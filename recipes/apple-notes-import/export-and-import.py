#!/usr/bin/env python3
"""
export-and-import.py — Export Apple Notes via AppleScript and import to Open Brain.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing dependency: requests")
    sys.exit(1)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://qokhwsxcujnmfyqbmmim.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_API_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
EMBEDDING_MODEL = "openai/text-embedding-3-small"
MAX_RETRIES = 3
RETRY_BACKOFF = 2
CHUNK_TARGET_WORDS = 400


def load_env(env_path: Path):
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, val = line.split('=', 1)
        os.environ.setdefault(key.strip(), val.strip())


def export_folder(folder_name: str) -> list[dict]:
    """Export all notes from an Apple Notes folder via AppleScript."""
    script = f'''
    tell application "Notes"
        set noteList to notes of folder "{folder_name}"
        set output to ""
        repeat with n in noteList
            set noteName to name of n
            set noteBody to plaintext of n
            set noteDate to modification date of n as string
            set output to output & "===SEPARATOR===" & linefeed & noteName & linefeed & "===DATE===" & linefeed & noteDate & linefeed & "===BODY===" & linefeed & noteBody & linefeed
        end repeat
        return output
    end tell
    '''
    try:
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"  ⚠️  AppleScript error for {folder_name}: {result.stderr[:200]}")
            return []
    except subprocess.TimeoutExpired:
        print(f"  ⚠️  Timeout exporting {folder_name}")
        return []

    notes = []
    raw = result.stdout
    parts = raw.split("===SEPARATOR===")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "===DATE===" in part and "===BODY===" in part:
            header, rest = part.split("===DATE===", 1)
            date_part, body = rest.split("===BODY===", 1)
            notes.append({
                "name": header.strip(),
                "date": date_part.strip(),
                "body": body.strip(),
                "folder": folder_name,
            })
        elif "===BODY===" in part:
            header, body = part.split("===BODY===", 1)
            notes.append({
                "name": header.strip(),
                "date": "",
                "body": body.strip(),
                "folder": folder_name,
            })

    return notes


def get_embedding(text: str) -> list[float] | None:
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                json={"model": EMBEDDING_MODEL, "input": text[:8000]},
                timeout=30,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))
                continue
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(RETRY_BACKOFF * (2 ** attempt))
    return None


def insert_thought(content: str, embedding: list, metadata: dict) -> bool:
    full_metadata = {
        "type": "observation",
        "people": [],
        "source": "apple-notes",
        "topics": metadata.get("topics", []),
        "action_items": [],
        "dates_mentioned": [],
        **{k: v for k, v in metadata.items() if k not in ("topics",)},
    }
    row = {"content": content, "embedding": embedding, "metadata": full_metadata}
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{SUPABASE_URL}/rest/v1/thoughts",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                json=row, timeout=30,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))
                continue
            resp.raise_for_status()
            return True
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"    ❌ Insert failed: {e}")
                return False
            time.sleep(RETRY_BACKOFF * (2 ** attempt))
    return False


def chunk_text(text: str, target: int = CHUNK_TARGET_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= target:
        return [text]
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current, current_len = [], 0
    for s in sentences:
        slen = len(s.split())
        if current_len + slen > target and current:
            chunks.append(' '.join(current))
            current, current_len = [s], slen
        else:
            current.append(s)
            current_len += slen
    if current:
        chunks.append(' '.join(current))
    return chunks


def main():
    script_dir = Path(__file__).parent
    load_env(script_dir.parent / "obsidian-vault-import" / ".env")

    global SUPABASE_KEY, OPENROUTER_KEY
    SUPABASE_KEY = os.environ.get("SUPABASE_API_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

    if not SUPABASE_KEY or not OPENROUTER_KEY:
        print("❌ Missing credentials")
        sys.exit(1)

    folders = [
        "Book Quotes", "Tim Keller", "Child Dedications",
        "Prepare/Enrich", "D.Min. Project", "The Letter of James ",
        "Individualism", "Vibe Coding ", "Notes",
    ]

    dry_run = "--dry-run" in sys.argv
    verbose = "--verbose" in sys.argv
    limit = None
    for arg in sys.argv:
        if arg.startswith("--limit="):
            limit = int(arg.split("=")[1])

    total_inserted = 0
    total_skipped = 0
    total_notes = 0

    for folder in folders:
        print(f"\n📁 Exporting: {folder}")
        notes = export_folder(folder)
        print(f"   Found {len(notes)} notes")

        for idx, note in enumerate(notes):
            if limit and total_notes >= limit:
                break

            body = note["body"].strip()
            wc = len(body.split())
            if wc < 15:
                total_skipped += 1
                continue

            total_notes += 1
            name = note["name"]

            if verbose:
                print(f"  [{total_notes}] {name} ({wc} words)")

            chunks = chunk_text(body)
            for ci, chunk in enumerate(chunks, 1):
                thought = f"[Apple Note: {folder}/{name} | Part {ci}/{len(chunks)}] {chunk}"
                metadata = {
                    "topics": [folder.strip(), name.split()[0] if name else "misc"],
                    "folder": folder,
                    "note_name": name,
                    "chunk": ci,
                    "total_chunks": len(chunks),
                }

                if dry_run:
                    if verbose and ci == 1:
                        print(f"    → {len(chunks)} chunks")
                    continue

                time.sleep(0.15)
                embedding = get_embedding(thought)
                if not embedding:
                    continue

                if insert_thought(thought, embedding, metadata):
                    total_inserted += 1
                    if verbose:
                        print(f"    ✅ Chunk {ci}/{len(chunks)}")

        if limit and total_notes >= limit:
            break

    print(f"\n{'='*50}")
    print(f"Notes processed: {total_notes}")
    print(f"Notes skipped (too short): {total_skipped}")
    if not dry_run:
        print(f"Thoughts inserted: {total_inserted}")
    else:
        print("(Dry run)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
