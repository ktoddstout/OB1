#!/usr/bin/env python3
"""Import pre-exported Apple Notes text files into Open Brain."""

import os
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing: requests")
    sys.exit(1)


def load_env(p):
    if not p.exists(): return
    for l in p.read_text().splitlines():
        l = l.strip()
        if not l or l.startswith('#') or '=' not in l: continue
        k, v = l.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())


load_env(Path(__file__).parent.parent / "obsidian-vault-import" / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_API_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
EMBEDDING_MODEL = "openai/text-embedding-3-small"
EXPORT_DIR = Path("/tmp/apple_notes_export")


def get_embedding(text):
    for attempt in range(3):
        try:
            r = requests.post("https://openrouter.ai/api/v1/embeddings",
                headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                json={"model": EMBEDDING_MODEL, "input": text[:8000]}, timeout=30)
            if r.status_code in (429,) or r.status_code >= 500:
                time.sleep(2 ** attempt); continue
            r.raise_for_status()
            return r.json()["data"][0]["embedding"]
        except: time.sleep(2 ** attempt)
    return None


def insert(content, embedding, metadata):
    meta = {"type":"observation","people":[],"source":"apple-notes",
            "topics":metadata.get("topics",[]),"action_items":[],"dates_mentioned":[],
            **{k:v for k,v in metadata.items() if k!="topics"}}
    for attempt in range(3):
        try:
            r = requests.post(f"{SUPABASE_URL}/rest/v1/thoughts",
                headers={"apikey":SUPABASE_KEY,"Authorization":f"Bearer {SUPABASE_KEY}",
                         "Content-Type":"application/json","Prefer":"return=minimal"},
                json={"content":content,"embedding":embedding,"metadata":meta}, timeout=30)
            if r.status_code in (429,) or r.status_code >= 500:
                time.sleep(2 ** attempt); continue
            r.raise_for_status(); return True
        except: time.sleep(2 ** attempt)
    return False


def chunk(text, target=400):
    words = text.split()
    if len(words) <= target: return [text]
    chunks, cur, cl = [], [], 0
    for s in re.split(r'(?<=[.!?])\s+', text):
        sl = len(s.split())
        if cl + sl > target and cur:
            chunks.append(' '.join(cur)); cur, cl = [s], sl
        else: cur.append(s); cl += sl
    if cur: chunks.append(' '.join(cur))
    return chunks


def parse_file(path):
    notes = []
    raw = path.read_text(errors='replace')
    for part in raw.split("===SEPARATOR==="):
        part = part.strip()
        if not part or "===BODY===" not in part: continue
        header, body = part.split("===BODY===", 1)
        notes.append({"name": header.strip(), "body": body.strip()})
    return notes


def main():
    verbose = "--verbose" in sys.argv
    files = sorted(EXPORT_DIR.glob("*.txt"))
    if not files:
        print("No exported files found in /tmp/apple_notes_export/")
        sys.exit(1)

    print(f"Found {len(files)} export files")
    total_notes = 0
    total_inserted = 0
    total_skipped = 0
    consecutive_failures = 0

    for f in files:
        folder = f.stem.rsplit("_", 2)[0].replace("_", " ")
        notes = parse_file(f)
        if verbose:
            print(f"\n📄 {f.name}: {len(notes)} notes")

        for note in notes:
            body = note["body"]
            wc = len(body.split())
            if wc < 15:
                total_skipped += 1; continue
            total_notes += 1
            name = note["name"]
            chunks = chunk(body)
            if verbose:
                print(f"  [{total_notes}] {name[:60]} ({wc}w → {len(chunks)}ch)")

            for ci, ch in enumerate(chunks, 1):
                thought = f"[Apple Note: {folder}/{name} | Part {ci}/{len(chunks)}] {ch}"
                time.sleep(0.15)
                emb = get_embedding(thought)
                if not emb:
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        print("\n❌ 10 consecutive failures — aborting.")
                        print(f"Inserted so far: {total_inserted}")
                        sys.exit(1)
                    continue
                if insert(thought, emb, {"topics":[folder, name.split()[0]],"folder":folder,"note_name":name}):
                    total_inserted += 1; consecutive_failures = 0
                else:
                    consecutive_failures += 1

    print(f"\n{'='*50}")
    print(f"Notes: {total_notes}, Skipped: {total_skipped}, Inserted: {total_inserted}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
