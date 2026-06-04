#!/usr/bin/env python3
"""
import-omnioutliner.py — Import OmniOutliner sermon notes into Open Brain.

Extracts text from .oo3 and .ooutline XML files, chunks them,
generates embeddings, and inserts into Supabase.

Usage:
  python import-omnioutliner.py --dry-run --verbose
  python import-omnioutliner.py --limit 5 --verbose
  python import-omnioutliner.py --verbose
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing dependency: requests")
    sys.exit(1)

# ── Config ───────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "openai/text-embedding-3-small"
MAX_RETRIES = 3
RETRY_BACKOFF = 2
CHUNK_TARGET_WORDS = 400
OMNI_ROOT = Path("/Users/toddstout/OmniPresence")
OO_NS = "http://www.omnigroup.com/namespace/OmniOutliner/v5"


# ── Env ──────────────────────────────────────────────────────────────────────

def load_env(env_path: Path):
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, val = line.split('=', 1)
        os.environ.setdefault(key.strip(), val.strip())


# ── OmniOutliner parsing ────────────────────────────────────────────────────

def extract_oo_text(contents_xml: Path) -> str:
    """Extract all text content from an OmniOutliner contents.xml."""
    try:
        tree = ET.parse(str(contents_xml))
        texts = []
        for lit in tree.iter(f'{{{OO_NS}}}lit'):
            if lit.text and len(lit.text.strip()) > 2:
                texts.append(lit.text.strip())
        return '\n\n'.join(texts)
    except Exception as e:
        return ''


def find_oo_files(root: Path) -> list[tuple[str, str, Path]]:
    """Find all .oo3 and .ooutline files. Returns (series, title, contents_xml_path)."""
    files = []
    for ext in ('*.oo3', '*.ooutline'):
        for bundle in root.rglob(ext):
            contents = bundle / 'contents.xml'
            if not contents.exists():
                continue
            # Determine series from parent folder
            rel = bundle.relative_to(root)
            parts = rel.parts
            if len(parts) > 1:
                series = parts[0]
            else:
                series = ""
            title = bundle.stem
            files.append((series, title, contents))
    return sorted(files, key=lambda x: (x[0], x[1]))


# ── API helpers ──────────────────────────────────────────────────────────────

def get_embedding(text: str, api_key: str) -> list[float] | None:
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
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
                print(f"    ❌ Embedding failed: {e}")
                return None
            time.sleep(RETRY_BACKOFF * (2 ** attempt))
    return None


def insert_thought(content: str, embedding: list, metadata: dict,
                   supabase_url: str, supabase_key: str) -> bool:
    full_metadata = {
        "type": "observation",
        "people": [],
        "source": metadata.get("source", "omnioutliner"),
        "topics": metadata.get("topics", []),
        "action_items": [],
        "dates_mentioned": [],
        **{k: v for k, v in metadata.items() if k not in ("source", "topics")},
    }
    row = {
        "content": content,
        "embedding": embedding,
        "metadata": full_metadata,
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{supabase_url}/rest/v1/thoughts",
                headers={
                    "apikey": supabase_key,
                    "Authorization": f"Bearer {supabase_key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                json=row,
                timeout=30,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))
                continue
            if resp.status_code == 409:
                return True
            resp.raise_for_status()
            return True
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"    ❌ Insert failed: {e}")
                return False
            time.sleep(RETRY_BACKOFF * (2 ** attempt))
    return False


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, target_words: int = CHUNK_TARGET_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= target_words:
        return [text]
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current = []
    current_len = 0
    for sentence in sentences:
        slen = len(sentence.split())
        if current_len + slen > target_words and current:
            chunks.append(' '.join(current))
            current = [sentence]
            current_len = slen
        else:
            current.append(sentence)
            current_len += slen
    if current:
        chunks.append(' '.join(current))
    return chunks


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Import OmniOutliner files into Open Brain")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    load_env(script_dir.parent / "obsidian-vault-import" / ".env")

    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_API_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not supabase_url or not supabase_key:
        print("❌ Missing SUPABASE_URL or API key")
        sys.exit(1)
    if not openrouter_key:
        print("❌ Missing OPENROUTER_API_KEY")
        sys.exit(1)

    # Discover files
    files = find_oo_files(OMNI_ROOT)
    print(f"OmniOutliner files found: {len(files)}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE IMPORT'}")
    print()

    if args.limit:
        files = files[:args.limit]

    # Preflight
    if not args.dry_run:
        print("Preflight check...")
        try:
            resp = requests.get(
                f"{supabase_url}/rest/v1/thoughts?select=id&limit=1",
                headers={"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            print("  ✅ Supabase OK")
        except Exception as e:
            print(f"  ❌ Supabase failed: {e}")
            sys.exit(1)
        try:
            test_resp = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={"Authorization": f"Bearer {openrouter_key}"},
                json={"model": EMBEDDING_MODEL, "input": "test"},
                timeout=10,
            )
            test_resp.raise_for_status()
            print("  ✅ OpenRouter OK")
        except Exception as e:
            print(f"  ❌ OpenRouter failed: {e}")
            sys.exit(1)
        print()

    total_thoughts = 0
    total_inserted = 0
    total_skipped = 0
    consecutive_failures = 0

    for idx, (series, title, contents_path) in enumerate(files, 1):
        text = extract_oo_text(contents_path)
        text = re.sub(r'\s+', ' ', text).strip()
        wc = len(text.split())

        if wc < 30:
            if args.verbose:
                print(f"  [{idx}/{len(files)}] {series}/{title} — skipped ({wc} words)")
            total_skipped += 1
            continue

        if args.verbose:
            print(f"  [{idx}/{len(files)}] {series}/{title} — {wc} words")

        chunks = chunk_text(text)
        if args.verbose:
            print(f"    → {len(chunks)} chunks")

        for ci, chunk in enumerate(chunks, 1):
            label = f"{series}: {title}" if series else title
            thought_text = f"[Sermon Note: {label} | Part {ci}/{len(chunks)}] {chunk}"

            # Determine topics from series name
            topics = ["sermon"]
            if series:
                topics.append(series.split(")")[-1].strip() if ")" in series else series)

            metadata = {
                "source": "omnioutliner",
                "topics": topics[:5],
                "series": series,
                "title": title,
                "chunk": ci,
                "total_chunks": len(chunks),
            }

            total_thoughts += 1

            if args.dry_run:
                if args.verbose and ci <= 2:
                    preview = thought_text[:120].replace('\n', ' ')
                    print(f"      [{ci}/{len(chunks)}] {preview}...")
                continue

            time.sleep(0.15)
            embedding = get_embedding(thought_text, openrouter_key)
            if not embedding:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    print("\n❌ 10 consecutive failures — aborting.")
                    sys.exit(1)
                continue

            success = insert_thought(
                thought_text, embedding, metadata,
                supabase_url, supabase_key,
            )
            if success:
                total_inserted += 1
                consecutive_failures = 0
                if args.verbose:
                    print(f"      ✅ Chunk {ci}/{len(chunks)}")
            else:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    print("\n❌ 10 consecutive failures — aborting.")
                    sys.exit(1)

        if not args.dry_run:
            time.sleep(1)

    print()
    print("=" * 50)
    print(f"Files processed: {len(files)}")
    print(f"Files skipped:   {total_skipped}")
    print(f"Thoughts generated: {total_thoughts}")
    if not args.dry_run:
        print(f"Thoughts inserted: {total_inserted}")
    else:
        print("(Dry run — nothing inserted)")
    print("=" * 50)


if __name__ == "__main__":
    main()
