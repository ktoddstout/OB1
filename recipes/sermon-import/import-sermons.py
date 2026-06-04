#!/usr/bin/env python3
"""
import-sermons.py — Import sermon transcripts from all_sermons.json into Open Brain.

Reads a JSON file keyed by YouTube video IDs, chunks each transcript into
atomic thoughts, generates embeddings via OpenRouter, and inserts into Supabase.

Usage:
  python import-sermons.py /path/to/all_sermons.json --dry-run
  python import-sermons.py /path/to/all_sermons.json --limit 5 --verbose
  python import-sermons.py /path/to/all_sermons.json --verbose
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing dependency: requests")
    print("Run: pip install requests")
    sys.exit(1)


# ── Config ───────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIMS = 1536
LLM_MODEL = "openai/gpt-4o-mini"

MAX_RETRIES = 3
RETRY_BACKOFF = 2

# Chunk target: ~400 words per thought (sermon transcripts are long)
CHUNK_TARGET_WORDS = 400


# ── Env ──────────────────────────────────────────────────────────────────────

def load_env(env_path: Path):
    """Load .env file into os.environ."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, val = line.split('=', 1)
        os.environ.setdefault(key.strip(), val.strip())


# ── API helpers ──────────────────────────────────────────────────────────────

def get_embedding(text: str, api_key: str) -> list[float] | None:
    """Generate embedding via OpenRouter."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": EMBEDDING_MODEL, "input": text},
                timeout=30,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = RETRY_BACKOFF * (2 ** attempt)
                print(f"    ⏳ Rate limited, waiting {wait}s...")
                time.sleep(wait)
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
                   supabase_url: str, supabase_key: str, fingerprint: str) -> bool:
    """Insert a thought into the Supabase thoughts table."""
    # Merge source metadata into the format Open Brain expects
    full_metadata = {
        "type": "observation",
        "people": [],
        "source": metadata.get("source", "sermon"),
        "topics": ["sermon", "theology"],
        "action_items": [],
        "dates_mentioned": [],
        **{k: v for k, v in metadata.items() if k != "source"},
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
                wait = RETRY_BACKOFF * (2 ** attempt)
                time.sleep(wait)
                continue
            if resp.status_code == 409:
                # Duplicate fingerprint — already imported
                return True
            resp.raise_for_status()
            return True
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"    ❌ Insert failed: {e}")
                return False
            time.sleep(RETRY_BACKOFF * (2 ** attempt))
    return False


def summarize_chunk(chunk_text: str, video_id: str, chunk_num: int,
                    api_key: str) -> str:
    """Use LLM to create a clean summary of a sermon chunk."""
    prompt = f"""You are processing a sermon transcript chunk. Clean it up into a readable,
standalone passage that preserves the speaker's key points and message.

Rules:
- Remove filler words, false starts, and [Music] tags
- Keep the theological content and key arguments intact
- Preserve any scripture references
- Keep it under 300 words
- Write in third person ("The speaker discusses..." or "Pastor Todd explains...")
- Make it standalone — someone reading this should understand the point without context

Sermon chunk (video {video_id}, part {chunk_num}):
{chunk_text[:3000]}"""

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                },
                timeout=60,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(RETRY_BACKOFF * (2 ** attempt))
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"    ⚠️  Summarize failed, using raw chunk: {e}")
                return chunk_text[:1500]
            time.sleep(RETRY_BACKOFF * (2 ** attempt))
    return chunk_text[:1500]


# ── Chunking ─────────────────────────────────────────────────────────────────

def clean_transcript(text: str) -> str:
    """Basic cleanup of transcript text."""
    # Remove [Music] tags
    text = re.sub(r'\[Music\]', '', text, flags=re.IGNORECASE)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def chunk_transcript(text: str, target_words: int = CHUNK_TARGET_WORDS) -> list[str]:
    """Split transcript into chunks of roughly target_words."""
    words = text.split()
    if len(words) <= target_words:
        return [text]

    chunks = []
    # Try to split on sentence boundaries near the target
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


def content_fingerprint(text: str) -> str:
    """SHA-256 hash of normalized content."""
    normalized = re.sub(r'\s+', ' ', text.strip().lower())
    return hashlib.sha256(normalized.encode()).hexdigest()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Import sermons into Open Brain")
    parser.add_argument("json_file", help="Path to all_sermons.json")
    parser.add_argument("--dry-run", action="store_true", help="Preview without inserting")
    parser.add_argument("--limit", type=int, help="Only process first N sermons")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM summarization, use raw chunks")
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress")
    args = parser.parse_args()

    # Load env
    script_dir = Path(__file__).parent
    load_env(script_dir / ".env")
    # Also check parent recipe dir
    load_env(script_dir.parent / "obsidian-vault-import" / ".env")

    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not supabase_url or not supabase_key:
        print("❌ Missing SUPABASE_URL or SUPABASE_API_KEY in environment or .env")
        sys.exit(1)
    if not openrouter_key:
        print("❌ Missing OPENROUTER_API_KEY in environment or .env")
        sys.exit(1)

    # Load sermons
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
        sys.exit(1)

    with open(json_path) as f:
        sermons = json.load(f)

    print(f"Source:   {json_path}")
    print(f"Sermons:  {len(sermons)}")
    print(f"Mode:     {'DRY RUN' if args.dry_run else 'LIVE IMPORT'}")
    print(f"LLM:      {'disabled' if args.no_llm else 'enabled (summarize chunks)'}")
    print()

    # Preflight check (unless dry run)
    if not args.dry_run:
        print("Preflight check...")
        try:
            resp = requests.get(
                f"{supabase_url}/rest/v1/thoughts?select=id&limit=1",
                headers={
                    "apikey": supabase_key,
                    "Authorization": f"Bearer {supabase_key}",
                },
                timeout=10,
            )
            resp.raise_for_status()
            print("  ✅ Supabase connection OK")
        except Exception as e:
            print(f"  ❌ Supabase connection failed: {e}")
            sys.exit(1)

        try:
            test_resp = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={"Authorization": f"Bearer {openrouter_key}"},
                json={"model": EMBEDDING_MODEL, "input": "test"},
                timeout=10,
            )
            test_resp.raise_for_status()
            print("  ✅ OpenRouter API key OK")
        except Exception as e:
            print(f"  ❌ OpenRouter API key failed: {e}")
            sys.exit(1)
        print()

    items = list(sermons.items())
    if args.limit:
        items = items[:args.limit]

    total_thoughts = 0
    total_inserted = 0
    total_skipped = 0
    consecutive_failures = 0

    for idx, (video_id, transcript) in enumerate(items, 1):
        cleaned = clean_transcript(transcript)
        wc = len(cleaned.split())

        if wc < 50:
            if args.verbose:
                print(f"  [{idx}] {video_id} — skipped (only {wc} words)")
            total_skipped += 1
            continue

        chunks = chunk_transcript(cleaned)

        if args.verbose:
            print(f"  [{idx}/{len(items)}] {video_id} — {wc} words → {len(chunks)} chunks")

        for ci, chunk in enumerate(chunks, 1):
            # Summarize via LLM or use raw
            if args.no_llm:
                thought_text = f"[Sermon: {video_id} | Part {ci}/{len(chunks)}] {chunk}"
            else:
                if args.verbose:
                    print(f"    Summarizing chunk {ci}/{len(chunks)}...")
                summary = summarize_chunk(chunk, video_id, ci, openrouter_key)
                thought_text = f"[Sermon: {video_id} | Part {ci}/{len(chunks)}] {summary}"

            fp = content_fingerprint(thought_text)
            metadata = {
                "source": "sermon",
                "youtube_id": video_id,
                "youtube_url": f"https://youtube.com/watch?v={video_id}",
                "chunk": ci,
                "total_chunks": len(chunks),
                "word_count": wc,
            }

            total_thoughts += 1

            if args.dry_run:
                if args.verbose:
                    preview = thought_text[:120].replace('\n', ' ')
                    print(f"    → [{ci}/{len(chunks)}] {preview}...")
                continue

            # Generate embedding
            time.sleep(0.15)  # Rate limiting
            embedding = get_embedding(thought_text, openrouter_key)
            if not embedding:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    print("\n❌ 10 consecutive failures — aborting.")
                    sys.exit(1)
                continue

            # Insert
            success = insert_thought(
                thought_text, embedding, metadata,
                supabase_url, supabase_key, fp,
            )
            if success:
                total_inserted += 1
                consecutive_failures = 0
                if args.verbose:
                    print(f"    ✅ Chunk {ci}/{len(chunks)} inserted")
            else:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    print("\n❌ 10 consecutive failures — aborting.")
                    sys.exit(1)

        # Pause between sermons
        if not args.dry_run:
            time.sleep(1)

    print()
    print("=" * 50)
    print(f"Sermons processed: {len(items)}")
    print(f"Sermons skipped:   {total_skipped}")
    print(f"Thoughts generated: {total_thoughts}")
    if not args.dry_run:
        print(f"Thoughts inserted: {total_inserted}")
    else:
        print("(Dry run — nothing inserted)")
    print("=" * 50)


if __name__ == "__main__":
    main()
