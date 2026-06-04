#!/usr/bin/env python3
"""
import-documents.py — Import .ppr (teleprompter) and .pdf files into Open Brain.

Extracts text from SQLite-based .ppr files and PDF files, chunks them,
generates embeddings, and inserts into Supabase.

Usage:
  python import-documents.py --dry-run --verbose
  python import-documents.py --limit 5 --verbose
  python import-documents.py --verbose
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing dependency: requests")
    sys.exit(1)


# ── Config ───────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "openai/text-embedding-3-small"
LLM_MODEL = "openai/gpt-4o-mini"
MAX_RETRIES = 3
RETRY_BACKOFF = 2
CHUNK_TARGET_WORDS = 400

# Sensitive file patterns to skip
SENSITIVE_PATTERNS = [
    r'tax', r'990', r'bank', r'chase', r'debt', r'loan', r'will.*testament',
    r'ez.?pass', r'mta', r'myers.?briggs', r'applicat', r'wedding',
    r'hold.?harmless', r'score', r'pacific.?debt',
]

# Documents folder scan paths
DOCUMENTS_ROOT = Path("/Users/toddstout/Documents")


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


# ── File extraction ──────────────────────────────────────────────────────────

def extract_ppr_text(path: Path) -> str:
    """Extract text content from a .ppr SQLite file."""
    try:
        conn = sqlite3.connect(str(path))
        cursor = conn.execute("SELECT * FROM texts ORDER BY id")
        rows = cursor.fetchall()
        conn.close()
        # Column index 1 is typically the text content
        texts = []
        for row in rows:
            for col in row[1:]:
                if isinstance(col, str) and len(col) > 20:
                    texts.append(col)
        return '\n\n'.join(texts)
    except Exception as e:
        print(f"    ⚠️  Failed to read .ppr: {e}")
        return ''


def extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF using macOS built-in tools."""
    try:
        # Try mdimport/textutil approach first
        result = subprocess.run(
            ['mdimport', '-d1', str(path)],
            capture_output=True, text=True, timeout=10
        )
    except Exception:
        pass

    # Use python-based approach with subprocess
    try:
        # macOS: use 'strings' as a basic fallback, or try pdftotext if available
        result = subprocess.run(
            ['pdftotext', str(path), '-'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except FileNotFoundError:
        pass

    # Fallback: use Python to read with basic text extraction
    try:
        # Try using macOS Automator / textutil
        result = subprocess.run(
            ['strings', str(path)],
            capture_output=True, text=True, timeout=30
        )
        if result.stdout.strip():
            # Filter to lines that look like real text
            lines = [l for l in result.stdout.splitlines()
                     if len(l) > 10 and any(c.isalpha() for c in l)]
            return '\n'.join(lines[:500])  # Cap at 500 lines
    except Exception:
        pass

    return ''


def is_sensitive(filename: str) -> bool:
    """Check if a file likely contains sensitive personal/financial data."""
    lower = filename.lower()
    for pattern in SENSITIVE_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


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


def summarize_chunk(chunk_text: str, title: str, api_key: str) -> str:
    prompt = f"""Summarize this document excerpt into a clear, standalone passage.
Preserve key points, scripture references, and theological content.
Keep it under 300 words. Write in third person.

Document: {title}
Content:
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
                return chunk_text[:1500]
            time.sleep(RETRY_BACKOFF * (2 ** attempt))
    return chunk_text[:1500]


def insert_thought(content: str, embedding: list, metadata: dict,
                   supabase_url: str, supabase_key: str) -> bool:
    full_metadata = {
        "type": "observation",
        "people": [],
        "source": metadata.get("source", "document"),
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
    parser = argparse.ArgumentParser(description="Import documents into Open Brain")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ppr-only", action="store_true", help="Only import .ppr files")
    parser.add_argument("--pdf-only", action="store_true", help="Only import .pdf files")
    args = parser.parse_args()

    # Load env
    script_dir = Path(__file__).parent
    load_env(script_dir / ".env")
    load_env(script_dir.parent / "obsidian-vault-import" / ".env")

    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not supabase_url or not supabase_key:
        print("❌ Missing SUPABASE_URL or SUPABASE_API_KEY")
        sys.exit(1)
    if not openrouter_key:
        print("❌ Missing OPENROUTER_API_KEY")
        sys.exit(1)

    # Discover files
    files = []

    if not args.pdf_only:
        # Find .ppr files
        for ppr in DOCUMENTS_ROOT.rglob("*.ppr"):
            if not is_sensitive(ppr.name):
                files.append(("ppr", ppr))

    if not args.ppr_only:
        # Find .pdf files
        for pdf in DOCUMENTS_ROOT.rglob("*.pdf"):
            if not is_sensitive(pdf.name):
                files.append(("pdf", pdf))

    print(f"Documents found: {len(files)}")
    print(f"  .ppr files: {sum(1 for t, _ in files if t == 'ppr')}")
    print(f"  .pdf files: {sum(1 for t, _ in files if t == 'pdf')}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE IMPORT'}")
    print(f"Sensitive files skipped: auto-filtered")
    print()

    if args.limit:
        files = files[:args.limit]

    # Preflight
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

    for idx, (ftype, fpath) in enumerate(files, 1):
        title = fpath.stem
        rel_path = str(fpath.relative_to(DOCUMENTS_ROOT))

        # Extract text
        if ftype == "ppr":
            text = extract_ppr_text(fpath)
        else:
            text = extract_pdf_text(fpath)

        # Clean
        text = re.sub(r'\s+', ' ', text).strip()
        wc = len(text.split())

        if wc < 30:
            if args.verbose:
                print(f"  [{idx}/{len(files)}] {rel_path} — skipped ({wc} words)")
            total_skipped += 1
            continue

        if args.verbose:
            print(f"  [{idx}/{len(files)}] {rel_path} — {wc} words")

        chunks = chunk_text(text)
        if args.verbose:
            print(f"    → {len(chunks)} chunks")

        for ci, chunk in enumerate(chunks, 1):
            if args.no_llm:
                thought_text = f"[{ftype.upper()}: {title} | Part {ci}/{len(chunks)}] {chunk}"
            else:
                if args.verbose:
                    print(f"    Summarizing chunk {ci}/{len(chunks)}...")
                summary = summarize_chunk(chunk, title, openrouter_key)
                thought_text = f"[{ftype.upper()}: {title} | Part {ci}/{len(chunks)}] {summary}"

            metadata = {
                "source": f"document-{ftype}",
                "topics": ["sermon" if ftype == "ppr" else "document", title.split()[0] if title else "misc"],
                "file_type": ftype,
                "file_name": fpath.name,
                "file_path": rel_path,
                "chunk": ci,
                "total_chunks": len(chunks),
            }

            total_thoughts += 1

            if args.dry_run:
                if args.verbose:
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
