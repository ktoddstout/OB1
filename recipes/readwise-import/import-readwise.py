#!/usr/bin/env python3
"""
import-readwise.py — Bulk import all Readwise highlights into Open Brain.
"""

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
READWISE_TOKEN = os.environ.get("READWISE_TOKEN", "")
EMBEDDING_MODEL = "openai/text-embedding-3-small"

# Readwise API - we'll use the Readwise export API
READWISE_HIGHLIGHTS_URL = "https://readwise.io/api/v2/highlights/"


def get_readwise_highlights(page=1, page_size=100):
    """Fetch a page of highlights from Readwise API."""
    # We need the Readwise token. If not in env, try to get from the API directly.
    # Since we have MCP access, we'll use the REST API with pagination.
    headers = {}
    if READWISE_TOKEN:
        headers["Authorization"] = f"Token {READWISE_TOKEN}"

    resp = requests.get(
        READWISE_HIGHLIGHTS_URL,
        headers=headers,
        params={"page": page, "page_size": page_size},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


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
        except:
            time.sleep(2 ** attempt)
    return None


def insert(content, embedding, metadata):
    meta = {"type": "observation", "people": [], "source": "readwise",
            "topics": metadata.get("topics", []), "action_items": [],
            "dates_mentioned": [], **{k: v for k, v in metadata.items() if k != "topics"}}
    for attempt in range(3):
        try:
            r = requests.post(f"{SUPABASE_URL}/rest/v1/thoughts",
                headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
                         "Content-Type": "application/json", "Prefer": "return=minimal"},
                json={"content": content, "embedding": embedding, "metadata": meta}, timeout=30)
            if r.status_code in (429,) or r.status_code >= 500:
                time.sleep(2 ** attempt); continue
            r.raise_for_status(); return True
        except:
            time.sleep(2 ** attempt)
    return False


def main():
    verbose = "--verbose" in sys.argv

    if not SUPABASE_KEY or not OPENROUTER_KEY:
        print("❌ Missing SUPABASE or OPENROUTER credentials")
        sys.exit(1)

    if not READWISE_TOKEN:
        print("❌ Missing READWISE_TOKEN in .env")
        print("Get your token from https://readwise.io/access_token")
        sys.exit(1)

    # Preflight
    print("Preflight check...")
    try:
        data = get_readwise_highlights(page=1, page_size=1)
        total = data.get("count", 0)
        print(f"  ✅ Readwise API OK — {total} total highlights")
    except Exception as e:
        print(f"  ❌ Readwise API failed: {e}")
        sys.exit(1)

    # Fetch all books for title lookup
    print("Fetching book metadata...")
    books = {}
    book_page = 1
    while True:
        try:
            r = requests.get("https://readwise.io/api/v2/books/",
                headers={"Authorization": f"Token {READWISE_TOKEN}"},
                params={"page": book_page, "page_size": 100}, timeout=30)
            r.raise_for_status()
            bdata = r.json()
            for b in bdata.get("results", []):
                books[b["id"]] = {"title": b.get("title", "Unknown"), "author": b.get("author", "")}
            if not bdata.get("next"):
                break
            book_page += 1
        except:
            break
    print(f"  Loaded {len(books)} books/sources")

    # Page through all highlights
    page = 1
    page_size = 100
    total_inserted = 0
    total_skipped = 0
    consecutive_failures = 0

    print(f"\nImporting {total} highlights...")

    while True:
        try:
            data = get_readwise_highlights(page=page, page_size=page_size)
        except Exception as e:
            print(f"  ❌ Failed to fetch page {page}: {e}")
            break

        results = data.get("results", [])
        if not results:
            break

        for h in results:
            text = h.get("text", "").strip()
            if len(text.split()) < 5:
                total_skipped += 1
                continue

            book_id = h.get("book_id")
            book_info = books.get(book_id, {"title": "Unknown", "author": ""})
            title = book_info["title"]
            author = book_info["author"]
            note = h.get("note", "").strip()
            tags = [t["name"] for t in h.get("tags", [])]

            # Build thought content
            source_label = f"{title}"
            if author:
                source_label += f" by {author}"
            thought = f"[Readwise: {source_label}] {text}"
            if note:
                thought += f"\n\nNote: {note}"

            metadata = {
                "topics": tags[:5] if tags else [title.split()[0] if title else "misc"],
                "book_title": title,
                "author": author,
                "readwise_id": h.get("id"),
                "highlighted_at": h.get("highlighted_at", ""),
            }

            time.sleep(0.15)
            emb = get_embedding(thought)
            if not emb:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    print(f"\n❌ 10 consecutive failures at page {page}")
                    print(f"Inserted so far: {total_inserted}")
                    sys.exit(1)
                continue

            if insert(thought, emb, metadata):
                total_inserted += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    print(f"\n❌ 10 consecutive failures at page {page}")
                    print(f"Inserted so far: {total_inserted}")
                    sys.exit(1)

        processed = (page - 1) * page_size + len(results)
        if verbose or page % 5 == 0:
            print(f"  Page {page}: {processed}/{total} processed, {total_inserted} inserted")

        if not data.get("next"):
            break
        page += 1

    print(f"\n{'='*50}")
    print(f"Total highlights: {total}")
    print(f"Skipped (too short): {total_skipped}")
    print(f"Thoughts inserted: {total_inserted}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
