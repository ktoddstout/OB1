#!/usr/bin/env python3
"""Import PDFs and docs from iCloud Drive into Open Brain."""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing: requests"); sys.exit(1)


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

ICLOUD = Path("/Users/toddstout/Library/Mobile Documents/com~apple~CloudDocs")

SCAN_DIRS = [
    ICLOUD / "Sermons",
    ICLOUD / "Redeemer City Ministry Year",
    ICLOUD / "Baptism",
    ICLOUD / "FaithLab",
    ICLOUD / "Prepare:Enrich",
    ICLOUD / "Advent Hope",
    ICLOUD / "Administration",
    ICLOUD / "Ministry",
]


def extract_pdf(path):
    try:
        r = subprocess.run(['pdftotext', str(path), '-'], capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except FileNotFoundError:
        pass
    try:
        r = subprocess.run(['strings', str(path)], capture_output=True, text=True, timeout=30)
        lines = [l for l in r.stdout.splitlines() if len(l) > 10 and any(c.isalpha() for c in l)]
        return '\n'.join(lines[:500])
    except:
        pass
    return ''


def extract_docx(path):
    try:
        r = subprocess.run(['textutil', '-convert', 'txt', '-stdout', str(path)],
                          capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            return r.stdout.strip()
    except:
        pass
    return ''


def extract_pages(path):
    return extract_docx(path)  # textutil handles .pages too


def extract_text(path):
    ext = path.suffix.lower()
    if ext == '.pdf':
        return extract_pdf(path)
    elif ext == '.docx':
        return extract_docx(path)
    elif ext == '.pages':
        return extract_pages(path)
    elif ext in ('.txt', '.md', '.rtf'):
        try:
            return path.read_text(errors='replace')
        except:
            return ''
    return ''


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
    meta = {"type": "observation", "people": [], "source": "icloud-drive",
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


SENSITIVE = ['tax', 'pay stub', 'password', 'ssn', 'social security', 'bank account']

def is_sensitive(name):
    lower = name.lower()
    return any(s in lower for s in SENSITIVE)


def main():
    verbose = "--verbose" in sys.argv

    if not SUPABASE_KEY or not OPENROUTER_KEY:
        print("❌ Missing credentials"); sys.exit(1)

    # Discover files
    files = []
    for d in SCAN_DIRS:
        if not d.exists():
            continue
        for f in d.rglob("*"):
            if f.suffix.lower() in ('.pdf', '.docx', '.pages', '.txt', '.md', '.rtf'):
                if not is_sensitive(f.name):
                    files.append(f)

    print(f"Files found: {len(files)}")
    print(f"Scanning: {', '.join(d.name for d in SCAN_DIRS if d.exists())}")
    print()

    total_inserted = 0
    total_skipped = 0
    consecutive_failures = 0

    for idx, f in enumerate(files, 1):
        text = extract_text(f)
        text = re.sub(r'\s+', ' ', text).strip()
        wc = len(text.split())

        if wc < 30:
            if verbose:
                print(f"  [{idx}/{len(files)}] {f.name} — skipped ({wc}w)")
            total_skipped += 1
            continue

        folder = f.parent.name
        title = f.stem

        if verbose:
            print(f"  [{idx}/{len(files)}] {folder}/{f.name} — {wc}w")

        chunks = chunk(text)
        for ci, ch in enumerate(chunks, 1):
            thought = f"[iCloud: {folder}/{title} | Part {ci}/{len(chunks)}] {ch}"
            time.sleep(0.15)
            emb = get_embedding(thought)
            if not emb:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    print(f"\n❌ 10 failures. Inserted: {total_inserted}")
                    sys.exit(1)
                continue

            title_word = title.split()[0] if title.strip() else "untitled"
            if insert(thought, emb, {"topics": [folder, title_word], "folder": folder, "file": f.name}):
                total_inserted += 1
                consecutive_failures = 0
                if verbose:
                    print(f"    ✅ {ci}/{len(chunks)}")
            else:
                consecutive_failures += 1

        time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"Files: {len(files)}, Skipped: {total_skipped}, Inserted: {total_inserted}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
