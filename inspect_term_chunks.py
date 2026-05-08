from __future__ import annotations

import re
from collections import defaultdict

from app.metadata_db import MetadataDB
from app.qdrant_client import get_chunks_by_source_id


TERMS = [
    "Atman",
    "Purusha",
    "yama",
    "niyama",
]


def has_term(text: str, term: str) -> bool:
    return re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])",
        text or "",
        flags=re.IGNORECASE,
    ) is not None


def snippet(text: str, term: str, radius: int = 260) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    m = re.search(re.escape(term), clean, flags=re.IGNORECASE)
    if not m:
        return clean[: radius * 2]
    start = max(0, m.start() - radius)
    end = min(len(clean), m.end() + radius)
    value = clean[start:end]
    if start > 0:
        value = "... " + value
    if end < len(clean):
        value += " ..."
    return value


db = MetadataDB()
sources = db.list_active_files()

print(f"Active sources: {len(sources)}")
print("=" * 100)

for source in sources:
    source_id = source.get("source_id")
    source_path = source.get("source_path") or source.get("path") or source.get("source") or "unknown"

    if not source_id:
        continue

    chunks = get_chunks_by_source_id(source_id)
    if not chunks:
        continue

    counts = defaultdict(int)
    hits_by_term = defaultdict(list)

    for chunk in chunks:
        text = chunk.get("text") or ""
        for term in TERMS:
            if has_term(text, term):
                counts[term] += 1
                hits_by_term[term].append(chunk)

    if not any(counts.values()):
        continue

    print()
    print(f"SOURCE: {source_path}")
    print(f"SOURCE_ID: {source_id}")
    print(f"TOTAL_CHUNKS: {len(chunks)}")
    for term in TERMS:
        print(f"{term}: {counts[term]} chunks")

    print("-" * 100)

    for term in TERMS:
        print()
        print(f"TOP MATCHES FOR: {term}")
        for chunk in hits_by_term[term][:8]:
            idx = chunk.get("chunk_index")
            page = chunk.get("page_number") or chunk.get("page_start")
            print(f"\nChunk: {idx} | Page: {page}")
            print(snippet(chunk.get("text") or "", term))
