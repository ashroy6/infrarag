from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from app.metadata_db import MetadataDB
from app.qdrant_client import get_chunks_by_source_id

DB_PATH = os.getenv("METADATA_DB_PATH", "/app/data/infrarag.db")

MAX_FTS_INDEX_SOURCES_PER_SEARCH = int(os.getenv("MAX_FTS_INDEX_SOURCES_PER_SEARCH", "20"))
MAX_FTS_RESULTS = int(os.getenv("MAX_FTS_RESULTS", "80"))


def _connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def ensure_keyword_index() -> None:
    """
    Creates a SQLite FTS5 index for fast keyword/exact phrase search.

    This is separate from Qdrant:
    - Qdrant = vector/semantic search
    - SQLite FTS5 = keyword/exact phrase search
    """
    with _connect() as conn:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                text,
                source_id UNINDEXED,
                source UNINDEXED,
                source_type UNINDEXED,
                file_type UNINDEXED,
                parser_type UNINDEXED,
                page_number UNINDEXED,
                page_start UNINDEXED,
                page_end UNINDEXED,
                chunk_index UNINDEXED,
                tokenize = 'unicode61'
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS keyword_index_sources (
                source_id TEXT PRIMARY KEY,
                chunk_count INTEGER DEFAULT 0,
                indexed_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def _indexed_source_ids() -> set[str]:
    ensure_keyword_index()

    with _connect() as conn:
        rows = conn.execute("SELECT source_id FROM keyword_index_sources").fetchall()
        return {str(row["source_id"]) for row in rows if row["source_id"]}


def index_source(source_id: str) -> dict[str, Any]:
    """
    Rebuilds FTS rows for one source_id from Qdrant chunk payloads.
    """
    ensure_keyword_index()

    clean_source_id = str(source_id or "").strip()
    if not clean_source_id:
        return {"source_id": source_id, "indexed": False, "chunks": 0, "reason": "empty_source_id"}

    chunks = get_chunks_by_source_id(clean_source_id)

    with _connect() as conn:
        conn.execute("DELETE FROM chunk_fts WHERE source_id = ?", (clean_source_id,))
        conn.execute("DELETE FROM keyword_index_sources WHERE source_id = ?", (clean_source_id,))

        rows: list[tuple[Any, ...]] = []
        for chunk in chunks:
            text = str(chunk.get("text") or "").strip()
            if not text:
                continue

            rows.append(
                (
                    text,
                    str(chunk.get("source_id") or clean_source_id),
                    str(chunk.get("source") or ""),
                    str(chunk.get("source_type") or ""),
                    str(chunk.get("file_type") or ""),
                    str(chunk.get("parser_type") or ""),
                    chunk.get("page_number"),
                    chunk.get("page_start"),
                    chunk.get("page_end"),
                    int(chunk.get("chunk_index") or 0),
                )
            )

        if rows:
            conn.executemany(
                """
                INSERT INTO chunk_fts (
                    text,
                    source_id,
                    source,
                    source_type,
                    file_type,
                    parser_type,
                    page_number,
                    page_start,
                    page_end,
                    chunk_index
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

        conn.execute(
            """
            INSERT INTO keyword_index_sources (source_id, chunk_count, indexed_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(source_id) DO UPDATE SET
                chunk_count = excluded.chunk_count,
                indexed_at = CURRENT_TIMESTAMP
            """,
            (clean_source_id, len(rows)),
        )

    return {
        "source_id": clean_source_id,
        "indexed": True,
        "chunks": len(rows),
        "reason": "indexed_from_qdrant",
    }


def ensure_sources_indexed(
    *,
    source_id: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
    allowed_source_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Lazy indexer.

    First keyword search indexes missing active sources once.
    After that FTS5 search is fast.
    """
    ensure_keyword_index()

    indexed = _indexed_source_ids()
    db = MetadataDB()

    if source_id:
        candidate_records = []
        record = db.get_file(source_id)
        if record:
            candidate_records.append(record)
    else:
        candidate_records = db.list_active_files(source_type=source_type)

    allowed = None
    if allowed_source_ids is not None:
        allowed = {str(item).strip() for item in allowed_source_ids if str(item).strip()}

    indexed_now: list[dict[str, Any]] = []
    skipped = 0

    for record in candidate_records:
        sid = str(record.get("source_id") or "").strip()
        if not sid:
            skipped += 1
            continue

        if allowed is not None and sid not in allowed:
            skipped += 1
            continue

        if file_type and str(record.get("file_type") or "") != file_type:
            skipped += 1
            continue

        if sid in indexed:
            continue

        if len(indexed_now) >= MAX_FTS_INDEX_SOURCES_PER_SEARCH:
            break

        indexed_now.append(index_source(sid))

    return {
        "indexed_now": indexed_now,
        "indexed_now_count": len(indexed_now),
        "skipped": skipped,
    }


def rebuild_keyword_index() -> dict[str, Any]:
    """
    Full manual rebuild. Useful after large ingestion/import.
    """
    ensure_keyword_index()

    db = MetadataDB()
    records = db.list_active_files()

    with _connect() as conn:
        conn.execute("DELETE FROM chunk_fts")
        conn.execute("DELETE FROM keyword_index_sources")

    results = []
    for record in records:
        sid = str(record.get("source_id") or "").strip()
        if sid:
            results.append(index_source(sid))

    return {
        "sources_indexed": len(results),
        "chunks_indexed": sum(int(item.get("chunks") or 0) for item in results),
        "results": results,
    }


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _terms_for_match(query: str) -> list[str]:
    raw_terms = re.findall(r"[A-Za-z0-9_./-]+", query or "")
    terms: list[str] = []

    stop = {
        "a", "an", "the", "is", "are", "was", "were", "what", "who", "where",
        "when", "which", "how", "why", "do", "does", "did", "tell", "me",
        "about", "find", "show", "give", "explain", "compare", "and", "or",
    }

    for term in raw_terms:
        clean = term.strip().lower()
        if clean and clean not in stop and len(clean) > 1:
            terms.append(clean)

    return terms[:12]


def _escape_fts_phrase(value: str) -> str:
    clean = _normalize_text(value).replace('"', '""')
    return f'"{clean}"'


def build_fts_query(query: str, exact_phrases: list[str] | None = None) -> str:
    phrases = [_normalize_text(item) for item in (exact_phrases or []) if _normalize_text(item)]

    if phrases:
        return " OR ".join(_escape_fts_phrase(item) for item in phrases)

    terms = _terms_for_match(query)
    if not terms:
        return _escape_fts_phrase(query)

    # OR is intentional. It improves recall for entity/comparison searches.
    return " OR ".join(_escape_fts_phrase(term) for term in terms)


def _page_ok(row: sqlite3.Row, page_start: int | None, page_end: int | None) -> bool:
    page_number = row["page_number"]

    if page_number is None:
        return True

    try:
        page = int(page_number)
    except (TypeError, ValueError):
        return True

    if page_start is not None and page < page_start:
        return False

    if page_end is not None and page > page_end:
        return False

    return True


def search_keyword_index(
    *,
    query: str,
    limit: int = 50,
    source_id: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    allowed_source_ids: list[str] | None = None,
    exact_phrases: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Fast keyword search using SQLite FTS5.

    Returns hit dicts shaped like Qdrant hits so hybrid_retrieve.py can merge them.
    """
    ensure_sources_indexed(
        source_id=source_id,
        source_type=source_type,
        file_type=file_type,
        allowed_source_ids=allowed_source_ids,
    )

    fts_query = build_fts_query(query, exact_phrases=exact_phrases)
    safe_limit = max(1, min(int(limit), MAX_FTS_RESULTS))

    where = ["chunk_fts MATCH ?"]
    params: list[Any] = [fts_query]

    if source_id:
        where.append("source_id = ?")
        params.append(source_id)

    if source_type:
        where.append("source_type = ?")
        params.append(source_type)

    if file_type:
        where.append("file_type = ?")
        params.append(file_type)

    if allowed_source_ids is not None:
        clean_allowed = [str(item).strip() for item in allowed_source_ids if str(item).strip()]
        if not clean_allowed:
            return []

        placeholders = ",".join(["?"] * len(clean_allowed))
        where.append(f"source_id IN ({placeholders})")
        params.extend(clean_allowed)

    params.append(safe_limit * 2)

    sql = f"""
        SELECT
            text,
            source_id,
            source,
            source_type,
            file_type,
            parser_type,
            page_number,
            page_start,
            page_end,
            chunk_index,
            bm25(chunk_fts) AS rank
        FROM chunk_fts
        WHERE {" AND ".join(where)}
        ORDER BY rank
        LIMIT ?
    """

    hits: list[dict[str, Any]] = []

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    for row in rows:
        if not _page_ok(row, page_start, page_end):
            continue

        try:
            rank = float(row["rank"] or 0.0)
        except (TypeError, ValueError):
            rank = 0.0

        # bm25 lower is better. Convert to a friendly bounded score.
        score = 1.0 / (1.0 + abs(rank))

        hits.append(
            {
                "score": round(score, 6),
                "source": row["source"],
                "source_id": row["source_id"],
                "chunk_index": int(row["chunk_index"] or 0),
                "text": row["text"],
                "file_type": row["file_type"],
                "source_type": row["source_type"],
                "parser_type": row["parser_type"],
                "page_number": row["page_number"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "retrieval_channel": "fts_keyword",
                "fts_rank": rank,
                "fts_query": fts_query,
            }
        )

        if len(hits) >= safe_limit:
            break

    return hits
