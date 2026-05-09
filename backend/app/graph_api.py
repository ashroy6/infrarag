from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Query

from app.graph_store import GraphStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["graph"])

GRAPH_REBUILD_MAX_CHUNKS_PER_SOURCE = int(os.getenv("GRAPH_REBUILD_MAX_CHUNKS_PER_SOURCE", "1200"))
GRAPH_REBUILD_MAX_TEXT_CHARS_PER_CHUNK = int(os.getenv("GRAPH_REBUILD_MAX_TEXT_CHARS_PER_CHUNK", "4000"))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalise_graph_chunks(
    chunks: list[dict[str, Any]],
    *,
    source_id: str,
    max_chunks: int,
) -> tuple[list[dict[str, Any]], bool, int]:
    """
    Converts Qdrant chunk payloads into graph extractor input.

    Graph rebuild must be bounded. Large files such as Bible/book corpora can
    contain thousands of chunks; trying to graph every chunk in one HTTP request
    can timeout or make the UI unusable.
    """
    sorted_chunks = sorted(chunks, key=lambda item: _safe_int(item.get("chunk_index"), 0))
    original_count = len(sorted_chunks)
    safe_max = max(1, int(max_chunks or GRAPH_REBUILD_MAX_CHUNKS_PER_SOURCE))
    limited_chunks = sorted_chunks[:safe_max]
    partial = original_count > len(limited_chunks)

    graph_chunks: list[dict[str, Any]] = []
    for chunk in limited_chunks:
        chunk_index = _safe_int(chunk.get("chunk_index"), 0)
        text = str(chunk.get("text") or "")
        if not text.strip():
            continue

        graph_chunks.append(
            {
                "chunk_id": f"{source_id}:{chunk_index}",
                "chunk_index": chunk_index,
                "qdrant_point_id": str(chunk.get("qdrant_point_id") or ""),
                "text": text[:GRAPH_REBUILD_MAX_TEXT_CHARS_PER_CHUNK],
                "page_number": chunk.get("page_number"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
            }
        )

    return graph_chunks, partial, original_count


def _rebuild_one_source(
    record: dict[str, Any],
    *,
    max_chunks_per_source: int,
) -> dict[str, Any]:
    from app.graph_extractor import build_graph_for_file
    from app.qdrant_client import get_chunks_by_source_id

    source_id = str(record.get("source_id") or "").strip()
    if not source_id:
        return {
            "status": "skipped",
            "reason": "missing_source_id",
            "source_path": record.get("source_path", ""),
        }

    chunks = get_chunks_by_source_id(source_id)
    if not chunks:
        return {
            "status": "skipped",
            "reason": "no_chunks",
            "source_id": source_id,
            "source_path": record.get("source_path", ""),
        }

    graph_chunks, partial, original_chunk_count = _normalise_graph_chunks(
        chunks,
        source_id=source_id,
        max_chunks=max_chunks_per_source,
    )

    if not graph_chunks:
        return {
            "status": "skipped",
            "reason": "no_non_empty_chunks",
            "source_id": source_id,
            "source_path": record.get("source_path", ""),
            "original_chunks": original_chunk_count,
        }

    graph_payload = build_graph_for_file(
        source_id=source_id,
        source_type=record.get("source_type", ""),
        source_path=record.get("source_path", ""),
        file_type=record.get("file_type", ""),
        parser_type=record.get("parser_type", ""),
        chunks=graph_chunks,
    )

    store = GraphStore()
    store.replace_source_graph(
        source_id=source_id,
        nodes=graph_payload.get("nodes", []),
        edges=graph_payload.get("edges", []),
    )

    return {
        "status": "rebuilt_partial" if partial else "rebuilt",
        "source_id": source_id,
        "source_path": record.get("source_path", ""),
        "original_chunks": original_chunk_count,
        "graph_chunks_used": len(graph_chunks),
        "partial": partial,
        "nodes": len(graph_payload.get("nodes", [])),
        "edges": len(graph_payload.get("edges", [])),
    }


@router.get("/stats")
def graph_stats() -> dict[str, Any]:
    store = GraphStore()
    return {"stats": store.stats()}


@router.get("")
def get_graph(
    mode: str = Query("structured", pattern="^(structured|dense)$"),
    source_id: str | None = None,
    q: str | None = None,
    node_type: str | None = None,
    edge_type: str | None = None,
    limit: int = 600,
) -> dict[str, Any]:
    store = GraphStore()
    return store.get_graph(
        mode=mode,
        source_id=source_id,
        q=q,
        node_type=node_type,
        edge_type=edge_type,
        limit=limit,
    )


@router.get("/search")
def search_graph(
    q: str,
    mode: str = Query("structured", pattern="^(structured|dense)$"),
    source_id: str | None = None,
    limit: int = 600,
) -> dict[str, Any]:
    store = GraphStore()
    return store.get_graph(
        mode=mode,
        source_id=source_id,
        q=q,
        limit=limit,
    )


@router.get("/source/{source_id}")
def graph_for_source(
    source_id: str,
    mode: str = Query("structured", pattern="^(structured|dense)$"),
    limit: int = 600,
) -> dict[str, Any]:
    store = GraphStore()
    return store.get_graph(
        mode=mode,
        source_id=source_id,
        limit=limit,
    )


@router.post("/rebuild-source/{source_id}")
def rebuild_graph_for_source(
    source_id: str,
    max_chunks_per_source: int = GRAPH_REBUILD_MAX_CHUNKS_PER_SOURCE,
) -> dict[str, Any]:
    from app.metadata_db import MetadataDB

    metadata_db = MetadataDB()
    record = metadata_db.get_file(source_id)

    if not record:
        return {
            "status": "not_found",
            "source_id": source_id,
            "message": "Source not found in metadata DB",
        }

    try:
        return _rebuild_one_source(
            dict(record),
            max_chunks_per_source=max_chunks_per_source,
        )
    except Exception as exc:
        logger.exception("Graph rebuild failed for source_id=%s", source_id)
        return {
            "status": "failed",
            "source_id": source_id,
            "source_path": dict(record).get("source_path", ""),
            "error": str(exc),
        }


@router.post("/rebuild")
def rebuild_graph_all(
    limit: int = 50,
    max_chunks_per_source: int = GRAPH_REBUILD_MAX_CHUNKS_PER_SOURCE,
) -> dict[str, Any]:
    from app.metadata_db import MetadataDB

    safe_limit = max(1, min(int(limit or 50), 500))
    safe_max_chunks = max(1, min(int(max_chunks_per_source or GRAPH_REBUILD_MAX_CHUNKS_PER_SOURCE), 5000))

    metadata_db = MetadataDB()
    store = GraphStore()

    files = metadata_db.list_active_files()[:safe_limit]
    rebuilt: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for record in files:
        record_dict = dict(record)
        source_id = str(record_dict.get("source_id") or "").strip()

        try:
            result = _rebuild_one_source(
                record_dict,
                max_chunks_per_source=safe_max_chunks,
            )

            if result.get("status", "").startswith("rebuilt"):
                rebuilt.append(result)
            else:
                skipped.append(result)

        except Exception as exc:
            logger.exception("Graph rebuild failed for source_id=%s", source_id)
            failed.append(
                {
                    "status": "failed",
                    "source_id": source_id,
                    "source_path": record_dict.get("source_path", ""),
                    "error": str(exc),
                }
            )
            continue

    stats = store.stats()
    return {
        "status": "complete_with_failures" if failed else "complete",
        "limit": safe_limit,
        "max_chunks_per_source": safe_max_chunks,
        "rebuilt_count": len(rebuilt),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "rebuilt": rebuilt,
        "skipped": skipped,
        "failed": failed,
        "stats": stats,
    }
