from __future__ import annotations

import os
from typing import Any

from app.graph_store import GraphStore
from app.qdrant_client import get_chunks_by_refs

GRAPH_CONTEXT_ENABLED_DEFAULT = os.getenv("GRAPH_CONTEXT_ENABLED_DEFAULT", "false").lower() == "true"
GRAPH_SEED_CHUNKS = int(os.getenv("GRAPH_SEED_CHUNKS", "3"))
GRAPH_MAX_NEIGHBOUR_CHUNKS = int(os.getenv("GRAPH_MAX_NEIGHBOUR_CHUNKS", "3"))
GRAPH_MAX_CONTEXT_CHARS = int(os.getenv("GRAPH_MAX_CONTEXT_CHARS", "2500"))

# Keep this strict. next/related_to are intentionally excluded by default.
GRAPH_ALLOWED_EDGES = tuple(
    item.strip()
    for item in os.getenv("GRAPH_ALLOWED_EDGES", "mentions,defines,contains").split(",")
    if item.strip()
)


def _chunk_key(chunk: dict[str, Any]) -> tuple[str, int]:
    source_id = str(chunk.get("source_id") or "")
    try:
        chunk_index = int(chunk.get("chunk_index"))
    except (TypeError, ValueError):
        chunk_index = -1
    return source_id, chunk_index


def _seed_chunks(vector_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []

    for hit in vector_hits:
        source_id, chunk_index = _chunk_key(hit)
        if not source_id or chunk_index < 0:
            continue
        seeds.append(hit)
        if len(seeds) >= GRAPH_SEED_CHUNKS:
            break

    return seeds


def expand_with_graph_context(
    vector_hits: list[dict[str, Any]],
    max_graph_chunks: int = GRAPH_MAX_NEIGHBOUR_CHUNKS,
    max_context_chars: int = GRAPH_MAX_CONTEXT_CHARS,
    allowed_edges: tuple[str, ...] = GRAPH_ALLOWED_EDGES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Adds a small number of graph-neighbour chunks to normal vector retrieval.

    Safe design:
    - uses vector hits as seeds
    - follows only strict edge types by default
    - fetches exact chunk text from Qdrant
    - dedupes against existing vector hits
    - caps graph chunk count and total added characters
    """
    if not vector_hits:
        return vector_hits, {
            "graph_context_enabled": True,
            "graph_chunks_added": 0,
            "graph_reason": "No vector hits to expand.",
        }

    safe_max_chunks = max(0, min(int(max_graph_chunks or 0), 5))
    if safe_max_chunks <= 0:
        return vector_hits, {
            "graph_context_enabled": True,
            "graph_chunks_added": 0,
            "graph_reason": "Graph max chunk cap is zero.",
        }

    seeds = _seed_chunks(vector_hits)
    if not seeds:
        return vector_hits, {
            "graph_context_enabled": True,
            "graph_chunks_added": 0,
            "graph_reason": "No seed chunks with source_id/chunk_index.",
        }

    existing_keys = {_chunk_key(hit) for hit in vector_hits}

    store = GraphStore()
    refs = store.get_related_chunk_refs(
        seed_chunks=[
            {
                "source_id": seed.get("source_id"),
                "chunk_index": seed.get("chunk_index"),
            }
            for seed in seeds
        ],
        allowed_edges=allowed_edges,
        max_refs=safe_max_chunks * 3,
    )

    refs = [
        ref for ref in refs
        if (str(ref.get("source_id") or ""), int(ref.get("chunk_index", -1))) not in existing_keys
    ]

    if not refs:
        return vector_hits, {
            "graph_context_enabled": True,
            "graph_chunks_added": 0,
            "graph_reason": "No safe graph neighbours found.",
        }

    fetched = get_chunks_by_refs(refs[: safe_max_chunks * 3])

    by_ref = {
        (str(ref.get("source_id") or ""), int(ref.get("chunk_index", -1))): ref
        for ref in refs
    }

    graph_chunks: list[dict[str, Any]] = []
    total_chars = 0

    for chunk in fetched:
        key = _chunk_key(chunk)
        if key in existing_keys:
            continue

        text = str(chunk.get("text") or "")
        if not text.strip():
            continue

        if graph_chunks and total_chars + len(text) > max_context_chars:
            break

        ref = by_ref.get(key, {})
        item = dict(chunk)
        item["retrieval_mode"] = "graph_context"
        item["graph_context"] = True
        item["graph_reason"] = ref.get("reason", "graph_neighbour")
        item["graph_edge_type"] = ref.get("edge_type")
        item["graph_connector_label"] = ref.get("connector_label")
        item["score"] = min(float(item.get("score", 0.75) or 0.75), 0.92)

        graph_chunks.append(item)
        existing_keys.add(key)
        total_chars += len(text)

        if len(graph_chunks) >= safe_max_chunks:
            break

    combined = vector_hits + graph_chunks

    return combined, {
        "graph_context_enabled": True,
        "graph_chunks_added": len(graph_chunks),
        "graph_reason": (
            f"Added {len(graph_chunks)} graph neighbour chunk(s) using edges: "
            f"{', '.join(allowed_edges)}."
        ),
    }
