from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.graph_store import GraphStore

router = APIRouter(prefix="/graph", tags=["graph"])


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
def rebuild_graph_for_source(source_id: str) -> dict[str, Any]:
    from app.graph_extractor import build_graph_for_file
    from app.metadata_db import MetadataDB
    from app.qdrant_client import get_chunks_by_source_id

    metadata_db = MetadataDB()
    record = metadata_db.get_file(source_id)

    if not record:
        return {
            "status": "not_found",
            "source_id": source_id,
            "message": "Source not found in metadata DB",
        }

    chunks = get_chunks_by_source_id(source_id)

    if not chunks:
        return {
            "status": "no_chunks",
            "source_id": source_id,
            "message": "No Qdrant chunks found for this source",
        }

    graph_chunks = []
    for chunk in chunks:
        graph_chunks.append(
            {
                "chunk_id": f"{source_id}:{chunk.get('chunk_index')}",
                "chunk_index": int(chunk.get("chunk_index") or 0),
                "qdrant_point_id": "",
                "text": chunk.get("text", ""),
                "page_number": chunk.get("page_number"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
            }
        )

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
        "status": "rebuilt",
        "source_id": source_id,
        "source_path": record.get("source_path", ""),
        "chunks": len(graph_chunks),
        "nodes": len(graph_payload.get("nodes", [])),
        "edges": len(graph_payload.get("edges", [])),
    }


@router.post("/rebuild")
def rebuild_graph_all(limit: int = 50) -> dict[str, Any]:
    from app.graph_extractor import build_graph_for_file
    from app.metadata_db import MetadataDB
    from app.qdrant_client import get_chunks_by_source_id

    safe_limit = max(1, min(int(limit or 50), 500))
    metadata_db = MetadataDB()
    store = GraphStore()

    files = metadata_db.list_active_files()[:safe_limit]
    rebuilt = []
    skipped = []

    for record in files:
        source_id = record.get("source_id")
        if not source_id:
            continue

        chunks = get_chunks_by_source_id(source_id)

        if not chunks:
            skipped.append(
                {
                    "source_id": source_id,
                    "source_path": record.get("source_path", ""),
                    "reason": "no_chunks",
                }
            )
            continue

        graph_chunks = []
        for chunk in chunks:
            graph_chunks.append(
                {
                    "chunk_id": f"{source_id}:{chunk.get('chunk_index')}",
                    "chunk_index": int(chunk.get("chunk_index") or 0),
                    "qdrant_point_id": "",
                    "text": chunk.get("text", ""),
                    "page_number": chunk.get("page_number"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                }
            )

        graph_payload = build_graph_for_file(
            source_id=source_id,
            source_type=record.get("source_type", ""),
            source_path=record.get("source_path", ""),
            file_type=record.get("file_type", ""),
            parser_type=record.get("parser_type", ""),
            chunks=graph_chunks,
        )

        store.replace_source_graph(
            source_id=source_id,
            nodes=graph_payload.get("nodes", []),
            edges=graph_payload.get("edges", []),
        )

        rebuilt.append(
            {
                "source_id": source_id,
                "source_path": record.get("source_path", ""),
                "chunks": len(graph_chunks),
                "nodes": len(graph_payload.get("nodes", [])),
                "edges": len(graph_payload.get("edges", [])),
            }
        )

    return {
        "status": "complete",
        "limit": safe_limit,
        "rebuilt_count": len(rebuilt),
        "skipped_count": len(skipped),
        "rebuilt": rebuilt,
        "skipped": skipped,
        "stats": store.stats(),
    }
