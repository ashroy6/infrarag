from __future__ import annotations

import logging
import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    Range,
    VectorParams,
)

logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "infrarag_docs")

_SCROLL_BATCH_SIZE = 1000
_DELETE_BATCH_SIZE = 500


def get_client() -> QdrantClient:
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        check_compatibility=False,
    )


def collection_exists(client: QdrantClient) -> bool:
    try:
        collections = [c.name for c in client.get_collections().collections]
        return QDRANT_COLLECTION in collections
    except Exception:
        logger.exception("Failed to check Qdrant collections")
        return False


def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    if collection_exists(client):
        return

    logger.info("Creating Qdrant collection '%s' with vector size %s", QDRANT_COLLECTION, vector_size)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def _build_filter(
    source_id: str | None = None,
    source: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> Filter | None:
    conditions: list[FieldCondition] = []

    if source_id:
        conditions.append(
            FieldCondition(
                key="source_id",
                match=MatchValue(value=source_id),
            )
        )

    if source:
        conditions.append(
            FieldCondition(
                key="source",
                match=MatchValue(value=source),
            )
        )

    if source_type:
        conditions.append(
            FieldCondition(
                key="source_type",
                match=MatchValue(value=source_type),
            )
        )

    if file_type:
        conditions.append(
            FieldCondition(
                key="file_type",
                match=MatchValue(value=file_type),
            )
        )

    if page_start is not None:
        conditions.append(
            FieldCondition(
                key="page_number",
                range=Range(gte=page_start),
            )
        )

    if page_end is not None:
        conditions.append(
            FieldCondition(
                key="page_number",
                range=Range(lte=page_end),
            )
        )

    if not conditions:
        return None

    return Filter(must=conditions)


def search(
    query_vector: list[float],
    limit: int = 5,
    source_id: str | None = None,
    source: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[dict[str, Any]]:
    client = get_client()

    if not collection_exists(client):
        logger.info("Qdrant collection '%s' does not exist yet; returning no search results", QDRANT_COLLECTION)
        return []

    search_filter = _build_filter(
        source_id=source_id,
        source=source,
        source_type=source_type,
        file_type=file_type,
        page_start=page_start,
        page_end=page_end,
    )

    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=search_filter,
        limit=limit,
    )

    hits: list[dict[str, Any]] = []
    for item in response.points:
        payload = item.payload or {}
        hits.append(
            {
                "score": item.score,
                "source": payload.get("source", "unknown"),
                "source_id": payload.get("source_id", ""),
                "chunk_index": payload.get("chunk_index", -1),
                "text": payload.get("text", ""),
                "file_type": payload.get("file_type", ""),
                "source_type": payload.get("source_type", ""),
                "parser_type": payload.get("parser_type", ""),
                "page_number": payload.get("page_number"),
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
            }
        )

    return hits


def _scroll_source_point_ids(client: QdrantClient, source_id: str) -> list[Any]:
    point_ids: list[Any] = []
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source_id",
                        match=MatchValue(value=source_id),
                    )
                ]
            ),
            limit=_SCROLL_BATCH_SIZE,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )

        point_ids.extend([p.id for p in points if p.id is not None])

        if next_offset is None:
            break

        offset = next_offset

    return point_ids


def delete_points_by_source_id(source_id: str) -> None:
    client = get_client()

    if not collection_exists(client):
        logger.info(
            "Qdrant collection '%s' does not exist yet; skipping delete for source_id=%s",
            QDRANT_COLLECTION,
            source_id,
        )
        return

    ids = _scroll_source_point_ids(client, source_id)

    if not ids:
        return

    logger.info("Deleting %s Qdrant points for source_id=%s", len(ids), source_id)

    for start in range(0, len(ids), _DELETE_BATCH_SIZE):
        batch = ids[start:start + _DELETE_BATCH_SIZE]
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=PointIdsList(points=batch),
        )
