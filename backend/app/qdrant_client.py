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
    allowed_source_ids: list[str] | None = None,
) -> Filter | None:
    conditions: list[FieldCondition] = []
    should_conditions: list[FieldCondition] = []

    clean_allowed_source_ids = [
        str(item).strip()
        for item in (allowed_source_ids or [])
        if str(item).strip()
    ]

    # If caller explicitly passes an empty allow-list, return no results.
    if allowed_source_ids is not None and not clean_allowed_source_ids:
        conditions.append(
            FieldCondition(
                key="source_id",
                match=MatchValue(value="__NO_ALLOWED_SOURCES__"),
            )
        )

    # If a source_id is requested inside an allow-list, enforce intersection.
    # If source_id is outside the allow-list, force no result.
    if source_id and clean_allowed_source_ids and source_id not in clean_allowed_source_ids:
        conditions.append(
            FieldCondition(
                key="source_id",
                match=MatchValue(value="__SOURCE_NOT_ALLOWED__"),
            )
        )

    if not source_id and len(clean_allowed_source_ids) == 1:
        conditions.append(
            FieldCondition(
                key="source_id",
                match=MatchValue(value=clean_allowed_source_ids[0]),
            )
        )

    if not source_id and len(clean_allowed_source_ids) > 1:
        should_conditions.extend(
            FieldCondition(key="source_id", match=MatchValue(value=item))
            for item in clean_allowed_source_ids
        )

    if source_id:
        conditions.append(FieldCondition(key="source_id", match=MatchValue(value=source_id)))

    if source:
        conditions.append(FieldCondition(key="source", match=MatchValue(value=source)))

    if source_type:
        conditions.append(FieldCondition(key="source_type", match=MatchValue(value=source_type)))

    if file_type:
        conditions.append(FieldCondition(key="file_type", match=MatchValue(value=file_type)))

    if page_start is not None:
        conditions.append(FieldCondition(key="page_number", range=Range(gte=page_start)))

    if page_end is not None:
        conditions.append(FieldCondition(key="page_number", range=Range(lte=page_end)))

    if not conditions and not should_conditions:
        return None

    if should_conditions:
        return Filter(must=conditions, should=should_conditions)

    return Filter(must=conditions)


def _payload_to_hit(item: Any) -> dict[str, Any]:
    payload = item.payload or {}

    return {
        "score": getattr(item, "score", 1.0) or 1.0,
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
        "tenant_id": payload.get("tenant_id"),
        "owner_user_id": payload.get("owner_user_id"),
        "source_group": payload.get("source_group"),
        "connector": payload.get("connector"),
        "data_domain": payload.get("data_domain"),
        "security_level": payload.get("security_level"),
        "tags": payload.get("tags", []),
        "agent_access_enabled": payload.get("agent_access_enabled"),
        "knowledge_source_id": payload.get("knowledge_source_id"),
        "chunk_strategy": payload.get("chunk_strategy"),
        "reference_labels": payload.get("reference_labels", []),
        "references": payload.get("references", []),
        "section_number": payload.get("section_number", ""),
        "section_start": payload.get("section_start", ""),
        "section_end": payload.get("section_end", ""),
        "subsection_start": payload.get("subsection_start", ""),
        "subsection_end": payload.get("subsection_end", ""),
        "reference_type": payload.get("reference_type", ""),
        "heading": payload.get("heading", ""),
        "parent_title": payload.get("parent_title", ""),
        "parent_id": payload.get("parent_id", ""),
        "heading_path": payload.get("heading_path", ""),
        "prev_chunk_index": payload.get("prev_chunk_index"),
        "next_chunk_index": payload.get("next_chunk_index"),
    }


def search(
    query_vector: list[float],
    limit: int = 5,
    source_id: str | None = None,
    source: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    allowed_source_ids: list[str] | None = None,
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
        allowed_source_ids=allowed_source_ids,
    )

    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=search_filter,
        limit=limit,
    )

    hits: list[dict[str, Any]] = []
    for item in response.points:
        hits.append(_payload_to_hit(item))

    return hits


def get_chunks_by_source_id(source_id: str) -> list[dict[str, Any]]:
    """
    Loads all chunk payloads for a source from Qdrant.

    Used by full document/book summarisation.
    This reads the actual full chunk text from Qdrant payload, not only SQLite previews.
    """
    client = get_client()

    if not collection_exists(client):
        logger.info("Qdrant collection '%s' does not exist yet; returning no chunks", QDRANT_COLLECTION)
        return []

    chunks: list[dict[str, Any]] = []
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
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            chunks.append(_payload_to_hit(point))

        if next_offset is None:
            break

        offset = next_offset

    chunks.sort(key=lambda item: int(item.get("chunk_index", 0)))
    return chunks


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



def get_chunk_by_source_id_and_index(source_id: str, chunk_index: int) -> dict[str, Any] | None:
    """
    Return one exact chunk from Qdrant by source_id + chunk_index.

    Used by clickable citations in the frontend evidence modal.
    """
    client = get_client()

    if not collection_exists(client):
        logger.info("Qdrant collection '%s' does not exist yet; returning no chunk", QDRANT_COLLECTION)
        return None

    points, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="source_id",
                    match=MatchValue(value=source_id),
                ),
                FieldCondition(
                    key="chunk_index",
                    match=MatchValue(value=chunk_index),
                ),
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        return None

    return _payload_to_hit(points[0])


def get_chunks_by_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Fetches exact chunks by source_id + chunk_index.

    Kept intentionally simple because graph context should only fetch a few chunks.
    """
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for ref in refs:
        source_id = str(ref.get("source_id") or "").strip()
        try:
            chunk_index = int(ref.get("chunk_index"))
        except (TypeError, ValueError):
            continue

        key = (source_id, chunk_index)
        if not source_id or chunk_index < 0 or key in seen:
            continue

        seen.add(key)
        chunk = get_chunk_by_source_id_and_index(source_id, chunk_index)
        if chunk:
            results.append(chunk)

    return results


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
