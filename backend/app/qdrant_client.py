from __future__ import annotations

import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    VectorParams,
)

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "infrarag_docs")


def get_client() -> QdrantClient:
    return QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        check_compatibility=False,
    )


def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def search(query_vector: list[float], limit: int = 5) -> list[dict[str, Any]]:
    client = get_client()
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
    )

    hits = []
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
            }
        )
    return hits


def delete_points_by_source_id(source_id: str) -> None:
    client = get_client()

    points, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="source_id",
                    match=MatchValue(value=source_id),
                )
            ]
        ),
        limit=10000,
        with_payload=False,
        with_vectors=False,
    )

    ids = [p.id for p in points]

    if ids:
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=PointIdsList(points=ids),
        )
