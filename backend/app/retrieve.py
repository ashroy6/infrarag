from __future__ import annotations

from app.embedding_service import get_embedding
from app.qdrant_client import search


def retrieve_context(query: str, limit: int = 5):
    query_vector = get_embedding(query)
    return search(query_vector=query_vector, limit=limit)
