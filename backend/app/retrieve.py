import os

import requests
from qdrant_client import QdrantClient

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "infrarag_docs")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def get_embedding(text: str):
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={
            "model": EMBED_MODEL,
            "prompt": text
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["embedding"]


def retrieve_context(query: str, limit: int = 5):
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        check_compatibility=False,
    )

    query_vector = get_embedding(query)

    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
    )

    hits = []
    for item in response.points:
        payload = item.payload or {}
        hits.append({
            "score": item.score,
            "source": payload.get("source", "unknown"),
            "chunk_index": payload.get("chunk_index", -1),
            "text": payload.get("text", "")
        })

    return hits
