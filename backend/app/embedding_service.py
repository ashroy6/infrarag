from __future__ import annotations

import os

import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def get_embedding(text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": text,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    embeddings = data.get("embeddings", [])
    if not embeddings:
        raise RuntimeError("No embeddings returned from Ollama")

    return embeddings[0]
