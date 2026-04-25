from __future__ import annotations

import logging
import math
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "8"))
RERANKER_MAX_TEXT_CHARS = int(os.getenv("RERANKER_MAX_TEXT_CHARS", "1800"))


def _sigmoid(value: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-float(value)))
    except OverflowError:
        return 0.0 if value < 0 else 1.0


@lru_cache(maxsize=1)
def get_reranker_model():
    """
    Lazily loads the local BGE reranker.

    Important:
    - First request can be slow because the model loads into memory.
    - If the model is not available, caller should gracefully fall back.
    """
    from sentence_transformers import CrossEncoder

    logger.info("Loading reranker model: %s", RERANKER_MODEL)
    return CrossEncoder(RERANKER_MODEL)


def _prepare_pair(question: str, hit: dict[str, Any]) -> tuple[str, str]:
    source = hit.get("source", "") or ""
    text = hit.get("text", "") or ""

    chunk_text = f"Source: {source}\n{text}".strip()
    if len(chunk_text) > RERANKER_MAX_TEXT_CHARS:
        chunk_text = chunk_text[:RERANKER_MAX_TEXT_CHARS].rstrip()

    return question, chunk_text


def rerank_hits(
    question: str,
    hits: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    """
    Real cross-encoder reranking.

    Input:
    - question
    - candidate chunks from Qdrant/hybrid retrieval

    Output:
    - top_n chunks ordered by BGE relevance score

    Safe behavior:
    - If reranker is disabled or fails, return existing ranking.
    """
    if not RERANKER_ENABLED:
        return hits[:top_n]

    if not hits:
        return []

    try:
        model = get_reranker_model()
        pairs = [_prepare_pair(question, hit) for hit in hits]

        raw_scores = model.predict(
            pairs,
            batch_size=RERANKER_BATCH_SIZE,
            show_progress_bar=False,
        )

        reranked: list[dict[str, Any]] = []

        for hit, raw_score in zip(hits, raw_scores):
            item = dict(hit)
            raw = float(raw_score)

            item["reranker_raw_score"] = raw
            item["reranker_score"] = round(_sigmoid(raw), 6)
            item["pre_rerank_score"] = float(hit.get("score", 0.0) or 0.0)

            # Final score favors the real reranker but keeps weak signal from hybrid score.
            item["score"] = round(
                (0.85 * item["reranker_score"]) + (0.15 * item["pre_rerank_score"]),
                6,
            )

            reranked.append(item)

        reranked.sort(
            key=lambda x: (
                x.get("score", 0.0),
                x.get("reranker_score", 0.0),
                x.get("pre_rerank_score", 0.0),
            ),
            reverse=True,
        )

        return reranked[:top_n]

    except Exception as exc:
        logger.exception("BGE reranker failed; falling back to hybrid ranking: %s", exc)
        return hits[:top_n]
