from __future__ import annotations

from typing import Any


def no_evidence_response() -> dict[str, Any]:
    return {
        "answer": "No evidence found in the knowledge base.",
        "citations": [],
    }


def build_pipeline_response(
    answer: str,
    citations: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clean_answer = (answer or "").strip() or "No evidence found in the knowledge base."

    payload: dict[str, Any] = {
        "answer": clean_answer,
        "citations": citations or [],
    }

    if extra:
        payload.update(extra)

    return payload
