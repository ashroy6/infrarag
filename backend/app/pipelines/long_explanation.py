from __future__ import annotations

import os
from typing import Any

from app.context_utils import build_citations, build_context_text, compact_chunks
from app.llm_client import generate_text
from app.prompts import LONG_EXPLANATION_PROMPT
from app.response_formatter import no_evidence_response
from app.retrieve import retrieve_context

MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.35"))
LONG_EXPLANATION_LIMIT = int(os.getenv("LONG_EXPLANATION_LIMIT", "10"))
LONG_EXPLANATION_NUM_PREDICT = int(os.getenv("LONG_EXPLANATION_NUM_PREDICT", "1400"))


def run(
    question: str,
    chat_context: str = "",
    source_id: str | None = None,
    source: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> dict[str, Any]:
    chunks = retrieve_context(
        question,
        limit=LONG_EXPLANATION_LIMIT,
        source_id=source_id,
        source=source,
        source_type=source_type,
        file_type=file_type,
        page_start=page_start,
        page_end=page_end,
    )

    if not chunks:
        return no_evidence_response()

    top_score = max(float(c.get("score", 0.0) or 0.0) for c in chunks)
    if top_score < MIN_SCORE_THRESHOLD:
        return no_evidence_response()

    compacted = compact_chunks(chunks, max_total_chars=11000)
    citations = build_citations(compacted)
    context_text = build_context_text(compacted)

    prompt = LONG_EXPLANATION_PROMPT.format(
        question=question,
        chat_context=chat_context or "No recent conversation context.",
        context_text=context_text,
    )

    answer = generate_text(
        prompt,
        temperature=0.0,
        num_predict=LONG_EXPLANATION_NUM_PREDICT,
    )

    if answer.strip() == "No evidence found in the knowledge base.":
        return no_evidence_response()

    return {
        "answer": answer,
        "citations": citations,
        "verification_context_text": context_text,
    }
