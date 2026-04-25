from __future__ import annotations

import os
from typing import Any

from app.context_utils import build_citations, build_context_text, compact_chunks
from app.llm_client import generate_text
from app.prompts import REPO_EXPLANATION_PROMPT
from app.response_formatter import no_evidence_response
from app.retrieve import retrieve_context

REPO_EXPLANATION_LIMIT = int(os.getenv("REPO_EXPLANATION_LIMIT", "10"))
REPO_EXPLANATION_NUM_PREDICT = int(os.getenv("REPO_EXPLANATION_NUM_PREDICT", "1400"))


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
    expanded_question = (
        f"{question}\n"
        "Prioritize README, architecture docs, Terraform files, Docker files, docker-compose files, "
        "GitHub Actions workflows, modules, deployment files, and runbooks."
    )

    chunks = retrieve_context(
        expanded_question,
        limit=REPO_EXPLANATION_LIMIT,
        source_id=source_id,
        source=source,
        source_type=source_type,
        file_type=file_type,
        page_start=page_start,
        page_end=page_end,
    )

    if not chunks:
        return no_evidence_response()

    compacted = compact_chunks(chunks, max_total_chars=12000)
    context_text = build_context_text(compacted)

    prompt = REPO_EXPLANATION_PROMPT.format(
        question=question,
        chat_context=chat_context or "No recent conversation context.",
        context_text=context_text,
    )

    answer = generate_text(
        prompt,
        temperature=0.0,
        num_predict=REPO_EXPLANATION_NUM_PREDICT,
    )

    return {
        "answer": answer,
        "citations": build_citations(compacted),
    }
