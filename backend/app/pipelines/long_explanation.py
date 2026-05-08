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
COMPARISON_NUM_PREDICT = int(os.getenv("COMPARISON_NUM_PREDICT", "700"))


def _chunk_meta(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if not chunks:
        return {
            "retriever_used": None,
            "retrieval_mode": None,
            "query_shape": None,
            "reranker_used": None,
            "retrieval_speed": None,
            "primary_entity": None,
            "comparison_entities": [],
        }

    first = chunks[0]
    return {
        "retriever_used": first.get("retriever_used"),
        "retrieval_mode": first.get("retrieval_mode"),
        "query_shape": first.get("query_shape"),
        "reranker_used": first.get("reranker_used"),
        "retrieval_speed": first.get("retrieval_speed"),
        "primary_entity": first.get("primary_entity"),
        "comparison_entities": first.get("comparison_entities") or [],
    }


def _with_meta(payload: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    return {**payload, **_chunk_meta(chunks)}


def _is_comparison(chunks: list[dict[str, Any]]) -> bool:
    if not chunks:
        return False
    first = chunks[0]
    entities = first.get("comparison_entities") or []
    return first.get("query_shape") == "comparison" and len(entities) >= 2


def _comparison_prompt(
    *,
    question: str,
    chat_context: str,
    context_text: str,
    chunks: list[dict[str, Any]],
) -> str:
    """
    Strict generic comparison prompt.

    No domain-specific hardcoding.
    Works for any comparison query where retrieval provides comparison_entities.
    """
    first = chunks[0] if chunks else {}
    entities = first.get("comparison_entities") or []
    entity_a = str(entities[0]) if len(entities) > 0 else "Item A"
    entity_b = str(entities[1]) if len(entities) > 1 else "Item B"

    counts = first.get("comparison_entity_counts") or {}
    weak_entities = first.get("comparison_weak_entities") or []

    weak_instruction = ""
    if weak_entities:
        weak_instruction = (
            "\nIf the retrieved context has weak or missing evidence for one compared item, "
            "say that clearly. Do not invent missing details."
        )

    return f"""
You are answering a retrieval-grounded comparison question.

Question:
{question}

Recent chat context:
{chat_context or "No recent conversation context."}

Compared items:
- {entity_a}
- {entity_b}

Evidence balance diagnostics:
{counts}

Retrieved context:
{context_text}

Instructions:
- Answer the user's comparison question directly.
- Use only the retrieved context.
- Do not dump or restate the context line-by-line.
- Do not write a long numbered list unless the user asked for one.
- The answer must discuss both compared items: {entity_a} and {entity_b}.
- Start with a short direct comparison.
- Then give:
  1. What {entity_a} means according to the evidence.
  2. What {entity_b} means according to the evidence.
  3. How the evidence relates them or distinguishes them.
- If the context explains that one term is substituted for, equivalent to, or philosophically different from the other, say that clearly.
- If the evidence mainly supports one item and only weakly supports the other, say that clearly.
- Do not shift the main comparison to a third concept unless the context uses that third concept to explain the relationship between {entity_a} and {entity_b}.
- Keep the answer concise but complete.
{weak_instruction}

Final answer:
""".strip()


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
        return _with_meta(no_evidence_response(), chunks)

    top_score = max(float(c.get("score", 0.0) or 0.0) for c in chunks)
    if top_score < MIN_SCORE_THRESHOLD:
        return _with_meta(no_evidence_response(), chunks)

    compacted = compact_chunks(chunks, max_total_chars=11000)
    citations = build_citations(compacted)
    context_text = build_context_text(compacted)

    if _is_comparison(compacted):
        prompt = _comparison_prompt(
            question=question,
            chat_context=chat_context,
            context_text=context_text,
            chunks=compacted,
        )
        num_predict = COMPARISON_NUM_PREDICT
    else:
        prompt = LONG_EXPLANATION_PROMPT.format(
            question=question,
            chat_context=chat_context or "No recent conversation context.",
            context_text=context_text,
        )
        num_predict = LONG_EXPLANATION_NUM_PREDICT

    answer = generate_text(
        prompt,
        temperature=0.0,
        num_predict=num_predict,
    )

    if answer.strip() == "No evidence found in the knowledge base.":
        return _with_meta(no_evidence_response(), compacted)

    return {
        "answer": answer,
        "citations": citations,
        "verification_context_text": context_text,
        **_chunk_meta(compacted),
    }
