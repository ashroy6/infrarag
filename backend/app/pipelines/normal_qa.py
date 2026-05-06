from __future__ import annotations

import os
from typing import Any

from app.answer_verifier import answer_denies_evidence, should_verify_answer, verify_answer
from app.context_utils import build_citations, build_context_text, compact_chunks
from app.llm_client import generate_text
from app.prompts import DENIAL_RECOVERY_PROMPT, NORMAL_QA_PROMPT
from app.response_formatter import no_evidence_response
from app.retrieve import retrieve_context

MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.35"))
NORMAL_QA_LIMIT = int(os.getenv("NORMAL_QA_LIMIT", "6"))
NORMAL_QA_NUM_PREDICT = int(os.getenv("NORMAL_QA_NUM_PREDICT", "500"))


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
        limit=NORMAL_QA_LIMIT,
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

    compacted = compact_chunks(chunks, max_total_chars=7000)
    citations = build_citations(compacted)
    context_text = build_context_text(compacted)

    prompt = NORMAL_QA_PROMPT.format(
        question=question,
        context_text=context_text,
    )

    answer = generate_text(
        prompt,
        temperature=0.0,
        num_predict=NORMAL_QA_NUM_PREDICT,
    )

    verification_result: dict[str, Any] = {
        "verification_verdict": "skipped",
        "unsupported_claims": [],
        "verification_reason": "Verifier skipped for this answer.",
        "verified": False,
    }

    if should_verify_answer(
        pipeline_used="normal_qa",
        routing={"source_strategy": "cluster_by_best_source"},
        answer=answer,
        citations=citations,
    ):
        verification_result = verify_answer(
            question=question,
            pipeline_used="normal_qa",
            context_text=context_text,
            draft_answer=answer,
        )

        verdict = verification_result.get("verification_verdict")
        corrected = str(verification_result.get("corrected_answer") or "").strip()

        if verdict in {"needs_revision", "insufficient_evidence"} and corrected:
            answer = corrected

    # Generic recovery:
    # If retrieval produced citations but the model still denies evidence,
    # run a stricter second pass before giving up.
    if answer_denies_evidence(answer, citations):
        recovery_prompt = DENIAL_RECOVERY_PROMPT.format(
            question=question,
            context_text=context_text,
        )
        recovered = generate_text(
            recovery_prompt,
            temperature=0.0,
            num_predict=NORMAL_QA_NUM_PREDICT,
        ).strip()

        if recovered and not answer_denies_evidence(recovered, citations):
            answer = recovered
            verification_result = {
                **verification_result,
                "verification_verdict": "recovered",
                "verification_reason": "Second-pass evidence recovery corrected a false no-evidence answer.",
                "verified": False,
            }

    # Important:
    # If retrieval produced citations, do not hide them just because the model failed.
    # Returning citations keeps the system debuggable and allows the UI/user to inspect evidence.
    if answer.strip() == "No evidence found in the knowledge base." and citations:
        return {
            "answer": (
                "Relevant evidence was retrieved, but the answer model failed to extract a supported answer from it. "
                "Open the citations to inspect the retrieved chunks, or ask a narrower question."
            ),
            "citations": citations,
            "verification_context_text": context_text,
            "verification_verdict": "answer_generation_failed_with_evidence",
            "unsupported_claims": [],
            "verification_reason": "Model returned no-evidence despite retrieved citations.",
            "verified": False,
        }

    if answer_denies_evidence(answer, citations) and citations:
        return {
            "answer": (
                "Relevant evidence was retrieved, but the answer model denied or failed to use it. "
                "Open the citations to inspect the retrieved chunks, or ask a narrower question."
            ),
            "citations": citations,
            "verification_context_text": context_text,
            "verification_verdict": "answer_denied_retrieved_evidence",
            "unsupported_claims": [],
            "verification_reason": "Model denied evidence despite retrieved citations.",
            "verified": False,
        }

    if answer.strip() == "No evidence found in the knowledge base.":
        return no_evidence_response()

    if answer_denies_evidence(answer, citations):
        return no_evidence_response()

    return {
        "answer": answer,
        "citations": citations,
        "verification_context_text": context_text,
        **verification_result,
    }
