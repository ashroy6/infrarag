from __future__ import annotations

import re
import time
from typing import Any

from app.answer_verifier import should_verify_answer, verify_answer
from app.audit import save_audit_event
from app.chat_history import (
    add_assistant_message,
    add_user_message,
    ensure_conversation,
    get_recent_chat_context,
)
from app.llm_client import CHAT_MODEL
from app.followup_resolver import resolve_followup_question
from app.router import decide_intent
from app.rag_metrics import (
    RAG_ANSWER_LATENCY_SECONDS,
    RAG_NO_EVIDENCE_TOTAL,
    RAG_PIPELINE_TOTAL,
    RAG_PLANNER_FALLBACK_TOTAL,
    RAG_QUESTIONS_TOTAL,
    RAG_VERIFIER_TOTAL,
)

from app.pipelines import document_summary
from app.pipelines import incident_runbook
from app.pipelines import long_explanation
from app.pipelines import normal_qa
from app.pipelines import repo_explanation


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", (question or "").strip())


def _run_selected_pipeline(
    pipeline_used: str,
    question: str,
    chat_context: str,
    source_id: str | None = None,
    source: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> dict[str, Any]:
    common_args = {
        "question": question,
        "chat_context": chat_context,
        "source_id": source_id,
        "source": source,
        "source_type": source_type,
        "file_type": file_type,
        "page_start": page_start,
        "page_end": page_end,
    }

    if pipeline_used == "document_summary":
        return document_summary.run(**common_args)

    if pipeline_used == "long_explanation":
        return long_explanation.run(**common_args)

    if pipeline_used == "repo_explanation":
        return repo_explanation.run(**common_args)

    if pipeline_used == "incident_runbook":
        return incident_runbook.run(**common_args)

    return normal_qa.run(**common_args)


def run_ask(
    question: str,
    conversation_id: str | None = None,
    source_id: str | None = None,
    source: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    normalized_question = normalize_question(question)

    resolved_conversation_id = ensure_conversation(conversation_id, normalized_question)
    add_user_message(resolved_conversation_id, normalized_question)

    chat_context = get_recent_chat_context(resolved_conversation_id, limit=10)

    followup = resolve_followup_question(normalized_question, chat_context)
    effective_question = followup.get("resolved_question") or normalized_question

    routing = decide_intent(effective_question, chat_context=chat_context)
    routing["original_question"] = normalized_question
    routing["resolved_question"] = effective_question
    routing["is_followup"] = followup.get("is_followup", False)
    routing["followup_reason"] = followup.get("reason", "")

    selected_pipeline = routing.get("pipeline_used", "normal_qa")
    RAG_PIPELINE_TOTAL.labels(pipeline=selected_pipeline).inc()
    if str(routing.get("router") or "") == "planner_fallback":
        RAG_PLANNER_FALLBACK_TOTAL.labels(reason="planner_fallback").inc()

    result = _run_selected_pipeline(
        pipeline_used=selected_pipeline,
        question=effective_question,
        chat_context=chat_context,
        source_id=source_id,
        source=source,
        source_type=source_type,
        file_type=file_type,
        page_start=page_start,
        page_end=page_end,
    )

    answer = result.get("answer", "No evidence found in the knowledge base.")
    citations = result.get("citations", [])

    if answer.strip() == "No evidence found in the knowledge base.":
        RAG_NO_EVIDENCE_TOTAL.labels(pipeline=selected_pipeline).inc()
        citations = []

    verification_result: dict[str, Any] = {
        "verification_verdict": "skipped",
        "unsupported_claims": [],
        "verification_reason": "Verifier skipped for this answer.",
        "verified": False,
    }

    # Exact file/code explanation already retrieves one specific file by source_id.
    # Verification adds large latency and little value for this narrow case.
    # Keep verifier enabled for repo-level, long explanation, incident, and normal higher-risk answers.
    skip_verifier = (
        selected_pipeline == "repo_explanation"
        and result.get("retrieval_mode") == "exact_file_source_id_lookup"
    )

    if skip_verifier:
        verification_result = {
            "verification_verdict": "skipped",
            "unsupported_claims": [],
            "verification_reason": "Verifier skipped for exact file/code explanation.",
            "verified": False,
        }

    if (not skip_verifier) and should_verify_answer(
        pipeline_used=selected_pipeline,
        routing=routing,
        answer=answer,
        citations=citations,
    ):
        verification_result = verify_answer(
            question=normalized_question,
            pipeline_used=selected_pipeline,
            context_text=result.get("verification_context_text", ""),
            draft_answer=answer,
        )

        verifier_verdict = verification_result.get("verification_verdict")
        RAG_VERIFIER_TOTAL.labels(
            pipeline=selected_pipeline,
            verdict=str(verifier_verdict or "unknown"),
        ).inc()

        # Important:
        # If verifier says the draft is valid, keep the original detailed draft.
        # Only replace the answer when verifier says revision/insufficient evidence.
        if verifier_verdict in {"needs_revision", "insufficient_evidence"}:
            answer = verification_result.get("corrected_answer", answer)

        if answer.strip() == "No evidence found in the knowledge base.":
            citations = []

    add_assistant_message(
        conversation_id=resolved_conversation_id,
        answer=answer,
        citations=citations,
        intent=routing.get("intent"),
        pipeline_used=selected_pipeline,
    )

    latency_seconds = time.perf_counter() - started
    latency_ms = int(latency_seconds * 1000)
    RAG_QUESTIONS_TOTAL.labels(pipeline=selected_pipeline, status="success").inc()
    RAG_ANSWER_LATENCY_SECONDS.labels(pipeline=selected_pipeline, status="success").observe(latency_seconds)

    audit_id = save_audit_event(
        conversation_id=resolved_conversation_id,
        question=normalized_question,
        answer=answer,
        routing={**routing, **verification_result},
        citations=citations,
        model=CHAT_MODEL,
        latency_ms=latency_ms,
    )

    response: dict[str, Any] = {
        "question": normalized_question,
        "resolved_question": effective_question,
        "is_followup": followup.get("is_followup", False),
        "followup_reason": followup.get("reason", ""),
        "answer": answer,
        "citations": citations,
        "conversation_id": resolved_conversation_id,
        "audit_id": audit_id,
        "pipeline_used": selected_pipeline,
        "pipeline_label": routing.get("pipeline_label"),
        "intent": routing.get("intent"),
        "intent_confidence": routing.get("confidence"),
        "intent_reason": routing.get("reason"),
        "router": routing.get("router"),
        "latency_ms": latency_ms,
        "verification_verdict": verification_result.get("verification_verdict"),
        "unsupported_claims": verification_result.get("unsupported_claims", []),
        "verification_reason": verification_result.get("verification_reason"),
        "verified": verification_result.get("verified", False),
    }

    for key, value in result.items():
        if key not in response and key not in {"answer", "citations", "verification_context_text"}:
            response[key] = value

    return response
