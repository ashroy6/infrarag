from __future__ import annotations

import re
import time
from typing import Any

from app.audit import save_audit_event
from app.chat_history import (
    add_assistant_message,
    add_user_message,
    ensure_conversation,
    get_recent_chat_context,
)
from app.llm_client import CHAT_MODEL
from app.router import decide_intent

from app.pipelines import document_summary
from app.pipelines import incident_runbook
from app.pipelines import long_explanation
from app.pipelines import normal_qa
from app.pipelines import repo_explanation


def normalize_question(question: str) -> str:
    q = (question or "").strip()

    typo_map = {
        r"\bsmadhi\b": "samadhi",
        r"\bsamadhi\b": "samadhi",
        r"\bsummarise\b": "summarize",
        r"\bpatanjalii\b": "patanjali",
    }

    for pattern, replacement in typo_map.items():
        q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)

    return q


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

    routing = decide_intent(normalized_question)
    selected_pipeline = routing.get("pipeline_used", "normal_qa")

    resolved_conversation_id = ensure_conversation(conversation_id, normalized_question)
    add_user_message(resolved_conversation_id, normalized_question)

    chat_context = get_recent_chat_context(resolved_conversation_id, limit=10)

    result = _run_selected_pipeline(
        pipeline_used=selected_pipeline,
        question=normalized_question,
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

    add_assistant_message(
        conversation_id=resolved_conversation_id,
        answer=answer,
        citations=citations,
        intent=routing.get("intent"),
        pipeline_used=selected_pipeline,
    )

    latency_ms = int((time.perf_counter() - started) * 1000)

    audit_id = save_audit_event(
        conversation_id=resolved_conversation_id,
        question=normalized_question,
        answer=answer,
        routing=routing,
        citations=citations,
        model=CHAT_MODEL,
        latency_ms=latency_ms,
    )

    response: dict[str, Any] = {
        "question": normalized_question,
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
    }

    for key, value in result.items():
        if key not in response and key not in {"answer", "citations"}:
            response[key] = value

    return response
