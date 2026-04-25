from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Iterator

from app.audit import save_audit_event
from app.chat_history import (
    add_assistant_message,
    add_user_message,
    ensure_conversation,
    get_recent_chat_context,
)
from app.context_utils import build_citations, build_context_text, compact_chunks
from app.llm_client import CHAT_MODEL, stream_generate_text
from app.pipelines.normal_qa import run as run_normal_qa
from app.prompts import (
    INCIDENT_RUNBOOK_PROMPT,
    LONG_EXPLANATION_PROMPT,
    NORMAL_QA_PROMPT,
    REPO_EXPLANATION_PROMPT,
)
from app.response_formatter import no_evidence_response
from app.retrieve import retrieve_context
from app.router import decide_intent
from app.summary_jobs import start_document_summary_job

MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.35"))

NORMAL_QA_LIMIT = int(os.getenv("NORMAL_QA_LIMIT", "6"))
LONG_EXPLANATION_LIMIT = int(os.getenv("LONG_EXPLANATION_LIMIT", "10"))
REPO_EXPLANATION_LIMIT = int(os.getenv("REPO_EXPLANATION_LIMIT", "10"))
INCIDENT_RUNBOOK_LIMIT = int(os.getenv("INCIDENT_RUNBOOK_LIMIT", "8"))

NORMAL_QA_NUM_PREDICT = int(os.getenv("NORMAL_QA_NUM_PREDICT", "500"))
LONG_EXPLANATION_NUM_PREDICT = int(os.getenv("LONG_EXPLANATION_NUM_PREDICT", "1400"))
REPO_EXPLANATION_NUM_PREDICT = int(os.getenv("REPO_EXPLANATION_NUM_PREDICT", "1400"))
INCIDENT_RUNBOOK_NUM_PREDICT = int(os.getenv("INCIDENT_RUNBOOK_NUM_PREDICT", "1200"))


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def normalize_question(question: str) -> str:
    q = (question or "").strip()

    typo_map = {
        r"\bsmadhi\b": "samadhi",
        r"\bsamadhi\b": "samadhi",
        r"\bsummarise\b": "summarize",
    }

    for pattern, replacement in typo_map.items():
        q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)

    return q


def _is_exact_fact_question(question: str) -> bool:
    q = (question or "").lower()
    return bool(
        re.search(
            r"\b(address|registered office|office address|company address|postcode|post code|company number|registration number|registered in|where is it registered)\b",
            q,
        )
    )


def _retrieve_for_pipeline(
    pipeline_used: str,
    question: str,
    source_id: str | None,
    source: str | None,
    source_type: str | None,
    file_type: str | None,
    page_start: int | None,
    page_end: int | None,
) -> tuple[list[dict[str, Any]], str, int]:
    if pipeline_used == "long_explanation":
        query = question
        limit = LONG_EXPLANATION_LIMIT
        num_predict = LONG_EXPLANATION_NUM_PREDICT

    elif pipeline_used == "repo_explanation":
        query = (
            f"{question}\n"
            "Prioritize README, architecture docs, infrastructure files, container files, "
            "workflow files, deployment files, modules, and runbooks."
        )
        limit = REPO_EXPLANATION_LIMIT
        num_predict = REPO_EXPLANATION_NUM_PREDICT

    elif pipeline_used == "incident_runbook":
        query = (
            f"{question}\n"
            "Prioritize runbooks, troubleshooting docs, deployment docs, rollback docs, "
            "monitoring docs, alert docs, and operational procedures."
        )
        limit = INCIDENT_RUNBOOK_LIMIT
        num_predict = INCIDENT_RUNBOOK_NUM_PREDICT

    else:
        query = question
        limit = NORMAL_QA_LIMIT
        num_predict = NORMAL_QA_NUM_PREDICT

    chunks = retrieve_context(
        query,
        limit=limit,
        source_id=source_id,
        source=source,
        source_type=source_type,
        file_type=file_type,
        page_start=page_start,
        page_end=page_end,
    )

    return chunks, query, num_predict


def _build_prompt(
    pipeline_used: str,
    question: str,
    chat_context: str,
    context_text: str,
) -> str:
    if pipeline_used == "long_explanation":
        return LONG_EXPLANATION_PROMPT.format(
            question=question,
            chat_context=chat_context or "No recent conversation context.",
            context_text=context_text,
        )

    if pipeline_used == "repo_explanation":
        return REPO_EXPLANATION_PROMPT.format(
            question=question,
            chat_context=chat_context or "No recent conversation context.",
            context_text=context_text,
        )

    if pipeline_used == "incident_runbook":
        return INCIDENT_RUNBOOK_PROMPT.format(
            question=question,
            chat_context=chat_context or "No recent conversation context.",
            context_text=context_text,
        )

    return NORMAL_QA_PROMPT.format(
        question=question,
        context_text=context_text,
    )


def stream_ask_events(
    question: str,
    conversation_id: str | None = None,
    source_id: str | None = None,
    source: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> Iterator[str]:
    started = time.perf_counter()

    try:
        normalized_question = normalize_question(question)
        routing = decide_intent(normalized_question)
        pipeline_used = routing.get("pipeline_used", "normal_qa")

        resolved_conversation_id = ensure_conversation(conversation_id, normalized_question)
        add_user_message(resolved_conversation_id, normalized_question)

        yield _sse(
            "meta",
            {
                "question": normalized_question,
                "conversation_id": resolved_conversation_id,
                "pipeline_used": pipeline_used,
                "pipeline_label": routing.get("pipeline_label"),
                "intent": routing.get("intent"),
                "intent_confidence": routing.get("confidence"),
                "intent_reason": routing.get("reason"),
                "router": routing.get("router"),
            },
        )

        if pipeline_used == "document_summary":
            job_id = start_document_summary_job(
                question=normalized_question,
                conversation_id=resolved_conversation_id,
                routing=routing,
                source_id=source_id,
                page_start=page_start,
                page_end=page_end,
            )

            yield _sse(
                "summary_job",
                {
                    "job_id": job_id,
                    "conversation_id": resolved_conversation_id,
                    "message": "Document summary job started.",
                },
            )
            yield _sse("done", {"status": "summary_job_started", "job_id": job_id})
            return

        # Exact structured facts are better handled deterministically than streamed through LLM.
        if pipeline_used == "normal_qa" and _is_exact_fact_question(normalized_question):
            result = run_normal_qa(
                question=normalized_question,
                chat_context="",
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
                citations = []

            yield _sse("citations", {"citations": citations})
            yield _sse("token", {"token": answer})

            add_assistant_message(
                conversation_id=resolved_conversation_id,
                answer=answer,
                citations=citations,
                intent=routing.get("intent"),
                pipeline_used=pipeline_used,
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

            yield _sse(
                "done",
                {
                    "status": "done",
                    "audit_id": audit_id,
                    "latency_ms": latency_ms,
                    "citations": citations,
                },
            )
            return

        chunks, _, num_predict = _retrieve_for_pipeline(
            pipeline_used=pipeline_used,
            question=normalized_question,
            source_id=source_id,
            source=source,
            source_type=source_type,
            file_type=file_type,
            page_start=page_start,
            page_end=page_end,
        )

        if not chunks:
            response = no_evidence_response()
            answer = response["answer"]
            citations = []
            yield _sse("citations", {"citations": citations})
            yield _sse("token", {"token": answer})

            latency_ms = int((time.perf_counter() - started) * 1000)
            yield _sse("done", {"status": "done", "latency_ms": latency_ms, "citations": citations})
            return

        top_score = max(float(c.get("score", 0.0) or 0.0) for c in chunks)
        if top_score < MIN_SCORE_THRESHOLD:
            answer = "No evidence found in the knowledge base."
            citations = []
            yield _sse("citations", {"citations": citations})
            yield _sse("token", {"token": answer})

            latency_ms = int((time.perf_counter() - started) * 1000)
            yield _sse("done", {"status": "done", "latency_ms": latency_ms, "citations": citations})
            return

        compacted = compact_chunks(chunks, max_total_chars=11000)
        citations = build_citations(compacted)
        context_text = build_context_text(compacted)
        chat_context = get_recent_chat_context(resolved_conversation_id, limit=10)

        prompt = _build_prompt(
            pipeline_used=pipeline_used,
            question=normalized_question,
            chat_context=chat_context,
            context_text=context_text,
        )

        yield _sse("citations", {"citations": citations})

        answer_parts: list[str] = []

        for token in stream_generate_text(
            prompt,
            temperature=0.0,
            num_predict=num_predict,
            timeout=600,
        ):
            answer_parts.append(token)
            yield _sse("token", {"token": token})

        answer = "".join(answer_parts).strip() or "No evidence found in the knowledge base."

        if answer == "No evidence found in the knowledge base.":
            citations = []

        add_assistant_message(
            conversation_id=resolved_conversation_id,
            answer=answer,
            citations=citations,
            intent=routing.get("intent"),
            pipeline_used=pipeline_used,
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

        yield _sse(
            "done",
            {
                "status": "done",
                "audit_id": audit_id,
                "latency_ms": latency_ms,
                "citations": citations,
            },
        )

    except Exception as exc:
        yield _sse("error", {"message": str(exc)})
