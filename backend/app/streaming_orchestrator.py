from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import PurePosixPath
from typing import Any, Iterator

from app.answer_verifier import answer_denies_evidence, should_verify_answer, verify_answer
from app.audit import save_audit_event
from app.cancel_registry import cancel_reason, is_cancelled, register_request, unregister_request
from app.chat_history import (
    add_assistant_message,
    add_user_message,
    ensure_conversation,
    get_recent_chat_context,
)
from app.context_utils import build_citations, build_context_text, compact_chunks
from app.followup_resolver import resolve_followup_question
from app.graph_retrieval import expand_with_graph_context
from app.llm_client import CHAT_MODEL, LLMCancelled, generate_text, stream_generate_text
from app.metadata_db import MetadataDB
from app.prompts import (
    CODE_FILE_EXPLANATION_PROMPT,
    DENIAL_RECOVERY_PROMPT,
    INCIDENT_RUNBOOK_PROMPT,
    LONG_EXPLANATION_PROMPT,
    LONG_EXPLANATION_RETRY_PROMPT,
    NORMAL_QA_PROMPT,
    REPO_EXPLANATION_PROMPT,
)
from app.qdrant_client import get_chunks_by_source_id
from app.rag_metrics import (
    RAG_ANSWER_LATENCY_SECONDS,
    RAG_GRAPH_CHUNKS_ADDED_TOTAL,
    RAG_GRAPH_CONTEXT_ENABLED_TOTAL,
    RAG_NO_EVIDENCE_TOTAL,
    RAG_OLLAMA_TIMEOUT_TOTAL,
    RAG_PIPELINE_TOTAL,
    RAG_PLANNER_FALLBACK_TOTAL,
    RAG_QUESTIONS_TOTAL,
    RAG_VERIFIER_TOTAL,
)
from app.response_formatter import no_evidence_response
from app.retrieve import retrieve_context
from app.router import decide_intent
from app.source_resolver import resolve_source_for_question
from app.summary_jobs import start_document_summary_job

MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.35"))

NORMAL_QA_LIMIT = int(os.getenv("NORMAL_QA_LIMIT", "6"))
LONG_EXPLANATION_LIMIT = int(os.getenv("LONG_EXPLANATION_LIMIT", "10"))
REPO_EXPLANATION_LIMIT = int(os.getenv("REPO_EXPLANATION_LIMIT", "10"))
INCIDENT_RUNBOOK_LIMIT = int(os.getenv("INCIDENT_RUNBOOK_LIMIT", "8"))

NORMAL_QA_NUM_PREDICT = int(os.getenv("NORMAL_QA_NUM_PREDICT", "500"))
LONG_EXPLANATION_NUM_PREDICT = int(os.getenv("LONG_EXPLANATION_NUM_PREDICT", "1400"))
REPO_EXPLANATION_NUM_PREDICT = int(os.getenv("REPO_EXPLANATION_NUM_PREDICT", "1400"))
CODE_FILE_NUM_PREDICT = int(os.getenv("CODE_FILE_NUM_PREDICT", "900"))
INCIDENT_RUNBOOK_NUM_PREDICT = int(os.getenv("INCIDENT_RUNBOOK_NUM_PREDICT", "1200"))

EXACT_FILE_LIMIT = int(os.getenv("EXACT_FILE_LIMIT", "50"))
DEFAULT_DB_PATH = "/app/data/infrarag.db"

FILE_PATTERN = re.compile(
    r"(?P<file>[A-Za-z0-9_\-./]+(?:\.(?:py|js|ts|tsx|jsx|json|yaml|yml|md|txt|tf|sh|sql|html|css)|Dockerfile|dockerfile))",
    re.IGNORECASE,
)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", (question or "").strip())


def _db_path() -> str:
    return (
        os.getenv("INFRARAG_DB_PATH")
        or os.getenv("METADATA_DB_PATH")
        or os.getenv("DATABASE_PATH")
        or DEFAULT_DB_PATH
    )


def _normalize_path(value: str) -> str:
    return re.sub(r"/+", "/", (value or "").replace("\\", "/").strip().lower())


def _basename(value: str) -> str:
    normalized = _normalize_path(value)
    if not normalized:
        return ""
    return PurePosixPath(normalized).name


def _extract_requested_file(question: str) -> str | None:
    match = FILE_PATTERN.search(question or "")
    if not match:
        return None

    requested_file = match.group("file").strip().strip("`'\".,:;()[]{}")
    return requested_file or None


def _resolve_source_from_files_table(requested_file: str) -> dict[str, str] | None:
    wanted = _normalize_path(requested_file)
    wanted_base = _basename(wanted)

    if not wanted and not wanted_base:
        return None

    db_file = _db_path()

    try:
        con = sqlite3.connect(db_file)
        con.row_factory = sqlite3.Row
    except Exception:
        return None

    try:
        rows = con.execute(
            """
            SELECT source_id, source_path, status, file_type, source_type, chunk_count, last_ingested_at
            FROM files
            WHERE source_id IS NOT NULL
              AND source_path IS NOT NULL
              AND (
                    LOWER(REPLACE(source_path, '\\', '/')) LIKE ?
                 OR LOWER(REPLACE(source_path, '\\', '/')) LIKE ?
              )
            ORDER BY
              CASE WHEN status = 'active' THEN 0 ELSE 1 END,
              last_ingested_at DESC
            LIMIT 100
            """,
            (f"%{wanted}%", f"%/{wanted_base}"),
        ).fetchall()
    except Exception:
        return None
    finally:
        con.close()

    if not rows:
        return None

    def rank(row: sqlite3.Row) -> tuple[int, int, int]:
        source_path = _normalize_path(str(row["source_path"]))
        source_base = _basename(source_path)
        status = str(row["status"] or "")

        active_rank = 0 if status == "active" else 1

        if source_path == wanted:
            match_rank = 0
        elif source_path.endswith("/" + wanted):
            match_rank = 1
        elif source_base == wanted_base:
            match_rank = 2
        elif wanted in source_path:
            match_rank = 3
        else:
            match_rank = 9

        return active_rank, match_rank, len(source_path)

    best = sorted(rows, key=rank)[0]

    return {
        "source_id": str(best["source_id"]),
        "source_path": str(best["source_path"]),
    }


def _pipeline_limits(pipeline_used: str) -> tuple[int, int]:
    if pipeline_used == "long_explanation":
        return LONG_EXPLANATION_LIMIT, LONG_EXPLANATION_NUM_PREDICT

    if pipeline_used == "repo_explanation":
        return REPO_EXPLANATION_LIMIT, REPO_EXPLANATION_NUM_PREDICT

    if pipeline_used == "incident_runbook":
        return INCIDENT_RUNBOOK_LIMIT, INCIDENT_RUNBOOK_NUM_PREDICT

    return NORMAL_QA_LIMIT, NORMAL_QA_NUM_PREDICT


def _build_prompt(
    pipeline_used: str,
    question: str,
    chat_context: str,
    context_text: str,
    exact_file_mode: bool = False,
) -> str:
    if pipeline_used == "long_explanation":
        return LONG_EXPLANATION_PROMPT.format(
            question=question,
            chat_context=chat_context or "No recent conversation context.",
            context_text=context_text,
        )

    if pipeline_used == "repo_explanation" and exact_file_mode:
        return CODE_FILE_EXPLANATION_PROMPT.format(
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


def _long_answer_too_short(answer: str, pipeline_used: str, citations: list[dict[str, Any]]) -> bool:
    if pipeline_used != "long_explanation":
        return False

    if not citations:
        return False

    clean = re.sub(r"\s+", " ", (answer or "").strip())
    if not clean:
        return True

    # Generic completeness check:
    # long_explanation should not return only one tiny section when evidence exists.
    section_count = len(re.findall(r"(?m)^\s*\d+\.\s+", answer or ""))
    word_count = len(re.findall(r"\b\w+\b", clean))

    if word_count < 120:
        return True

    if section_count < 3:
        return True

    return False


def _chunk_index(chunk: dict[str, Any]) -> int:
    candidates: list[Any] = [
        chunk.get("chunk_index"),
        chunk.get("index"),
        chunk.get("chunk"),
    ]

    metadata = chunk.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend(
            [
                metadata.get("chunk_index"),
                metadata.get("index"),
                metadata.get("chunk"),
            ]
        )

    payload = chunk.get("payload")
    if isinstance(payload, dict):
        candidates.extend(
            [
                payload.get("chunk_index"),
                payload.get("index"),
                payload.get("chunk"),
            ]
        )

        payload_metadata = payload.get("metadata")
        if isinstance(payload_metadata, dict):
            candidates.extend(
                [
                    payload_metadata.get("chunk_index"),
                    payload_metadata.get("index"),
                    payload_metadata.get("chunk"),
                ]
            )

    for value in candidates:
        try:
            return int(value)
        except (TypeError, ValueError):
            continue

    return 0


def _is_file_explanation_question(question: str) -> bool:
    if not _extract_requested_file(question):
        return False

    q = " ".join((question or "").lower().split())

    intent_words = (
        "explain",
        "line by line",
        "walk through",
        "what does",
        "how does",
        "review",
        "improve",
        "summarize",
        "summarise",
        "analyse",
        "analyze",
        "describe",
    )

    return any(word in q for word in intent_words)


def _exact_file_plan(requested_file: str) -> dict[str, Any]:
    return {
        "rewritten_queries": [requested_file],
        "candidate_top_k": 80,
        "final_top_k": 50,
        "source_strategy": "cluster_by_best_source",
        "router": "exact_file_source_id_lookup",
        "question_type": "file_explanation",
    }


def _retrieve_exact_file_chunks(
    *,
    requested_file: str,
    source_id: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    resolved = None

    if source_id:
        resolved = {
            "source_id": source_id,
            "source_path": "",
        }
    else:
        resolved = _resolve_source_from_files_table(requested_file)

    if not resolved or not resolved.get("source_id"):
        return [], None

    # Exact file mode must not use retrieve_context().
    # retrieve_context() performs embedding, vector search, clustering, and reranking.
    # We already resolved the exact source_id from SQLite, so load all file chunks
    # directly from Qdrant by source_id. This is faster and avoids unrelated context.
    chunks = get_chunks_by_source_id(resolved["source_id"])

    # Optional defensive filters. source_id should already be exact, but keep these
    # so UI filters do not accidentally return unrelated source types/file types.
    if source_type:
        chunks = [chunk for chunk in chunks if chunk.get("source_type") == source_type]

    if file_type:
        chunks = [chunk for chunk in chunks if chunk.get("file_type") == file_type]

    if page_start is not None:
        chunks = [
            chunk for chunk in chunks
            if chunk.get("page_number") is None or int(chunk.get("page_number") or 0) >= page_start
        ]

    if page_end is not None:
        chunks = [
            chunk for chunk in chunks
            if chunk.get("page_number") is None or int(chunk.get("page_number") or 0) <= page_end
        ]

    chunks.sort(key=_chunk_index)
    return chunks, resolved


def _exact_file_question_for_prompt(question: str, requested_file: str) -> str:
    return (
        f"{question}\n\n"
        f"Important: Explain only the requested file `{requested_file}` using the retrieved file chunks. "
        f"Do not explain README, Docker, GitHub Actions, CI/CD, dependencies, or repo-level workflow unless those details are inside this exact file. "
        f"If the user asked line by line, explain the code in source order. "
        f"If evidence is missing, say exactly what is missing."
    )



def _default_regular_chat_source_ids() -> list[str]:
    """
    Main Chat must not search Agent Studio sources by default.

    Agent Studio has its own source assignment and allowed_source_ids path.
    Regular InfraRAG chat should only search source_group='regular_chat'
    unless the user/UI explicitly selected a source_id.
    """
    try:
        records = MetadataDB().list_active_files(source_group="regular_chat")
    except Exception:
        return []

    source_ids: list[str] = []
    for record in records:
        source_id = str(record.get("source_id") or "").strip()
        if source_id and source_id not in source_ids:
            source_ids.append(source_id)

    return source_ids


def _clean_direct_snippet(text: str, max_chars: int = 900) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + " ..."


def _extract_entity_from_lookup_question(question: str) -> str:
    q = re.sub(r"\s+", " ", str(question or "").strip())
    patterns = [
        r"^\s*who\s+is\s+(.+?)\s*\??$",
        r"^\s*who\s+was\s+(.+?)\s*\??$",
        r"^\s*what\s+is\s+(.+?)\s*\??$",
        r"^\s*where\s+is\s+(.+?)\s*\??$",
        r"^\s*tell\s+me\s+about\s+(.+?)\s*\??$",
        r"^\s*define\s+(.+?)\s*\??$",
    ]

    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .,:;?!'\"")

    return q.strip(" .,:;?!'\"")


def _direct_answer_from_chunks(
    *,
    question: str,
    chunks: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    query_shape: str,
    retrieval_mode: str,
) -> str:
    if not chunks:
        return "No evidence found in the knowledge base."

    top = chunks[0]
    source = top.get("source") or "retrieved source"
    chunk_index = top.get("chunk_index", "")
    snippet = _clean_direct_snippet(top.get("text", ""), max_chars=900)

    if query_shape == "exact_phrase":
        return (
            f"Found a matching passage in {source}, chunk {chunk_index}.\n\n"
            f"{snippet}\n\n"
            f"Open citation [1] for the full evidence."
        )

    if query_shape == "entity_lookup":
        entity = _extract_entity_from_lookup_question(question)
        return (
            f"Based on the top retrieved evidence, {entity} is mentioned in {source}, chunk {chunk_index}.\n\n"
            f"{snippet}\n\n"
            f"Open the citations for full context."
        )

    return (
        f"Top matching evidence found in {source}, chunk {chunk_index}.\n\n"
        f"{snippet}\n\n"
        f"Open the citations for full context."
    )


def _can_use_direct_answer(
    *,
    retrieval_speed: str,
    pipeline_used: str,
    query_shape: str,
    exact_file_mode: bool,
) -> bool:
    if retrieval_speed != "direct":
        return False

    if exact_file_mode:
        return False

    if pipeline_used not in {"normal_qa"}:
        return False

    return query_shape in {"entity_lookup", "exact_phrase", "normal_qa", "section_summary"}


def _mode_label_from_speed(retrieval_speed: str) -> str:
    clean = str(retrieval_speed or "normal").lower()
    if clean == "direct":
        return "Direct"
    if clean == "fast":
        return "Quick AI"
    if clean == "background_summary":
        return "Background Summary"
    return "Deep AI"


def _build_answer_metadata(
    *,
    routing: dict[str, Any],
    pipeline_used: str,
    retrieval_speed: str,
    actual_retrieval_mode: str,
    actual_retriever_used: str,
    actual_query_shape: str,
    actual_reranker_used: Any,
    actual_neighbour_window: Any,
    actual_retrieval_reason: str,
    actual_adaptive_retrieval: bool,
    retrieved_chunks: int,
    use_graph_context: bool,
    graph_chunks_added: int,
    verification_result: dict[str, Any],
    latency_ms: int,
    requested_file: str | None = None,
    resolved_file: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "pipeline_used": pipeline_used,
        "pipeline_label": routing.get("pipeline_label") or pipeline_used,
        "intent": routing.get("intent"),
        "intent_confidence": routing.get("confidence"),
        "intent_reason": routing.get("reason"),
        "router": routing.get("router"),
        "question_type": routing.get("question_type"),
        "source_strategy": routing.get("source_strategy"),
        "retrieval_speed": retrieval_speed,
        "mode_label": _mode_label_from_speed(retrieval_speed),
        "retrieval_mode": actual_retrieval_mode,
        "retriever_used": actual_retriever_used or actual_retrieval_mode,
        "query_shape": actual_query_shape,
        "reranker_used": actual_reranker_used,
        "neighbour_window": actual_neighbour_window,
        "retrieval_planner_reason": actual_retrieval_reason,
        "adaptive_retrieval": actual_adaptive_retrieval,
        "retrieved_chunks": retrieved_chunks,
        "verification_verdict": verification_result.get("verification_verdict"),
        "verification_reason": verification_result.get("verification_reason"),
        "unsupported_claims": verification_result.get("unsupported_claims", []),
        "verified": verification_result.get("verified", False),
        "graph_context_enabled": bool(use_graph_context),
        "graph_chunks_added": graph_chunks_added,
        "latency_ms": latency_ms,
        "progress": 100,
        "progress_label": "Done.",
        "requested_file": requested_file,
        "resolved_source_id": resolved_file.get("source_id") if resolved_file else None,
        "resolved_source_path": resolved_file.get("source_path") if resolved_file else None,
    }





def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _new_timing_breakdown() -> dict[str, int | None]:
    return {
        "planning_ms": None,
        "retrieval_ms": None,
        "graph_ms": None,
        "context_build_ms": None,
        "time_to_first_token_ms": None,
        "ollama_ms": None,
        "audit_save_ms": None,
        "total_latency_ms": None,
    }


def _finalize_timing(timings: dict[str, int | None], started_at: float) -> dict[str, int | None]:
    timings["total_latency_ms"] = _elapsed_ms(started_at)
    return timings


def _timing_payload(timings: dict[str, int | None]) -> dict[str, int | None]:
    return {
        "planning_ms": timings.get("planning_ms"),
        "retrieval_ms": timings.get("retrieval_ms"),
        "graph_ms": timings.get("graph_ms"),
        "context_build_ms": timings.get("context_build_ms"),
        "time_to_first_token_ms": timings.get("time_to_first_token_ms"),
        "ollama_ms": timings.get("ollama_ms"),
        "audit_save_ms": timings.get("audit_save_ms"),
        "total_latency_ms": timings.get("total_latency_ms"),
    }


def stream_ask_events(
    question: str,
    conversation_id: str | None = None,
    source_id: str | None = None,
    source: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    request_id: str | None = None,
    use_graph_context: bool = False,
    retrieval_speed: str = "normal",
) -> Iterator[str]:
    started = time.perf_counter()
    timings = _new_timing_breakdown()
    planning_started = time.perf_counter()
    request_id = register_request(request_id)

    def cancelled() -> bool:
        return is_cancelled(request_id)

    def cancel_payload(stage: str) -> dict[str, Any]:
        return {
            "status": "cancelled",
            "request_id": request_id,
            "stage": stage,
            "reason": cancel_reason(request_id) or "Cancelled by user.",
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    try:
        normalized_question = normalize_question(question)
        speed_value = str(retrieval_speed or "").lower()
        clean_retrieval_speed = "direct" if speed_value == "direct" else ("fast" if speed_value == "fast" else "normal")

        yield _sse(
            "request",
            {
                "request_id": request_id,
                "progress": 5,
                "progress_label": "Request registered.",
            },
        )

        resolved_conversation_id = ensure_conversation(conversation_id, normalized_question)
        add_user_message(resolved_conversation_id, normalized_question)

        if cancelled():
            yield _sse("cancelled", cancel_payload("before_planning"))
            return

        chat_context = get_recent_chat_context(resolved_conversation_id, limit=10)

        followup = resolve_followup_question(normalized_question, chat_context)
        effective_question = followup.get("resolved_question") or normalized_question

        source_resolution = resolve_source_for_question(
            effective_question,
            source_id=source_id,
            source_type=source_type,
            file_type=file_type,
        )

        if not source_id and source_resolution.get("source_id"):
            source_id = str(source_resolution.get("source_id"))

        routing = decide_intent(effective_question, chat_context=chat_context)
        routing["original_question"] = normalized_question
        routing["resolved_question"] = effective_question
        routing["is_followup"] = followup.get("is_followup", False)
        routing["followup_reason"] = followup.get("reason", "")

        pipeline_used = routing.get("pipeline_used", "normal_qa")
        RAG_PIPELINE_TOTAL.labels(pipeline=pipeline_used).inc()

        router_name = str(routing.get("router") or "")
        route_reason = str(routing.get("reason") or "")
        if router_name == "planner_fallback":
            RAG_PLANNER_FALLBACK_TOTAL.labels(reason="planner_fallback").inc()
        if "timeout" in route_reason.lower() or "timed out" in route_reason.lower():
            RAG_OLLAMA_TIMEOUT_TOTAL.labels(stage="planner").inc()
        requested_file = _extract_requested_file(effective_question)
        exact_file_mode = (
            pipeline_used == "repo_explanation"
            and requested_file is not None
            and _is_file_explanation_question(effective_question)
        )

        planned_retrieval_mode = (
            "exact_file_source_id_lookup"
            if exact_file_mode
            else ("vector_graph_search" if use_graph_context else "adaptive_hybrid_search_direct" if clean_retrieval_speed == "direct" else ("adaptive_hybrid_search_fast" if clean_retrieval_speed == "fast" else "adaptive_hybrid_search"))
        )

        if cancelled():
            yield _sse("cancelled", cancel_payload("after_planning"))
            return

        timings["planning_ms"] = _elapsed_ms(planning_started)

        yield _sse(
            "meta",
            {
                "request_id": request_id,
                "question": normalized_question,
                "resolved_question": effective_question,
                "is_followup": followup.get("is_followup", False),
                "followup_reason": followup.get("reason", ""),
                "conversation_id": resolved_conversation_id,
                "pipeline_used": pipeline_used,
                "pipeline_label": routing.get("pipeline_label"),
                "intent": routing.get("intent"),
                "intent_confidence": routing.get("confidence"),
                "intent_reason": routing.get("reason"),
                "router": routing.get("router"),
                "question_type": routing.get("question_type"),
                "source_strategy": routing.get("source_strategy"),
                "retrieval_mode": planned_retrieval_mode,
                "retrieval_speed": clean_retrieval_speed,
                "retriever_used": "",
                "query_shape": "",
                "reranker_used": None,
                "retrieved_chunks": 0,
                "graph_context_enabled": bool(use_graph_context and not exact_file_mode),
                "graph_chunks_added": 0,
                "requested_file": requested_file,
                "source_resolution_mode": source_resolution.get("mode"),
                "source_resolution_reason": source_resolution.get("reason"),
                "auto_source_id": source_resolution.get("source_id"),
                "auto_source_path": source_resolution.get("source_path"),
                "verification_verdict": "pending" if pipeline_used in {"long_explanation", "repo_explanation", "incident_runbook"} else "skipped",
                "progress": 15,
                "progress_label": "Planning complete. Searching evidence.",
                **_timing_payload(timings),
            },
        )

        if pipeline_used == "document_summary":
            if cancelled():
                yield _sse("cancelled", cancel_payload("before_summary_job"))
                return

            job_id = start_document_summary_job(
                question=effective_question,
                conversation_id=resolved_conversation_id,
                routing=routing,
                source_id=source_id,
                page_start=page_start,
                page_end=page_end,
            )

            yield _sse(
                "summary_job",
                {
                    "request_id": request_id,
                    "job_id": job_id,
                    "conversation_id": resolved_conversation_id,
                    "message": "Document summary job started.",
                    "progress": 10,
                    "progress_label": "Document summary job started.",
                },
            )
            yield _sse("done", {"status": "summary_job_started", "job_id": job_id, "request_id": request_id})
            return

        retrieval_limit, num_predict = _pipeline_limits(pipeline_used)

        if clean_retrieval_speed == "fast":
            retrieval_limit = min(retrieval_limit, 3)
            if pipeline_used == "normal_qa":
                num_predict = min(num_predict, 180)
            elif pipeline_used == "incident_runbook":
                num_predict = min(num_predict, 450)
            elif pipeline_used == "repo_explanation":
                num_predict = min(num_predict, 500)
            elif pipeline_used == "long_explanation":
                num_predict = min(num_predict, 700)

        if clean_retrieval_speed == "direct":
            retrieval_limit = min(retrieval_limit, 3)
            num_predict = 0

        # Exact file/code explanation should be smaller and faster than repo-wide explanation.
        # It already uses exact chunks by source_id, so avoid oversized generation.
        if exact_file_mode:
            num_predict = CODE_FILE_NUM_PREDICT

        if cancelled():
            yield _sse("cancelled", cancel_payload("before_retrieval"))
            return

        resolved_file: dict[str, str] | None = None

        # Default Chat source isolation:
        # If no specific source is selected, only search regular_chat sources.
        # This prevents Agent Studio demo/customer/hr/github/legal data leaking into main chat answers.
        default_allowed_source_ids = None
        if not source_id:
            default_allowed_source_ids = _default_regular_chat_source_ids()

        retrieval_started = time.perf_counter()

        if exact_file_mode and requested_file:
            chunks, resolved_file = _retrieve_exact_file_chunks(
                requested_file=requested_file,
                source_id=source_id,
                source_type=source_type,
                file_type=file_type,
                page_start=page_start,
                page_end=page_end,
            )
        else:
            retrieval_plan = routing.get("planner") or routing
            chunks = retrieve_context(
                effective_question,
                limit=retrieval_limit,
                source_id=source_id,
                source=source,
                source_type=source_type,
                file_type=file_type,
                page_start=page_start,
                page_end=page_end,
                retrieval_plan=retrieval_plan,
                use_graph_context=False,
                graph_max_chunks=0,
                allowed_source_ids=default_allowed_source_ids,
                retrieval_speed=clean_retrieval_speed,
            )

        timings["retrieval_ms"] = _elapsed_ms(retrieval_started)

        graph_context_meta: dict[str, Any] = {}
        graph_chunks_added = 0

        if use_graph_context and not exact_file_mode and chunks:
            graph_started = time.perf_counter()

            chunks, graph_context_meta = expand_with_graph_context(
                chunks,
                max_graph_chunks=3,
            )

            timings["graph_ms"] = _elapsed_ms(graph_started)

            for hit in chunks:
                hit.setdefault("graph_context_enabled", True)

            if chunks:
                chunks[0]["graph_context_meta"] = graph_context_meta

            graph_chunks_added = len([c for c in chunks if c.get("graph_context")])
        else:
            timings["graph_ms"] = 0

        if cancelled():
            yield _sse("cancelled", cancel_payload("after_retrieval"))
            return

        actual_retrieval_mode = (
            "exact_file_source_id_lookup"
            if exact_file_mode
            else (
                str(chunks[0].get("retrieval_mode") or planned_retrieval_mode)
                if chunks
                else planned_retrieval_mode
            )
        )
        actual_query_shape = (
            str(chunks[0].get("query_shape") or "")
            if chunks
            else ""
        )
        actual_retrieval_reason = (
            str(chunks[0].get("retrieval_planner_reason") or "")
            if chunks
            else ""
        )
        actual_adaptive_retrieval = bool(
            chunks and chunks[0].get("adaptive_retrieval", False)
        )
        actual_retriever_used = (
            str(chunks[0].get("retriever_used") or "")
            if chunks
            else ""
        )
        actual_reranker_used = (
            chunks[0].get("reranker_used")
            if chunks
            else None
        )
        actual_neighbour_window = (
            chunks[0].get("neighbour_window")
            if chunks
            else None
        )

        if not chunks:
            RAG_NO_EVIDENCE_TOTAL.labels(pipeline=pipeline_used).inc()
            answer = no_evidence_response()["answer"]
            citations: list[dict[str, Any]] = []

            yield _sse(
                "citations",
                {
                    "citations": citations,
                    "progress": 35,
                    "progress_label": "No evidence found.",
                    "retrieval_mode": actual_retrieval_mode,
                    "retrieval_speed": clean_retrieval_speed,
                    "retriever_used": actual_retriever_used,
                    "query_shape": actual_query_shape,
                    "reranker_used": actual_reranker_used,
                    "neighbour_window": actual_neighbour_window,
                    "retrieval_planner_reason": actual_retrieval_reason,
                    "adaptive_retrieval": actual_adaptive_retrieval,
                    "requested_file": requested_file,
                    "resolved_source_id": resolved_file.get("source_id") if resolved_file else None,
                    "resolved_source_path": resolved_file.get("source_path") if resolved_file else None,
                    "retrieved_chunks": 0,
                    **_timing_payload(timings),
                },
            )
            yield _sse("token", {"token": answer, "progress": 100, "progress_label": "Done."})

            latency_ms = int((time.perf_counter() - started) * 1000)
            yield _sse(
                "done",
                {
                    "status": "done",
                    "request_id": request_id,
                    "latency_ms": latency_ms,
                    "citations": citations,
                    "progress": 100,
                    "progress_label": "Done.",
                    "retrieval_mode": actual_retrieval_mode,
                    "retrieval_speed": clean_retrieval_speed,
                    "retriever_used": actual_retriever_used,
                    "query_shape": actual_query_shape,
                    "reranker_used": actual_reranker_used,
                    "neighbour_window": actual_neighbour_window,
                    "retrieval_planner_reason": actual_retrieval_reason,
                    "adaptive_retrieval": actual_adaptive_retrieval,
                    "requested_file": requested_file,
                    "resolved_source_id": resolved_file.get("source_id") if resolved_file else None,
                    "resolved_source_path": resolved_file.get("source_path") if resolved_file else None,
                    "retrieved_chunks": 0,
                    **_timing_payload(timings),
                },
            )
            return

        if use_graph_context and not exact_file_mode:
            RAG_GRAPH_CONTEXT_ENABLED_TOTAL.labels(pipeline=pipeline_used).inc()
            if graph_chunks_added > 0:
                RAG_GRAPH_CHUNKS_ADDED_TOTAL.labels(pipeline=pipeline_used).inc(graph_chunks_added)

        if not exact_file_mode:
            top_score = max(float(c.get("score", 0.0) or 0.0) for c in chunks)
            if top_score < MIN_SCORE_THRESHOLD:
                RAG_NO_EVIDENCE_TOTAL.labels(pipeline=pipeline_used).inc()
                answer = "No evidence found in the knowledge base."
                citations = []

                yield _sse(
                    "citations",
                    {
                        "citations": citations,
                        "progress": 35,
                        "progress_label": "Evidence below confidence threshold.",
                    },
                )
                yield _sse("token", {"token": answer, "progress": 100, "progress_label": "Done."})

                latency_ms = int((time.perf_counter() - started) * 1000)
                yield _sse(
                    "done",
                    {
                        "status": "done",
                        "request_id": request_id,
                        "latency_ms": latency_ms,
                        "citations": citations,
                        "progress": 100,
                        "progress_label": "Done.",
                    },
                )
                return

        context_build_started = time.perf_counter()

        compacted = compact_chunks(
            chunks,
            max_total_chars=18000 if exact_file_mode else 11000,
        )
        citations = build_citations(compacted)
        context_text = build_context_text(compacted)

        timings["context_build_ms"] = _elapsed_ms(context_build_started)

        if _can_use_direct_answer(
            retrieval_speed=clean_retrieval_speed,
            pipeline_used=pipeline_used,
            query_shape=actual_query_shape,
            exact_file_mode=exact_file_mode,
        ):
            answer = _direct_answer_from_chunks(
                question=effective_question,
                chunks=compacted,
                citations=citations,
                query_shape=actual_query_shape,
                retrieval_mode=actual_retrieval_mode,
            )

            yield _sse(
                "citations",
                {
                    "citations": citations,
                    "progress": 35,
                    "progress_label": "Evidence found. Returning direct answer.",
                    "retrieval_mode": actual_retrieval_mode,
                    "retrieval_speed": clean_retrieval_speed,
                    "retriever_used": actual_retriever_used,
                    "query_shape": actual_query_shape,
                    "reranker_used": actual_reranker_used,
                    "neighbour_window": actual_neighbour_window,
                    "retrieval_planner_reason": actual_retrieval_reason,
                    "adaptive_retrieval": actual_adaptive_retrieval,
                    "requested_file": requested_file,
                    "resolved_source_id": resolved_file.get("source_id") if resolved_file else None,
                    "resolved_source_path": resolved_file.get("source_path") if resolved_file else None,
                    "retrieved_chunks": len(compacted),
                    "graph_context_enabled": bool(use_graph_context and not exact_file_mode),
                    "graph_chunks_added": graph_chunks_added if not exact_file_mode else 0,
                    "graph_reason": graph_context_meta.get("graph_reason", ""),
                    **_timing_payload(timings),
                },
            )

            yield _sse(
                "token",
                {
                    "token": answer,
                    "progress": 95,
                    "progress_label": "Direct answer ready.",
                },
            )

            verification_result = {
                "verification_verdict": "skipped_direct",
                "unsupported_claims": [],
                "verification_reason": "Direct mode skipped Ollama and returned retrieved evidence snippet.",
                "verified": False,
            }

            timings = _finalize_timing(timings, started)
            latency_seconds = time.perf_counter() - started
            latency_ms = int(latency_seconds * 1000)

            answer_metadata = _build_answer_metadata(
                routing=routing,
                pipeline_used=pipeline_used,
                retrieval_speed=clean_retrieval_speed,
                actual_retrieval_mode=actual_retrieval_mode,
                actual_retriever_used=actual_retriever_used,
                actual_query_shape=actual_query_shape,
                actual_reranker_used=actual_reranker_used,
                actual_neighbour_window=actual_neighbour_window,
                actual_retrieval_reason=actual_retrieval_reason,
                actual_adaptive_retrieval=actual_adaptive_retrieval,
                retrieved_chunks=len(compacted),
                use_graph_context=bool(use_graph_context and not exact_file_mode),
                graph_chunks_added=graph_chunks_added if not exact_file_mode else 0,
                verification_result=verification_result,
                latency_ms=latency_ms,
                requested_file=requested_file,
                resolved_file=resolved_file,
            )
            answer_metadata.update(_timing_payload(timings))

            audit_save_started = time.perf_counter()

            add_assistant_message(
                conversation_id=resolved_conversation_id,
                answer=answer,
                citations=citations,
                intent=routing.get("intent"),
                pipeline_used=pipeline_used,
                metadata=answer_metadata,
            )
            RAG_QUESTIONS_TOTAL.labels(pipeline=pipeline_used, status="success").inc()
            RAG_ANSWER_LATENCY_SECONDS.labels(pipeline=pipeline_used, status="success").observe(latency_seconds)
            audit_id = save_audit_event(
                conversation_id=resolved_conversation_id,
                question=normalized_question,
                answer=answer,
                routing={**routing, **verification_result},
                citations=citations,
                model="direct_backend_snippet",
                latency_ms=latency_ms,
            )

            timings["audit_save_ms"] = _elapsed_ms(audit_save_started)
            timings = _finalize_timing(timings, started)

            yield _sse(
                "done",
                {
                    "status": "done",
                    "request_id": request_id,
                    "audit_id": audit_id,
                    "latency_ms": latency_ms,
                    "citations": citations,
                    "verification_verdict": verification_result.get("verification_verdict"),
                    "unsupported_claims": [],
                    "verification_reason": verification_result.get("verification_reason"),
                    "verified": False,
                    "progress": 100,
                    "progress_label": "Done.",
                    "retrieval_mode": actual_retrieval_mode,
                    "retrieval_speed": clean_retrieval_speed,
                    "retriever_used": actual_retriever_used,
                    "query_shape": actual_query_shape,
                    "reranker_used": actual_reranker_used,
                    "neighbour_window": actual_neighbour_window,
                    "retrieval_planner_reason": actual_retrieval_reason,
                    "adaptive_retrieval": actual_adaptive_retrieval,
                    "requested_file": requested_file,
                    "resolved_source_id": resolved_file.get("source_id") if resolved_file else None,
                    "resolved_source_path": resolved_file.get("source_path") if resolved_file else None,
                    "retrieved_chunks": len(compacted),
                    **_timing_payload(timings),
                },
            )
            return

        prompt_question = (
            _exact_file_question_for_prompt(effective_question, requested_file)
            if exact_file_mode and requested_file
            else effective_question
        )

        prompt = _build_prompt(
            pipeline_used=pipeline_used,
            question=prompt_question,
            chat_context=chat_context,
            context_text=context_text,
            exact_file_mode=exact_file_mode,
        )

        yield _sse(
            "citations",
            {
                "citations": citations,
                "progress": 35,
                "progress_label": "Evidence found. Generating answer.",
                "retrieval_mode": actual_retrieval_mode,
                    "retrieval_speed": clean_retrieval_speed,
                    "retriever_used": actual_retriever_used,
                    "query_shape": actual_query_shape,
                    "reranker_used": actual_reranker_used,
                    "neighbour_window": actual_neighbour_window,
                    "retrieval_planner_reason": actual_retrieval_reason,
                    "adaptive_retrieval": actual_adaptive_retrieval,
                "requested_file": requested_file,
                "resolved_source_id": resolved_file.get("source_id") if resolved_file else None,
                "resolved_source_path": resolved_file.get("source_path") if resolved_file else None,
                "retrieved_chunks": len(compacted),
                "graph_context_enabled": bool(use_graph_context and not exact_file_mode),
                "graph_chunks_added": graph_chunks_added if not exact_file_mode else 0,
                "graph_reason": graph_context_meta.get("graph_reason", ""),
            },
        )

        answer_parts: list[str] = []
        ollama_started = time.perf_counter()
        first_token_recorded = False

        try:
            for token in stream_generate_text(
                prompt,
                temperature=0.0,
                num_predict=num_predict,
                timeout=600,
                cancel_check=cancelled,
            ):
                if cancelled():
                    yield _sse("cancelled", cancel_payload("answer_generation"))
                    return

                if not first_token_recorded:
                    timings["time_to_first_token_ms"] = _elapsed_ms(started)
                    first_token_recorded = True

                answer_parts.append(token)
                progress = min(85, 45 + (len("".join(answer_parts)) // 120))
                yield _sse("token", {"token": token, "progress": progress, "progress_label": "Generating answer."})

        except LLMCancelled:
            yield _sse("cancelled", cancel_payload("answer_generation"))
            return

        timings["ollama_ms"] = _elapsed_ms(ollama_started)

        answer = "".join(answer_parts).strip() or "No evidence found in the knowledge base."

        # Generic long-answer completeness recovery.
        # If long_explanation returns a tiny answer despite retrieved evidence,
        # run one stricter expansion pass before verifier.
        #
        # Important:
        # This retry must stream. Do not use generate_text() here, because that
        # waits for the full answer and then dumps everything into the UI at once.
        if _long_answer_too_short(answer, pipeline_used, citations):
            yield _sse(
                "verification_status",
                {
                    "message": "Expanding short long-form answer...",
                    "verification_verdict": "expansion_running",
                    "progress": 88,
                    "progress_label": "Expanding answer.",
                },
            )

            retry_prompt = LONG_EXPLANATION_RETRY_PROMPT.format(
                question=effective_question,
                short_answer=answer,
                context_text=context_text,
            )

            # Tell the frontend to clear the tiny first answer before streaming
            # the expanded replacement.
            yield _sse(
                "answer_replace",
                {
                    "answer": "",
                    "progress": 88,
                    "progress_label": "Streaming expanded answer.",
                },
            )

            expanded_parts: list[str] = []

            try:
                for token in stream_generate_text(
                    retry_prompt,
                    temperature=0.0,
                    num_predict=max(num_predict, 1800),
                    timeout=600,
                    cancel_check=cancelled,
                ):
                    if cancelled():
                        yield _sse("cancelled", cancel_payload("answer_expansion"))
                        return

                    expanded_parts.append(token)
                    expanded_text = "".join(expanded_parts)
                    progress = min(94, 88 + (len(expanded_text) // 450))

                    yield _sse(
                        "token",
                        {
                            "token": token,
                            "progress": progress,
                            "progress_label": "Streaming expanded answer.",
                        },
                    )

            except LLMCancelled:
                yield _sse("cancelled", cancel_payload("answer_expansion"))
                return
            except Exception:
                expanded_parts = []

            expanded = "".join(expanded_parts).strip()

            if expanded and not _long_answer_too_short(expanded, pipeline_used, citations):
                answer = expanded
                verification_result = {
                    "verification_verdict": "expanded",
                    "unsupported_claims": [],
                    "verification_reason": "Expanded short long-form answer using retrieved context.",
                    "verified": False,
                }

                yield _sse(
                    "verification",
                    {
                        "verification_verdict": "expanded",
                        "unsupported_claims": [],
                        "verification_reason": "Expanded short long-form answer using retrieved context.",
                        "verified": False,
                        "corrected_answer": answer,
                        "progress": 95,
                        "progress_label": "Expansion complete.",
                    },
                )

        # Do not clear citations before verification.
        # If the model falsely denies evidence while citations exist, the verifier needs
        # both the draft answer and retrieved context to repair the answer.
        if cancelled():
            yield _sse("cancelled", cancel_payload("after_answer_generation"))
            return

        verification_result: dict[str, Any] = {
            "verification_verdict": "skipped",
            "unsupported_claims": [],
            "verification_reason": "Verifier skipped for this answer.",
            "verified": False,
        }

        # Exact file/code explanation already retrieves one specific file by source_id.
        # Verification adds large latency and little value for this narrow case.
        # For normal Q&A, verifier stays skipped unless the model denies evidence
        # even though retrieval produced citations.
        if (not exact_file_mode) and should_verify_answer(
            pipeline_used=pipeline_used,
            routing=routing,
            answer=answer,
            citations=citations,
        ):
            yield _sse(
                "verification_status",
                {
                    "message": "Verifying answer against retrieved evidence...",
                    "verification_verdict": "running",
                    "progress": 90,
                    "progress_label": "Verifying answer.",
                },
            )

            try:
                verification_result = verify_answer(
                    question=effective_question,
                    pipeline_used=pipeline_used,
                    context_text=context_text,
                    draft_answer=answer,
                    cancel_check=cancelled,
                )
            except LLMCancelled:
                yield _sse("cancelled", cancel_payload("verification"))
                return

            if cancelled():
                yield _sse("cancelled", cancel_payload("after_verification"))
                return

            verifier_verdict = verification_result.get("verification_verdict")
            RAG_VERIFIER_TOTAL.labels(
                pipeline=pipeline_used,
                verdict=str(verifier_verdict or "unknown"),
            ).inc()

            if verifier_verdict in {"needs_revision", "insufficient_evidence"}:
                corrected_answer = verification_result.get("corrected_answer", answer)
                if corrected_answer.strip():
                    answer = corrected_answer

            # Do not clear citations here.
            # If retrieved evidence exists, citations must remain visible for debugging/trust.
            if answer.strip() == "No evidence found in the knowledge base." and not citations:
                citations = []

            if answer_denies_evidence(answer, citations) and not citations:
                citations = []

            yield _sse(
                "verification",
                {
                    "verification_verdict": verification_result.get("verification_verdict"),
                    "unsupported_claims": verification_result.get("unsupported_claims", []),
                    "verification_reason": verification_result.get("verification_reason"),
                    "verified": verification_result.get("verified", False),
                    "corrected_answer": answer,
                    "progress": 95,
                    "progress_label": "Verification complete.",
                },
            )

        # Generic recovery:
        # If retrieval produced citations but the model still denies evidence,
        # run a stricter second pass before clearing citations.
        if (not exact_file_mode) and answer_denies_evidence(answer, citations):
            yield _sse(
                "verification_status",
                {
                    "message": "Recovering answer from retrieved evidence...",
                    "verification_verdict": "recovery_running",
                    "progress": 96,
                    "progress_label": "Recovering answer.",
                },
            )

            recovery_prompt = DENIAL_RECOVERY_PROMPT.format(
                question=effective_question,
                context_text=context_text,
            )

            try:
                recovered = generate_text(
                    recovery_prompt,
                    temperature=0.0,
                    num_predict=num_predict,
                    timeout=600,
                ).strip()
            except Exception:
                recovered = ""

            if recovered and not answer_denies_evidence(recovered, citations):
                answer = recovered
                verification_result = {
                    **verification_result,
                    "verification_verdict": "recovered",
                    "verification_reason": "Second-pass evidence recovery corrected a false no-evidence answer.",
                    "verified": False,
                }

                yield _sse(
                    "verification",
                    {
                        "verification_verdict": "recovered",
                        "unsupported_claims": [],
                        "verification_reason": "Second-pass evidence recovery corrected a false no-evidence answer.",
                        "verified": False,
                        "corrected_answer": answer,
                        "progress": 98,
                        "progress_label": "Recovery complete.",
                    },
                )

        # Final guard:
        # Never hide citations if retrieval found evidence.
        # If the model failed, show a transparent failure message with citations intact.
        if answer.strip() == "No evidence found in the knowledge base." and citations:
            answer = (
                "Relevant evidence was retrieved, but the answer model failed to extract a supported answer from it. "
                "Open the citations to inspect the retrieved chunks, or ask a narrower question."
            )
            verification_result = {
                **verification_result,
                "verification_verdict": "answer_generation_failed_with_evidence",
                "verification_reason": "Model returned no-evidence despite retrieved citations.",
                "verified": False,
            }
        elif answer.strip() == "No evidence found in the knowledge base.":
            citations = []

        if answer_denies_evidence(answer, citations) and citations:
            answer = (
                "Relevant evidence was retrieved, but the answer model denied or failed to use it. "
                "Open the citations to inspect the retrieved chunks, or ask a narrower question."
            )
            verification_result = {
                **verification_result,
                "verification_verdict": "answer_denied_retrieved_evidence",
                "verification_reason": "Model denied evidence despite retrieved citations.",
                "verified": False,
            }
        elif answer_denies_evidence(answer, citations):
            citations = []

        if cancelled():
            yield _sse("cancelled", cancel_payload("before_save"))
            return

        timings = _finalize_timing(timings, started)
        latency_seconds = time.perf_counter() - started
        latency_ms = int(latency_seconds * 1000)

        answer_metadata = _build_answer_metadata(
            routing=routing,
            pipeline_used=pipeline_used,
            retrieval_speed=clean_retrieval_speed,
            actual_retrieval_mode=actual_retrieval_mode,
            actual_retriever_used=actual_retriever_used,
            actual_query_shape=actual_query_shape,
            actual_reranker_used=actual_reranker_used,
            actual_neighbour_window=actual_neighbour_window,
            actual_retrieval_reason=actual_retrieval_reason,
            actual_adaptive_retrieval=actual_adaptive_retrieval,
            retrieved_chunks=len(compacted),
            use_graph_context=bool(use_graph_context and not exact_file_mode),
            graph_chunks_added=graph_chunks_added if not exact_file_mode else 0,
            verification_result=verification_result,
            latency_ms=latency_ms,
            requested_file=requested_file,
            resolved_file=resolved_file,
        )
        answer_metadata.update(_timing_payload(timings))

        audit_save_started = time.perf_counter()

        add_assistant_message(
            conversation_id=resolved_conversation_id,
            answer=answer,
            citations=citations,
            intent=routing.get("intent"),
            pipeline_used=pipeline_used,
            metadata=answer_metadata,
        )
        RAG_QUESTIONS_TOTAL.labels(pipeline=pipeline_used, status="success").inc()
        RAG_ANSWER_LATENCY_SECONDS.labels(pipeline=pipeline_used, status="success").observe(latency_seconds)
        audit_id = save_audit_event(
            conversation_id=resolved_conversation_id,
            question=normalized_question,
            answer=answer,
            routing={**routing, **verification_result},
            citations=citations,
            model=CHAT_MODEL,
            latency_ms=latency_ms,
        )

        timings["audit_save_ms"] = _elapsed_ms(audit_save_started)
        timings = _finalize_timing(timings, started)

        yield _sse(
            "done",
            {
                "status": "done",
                "request_id": request_id,
                "audit_id": audit_id,
                "latency_ms": latency_ms,
                "citations": citations,
                "verification_verdict": verification_result.get("verification_verdict"),
                "unsupported_claims": verification_result.get("unsupported_claims", []),
                "verification_reason": verification_result.get("verification_reason"),
                "verified": verification_result.get("verified", False),
                "progress": 100,
                "progress_label": "Done.",
                "retrieval_mode": actual_retrieval_mode,
                    "retrieval_speed": clean_retrieval_speed,
                    "retriever_used": actual_retriever_used,
                    "query_shape": actual_query_shape,
                    "reranker_used": actual_reranker_used,
                    "neighbour_window": actual_neighbour_window,
                    "retrieval_planner_reason": actual_retrieval_reason,
                    "adaptive_retrieval": actual_adaptive_retrieval,
                "requested_file": requested_file,
                "resolved_source_id": resolved_file.get("source_id") if resolved_file else None,
                "resolved_source_path": resolved_file.get("source_path") if resolved_file else None,
                "retrieved_chunks": len(compacted),
                **_timing_payload(timings),
            },
        )

    except Exception as exc:
        error_text = str(exc)
        metric_pipeline = locals().get("pipeline_used", "unknown")
        RAG_QUESTIONS_TOTAL.labels(pipeline=metric_pipeline, status="error").inc()
        RAG_ANSWER_LATENCY_SECONDS.labels(pipeline=metric_pipeline, status="error").observe(time.perf_counter() - started)
        if "timeout" in error_text.lower() or "timed out" in error_text.lower():
            RAG_OLLAMA_TIMEOUT_TOTAL.labels(stage="answer_generation").inc()
        yield _sse("error", {"message": error_text, "request_id": request_id})
    finally:
        unregister_request(request_id)
