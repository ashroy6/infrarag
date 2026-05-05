from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import PurePosixPath
from typing import Any, Iterator

from app.answer_verifier import should_verify_answer, verify_answer
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
from app.llm_client import CHAT_MODEL, LLMCancelled, stream_generate_text
from app.prompts import (
    CODE_FILE_EXPLANATION_PROMPT,
    INCIDENT_RUNBOOK_PROMPT,
    LONG_EXPLANATION_PROMPT,
    NORMAL_QA_PROMPT,
    REPO_EXPLANATION_PROMPT,
)
from app.qdrant_client import get_chunks_by_source_id
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
) -> Iterator[str]:
    started = time.perf_counter()
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

        routing = decide_intent(effective_question, chat_context=chat_context)
        routing["original_question"] = normalized_question
        routing["resolved_question"] = effective_question
        routing["is_followup"] = followup.get("is_followup", False)
        routing["followup_reason"] = followup.get("reason", "")

        pipeline_used = routing.get("pipeline_used", "normal_qa")
        requested_file = _extract_requested_file(effective_question)
        exact_file_mode = (
            pipeline_used == "repo_explanation"
            and requested_file is not None
            and _is_file_explanation_question(effective_question)
        )

        if cancelled():
            yield _sse("cancelled", cancel_payload("after_planning"))
            return

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
                "retrieval_mode": "exact_file_source_id_lookup" if exact_file_mode else "vector_search",
                "requested_file": requested_file,
                "verification_verdict": "pending" if pipeline_used in {"long_explanation", "repo_explanation", "incident_runbook"} else "skipped",
                "progress": 15,
                "progress_label": "Planning complete. Searching evidence.",
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

        # Exact file/code explanation should be smaller and faster than repo-wide explanation.
        # It already uses exact chunks by source_id, so avoid oversized generation.
        if exact_file_mode:
            num_predict = CODE_FILE_NUM_PREDICT

        if cancelled():
            yield _sse("cancelled", cancel_payload("before_retrieval"))
            return

        resolved_file: dict[str, str] | None = None

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
            )

        if cancelled():
            yield _sse("cancelled", cancel_payload("after_retrieval"))
            return

        if not chunks:
            answer = no_evidence_response()["answer"]
            citations: list[dict[str, Any]] = []

            yield _sse(
                "citations",
                {
                    "citations": citations,
                    "progress": 35,
                    "progress_label": "No evidence found.",
                    "retrieval_mode": "exact_file_source_id_lookup" if exact_file_mode else "vector_search",
                    "requested_file": requested_file,
                    "resolved_source_id": resolved_file.get("source_id") if resolved_file else None,
                    "resolved_source_path": resolved_file.get("source_path") if resolved_file else None,
                    "retrieved_chunks": 0,
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
                    "retrieval_mode": "exact_file_source_id_lookup" if exact_file_mode else "vector_search",
                    "requested_file": requested_file,
                    "resolved_source_id": resolved_file.get("source_id") if resolved_file else None,
                    "resolved_source_path": resolved_file.get("source_path") if resolved_file else None,
                    "retrieved_chunks": 0,
                },
            )
            return

        if not exact_file_mode:
            top_score = max(float(c.get("score", 0.0) or 0.0) for c in chunks)
            if top_score < MIN_SCORE_THRESHOLD:
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

        compacted = compact_chunks(
            chunks,
            max_total_chars=18000 if exact_file_mode else 11000,
        )
        citations = build_citations(compacted)
        context_text = build_context_text(compacted)

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
                "retrieval_mode": "exact_file_source_id_lookup" if exact_file_mode else "vector_search",
                "requested_file": requested_file,
                "resolved_source_id": resolved_file.get("source_id") if resolved_file else None,
                "resolved_source_path": resolved_file.get("source_path") if resolved_file else None,
                "retrieved_chunks": len(compacted),
            },
        )

        answer_parts: list[str] = []

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

                answer_parts.append(token)
                progress = min(85, 45 + (len("".join(answer_parts)) // 120))
                yield _sse("token", {"token": token, "progress": progress, "progress_label": "Generating answer."})

        except LLMCancelled:
            yield _sse("cancelled", cancel_payload("answer_generation"))
            return

        answer = "".join(answer_parts).strip() or "No evidence found in the knowledge base."

        if answer == "No evidence found in the knowledge base.":
            citations = []

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
        # Keep verifier enabled for repo-level, long explanation, incident, and normal higher-risk answers.
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

            if verifier_verdict in {"needs_revision", "insufficient_evidence"}:
                corrected_answer = verification_result.get("corrected_answer", answer)
                if corrected_answer.strip():
                    answer = corrected_answer

            if answer.strip() == "No evidence found in the knowledge base.":
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

        if cancelled():
            yield _sse("cancelled", cancel_payload("before_save"))
            return

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
            routing={**routing, **verification_result},
            citations=citations,
            model=CHAT_MODEL,
            latency_ms=latency_ms,
        )

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
                "retrieval_mode": "exact_file_source_id_lookup" if exact_file_mode else "vector_search",
                "requested_file": requested_file,
                "resolved_source_id": resolved_file.get("source_id") if resolved_file else None,
                "resolved_source_path": resolved_file.get("source_path") if resolved_file else None,
                "retrieved_chunks": len(compacted),
            },
        )

    except Exception as exc:
        yield _sse("error", {"message": str(exc), "request_id": request_id})
    finally:
        unregister_request(request_id)
