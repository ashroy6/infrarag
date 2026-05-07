from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from app.audit import save_audit_event
from app.chat_history import add_assistant_message
from app.context_utils import build_citations, build_context_text, compact_chunks, load_all_source_chunks
from app.llm_client import CHAT_MODEL, generate_text
from app.metadata_db import MetadataDB
from app.prompts import DOCUMENT_SUMMARY_MAP_PROMPT, DOCUMENT_SUMMARY_REDUCE_PROMPT
from app.router import PIPELINE_LABELS
from app.retrieve import retrieve_context

DB_PATH = os.getenv("METADATA_DB_PATH", "/app/data/infrarag.db")

DOC_SUMMARY_BATCH_SIZE = int(os.getenv("DOC_SUMMARY_BATCH_SIZE", "20"))
DOC_SUMMARY_MAX_BATCHES = int(os.getenv("DOC_SUMMARY_MAX_BATCHES", "50"))
DOC_SUMMARY_MAP_NUM_PREDICT = int(os.getenv("DOC_SUMMARY_MAP_NUM_PREDICT", "220"))
DOC_SUMMARY_REDUCE_NUM_PREDICT = int(os.getenv("DOC_SUMMARY_REDUCE_NUM_PREDICT", "1600"))


def _connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summary_jobs (
                job_id TEXT PRIMARY KEY,
                conversation_id TEXT,
                question TEXT NOT NULL,
                source_id TEXT,
                status TEXT NOT NULL,
                progress_current INTEGER DEFAULT 0,
                progress_total INTEGER DEFAULT 0,
                answer TEXT,
                citations_json TEXT,
                error TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_summary_jobs_created_at
            ON summary_jobs(created_at)
            """
        )


def _update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return

    allowed = {
        "status",
        "progress_current",
        "progress_total",
        "answer",
        "citations_json",
        "error",
        "source_id",
    }

    updates = []
    values = []

    for key, value in fields.items():
        if key not in allowed:
            continue
        updates.append(f"{key} = ?")
        values.append(value)

    updates.append("updated_at = CURRENT_TIMESTAMP")
    values.append(job_id)

    with _connect() as conn:
        conn.execute(
            f"""
            UPDATE summary_jobs
            SET {", ".join(updates)}
            WHERE job_id = ?
            """,
            values,
        )


def get_summary_job(job_id: str) -> dict[str, Any] | None:
    _init_db()

    with _connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM summary_jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()

    if not row:
        return None

    item = dict(row)

    try:
        item["citations"] = json.loads(item.get("citations_json") or "[]")
    except json.JSONDecodeError:
        item["citations"] = []

    return item


def _batch_chunks(chunks: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]


def _filter_by_page_range(
    chunks: list[dict[str, Any]],
    page_start: int | None,
    page_end: int | None,
) -> list[dict[str, Any]]:
    if page_start is None and page_end is None:
        return chunks

    start = page_start if page_start is not None else 1
    end = page_end if page_end is not None else 10**9

    filtered: list[dict[str, Any]] = []
    for chunk in chunks:
        page = chunk.get("page_number")
        if page is None:
            filtered.append(chunk)
            continue

        try:
            page_number = int(page)
        except (TypeError, ValueError):
            continue

        if start <= page_number <= end:
            filtered.append(chunk)

    return filtered



def _resolve_summary_source_id(
    question: str,
    explicit_source_id: str | None,
    routing: dict[str, Any],
    page_start: int | None,
    page_end: int | None,
) -> str | None:
    """
    Resolve the best source for a full document summary.

    Generic behaviour:
    - If the user selected a source, use it.
    - Otherwise retrieve matching chunks first.
    - Group retrieved chunks by source_id.
    - Pick the source with the strongest cluster score.
    """
    if explicit_source_id:
        return explicit_source_id

    retrieval_plan = routing.get("planner") or routing

    try:
        candidate_chunks = retrieve_context(
            question,
            limit=10,
            page_start=page_start,
            page_end=page_end,
            retrieval_plan=retrieval_plan,
        )
    except Exception:
        return None

    if not candidate_chunks:
        return None

    grouped: dict[str, list[dict[str, Any]]] = {}

    for chunk in candidate_chunks:
        sid = chunk.get("source_id")
        if not sid:
            continue
        grouped.setdefault(str(sid), []).append(chunk)

    if not grouped:
        return None

    best_source_id = None
    best_score = -1.0

    for sid, chunks in grouped.items():
        scores = sorted(
            [float(c.get("score", 0.0) or 0.0) for c in chunks],
            reverse=True,
        )

        top_scores = scores[:5]
        avg_top = sum(top_scores) / max(len(top_scores), 1)
        max_score = max(scores) if scores else 0.0
        count_bonus = min(len(chunks), 6) * 0.03

        cluster_score = (0.60 * avg_top) + (0.30 * max_score) + count_bonus

        if cluster_score > best_score:
            best_score = cluster_score
            best_source_id = sid

    return best_source_id


def _run_document_summary_job(
    job_id: str,
    question: str,
    conversation_id: str | None,
    routing: dict[str, Any],
    source_id: str | None,
    page_start: int | None,
    page_end: int | None,
) -> None:
    started = time.perf_counter()

    try:
        _update_job(job_id, status="running")

        resolved_input_source_id = _resolve_summary_source_id(
            question=question,
            explicit_source_id=source_id,
            routing=routing,
            page_start=page_start,
            page_end=page_end,
        )

        resolved_source_id, chunks = load_all_source_chunks(resolved_input_source_id)

        if not resolved_source_id or not chunks:
            answer = "No evidence found in the knowledge base."
            _update_job(
                job_id,
                status="done",
                source_id=resolved_source_id or "",
                progress_current=0,
                progress_total=0,
                answer=answer,
                citations_json="[]",
            )
            return

        chunks = _filter_by_page_range(chunks, page_start, page_end)
        if not chunks:
            answer = "No evidence found in the knowledge base."
            _update_job(
                job_id,
                status="done",
                source_id=resolved_source_id,
                progress_current=0,
                progress_total=0,
                answer=answer,
                citations_json="[]",
            )
            return

        all_batches = _batch_chunks(chunks, DOC_SUMMARY_BATCH_SIZE)
        batches = all_batches[:DOC_SUMMARY_MAX_BATCHES]

        _update_job(
            job_id,
            source_id=resolved_source_id,
            progress_total=len(batches) + 1,
            progress_current=0,
        )

        partial_summaries: list[str] = []

        for index, batch in enumerate(batches, start=1):
            compacted = compact_chunks(
                batch,
                max_chars_per_chunk=900,
                max_total_chars=7000,
            )

            prompt = DOCUMENT_SUMMARY_MAP_PROMPT.format(
                context_text=build_context_text(compacted)
            )

            partial = generate_text(
                prompt,
                temperature=0.0,
                num_predict=DOC_SUMMARY_MAP_NUM_PREDICT,
                timeout=300,
            )

            if partial.strip():
                partial_summaries.append(f"Batch {index} summary:\n{partial.strip()}")

            _update_job(job_id, progress_current=index)

        if not partial_summaries:
            answer = "No evidence found in the knowledge base."
            citations: list[dict[str, Any]] = []
        else:
            reduce_prompt = DOCUMENT_SUMMARY_REDUCE_PROMPT.format(
                question=question,
                partial_summaries="\n\n".join(partial_summaries),
            )

            answer = generate_text(
                reduce_prompt,
                temperature=0.0,
                num_predict=DOC_SUMMARY_REDUCE_NUM_PREDICT,
                timeout=600,
            )

            if len(all_batches) > len(batches):
                answer += (
                    f"\n\nNote: This run summarised the first {len(batches)} "
                    f"of {len(all_batches)} batches. Increase DOC_SUMMARY_MAX_BATCHES "
                    f"for a complete full-source summary."
                )

            citation_chunks = compact_chunks(
                chunks[:20],
                max_chars_per_chunk=800,
                max_total_chars=6000,
            )
            citations = build_citations(citation_chunks)

        _update_job(
            job_id,
            status="done",
            progress_current=len(batches) + 1,
            progress_total=len(batches) + 1,
            answer=answer,
            citations_json=json.dumps(citations, ensure_ascii=False),
        )

        latency_ms = int((time.perf_counter() - started) * 1000)

        summary_metadata = {
            "pipeline_used": routing.get("pipeline_used") or "document_summary",
            "pipeline_label": routing.get("pipeline_label") or "Full document/book summarisation",
            "intent": routing.get("intent") or "document_summary",
            "intent_confidence": routing.get("confidence"),
            "intent_reason": "Background document summary completed.",
            "router": "background",
            "retrieval_speed": "background_summary",
            "mode_label": "Background Summary",
            "retrieval_mode": "full_source_chunks",
            "retriever_used": "Qdrant source scroll",
            "query_shape": "document_summary",
            "reranker_used": False,
            "neighbour_window": 0,
            "retrieved_chunks": len(citations),
            "verification_verdict": "n/a",
            "verification_reason": "Verifier not used for background document summary.",
            "graph_context_enabled": False,
            "graph_chunks_added": 0,
            "latency_ms": latency_ms,
            "progress": 100,
            "progress_label": "Done.",
            "source_id": resolved_source_id,
        }

        if conversation_id:
            add_assistant_message(
                conversation_id=conversation_id,
                answer=answer,
                citations=citations,
                intent=routing.get("intent"),
                pipeline_used=routing.get("pipeline_used"),
                metadata=summary_metadata,
            )

        save_audit_event(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            routing={**routing, **summary_metadata},
            citations=citations,
            model=CHAT_MODEL,
            latency_ms=latency_ms,
        )

    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
        )


def start_document_summary_job(
    question: str,
    conversation_id: str | None,
    routing: dict[str, Any],
    source_id: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> str:
    _init_db()

    job_id = str(uuid.uuid4())

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO summary_jobs (
                job_id,
                conversation_id,
                question,
                source_id,
                status,
                progress_current,
                progress_total
            )
            VALUES (?, ?, ?, ?, 'queued', 0, 0)
            """,
            (job_id, conversation_id, question, source_id),
        )

    thread = threading.Thread(
        target=_run_document_summary_job,
        kwargs={
            "job_id": job_id,
            "question": question,
            "conversation_id": conversation_id,
            "routing": routing,
            "source_id": source_id,
            "page_start": page_start,
            "page_end": page_end,
        },
        daemon=True,
    )
    thread.start()

    return job_id
