from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Any

from app.context_utils import build_citations, build_context_text, compact_chunks
from app.llm_client import CHAT_MODEL, generate_text
from app.metadata_db import MetadataDB
from app.retrieve import retrieve_context

AGENT_DEFAULT_LIMIT = 8
AGENT_NUM_PREDICT = 900
AGENT_MAX_CONTEXT_CHARS = 12000

MAX_CHUNKS_PER_SOURCE_DEFAULT = 2

CSV_INTENT_TERMS = {
    "candidate",
    "candidates",
    "profile",
    "profiles",
    "csv",
    "spreadsheet",
    "table",
    "row",
    "rows",
    "invoice",
    "invoices",
    "vendor",
    "vendors",
    "purchase",
    "order",
    "orders",
    "ticket",
    "tickets",
    "contract",
    "register",
}

PREFERRED_TEXT_FILE_TYPES = {
    ".md",
    ".docx",
    ".txt",
    ".pdf",
    ".rst",
}

STRUCTURED_FILE_TYPES = {
    ".csv",
    ".json",
}


AGENT_RUN_PROMPT = """
You are an enterprise Agent Studio assistant.

You must answer only from the retrieved context assigned to this agent.
Do not use general knowledge.
Do not invent facts.
If the retrieved context is insufficient, say exactly what is missing.

Agent name:
{agent_name}

Agent description:
{agent_description}

Agent instructions:
{agent_instructions}

User task:
{user_task}

Retrieved context:
{context_text}

Rules:
- Use only the retrieved context.
- Keep the answer practical and structured.
- Include only facts supported by the context.
- If approval or escalation is required by the context, mention it.
- If no relevant evidence is present, reply exactly: No evidence found in this agent's assigned knowledge sources.

Return only the final answer.
""".strip()


def _safe_json(value: Any) -> str:
    return json.dumps(value or [], ensure_ascii=False)


def _run_status(status: str) -> str:
    return str(status or "unknown").strip().lower()


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _tokens(value: str) -> set[str]:
    text = _normalise_text(value)
    return set(re.findall(r"[a-zA-Z0-9_\-]+", text))


def _file_name(hit: dict[str, Any]) -> str:
    source = str(hit.get("source") or "")
    if not source:
        return ""
    return PurePosixPath(source.replace("\\", "/")).name.lower()


def _file_type(hit: dict[str, Any]) -> str:
    return str(hit.get("file_type") or "").lower()


def _score(hit: dict[str, Any]) -> float:
    for key in ("reranker_score", "score", "pre_cluster_score", "vector_score"):
        try:
            return float(hit.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
    return 0.0


def _query_suggests_structured_file(user_task: str) -> bool:
    task_tokens = _tokens(user_task)
    return bool(task_tokens.intersection(CSV_INTENT_TERMS))


def _filename_relevance_boost(user_task: str, hit: dict[str, Any]) -> float:
    """
    Adds a tiny deterministic boost when filename terms match the task.
    This helps keep leave_policy.md/onboarding_checklist.docx above unrelated CSV rows.
    """
    task_tokens = _tokens(user_task)
    file_tokens = _tokens(_file_name(hit))

    if not task_tokens or not file_tokens:
        return 0.0

    overlap = task_tokens.intersection(file_tokens)
    return min(0.08, len(overlap) * 0.035)


def _agent_relevance_score(user_task: str, hit: dict[str, Any]) -> float:
    base = _score(hit)
    file_type = _file_type(hit)

    adjusted = base + _filename_relevance_boost(user_task, hit)

    structured_requested = _query_suggests_structured_file(user_task)

    # Do not punish CSV/JSON when user actually asks about rows, tickets, candidates, invoices, etc.
    if file_type in STRUCTURED_FILE_TYPES and not structured_requested:
        adjusted -= 0.08

    if file_type in PREFERRED_TEXT_FILE_TYPES:
        adjusted += 0.02

    return round(max(0.0, adjusted), 6)


def _clean_agent_chunks(
    *,
    user_task: str,
    chunks: list[dict[str, Any]],
    limit: int,
    max_chunks_per_source: int = MAX_CHUNKS_PER_SOURCE_DEFAULT,
) -> list[dict[str, Any]]:
    """
    Cleans retrieval noise for agent runs while preserving backend permissions.

    It does not add any new source access. It only reorders and trims chunks already
    retrieved from the agent's assigned allowed_source_ids.
    """
    if not chunks:
        return []

    safe_limit = max(1, min(int(limit), 12))
    structured_requested = _query_suggests_structured_file(user_task)

    ranked: list[dict[str, Any]] = []
    for hit in chunks:
        item = dict(hit)
        item["agent_relevance_score"] = _agent_relevance_score(user_task, item)
        ranked.append(item)

    ranked.sort(
        key=lambda item: (
            item.get("agent_relevance_score", 0.0),
            _score(item),
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    per_source_count: dict[str, int] = defaultdict(int)

    for hit in ranked:
        source_id = str(hit.get("source_id") or hit.get("source") or "unknown")
        file_type = _file_type(hit)

        # For normal policy/document questions, avoid flooding citations with CSV/JSON.
        if (
            file_type in STRUCTURED_FILE_TYPES
            and not structured_requested
            and len(selected) >= 3
        ):
            continue

        if per_source_count[source_id] >= max_chunks_per_source:
            continue

        selected.append(hit)
        per_source_count[source_id] += 1

        if len(selected) >= safe_limit:
            break

    # If filtering was too aggressive, backfill from ranked list without duplicates.
    # Do not backfill CSV/JSON for normal policy/document questions.
    # Structured sources are included only when the user asks about candidates,
    # invoices, tickets, rows, tables, vendors, contracts, etc.
    seen_keys = {
        (str(item.get("source_id") or item.get("source")), item.get("chunk_index"))
        for item in selected
    }

    for hit in ranked:
        if len(selected) >= safe_limit:
            break

        file_type = _file_type(hit)
        if file_type in STRUCTURED_FILE_TYPES and not structured_requested:
            continue

        key = (str(hit.get("source_id") or hit.get("source")), hit.get("chunk_index"))
        if key in seen_keys:
            continue

        selected.append(hit)
        seen_keys.add(key)

    selected.sort(
        key=lambda item: (
            item.get("agent_relevance_score", 0.0),
            _score(item),
        ),
        reverse=True,
    )

    return selected[:safe_limit]


def run_agent(
    *,
    agent_id: str,
    user_task: str,
    conversation_id: str | None = None,
    tenant_id: str = "local",
    user_id: str = "ashish",
    limit: int = AGENT_DEFAULT_LIMIT,
) -> dict[str, Any]:
    started = time.perf_counter()
    db = MetadataDB()

    agent = db.get_agent(agent_id)
    if not agent:
        raise ValueError(f"Agent not found: {agent_id}")

    if _run_status(agent.get("status")) != "active":
        raise ValueError(f"Agent is not active: {agent_id}")

    clean_task = (user_task or "").strip()
    if not clean_task:
        raise ValueError("user_task is required")

    allowed_source_ids = db.list_agent_allowed_source_ids(agent_id)

    agent_run_id = db.create_agent_run(
        agent_id=agent_id,
        user_task=clean_task,
        tenant_id=tenant_id,
        user_id=user_id,
        conversation_id=conversation_id,
        status="running",
    )

    db.add_agent_run_event(
        agent_run_id=agent_run_id,
        event_type="run_started",
        event_status="info",
        message="Agent run started.",
        payload={
            "agent_id": agent_id,
            "agent_name": agent.get("agent_name"),
            "allowed_source_count": len(allowed_source_ids),
        },
    )

    if not allowed_source_ids:
        answer = "No evidence found in this agent's assigned knowledge sources."
        latency_ms = int((time.perf_counter() - started) * 1000)

        db.update_agent_run(
            agent_run_id=agent_run_id,
            status="completed",
            answer=answer,
            sources_json="[]",
            tools_json="[]",
            approval_status="not_required",
            latency_ms=latency_ms,
        )

        db.add_agent_run_event(
            agent_run_id=agent_run_id,
            event_type="no_sources",
            event_status="warning",
            message="Agent has no assigned active source IDs.",
            payload={},
        )

        return {
            "status": "completed",
            "agent_run_id": agent_run_id,
            "agent": agent,
            "answer": answer,
            "citations": [],
            "allowed_source_ids": [],
            "latency_ms": latency_ms,
            "model": CHAT_MODEL,
        }

    db.add_agent_run_event(
        agent_run_id=agent_run_id,
        event_type="retrieval_started",
        event_status="info",
        message="Retrieving from assigned agent sources only.",
        payload={"allowed_source_ids": allowed_source_ids},
    )

    safe_limit = max(1, min(int(limit), 12))

    chunks = retrieve_context(
        query=clean_task,
        limit=safe_limit,
        retrieval_plan={
            "rewritten_queries": [clean_task],
            "candidate_top_k": 50,
            "final_top_k": max(6, safe_limit),
            "source_strategy": "allow_multiple_sources",
        },
        allowed_source_ids=allowed_source_ids,
        use_graph_context=False,
    )

    cleaned_chunks = _clean_agent_chunks(
        user_task=clean_task,
        chunks=chunks,
        limit=safe_limit,
    )

    compacted = compact_chunks(cleaned_chunks, max_total_chars=AGENT_MAX_CONTEXT_CHARS)
    citations = build_citations(compacted)
    context_text = build_context_text(compacted)

    db.add_agent_run_event(
        agent_run_id=agent_run_id,
        event_type="retrieval_completed",
        event_status="info",
        message="Agent retrieval completed and cleaned.",
        payload={
            "retrieved_chunks_before_cleanup": len(chunks),
            "retrieved_chunks_after_cleanup": len(compacted),
            "citation_count": len(citations),
        },
    )

    if not compacted:
        answer = "No evidence found in this agent's assigned knowledge sources."
    else:
        prompt = AGENT_RUN_PROMPT.format(
            agent_name=agent.get("agent_name") or "",
            agent_description=agent.get("agent_description") or "",
            agent_instructions=agent.get("instructions") or "",
            user_task=clean_task,
            context_text=context_text,
        )

        db.add_agent_run_event(
            agent_run_id=agent_run_id,
            event_type="generation_started",
            event_status="info",
            message="Generating agent answer with assigned-source context.",
            payload={"model": CHAT_MODEL},
        )

        answer = generate_text(
            prompt,
            temperature=0.0,
            num_predict=AGENT_NUM_PREDICT,
            timeout=600,
        ).strip()

        if not answer:
            answer = "No evidence found in this agent's assigned knowledge sources."

    latency_ms = int((time.perf_counter() - started) * 1000)

    db.update_agent_run(
        agent_run_id=agent_run_id,
        status="completed",
        answer=answer,
        sources_json=_safe_json(citations),
        tools_json="[]",
        approval_status="not_required",
        latency_ms=latency_ms,
    )

    db.add_agent_run_event(
        agent_run_id=agent_run_id,
        event_type="run_completed",
        event_status="success",
        message="Agent run completed.",
        payload={
            "latency_ms": latency_ms,
            "citation_count": len(citations),
        },
    )

    return {
        "status": "completed",
        "agent_run_id": agent_run_id,
        "agent": agent,
        "answer": answer,
        "citations": citations,
        "allowed_source_ids": allowed_source_ids,
        "retrieved_chunks": len(compacted),
        "latency_ms": latency_ms,
        "model": CHAT_MODEL,
    }
