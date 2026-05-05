from __future__ import annotations

import os
import re
import sqlite3
from pathlib import PurePosixPath
from typing import Any

from app.context_utils import build_citations, build_context_text, compact_chunks
from app.llm_client import generate_text
from app.prompts import CODE_FILE_EXPLANATION_PROMPT, REPO_EXPLANATION_PROMPT
from app.qdrant_client import get_chunks_by_source_id
from app.response_formatter import no_evidence_response
from app.retrieve import retrieve_context

MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.35"))
REPO_EXPLANATION_LIMIT = int(os.getenv("REPO_EXPLANATION_LIMIT", "10"))
REPO_EXPLANATION_NUM_PREDICT = int(os.getenv("REPO_EXPLANATION_NUM_PREDICT", "1400"))
CODE_FILE_NUM_PREDICT = int(os.getenv("CODE_FILE_NUM_PREDICT", "900"))
EXACT_FILE_LIMIT = int(os.getenv("EXACT_FILE_LIMIT", "50"))

DEFAULT_DB_PATH = "/app/data/infrarag.db"

FILE_PATTERN = re.compile(
    r"(?P<file>[A-Za-z0-9_\-./]+(?:\.(?:py|js|ts|tsx|jsx|json|yaml|yml|md|txt|tf|sh|sql|html|css)|Dockerfile|dockerfile))",
    re.IGNORECASE,
)


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
    """
    Resolve a requested file/path to the indexed source_id using SQLite.

    This is generic:
    - no project-specific file names
    - matches basename or partial path
    - prefers active files
    - prefers exact path/basename matches
    """
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


def _generate_answer(
    *,
    question: str,
    chat_context: str,
    chunks: list[dict[str, Any]],
    max_total_chars: int,
    exact_file_mode: bool = False,
) -> dict[str, Any]:
    compacted = compact_chunks(chunks, max_total_chars=max_total_chars)
    citations = build_citations(compacted)
    context_text = build_context_text(compacted)

    prompt_template = CODE_FILE_EXPLANATION_PROMPT if exact_file_mode else REPO_EXPLANATION_PROMPT

    prompt = prompt_template.format(
        question=question,
        chat_context=chat_context or "No recent conversation context.",
        context_text=context_text,
    )

    answer = generate_text(
        prompt,
        temperature=0.0,
        num_predict=CODE_FILE_NUM_PREDICT if exact_file_mode else REPO_EXPLANATION_NUM_PREDICT,
    )

    if answer.strip() == "No evidence found in the knowledge base.":
        return no_evidence_response()

    return {
        "answer": answer,
        "citations": citations,
        "verification_context_text": context_text,
        "retrieved_chunks": len(compacted),
    }


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
    requested_file = _extract_requested_file(question)

    if requested_file and _is_file_explanation_question(question):
        chunks, resolved = _retrieve_exact_file_chunks(
            requested_file=requested_file,
            source_id=source_id,
            source_type=source_type,
            file_type=file_type,
            page_start=page_start,
            page_end=page_end,
        )

        if not chunks:
            response = no_evidence_response()
            response.update(
                {
                    "retrieval_mode": "exact_file_source_id_lookup",
                    "requested_file": requested_file,
                    "resolved_source_id": resolved.get("source_id") if resolved else None,
                    "resolved_source_path": resolved.get("source_path") if resolved else None,
                    "retrieved_chunks": 0,
                }
            )
            return response

        exact_question = (
            f"{question}\n\n"
            f"Important: Explain only the requested file `{requested_file}` using the retrieved file chunks. "
            f"Do not explain README, Docker, GitHub Actions, CI/CD, dependencies, or repo-level workflow unless those details are inside this exact file. "
            f"If the user asked line by line, explain the code in source order. "
            f"If evidence is missing, say exactly what is missing."
        )

        response = _generate_answer(
            question=exact_question,
            chat_context=chat_context,
            chunks=chunks,
            max_total_chars=18000,
            exact_file_mode=True,
        )

        response.update(
            {
                "retrieval_mode": "exact_file_source_id_lookup",
                "requested_file": requested_file,
                "resolved_source_id": resolved.get("source_id") if resolved else None,
                "resolved_source_path": resolved.get("source_path") if resolved else None,
            }
        )
        return response

    chunks = retrieve_context(
        question,
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

    top_score = max(float(chunk.get("score", 0.0) or 0.0) for chunk in chunks)
    if top_score < MIN_SCORE_THRESHOLD:
        return no_evidence_response()

    response = _generate_answer(
        question=question,
        chat_context=chat_context,
        chunks=chunks,
        max_total_chars=12000,
        exact_file_mode=False,
    )

    response.update(
        {
            "retrieval_mode": "repo_vector_search",
        }
    )

    return response
