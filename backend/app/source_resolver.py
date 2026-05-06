from __future__ import annotations

import os
import re
import sqlite3
from pathlib import PurePosixPath
from typing import Any

DB_PATH = os.getenv("METADATA_DB_PATH", "/app/data/infrarag.db")

CV_HINTS = {
    "cv",
    "resume",
    "résumé",
    "profile",
    "candidate profile",
    "professional profile",
}

DOCUMENT_HINTS = {
    "this document",
    "the document",
    "this file",
    "the file",
    "this source",
    "the source",
    "uploaded document",
    "uploaded file",
}

PROFILE_FILE_HINTS = {
    "cv",
    "resume",
    "profile",
    "ashish",
}

PROFILE_EXTENSIONS = {
    ".pdf",
    ".docx",
}

DOC_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".md",
    ".txt",
    ".rst",
}


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _basename(path: str) -> str:
    clean_path = (path or "").replace("\\", "/")
    return PurePosixPath(clean_path).name.lower()


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", _clean(value))
        if len(token) >= 2
    }


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _load_active_sources() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                source_id,
                source_type,
                source_path,
                file_type,
                parser_type,
                chunk_count,
                status,
                last_ingested_at
            FROM files
            WHERE status = 'active'
            ORDER BY last_ingested_at DESC
            LIMIT 1000
            """
        ).fetchall()

    return [dict(row) for row in rows]


def _has_any_phrase(question: str, phrases: set[str]) -> bool:
    q = _clean(question)
    return any(phrase in q for phrase in phrases)


def _looks_like_profile_query(question: str) -> bool:
    q = _clean(question)

    if _has_any_phrase(q, CV_HINTS):
        return True

    profile_markers = (
        "certification",
        "certifications",
        "certificate",
        "certificates",
        "skills",
        "experience",
        "work experience",
        "education",
    )

    return any(marker in q for marker in profile_markers) and any(
        person_marker in q for person_marker in ("ashish", "dev")
    )


def _looks_like_document_query(question: str) -> bool:
    q = _clean(question)
    return _has_any_phrase(q, DOCUMENT_HINTS)


def _score_source_for_question(source: dict[str, Any], question: str) -> int:
    q = _clean(question)
    q_tokens = _tokens(q)

    source_path = str(source.get("source_path") or "")
    source_type = str(source.get("source_type") or "")
    file_type = str(source.get("file_type") or "").lower()
    base = _basename(source_path)
    base_tokens = _tokens(base)

    score = 0

    # Prefer uploaded files for vague "this profile/document" questions.
    if source_type == "upload":
        score += 2

    # Prefer document-like file types.
    if file_type in DOC_EXTENSIONS:
        score += 2

    # Prefer CV/resume/profile filenames for profile queries.
    if _looks_like_profile_query(q):
        if file_type in PROFILE_EXTENSIONS:
            score += 3

        if any(hint in base for hint in PROFILE_FILE_HINTS):
            score += 5

        if "cv" in q and "cv" in base:
            score += 4
        if "resume" in q and "resume" in base:
            score += 4
        if "profile" in q and "profile" in base:
            score += 4
        if "ashish" in q and "ashish" in base:
            score += 4
        if "dev" in q and "dev" in base:
            score += 2

    # Token overlap with filename/path.
    overlap = q_tokens.intersection(base_tokens)
    score += min(len(overlap), 6)

    # Penalise code files for profile/document questions.
    if file_type in {".py", ".js", ".ts", ".tsx", ".jsx", ".tf", ".sh", ".sql"}:
        score -= 3

    return score


def resolve_source_for_question(
    question: str,
    source_id: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
) -> dict[str, Any]:
    """
    Resolve vague source references without hardcoding document content.

    Priority:
    1. If UI already provided source_id, keep it.
    2. If question says CV/resume/profile, prefer latest matching uploaded CV/profile source.
    3. If question says this document/file/source, prefer latest uploaded document-like source.
    4. Otherwise do not force a source.
    """
    cleaned = _clean(question)

    if source_id:
        return {
            "source_id": source_id,
            "mode": "selected_source",
            "reason": "Source was explicitly selected by the UI/request.",
            "score": None,
        }

    should_resolve = _looks_like_profile_query(cleaned) or _looks_like_document_query(cleaned)

    if not should_resolve:
        return {
            "source_id": None,
            "mode": "all_sources",
            "reason": "Question does not contain a source-specific profile/document hint.",
            "score": None,
        }

    sources = _load_active_sources()

    if source_type:
        sources = [item for item in sources if item.get("source_type") == source_type]

    if file_type:
        wanted_type = file_type.lower()
        sources = [item for item in sources if str(item.get("file_type") or "").lower() == wanted_type]

    if not sources:
        return {
            "source_id": None,
            "mode": "all_sources",
            "reason": "No active sources matched source filters.",
            "score": None,
        }

    scored = []
    for item in sources:
        score = _score_source_for_question(item, cleaned)
        scored.append((score, item))

    scored.sort(
        key=lambda pair: (
            pair[0],
            str(pair[1].get("last_ingested_at") or ""),
        ),
        reverse=True,
    )

    best_score, best = scored[0]

    # Conservative threshold:
    # Avoid forcing source if signal is weak.
    if best_score < 4:
        return {
            "source_id": None,
            "mode": "all_sources",
            "reason": "Source hint was weak; searching all sources.",
            "score": best_score,
        }

    return {
        "source_id": best.get("source_id"),
        "source_path": best.get("source_path"),
        "source_type": best.get("source_type"),
        "file_type": best.get("file_type"),
        "mode": "auto_resolved_source",
        "reason": "Auto-selected best matching active source for profile/document-style question.",
        "score": best_score,
    }
