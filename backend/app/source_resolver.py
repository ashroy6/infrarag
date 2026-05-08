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

CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".tf",
    ".sh",
    ".sql",
    ".yaml",
    ".yml",
    ".json",
}

ENTITY_LOOKUP_RE = re.compile(
    r"^\s*(who|what|where)\s+(is|are|was|were)\s+([a-zA-Z0-9_' -]{2,80})\??\s*$",
    re.IGNORECASE,
)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _basename(path: str) -> str:
    clean_path = (path or "").replace("\\", "/")
    return PurePosixPath(clean_path).name.lower()


def _stem_filename(path: str) -> str:
    base = _basename(path)
    for ext in DOC_EXTENSIONS.union(CODE_EXTENSIONS):
        if base.endswith(ext):
            return base[: -len(ext)]
    return base


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


def _looks_like_short_entity_lookup(question: str) -> bool:
    q = _clean(question)
    if not ENTITY_LOOKUP_RE.match(q):
        return False

    technical_markers = (
        "docker",
        "kubernetes",
        "terraform",
        "aws",
        "azure",
        "gcp",
        "python",
        "fastapi",
        "qdrant",
        "ollama",
        "prometheus",
        "grafana",
        "github",
        "gitlab",
        "pipeline",
        "workflow",
        "repo",
        "code",
        "function",
        "class",
        "module",
    )

    return not any(marker in q for marker in technical_markers)


def _is_document_like_source(source: dict[str, Any]) -> bool:
    file_type = str(source.get("file_type") or "").lower()
    source_type = str(source.get("source_type") or "").lower()
    source_path = str(source.get("source_path") or "")
    base = _basename(source_path)

    if file_type in DOC_EXTENSIONS:
        return True

    if source_type == "upload" and not any(base.endswith(ext) for ext in CODE_EXTENSIONS):
        return True

    return False


def _uploaded_document_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []

    for item in sources:
        source_type = str(item.get("source_type") or "").lower()
        if source_type != "upload":
            continue

        if not _is_document_like_source(item):
            continue

        docs.append(item)

    return docs


def _match_source_mentioned_in_question(
    sources: list[dict[str, Any]],
    question: str,
) -> dict[str, Any] | None:
    """
    Resolve explicit source references like:
    - in 01-small-novel.md, who is Mira?
    - from small novel, who is Mira?
    - in Yoga-Aphorisms-of-Patanjali.pdf, what is samadhi?
    """
    q = _clean(question)
    q_tokens = _tokens(q)

    best: tuple[int, dict[str, Any]] | None = None

    for item in sources:
        source_path = str(item.get("source_path") or "")
        base = _basename(source_path)
        stem = _stem_filename(source_path)
        base_tokens = _tokens(base)
        stem_tokens = _tokens(stem)

        score = 0

        if base and base in q:
            score += 20

        if stem and stem in q:
            score += 15

        overlap = q_tokens.intersection(base_tokens.union(stem_tokens))
        score += min(len(overlap), 8)

        # Boost meaningful filename token matches like "novel", "patanjali", "yoga".
        meaningful_overlap = {
            token
            for token in overlap
            if token not in {"pdf", "docx", "txt", "md", "rst", "the", "and", "file", "document"}
        }
        score += len(meaningful_overlap) * 2

        if score <= 0:
            continue

        if best is None or score > best[0]:
            best = (score, item)

    if best and best[0] >= 4:
        return best[1]

    return None


def _score_source_for_question(source: dict[str, Any], question: str) -> int:
    q = _clean(question)
    q_tokens = _tokens(q)

    source_path = str(source.get("source_path") or "")
    source_type = str(source.get("source_type") or "")
    file_type = str(source.get("file_type") or "").lower()
    base = _basename(source_path)
    base_tokens = _tokens(base)

    score = 0

    if source_type == "upload":
        score += 2

    if file_type in DOC_EXTENSIONS:
        score += 2

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

    overlap = q_tokens.intersection(base_tokens)
    score += min(len(overlap), 6)

    if file_type in CODE_EXTENSIONS:
        score -= 3

    return score


def resolve_source_for_question(
    question: str,
    source_id: str | None = None,
    source_type: str | None = None,
    file_type: str | None = None,
) -> dict[str, Any]:
    """
    Resolve vague source references safely.

    Priority:
    1. If UI already provided source_id, keep it.
    2. If question explicitly mentions a filename/source name, use that source.
    3. If question says CV/resume/profile, prefer matching uploaded CV/profile source.
    4. If question says this document/file/source, use the latest uploaded document.
    5. If there is exactly one uploaded document and the question is a short entity lookup, use it.
    6. If there are multiple uploaded documents and the question is ambiguous, do not force a source.
    """
    cleaned = _clean(question)

    if source_id:
        return {
            "source_id": source_id,
            "mode": "selected_source",
            "reason": "Source was explicitly selected by the UI/request.",
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

    explicit_source = _match_source_mentioned_in_question(sources, cleaned)
    if explicit_source:
        return {
            "source_id": explicit_source.get("source_id"),
            "source_path": explicit_source.get("source_path"),
            "source_type": explicit_source.get("source_type"),
            "file_type": explicit_source.get("file_type"),
            "mode": "explicit_source_mention",
            "reason": "Question mentioned a filename/source name, so retrieval was scoped to that source.",
            "score": 20,
        }

    uploaded_docs = _uploaded_document_sources(sources)

    # "this document" should mean the latest uploaded document.
    if _looks_like_document_query(cleaned):
        if uploaded_docs:
            latest_doc = uploaded_docs[0]
            return {
                "source_id": latest_doc.get("source_id"),
                "source_path": latest_doc.get("source_path"),
                "source_type": latest_doc.get("source_type"),
                "file_type": latest_doc.get("file_type"),
                "mode": "latest_uploaded_document_reference",
                "reason": "Question referred to this/the uploaded document, so retrieval used the latest uploaded document.",
                "score": 10,
            }

    # Safe shortcut only when there is one uploaded document.
    # Do NOT blindly select the latest upload when many documents exist.
    if _looks_like_short_entity_lookup(cleaned) and len(uploaded_docs) == 1:
        only_doc = uploaded_docs[0]
        return {
            "source_id": only_doc.get("source_id"),
            "source_path": only_doc.get("source_path"),
            "source_type": only_doc.get("source_type"),
            "file_type": only_doc.get("file_type"),
            "mode": "single_uploaded_document_entity_lookup",
            "reason": "Short entity lookup was scoped to the only active uploaded document.",
            "score": 10,
        }

    should_resolve = _looks_like_profile_query(cleaned)

    if not should_resolve:
        return {
            "source_id": None,
            "mode": "all_sources",
            "reason": "Question is ambiguous across multiple sources; no source was forced.",
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
