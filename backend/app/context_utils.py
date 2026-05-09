from __future__ import annotations

import os
from typing import Any

from app.metadata_db import MetadataDB
from app.qdrant_client import get_chunks_by_source_id

MAX_CONTEXT_CHARS_PER_CHUNK = int(os.getenv("MAX_CONTEXT_CHARS_PER_CHUNK", "1400"))
MAX_TOTAL_CONTEXT_CHARS = int(os.getenv("MAX_TOTAL_CONTEXT_CHARS", "8000"))


def trim_text(text: str, max_chars: int) -> str:
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + " ..."


def compact_chunks(
    chunks: list[dict[str, Any]],
    max_chars_per_chunk: int = MAX_CONTEXT_CHARS_PER_CHUNK,
    max_total_chars: int = MAX_TOTAL_CONTEXT_CHARS,
) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    total_chars = 0

    for chunk in chunks:
        text = trim_text(chunk.get("text", ""), max_chars_per_chunk)
        if not text:
            continue

        projected = total_chars + len(text)
        if compacted and projected > max_total_chars:
            break

        item = dict(chunk)
        item["text"] = text
        compacted.append(item)
        total_chars += len(text)

    return compacted


def _format_reference_meta(chunk: dict[str, Any]) -> str:
    parts: list[str] = []

    if chunk.get("structured_reference"):
        parts.append(f"MatchedRef={chunk.get('structured_reference')}")

    if chunk.get("reference_labels"):
        labels = chunk.get("reference_labels") or []
        if isinstance(labels, list) and labels:
            parts.append("Refs=" + ", ".join(str(x) for x in labels[:12]))

    for key, label in (
        ("heading", "Heading"),
        ("parent_title", "Parent"),
        ("heading_path", "Path"),
        ("section_number", "Section"),
        ("subsection_start", "SubsectionStart"),
        ("subsection_end", "SubsectionEnd"),
        ("reference_type", "RefType"),
    ):
        value = chunk.get(key)
        if value not in (None, "", []):
            parts.append(f"{label}={value}")

    return " | ".join(parts)


def build_context_text(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []

    for chunk in chunks:
        page_info = ""
        if chunk.get("page_number") is not None:
            page_info = f" | Page: {chunk['page_number']}"
        elif chunk.get("page_start") is not None and chunk.get("page_end") is not None:
            page_info = f" | Pages: {chunk['page_start']}-{chunk['page_end']}"

        score = float(chunk.get("score", 0.0) or 0.0)
        reference_info = _format_reference_meta(chunk)
        reference_line = f" | {reference_info}" if reference_info else ""

        parts.append(
            "[Source: {source} | Source ID: {source_id} | Chunk: {chunk_index}{page_info} | Score: {score:.4f}{reference_line}]\n{text}".format(
                source=chunk.get("source", "unknown"),
                source_id=chunk.get("source_id", ""),
                chunk_index=chunk.get("chunk_index", -1),
                page_info=page_info,
                score=score,
                reference_line=reference_line,
                text=chunk.get("text", ""),
            )
        )

    return "\n\n".join(parts)


def build_citations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks, start=1):
        item: dict[str, Any] = {
            "citation_id": idx,
            "source": chunk.get("source", "unknown"),
            "source_id": chunk.get("source_id", ""),
            "chunk_index": chunk.get("chunk_index", -1),
            "score": float(chunk.get("score", 0.0) or 0.0),
            "file_type": chunk.get("file_type", ""),
            "source_type": chunk.get("source_type", ""),
        }

        for key in (
            "chunk_strategy",
            "reference_labels",
            "references",
            "section_number",
            "section_start",
            "section_end",
            "subsection_start",
            "subsection_end",
            "reference_type",
            "heading",
            "parent_title",
            "parent_id",
            "heading_path",
            "prev_chunk_index",
            "next_chunk_index",
            "structured_reference",
            "structured_reference_context_match",
            "structured_reference_fallback",
            "retrieval_channel",
            "retrieval_planner_reason",
        ):
            value = chunk.get(key)
            if value not in (None, "", []):
                item[key] = value

        if chunk.get("page_number") is not None:
            item["page_number"] = chunk.get("page_number")
        if chunk.get("page_start") is not None:
            item["page_start"] = chunk.get("page_start")
        if chunk.get("page_end") is not None:
            item["page_end"] = chunk.get("page_end")

        citations.append(item)

    return citations


def resolve_source_id(source_id: str | None = None) -> str | None:
    if source_id:
        return source_id

    db = MetadataDB()
    latest = db.get_latest_active_source()
    if latest:
        return latest.get("source_id")

    return None


def load_all_source_chunks(source_id: str | None = None) -> tuple[str | None, list[dict[str, Any]]]:
    resolved_source_id = resolve_source_id(source_id)
    if not resolved_source_id:
        return None, []

    chunks = get_chunks_by_source_id(resolved_source_id)
    return resolved_source_id, chunks
