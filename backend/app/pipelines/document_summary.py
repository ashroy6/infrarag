from __future__ import annotations

import os
from typing import Any

from app.context_utils import build_citations, build_context_text, compact_chunks, load_all_source_chunks
from app.llm_client import generate_text
from app.metadata_db import MetadataDB
from app.prompts import DOCUMENT_SUMMARY_MAP_PROMPT, DOCUMENT_SUMMARY_REDUCE_PROMPT
from app.response_formatter import no_evidence_response

# Faster defaults for local Ollama/laptop testing.
# Full-book production streaming/background mode comes later.
DOC_SUMMARY_BATCH_SIZE = int(os.getenv("DOC_SUMMARY_BATCH_SIZE", "20"))
DOC_SUMMARY_MAX_BATCHES = int(os.getenv("DOC_SUMMARY_MAX_BATCHES", "12"))
DOC_SUMMARY_MAP_NUM_PREDICT = int(os.getenv("DOC_SUMMARY_MAP_NUM_PREDICT", "220"))
DOC_SUMMARY_REDUCE_NUM_PREDICT = int(os.getenv("DOC_SUMMARY_REDUCE_NUM_PREDICT", "1800"))


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
    resolved_source_id, chunks = load_all_source_chunks(source_id)

    if not resolved_source_id or not chunks:
        return no_evidence_response()

    chunks = _filter_by_page_range(chunks, page_start, page_end)

    if not chunks:
        return no_evidence_response()

    all_batches = _batch_chunks(chunks, DOC_SUMMARY_BATCH_SIZE)
    batches = all_batches[:DOC_SUMMARY_MAX_BATCHES]

    partial_summaries: list[str] = []

    for index, batch in enumerate(batches, start=1):
        compacted = compact_chunks(
            batch,
            max_chars_per_chunk=900,
            max_total_chars=7000,
        )
        context_text = build_context_text(compacted)

        prompt = DOCUMENT_SUMMARY_MAP_PROMPT.format(context_text=context_text)
        partial = generate_text(
            prompt,
            temperature=0.0,
            num_predict=DOC_SUMMARY_MAP_NUM_PREDICT,
            timeout=240,
        )

        if partial.strip():
            partial_summaries.append(f"Batch {index} summary:\n{partial.strip()}")

    if not partial_summaries:
        return no_evidence_response()

    combined = "\n\n".join(partial_summaries)

    reduce_prompt = DOCUMENT_SUMMARY_REDUCE_PROMPT.format(
        question=question,
        partial_summaries=combined,
    )

    final_answer = generate_text(
        reduce_prompt,
        temperature=0.0,
        num_predict=DOC_SUMMARY_REDUCE_NUM_PREDICT,
        timeout=360,
    )

    db = MetadataDB()
    source_record = db.get_file(resolved_source_id)

    citation_chunks = compact_chunks(
        chunks[:20],
        max_chars_per_chunk=800,
        max_total_chars=6000,
    )

    note = ""
    if len(all_batches) > len(batches):
        note = (
            f"\n\nNote: This local fast-summary run used the first {len(batches)} "
            f"of {len(all_batches)} batches. Full streaming/background summary mode will cover the complete source."
        )

    return {
        "answer": final_answer + note,
        "citations": build_citations(citation_chunks),
        "summary_source_id": resolved_source_id,
        "summary_source_path": source_record.get("source_path") if source_record else None,
        "summary_chunk_count": len(chunks),
        "summary_batch_count": len(batches),
        "summary_total_batches": len(all_batches),
        "summary_limited": len(all_batches) > len(batches),
    }
