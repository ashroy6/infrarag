from __future__ import annotations

import csv
import io
import os
import re
from pathlib import Path
from typing import Any

from app.code_chunker import chunk_generic_code, chunk_python, chunk_terraform
from app.text_chunker import chunk_markdown, chunk_yaml, fixed_chunk_text, split_large_chunks

DEFAULT_CHUNK_SIZE = int(os.getenv("DOCUMENT_CHUNK_SIZE", "1400"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DOCUMENT_CHUNK_OVERLAP", "180"))
LARGE_TEXT_THRESHOLD_BYTES = int(os.getenv("LARGE_TEXT_THRESHOLD_BYTES", "1000000"))
LARGE_TEXT_CHUNK_SIZE = int(os.getenv("LARGE_TEXT_CHUNK_SIZE", "5000"))
LARGE_TEXT_CHUNK_OVERLAP = int(os.getenv("LARGE_TEXT_CHUNK_OVERLAP", "500"))

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".tf", ".hcl", ".sh", ".bash",
    ".go", ".java", ".rs", ".sql", ".html", ".css", ".json",
}

HEADING_RE = re.compile(
    r"""
    ^\s*(
        \#{1,6}\s+.+                                      |
        (chapter|section|part|appendix|article|clause)\s+[\w.\-:]+.* |
        \d{1,4}(\.\d{1,4}){0,5}\s+.+                       |
        [A-Z][A-Z0-9 ,.'’"()/_-]{8,120}
    )\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

STRUCTURED_RECORD_RE = re.compile(
    r"""
    ^\s*(
        [A-Za-z][A-Za-z0-9 _.-]{0,60}\s+\d{1,4}:\d{1,4}\b.* |
        \d{1,4}:\d{1,4}\b.*                                 |
        (ticket|case|incident|invoice|order|request|clause|section)\s*[:#-]?\s*[A-Za-z0-9_.-]+.* |
        [A-Z]{2,10}-\d{2,10}\b.*                            |
        \d{1,5}(\.\d{1,5}){1,6}\s+.*
    )\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

EMAIL_BOUNDARY_RE = re.compile(
    r"^\s*(from:|to:|subject:|date:|sent:)\s+",
    re.IGNORECASE,
)


def _clean_text(value: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", (value or "").strip())


def _clean_chunk_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for item in items:
        text = _clean_text(str(item.get("text") or ""))
        if not text:
            continue
        clean_item = dict(item)
        clean_item["text"] = text
        cleaned.append(clean_item)
    return cleaned


def _with_common_metadata(
    chunks: list[str],
    *,
    chunk_strategy: str,
    section_title: str | None = None,
    record_type: str | None = None,
    page_number: int | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for chunk in chunks:
        text = _clean_text(chunk)
        if not text:
            continue

        item: dict[str, Any] = {
            "text": text,
            "chunk_strategy": chunk_strategy,
        }

        if section_title:
            item["section_title"] = section_title
        if record_type:
            item["record_type"] = record_type
        if page_number is not None:
            item["page_number"] = page_number
            item["page_start"] = page_start if page_start is not None else page_number
            item["page_end"] = page_end if page_end is not None else page_number

        records.append(item)

    return records


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        clean = line.strip().strip("#").strip()
        if clean and HEADING_RE.match(line.strip()):
            return clean[:180]
    return None


def _looks_like_structured_records(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 5:
        return False

    sample = lines[: min(len(lines), 300)]
    matches = sum(1 for line in sample if STRUCTURED_RECORD_RE.match(line))
    return matches >= 5 and matches / max(len(sample), 1) >= 0.08


def _looks_like_email_thread(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    sample = lines[: min(len(lines), 200)]
    matches = sum(1 for line in sample if EMAIL_BOUNDARY_RE.match(line))
    return matches >= 3


def _split_by_headings(text: str, strategy: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    sections: list[dict[str, Any]] = []
    current: list[str] = []
    current_title: str | None = None

    for line in lines:
        stripped = line.strip()
        is_heading = bool(stripped and HEADING_RE.match(stripped))

        if is_heading and current:
            section_text = "\n".join(current).strip()
            section_chunks = split_large_chunks(
                fixed_chunk_text(section_text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP),
                max_size=max(DEFAULT_CHUNK_SIZE + 500, 1800),
            )
            sections.extend(
                _with_common_metadata(
                    section_chunks,
                    chunk_strategy=strategy,
                    section_title=current_title or _first_heading(section_text),
                )
            )
            current = [line]
            current_title = stripped.strip("#").strip()[:180]
        else:
            current.append(line)
            if is_heading and not current_title:
                current_title = stripped.strip("#").strip()[:180]

    if current:
        section_text = "\n".join(current).strip()
        section_chunks = split_large_chunks(
            fixed_chunk_text(section_text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP),
            max_size=max(DEFAULT_CHUNK_SIZE + 500, 1800),
        )
        sections.extend(
            _with_common_metadata(
                section_chunks,
                chunk_strategy=strategy,
                section_title=current_title or _first_heading(section_text),
            )
        )

    return _clean_chunk_items(sections)


def _split_structured_records(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    chunks: list[dict[str, Any]] = []
    current: list[str] = []
    current_title: str | None = None

    def flush() -> None:
        nonlocal current, current_title
        record_text = "\n".join(current).strip()
        if not record_text:
            current = []
            current_title = None
            return

        if len(record_text) > DEFAULT_CHUNK_SIZE + 600:
            parts = fixed_chunk_text(
                record_text,
                chunk_size=DEFAULT_CHUNK_SIZE,
                overlap=DEFAULT_CHUNK_OVERLAP,
            )
        else:
            parts = [record_text]

        chunks.extend(
            _with_common_metadata(
                parts,
                chunk_strategy="structured_record_chunking",
                section_title=current_title,
                record_type="structured_record",
            )
        )
        current = []
        current_title = None

    for line in lines:
        stripped = line.strip()
        starts_record = bool(stripped and STRUCTURED_RECORD_RE.match(stripped))

        if starts_record and current:
            flush()

        current.append(line)

        if starts_record and not current_title:
            current_title = stripped[:180]

    if current:
        flush()

    if len(chunks) < 3:
        return _with_common_metadata(
            fixed_chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP),
            chunk_strategy="plain_text_chunking",
        )

    return _clean_chunk_items(chunks)


def _chunk_csv_text(text: str) -> list[dict[str, Any]]:
    clean = text.strip()
    if not clean:
        return []

    try:
        reader = csv.DictReader(io.StringIO(clean))
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError("CSV has no header")

        chunks: list[dict[str, Any]] = []
        for row_number, row in enumerate(reader, start=1):
            parts = [f"row_number: {row_number}"]
            for key in fieldnames:
                value = row.get(key)
                if value is not None and str(value).strip():
                    parts.append(f"{key}: {str(value).strip()}")

            row_text = "\n".join(parts).strip()
            if row_text:
                chunks.append(
                    {
                        "text": row_text,
                        "chunk_strategy": "table_aware_extraction",
                        "record_type": "table_row",
                        "row_number": row_number,
                    }
                )

        if chunks:
            return chunks
    except Exception:
        pass

    return _with_common_metadata(
        fixed_chunk_text(clean, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP),
        chunk_strategy="plain_text_chunking",
    )


def _chunk_email_thread(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    chunks: list[dict[str, Any]] = []
    current: list[str] = []

    for line in lines:
        if EMAIL_BOUNDARY_RE.match(line) and current and len("\n".join(current)) > 300:
            chunks.extend(
                _with_common_metadata(
                    fixed_chunk_text("\n".join(current), chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP),
                    chunk_strategy="email_thread_chunking",
                    record_type="email_message",
                )
            )
            current = [line]
        else:
            current.append(line)

    if current:
        chunks.extend(
            _with_common_metadata(
                fixed_chunk_text("\n".join(current), chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP),
                chunk_strategy="email_thread_chunking",
                record_type="email_message",
            )
        )

    return _clean_chunk_items(chunks)


def _chunk_pdf(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    page_records = parsed.get("metadata", {}).get("pages", []) or []
    chunks: list[dict[str, Any]] = []

    for page in page_records:
        page_number = page.get("page_number")
        page_text = _clean_text(page.get("text") or "")
        if not page_text:
            continue

        try:
            clean_page_number = int(page_number) if page_number is not None else None
        except (TypeError, ValueError):
            clean_page_number = None

        if _looks_like_structured_records(page_text):
            page_chunks = _split_structured_records(page_text)
            for item in page_chunks:
                item["chunk_strategy"] = "structured_record_chunking"
                item["page_number"] = clean_page_number
                item["page_start"] = clean_page_number
                item["page_end"] = clean_page_number
                chunks.append(item)
            continue

        if _first_heading(page_text):
            page_chunks = _split_by_headings(page_text, strategy="section_aware_chunking")
            for item in page_chunks:
                item["page_number"] = clean_page_number
                item["page_start"] = clean_page_number
                item["page_end"] = clean_page_number
                chunks.append(item)
            continue

        page_chunks = fixed_chunk_text(
            page_text,
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap=DEFAULT_CHUNK_OVERLAP,
        )
        chunks.extend(
            _with_common_metadata(
                page_chunks,
                chunk_strategy="page_aware_chunking",
                page_number=clean_page_number,
                page_start=clean_page_number,
                page_end=clean_page_number,
            )
        )

    return _clean_chunk_items(chunks)


def chunk_document(parsed: dict[str, Any], source_path: str | None = None) -> list[dict[str, Any]]:
    """
    Universal chunking entry point.

    It keeps answer pipelines stable and moves document intelligence into ingestion:
    - code_ast_chunking for code
    - page_aware_chunking for PDFs
    - heading/section-aware chunking for docs/books/manuals
    - table-aware row chunks for CSV
    - structured records for numbered clauses, tickets, invoices, verses, etc.
    - plain_text_chunking fallback
    """
    text = _clean_text(parsed.get("text") or "")
    file_type = str(parsed.get("file_type") or "").lower()
    parser_type = str(parsed.get("parser_type") or "").lower()
    source_name = Path(source_path or "").name.lower()

    if not text and parser_type != "pdf":
        return []

    if parser_type == "pdf":
        return _chunk_pdf(parsed)

    if file_type == ".csv":
        return _clean_chunk_items(_chunk_csv_text(text))

    if file_type == ".py":
        return _with_common_metadata(
            chunk_python(text),
            chunk_strategy="code_ast_chunking",
            record_type="python_symbol",
        )

    if file_type in {".tf", ".hcl"}:
        return _with_common_metadata(
            chunk_terraform(text),
            chunk_strategy="code_ast_chunking",
            record_type="terraform_block",
        )

    if parser_type == "code" or file_type in CODE_EXTENSIONS or source_name == "dockerfile":
        return _with_common_metadata(
            chunk_generic_code(text),
            chunk_strategy="code_ast_chunking",
            record_type="code_block",
        )

    if file_type in {".yml", ".yaml"}:
        return _with_common_metadata(
            chunk_yaml(text),
            chunk_strategy="heading_aware_chunking",
            record_type="yaml_block",
        )

    if file_type == ".md":
        chunks = chunk_markdown(text)
        return _with_common_metadata(
            chunks,
            chunk_strategy="heading_aware_chunking",
            section_title=_first_heading(text),
        )

    if _looks_like_email_thread(text):
        return _chunk_email_thread(text)

    if _looks_like_structured_records(text):
        return _split_structured_records(text)

    if file_type in {".rst", ".docx", ".html", ".htm"} or _first_heading(text):
        return _split_by_headings(text, strategy="section_aware_chunking")

    encoded_size = len(text.encode("utf-8", errors="ignore"))
    if encoded_size >= LARGE_TEXT_THRESHOLD_BYTES:
        return _with_common_metadata(
            fixed_chunk_text(
                text,
                chunk_size=LARGE_TEXT_CHUNK_SIZE,
                overlap=LARGE_TEXT_CHUNK_OVERLAP,
            ),
            chunk_strategy="plain_text_chunking",
        )

    return _with_common_metadata(
        fixed_chunk_text(
            text,
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap=DEFAULT_CHUNK_OVERLAP,
        ),
        chunk_strategy="plain_text_chunking",
    )
