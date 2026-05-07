from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.metadata_db import MetadataDB

CODE_TYPES = {".py", ".js", ".ts", ".tsx", ".jsx", ".tf", ".hcl", ".sh", ".go", ".java", ".rs", ".sql"}
TABLE_TYPES = {".csv", ".xlsx", ".xls"}
DOC_TYPES = {".txt", ".md", ".rst", ".pdf", ".docx", ".html", ".htm"}


def _json_load(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _source_size(chunk_count: int) -> str:
    if chunk_count >= 150:
        return "large"
    if chunk_count >= 40:
        return "medium"
    return "small"


def _count_strategy(chunks: list[dict[str, Any]], strategy: str) -> int:
    return sum(1 for item in chunks if item.get("chunk_strategy") == strategy)


def build_source_profile(
    *,
    parsed: dict[str, Any],
    chunks: list[dict[str, Any]],
    source_path: str,
) -> dict[str, Any]:
    file_type = str(parsed.get("file_type") or Path(source_path).suffix or "").lower()
    parser_type = str(parsed.get("parser_type") or "").lower()
    chunk_count = len(chunks)

    strategies = sorted(
        {
            str(item.get("chunk_strategy") or "unknown")
            for item in chunks
            if str(item.get("chunk_strategy") or "").strip()
        }
    )

    has_pages = any(item.get("page_number") is not None for item in chunks)
    has_sections = any(str(item.get("section_title") or "").strip() for item in chunks)
    has_records = any(str(item.get("record_type") or "").strip() for item in chunks)
    has_tables = file_type in TABLE_TYPES or "table_aware_extraction" in strategies
    has_code = file_type in CODE_TYPES or parser_type == "code" or "code_ast_chunking" in strategies

    if has_code:
        source_kind = "code"
    elif has_tables:
        source_kind = "table"
    elif has_records:
        source_kind = "structured_document"
    elif has_sections or has_pages:
        source_kind = "document"
    elif file_type in DOC_TYPES:
        source_kind = "document"
    else:
        source_kind = "plain_text"

    return {
        "source_kind": source_kind,
        "source_size": _source_size(chunk_count),
        "chunk_count": chunk_count,
        "file_type": file_type,
        "parser_type": parser_type,
        "chunking_strategies": strategies,
        "primary_chunking_strategy": strategies[0] if len(strategies) == 1 else ("mixed" if strategies else "unknown"),
        "has_code": has_code,
        "has_tables": has_tables,
        "has_sections": has_sections,
        "has_headings": has_sections,
        "has_pages": has_pages,
        "has_records": has_records,
        "plain_text_chunks": _count_strategy(chunks, "plain_text_chunking"),
        "section_chunks": _count_strategy(chunks, "section_aware_chunking"),
        "heading_chunks": _count_strategy(chunks, "heading_aware_chunking"),
        "page_chunks": _count_strategy(chunks, "page_aware_chunking"),
        "table_chunks": _count_strategy(chunks, "table_aware_extraction"),
        "code_chunks": _count_strategy(chunks, "code_ast_chunking"),
        "email_chunks": _count_strategy(chunks, "email_thread_chunking"),
        "structured_record_chunks": _count_strategy(chunks, "structured_record_chunking"),
    }


def get_source_profile(source_id: str | None = None) -> dict[str, Any]:
    db = MetadataDB()

    if source_id:
        record = db.get_file(source_id)
    else:
        record = db.get_latest_active_source()

    if not record:
        return {
            "source_id": source_id,
            "known": False,
            "source_kind": "unknown",
            "source_size": "small",
            "chunk_count": 0,
            "file_type": "",
            "parser_type": "",
            "has_code": False,
            "has_tables": False,
            "has_sections": False,
            "has_headings": False,
            "has_pages": False,
            "has_records": False,
            "chunking_strategies": [],
            "primary_chunking_strategy": "unknown",
        }

    metadata = _json_load(record.get("metadata_json"))
    profile = metadata.get("source_profile") if isinstance(metadata.get("source_profile"), dict) else {}

    file_type = str(record.get("file_type") or "").lower()
    parser_type = str(record.get("parser_type") or "").lower()
    chunk_count = int(record.get("chunk_count") or profile.get("chunk_count") or 0)

    fallback_kind = "code" if file_type in CODE_TYPES or parser_type == "code" else "document"

    return {
        "source_id": record.get("source_id"),
        "known": True,
        "source_kind": profile.get("source_kind", fallback_kind),
        "chunk_count": chunk_count,
        "source_size": profile.get("source_size", _source_size(chunk_count)),
        "file_type": file_type,
        "parser_type": parser_type,
        "chunking_strategies": profile.get("chunking_strategies", []),
        "primary_chunking_strategy": profile.get("primary_chunking_strategy", "unknown"),
        "has_code": bool(profile.get("has_code", file_type in CODE_TYPES or parser_type == "code")),
        "has_tables": bool(profile.get("has_tables", file_type in TABLE_TYPES)),
        "has_sections": bool(profile.get("has_sections", file_type in {".md", ".pdf", ".docx", ".rst"})),
        "has_headings": bool(profile.get("has_headings", file_type in {".md", ".docx", ".rst"})),
        "has_pages": bool(profile.get("has_pages", file_type == ".pdf")),
        "has_records": bool(profile.get("has_records", False)),
        "source_path": record.get("source_path"),
        "source_type": record.get("source_type"),
        "data_domain": record.get("data_domain"),
        "source_group": record.get("source_group"),
        "metadata": metadata,
    }
