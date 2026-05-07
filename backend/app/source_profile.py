from __future__ import annotations

from typing import Any

from app.metadata_db import MetadataDB


def get_source_profile(source_id: str | None = None) -> dict[str, Any]:
    """
    Lightweight source profile.

    Phase-1 only uses metadata already present in SQLite.
    Later we can enrich this during ingestion with heading/code/table statistics.
    """
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
            "chunk_count": 0,
            "file_type": "",
            "parser_type": "",
            "has_code": False,
            "has_tables": False,
            "has_sections": False,
        }

    file_type = str(record.get("file_type") or "").lower()
    parser_type = str(record.get("parser_type") or "").lower()
    chunk_count = int(record.get("chunk_count") or 0)

    code_types = {".py", ".js", ".ts", ".tsx", ".jsx", ".tf", ".sh", ".go", ".java", ".rs"}

    return {
        "source_id": record.get("source_id"),
        "known": True,
        "source_kind": "code" if file_type in code_types or parser_type == "code" else "document",
        "chunk_count": chunk_count,
        "source_size": "large" if chunk_count >= 150 else ("medium" if chunk_count >= 40 else "small"),
        "file_type": file_type,
        "parser_type": parser_type,
        "has_code": file_type in code_types or parser_type == "code",
        "has_tables": file_type in {".csv", ".xlsx"} or "table" in parser_type,
        "has_sections": file_type in {".md", ".pdf", ".docx", ".rst"},
        "source_path": record.get("source_path"),
        "source_type": record.get("source_type"),
        "data_domain": record.get("data_domain"),
        "source_group": record.get("source_group"),
    }
