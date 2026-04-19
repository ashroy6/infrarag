from __future__ import annotations

from pathlib import Path

from app.metadata_db import MetadataDB
from app.state_store import build_source_id, sha256_file


def file_needs_ingest(source_type: str, source_path: str, metadata_db: MetadataDB) -> tuple[bool, str]:
    source_id = build_source_id(source_type, source_path)
    current_hash = sha256_file(source_path)
    existing = metadata_db.get_file(source_id)

    if not existing:
        return True, current_hash

    if existing["file_hash"] != current_hash:
        return True, current_hash

    return False, current_hash


def collect_changed_files(source_type: str, file_paths: list[str], metadata_db: MetadataDB) -> list[dict]:
    changed = []

    for path in file_paths:
        p = Path(path)
        if not p.is_file():
            continue

        needs_ingest, file_hash = file_needs_ingest(source_type, str(p), metadata_db)
        if needs_ingest:
            changed.append(
                {
                    "path": str(p),
                    "source_id": build_source_id(source_type, str(p)),
                    "file_hash": file_hash,
                }
            )

    return changed
