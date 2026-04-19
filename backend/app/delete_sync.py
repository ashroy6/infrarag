from __future__ import annotations

from pathlib import Path

from app.metadata_db import MetadataDB
from app.qdrant_client import delete_points_by_source_id


def sync_deleted_files(source_type: str, current_paths: list[str], metadata_db: MetadataDB) -> list[str]:
    current_set = {str(Path(p)) for p in current_paths}
    existing_files = metadata_db.list_active_files(source_type=source_type)

    deleted_source_ids: list[str] = []

    for record in existing_files:
        source_path = str(Path(record["source_path"]))
        if source_path not in current_set:
            source_id = record["source_id"]
            delete_points_by_source_id(source_id)
            metadata_db.mark_deleted(source_id)
            deleted_source_ids.append(source_id)

    return deleted_source_ids
