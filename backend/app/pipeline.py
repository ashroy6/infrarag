from __future__ import annotations

import uuid
from pathlib import Path

from qdrant_client.models import PointStruct

from app.code_chunker import chunk_generic_code, chunk_python, chunk_terraform
from app.code_parser import parse_code_file, supports as supports_code
from app.delete_sync import sync_deleted_files
from app.embedding_service import get_embedding
from app.incremental_ingest import collect_changed_files
from app.metadata_db import MetadataDB
from app.pdf_parser import parse_pdf_file, supports as supports_pdf
from app.qdrant_client import delete_points_by_source_id, ensure_collection, get_client, QDRANT_COLLECTION
from app.text_chunker import chunk_markdown, chunk_yaml, fixed_chunk_text
from app.text_parser import parse_text_file, supports as supports_text

SUPPORTED_GENERIC_EXTENSIONS = {
    ".md", ".txt", ".tf", ".py", ".yml", ".yaml", ".json", ".sh", ".cfg",
    ".ini", ".log", ".sql", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".pdf"
}


def is_supported_file(path: Path) -> bool:
    if path.name.lower() == "dockerfile":
        return True
    return path.suffix.lower() in SUPPORTED_GENERIC_EXTENSIONS


def discover_files(paths: list[str]) -> list[str]:
    files: list[str] = []

    for raw in paths:
        p = Path(raw)
        if p.is_file():
            if is_supported_file(p):
                files.append(str(p))
        elif p.is_dir():
            for item in p.rglob("*"):
                if item.is_file() and is_supported_file(item):
                    files.append(str(item))

    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    return unique_files


def parse_file(path: str) -> dict:
    if supports_pdf(path):
        return parse_pdf_file(path)

    if supports_code(path):
        return parse_code_file(path)

    if supports_text(path):
        return parse_text_file(path)

    raise ValueError(f"Unsupported file: {path}")


def chunk_parsed_content(parsed: dict) -> list[str]:
    text = parsed["text"]
    file_type = parsed.get("file_type", "")
    parser_type = parsed.get("parser_type", "")

    if parser_type == "pdf":
        return fixed_chunk_text(text)

    if file_type == ".md":
        return chunk_markdown(text)

    if file_type in {".yml", ".yaml"}:
        return chunk_yaml(text)

    if file_type == ".py":
        return chunk_python(text)

    if file_type == ".tf":
        return chunk_terraform(text)

    if parser_type == "code":
        return chunk_generic_code(text)

    return fixed_chunk_text(text)


def should_run_delete_sync(source_type: str) -> bool:
    return source_type in {"local", "git"}


def ingest_paths(paths: list[str], source_type: str = "local") -> dict:
    metadata_db = MetadataDB()
    all_files = discover_files(paths)

    deleted = []
    if should_run_delete_sync(source_type):
        deleted = sync_deleted_files(
            source_type=source_type,
            current_paths=all_files,
            metadata_db=metadata_db,
        )

    changed_files = collect_changed_files(
        source_type=source_type,
        file_paths=all_files,
        metadata_db=metadata_db,
    )

    if not changed_files:
        return {
            "message": "No new or changed files to ingest",
            "total_discovered": len(all_files),
            "changed_files": 0,
            "deleted_sources": len(deleted),
        }

    client = get_client()
    total_files = 0
    total_chunks = 0
    vector_size = None

    for item in changed_files:
        file_path = item["path"]
        source_id = item["source_id"]
        file_hash = item["file_hash"]

        parsed = parse_file(file_path)
        chunks = chunk_parsed_content(parsed)

        delete_points_by_source_id(source_id)

        points: list[PointStruct] = []
        chunk_records: list[dict] = []

        for idx, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)

            if vector_size is None:
                vector_size = len(embedding)
                ensure_collection(client, vector_size)

            point_id = str(uuid.uuid4())
            chunk_id = str(uuid.uuid4())

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "source": file_path,
                        "source_id": source_id,
                        "source_type": source_type,
                        "chunk_index": idx,
                        "text": chunk,
                        "file_type": parsed.get("file_type", ""),
                        "parser_type": parsed.get("parser_type", ""),
                    },
                )
            )

            chunk_records.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "qdrant_point_id": point_id,
                    "text_preview": chunk[:200],
                }
            )

        if points:
            client.upsert(collection_name=QDRANT_COLLECTION, points=points)

        metadata_db.upsert_file(
            source_id=source_id,
            source_type=source_type,
            source_path=file_path,
            file_hash=file_hash,
            file_type=parsed.get("file_type", ""),
            parser_type=parsed.get("parser_type", ""),
            chunk_count=len(chunks),
            status="active",
        )
        metadata_db.replace_chunks(source_id=source_id, chunk_records=chunk_records)

        total_files += 1
        total_chunks += len(chunks)

    return {
        "message": "Ingestion complete",
        "total_discovered": len(all_files),
        "changed_files": total_files,
        "total_chunks": total_chunks,
        "deleted_sources": len(deleted),
    }
