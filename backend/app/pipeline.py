from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from qdrant_client.models import PointStruct

from app.code_chunker import chunk_generic_code, chunk_python, chunk_terraform
from app.code_parser import parse_code_file, supports as supports_code
from app.delete_sync import sync_deleted_files
from app.docx_parser import parse_docx_file, supports as supports_docx
from app.embedding_service import get_embedding
from app.graph_extractor import build_graph_for_file
from app.graph_store import GraphStore
from app.incremental_ingest import collect_changed_files
from app.metadata_db import MetadataDB
from app.pdf_parser import parse_pdf_file, supports as supports_pdf
from app.qdrant_client import QDRANT_COLLECTION, delete_points_by_source_id, ensure_collection, get_client, get_chunks_by_source_id
from app.text_chunker import chunk_markdown, chunk_yaml, fixed_chunk_text
from app.text_parser import parse_text_file, supports as supports_text

logger = logging.getLogger(__name__)

SUPPORTED_GENERIC_EXTENSIONS = {
    ".md",
    ".txt",
    ".rst",
    ".tf",
    ".hcl",
    ".py",
    ".yml",
    ".yaml",
    ".json",
    ".sh",
    ".bash",
    ".cfg",
    ".ini",
    ".conf",
    ".env",
    ".log",
    ".csv",
    ".sql",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".pdf",
    ".docx",
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
    for file_path in files:
        if file_path not in seen:
            seen.add(file_path)
            unique_files.append(file_path)

    return unique_files


def parse_file(path: str) -> dict[str, Any]:
    if supports_pdf(path):
        return parse_pdf_file(path)

    if supports_docx(path):
        return parse_docx_file(path)

    if supports_code(path):
        return parse_code_file(path)

    if supports_text(path):
        return parse_text_file(path)

    raise ValueError(f"Unsupported file: {path}")


def _clean_chunks(chunks: list[str]) -> list[str]:
    cleaned: list[str] = []
    for chunk in chunks:
        text = (chunk or "").strip()
        if text:
            cleaned.append(text)
    return cleaned


def chunk_parsed_content(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    text = parsed["text"]
    file_type = parsed.get("file_type", "")
    parser_type = parsed.get("parser_type", "")

    if parser_type == "pdf":
        page_records = parsed.get("metadata", {}).get("pages", []) or []
        page_chunks: list[dict[str, Any]] = []

        for page in page_records:
            page_number = page.get("page_number")
            page_text = (page.get("text") or "").strip()
            if not page_text:
                continue

            chunks = _clean_chunks(fixed_chunk_text(page_text))
            for chunk in chunks:
                page_chunks.append(
                    {
                        "text": chunk,
                        "page_number": page_number,
                        "page_start": page_number,
                        "page_end": page_number,
                    }
                )

        return page_chunks

    if file_type == ".md":
        chunks = _clean_chunks(chunk_markdown(text))
    elif file_type in {".yml", ".yaml"}:
        chunks = _clean_chunks(chunk_yaml(text))
    elif file_type == ".py":
        chunks = _clean_chunks(chunk_python(text))
    elif file_type == ".tf":
        chunks = _clean_chunks(chunk_terraform(text))
    elif parser_type == "code":
        chunks = _clean_chunks(chunk_generic_code(text))
    else:
        chunks = _clean_chunks(fixed_chunk_text(text))

    return [{"text": chunk} for chunk in chunks]


def should_run_delete_sync(source_type: str) -> bool:
    return source_type in {"local", "git"}


def build_and_save_graph_for_ingested_file(
    *,
    source_id: str,
    source_type: str,
    source_path: str,
    parsed: dict[str, Any],
    chunk_records: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Builds graph rows immediately after a file is ingested.

    This keeps graph generation source-scoped and idempotent:
    same source_id -> old graph removed -> fresh graph inserted.
    """
    graph_chunks: list[dict[str, Any]] = []

    for idx, chunk_info in enumerate(chunks):
        chunk_text = (chunk_info.get("text") or "").strip()
        if not chunk_text:
            continue

        matching_record = chunk_records[idx] if idx < len(chunk_records) else {}

        graph_chunks.append(
            {
                "chunk_id": matching_record.get("chunk_id") or f"{source_id}:{idx}",
                "chunk_index": idx,
                "qdrant_point_id": matching_record.get("qdrant_point_id", ""),
                "text": chunk_text,
                "page_number": chunk_info.get("page_number"),
                "page_start": chunk_info.get("page_start"),
                "page_end": chunk_info.get("page_end"),
            }
        )

    graph_payload = build_graph_for_file(
        source_id=source_id,
        source_type=source_type,
        source_path=source_path,
        file_type=parsed.get("file_type", ""),
        parser_type=parsed.get("parser_type", ""),
        chunks=graph_chunks,
    )

    store = GraphStore()
    store.replace_source_graph(
        source_id=source_id,
        nodes=graph_payload.get("nodes", []),
        edges=graph_payload.get("edges", []),
    )

    return {
        "source_id": source_id,
        "chunks": len(graph_chunks),
        "nodes": len(graph_payload.get("nodes", [])),
        "edges": len(graph_payload.get("edges", [])),
    }


def build_graph_for_existing_source(
    *,
    source_id: str,
    source_type: str,
    source_path: str,
    file_type: str,
    parser_type: str,
) -> dict[str, Any]:
    """
    Builds or refreshes graph rows for an already-ingested source.

    Used for duplicate uploads:
    - duplicate file content is not re-ingested
    - existing Qdrant chunks are reused
    - missing/stale graph rows are rebuilt
    """
    existing_chunks = get_chunks_by_source_id(source_id)

    if not existing_chunks:
        return {
            "source_id": source_id,
            "source_path": source_path,
            "chunks": 0,
            "nodes": 0,
            "edges": 0,
            "status": "skipped_no_chunks",
        }

    graph_chunks: list[dict[str, Any]] = []

    for chunk in existing_chunks:
        try:
            chunk_index = int(chunk.get("chunk_index") or 0)
        except (TypeError, ValueError):
            chunk_index = 0

        graph_chunks.append(
            {
                "chunk_id": f"{source_id}:{chunk_index}",
                "chunk_index": chunk_index,
                "qdrant_point_id": "",
                "text": chunk.get("text", ""),
                "page_number": chunk.get("page_number"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
            }
        )

    graph_chunks.sort(key=lambda item: int(item.get("chunk_index") or 0))

    graph_payload = build_graph_for_file(
        source_id=source_id,
        source_type=source_type,
        source_path=source_path,
        file_type=file_type,
        parser_type=parser_type,
        chunks=graph_chunks,
    )

    store = GraphStore()
    store.replace_source_graph(
        source_id=source_id,
        nodes=graph_payload.get("nodes", []),
        edges=graph_payload.get("edges", []),
    )

    return {
        "source_id": source_id,
        "source_path": source_path,
        "chunks": len(graph_chunks),
        "nodes": len(graph_payload.get("nodes", [])),
        "edges": len(graph_payload.get("edges", [])),
        "status": "rebuilt_existing_duplicate_source",
    }


def ingest_paths(paths: list[str], source_type: str = "local") -> dict[str, Any]:
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
            "failed_files": [],
            "duplicate_files": [],
            "graph_built": [],
            "graph_failed": [],
        }

    client = get_client()
    total_files = 0
    total_chunks = 0
    vector_size = None
    failed_files: list[dict[str, str]] = []
    duplicate_files: list[dict[str, str]] = []
    graph_built: list[dict[str, Any]] = []
    graph_failed: list[dict[str, str]] = []

    for item in changed_files:
        file_path = item["path"]
        source_id = item["source_id"]
        file_hash = item["file_hash"]

        try:
            if source_type == "upload":
                existing = metadata_db.get_active_file_by_hash(file_hash=file_hash, source_type="upload")
                if existing and existing["source_path"] != file_path:
                    existing_record = dict(existing)

                    logger.info(
                        "Skipping duplicate upload %s; already ingested as %s",
                        file_path,
                        existing_record.get("source_path", ""),
                    )

                    duplicate_info = {
                        "path": file_path,
                        "existing_source_id": existing_record.get("source_id", ""),
                        "existing_source_path": existing_record.get("source_path", ""),
                    }

                    try:
                        duplicate_graph_result = build_graph_for_existing_source(
                            source_id=existing_record.get("source_id", ""),
                            source_type=existing_record.get("source_type", source_type),
                            source_path=existing_record.get("source_path", ""),
                            file_type=existing_record.get("file_type", ""),
                            parser_type=existing_record.get("parser_type", ""),
                        )
                        duplicate_graph_result["reason"] = "duplicate_upload_existing_source"
                        graph_built.append(duplicate_graph_result)
                        duplicate_info["graph_status"] = duplicate_graph_result.get("status", "unknown")
                    except Exception as graph_exc:
                        logger.exception(
                            "Failed to build graph for duplicate existing source: %s",
                            existing_record.get("source_path", ""),
                        )
                        graph_failed.append(
                            {
                                "path": existing_record.get("source_path", ""),
                                "source_id": existing_record.get("source_id", ""),
                                "error": str(graph_exc),
                                "reason": "duplicate_upload_existing_source",
                            }
                        )
                        duplicate_info["graph_status"] = "failed"

                    Path(file_path).unlink(missing_ok=True)
                    duplicate_files.append(duplicate_info)
                    continue

            parsed = parse_file(file_path)
            chunks = chunk_parsed_content(parsed)

            if not chunks:
                logger.warning("Skipping file with no chunkable content: %s", file_path)
                failed_files.append(
                    {
                        "path": file_path,
                        "error": "No chunkable content produced",
                    }
                )
                continue

            delete_points_by_source_id(source_id)

            points: list[PointStruct] = []
            chunk_records: list[dict[str, Any]] = []

            for idx, chunk_info in enumerate(chunks):
                chunk_text = (chunk_info.get("text") or "").strip()
                if not chunk_text:
                    continue

                embedding = get_embedding(chunk_text)

                if vector_size is None:
                    vector_size = len(embedding)
                    ensure_collection(client, vector_size)

                point_id = str(uuid.uuid4())
                chunk_id = str(uuid.uuid4())

                payload = {
                    "source": file_path,
                    "source_id": source_id,
                    "source_type": source_type,
                    "chunk_index": idx,
                    "text": chunk_text,
                    "file_type": parsed.get("file_type", ""),
                    "parser_type": parsed.get("parser_type", ""),
                }

                if chunk_info.get("page_number") is not None:
                    payload["page_number"] = chunk_info["page_number"]
                if chunk_info.get("page_start") is not None:
                    payload["page_start"] = chunk_info["page_start"]
                if chunk_info.get("page_end") is not None:
                    payload["page_end"] = chunk_info["page_end"]

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload,
                    )
                )

                preview_prefix = ""
                if chunk_info.get("page_number") is not None:
                    preview_prefix = f"[Page {chunk_info['page_number']}] "

                chunk_records.append(
                    {
                        "chunk_id": chunk_id,
                        "chunk_index": idx,
                        "qdrant_point_id": point_id,
                        "text_preview": f"{preview_prefix}{chunk_text[:200]}",
                    }
                )

            if not points:
                failed_files.append(
                    {
                        "path": file_path,
                        "error": "No non-empty chunks produced",
                    }
                )
                continue

            client.upsert(collection_name=QDRANT_COLLECTION, points=points)

            metadata_db.upsert_file(
                source_id=source_id,
                source_type=source_type,
                source_path=file_path,
                file_hash=file_hash,
                file_type=parsed.get("file_type", ""),
                parser_type=parsed.get("parser_type", ""),
                chunk_count=len(points),
                status="active",
            )
            metadata_db.replace_chunks(source_id=source_id, chunk_records=chunk_records)

            try:
                graph_result = build_and_save_graph_for_ingested_file(
                    source_id=source_id,
                    source_type=source_type,
                    source_path=file_path,
                    parsed=parsed,
                    chunk_records=chunk_records,
                    chunks=chunks,
                )
                graph_built.append(graph_result)
            except Exception as graph_exc:
                logger.exception("Failed to build graph for file: %s", file_path)
                graph_failed.append(
                    {
                        "path": file_path,
                        "source_id": source_id,
                        "error": str(graph_exc),
                    }
                )

            total_files += 1
            total_chunks += len(points)

        except Exception as exc:
            logger.exception("Failed to ingest file: %s", file_path)
            failed_files.append(
                {
                    "path": file_path,
                    "error": str(exc),
                }
            )

    return {
        "message": "Ingestion complete" if not failed_files else "Ingestion complete with some file failures",
        "total_discovered": len(all_files),
        "changed_files": total_files,
        "total_chunks": total_chunks,
        "deleted_sources": len(deleted),
        "failed_files": failed_files,
        "duplicate_files": duplicate_files,
        "graph_built": graph_built,
        "graph_failed": graph_failed,
    }
