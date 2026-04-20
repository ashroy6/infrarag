from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

from fastapi import UploadFile

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/data/uploads"))

_SAFE_PART_RE = re.compile(r"[^A-Za-z0-9._-]")


def ensure_upload_dir() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def _sanitize_part(value: str) -> str:
    cleaned = _SAFE_PART_RE.sub("_", value.strip())
    cleaned = cleaned.strip(" .")
    return cleaned or "item"


def _sanitize_relative_path(raw_name: str) -> Path:
    raw_path = Path(raw_name)
    safe_parts: list[str] = []

    for part in raw_path.parts:
        if part in {"", ".", ".."}:
            continue
        safe_parts.append(_sanitize_part(part))

    if not safe_parts:
        safe_parts = ["uploaded_file"]

    return Path(*safe_parts)


async def _write_upload_to_path(file: UploadFile, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f"{destination.name}.part-{uuid.uuid4().hex[:8]}")

    try:
        with temp_path.open("wb") as out_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)

        temp_path.replace(destination)
        return str(destination)
    finally:
        await file.close()
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


async def save_upload(file: UploadFile) -> str:
    ensure_upload_dir()
    filename = file.filename or "uploaded_file"
    safe_name = _sanitize_relative_path(filename).name
    destination = UPLOAD_DIR / safe_name
    return await _write_upload_to_path(file, destination)


async def save_uploads(files: list[UploadFile], folder_name: str | None = None) -> tuple[str, list[str]]:
    ensure_upload_dir()

    safe_folder_name = _sanitize_part(folder_name or "folder_upload")
    base_dir = UPLOAD_DIR / safe_folder_name
    base_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []

    for file in files:
        raw_name = file.filename or "uploaded_file"
        relative_path = _sanitize_relative_path(raw_name)
        destination = base_dir / relative_path
        saved_path = await _write_upload_to_path(file, destination)
        saved_paths.append(saved_path)

    return str(base_dir), saved_paths
