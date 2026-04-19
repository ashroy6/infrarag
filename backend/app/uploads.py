from __future__ import annotations

import os
from pathlib import Path

from fastapi import UploadFile

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/data/uploads"))


def ensure_upload_dir() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


async def save_upload(file: UploadFile) -> str:
    ensure_upload_dir()
    filename = file.filename or "uploaded_file"
    destination = UPLOAD_DIR / Path(filename).name
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("wb") as out_file:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out_file.write(chunk)

    await file.close()
    return str(destination)


async def save_uploads(files: list[UploadFile], folder_name: str | None = None) -> tuple[str, list[str]]:
    ensure_upload_dir()

    if folder_name:
        base_dir = UPLOAD_DIR / folder_name
    else:
        base_dir = UPLOAD_DIR / "folder_upload"

    base_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []

    for file in files:
        raw_name = file.filename or "uploaded_file"
        relative_path = Path(raw_name)
        destination = base_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("wb") as out_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)

        await file.close()
        saved_paths.append(str(destination))

    return str(base_dir), saved_paths
