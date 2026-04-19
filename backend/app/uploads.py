from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import UploadFile

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/data/uploads"))


def ensure_upload_dir() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


async def save_upload(file: UploadFile) -> str:
    ensure_upload_dir()
    destination = UPLOAD_DIR / file.filename

    with destination.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    return str(destination)
