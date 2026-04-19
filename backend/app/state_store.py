from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_source_id(source_type: str, source_path: str) -> str:
    raw = f"{source_type}:{source_path}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
