from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".log", ".json", ".yaml", ".yml",
    ".csv", ".ini", ".cfg", ".conf", ".env"
}


def supports(path: str | Path) -> bool:
    return Path(path).suffix.lower() in TEXT_EXTENSIONS


def _read_text_with_fallback(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8", errors="ignore")


def _parse_json_text(raw_text: str) -> tuple[str, dict[str, Any]]:
    try:
        data = json.loads(raw_text)
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        return pretty, {"json_valid": True}
    except json.JSONDecodeError:
        return raw_text, {"json_valid": False, "json_parse_fallback": "raw_text"}


def _parse_csv_text(p: Path) -> str:
    rows: list[str] = []
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as file_obj:
        reader = csv.reader(file_obj)
        for row in reader:
            rows.append(" | ".join(cell.strip() for cell in row))
    return "\n".join(rows)


def parse_text_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".json":
        raw_text = _read_text_with_fallback(p)
        text, extra_metadata = _parse_json_text(raw_text)

    elif ext == ".csv":
        text = _parse_csv_text(p)
        extra_metadata = {}

    else:
        text = _read_text_with_fallback(p)
        extra_metadata = {}

    return {
        "path": str(p),
        "parser_type": "text",
        "file_type": ext,
        "text": text,
        "metadata": {
            "filename": p.name,
            "parent": str(p.parent),
            **extra_metadata,
        },
    }
