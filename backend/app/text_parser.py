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


def parse_text_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".json":
        data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        text = json.dumps(data, indent=2, ensure_ascii=False)

    elif ext == ".csv":
        rows = []
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(" | ".join(row))
        text = "\n".join(rows)

    else:
        text = p.read_text(encoding="utf-8", errors="ignore")

    return {
        "path": str(p),
        "parser_type": "text",
        "file_type": ext,
        "text": text,
        "metadata": {
            "filename": p.name,
            "parent": str(p.parent),
        },
    }
