from __future__ import annotations

from pathlib import Path
from typing import Any

from pypdf import PdfReader


def supports(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".pdf"


def parse_pdf_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    reader = PdfReader(str(p))

    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")

    text = "\n\n".join(pages).strip()

    return {
        "path": str(p),
        "parser_type": "pdf",
        "file_type": ".pdf",
        "text": text,
        "metadata": {
            "filename": p.name,
            "parent": str(p.parent),
            "page_count": len(reader.pages),
        },
    }
