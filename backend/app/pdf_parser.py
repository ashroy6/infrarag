from __future__ import annotations

from pathlib import Path
from typing import Any

from pypdf import PdfReader


def supports(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".pdf"


def parse_pdf_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    reader = PdfReader(str(p))

    pages: list[dict[str, Any]] = []
    page_texts: list[str] = []

    for idx, page in enumerate(reader.pages, start=1):
        try:
            extracted = (page.extract_text() or "").strip()
        except Exception:
            extracted = ""

        pages.append(
            {
                "page_number": idx,
                "text": extracted,
            }
        )

        if extracted:
            page_texts.append(f"[Page {idx}]\n{extracted}")

    text = "\n\n".join(page_texts).strip()

    return {
        "path": str(p),
        "parser_type": "pdf",
        "file_type": ".pdf",
        "text": text,
        "metadata": {
            "filename": p.name,
            "parent": str(p.parent),
            "page_count": len(reader.pages),
            "pages": pages,
        },
    }
