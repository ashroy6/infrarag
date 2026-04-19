from __future__ import annotations

from pathlib import Path
from typing import Any

CODE_EXTENSIONS = {
    ".py", ".tf", ".hcl", ".sh", ".bash", ".js", ".ts", ".tsx",
    ".jsx", ".java", ".go", ".rs", ".sql"
}


def supports(path: str | Path) -> bool:
    p = Path(path)
    return p.suffix.lower() in CODE_EXTENSIONS or p.name.lower() == "dockerfile"


def detect_language(path: Path) -> str:
    if path.name.lower() == "dockerfile":
        return "dockerfile"

    mapping = {
        ".py": "python",
        ".tf": "terraform",
        ".hcl": "hcl",
        ".sh": "shell",
        ".bash": "bash",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".sql": "sql",
    }
    return mapping.get(path.suffix.lower(), "code")


def parse_code_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")

    return {
        "path": str(p),
        "parser_type": "code",
        "file_type": p.suffix.lower() if p.name.lower() != "dockerfile" else "dockerfile",
        "language": detect_language(p),
        "text": text,
        "metadata": {
            "filename": p.name,
            "parent": str(p.parent),
            "line_count": len(text.splitlines()),
        },
    }
