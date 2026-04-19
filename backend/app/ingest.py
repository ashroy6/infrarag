from __future__ import annotations

import sys
from pathlib import Path

from app.pipeline import ingest_paths

DOCS_ROOT = Path("/docs")


def resolve_cli_paths(root: Path, cli_args: list[str]) -> list[str]:
    resolved = []

    for rel in cli_args:
        rel = rel.strip()
        if not rel:
            continue

        if rel.startswith("/"):
            p = Path(rel)
        else:
            p = root / rel

        if p.exists():
            resolved.append(str(p))
        else:
            print(f"WARNING: path not found, skipping: {rel}")

    return resolved


def main():
    if len(sys.argv) < 2:
        print("Usage: python app/ingest.py <path1> <path2> ...")
        print("Paths may be absolute or relative to /docs")
        sys.exit(1)

    resolved = resolve_cli_paths(DOCS_ROOT, sys.argv[1:])
    if not resolved:
        print("No valid paths found")
        sys.exit(1)

    result = ingest_paths(resolved, source_type="local")
    print(result)


if __name__ == "__main__":
    main()
