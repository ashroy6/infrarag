from __future__ import annotations

import ast
import re


def chunk_python(text: str, max_size: int = 1800) -> list[str]:
    from app.text_chunker import fixed_chunk_text, split_large_chunks

    try:
        tree = ast.parse(text)
        lines = text.splitlines()
        chunks = []

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = getattr(node, "end_lineno", node.lineno)
                chunk = "\n".join(lines[start:end]).strip()
                if chunk:
                    chunks.append(chunk)

        if chunks:
            return split_large_chunks(chunks, max_size=max_size)

        return fixed_chunk_text(text)
    except Exception:
        return fixed_chunk_text(text)


def chunk_terraform(text: str, max_size: int = 1800) -> list[str]:
    from app.text_chunker import split_large_chunks

    lines = text.splitlines()
    chunks = []
    current = []

    block_start = re.compile(r'^\s*(resource|module|variable|output|locals|data|provider|terraform)\b')

    for line in lines:
        if block_start.match(line) and current:
            chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        chunks.append("\n".join(current).strip())

    return split_large_chunks(chunks, max_size=max_size)


def chunk_generic_code(text: str) -> list[str]:
    from app.text_chunker import fixed_chunk_text
    return fixed_chunk_text(text)
