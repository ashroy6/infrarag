from __future__ import annotations


def fixed_chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = max(end - overlap, 0)

    return chunks


def split_large_chunks(chunks: list[str], max_size: int = 1800) -> list[str]:
    final_chunks = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        if len(chunk) <= max_size:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(fixed_chunk_text(chunk))

    return final_chunks


def chunk_markdown(text: str, max_size: int = 1800) -> list[str]:
    import re

    lines = text.splitlines()
    chunks = []
    current = []

    for line in lines:
        if re.match(r"^\s{0,3}#{1,6}\s+", line) and current:
            chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        chunks.append("\n".join(current).strip())

    return split_large_chunks(chunks, max_size=max_size)


def chunk_yaml(text: str, max_size: int = 1800) -> list[str]:
    lines = text.splitlines()
    chunks = []
    current = []

    for line in lines:
        stripped = line.strip()
        is_top_level_key = (
            line
            and not line.startswith((" ", "\t"))
            and stripped.endswith(":")
            and not stripped.startswith("-")
        )

        if is_top_level_key and current:
            chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        chunks.append("\n".join(current).strip())

    return split_large_chunks(chunks, max_size=max_size)
