import ast
import os
import re
import sys
import uuid
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

DOCS_ROOT = Path("/docs")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "infrarag_docs")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

MAX_CHUNK_SIZE = 1800
FALLBACK_CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

ALLOWED_EXTENSIONS = {
    ".md", ".txt", ".tf", ".py", ".yml", ".yaml", ".json", ".sh", ".cfg", ".ini"
}


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def fixed_chunk_text(text: str, chunk_size: int = FALLBACK_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = text.strip()
    if not text:
        return []

    chunks = []
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


def split_large_chunks(chunks, max_size: int = MAX_CHUNK_SIZE):
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


def chunk_markdown(text: str):
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

    return split_large_chunks(chunks)


def chunk_python(text: str):
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
            return split_large_chunks(chunks)

        return fixed_chunk_text(text)
    except Exception:
        return fixed_chunk_text(text)


def chunk_terraform(text: str):
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

    return split_large_chunks(chunks)


def chunk_yaml(text: str):
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

    return split_large_chunks(chunks)


def chunk_by_extension(path: Path, text: str):
    ext = path.suffix.lower()

    if ext == ".md":
        return chunk_markdown(text)
    if ext == ".py":
        return chunk_python(text)
    if ext == ".tf":
        return chunk_terraform(text)
    if ext in {".yml", ".yaml"}:
        return chunk_yaml(text)

    return fixed_chunk_text(text)


def get_embedding(text: str):
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={
            "model": EMBED_MODEL,
            "prompt": text
        },
        timeout=120
    )
    response.raise_for_status()
    data = response.json()
    return data["embedding"]


def ensure_collection(client: QdrantClient, vector_size: int):
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def resolve_cli_paths(root: Path, cli_args: list[str]):
    selected_files = []

    for rel in cli_args:
        rel = rel.strip().lstrip("/")
        full_path = root / rel

        if full_path.is_file():
            if full_path.suffix.lower() in ALLOWED_EXTENSIONS:
                selected_files.append(full_path)
            else:
                print(f"Skipping unsupported file type: {rel}")

        elif full_path.is_dir():
            for path in full_path.rglob("*"):
                if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
                    selected_files.append(path)

        else:
            print(f"WARNING: path not found, skipping: {rel}")

    seen = set()
    unique_files = []
    for path in selected_files:
        if path not in seen:
            seen.add(path)
            unique_files.append(path)

    return unique_files


def main():
    if len(sys.argv) < 2:
        print("Usage: python app/ingest.py <path1> <path2> ...")
        print("Paths must be relative to /docs")
        print("Example: python app/ingest.py repos/ml_data_pipeline/src terraform/module-s3 aws/eks/README.md")
        sys.exit(1)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)

    files = resolve_cli_paths(DOCS_ROOT, sys.argv[1:])
    if not files:
        print("No selected files found under /docs")
        return

    print(f"Selected {len(files)} files for ingestion")

    all_points = []
    first_vector_size = None

    for file_path in files:
        rel_path = str(file_path.relative_to(DOCS_ROOT))
        text = read_text_file(file_path)

        if not text.strip():
            continue

        chunks = chunk_by_extension(file_path, text)
        print(f"Ingesting {rel_path} -> {len(chunks)} chunks")

        for idx, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)

            if first_vector_size is None:
                first_vector_size = len(embedding)
                ensure_collection(client, first_vector_size)

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "source": rel_path,
                    "chunk_index": idx,
                    "text": chunk,
                    "file_type": file_path.suffix.lower(),
                }
            )
            all_points.append(point)

            if len(all_points) >= 64:
                client.upsert(collection_name=QDRANT_COLLECTION, points=all_points)
                all_points = []

    if all_points:
        client.upsert(collection_name=QDRANT_COLLECTION, points=all_points)

    print("Ingestion complete")


if __name__ == "__main__":
    main()
