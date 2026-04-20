from __future__ import annotations

import os
import sqlite3
from typing import Any

import requests
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.git_connector import clone_or_pull_repo
from app.metadata_db import MetadataDB
from app.pipeline import ingest_paths
from app.qdrant_client import delete_points_by_source_id
from app.retrieve import retrieve_context
from app.uploads import save_upload, save_uploads

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
METADATA_DB_PATH = os.getenv("METADATA_DB_PATH", "/app/data/infrarag.db")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90"))
MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.35"))

app = FastAPI(title="InfraRAG Backend", version="2.6.0")

REQUEST_COUNT = Counter("infrarag_requests_total", "Total API requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("infrarag_request_latency_seconds", "Request latency", ["endpoint"])


class InfraRAGError(Exception):
    def __init__(self, message: str, status_code: int = 500, code: str = "internal_error"):
        self.message = message
        self.status_code = status_code
        self.code = code
        super().__init__(message)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path

    with REQUEST_LATENCY.labels(endpoint=endpoint).time():
        try:
            response = await call_next(request)
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(response.status_code)).inc()
            return response
        except Exception:
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status="500").inc()
            raise


@app.exception_handler(InfraRAGError)
async def infrarag_error_handler(_: Request, exc: InfraRAGError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Request validation failed",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_server_error",
                "message": str(exc),
            }
        },
    )


def build_context_text(context_chunks: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        [
            f"[Source: {c['source']} | Chunk: {c['chunk_index']} | Score: {c.get('score', 0):.4f}]\n{c['text']}"
            for c in context_chunks
        ]
    )


def build_prompt(question: str, context_text: str) -> str:
    return f"""
You are InfraRAG, a private DevOps and cloud assistant.

Answer the user's question using only the retrieved context below.
Do not invent facts.
If the context does not support the answer, reply exactly:
No evidence found in the knowledge base.

Instructions:
- Write one clear technical paragraph in about 5 to 7 lines when enough evidence exists.
- Be specific and direct, not vague.
- Do not say "it appears", "it seems", or "probably" if the evidence is clear.
- If the question is about code, explain what the code does, where it does it, and what the outcome is.
- If the question is about a repo or project, explain:
  1. the main purpose,
  2. the key workflow or execution flow,
  3. the main tools or frameworks only if they are explicitly supported by the context.
- Prefer concrete workflow steps such as train, validate, predict, test, analyze when present in the context.
- Do not mention anything that is not supported by the retrieved context.

Format:
Answer:
<grounded technical answer>

Evidence:
- [Source: ... | Chunk: ...]
- [Source: ... | Chunk: ...]

Question:
{question}

Retrieved Context:
{context_text}
""".strip()


def get_sqlite_connection():
    conn = sqlite3.connect(METADATA_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ask_ollama(question: str, context_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if not context_chunks:
        return {
            "answer": "No evidence found in the knowledge base.",
            "citations": [],
        }

    top_score = max(c.get("score", 0) for c in context_chunks)
    if top_score < MIN_SCORE_THRESHOLD:
        return {
            "answer": "No evidence found in the knowledge base.",
            "citations": [],
        }

    context_text = build_context_text(context_chunks)
    prompt = build_prompt(question, context_text)

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 220,
                },
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
    except requests.Timeout as exc:
        raise InfraRAGError("Ollama request timed out", status_code=504, code="ollama_timeout") from exc
    except requests.RequestException as exc:
        raise InfraRAGError(f"Ollama request failed: {exc}", status_code=502, code="ollama_request_failed") from exc

    if response.status_code >= 400:
        body_preview = response.text[:500]
        raise InfraRAGError(
            f"Ollama returned HTTP {response.status_code}: {body_preview}",
            status_code=502,
            code="ollama_bad_response",
        )

    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        raise InfraRAGError(
            f"Ollama returned non-JSON content-type: {content_type or 'unknown'}",
            status_code=502,
            code="ollama_non_json_response",
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise InfraRAGError("Failed to parse Ollama JSON response", status_code=502, code="ollama_invalid_json") from exc

    citations = [
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "score": c["score"],
        }
        for c in context_chunks
    ]

    answer = (data.get("response") or "").strip()
    if not answer:
        answer = "No evidence found in the knowledge base."

    return {
        "answer": answer,
        "citations": citations,
    }


@app.get("/")
def root():
    return {"message": "InfraRAG backend is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/retrieve")
def retrieve(q: str = ""):
    if not q:
        raise InfraRAGError("query parameter 'q' is required", status_code=400, code="missing_query")

    hits = retrieve_context(q, limit=5)
    return {"question": q, "results": hits}


@app.get("/ask")
def ask(q: str = ""):
    if not q:
        raise InfraRAGError("query parameter 'q' is required", status_code=400, code="missing_query")

    context_chunks = retrieve_context(q, limit=5)
    result = ask_ollama(q, context_chunks)

    return {
        "question": q,
        "answer": result["answer"],
        "citations": result["citations"],
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    saved_path = await save_upload(file)
    result = ingest_paths([saved_path], source_type="upload")
    return {
        "message": "Upload processed",
        "saved_path": saved_path,
        "ingest_result": result,
    }


@app.post("/upload-folder")
async def upload_folder(
    files: list[UploadFile] = File(...),
    folder_name: str = Form("folder_upload"),
):
    base_dir, saved_paths = await save_uploads(files, folder_name=folder_name)
    result = ingest_paths([base_dir], source_type="upload")

    return {
        "message": "Folder upload processed",
        "folder_name": folder_name,
        "saved_base_dir": base_dir,
        "saved_file_count": len(saved_paths),
        "ingest_result": result,
    }


@app.post("/ingest/local")
def ingest_local(paths: list[str]):
    return ingest_paths(paths, source_type="local")


@app.post("/ingest/git")
def ingest_git(repo_url: str, branch: str | None = None):
    repo_path = clone_or_pull_repo(repo_url=repo_url, branch=branch)
    result = ingest_paths([repo_path], source_type="git")
    return {
        "repo_path": repo_path,
        "ingest_result": result,
    }


@app.get("/admin/stats")
def admin_stats():
    with get_sqlite_connection() as conn:
        total_files = conn.execute("SELECT COUNT(*) AS c FROM files").fetchone()["c"]
        active_files = conn.execute("SELECT COUNT(*) AS c FROM files WHERE status = 'active'").fetchone()["c"]
        deleted_files = conn.execute("SELECT COUNT(*) AS c FROM files WHERE status = 'deleted'").fetchone()["c"]
        total_chunks = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]

        uploads = conn.execute("SELECT COUNT(*) AS c FROM files WHERE source_type = 'upload'").fetchone()["c"]
        locals_count = conn.execute("SELECT COUNT(*) AS c FROM files WHERE source_type = 'local'").fetchone()["c"]
        git_count = conn.execute("SELECT COUNT(*) AS c FROM files WHERE source_type = 'git'").fetchone()["c"]

    return {
        "total_files": total_files,
        "active_files": active_files,
        "deleted_files": deleted_files,
        "total_chunks": total_chunks,
        "uploads": uploads,
        "local_files": locals_count,
        "git_files": git_count,
    }


@app.get("/admin/sources")
def admin_sources(
    source_type: str | None = None,
    status: str | None = None,
    limit: int = 200,
):
    query = """
        SELECT
            source_id,
            source_type,
            source_path,
            file_hash,
            file_type,
            parser_type,
            chunk_count,
            status,
            last_ingested_at,
            last_seen_at
        FROM files
        WHERE 1=1
    """
    params = []

    if source_type:
        query += " AND source_type = ?"
        params.append(source_type)

    if status:
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY last_ingested_at DESC LIMIT ?"
    params.append(limit)

    with get_sqlite_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        items = [dict(row) for row in rows]

    return {"sources": items}


@app.get("/admin/chunks")
def admin_chunks(source_id: str, limit: int = 200):
    if not source_id:
        raise InfraRAGError("source_id is required", status_code=400, code="missing_source_id")

    with get_sqlite_connection() as conn:
        file_row = conn.execute(
            """
            SELECT source_id, source_type, source_path, file_type, parser_type, chunk_count, status
            FROM files
            WHERE source_id = ?
            """,
            (source_id,),
        ).fetchone()

        if not file_row:
            raise InfraRAGError("source not found", status_code=404, code="source_not_found")

        chunk_rows = conn.execute(
            """
            SELECT chunk_id, source_id, chunk_index, qdrant_point_id, text_preview, created_at
            FROM chunks
            WHERE source_id = ?
            ORDER BY chunk_index ASC
            LIMIT ?
            """,
            (source_id, limit),
        ).fetchall()

        file_info = dict(file_row)
        chunks = [dict(row) for row in chunk_rows]

    return {
        "source": file_info,
        "chunks": chunks,
    }


@app.post("/admin/reingest")
def admin_reingest(payload: dict):
    source_path = (payload or {}).get("source_path")
    source_type = (payload or {}).get("source_type", "local")

    if not source_path:
        raise InfraRAGError("source_path is required", status_code=400, code="missing_source_path")

    result = ingest_paths([source_path], source_type=source_type)
    return {
        "message": "Re-ingest complete",
        "source_path": source_path,
        "source_type": source_type,
        "result": result,
    }


@app.delete("/admin/source")
def admin_delete_source(source_id: str):
    if not source_id:
        raise InfraRAGError("source_id is required", status_code=400, code="missing_source_id")

    metadata_db = MetadataDB()
    record = metadata_db.get_file(source_id)

    if not record:
        raise InfraRAGError("source not found", status_code=404, code="source_not_found")

    delete_points_by_source_id(source_id)
    metadata_db.delete_file_and_chunks(source_id)

    return {
        "message": "Source deleted",
        "source_id": source_id,
        "source_path": record["source_path"],
    }


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)