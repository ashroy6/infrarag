from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.pipeline import ingest_paths
from app.ingestion_progress import ensure_progress_columns, set_current_ingestion_job, update_ingestion_progress

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("METADATA_DB_PATH", "/app/data/infrarag.db")
MAX_INGEST_WORKERS = int(os.getenv("MAX_INGEST_WORKERS", "1"))

_executor = ThreadPoolExecutor(max_workers=MAX_INGEST_WORKERS)
_schema_lock = threading.Lock()
_schema_ready = False
_active_jobs_lock = threading.Lock()
_active_jobs: set[str] = set()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def ensure_ingestion_jobs_table() -> None:
    global _schema_ready

    if _schema_ready:
        return

    with _schema_lock:
        if _schema_ready:
            return

        with _connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    paths_json TEXT NOT NULL,
                    result_json TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status
                ON ingestion_jobs(status)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_created_at
                ON ingestion_jobs(created_at)
                """
            )

        ensure_progress_columns()

        _schema_ready = True


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_load(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def _same_paths(paths_a: list[str], paths_b: list[str]) -> bool:
    return sorted(str(p) for p in paths_a) == sorted(str(p) for p in paths_b)


def _find_existing_active_job(paths: list[str], source_type: str) -> dict[str, Any] | None:
    ensure_ingestion_jobs_table()
    clean_paths = [str(p) for p in paths if str(p).strip()]

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM ingestion_jobs
            WHERE source_type = ?
              AND status IN ('queued', 'running')
            ORDER BY created_at DESC
            """,
            (source_type,),
        ).fetchall()

    for row in rows:
        item = dict(row)
        existing_paths = _json_load(item.get("paths_json")) or []
        if _same_paths(existing_paths, clean_paths):
            return item

    return None


def create_ingestion_job(paths: list[str], source_type: str) -> str:
    ensure_ingestion_jobs_table()

    clean_paths = [str(p) for p in paths if str(p).strip()]
    if not clean_paths:
        raise ValueError("At least one path is required for ingestion")

    existing = _find_existing_active_job(clean_paths, source_type)
    if existing:
        return str(existing["job_id"])

    job_id = str(uuid.uuid4())
    now = _utc_now()

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO ingestion_jobs (
                job_id,
                status,
                source_type,
                paths_json,
                created_at,
                updated_at,
                stage,
                total_units,
                processed_units,
                progress_percent,
                heartbeat_at
            )
            VALUES (?, 'queued', ?, ?, ?, ?, 'queued', 0, 0, 0, ?)
            """,
            (
                job_id,
                source_type,
                _json_dump(clean_paths),
                now,
                now,
                now,
            ),
        )

    return job_id


def _mark_job_running(job_id: str) -> None:
    now = _utc_now()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE ingestion_jobs
            SET status = 'running',
                stage = 'starting',
                started_at = COALESCE(started_at, ?),
                heartbeat_at = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (now, now, now, job_id),
        )


def _mark_job_done(job_id: str, result: dict[str, Any]) -> None:
    now = _utc_now()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE ingestion_jobs
            SET status = 'done',
                stage = 'done',
                result_json = ?,
                error_message = NULL,
                progress_percent = 100,
                finished_at = ?,
                heartbeat_at = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (_json_dump(result), now, now, now, job_id),
        )


def _mark_job_failed(job_id: str, error_message: str) -> None:
    now = _utc_now()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE ingestion_jobs
            SET status = 'failed',
                stage = 'failed',
                error_message = ?,
                finished_at = ?,
                heartbeat_at = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (error_message[:4000], now, now, now, job_id),
        )


def _run_ingestion_job(job_id: str, paths: list[str], source_type: str) -> None:
    with _active_jobs_lock:
        if job_id in _active_jobs:
            return
        _active_jobs.add(job_id)

    logger.info("Ingestion job started: job_id=%s source_type=%s paths=%s", job_id, source_type, paths)

    try:
        set_current_ingestion_job(job_id)
        _mark_job_running(job_id)
        update_ingestion_progress(stage="starting", total_units=0, processed_units=0, progress_percent=1, job_id=job_id)
        result = ingest_paths(paths, source_type=source_type)
        _mark_job_done(job_id, result)
        logger.info("Ingestion job done: job_id=%s", job_id)
    except Exception as exc:
        logger.exception("Ingestion job failed: job_id=%s", job_id)
        _mark_job_failed(job_id, str(exc))
    finally:
        set_current_ingestion_job(None)
        with _active_jobs_lock:
            _active_jobs.discard(job_id)


def _submit_existing_job(job_id: str, paths: list[str], source_type: str) -> None:
    _executor.submit(_run_ingestion_job, job_id, paths, source_type)


def submit_ingestion_job(paths: list[str], source_type: str) -> str:
    job_id = create_ingestion_job(paths=paths, source_type=source_type)

    job = get_ingestion_job(job_id)
    if not job:
        raise ValueError(f"Failed to load created ingestion job: {job_id}")

    if job["status"] in {"queued", "running"}:
        _submit_existing_job(job_id, job["paths"], job["source_type"])

    return job_id


def recover_and_start_pending_jobs() -> dict[str, Any]:
    """
    Called on backend startup.

    Any job marked running before a backend restart is stale because the
    in-memory worker thread died. Requeue it, then start queued jobs.
    """
    ensure_ingestion_jobs_table()

    now = _utc_now()

    with _connect() as conn:
        stale_rows = conn.execute(
            """
            SELECT job_id
            FROM ingestion_jobs
            WHERE status = 'running'
            """
        ).fetchall()

        stale_count = len(stale_rows)

        conn.execute(
            """
            UPDATE ingestion_jobs
            SET status = 'queued',
                error_message = NULL,
                updated_at = ?
            WHERE status = 'running'
            """,
            (now,),
        )

        queued_rows = conn.execute(
            """
            SELECT *
            FROM ingestion_jobs
            WHERE status = 'queued'
            ORDER BY created_at ASC
            """
        ).fetchall()

    started = 0

    for row in queued_rows:
        item = dict(row)
        job_id = str(item["job_id"])
        source_type = str(item["source_type"])
        paths = _json_load(item.get("paths_json")) or []

        if not paths:
            _mark_job_failed(job_id, "No paths found for queued ingestion job")
            continue

        _submit_existing_job(job_id, paths, source_type)
        started += 1

    logger.info(
        "Ingestion job recovery complete: stale_requeued=%s queued_started=%s",
        stale_count,
        started,
    )

    return {
        "stale_requeued": stale_count,
        "queued_started": started,
    }


def get_ingestion_job(job_id: str) -> dict[str, Any] | None:
    ensure_ingestion_jobs_table()

    with _connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM ingestion_jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()

    if not row:
        return None

    item = dict(row)
    item["paths"] = _json_load(item.pop("paths_json", None)) or []
    item["result"] = _json_load(item.pop("result_json", None))
    return item


def list_ingestion_jobs(limit: int = 50) -> list[dict[str, Any]]:
    ensure_ingestion_jobs_table()

    safe_limit = max(1, min(int(limit), 200))

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM ingestion_jobs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    jobs: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["paths"] = _json_load(item.pop("paths_json", None)) or []
        item["result"] = _json_load(item.pop("result_json", None))
        jobs.append(item)

    return jobs


def cancel_ingestion_job(job_id: str) -> dict[str, Any]:
    ensure_ingestion_jobs_table()
    now = _utc_now()

    with _connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM ingestion_jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()

        if not row:
            return {"ok": False, "error": "job_not_found"}

        current_status = str(row["status"]).lower()

        if current_status in {"done", "failed", "cancelled"}:
            return {
                "ok": True,
                "job_id": job_id,
                "status": current_status,
                "message": f"Job already {current_status}",
            }

        conn.execute(
            """
            UPDATE ingestion_jobs
            SET status = 'cancelled',
                stage = 'cancelled',
                error_message = 'Cancelled by user',
                finished_at = ?,
                heartbeat_at = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (now, now, now, job_id),
        )

    return {
        "ok": True,
        "job_id": job_id,
        "status": "cancelled",
        "message": "Job cancellation requested",
    }


def retry_ingestion_job(job_id: str) -> dict[str, Any]:
    ensure_ingestion_jobs_table()

    old_job = get_ingestion_job(job_id)
    if not old_job:
        return {"ok": False, "error": "job_not_found"}

    old_status = str(old_job.get("status") or "").lower()
    if old_status not in {"failed", "cancelled"}:
        return {
            "ok": False,
            "error": "job_not_retryable",
            "message": f"Only failed/cancelled jobs can be retried. Current status: {old_status}",
        }

    new_job_id = submit_ingestion_job(
        paths=old_job.get("paths") or [],
        source_type=old_job.get("source_type") or "upload",
    )

    return {
        "ok": True,
        "old_job_id": job_id,
        "new_job_id": new_job_id,
        "status_url": f"/ingestion-jobs/{new_job_id}",
    }
