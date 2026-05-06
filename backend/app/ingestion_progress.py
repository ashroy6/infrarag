from __future__ import annotations

import os
import sqlite3
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = os.getenv("METADATA_DB_PATH", "/app/data/infrarag.db")

_current_job_id: ContextVar[str | None] = ContextVar("current_ingestion_job_id", default=None)


def set_current_ingestion_job(job_id: str | None) -> None:
    _current_job_id.set(job_id)


def get_current_ingestion_job() -> str | None:
    return _current_job_id.get()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def ensure_progress_columns() -> None:
    with _connect() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(ingestion_jobs)").fetchall()}

        if "stage" not in columns:
            conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN stage TEXT DEFAULT 'queued'")
        if "total_units" not in columns:
            conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN total_units INTEGER DEFAULT 0")
        if "processed_units" not in columns:
            conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN processed_units INTEGER DEFAULT 0")
        if "progress_percent" not in columns:
            conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN progress_percent REAL DEFAULT 0")
        if "heartbeat_at" not in columns:
            conn.execute("ALTER TABLE ingestion_jobs ADD COLUMN heartbeat_at TEXT")


def update_ingestion_progress(
    *,
    stage: str,
    total_units: int | None = None,
    processed_units: int | None = None,
    progress_percent: float | None = None,
    job_id: str | None = None,
) -> None:
    active_job_id = job_id or get_current_ingestion_job()
    if not active_job_id:
        return

    ensure_progress_columns()

    updates: list[str] = [
        "stage = ?",
        "heartbeat_at = ?",
        "updated_at = ?",
    ]
    params: list[Any] = [stage, _utc_now(), _utc_now()]

    if total_units is not None:
        updates.append("total_units = ?")
        params.append(max(0, int(total_units)))

    if processed_units is not None:
        updates.append("processed_units = ?")
        params.append(max(0, int(processed_units)))

    if progress_percent is not None:
        clean_percent = max(0.0, min(100.0, float(progress_percent)))
        updates.append("progress_percent = ?")
        params.append(clean_percent)

    params.append(active_job_id)

    with _connect() as conn:
        conn.execute(
            f"""
            UPDATE ingestion_jobs
            SET {", ".join(updates)}
            WHERE job_id = ?
            """,
            params,
        )


def is_ingestion_cancelled(job_id: str | None = None) -> bool:
    active_job_id = job_id or get_current_ingestion_job()
    if not active_job_id:
        return False

    ensure_progress_columns()

    with _connect() as conn:
        row = conn.execute(
            """
            SELECT status
            FROM ingestion_jobs
            WHERE job_id = ?
            """,
            (active_job_id,),
        ).fetchone()

    if not row:
        return False

    return str(row["status"]).lower() in {"cancelled", "failed"}
