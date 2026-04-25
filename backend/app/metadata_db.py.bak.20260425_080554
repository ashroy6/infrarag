from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = os.getenv("METADATA_DB_PATH", "/app/data/infrarag.db")


class MetadataDB:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    source_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_type TEXT,
                    parser_type TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    last_ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_seen_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    qdrant_point_id TEXT NOT NULL,
                    text_preview TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(source_id) REFERENCES files(source_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_source_id
                ON chunks(source_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_files_status
                ON files(status)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_files_source_type
                ON files(source_type)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_files_file_hash
                ON files(file_hash)
                """
            )

    def upsert_file(
        self,
        source_id: str,
        source_type: str,
        source_path: str,
        file_hash: str,
        file_type: str,
        parser_type: str,
        chunk_count: int,
        status: str = "active",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO files (
                    source_id, source_type, source_path, file_hash, file_type,
                    parser_type, chunk_count, status, last_ingested_at, last_seen_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(source_id) DO UPDATE SET
                    source_type=excluded.source_type,
                    source_path=excluded.source_path,
                    file_hash=excluded.file_hash,
                    file_type=excluded.file_type,
                    parser_type=excluded.parser_type,
                    chunk_count=excluded.chunk_count,
                    status=excluded.status,
                    last_ingested_at=CURRENT_TIMESTAMP,
                    last_seen_at=CURRENT_TIMESTAMP
                """,
                (
                    source_id,
                    source_type,
                    source_path,
                    file_hash,
                    file_type,
                    parser_type,
                    chunk_count,
                    status,
                ),
            )

    def replace_chunks(
        self,
        source_id: str,
        chunk_records: list[dict[str, Any]],
    ) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, source_id, chunk_index, qdrant_point_id, text_preview
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        record["chunk_id"],
                        source_id,
                        record["chunk_index"],
                        record["qdrant_point_id"],
                        record["text_preview"],
                    )
                    for record in chunk_records
                ],
            )

    def get_file(self, source_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM files WHERE source_id = ?",
                (source_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_active_file_by_hash(
        self,
        file_hash: str,
        source_type: str | None = None,
    ) -> dict[str, Any] | None:
        query = "SELECT * FROM files WHERE file_hash = ? AND status = 'active'"
        params: list[Any] = [file_hash]

        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)

        query += " ORDER BY last_ingested_at DESC LIMIT 1"

        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
            return dict(row) if row else None

    def list_active_files(self, source_type: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM files WHERE status = 'active'"
        params: list[Any] = []

        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def mark_deleted(self, source_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE files
                SET status = 'deleted', last_seen_at = CURRENT_TIMESTAMP
                WHERE source_id = ?
                """,
                (source_id,),
            )

    def delete_file_and_chunks(self, source_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
            conn.execute("DELETE FROM files WHERE source_id = ?", (source_id,))

    def get_chunks_for_source(self, source_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM chunks
                WHERE source_id = ?
                ORDER BY chunk_index ASC
                """,
                (source_id,),
            ).fetchall()
            return [dict(row) for row in rows]
