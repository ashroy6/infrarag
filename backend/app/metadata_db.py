from __future__ import annotations

import os
import sqlite3
import uuid
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
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    sources_json TEXT,
                    intent TEXT,
                    pipeline_used TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    audit_id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    intent TEXT,
                    pipeline_used TEXT,
                    pipeline_label TEXT,
                    confidence REAL,
                    router_reason TEXT,
                    sources_json TEXT,
                    model TEXT,
                    latency_ms INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
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

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
                ON conversations(updated_at)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id
                ON chat_messages(conversation_id)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at
                ON audit_logs(created_at)
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

        query += " ORDER BY last_ingested_at DESC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_latest_active_source(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM files
                WHERE status = 'active'
                ORDER BY last_ingested_at DESC
                LIMIT 1
                """
            ).fetchone()
            return dict(row) if row else None

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
                SELECT *
                FROM chunks
                WHERE source_id = ?
                ORDER BY chunk_index ASC
                """,
                (source_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def create_conversation(self, title: str = "New Chat") -> str:
        conversation_id = str(uuid.uuid4())
        clean_title = (title or "New Chat").strip()[:120] or "New Chat"

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (conversation_id, title)
                VALUES (?, ?)
                """,
                (conversation_id, clean_title),
            )

        return conversation_id

    def list_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 200))

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT conversation_id, title, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT conversation_id, title, created_at, updated_at
                FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            return dict(row) if row else None

    def update_conversation_title(self, conversation_id: str, title: str) -> None:
        clean_title = (title or "New Chat").strip()[:120] or "New Chat"

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id = ?
                """,
                (clean_title, conversation_id),
            )

    def delete_conversation(self, conversation_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )

    def add_chat_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources_json: str | None = None,
        intent: str | None = None,
        pipeline_used: str | None = None,
    ) -> str:
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Invalid chat message role: {role}")

        message_id = str(uuid.uuid4())

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (
                    message_id, conversation_id, role, content,
                    sources_json, intent, pipeline_used
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    sources_json,
                    intent,
                    pipeline_used,
                ),
            )
            conn.execute(
                """
                UPDATE conversations
                SET updated_at = CURRENT_TIMESTAMP
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            )

        return message_id

    def get_chat_messages(
        self,
        conversation_id: str,
        limit: int = 100,
        newest_first: bool = False,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        direction = "DESC" if newest_first else "ASC"

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    message_id,
                    conversation_id,
                    role,
                    content,
                    sources_json,
                    intent,
                    pipeline_used,
                    created_at
                FROM chat_messages
                WHERE conversation_id = ?
                ORDER BY created_at {direction}
                LIMIT ?
                """,
                (conversation_id, safe_limit),
            ).fetchall()

        items = [dict(row) for row in rows]
        if newest_first:
            items.reverse()
        return items

    def get_recent_chat_messages(
        self,
        conversation_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return self.get_chat_messages(
            conversation_id=conversation_id,
            limit=limit,
            newest_first=True,
        )

    def save_audit_log(
        self,
        question: str,
        answer: str,
        intent: str | None = None,
        pipeline_used: str | None = None,
        pipeline_label: str | None = None,
        confidence: float | None = None,
        router_reason: str | None = None,
        sources_json: str | None = None,
        model: str | None = None,
        latency_ms: int | None = None,
        conversation_id: str | None = None,
    ) -> str:
        audit_id = str(uuid.uuid4())

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs (
                    audit_id,
                    conversation_id,
                    question,
                    answer,
                    intent,
                    pipeline_used,
                    pipeline_label,
                    confidence,
                    router_reason,
                    sources_json,
                    model,
                    latency_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    conversation_id,
                    question,
                    answer,
                    intent,
                    pipeline_used,
                    pipeline_label,
                    confidence,
                    router_reason,
                    sources_json,
                    model,
                    latency_ms,
                ),
            )

        return audit_id

    def list_audit_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM audit_logs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()

        return [dict(row) for row in rows]
