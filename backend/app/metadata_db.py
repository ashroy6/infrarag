from __future__ import annotations

import json
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

    @staticmethod
    def _json_dump(value: Any) -> str:
        return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _json_load(value: str | None) -> Any:
        if not value:
            return None
        try:
            return json.loads(value)
        except Exception:
            return None

    @staticmethod
    def _tags_dump(tags: list[str] | None) -> str:
        clean_tags = []
        for tag in tags or []:
            value = str(tag or "").strip()
            if value and value not in clean_tags:
                clean_tags.append(value)
        return json.dumps(clean_tags, ensure_ascii=False)

    def _table_columns(self, conn: sqlite3.Connection, table_name: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(row["name"]) for row in rows}

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_sql: str,
    ) -> None:
        columns = self._table_columns(conn, table_name)
        if column_name not in columns:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")

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

            # Backward-compatible Agent Studio metadata on existing file records.
            # Existing regular chat files remain valid and untouched.
            self._ensure_column(conn, "files", "tenant_id", "tenant_id TEXT DEFAULT 'local'")
            self._ensure_column(conn, "files", "owner_user_id", "owner_user_id TEXT DEFAULT 'ashish'")
            self._ensure_column(conn, "files", "source_group", "source_group TEXT DEFAULT 'regular_chat'")
            self._ensure_column(conn, "files", "connector", "connector TEXT DEFAULT 'file_upload'")
            self._ensure_column(conn, "files", "data_domain", "data_domain TEXT DEFAULT 'general'")
            self._ensure_column(conn, "files", "security_level", "security_level TEXT DEFAULT 'internal'")
            self._ensure_column(conn, "files", "tags_json", "tags_json TEXT DEFAULT '[]'")
            self._ensure_column(conn, "files", "metadata_json", "metadata_json TEXT DEFAULT '{}'")
            self._ensure_column(conn, "files", "agent_access_enabled", "agent_access_enabled INTEGER DEFAULT 0")
            self._ensure_column(conn, "files", "knowledge_source_id", "knowledge_source_id TEXT")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_sources (
                    knowledge_source_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'local',
                    source_name TEXT NOT NULL,
                    source_description TEXT,
                    connector TEXT NOT NULL DEFAULT 'file_upload',
                    source_group TEXT NOT NULL DEFAULT 'agent_studio',
                    data_domain TEXT NOT NULL DEFAULT 'general',
                    security_level TEXT NOT NULL DEFAULT 'internal',
                    owner_user_id TEXT NOT NULL DEFAULT 'ashish',
                    tags_json TEXT DEFAULT '[]',
                    metadata_json TEXT DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'local',
                    agent_name TEXT NOT NULL,
                    agent_description TEXT,
                    agent_type TEXT NOT NULL DEFAULT 'knowledge_agent',
                    instructions TEXT,
                    status TEXT NOT NULL DEFAULT 'draft',
                    created_by TEXT NOT NULL DEFAULT 'ashish',
                    metadata_json TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_sources (
                    agent_source_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'local',
                    agent_id TEXT NOT NULL,
                    knowledge_source_id TEXT NOT NULL,
                    access_level TEXT NOT NULL DEFAULT 'read',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(agent_id, knowledge_source_id),
                    FOREIGN KEY(agent_id) REFERENCES agents(agent_id) ON DELETE CASCADE,
                    FOREIGN KEY(knowledge_source_id) REFERENCES knowledge_sources(knowledge_source_id) ON DELETE CASCADE
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    agent_run_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL DEFAULT 'local',
                    agent_id TEXT NOT NULL,
                    conversation_id TEXT,
                    user_id TEXT NOT NULL DEFAULT 'ashish',
                    user_task TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'created',
                    answer TEXT,
                    sources_json TEXT,
                    tools_json TEXT DEFAULT '[]',
                    approval_status TEXT DEFAULT 'not_required',
                    latency_ms INTEGER,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(agent_id) REFERENCES agents(agent_id) ON DELETE CASCADE
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_run_events (
                    event_id TEXT PRIMARY KEY,
                    agent_run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_status TEXT NOT NULL DEFAULT 'info',
                    message TEXT NOT NULL,
                    payload_json TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(agent_run_id) REFERENCES agent_runs(agent_run_id) ON DELETE CASCADE
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

            self._ensure_column(conn, "chat_messages", "metadata_json", "metadata_json TEXT DEFAULT '{}'")

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

            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source_id ON chunks(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_status ON files(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_source_type ON files(source_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_file_hash ON files(file_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_source_group ON files(source_group)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_data_domain ON files(data_domain)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_agent_access ON files(agent_access_enabled)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_knowledge_source_id ON files(knowledge_source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_tenant_domain ON files(tenant_id, data_domain, source_group)")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_sources_tenant ON knowledge_sources(tenant_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_sources_domain ON knowledge_sources(data_domain)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_sources_status ON knowledge_sources(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_sources_group ON knowledge_sources(source_group)")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_agents_tenant ON agents(tenant_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_sources_agent_id ON agent_sources(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_sources_ks_id ON agent_sources(knowledge_source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_runs_agent_id ON agent_runs(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_run_events_run_id ON agent_run_events(agent_run_id)")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON chat_messages(conversation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at)")

    # ---------------------------------------------------------------------
    # File/source/chunk methods
    # ---------------------------------------------------------------------

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
        tenant_id: str = "local",
        owner_user_id: str = "ashish",
        source_group: str = "regular_chat",
        connector: str | None = None,
        data_domain: str = "general",
        security_level: str = "internal",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        agent_access_enabled: bool = False,
        knowledge_source_id: str | None = None,
    ) -> None:
        clean_connector = connector or source_type or "file_upload"

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO files (
                    source_id,
                    source_type,
                    source_path,
                    file_hash,
                    file_type,
                    parser_type,
                    chunk_count,
                    status,
                    last_ingested_at,
                    last_seen_at,
                    tenant_id,
                    owner_user_id,
                    source_group,
                    connector,
                    data_domain,
                    security_level,
                    tags_json,
                    metadata_json,
                    agent_access_enabled,
                    knowledge_source_id
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(source_id) DO UPDATE SET
                    source_type=excluded.source_type,
                    source_path=excluded.source_path,
                    file_hash=excluded.file_hash,
                    file_type=excluded.file_type,
                    parser_type=excluded.parser_type,
                    chunk_count=excluded.chunk_count,
                    status=excluded.status,
                    last_ingested_at=CURRENT_TIMESTAMP,
                    last_seen_at=CURRENT_TIMESTAMP,
                    tenant_id=excluded.tenant_id,
                    owner_user_id=excluded.owner_user_id,
                    source_group=excluded.source_group,
                    connector=excluded.connector,
                    data_domain=excluded.data_domain,
                    security_level=excluded.security_level,
                    tags_json=excluded.tags_json,
                    metadata_json=excluded.metadata_json,
                    agent_access_enabled=excluded.agent_access_enabled,
                    knowledge_source_id=excluded.knowledge_source_id
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
                    tenant_id,
                    owner_user_id,
                    source_group,
                    clean_connector,
                    data_domain,
                    security_level,
                    self._tags_dump(tags),
                    self._json_dump(metadata),
                    1 if agent_access_enabled else 0,
                    knowledge_source_id,
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
            row = conn.execute("SELECT * FROM files WHERE source_id = ?", (source_id,)).fetchone()
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

    def list_active_files(
        self,
        source_type: str | None = None,
        source_group: str | None = None,
        data_domain: str | None = None,
        agent_access_enabled: bool | None = None,
        knowledge_source_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM files WHERE status = 'active'"
        params: list[Any] = []

        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)

        if source_group:
            query += " AND source_group = ?"
            params.append(source_group)

        if data_domain:
            query += " AND data_domain = ?"
            params.append(data_domain)

        if agent_access_enabled is not None:
            query += " AND agent_access_enabled = ?"
            params.append(1 if agent_access_enabled else 0)

        if knowledge_source_id:
            query += " AND knowledge_source_id = ?"
            params.append(knowledge_source_id)

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

    # ---------------------------------------------------------------------
    # Knowledge source methods
    # ---------------------------------------------------------------------

    def create_knowledge_source(
        self,
        source_name: str,
        source_description: str | None = None,
        tenant_id: str = "local",
        connector: str = "file_upload",
        source_group: str = "agent_studio",
        data_domain: str = "general",
        security_level: str = "internal",
        owner_user_id: str = "ashish",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        status: str = "active",
        knowledge_source_id: str | None = None,
    ) -> str:
        ks_id = knowledge_source_id or str(uuid.uuid4())
        clean_name = (source_name or "Knowledge Source").strip()[:200] or "Knowledge Source"

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_sources (
                    knowledge_source_id,
                    tenant_id,
                    source_name,
                    source_description,
                    connector,
                    source_group,
                    data_domain,
                    security_level,
                    owner_user_id,
                    tags_json,
                    metadata_json,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(knowledge_source_id) DO UPDATE SET
                    tenant_id=excluded.tenant_id,
                    source_name=excluded.source_name,
                    source_description=excluded.source_description,
                    connector=excluded.connector,
                    source_group=excluded.source_group,
                    data_domain=excluded.data_domain,
                    security_level=excluded.security_level,
                    owner_user_id=excluded.owner_user_id,
                    tags_json=excluded.tags_json,
                    metadata_json=excluded.metadata_json,
                    status=excluded.status,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    ks_id,
                    tenant_id,
                    clean_name,
                    source_description,
                    connector,
                    source_group,
                    data_domain,
                    security_level,
                    owner_user_id,
                    self._tags_dump(tags),
                    self._json_dump(metadata),
                    status,
                ),
            )

        return ks_id

    def get_knowledge_source(self, knowledge_source_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM knowledge_sources
                WHERE knowledge_source_id = ?
                """,
                (knowledge_source_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_knowledge_sources(
        self,
        tenant_id: str = "local",
        source_group: str | None = None,
        data_domain: str | None = None,
        status: str | None = "active",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        query = "SELECT * FROM knowledge_sources WHERE tenant_id = ?"
        params: list[Any] = [tenant_id]

        if source_group:
            query += " AND source_group = ?"
            params.append(source_group)

        if data_domain:
            query += " AND data_domain = ?"
            params.append(data_domain)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(safe_limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def set_knowledge_source_status(self, knowledge_source_id: str, status: str) -> None:
        clean_status = (status or "active").strip().lower()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE knowledge_sources
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE knowledge_source_id = ?
                """,
                (clean_status, knowledge_source_id),
            )

    # ---------------------------------------------------------------------
    # Agent methods
    # ---------------------------------------------------------------------

    def create_agent(
        self,
        agent_name: str,
        agent_description: str | None = None,
        agent_type: str = "knowledge_agent",
        instructions: str | None = None,
        tenant_id: str = "local",
        created_by: str = "ashish",
        metadata: dict[str, Any] | None = None,
        status: str = "draft",
        agent_id: str | None = None,
    ) -> str:
        clean_agent_id = agent_id or str(uuid.uuid4())
        clean_name = (agent_name or "Untitled Agent").strip()[:160] or "Untitled Agent"

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agents (
                    agent_id,
                    tenant_id,
                    agent_name,
                    agent_description,
                    agent_type,
                    instructions,
                    status,
                    created_by,
                    metadata_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(agent_id) DO UPDATE SET
                    tenant_id=excluded.tenant_id,
                    agent_name=excluded.agent_name,
                    agent_description=excluded.agent_description,
                    agent_type=excluded.agent_type,
                    instructions=excluded.instructions,
                    status=excluded.status,
                    created_by=excluded.created_by,
                    metadata_json=excluded.metadata_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    clean_agent_id,
                    tenant_id,
                    clean_name,
                    agent_description,
                    agent_type,
                    instructions,
                    status,
                    created_by,
                    self._json_dump(metadata),
                ),
            )

        return clean_agent_id

    def get_agent(self, agent_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM agents
                WHERE agent_id = ?
                """,
                (agent_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_agents(
        self,
        tenant_id: str = "local",
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        query = "SELECT * FROM agents WHERE tenant_id = ?"
        params: list[Any] = [tenant_id]

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(safe_limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def set_agent_status(self, agent_id: str, status: str) -> None:
        clean_status = (status or "draft").strip().lower()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE agents
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE agent_id = ?
                """,
                (clean_status, agent_id),
            )

    def assign_knowledge_source_to_agent(
        self,
        agent_id: str,
        knowledge_source_id: str,
        tenant_id: str = "local",
        access_level: str = "read",
        status: str = "active",
    ) -> str:
        agent_source_id = str(uuid.uuid4())

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_sources (
                    agent_source_id,
                    tenant_id,
                    agent_id,
                    knowledge_source_id,
                    access_level,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(agent_id, knowledge_source_id) DO UPDATE SET
                    access_level=excluded.access_level,
                    status=excluded.status,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    agent_source_id,
                    tenant_id,
                    agent_id,
                    knowledge_source_id,
                    access_level,
                    status,
                ),
            )

        return agent_source_id

    def list_agent_sources(
        self,
        agent_id: str,
        status: str | None = "active",
    ) -> list[dict[str, Any]]:
        query = """
            SELECT
                agent_sources.*,
                knowledge_sources.source_name,
                knowledge_sources.connector,
                knowledge_sources.data_domain,
                knowledge_sources.security_level,
                knowledge_sources.source_group
            FROM agent_sources
            JOIN knowledge_sources
              ON knowledge_sources.knowledge_source_id = agent_sources.knowledge_source_id
            WHERE agent_sources.agent_id = ?
        """
        params: list[Any] = [agent_id]

        if status:
            query += " AND agent_sources.status = ?"
            params.append(status)

        query += " ORDER BY knowledge_sources.source_name ASC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def list_agent_allowed_source_ids(self, agent_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT files.source_id
                FROM agent_sources
                JOIN knowledge_sources
                  ON knowledge_sources.knowledge_source_id = agent_sources.knowledge_source_id
                JOIN files
                  ON files.knowledge_source_id = knowledge_sources.knowledge_source_id
                WHERE agent_sources.agent_id = ?
                  AND agent_sources.status = 'active'
                  AND knowledge_sources.status = 'active'
                  AND files.status = 'active'
                  AND files.agent_access_enabled = 1
                ORDER BY files.source_path ASC
                """,
                (agent_id,),
            ).fetchall()

        return [str(row["source_id"]) for row in rows]

    # ---------------------------------------------------------------------
    # Agent run methods
    # ---------------------------------------------------------------------

    def create_agent_run(
        self,
        agent_id: str,
        user_task: str,
        tenant_id: str = "local",
        user_id: str = "ashish",
        conversation_id: str | None = None,
        status: str = "created",
    ) -> str:
        agent_run_id = str(uuid.uuid4())

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_runs (
                    agent_run_id,
                    tenant_id,
                    agent_id,
                    conversation_id,
                    user_id,
                    user_task,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    agent_run_id,
                    tenant_id,
                    agent_id,
                    conversation_id,
                    user_id,
                    user_task,
                    status,
                ),
            )

        return agent_run_id

    def add_agent_run_event(
        self,
        agent_run_id: str,
        event_type: str,
        message: str,
        event_status: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> str:
        event_id = str(uuid.uuid4())

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_run_events (
                    event_id,
                    agent_run_id,
                    event_type,
                    event_status,
                    message,
                    payload_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    event_id,
                    agent_run_id,
                    event_type,
                    event_status,
                    message,
                    self._json_dump(payload),
                ),
            )
            conn.execute(
                """
                UPDATE agent_runs
                SET updated_at = CURRENT_TIMESTAMP
                WHERE agent_run_id = ?
                """,
                (agent_run_id,),
            )

        return event_id

    def update_agent_run(
        self,
        agent_run_id: str,
        status: str,
        answer: str | None = None,
        sources_json: str | None = None,
        tools_json: str | None = None,
        approval_status: str | None = None,
        latency_ms: int | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE agent_runs
                SET
                    status = ?,
                    answer = COALESCE(?, answer),
                    sources_json = COALESCE(?, sources_json),
                    tools_json = COALESCE(?, tools_json),
                    approval_status = COALESCE(?, approval_status),
                    latency_ms = COALESCE(?, latency_ms),
                    error_message = COALESCE(?, error_message),
                    updated_at = CURRENT_TIMESTAMP
                WHERE agent_run_id = ?
                """,
                (
                    status,
                    answer,
                    sources_json,
                    tools_json,
                    approval_status,
                    latency_ms,
                    error_message,
                    agent_run_id,
                ),
            )

    def list_agent_runs(
        self,
        agent_id: str | None = None,
        tenant_id: str = "local",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        query = "SELECT * FROM agent_runs WHERE tenant_id = ?"
        params: list[Any] = [tenant_id]

        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(safe_limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def list_agent_run_events(self, agent_run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM agent_run_events
                WHERE agent_run_id = ?
                ORDER BY created_at ASC
                """,
                (agent_run_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    # ---------------------------------------------------------------------
    # Conversation methods
    # ---------------------------------------------------------------------

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
        metadata_json: str | None = None,
    ) -> str:
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Invalid chat message role: {role}")

        message_id = str(uuid.uuid4())

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (
                    message_id, conversation_id, role, content,
                    sources_json, intent, pipeline_used, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    sources_json,
                    intent,
                    pipeline_used,
                    metadata_json or "{}",
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
                    metadata_json,
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

    # ---------------------------------------------------------------------
    # Audit methods
    # ---------------------------------------------------------------------

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
