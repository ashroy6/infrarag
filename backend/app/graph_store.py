from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = os.getenv("METADATA_DB_PATH", "/app/data/infrarag.db")


class GraphStore:
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
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    source_id TEXT,
                    chunk_index INTEGER,
                    source_path TEXT,
                    metadata_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_edges (
                    edge_id TEXT PRIMARY KEY,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    source_id TEXT,
                    weight REAL DEFAULT 1.0,
                    metadata_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_nodes_type
                ON graph_nodes(node_type)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_nodes_source_id
                ON graph_nodes(source_id)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_nodes_label
                ON graph_nodes(label)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_edges_source
                ON graph_edges(source_node_id)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_edges_target
                ON graph_edges(target_node_id)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_edges_type
                ON graph_edges(edge_type)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_edges_source_id
                ON graph_edges(source_id)
                """
            )

    @staticmethod
    def _json_dump(value: Any) -> str:
        return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _json_load(value: str | None) -> dict[str, Any]:
        if not value:
            return {}
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def replace_source_graph(
        self,
        source_id: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> None:
        """
        Replaces all graph rows for one source_id.

        This is deliberately source-scoped so re-ingest is idempotent:
        same source -> old graph removed -> fresh graph inserted.
        """
        clean_nodes = []
        seen_nodes: set[str] = set()

        for node in nodes:
            node_id = str(node.get("node_id") or "").strip()
            node_type = str(node.get("node_type") or "").strip()
            label = str(node.get("label") or "").strip()

            if not node_id or not node_type or not label:
                continue

            if node_id in seen_nodes:
                continue

            seen_nodes.add(node_id)

            clean_nodes.append(
                (
                    node_id,
                    node_type,
                    label[:240],
                    node.get("source_id") or source_id,
                    node.get("chunk_index"),
                    node.get("source_path"),
                    self._json_dump(node.get("metadata")),
                )
            )

        clean_edges = []
        seen_edges: set[str] = set()

        for edge in edges:
            edge_id = str(edge.get("edge_id") or "").strip()
            source_node_id = str(edge.get("source_node_id") or "").strip()
            target_node_id = str(edge.get("target_node_id") or "").strip()
            edge_type = str(edge.get("edge_type") or "").strip()

            if not edge_id or not source_node_id or not target_node_id or not edge_type:
                continue

            if source_node_id not in seen_nodes or target_node_id not in seen_nodes:
                continue

            if edge_id in seen_edges:
                continue

            seen_edges.add(edge_id)

            clean_edges.append(
                (
                    edge_id,
                    source_node_id,
                    target_node_id,
                    edge_type,
                    edge.get("source_id") or source_id,
                    float(edge.get("weight") or 1.0),
                    self._json_dump(edge.get("metadata")),
                )
            )

        with self._connect() as conn:
            conn.execute("DELETE FROM graph_edges WHERE source_id = ?", (source_id,))
            conn.execute("DELETE FROM graph_nodes WHERE source_id = ?", (source_id,))

            if clean_nodes:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO graph_nodes (
                        node_id,
                        node_type,
                        label,
                        source_id,
                        chunk_index,
                        source_path,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    clean_nodes,
                )

            if clean_edges:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO graph_edges (
                        edge_id,
                        source_node_id,
                        target_node_id,
                        edge_type,
                        source_id,
                        weight,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    clean_edges,
                )

    def delete_graph_for_source(self, source_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM graph_edges WHERE source_id = ?", (source_id,))
            conn.execute("DELETE FROM graph_nodes WHERE source_id = ?", (source_id,))

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            node_count = conn.execute("SELECT COUNT(*) AS c FROM graph_nodes").fetchone()["c"]
            edge_count = conn.execute("SELECT COUNT(*) AS c FROM graph_edges").fetchone()["c"]
            source_count = conn.execute(
                "SELECT COUNT(DISTINCT source_id) AS c FROM graph_nodes WHERE source_id IS NOT NULL"
            ).fetchone()["c"]

            node_types = conn.execute(
                """
                SELECT node_type, COUNT(*) AS c
                FROM graph_nodes
                GROUP BY node_type
                ORDER BY c DESC
                """
            ).fetchall()

            edge_types = conn.execute(
                """
                SELECT edge_type, COUNT(*) AS c
                FROM graph_edges
                GROUP BY edge_type
                ORDER BY c DESC
                """
            ).fetchall()

        return {
            "nodes": node_count,
            "edges": edge_count,
            "sources": source_count,
            "node_types": [dict(row) for row in node_types],
            "edge_types": [dict(row) for row in edge_types],
        }

    def get_graph(
        self,
        mode: str = "structured",
        source_id: str | None = None,
        q: str | None = None,
        node_type: str | None = None,
        edge_type: str | None = None,
        limit: int = 600,
    ) -> dict[str, Any]:
        mode = mode if mode in {"structured", "dense"} else "structured"
        safe_limit = max(25, min(int(limit or 600), 2000))

        params: list[Any] = []
        where = ["1=1"]

        if source_id:
            where.append("source_id = ?")
            params.append(source_id)

        if node_type:
            where.append("node_type = ?")
            params.append(node_type)

        if q:
            like = f"%{q.strip()}%"
            where.append("(label LIKE ? OR source_path LIKE ? OR node_id LIKE ?)")
            params.extend([like, like, like])

        where_sql = " AND ".join(where)

        if q:
            # Search mode: include matched nodes plus direct neighbours.
            with self._connect() as conn:
                matched_rows = conn.execute(
                    f"""
                    SELECT *
                    FROM graph_nodes
                    WHERE {where_sql}
                    ORDER BY
                        CASE node_type
                            WHEN 'file' THEN 1
                            WHEN 'chapter' THEN 2
                            WHEN 'section' THEN 3
                            WHEN 'chunk' THEN 4
                            WHEN 'service' THEN 5
                            WHEN 'resource' THEN 6
                            WHEN 'concept' THEN 7
                            ELSE 8
                        END,
                        label ASC
                    LIMIT ?
                    """,
                    params + [safe_limit],
                ).fetchall()

                matched_ids = [row["node_id"] for row in matched_rows]

                if not matched_ids:
                    return {"mode": mode, "nodes": [], "edges": [], "stats": self.stats()}

                placeholders = ",".join("?" for _ in matched_ids)

                edge_where = f"(source_node_id IN ({placeholders}) OR target_node_id IN ({placeholders}))"
                edge_params = matched_ids + matched_ids

                if edge_type:
                    edge_where += " AND edge_type = ?"
                    edge_params.append(edge_type)

                edge_rows = conn.execute(
                    f"""
                    SELECT *
                    FROM graph_edges
                    WHERE {edge_where}
                    LIMIT ?
                    """,
                    edge_params + [safe_limit * 3],
                ).fetchall()

                neighbour_ids = set(matched_ids)
                for edge in edge_rows:
                    neighbour_ids.add(edge["source_node_id"])
                    neighbour_ids.add(edge["target_node_id"])

                node_placeholders = ",".join("?" for _ in neighbour_ids)
                node_rows = conn.execute(
                    f"""
                    SELECT *
                    FROM graph_nodes
                    WHERE node_id IN ({node_placeholders})
                    LIMIT ?
                    """,
                    list(neighbour_ids) + [safe_limit],
                ).fetchall()

            return self._format_graph(node_rows=node_rows, edge_rows=edge_rows, mode=mode)

        # Default mode.
        if mode == "structured":
            order_sql = """
                CASE node_type
                    WHEN 'file' THEN 1
                    WHEN 'chapter' THEN 2
                    WHEN 'section' THEN 3
                    WHEN 'chunk' THEN 4
                    WHEN 'service' THEN 5
                    WHEN 'resource' THEN 6
                    WHEN 'concept' THEN 7
                    ELSE 8
                END,
                source_path ASC,
                chunk_index ASC,
                label ASC
            """
            default_limit = min(safe_limit, 450)
        else:
            order_sql = """
                CASE node_type
                    WHEN 'service' THEN 1
                    WHEN 'resource' THEN 2
                    WHEN 'concept' THEN 3
                    WHEN 'file' THEN 4
                    WHEN 'chapter' THEN 5
                    WHEN 'section' THEN 6
                    WHEN 'chunk' THEN 7
                    ELSE 8
                END,
                label ASC
            """
            default_limit = min(safe_limit, 1200)

        with self._connect() as conn:
            node_rows = conn.execute(
                f"""
                SELECT *
                FROM graph_nodes
                WHERE {where_sql}
                ORDER BY {order_sql}
                LIMIT ?
                """,
                params + [default_limit],
            ).fetchall()

            node_ids = [row["node_id"] for row in node_rows]
            if not node_ids:
                return {"mode": mode, "nodes": [], "edges": [], "stats": self.stats()}

            placeholders = ",".join("?" for _ in node_ids)
            edge_where = f"source_node_id IN ({placeholders}) AND target_node_id IN ({placeholders})"
            edge_params = node_ids + node_ids

            if edge_type:
                edge_where += " AND edge_type = ?"
                edge_params.append(edge_type)

            edge_rows = conn.execute(
                f"""
                SELECT *
                FROM graph_edges
                WHERE {edge_where}
                LIMIT ?
                """,
                edge_params + [default_limit * 3],
            ).fetchall()

        return self._format_graph(node_rows=node_rows, edge_rows=edge_rows, mode=mode)


    def get_related_chunk_refs(
        self,
        seed_chunks: list[dict[str, Any]],
        allowed_edges: tuple[str, ...] = ("mentions", "defines", "contains"),
        max_refs: int = 9,
    ) -> list[dict[str, Any]]:
        """
        Finds chunk nodes related to seed chunk nodes through safe graph edges.

        Strategy:
        seed chunk -> connector node -> related chunk

        Connector can be concept/service/resource/symbol/section/file depending on edge type.
        Results are capped and deduped. This is used for graph-assisted RAG context.
        """
        safe_max_refs = max(1, min(int(max_refs or 9), 30))
        clean_edges = tuple(edge for edge in allowed_edges if edge)
        if not clean_edges:
            return []

        clean_seeds: list[tuple[str, int]] = []
        for seed in seed_chunks:
            source_id = str(seed.get("source_id") or "").strip()
            try:
                chunk_index = int(seed.get("chunk_index"))
            except (TypeError, ValueError):
                continue

            if source_id and chunk_index >= 0:
                clean_seeds.append((source_id, chunk_index))

        if not clean_seeds:
            return []

        with self._connect() as conn:
            seed_node_rows = []
            for source_id, chunk_index in clean_seeds:
                row = conn.execute(
                    """
                    SELECT node_id, source_id, chunk_index, label
                    FROM graph_nodes
                    WHERE source_id = ?
                      AND node_type = 'chunk'
                      AND chunk_index = ?
                    LIMIT 1
                    """,
                    (source_id, chunk_index),
                ).fetchone()
                if row:
                    seed_node_rows.append(row)

            if not seed_node_rows:
                return []

            seed_node_ids = [row["node_id"] for row in seed_node_rows]
            seed_node_id_set = set(seed_node_ids)

            edge_placeholders = ",".join("?" for _ in clean_edges)
            seed_placeholders = ",".join("?" for _ in seed_node_ids)

            first_hop_edges = conn.execute(
                f"""
                SELECT *
                FROM graph_edges
                WHERE edge_type IN ({edge_placeholders})
                  AND (
                    source_node_id IN ({seed_placeholders})
                    OR target_node_id IN ({seed_placeholders})
                  )
                LIMIT ?
                """,
                list(clean_edges) + seed_node_ids + seed_node_ids + [safe_max_refs * 10],
            ).fetchall()

            connector_ids: set[str] = set()
            connector_reason: dict[str, dict[str, Any]] = {}

            for edge in first_hop_edges:
                source_node_id = edge["source_node_id"]
                target_node_id = edge["target_node_id"]

                other_id = target_node_id if source_node_id in seed_node_id_set else source_node_id
                if other_id in seed_node_id_set:
                    continue

                connector_ids.add(other_id)
                connector_reason.setdefault(
                    other_id,
                    {
                        "edge_type": edge["edge_type"],
                        "weight": float(edge["weight"] or 1.0),
                    },
                )

            if not connector_ids:
                return []

            connector_placeholders = ",".join("?" for _ in connector_ids)
            connector_rows = conn.execute(
                f"""
                SELECT node_id, node_type, label
                FROM graph_nodes
                WHERE node_id IN ({connector_placeholders})
                """,
                list(connector_ids),
            ).fetchall()

            connector_labels = {
                row["node_id"]: {
                    "node_type": row["node_type"],
                    "label": row["label"],
                }
                for row in connector_rows
            }

            second_hop_edges = conn.execute(
                f"""
                SELECT *
                FROM graph_edges
                WHERE edge_type IN ({edge_placeholders})
                  AND (
                    source_node_id IN ({connector_placeholders})
                    OR target_node_id IN ({connector_placeholders})
                  )
                LIMIT ?
                """,
                list(clean_edges) + list(connector_ids) + list(connector_ids) + [safe_max_refs * 20],
            ).fetchall()

            candidate_chunk_ids: set[str] = set()
            candidate_meta: dict[str, dict[str, Any]] = {}

            for edge in second_hop_edges:
                source_node_id = edge["source_node_id"]
                target_node_id = edge["target_node_id"]

                if source_node_id in connector_ids:
                    candidate_id = target_node_id
                    connector_id = source_node_id
                elif target_node_id in connector_ids:
                    candidate_id = source_node_id
                    connector_id = target_node_id
                else:
                    continue

                if candidate_id in seed_node_id_set:
                    continue

                candidate_chunk_ids.add(candidate_id)

                connector_info = connector_labels.get(connector_id, {})
                candidate_meta.setdefault(
                    candidate_id,
                    {
                        "edge_type": edge["edge_type"],
                        "connector_id": connector_id,
                        "connector_label": connector_info.get("label", ""),
                        "connector_type": connector_info.get("node_type", ""),
                        "weight": float(edge["weight"] or 1.0),
                    },
                )

            if not candidate_chunk_ids:
                return []

            candidate_placeholders = ",".join("?" for _ in candidate_chunk_ids)
            candidate_rows = conn.execute(
                f"""
                SELECT node_id, source_id, chunk_index, label, source_path
                FROM graph_nodes
                WHERE node_id IN ({candidate_placeholders})
                  AND node_type = 'chunk'
                  AND source_id IS NOT NULL
                  AND chunk_index IS NOT NULL
                """,
                list(candidate_chunk_ids),
            ).fetchall()

        results: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()

        edge_rank = {
            "defines": 4,
            "mentions": 3,
            "contains": 2,
            "related_to": 1,
            "next": 0,
        }

        for row in candidate_rows:
            source_id = str(row["source_id"] or "")
            try:
                chunk_index = int(row["chunk_index"])
            except (TypeError, ValueError):
                continue

            key = (source_id, chunk_index)
            if key in seen:
                continue

            meta = candidate_meta.get(row["node_id"], {})
            edge_type = str(meta.get("edge_type") or "")
            connector_label = str(meta.get("connector_label") or "")

            seen.add(key)
            results.append(
                {
                    "source_id": source_id,
                    "chunk_index": chunk_index,
                    "source_path": row["source_path"],
                    "edge_type": edge_type,
                    "connector_label": connector_label,
                    "connector_type": meta.get("connector_type", ""),
                    "reason": (
                        f"Related through {edge_type}"
                        + (f" via {connector_label}" if connector_label else "")
                    ),
                    "rank": edge_rank.get(edge_type, 0) + float(meta.get("weight") or 1.0),
                }
            )

        results.sort(key=lambda item: item.get("rank", 0), reverse=True)
        return results[:safe_max_refs]

    def _format_graph(
        self,
        node_rows: list[sqlite3.Row],
        edge_rows: list[sqlite3.Row],
        mode: str,
    ) -> dict[str, Any]:
        nodes = []
        for row in node_rows:
            metadata = self._json_load(row["metadata_json"])
            nodes.append(
                {
                    "id": row["node_id"],
                    "label": row["label"],
                    "type": row["node_type"],
                    "source_id": row["source_id"],
                    "chunk_index": row["chunk_index"],
                    "source_path": row["source_path"],
                    "metadata": metadata,
                }
            )

        node_ids = {node["id"] for node in nodes}

        edges = []
        for row in edge_rows:
            if row["source_node_id"] not in node_ids or row["target_node_id"] not in node_ids:
                continue

            metadata = self._json_load(row["metadata_json"])
            edges.append(
                {
                    "id": row["edge_id"],
                    "source": row["source_node_id"],
                    "target": row["target_node_id"],
                    "type": row["edge_type"],
                    "source_id": row["source_id"],
                    "weight": row["weight"],
                    "metadata": metadata,
                }
            )

        return {
            "mode": mode,
            "nodes": nodes,
            "edges": edges,
            "stats": self.stats(),
        }
