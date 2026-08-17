"""Text, vector-generation, settings, and diagnostic persistence for Recall."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from ....storage.sqlite.codec import _dt, _now


class RecallIndexRepositoryMixin:
    async def fts_search(
        self, profile_id: str, instance_id: str, query: str, *, limit: int = 80
    ) -> list[dict[str, Any]]:
        if not str(query or "").strip():
            return []
        rows = await self.db.fetch_all(
            """SELECT document.*, bm25(recall_documents_fts) AS bm25_score
            FROM recall_documents_fts
            JOIN recall_documents AS document
              ON document.document_key = recall_documents_fts.document_key
            WHERE recall_documents_fts MATCH ?
              AND recall_documents_fts.profile_id = ?
              AND recall_documents_fts.instance_id = ?
            ORDER BY bm25_score LIMIT ?""",
            (query, profile_id, instance_id, max(1, min(int(limit), 200))),
        )
        return [self._document(row) for row in rows]

    async def list_documents(self, profile_id: str, instance_id: str) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            "SELECT * FROM recall_documents WHERE profile_id = ? AND instance_id = ? "
            "ORDER BY document_key",
            (profile_id, instance_id),
        )
        return [self._document(row) for row in rows]

    async def documents_by_keys(
        self, profile_id: str, instance_id: str, keys: Sequence[str]
    ) -> list[dict[str, Any]]:
        normalized = tuple(dict.fromkeys(str(key) for key in keys if str(key)))
        if not normalized:
            return []
        result: list[dict[str, Any]] = []
        for offset in range(0, len(normalized), 400):
            batch = normalized[offset : offset + 400]
            placeholders = ",".join("?" for _ in batch)
            rows = await self.db.fetch_all(
                f"SELECT * FROM recall_documents WHERE profile_id = ? AND instance_id = ? "
                f"AND document_key IN ({placeholders})",
                (profile_id, instance_id, *batch),
            )
            result.extend(self._document(row) for row in rows)
        by_key = {item["document_key"]: item for item in result}
        return [by_key[key] for key in normalized if key in by_key]

    async def list_edges(self, profile_id: str, instance_id: str) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT source_document_key, target_document_key, edge_type, weight,
               valid_from, valid_until, evidence_json
            FROM recall_edges WHERE profile_id = ? AND instance_id = ? AND status = 'ACTIVE'
            ORDER BY edge_id""",
            (profile_id, instance_id),
        )
        return [self._record(row, json_columns=("evidence_json",)) for row in rows]

    async def list_graph(
        self, profile_id: str, instance_id: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        node_rows = await self.db.fetch_all(
            """SELECT node_key, node_type, stable_ref, label, document_key, scene_key,
               valid_from, valid_until, evidence_json
            FROM recall_graph_nodes WHERE profile_id = ? AND instance_id = ?
            ORDER BY node_key""",
            (profile_id, instance_id),
        )
        edge_rows = await self.db.fetch_all(
            """SELECT source_node_key, target_node_key, edge_type, weight,
               evidence_document_key, valid_from, valid_until, evidence_json
            FROM recall_graph_edges
            WHERE profile_id = ? AND instance_id = ? AND status = 'ACTIVE'
            ORDER BY edge_id""",
            (profile_id, instance_id),
        )
        return (
            [self._record(row, json_columns=("evidence_json",)) for row in node_rows],
            [self._record(row, json_columns=("evidence_json",)) for row in edge_rows],
        )

    async def list_scenes(self, profile_id: str, instance_id: str) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT scene.*, GROUP_CONCAT(member.document_key, char(31)) AS member_keys
            FROM recall_scenes AS scene
            LEFT JOIN recall_scene_members AS member ON member.scene_key = scene.scene_key
            WHERE scene.profile_id = ? AND scene.instance_id = ?
            GROUP BY scene.scene_key ORDER BY scene.scene_level, scene.scene_key""",
            (profile_id, instance_id),
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            item = self._record(row, json_columns=("evidence_json",))
            item["member_keys"] = tuple(
                value for value in str(item.get("member_keys") or "").split(chr(31)) if value
            )
            result.append(item)
        return result

    async def get_role_settings(self, profile_id: str) -> dict[str, Any]:
        row = await self.db.fetch_one(
            "SELECT * FROM recall_role_settings WHERE profile_id = ?", (profile_id,)
        )
        if row is None:
            return {
                "profile_id": profile_id,
                "embedding_provider_id": None,
                "rerank_provider_id": None,
                "version": 0,
            }
        return self._record(row)

    async def save_role_settings(
        self,
        profile_id: str,
        *,
        embedding_provider_id: str | None,
        rerank_provider_id: str | None,
        expected_version: int,
    ) -> dict[str, Any]:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> dict[str, Any]:
            current = conn.execute(
                "SELECT version FROM recall_role_settings WHERE profile_id = ?", (profile_id,)
            ).fetchone()
            actual = int(current[0]) if current else 0
            if actual != int(expected_version):
                raise ValueError("recall role settings version conflict")
            next_version = actual + 1
            conn.execute(
                """INSERT INTO recall_role_settings(
                    profile_id, embedding_provider_id, rerank_provider_id, version, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(profile_id) DO UPDATE SET
                    embedding_provider_id = excluded.embedding_provider_id,
                    rerank_provider_id = excluded.rerank_provider_id,
                    version = excluded.version, updated_at = excluded.updated_at""",
                (
                    profile_id,
                    self._nullable_provider(embedding_provider_id),
                    self._nullable_provider(rerank_provider_id),
                    next_version,
                    now,
                ),
            )
            self._enqueue_settings_rebuilds(conn, profile_id, next_version, now)
            row = conn.execute(
                "SELECT * FROM recall_role_settings WHERE profile_id = ?", (profile_id,)
            ).fetchone()
            assert row is not None
            return self._record(row)

        return await self.uow.run(operation)

    @staticmethod
    def _enqueue_settings_rebuilds(
        conn: sqlite3.Connection, profile_id: str, version: int, now: str
    ) -> None:
        instances = conn.execute(
            "SELECT instance_id FROM character_instances WHERE profile_id = ?",
            (profile_id,),
        ).fetchall()
        for row in instances:
            instance_id = str(row[0])
            conn.execute(
                """INSERT OR IGNORE INTO recall_index_outbox(
                    task_key, profile_id, instance_id, source_type, source_key,
                    operation, source_version, status, not_before, created_at, updated_at
                ) VALUES (?, ?, ?, 'SCOPE', ?, 'REBUILD', ?, 'PENDING', ?, ?, ?)""",
                (
                    f"scope:{profile_id}:{instance_id}:settings:{version}",
                    profile_id,
                    instance_id,
                    instance_id,
                    version,
                    now,
                    now,
                    now,
                ),
            )

    async def enqueue_rebuild(self, profile_id: str, instance_id: str) -> None:
        now = _dt(_now())
        key = f"scope:{profile_id}:{instance_id}:manual:{int(datetime.now(UTC).timestamp() * 1000)}"
        await self.uow.run(
            lambda conn: conn.execute(
                """INSERT INTO recall_index_outbox(
                    task_key, profile_id, instance_id, source_type, source_key,
                    operation, source_version, status, not_before, created_at, updated_at
                ) VALUES (?, ?, ?, 'SCOPE', ?, 'REBUILD', 0, 'PENDING', ?, ?, ?)""",
                (key, profile_id, instance_id, instance_id, now, now, now),
            )
        )

    async def ensure_generation(
        self,
        profile_id: str,
        instance_id: str,
        *,
        provider_id: str,
        provider_fingerprint: str,
        dimension: int,
    ) -> dict[str, Any]:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> dict[str, Any]:
            row = conn.execute(
                """SELECT * FROM recall_index_generations
                WHERE profile_id = ? AND instance_id = ? AND provider_fingerprint = ?""",
                (profile_id, instance_id, provider_fingerprint),
            ).fetchone()
            if row is None:
                row = self._insert_generation(
                    conn,
                    profile_id,
                    instance_id,
                    provider_id=provider_id,
                    provider_fingerprint=provider_fingerprint,
                    dimension=dimension,
                    now=now,
                )
            elif str(row["status"]) in {"FAILED", "RETIRED"} and not bool(row["active"]):
                conn.execute(
                    """UPDATE recall_index_generations SET status = 'BUILDING',
                       failure_reason = '', embedding_provider_id = ?, vector_dimension = ?,
                       updated_at = ? WHERE generation_id = ?""",
                    (provider_id, int(dimension), now, int(row["generation_id"])),
                )
            self._refresh_generation_counts(conn, profile_id, instance_id, now)
            current = conn.execute(
                "SELECT * FROM recall_index_generations WHERE generation_id = ?",
                (int(row["generation_id"]),),
            ).fetchone()
            assert current is not None
            return self._record(current)

        return await self.uow.run(operation)

    @staticmethod
    def _insert_generation(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        *,
        provider_id: str,
        provider_fingerprint: str,
        dimension: int,
        now: str,
    ) -> sqlite3.Row:
        cursor = conn.execute(
            """INSERT INTO recall_index_generations(
                profile_id, instance_id, embedding_provider_id,
                provider_fingerprint, vector_dimension, status, active,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'BUILDING', 0, ?, ?)""",
            (
                profile_id,
                instance_id,
                provider_id,
                provider_fingerprint,
                int(dimension),
                now,
                now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM recall_index_generations WHERE generation_id = ?",
            (int(cursor.lastrowid),),
        ).fetchone()
        assert row is not None
        return row

    async def active_generation(self, profile_id: str, instance_id: str) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            """SELECT * FROM recall_index_generations
            WHERE profile_id = ? AND instance_id = ? AND active = 1 AND status = 'READY'""",
            (profile_id, instance_id),
        )
        return self._record(row) if row is not None else None

    async def building_generations(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT * FROM recall_index_generations
            WHERE status = 'BUILDING'
               OR (active = 1 AND status = 'READY' AND embedded_count < document_count)
            ORDER BY updated_at, generation_id LIMIT ?""",
            (max(1, min(int(limit), 100)),),
        )
        return [self._record(row) for row in rows]

    async def missing_embedding_documents(
        self, generation_id: int, *, limit: int = 32
    ) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT document.document_key, document.content,
               document.source_fingerprint AS content_hash
            FROM recall_index_generations AS generation
            JOIN recall_documents AS document
              ON document.profile_id = generation.profile_id
             AND document.instance_id = generation.instance_id
             AND document.dense_eligible = 1
            LEFT JOIN recall_embeddings AS embedding
              ON embedding.document_key = document.document_key
             AND embedding.generation_id = generation.generation_id
            WHERE generation.generation_id = ? AND embedding.document_key IS NULL
            ORDER BY document.document_key LIMIT ?""",
            (int(generation_id), max(1, min(int(limit), 256))),
        )
        return [self._record(row) for row in rows]

    async def store_embeddings(
        self,
        generation_id: int,
        *,
        provider_fingerprint: str,
        dimension: int,
        rows: Sequence[Mapping[str, Any]],
    ) -> int:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            stored = sum(
                self._store_embedding_row(
                    conn,
                    generation_id,
                    provider_fingerprint=provider_fingerprint,
                    dimension=dimension,
                    row=row,
                    now=now,
                )
                for row in rows
            )
            generation = conn.execute(
                "SELECT profile_id, instance_id FROM recall_index_generations "
                "WHERE generation_id = ?",
                (int(generation_id),),
            ).fetchone()
            if generation is not None:
                self._refresh_generation_counts(conn, str(generation[0]), str(generation[1]), now)
            return stored

        return await self.uow.run(operation)

    @staticmethod
    def _store_embedding_row(
        conn: sqlite3.Connection,
        generation_id: int,
        *,
        provider_fingerprint: str,
        dimension: int,
        row: Mapping[str, Any],
        now: str,
    ) -> int:
        cursor = conn.execute(
            """INSERT INTO recall_embeddings(
                document_key, generation_id, vector_dimension, vector_blob,
                content_hash, provider_fingerprint, created_at
            ) SELECT document_key, ?, ?, ?, ?, ?, ? FROM recall_documents
            WHERE document_key = ? AND source_fingerprint = ?
            ON CONFLICT(document_key, generation_id) DO UPDATE SET
                vector_dimension = excluded.vector_dimension,
                vector_blob = excluded.vector_blob,
                content_hash = excluded.content_hash,
                provider_fingerprint = excluded.provider_fingerprint,
                created_at = excluded.created_at""",
            (
                int(generation_id),
                int(dimension),
                sqlite3.Binary(bytes(row["vector_blob"])),
                row["content_hash"],
                provider_fingerprint,
                now,
                row["document_key"],
                row["content_hash"],
            ),
        )
        return max(0, int(cursor.rowcount))

    async def embedding_rows(self, generation_id: int) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT embedding.document_key, embedding.vector_dimension,
               embedding.vector_blob, document.source_fingerprint,
               document.authority_status, document.source_type
            FROM recall_embeddings AS embedding
            JOIN recall_documents AS document ON document.document_key = embedding.document_key
            WHERE embedding.generation_id = ? ORDER BY embedding.document_key""",
            (int(generation_id),),
        )
        return [self._record(row) for row in rows]

    async def activate_generation(self, generation_id: int) -> bool:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                "SELECT * FROM recall_index_generations WHERE generation_id = ?",
                (int(generation_id),),
            ).fetchone()
            if row is None or str(row["status"]) != "BUILDING":
                return False
            self._refresh_generation_counts(
                conn, str(row["profile_id"]), str(row["instance_id"]), now
            )
            row = conn.execute(
                "SELECT * FROM recall_index_generations WHERE generation_id = ?",
                (int(generation_id),),
            ).fetchone()
            if row is None or int(row["embedded_count"]) != int(row["document_count"]):
                return False
            self._activate_generation_row(conn, row, generation_id, now)
            return True

        return await self.uow.run(operation)

    @staticmethod
    def _activate_generation_row(
        conn: sqlite3.Connection, row: sqlite3.Row, generation_id: int, now: str
    ) -> None:
        conn.execute(
            """UPDATE recall_index_generations SET active = 0,
               status = CASE WHEN active = 1 THEN 'RETIRED' ELSE status END,
               updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND generation_id <> ?""",
            (now, row["profile_id"], row["instance_id"], int(generation_id)),
        )
        conn.execute(
            """UPDATE recall_index_generations SET status = 'READY', active = 1,
               activated_at = ?, updated_at = ? WHERE generation_id = ?""",
            (now, now, int(generation_id)),
        )
        retired = conn.execute(
            """SELECT generation_id FROM recall_index_generations
            WHERE profile_id = ? AND instance_id = ? AND status = 'RETIRED'
            ORDER BY COALESCE(activated_at, updated_at) DESC, generation_id DESC""",
            (row["profile_id"], row["instance_id"]),
        ).fetchall()
        for old in retired[1:]:
            conn.execute(
                "DELETE FROM recall_index_generations WHERE generation_id = ?",
                (int(old[0]),),
            )

    async def fail_generation(self, generation_id: int, error: str) -> None:
        now = _dt(_now())
        await self.uow.run(
            lambda conn: conn.execute(
                """UPDATE recall_index_generations SET status = 'FAILED', active = 0,
                   failure_reason = ?, updated_at = ? WHERE generation_id = ?""",
                (str(error or "")[:500], now, int(generation_id)),
            )
        )

    async def index_status(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        counts = await self.db.fetch_one(
            """SELECT
                COUNT(*) AS documents,
                SUM(CASE WHEN dense_eligible = 1 THEN 1 ELSE 0 END) AS dense_documents
            FROM recall_documents WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        )
        pending = await self.db.fetch_one(
            """SELECT COUNT(*) AS pending FROM recall_index_outbox
            WHERE profile_id = ? AND instance_id = ?
              AND status IN ('PENDING', 'LEASED', 'FAILED')""",
            (profile_id, instance_id),
        )
        active = await self.active_generation(profile_id, instance_id)
        building = await self.db.fetch_one(
            """SELECT * FROM recall_index_generations
            WHERE profile_id = ? AND instance_id = ? AND status IN ('BUILDING', 'FAILED')
            ORDER BY generation_id DESC LIMIT 1""",
            (profile_id, instance_id),
        )
        return {
            "documents": int(counts["documents"] or 0) if counts else 0,
            "dense_documents": int(counts["dense_documents"] or 0) if counts else 0,
            "pending_tasks": int(pending["pending"] or 0) if pending else 0,
            "active_generation": active,
            "latest_build": self._record(building) if building is not None else None,
        }

    async def record_recall_report(
        self,
        profile_id: str,
        instance_id: str,
        query: str,
        report: Mapping[str, Any],
        *,
        current_message_id: int | None = None,
    ) -> None:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO recall_probe_reports(
                    profile_id, instance_id, current_message_id, query, report_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    profile_id,
                    instance_id,
                    current_message_id,
                    str(query or "")[:4000],
                    self._json(report),
                    now,
                ),
            )
            conn.execute(
                """DELETE FROM recall_probe_reports
                WHERE profile_id = ? AND instance_id = ? AND report_id NOT IN (
                    SELECT report_id FROM recall_probe_reports
                    WHERE profile_id = ? AND instance_id = ?
                    ORDER BY report_id DESC LIMIT 100
                )""",
                (profile_id, instance_id, profile_id, instance_id),
            )

        await self.uow.run(operation)

    async def list_recall_reports(
        self, profile_id: str, instance_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT * FROM recall_probe_reports WHERE profile_id = ?
            AND instance_id = ? ORDER BY report_id DESC LIMIT ?""",
            (profile_id, instance_id, max(1, min(int(limit), 100))),
        )
        return [self._record(row, json_columns=("report_json",)) for row in rows]


__all__ = ["RecallIndexRepositoryMixin"]
