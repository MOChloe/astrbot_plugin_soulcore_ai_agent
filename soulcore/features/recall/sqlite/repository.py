"""SQLite ownership for Recall's derived, rebuildable indexes."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from ....storage.sqlite.codec import _dt, _now
from ....storage.sqlite.engine import SqliteEngine
from ....storage.sqlite.uow import SqliteUnitOfWork
from ..ports import RecallProjectionSnapshot
from .index_repository import RecallIndexRepositoryMixin
from .repository_support import RecallSqliteRepositorySupport

_VISIBLE_OUTBOUND = ("PLATFORM_ACCEPTED_UNCONFIRMED", "SENT", "DELIVERED")


class SqliteRecallRepository(RecallIndexRepositoryMixin, RecallSqliteRepositorySupport):
    def __init__(self, engine: SqliteEngine) -> None:
        self.db = engine
        self.uow = SqliteUnitOfWork(engine)

    async def all_instance_scopes(self) -> list[tuple[str, str]]:
        rows = await self.db.fetch_all(
            "SELECT profile_id, instance_id FROM character_instances "
            "ORDER BY profile_id, instance_id"
        )
        return [(str(row["profile_id"]), str(row["instance_id"])) for row in rows]

    async def snapshot_scope(self, profile_id: str, instance_id: str) -> RecallProjectionSnapshot:
        return await self.uow.run(lambda conn: self._snapshot_scope(conn, profile_id, instance_id))

    def _snapshot_scope(
        self, conn: sqlite3.Connection, profile_id: str, instance_id: str
    ) -> RecallProjectionSnapshot:
        scope = (profile_id, instance_id)
        watermark_row = conn.execute(
            "SELECT COALESCE(MAX(task_id), 0) FROM recall_index_outbox "
            "WHERE profile_id = ? AND instance_id = ?",
            scope,
        ).fetchone()
        memories, memory_terms, memory_sources = self._snapshot_memories(conn, scope)
        world_info, world_terms, world_sources = self._snapshot_world_info(conn, scope)
        role_events, role_current = self._snapshot_role_timeline(conn, scope)
        summaries, messages = self._snapshot_conversation(conn, scope)
        return RecallProjectionSnapshot(
            profile_id,
            instance_id,
            int(watermark_row[0] if watermark_row else 0),
            memories,
            memory_terms,
            memory_sources,
            world_info,
            world_terms,
            world_sources,
            role_events,
            role_current,
            summaries,
            messages,
        )

    def _snapshot_memories(
        self, conn: sqlite3.Connection, scope: tuple[str, str]
    ) -> tuple[tuple[dict[str, Any], ...], ...]:
        memories = self._records(
            conn.execute(
                """SELECT entry.memory_id, entry.status, entry.current_revision,
                   entry.created_at AS entry_created_at, entry.updated_at AS entry_updated_at,
                   revision.revision, revision.brief, revision.ultra_brief,
                   revision.importance, revision.event_time, revision.origin,
                   revision.change_reason, revision.created_at
                FROM memories AS entry
                JOIN memory_revisions AS revision ON revision.memory_id = entry.memory_id
                WHERE entry.profile_id = ? AND entry.instance_id = ?
                  AND entry.status <> 'RETRACTED'
                ORDER BY entry.memory_id, revision.revision""",
                scope,
            )
        )
        terms = self._records(
            conn.execute(
                """SELECT term.memory_id, term.revision, term.term, term.normalized_term
                FROM memory_terms AS term
                JOIN memories AS entry ON entry.memory_id = term.memory_id
                WHERE entry.profile_id = ? AND entry.instance_id = ?
                  AND entry.status <> 'RETRACTED'
                ORDER BY term.memory_id, term.revision, term.normalized_term""",
                scope,
            )
        )
        sources = self._records(
            conn.execute(
                """SELECT source.memory_id, source.revision, source.message_id,
                   source.quote, source.occurred_at
                FROM memory_revision_sources AS source
                JOIN memories AS entry ON entry.memory_id = source.memory_id
                WHERE entry.profile_id = ? AND entry.instance_id = ?
                  AND entry.status <> 'RETRACTED'
                ORDER BY source.memory_id, source.revision, source.message_id""",
                scope,
            )
        )
        return memories, terms, sources

    def _snapshot_world_info(
        self, conn: sqlite3.Connection, scope: tuple[str, str]
    ) -> tuple[tuple[dict[str, Any], ...], ...]:
        facts = self._records(
            conn.execute(
                """SELECT entry.knowledge_fact_id, entry.status, entry.current_revision,
                   entry.created_at AS entry_created_at, entry.updated_at AS entry_updated_at,
                   revision.revision, revision.name, revision.aliases_json,
                   revision.definition, revision.brief, revision.importance,
                   revision.category, revision.session_specific_reason, revision.origin,
                   revision.change_reason, revision.created_at,
                   revision.valid_from, revision.valid_until
                FROM knowledge_fact_entries AS entry
                JOIN knowledge_fact_revisions AS revision
                  ON revision.knowledge_fact_id = entry.knowledge_fact_id
                WHERE entry.profile_id = ? AND entry.instance_id = ?
                  AND entry.status <> 'RETRACTED'
                ORDER BY entry.knowledge_fact_id, revision.revision""",
                scope,
            )
        )
        terms = self._records(
            conn.execute(
                """SELECT term.knowledge_fact_id, term.revision, term.term,
                   term.normalized_term, term.term_kind
                FROM knowledge_fact_terms AS term
                JOIN knowledge_fact_entries AS entry
                  ON entry.knowledge_fact_id = term.knowledge_fact_id
                WHERE entry.profile_id = ? AND entry.instance_id = ?
                  AND entry.status <> 'RETRACTED'
                ORDER BY term.knowledge_fact_id, term.revision, term.term_kind,
                   term.normalized_term""",
                scope,
            )
        )
        sources = self._records(
            conn.execute(
                """SELECT source.knowledge_fact_id, source.revision,
                   source.message_id, source.quote, message.occurred_at
                FROM knowledge_fact_revision_sources AS source
                JOIN knowledge_fact_entries AS entry
                  ON entry.knowledge_fact_id = source.knowledge_fact_id
                JOIN instance_messages AS message ON message.message_id = source.message_id
                WHERE entry.profile_id = ? AND entry.instance_id = ?
                  AND entry.status <> 'RETRACTED'
                ORDER BY source.knowledge_fact_id, source.revision, source.message_id""",
                scope,
            )
        )
        return facts, terms, sources

    def _snapshot_role_timeline(
        self, conn: sqlite3.Connection, scope: tuple[str, str]
    ) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
        events = self._records(
            conn.execute(
                """SELECT event_id, public_ref, source, content, frame_start_at,
                   frame_end_at, leftover_text, created_at
                FROM background_role_timeline_events
                WHERE profile_id = ? AND instance_id = ?
                ORDER BY frame_start_at, event_id""",
                scope,
            )
        )
        current = self._records(
            conn.execute(
                """SELECT revision, narrative_time, location, doing, body_state,
                   mood, intention, current_concern, as_of, source, source_event_id,
                   created_at, updated_at
                FROM background_role_current_views
                WHERE profile_id = ? AND instance_id = ?""",
                scope,
            )
        )
        return events, current

    def _snapshot_conversation(
        self, conn: sqlite3.Connection, scope: tuple[str, str]
    ) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
        summaries = self._records(
            conn.execute(
                """SELECT summary_id, version, covered_from_message_id,
                   covered_through_message_id, rendered_text, created_at
                FROM dialogue_summaries
                WHERE profile_id = ? AND instance_id = ? AND rendered_text <> ''
                ORDER BY version""",
                scope,
            )
        )
        visible = ",".join("?" for _ in _VISIBLE_OUTBOUND)
        messages = self._records(
            conn.execute(
                f"""SELECT message_id, direction, role, sender_id, sender_name, plain_text,
                   delivery_status, occurred_at, created_at
                FROM instance_messages
                WHERE profile_id = ? AND instance_id = ? AND plain_text <> ''
                  AND knowledge_eligibility = 'ELIGIBLE'
                  AND ((direction = 'INBOUND' AND delivery_status = 'RECEIVED') OR
                       (direction = 'OUTBOUND' AND delivery_status IN ({visible})))
                ORDER BY message_id""",
                (*scope, *_VISIBLE_OUTBOUND),
            )
        )
        return summaries, messages

    async def publish_projection(
        self,
        profile_id: str,
        instance_id: str,
        *,
        outbox_watermark: int,
        documents: Sequence[Mapping[str, Any]],
        fts_rows: Mapping[str, str],
        edges: Sequence[Mapping[str, Any]],
        scenes: Sequence[Mapping[str, Any]],
        scene_members: Sequence[Mapping[str, Any]],
        graph_nodes: Sequence[Mapping[str, Any]],
        graph_edges: Sequence[Mapping[str, Any]],
    ) -> dict[str, int]:
        now = _dt(_now())
        return await self.uow.run(
            lambda conn: self._publish_projection_transaction(
                conn,
                profile_id,
                instance_id,
                outbox_watermark=outbox_watermark,
                documents=documents,
                fts_rows=fts_rows,
                edges=edges,
                scenes=scenes,
                scene_members=scene_members,
                graph_nodes=graph_nodes,
                graph_edges=graph_edges,
                now=now,
            )
        )

    def _publish_projection_transaction(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        *,
        outbox_watermark: int,
        documents: Sequence[Mapping[str, Any]],
        fts_rows: Mapping[str, str],
        edges: Sequence[Mapping[str, Any]],
        scenes: Sequence[Mapping[str, Any]],
        scene_members: Sequence[Mapping[str, Any]],
        graph_nodes: Sequence[Mapping[str, Any]],
        graph_edges: Sequence[Mapping[str, Any]],
        now: str,
    ) -> dict[str, int]:
        desired, changed, removed = self._reconcile_projection_documents(
            conn,
            profile_id,
            instance_id,
            documents=documents,
            fts_rows=fts_rows,
            now=now,
        )
        self._replace_projection_edges(
            conn, profile_id, instance_id, edges=edges, desired=desired, now=now
        )
        self._replace_projection_scenes(
            conn,
            profile_id,
            instance_id,
            scenes=scenes,
            scene_members=scene_members,
            desired=desired,
            now=now,
        )
        self._replace_projection_graph(
            conn,
            profile_id,
            instance_id,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            desired_documents=set(desired),
            desired_scenes={str(item["scene_key"]) for item in scenes},
            now=now,
        )
        self._complete_projection_tasks(
            conn,
            profile_id,
            instance_id,
            outbox_watermark=outbox_watermark,
            now=now,
        )
        return {
            "documents": len(desired),
            "fts_documents": len(desired),
            "fts_rows": len(desired),
            "changed": len(changed),
            "removed": len(removed),
            "edges": len(edges),
            "scenes": len(scenes),
        }

    def _reconcile_projection_documents(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        *,
        documents: Sequence[Mapping[str, Any]],
        fts_rows: Mapping[str, str],
        now: str,
    ) -> tuple[dict[str, Mapping[str, Any]], set[str], set[str]]:
        existing = {
            str(row["document_key"]): str(row["source_fingerprint"])
            for row in conn.execute(
                "SELECT document_key, source_fingerprint FROM recall_documents "
                "WHERE profile_id = ? AND instance_id = ?",
                (profile_id, instance_id),
            )
        }
        fts_counts = {
            str(row["document_key"]): int(row["row_count"])
            for row in conn.execute(
                "SELECT document_key, COUNT(*) AS row_count FROM recall_documents_fts "
                "WHERE profile_id = ? AND instance_id = ? GROUP BY document_key",
                (profile_id, instance_id),
            )
        }
        desired = {str(item["document_key"]): item for item in documents}
        removed = set(existing) - set(desired)
        changed = {
            key
            for key, item in desired.items()
            if existing.get(key) != str(item["source_fingerprint"])
        }
        fts_removed = set(fts_counts) - set(desired)
        fts_refresh = {key for key in desired if key in changed or fts_counts.get(key, 0) != 1}
        for key in sorted(removed | fts_removed | fts_refresh):
            conn.execute(
                "DELETE FROM recall_documents_fts "
                "WHERE profile_id = ? AND instance_id = ? AND document_key = ?",
                (profile_id, instance_id, key),
            )
        if changed:
            conn.executemany(
                "DELETE FROM recall_embeddings WHERE document_key = ?",
                ((key,) for key in sorted(changed)),
            )
        if removed:
            conn.executemany(
                "DELETE FROM recall_documents WHERE document_key = ?",
                ((key,) for key in sorted(removed)),
            )
        for key in sorted(desired):
            self._upsert_projection_document(
                conn,
                profile_id,
                instance_id,
                key=key,
                item=desired[key],
                fts_text=fts_rows.get(key, ""),
                write_fts=key in fts_refresh,
                now=now,
            )
        return desired, changed, removed

    def _upsert_projection_document(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        *,
        key: str,
        item: Mapping[str, Any],
        fts_text: str,
        write_fts: bool,
        now: str,
    ) -> None:
        conn.execute(
            """INSERT INTO recall_documents(
                document_key, profile_id, instance_id, source_type, source_key,
                source_revision, authority_status, title, content, search_text,
                entity_names_json, valid_from, valid_until, recorded_from,
                recorded_until, occurred_at, evidence_json, source_fingerprint,
                dense_eligible, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_key) DO UPDATE SET
                authority_status = excluded.authority_status,
                title = excluded.title,
                content = excluded.content,
                search_text = excluded.search_text,
                entity_names_json = excluded.entity_names_json,
                valid_from = excluded.valid_from,
                valid_until = excluded.valid_until,
                recorded_from = excluded.recorded_from,
                recorded_until = excluded.recorded_until,
                occurred_at = excluded.occurred_at,
                evidence_json = excluded.evidence_json,
                source_fingerprint = excluded.source_fingerprint,
                dense_eligible = excluded.dense_eligible,
                updated_at = excluded.updated_at""",
            (
                key,
                profile_id,
                instance_id,
                item["source_type"],
                item["source_key"],
                int(item.get("source_revision") or 0),
                item["authority_status"],
                item.get("title", ""),
                item["content"],
                item["search_text"],
                self._json(item.get("entity_names", ())),
                item.get("valid_from") or None,
                item.get("valid_until") or None,
                item.get("recorded_from") or None,
                item.get("recorded_until") or None,
                item.get("occurred_at") or None,
                self._json(item.get("evidence", ())),
                item["source_fingerprint"],
                1 if item.get("dense_eligible", True) else 0,
                item.get("created_at") or now,
                now,
            ),
        )
        if write_fts:
            conn.execute(
                "INSERT INTO recall_documents_fts(tokens, document_key, profile_id, "
                "instance_id) VALUES (?, ?, ?, ?)",
                (fts_text, key, profile_id, instance_id),
            )

    def _replace_projection_edges(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        *,
        edges: Sequence[Mapping[str, Any]],
        desired: Mapping[str, Mapping[str, Any]],
        now: str,
    ) -> None:
        conn.execute(
            "DELETE FROM recall_edges WHERE profile_id = ? AND instance_id = ?",
            (profile_id, instance_id),
        )
        for edge in edges:
            if (
                edge["source_document_key"] not in desired
                or edge["target_document_key"] not in desired
            ):
                continue
            conn.execute(
                """INSERT OR IGNORE INTO recall_edges(
                    profile_id, instance_id, source_document_key, target_document_key,
                    edge_type, weight, status, valid_from, valid_until, evidence_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?)""",
                (
                    profile_id,
                    instance_id,
                    edge["source_document_key"],
                    edge["target_document_key"],
                    edge["edge_type"],
                    float(edge.get("weight") or 1.0),
                    edge.get("valid_from") or None,
                    edge.get("valid_until") or None,
                    self._json(edge.get("evidence", ())),
                    now,
                ),
            )

    def _replace_projection_scenes(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        *,
        scenes: Sequence[Mapping[str, Any]],
        scene_members: Sequence[Mapping[str, Any]],
        desired: Mapping[str, Mapping[str, Any]],
        now: str,
    ) -> None:
        conn.execute(
            "DELETE FROM recall_scenes WHERE profile_id = ? AND instance_id = ?",
            (profile_id, instance_id),
        )
        for scene in scenes:
            conn.execute(
                """INSERT INTO recall_scenes(
                    scene_key, profile_id, instance_id, parent_scene_key, scene_level,
                    title, summary, search_text, occurred_from, occurred_until,
                    evidence_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    scene["scene_key"],
                    profile_id,
                    instance_id,
                    scene.get("parent_scene_key") or None,
                    scene["scene_level"],
                    scene["title"],
                    scene["summary"],
                    scene["search_text"],
                    scene.get("occurred_from") or None,
                    scene.get("occurred_until") or None,
                    self._json(scene.get("evidence", ())),
                    now,
                    now,
                ),
            )
        for member in scene_members:
            if member["document_key"] not in desired:
                continue
            conn.execute(
                """INSERT INTO recall_scene_members(
                    scene_key, document_key, membership_weight, evidence_json
                ) VALUES (?, ?, ?, ?)""",
                (
                    member["scene_key"],
                    member["document_key"],
                    float(member.get("membership_weight") or 1.0),
                    self._json(member.get("evidence", ())),
                ),
            )

    def _replace_projection_graph(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        *,
        graph_nodes: Sequence[Mapping[str, Any]],
        graph_edges: Sequence[Mapping[str, Any]],
        desired_documents: set[str],
        desired_scenes: set[str],
        now: str,
    ) -> None:
        conn.execute(
            "DELETE FROM recall_graph_nodes WHERE profile_id = ? AND instance_id = ?",
            (profile_id, instance_id),
        )
        inserted_nodes = self._insert_projection_graph_nodes(
            conn,
            profile_id,
            instance_id,
            graph_nodes=graph_nodes,
            desired_documents=desired_documents,
            desired_scenes=desired_scenes,
            now=now,
        )
        self._insert_projection_graph_edges(
            conn,
            profile_id,
            instance_id,
            graph_edges=graph_edges,
            inserted_nodes=inserted_nodes,
            desired_documents=desired_documents,
            now=now,
        )

    def _insert_projection_graph_nodes(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        *,
        graph_nodes: Sequence[Mapping[str, Any]],
        desired_documents: set[str],
        desired_scenes: set[str],
        now: str,
    ) -> set[str]:
        inserted_nodes: set[str] = set()
        for node in graph_nodes:
            document_key = str(node.get("document_key") or "")
            scene_key = str(node.get("scene_key") or "")
            if document_key and document_key not in desired_documents:
                continue
            if scene_key and scene_key not in desired_scenes:
                continue
            conn.execute(
                """INSERT INTO recall_graph_nodes(
                    node_key, profile_id, instance_id, node_type, stable_ref, label,
                    document_key, scene_key, valid_from, valid_until, evidence_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    node["node_key"],
                    profile_id,
                    instance_id,
                    node["node_type"],
                    node["stable_ref"],
                    node.get("label") or "",
                    document_key or None,
                    scene_key or None,
                    node.get("valid_from") or None,
                    node.get("valid_until") or None,
                    self._json(node.get("evidence", ())),
                    now,
                ),
            )
            inserted_nodes.add(str(node["node_key"]))
        return inserted_nodes

    def _insert_projection_graph_edges(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        *,
        graph_edges: Sequence[Mapping[str, Any]],
        inserted_nodes: set[str],
        desired_documents: set[str],
        now: str,
    ) -> None:
        for edge in graph_edges:
            if (
                str(edge["source_node_key"]) not in inserted_nodes
                or str(edge["target_node_key"]) not in inserted_nodes
            ):
                continue
            evidence_document_key = str(edge.get("evidence_document_key") or "")
            if evidence_document_key and evidence_document_key not in desired_documents:
                continue
            conn.execute(
                """INSERT OR IGNORE INTO recall_graph_edges(
                    profile_id, instance_id, source_node_key, target_node_key,
                    edge_type, weight, status, evidence_document_key, valid_from,
                    valid_until, evidence_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?, ?)""",
                (
                    profile_id,
                    instance_id,
                    edge["source_node_key"],
                    edge["target_node_key"],
                    edge["edge_type"],
                    float(edge.get("weight") or 1.0),
                    evidence_document_key or None,
                    edge.get("valid_from") or None,
                    edge.get("valid_until") or None,
                    self._json(edge.get("evidence", ())),
                    now,
                ),
            )

    def _complete_projection_tasks(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        *,
        outbox_watermark: int,
        now: str,
    ) -> None:
        if int(outbox_watermark) > 0:
            conn.execute(
                """UPDATE recall_index_outbox SET status = 'COMPLETED',
                   lease_owner = NULL, lease_until = NULL, last_error = '', updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND task_id <= ?""",
                (now, profile_id, instance_id, int(outbox_watermark)),
            )
        self._refresh_generation_counts(conn, profile_id, instance_id, now)

    async def claim_pending_scope(
        self, worker_id: str, *, lease_seconds: int = 60
    ) -> dict[str, Any] | None:
        now_dt = datetime.now(UTC)
        now = _dt(now_dt)
        lease_until = _dt(now_dt + timedelta(seconds=max(10, int(lease_seconds))))

        def operation(conn: sqlite3.Connection) -> dict[str, Any] | None:
            conn.execute(
                """UPDATE recall_index_outbox SET status = 'PENDING', lease_owner = NULL,
                   lease_until = NULL, updated_at = ?
                WHERE status = 'LEASED' AND lease_until IS NOT NULL AND lease_until <= ?""",
                (now, now),
            )
            row = conn.execute(
                """SELECT profile_id, instance_id FROM recall_index_outbox
                WHERE status = 'PENDING' AND not_before <= ?
                ORDER BY task_id LIMIT 1""",
                (now,),
            ).fetchone()
            if row is None:
                return None
            profile, instance = str(row["profile_id"]), str(row["instance_id"])
            max_row = conn.execute(
                """SELECT MAX(task_id) FROM recall_index_outbox
                WHERE profile_id = ? AND instance_id = ? AND status = 'PENDING'
                  AND not_before <= ?""",
                (profile, instance, now),
            ).fetchone()
            watermark = int(max_row[0] or 0)
            conn.execute(
                """UPDATE recall_index_outbox SET status = 'LEASED', lease_owner = ?,
                   lease_token = lease_token + 1, lease_until = ?,
                   attempt_count = attempt_count + 1, updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND status = 'PENDING'
                  AND task_id <= ?""",
                (worker_id, lease_until, now, profile, instance, watermark),
            )
            return {
                "profile_id": profile,
                "instance_id": instance,
                "watermark": watermark,
                "lease_until": lease_until,
            }

        return await self.uow.run(operation)

    async def fail_claim(
        self,
        profile_id: str,
        instance_id: str,
        watermark: int,
        worker_id: str,
        error: str,
    ) -> None:
        now_dt = datetime.now(UTC)
        now, retry_at = _dt(now_dt), _dt(now_dt + timedelta(seconds=5))
        await self.uow.run(
            lambda conn: (
                conn.execute(
                    """UPDATE recall_index_outbox
                SET status = CASE WHEN attempt_count >= 5 THEN 'FAILED' ELSE 'PENDING' END,
                    lease_owner = NULL, lease_until = NULL, last_error = ?,
                    not_before = CASE WHEN attempt_count >= 5 THEN not_before ELSE ? END,
                    updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND task_id <= ?
                  AND status = 'LEASED' AND lease_owner = ?""",
                    (
                        str(error or "")[:500],
                        retry_at,
                        now,
                        profile_id,
                        instance_id,
                        int(watermark),
                        worker_id,
                    ),
                ).rowcount
            )
        )

    async def release_worker_leases(self, worker_id: str) -> int:
        now = _dt(_now())
        return await self.uow.run(
            lambda conn: (
                conn.execute(
                    """UPDATE recall_index_outbox SET status = 'PENDING', lease_owner = NULL,
                   lease_until = NULL, updated_at = ?
                WHERE status = 'LEASED' AND lease_owner = ?""",
                    (now, worker_id),
                ).rowcount
            )
        )


__all__ = ["RecallProjectionSnapshot", "SqliteRecallRepository"]
