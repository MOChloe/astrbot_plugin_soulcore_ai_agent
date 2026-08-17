from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from datetime import datetime

from ....storage.sqlite.codec import _dt, _parse
from ....storage.sqlite.inbound_admission_transactions import InboundAdmissionTransactions

KnowledgeRefresh = Callable[..., None]


class RecoverTurnBufferBatches:
    def __init__(
        self,
        *,
        inbound_admission: InboundAdmissionTransactions,
        refresh_knowledge_task: KnowledgeRefresh,
        now: datetime,
    ) -> None:
        self.inbound_admission = inbound_admission
        self.refresh_knowledge_task = refresh_knowledge_task
        self.now = now
        self.now_text = _dt(now)
        assert self.now_text is not None
        self.recovery_owner = f"turn-buffer-recovery:{uuid.uuid4().hex}"

    def __call__(self, conn: sqlite3.Connection) -> int:
        recovered = self._recover_expired_batches(conn)
        for identity, rows in self._claimed_orphans(conn).items():
            profile_id, instance_id = identity
            self._apply_admissions(conn, profile_id, instance_id, rows)
            batch_id = self._ensure_waiting_batch(conn, profile_id, instance_id)
            recovered += self._attach_and_complete(
                conn,
                profile_id,
                instance_id,
                batch_id,
                rows,
            )
        return recovered

    def _recover_expired_batches(self, conn: sqlite3.Connection) -> int:
        return int(
            conn.execute(
                """UPDATE conversation_turn_buffer_batches SET status = 'WAITING',
                requested_delay_seconds = NULL, ai_elapsed_seconds = NULL,
                remaining_delay_seconds = 0, due_at = ?, lease_owner = NULL,
                lease_until = NULL, lease_token = lease_token + 1,
                error_code = 'classification_lease_expired', version = version + 1,
                updated_at = ? WHERE status = 'CLASSIFYING' AND lease_until <= ?""",
                (self.now_text, self.now_text, self.now_text),
            ).rowcount
        )

    def _claimed_orphans(
        self,
        conn: sqlite3.Connection,
    ) -> dict[tuple[str, str], list[tuple[sqlite3.Row, int]]]:
        rows = conn.execute(
            """SELECT m.profile_id, m.instance_id, m.message_id,
            batch.batch_id
            FROM instance_messages m
            LEFT JOIN conversation_turn_buffer_members member
              ON member.profile_id = m.profile_id
             AND member.instance_id = m.instance_id
             AND member.message_id = m.message_id
            LEFT JOIN conversation_turn_buffer_batches batch
              ON batch.batch_id = member.batch_id AND batch.status IN
                ('PENDING','CLASSIFYING','WAITING','CLAIMED')
            WHERE m.direction = 'INBOUND' AND m.knowledge_eligibility = 'HELD'
              AND m.knowledge_eligibility_reason IN (
                'inbound_turn_buffer_pending', 'state_gate_pending_decision'
              )
              AND json_extract(m.metadata_json,
                '$.inbound_admission.status') = 'ADMITTING'
              AND json_extract(m.metadata_json,
                '$.inbound_admission.lease_until') <= ?
            ORDER BY m.profile_id, m.instance_id, m.message_id""",
            (self.now_text,),
        )
        grouped: dict[tuple[str, str], list[tuple[sqlite3.Row, int]]] = {}
        for row in rows:
            token = self._claim_orphan(conn, row)
            if token is not None:
                grouped.setdefault(
                    (str(row["profile_id"]), str(row["instance_id"])),
                    [],
                ).append((row, token))
        return grouped

    def _claim_orphan(self, conn: sqlite3.Connection, row: sqlite3.Row) -> int | None:
        return self.inbound_admission.claim_expired(
            conn,
            profile_id=str(row["profile_id"]),
            instance_id=str(row["instance_id"]),
            message_id=int(row["message_id"]),
            lease_owner=self.recovery_owner,
            now=self.now,
            lease_seconds=30,
        )

    def _apply_admissions(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        rows: list[tuple[sqlite3.Row, int]],
    ) -> None:
        for row, token in rows:
            admission = self.inbound_admission.apply(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                message_id=int(row["message_id"]),
                now=self.now,
                group_scope=False,
                refresh_knowledge_task=self.refresh_knowledge_task,
                lease_owner=self.recovery_owner,
                lease_token=token,
            )
            if not admission.ownership_valid:
                raise RuntimeError("turn-buffer orphan admission ownership changed")

    def _ensure_waiting_batch(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
    ) -> str:
        active = conn.execute(
            """SELECT batch_id FROM conversation_turn_buffer_batches
            WHERE profile_id = ? AND instance_id = ? AND status IN
              ('PENDING','CLASSIFYING','WAITING','CLAIMED')""",
            (profile_id, instance_id),
        ).fetchone()
        current_epoch = self._current_epoch(conn, profile_id, instance_id)
        if active is None:
            batch_id = f"turn:recovery:{uuid.uuid4().hex}"
            conn.execute(
                """INSERT INTO conversation_turn_buffer_batches(
                batch_id, profile_id, instance_id, activity_epoch, status,
                remaining_delay_seconds, due_at, error_code, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'WAITING', 0, ?,
                  'orphaned_held_messages', ?, ?)""",
                (
                    batch_id,
                    profile_id,
                    instance_id,
                    current_epoch,
                    self.now_text,
                    self.now_text,
                    self.now_text,
                ),
            )
            return batch_id
        batch_id = str(active["batch_id"])
        conn.execute(
            """UPDATE conversation_turn_buffer_batches SET generation = generation + 1,
            activity_epoch = ?,
            status = 'WAITING', requested_delay_seconds = NULL,
            ai_elapsed_seconds = NULL, remaining_delay_seconds = 0, due_at = ?,
            lease_owner = NULL, lease_until = NULL, lease_token = lease_token + 1,
            error_code = 'orphaned_held_messages', version = version + 1,
            updated_at = ? WHERE batch_id = ?""",
            (current_epoch, self.now_text, self.now_text, batch_id),
        )
        return batch_id

    @staticmethod
    def _current_epoch(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
    ) -> int:
        row = conn.execute(
            """SELECT activity_epoch FROM instance_core_state
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        ).fetchone()
        return int(row["activity_epoch"])

    def _attach_and_complete(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        batch_id: str,
        rows: list[tuple[sqlite3.Row, int]],
    ) -> int:
        ordinal_row = conn.execute(
            """SELECT COALESCE(MAX(ordinal), -1) FROM conversation_turn_buffer_members
            WHERE batch_id = ?""",
            (batch_id,),
        ).fetchone()
        ordinal = int(ordinal_row[0]) + 1
        recovered = 0
        for row, token in rows:
            if row["batch_id"] is None:
                conn.execute(
                    """INSERT OR IGNORE INTO conversation_turn_buffer_members(
                    batch_id, profile_id, instance_id, message_id, ordinal, added_at
                    ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        batch_id,
                        profile_id,
                        instance_id,
                        int(row["message_id"]),
                        ordinal,
                        self.now_text,
                    ),
                )
                ordinal += 1
            recovered += 1
            completed = self.inbound_admission.complete(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                message_id=int(row["message_id"]),
                lease_owner=self.recovery_owner,
                lease_token=token,
                now=self.now,
            )
            if not completed:
                raise RuntimeError("turn-buffer orphan handoff ownership changed")
        return recovered


class TurnBufferRecoveryMixin:
    async def recover_turn_buffer_batches(
        self,
        *,
        now: datetime,
    ) -> int:
        return await self.uow.run(
            RecoverTurnBufferBatches(
                inbound_admission=self._inbound_admission,
                refresh_knowledge_task=self._refresh_knowledge_task_sql,
                now=now,
            )
        )

    async def reconcile_turn_buffer_switches(
        self,
        *,
        now: datetime,
        profile_id: str | None = None,
    ) -> int:
        """Fence in-flight work and release waits after profile switch changes."""

        now_text = _dt(now)
        assert now_text is not None
        normalized_profile = str(profile_id or "").strip()

        def operation(conn: sqlite3.Connection) -> int:
            profile_clause = " AND batch.profile_id = ?" if normalized_profile else ""
            profile_params: tuple[object, ...] = (normalized_profile,) if normalized_profile else ()
            invalidated = conn.execute(
                f"""UPDATE conversation_turn_buffer_batches AS batch SET
                status = 'PENDING', due_at = NULL, lease_owner = NULL,
                lease_until = NULL, lease_token = lease_token + 1,
                error_code = CASE WHEN EXISTS (
                    SELECT 1 FROM role_profiles profile
                    WHERE profile.profile_id = batch.profile_id AND profile.enabled = 0
                ) THEN 'PROFILE_DISABLED' ELSE 'FEATURE_DISABLED' END,
                version = version + 1, updated_at = ?
                WHERE batch.status = 'CLASSIFYING'{profile_clause}
                  AND EXISTS (
                    SELECT 1 FROM role_profiles profile
                    WHERE profile.profile_id = batch.profile_id
                      AND (profile.enabled = 0 OR profile.turn_buffer_enabled = 0)
                  )""",
                (now_text, *profile_params),
            ).rowcount
            released = conn.execute(
                f"""UPDATE conversation_turn_buffer_batches AS batch SET
                remaining_delay_seconds = 0, due_at = ?,
                error_code = 'FEATURE_DISABLED', version = version + 1,
                updated_at = ? WHERE batch.status = 'WAITING'{profile_clause}
                  AND batch.due_at > ? AND EXISTS (
                    SELECT 1 FROM role_profiles profile
                    WHERE profile.profile_id = batch.profile_id
                      AND profile.enabled = 1 AND profile.turn_buffer_enabled = 0
                  )""",
                (now_text, now_text, *profile_params, now_text),
            ).rowcount
            return int(invalidated + released)

        return await self.uow.run(operation)

    async def next_turn_buffer_due_at(self) -> datetime | None:
        row = await self.db.fetch_one(
            """SELECT MIN(due_at) AS due_at FROM conversation_turn_buffer_batches
            WHERE status = 'WAITING' AND EXISTS (
                SELECT 1 FROM role_profiles profile
                WHERE profile.profile_id = conversation_turn_buffer_batches.profile_id
                  AND profile.enabled = 1
            )"""
        )
        return _parse(row["due_at"]) if row and row["due_at"] else None

    async def _recover_expired_classification_leases(self, *, now: datetime) -> int:
        now_text = _dt(now)
        assert now_text is not None
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE conversation_turn_buffer_batches SET status = 'WAITING',
                requested_delay_seconds = NULL, ai_elapsed_seconds = NULL,
                remaining_delay_seconds = 0, due_at = ?, lease_owner = NULL,
                lease_until = NULL, lease_token = lease_token + 1,
                error_code = 'classification_lease_expired', version = version + 1,
                updated_at = ? WHERE status = 'CLASSIFYING' AND lease_until <= ?""",
                (now_text, now_text, now_text),
            ),
            transaction=True,
        )
        return int(cursor.rowcount)
