from __future__ import annotations

from dataclasses import dataclass

from .support import Any, _dt, datetime, sqlite3, uuid


@dataclass(frozen=True, slots=True)
class DeferredBatchAppendContext:
    profile_id: str
    instance_id: str
    message_id: int
    due_at: datetime
    activity_epoch: int
    gate_generation: int
    creation_key: str
    identifier: str
    message_ref: str | None
    idempotency_key: str | None
    received_at: datetime | None
    now: str


class DeferredBatchAppendTransaction:
    def __init__(self, context: DeferredBatchAppendContext) -> None:
        self.context = context

    def __call__(self, conn: sqlite3.Connection) -> str:
        message = self._load_inbound_message(conn)
        stable_ref = str(
            self.context.message_ref
            or message["idempotency_key"]
            or f"ledger:{self.context.message_id}"
        )
        stable_key = str(self.context.idempotency_key or stable_ref)
        self._insert_batch(conn)
        batch = self._load_appendable_batch(conn)
        self._refresh_batch(conn, batch)
        self._insert_item(conn, batch, message, stable_ref, stable_key)
        self._hold_message(conn)
        return str(batch["batch_id"])

    def _load_inbound_message(self, conn: sqlite3.Connection) -> sqlite3.Row:
        context = self.context
        message = conn.execute(
            """SELECT direction, idempotency_key, occurred_at,
            knowledge_eligibility FROM instance_messages WHERE profile_id = ?
            AND instance_id = ? AND message_id = ?""",
            (context.profile_id, context.instance_id, context.message_id),
        ).fetchone()
        if message is None or message["direction"] != "INBOUND":
            raise ValueError("deferred item must reference an inbound instance message")
        return message

    def _insert_batch(self, conn: sqlite3.Connection) -> None:
        context = self.context
        conn.execute(
            """INSERT INTO deferred_message_batches(
                batch_id, profile_id, instance_id, due_at, activity_epoch,
                gate_generation, creation_key, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id, instance_id, creation_key) DO NOTHING""",
            (
                context.identifier,
                context.profile_id,
                context.instance_id,
                _dt(context.due_at),
                context.activity_epoch,
                context.gate_generation,
                context.creation_key,
                context.now,
                context.now,
            ),
        )

    def _load_appendable_batch(self, conn: sqlite3.Connection) -> sqlite3.Row:
        context = self.context
        row = conn.execute(
            """SELECT batch_id, status FROM deferred_message_batches
            WHERE profile_id = ? AND instance_id = ? AND creation_key = ?""",
            (context.profile_id, context.instance_id, context.creation_key),
        ).fetchone()
        if row is None or row["status"] not in {"PENDING", "CLAIMED"}:
            raise ValueError("deferred batch is no longer appendable")
        return row

    def _refresh_batch(self, conn: sqlite3.Connection, batch: sqlite3.Row) -> None:
        context = self.context
        if batch["status"] == "CLAIMED":
            conn.execute(
                """UPDATE deferred_message_batches SET status = 'PENDING',
                due_at = MAX(due_at, ?), activity_epoch = MAX(activity_epoch, ?),
                lease_until = NULL, lease_token = lease_token + 1,
                version = version + 1, updated_at = ?
                WHERE batch_id = ? AND status = 'CLAIMED'""",
                (
                    _dt(context.due_at),
                    context.activity_epoch,
                    context.now,
                    batch["batch_id"],
                ),
            )
            return
        conn.execute(
            """UPDATE deferred_message_batches SET due_at = MAX(due_at, ?),
            activity_epoch = MAX(activity_epoch, ?), updated_at = ?
            WHERE batch_id = ? AND status = 'PENDING'""",
            (
                _dt(context.due_at),
                context.activity_epoch,
                context.now,
                batch["batch_id"],
            ),
        )

    def _insert_item(
        self,
        conn: sqlite3.Connection,
        batch: sqlite3.Row,
        message: sqlite3.Row,
        stable_ref: str,
        stable_key: str,
    ) -> None:
        context = self.context
        conn.execute(
            """INSERT INTO deferred_message_items(
                batch_id, profile_id, instance_id, message_id, message_ref,
                idempotency_key, activity_epoch, received_at, added_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING""",
            (
                batch["batch_id"],
                context.profile_id,
                context.instance_id,
                context.message_id,
                stable_ref,
                stable_key,
                context.activity_epoch,
                _dt(context.received_at) or message["occurred_at"],
                context.now,
            ),
        )

    def _hold_message(self, conn: sqlite3.Connection) -> None:
        context = self.context
        cursor = conn.execute(
            """UPDATE instance_messages SET knowledge_eligibility = 'HELD',
            knowledge_eligibility_reason = 'state_gate_deferred'
            WHERE profile_id = ? AND instance_id = ? AND message_id = ?
              AND (knowledge_eligibility != 'HELD'
                   OR knowledge_eligibility_reason != 'state_gate_deferred')""",
            (context.profile_id, context.instance_id, context.message_id),
        )
        if cursor.rowcount:
            conn.execute(
                """UPDATE knowledge_processing_state SET
                processing_version = processing_version + 1,
                updated_at = ? WHERE profile_id = ? AND instance_id = ?""",
                (context.now, context.profile_id, context.instance_id),
            )


@dataclass(frozen=True, slots=True)
class DeferredBatchClaimContext:
    now_text: str
    orphan_cutoff: str
    lease_until: str
    limit: int
    profile_id: str | None
    instance_id: str | None
    include_policy_disabled: bool = False


class DeferredBatchClaimTransaction:
    def __init__(self, context: DeferredBatchClaimContext) -> None:
        self.context = context

    def __call__(self, conn: sqlite3.Connection) -> list[str]:
        for message in self._orphan_messages(conn):
            self._recover_orphan(conn, message)
        self._release_expired_leases(conn)
        return self._claim_due(conn)

    def _orphan_messages(self, conn: sqlite3.Connection) -> list[sqlite3.Row]:
        return list(
            conn.execute(
                """SELECT m.*, core.activity_epoch,
                COALESCE(gate.generation, 1) AS gate_generation,
                instance.initialization_state AS instance_initialization_state
                FROM instance_messages m
                JOIN instance_core_state core ON core.profile_id = m.profile_id
                  AND core.instance_id = m.instance_id
                JOIN character_instances instance ON instance.profile_id = m.profile_id
                  AND instance.instance_id = m.instance_id
                LEFT JOIN instance_state_gate_snapshots gate
                  ON gate.profile_id = m.profile_id AND gate.instance_id = m.instance_id
                LEFT JOIN deferred_message_items item
                  ON item.profile_id = m.profile_id AND item.instance_id = m.instance_id
                 AND item.message_id = m.message_id
                WHERE m.direction = 'INBOUND' AND m.knowledge_eligibility = 'HELD'
                  AND (
                    (m.knowledge_eligibility_reason = 'state_gate_pending_decision'
                     AND m.created_at <= ?)
                    OR
                    (m.knowledge_eligibility_reason = 'instance_initialization_pending'
                     AND m.created_at <= ?)
                  )
                  AND item.message_id IS NULL""",
                (self.context.orphan_cutoff, self.context.orphan_cutoff),
            )
        )

    def _recover_orphan(self, conn: sqlite3.Connection, message: sqlite3.Row) -> None:
        context = self.context
        initialization_trigger = (
            str(message["knowledge_eligibility_reason"]) == "instance_initialization_pending"
        )
        creation_key = (
            f"instance-initialization:{int(message['message_id'])}"
            if initialization_trigger
            else f"state-gate-recovery:{int(message['message_id'])}"
        )
        batch_id = creation_key if initialization_trigger else f"defer:recovery:{uuid.uuid4().hex}"
        due_at = (
            context.now_text
            if initialization_trigger and str(message["instance_initialization_state"]) == "READY"
            else ("9999-12-31T23:59:59+00:00" if initialization_trigger else context.now_text)
        )
        resolution_reason = (
            "recovered_instance_initialization_trigger"
            if initialization_trigger
            else "recovered_pending_decision"
        )
        conn.execute(
            """INSERT INTO deferred_message_batches(
            batch_id, profile_id, instance_id, due_at, activity_epoch,
            gate_generation, creation_key, resolution_reason, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id, instance_id, creation_key) DO NOTHING""",
            (
                batch_id,
                message["profile_id"],
                message["instance_id"],
                due_at,
                int(message["activity_epoch"]),
                1 if initialization_trigger else max(1, int(message["gate_generation"])),
                creation_key,
                resolution_reason,
                context.now_text,
                context.now_text,
            ),
        )
        selected = self._selected_recovery_batch(conn, message, creation_key)
        if selected is None:
            return
        stable = str(message["idempotency_key"] or f"ledger:{int(message['message_id'])}")
        self._insert_recovered_item(
            conn,
            message,
            selected["batch_id"],
            stable,
            reason=(
                "instance_initialization_deferred_recovered"
                if initialization_trigger
                else "state_gate_deferred_recovered"
            ),
        )

    @staticmethod
    def _selected_recovery_batch(
        conn: sqlite3.Connection, message: sqlite3.Row, creation_key: str
    ) -> sqlite3.Row | None:
        return conn.execute(
            """SELECT batch_id FROM deferred_message_batches
            WHERE profile_id = ? AND instance_id = ? AND creation_key = ?""",
            (message["profile_id"], message["instance_id"], creation_key),
        ).fetchone()

    def _insert_recovered_item(
        self,
        conn: sqlite3.Connection,
        message: sqlite3.Row,
        batch_id: str,
        stable: str,
        *,
        reason: str,
    ) -> None:
        conn.execute(
            """INSERT OR IGNORE INTO deferred_message_items(
            batch_id, profile_id, instance_id, message_id, message_ref,
            idempotency_key, activity_epoch, received_at, added_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                batch_id,
                message["profile_id"],
                message["instance_id"],
                int(message["message_id"]),
                stable,
                stable,
                int(message["activity_epoch"]),
                message["occurred_at"],
                self.context.now_text,
            ),
        )
        conn.execute(
            """UPDATE instance_messages SET
            knowledge_eligibility_reason = ?
            WHERE profile_id = ? AND instance_id = ? AND message_id = ?
              AND knowledge_eligibility = 'HELD'""",
            (
                reason,
                message["profile_id"],
                message["instance_id"],
                int(message["message_id"]),
            ),
        )

    def _release_expired_leases(self, conn: sqlite3.Connection) -> None:
        context = self.context
        # A claimed batch handed to a durable MAIN_CORE task remains owned
        # while that task is queued, paused, running, or awaiting explicit
        # reconciliation.  Extend the exact persisted fence before releasing
        # unrelated expired claims; otherwise queue delay manufactures a new
        # task generation for the same messages.
        conn.execute(
            """UPDATE deferred_message_batches SET lease_until = ?, updated_at = ?
            WHERE status = 'CLAIMED' AND lease_until <= ? AND EXISTS (
                SELECT 1 FROM ai_tasks AS task
                WHERE task.profile_id = deferred_message_batches.profile_id
                  AND task.instance_id = deferred_message_batches.instance_id
                  AND task.task_type = 'MAIN_CORE'
                  AND task.status IN (
                    'READY','SCHEDULED','RETRY_WAIT','RUNNING',
                    'PAUSE_REQUESTED','PAUSED','CANCEL_REQUESTED',
                    'RECOVERY_REQUIRED'
                  )
                  AND json_extract(
                    task.input_json,
                    '$.payload.metadata.deferred_gate_fence.batch_ref'
                  ) = deferred_message_batches.batch_id
                  AND CAST(json_extract(
                    task.input_json,
                    '$.payload.metadata.deferred_gate_fence.version'
                  ) AS INTEGER) = deferred_message_batches.version
                  AND CAST(json_extract(
                    task.input_json,
                    '$.payload.metadata.deferred_gate_fence.lease_token'
                  ) AS INTEGER) = deferred_message_batches.lease_token
                  AND CAST(json_extract(
                    task.input_json,
                    '$.payload.metadata.deferred_gate_fence.gate_generation'
                  ) AS INTEGER) = deferred_message_batches.gate_generation
                  AND CAST(json_extract(
                    task.input_json,
                    '$.payload.metadata.deferred_gate_fence.activity_epoch'
                  ) AS INTEGER) = deferred_message_batches.activity_epoch
            )""",
            (context.lease_until, context.now_text, context.now_text),
        )
        conn.execute(
            """UPDATE deferred_message_batches SET status = 'PENDING',
            lease_until = NULL, lease_token = lease_token + 1,
            version = version + 1, updated_at = ? WHERE status = 'CLAIMED'
            AND lease_until <= ?""",
            (context.now_text, context.now_text),
        )

    def _claim_due(self, conn: sqlite3.Connection) -> list[str]:
        context = self.context
        if context.include_policy_disabled:
            sql = """SELECT batch.batch_id FROM deferred_message_batches batch
                JOIN character_instances instance
                  ON instance.profile_id = batch.profile_id
                 AND instance.instance_id = batch.instance_id
                JOIN scope_state_gate_policies policy
                  ON policy.profile_id = batch.profile_id
                 AND policy.scope = instance.scope
                LEFT JOIN instance_state_gate_overrides gate_override
                  ON gate_override.profile_id = batch.profile_id
                 AND gate_override.instance_id = batch.instance_id
                WHERE batch.status = 'PENDING' AND (
                    batch.due_at <= ?
                    OR COALESCE(gate_override.enabled, policy.enabled) = 0
                )"""
        else:
            sql = """SELECT batch.batch_id FROM deferred_message_batches batch
                WHERE batch.status = 'PENDING' AND batch.due_at <= ?"""
        sql += """ AND (
            batch.creation_key NOT LIKE 'instance-initialization:%'
            OR EXISTS (
                SELECT 1 FROM character_instances opening_instance
                WHERE opening_instance.profile_id = batch.profile_id
                  AND opening_instance.instance_id = batch.instance_id
                  AND opening_instance.initialization_state = 'READY'
            )
        )"""
        params: list[Any] = [context.now_text]
        if context.profile_id is not None:
            sql += " AND batch.profile_id = ?"
            params.append(context.profile_id)
        if context.instance_id is not None:
            sql += " AND batch.instance_id = ?"
            params.append(context.instance_id)
        sql += " ORDER BY batch.due_at, batch.batch_id LIMIT ?"
        params.append(max(0, context.limit))
        claimed: list[str] = []
        for row in conn.execute(sql, params):
            if self._claim_one(conn, str(row[0])):
                claimed.append(str(row[0]))
        return claimed

    def _claim_one(self, conn: sqlite3.Connection, batch_id: str) -> bool:
        context = self.context
        cursor = conn.execute(
            """UPDATE deferred_message_batches SET status = 'CLAIMED',
            lease_until = ?, lease_token = lease_token + 1,
            version = version + 1, updated_at = ? WHERE batch_id = ?
            AND status = 'PENDING'""",
            (context.lease_until, context.now_text, batch_id),
        )
        return bool(cursor.rowcount)
