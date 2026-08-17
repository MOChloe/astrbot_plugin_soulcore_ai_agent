from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime

from ....storage.sqlite.codec import _dt, _now
from ....storage.sqlite.inbound_admission_transactions import InboundAdmissionTransactions

KnowledgeRefresh = Callable[..., None]


def normalize_turn_buffer_append(
    message_ids: Sequence[int],
    activity_epoch: int,
    now: datetime,
    admission_message_id: int | None,
    admission_lease_owner: str | None,
    admission_lease_token: int | None,
) -> tuple[tuple[int, ...], str]:
    ids = tuple(int(value) for value in message_ids)
    if not ids or any(value < 1 for value in ids):
        raise ValueError("turn buffer requires persisted message ids")
    if len(ids) != len(set(ids)) or ids != tuple(sorted(ids)):
        raise ValueError("turn buffer message ids must be unique and ordered")
    if int(activity_epoch) < 0:
        raise ValueError("activity_epoch cannot be negative")
    handoff_values = (
        admission_message_id,
        admission_lease_owner,
        admission_lease_token,
    )
    if any(value is not None for value in handoff_values) and not all(
        value is not None for value in handoff_values
    ):
        raise ValueError("turn buffer admission handoff requires message, owner, and token")
    now_text = _dt(now)
    assert now_text is not None
    return ids, now_text


class AppendTurnBufferBatch:
    def __init__(
        self,
        *,
        inbound_admission: InboundAdmissionTransactions,
        profile_id: str,
        instance_id: str,
        message_ids: tuple[int, ...],
        activity_epoch: int,
        now_text: str,
        admission_message_id: int | None,
        admission_lease_owner: str | None,
        admission_lease_token: int | None,
    ) -> None:
        self.inbound_admission = inbound_admission
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.message_ids = message_ids
        self.activity_epoch = int(activity_epoch)
        self.now_text = now_text
        self.admission_message_id = admission_message_id
        self.admission_lease_owner = admission_lease_owner
        self.admission_lease_token = admission_lease_token

    def __call__(self, conn: sqlite3.Connection) -> str:
        self._validate_members(conn)
        current = self._active_batch(conn)
        if current is None:
            batch_id = self._create_batch(conn)
        else:
            batch_id = str(current["batch_id"])
            if self._matches_current(conn, current, batch_id):
                self._complete_handoff(conn)
                return batch_id
            self._refresh_batch(conn, batch_id)
        self._replace_members(conn, batch_id)
        self._hold_messages(conn)
        self._complete_handoff(conn)
        return batch_id

    def _validate_members(self, conn: sqlite3.Connection) -> None:
        placeholders = ",".join("?" for _ in self.message_ids)
        rows = conn.execute(
            f"""SELECT message_id FROM instance_messages
            WHERE profile_id = ? AND instance_id = ?
              AND message_id IN ({placeholders}) AND direction = 'INBOUND'
              AND delivery_status = 'RECEIVED'
            ORDER BY message_id""",
            (self.profile_id, self.instance_id, *self.message_ids),
        )
        if tuple(int(row[0]) for row in rows) != self.message_ids:
            raise ValueError("turn buffer members must be received inbound ledger messages")

    def _active_batch(self, conn: sqlite3.Connection) -> sqlite3.Row | None:
        return conn.execute(
            """SELECT * FROM conversation_turn_buffer_batches
            WHERE profile_id = ? AND instance_id = ?
              AND status IN ('PENDING','CLASSIFYING','WAITING','CLAIMED')""",
            (self.profile_id, self.instance_id),
        ).fetchone()

    def _create_batch(self, conn: sqlite3.Connection) -> str:
        batch_id = f"turn:{uuid.uuid4().hex}"
        conn.execute(
            """INSERT INTO conversation_turn_buffer_batches(
            batch_id, profile_id, instance_id, activity_epoch,
            created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                batch_id,
                self.profile_id,
                self.instance_id,
                self.activity_epoch,
                self.now_text,
                self.now_text,
            ),
        )
        return batch_id

    def _matches_current(
        self,
        conn: sqlite3.Connection,
        current: sqlite3.Row,
        batch_id: str,
    ) -> bool:
        old_ids = tuple(
            int(row[0])
            for row in conn.execute(
                """SELECT message_id FROM conversation_turn_buffer_members
                WHERE batch_id = ? ORDER BY ordinal""",
                (batch_id,),
            )
        )
        return old_ids == self.message_ids and int(current["activity_epoch"]) == self.activity_epoch

    def _refresh_batch(self, conn: sqlite3.Connection, batch_id: str) -> None:
        conn.execute(
            """UPDATE conversation_turn_buffer_batches SET
            generation = generation + 1, activity_epoch = ?, status = 'PENDING',
            requested_delay_seconds = NULL, ai_elapsed_seconds = NULL,
            remaining_delay_seconds = NULL, due_at = NULL,
            lease_owner = NULL, lease_until = NULL, lease_token = lease_token + 1,
            main_core_task_ref = NULL, error_code = '', resolution_outcome = '',
            version = version + 1, updated_at = ?, resolved_at = NULL
            WHERE batch_id = ?""",
            (self.activity_epoch, self.now_text, batch_id),
        )
        conn.execute(
            "DELETE FROM conversation_turn_buffer_members WHERE batch_id = ?",
            (batch_id,),
        )

    def _replace_members(self, conn: sqlite3.Connection, batch_id: str) -> None:
        conn.executemany(
            """INSERT INTO conversation_turn_buffer_members(
            batch_id, profile_id, instance_id, message_id, ordinal, added_at
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (
                    batch_id,
                    self.profile_id,
                    self.instance_id,
                    message_id,
                    ordinal,
                    self.now_text,
                )
                for ordinal, message_id in enumerate(self.message_ids)
            ],
        )

    def _hold_messages(self, conn: sqlite3.Connection) -> None:
        placeholders = ",".join("?" for _ in self.message_ids)
        changed = conn.execute(
            f"""UPDATE instance_messages SET knowledge_eligibility = 'HELD',
            knowledge_eligibility_reason = 'inbound_turn_buffer_pending'
            WHERE profile_id = ? AND instance_id = ? AND message_id IN ({placeholders})
              AND knowledge_eligibility != 'EXCLUDED'
              AND (knowledge_eligibility != 'HELD' OR
                   knowledge_eligibility_reason != 'inbound_turn_buffer_pending')""",
            (self.profile_id, self.instance_id, *self.message_ids),
        ).rowcount
        if changed:
            conn.execute(
                """UPDATE knowledge_processing_state SET
                processing_version = processing_version + 1, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?""",
                (self.now_text, self.profile_id, self.instance_id),
            )

    def _complete_handoff(self, conn: sqlite3.Connection) -> None:
        if self.admission_message_id is None:
            return
        assert self.admission_lease_owner is not None
        assert self.admission_lease_token is not None
        completed = self.inbound_admission.complete(
            conn,
            profile_id=self.profile_id,
            instance_id=self.instance_id,
            message_id=int(self.admission_message_id),
            lease_owner=self.admission_lease_owner,
            lease_token=int(self.admission_lease_token),
            now=_now(),
        )
        if not completed:
            raise RuntimeError("turn buffer admission ownership changed before handoff")
