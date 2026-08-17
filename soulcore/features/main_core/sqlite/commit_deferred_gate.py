"""Deferred-gate fence settlement inside the instance Main Core transaction."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ....contracts.deferred_gate import DeferredGateCommitFence


@dataclass(frozen=True, slots=True)
class DeferredGateCommitSettlement:
    owner: Any
    profile_id: str
    instance_id: str
    expected_activity_epoch: int
    run_id: int
    fence: DeferredGateCommitFence | None
    now: str

    def claim_is_current(self, conn: sqlite3.Connection) -> bool:
        fence = self.fence
        if fence is None:
            return True
        if fence.activity_epoch != self.expected_activity_epoch:
            return False
        row = conn.execute(
            """SELECT 1 FROM deferred_message_batches
            WHERE profile_id = ? AND instance_id = ? AND batch_id = ?
              AND status = 'CLAIMED' AND gate_generation = ?
              AND activity_epoch = ? AND version = ? AND lease_token = ?
              AND lease_until IS NOT NULL AND lease_until > ?""",
            (
                self.profile_id,
                self.instance_id,
                fence.batch_ref,
                fence.gate_generation,
                fence.activity_epoch,
                fence.version,
                fence.lease_token,
                self.now,
            ),
        ).fetchone()
        return row is not None

    def resolve(self, conn: sqlite3.Connection) -> None:
        fence = self.fence
        if fence is None:
            return
        changed = conn.execute(
            """UPDATE deferred_message_batches SET status = 'RESOLVED',
            resolution_reason = 'merged_into_foreground',
            resolution_run_id = ?, resolved_at = ?, lease_until = NULL,
            lease_token = lease_token + 1, version = version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND batch_id = ?
              AND status = 'CLAIMED' AND gate_generation = ?
              AND activity_epoch = ? AND version = ? AND lease_token = ?
              AND lease_until IS NOT NULL AND lease_until > ?""",
            (
                self.run_id,
                self.now,
                self.now,
                self.profile_id,
                self.instance_id,
                fence.batch_ref,
                fence.gate_generation,
                fence.activity_epoch,
                fence.version,
                fence.lease_token,
                self.now,
            ),
        ).rowcount
        if changed != 1:
            raise RuntimeError("deferred-gate ownership changed during Main Core commit")
        conn.execute(
            """UPDATE deferred_message_items SET status = 'RESOLVED',
            resolved_at = ? WHERE batch_id = ? AND status = 'PENDING'""",
            (self.now, fence.batch_ref),
        )
        conn.execute(
            """UPDATE instance_state_gate_snapshots
            SET reason_code = '', expression_context = '',
                not_before_at = NULL, until_at = NULL, source_run_id = NULL,
                version = version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?
              AND action = 'OPEN' AND reason_code = 'TEMPORARY_ABSENCE'
              AND generation = ?
              AND NOT EXISTS (
                SELECT 1 FROM deferred_message_batches AS pending
                WHERE pending.profile_id = instance_state_gate_snapshots.profile_id
                  AND pending.instance_id = instance_state_gate_snapshots.instance_id
                  AND pending.gate_generation = ?
                  AND pending.status IN ('PENDING', 'CLAIMED')
              )""",
            (
                self.now,
                self.profile_id,
                self.instance_id,
                fence.gate_generation + 1,
                fence.gate_generation,
            ),
        )
        released = conn.execute(
            """UPDATE instance_messages SET knowledge_eligibility = 'ELIGIBLE',
            knowledge_eligibility_reason = 'state_gate_resolved_foreground'
            WHERE profile_id = ? AND instance_id = ?
              AND knowledge_eligibility != 'EXCLUDED'
              AND message_id IN (
                SELECT message_id FROM deferred_message_items WHERE batch_id = ?
              )""",
            (self.profile_id, self.instance_id, fence.batch_ref),
        ).rowcount
        if released:
            conn.execute(
                """UPDATE knowledge_processing_state SET
                processing_version = processing_version + 1, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?""",
                (self.now, self.profile_id, self.instance_id),
            )
            self.owner._refresh_knowledge_task_sql(
                conn,
                self.profile_id,
                self.instance_id,
                now_dt=datetime.fromisoformat(self.now),
            )


__all__ = ["DeferredGateCommitSettlement"]
