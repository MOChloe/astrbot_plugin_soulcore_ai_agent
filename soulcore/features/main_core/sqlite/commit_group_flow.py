"""Group-run fence and window settlement in the Main Core transaction."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from ....contracts.group_flow import GroupRunFence


@dataclass(frozen=True, slots=True)
class GroupFlowCommitSettlement:
    profile_id: str
    instance_id: str
    fence: GroupRunFence | None
    source_run_id: int
    segment_index: int | None
    has_visible_output: bool
    final_commit: bool
    now: str

    def claim_is_current(self, conn: sqlite3.Connection) -> bool:
        fence = self.fence
        if fence is None:
            return True
        if self._has_prior_expression(conn):
            return self._continuation_fence_is_current(conn, fence)
        row = conn.execute(
            """SELECT 1 FROM group_flow_windows
            WHERE profile_id = ? AND instance_id = ? AND window_id = ?
              AND status = 'RUNNING' AND frozen_through_message_id = ?
              AND lease_token = ? AND version = ? AND main_core_task_ref = ?""",
            (
                self.profile_id,
                self.instance_id,
                fence.window_id,
                fence.frozen_through_message_id,
                fence.lease_token,
                fence.version,
                fence.main_core_task_ref,
            ),
        ).fetchone()
        return row is not None

    def resolve(self, conn: sqlite3.Connection) -> None:
        fence = self.fence
        if fence is None:
            return
        # A single MainCore run may commit several ordered expression
        # segments.  The first visible segment owns the one-time transition;
        # later segments and the final silent commit must preserve that
        # settlement instead of pretending the run produced no output.
        if self._has_prior_expression(conn):
            return
        if self.has_visible_output:
            self._wait_for_first_attempt(conn, fence)
            return
        if not self.final_commit:
            raise RuntimeError("a non-final group segment cannot be silent")
        self._resolve_without_output(conn, fence)

    def _has_prior_expression(self, conn: sqlite3.Connection) -> bool:
        if self.segment_index is None:
            row = conn.execute(
                """SELECT 1 FROM instance_expression_batches
                WHERE profile_id = ? AND instance_id = ? AND source_run_id = ?
                LIMIT 1""",
                (self.profile_id, self.instance_id, int(self.source_run_id)),
            ).fetchone()
            return row is not None
        row = conn.execute(
            """SELECT 1 FROM instance_expression_batches
            WHERE profile_id = ? AND instance_id = ? AND source_run_id = ?
              AND segment_index < ? LIMIT 1""",
            (
                self.profile_id,
                self.instance_id,
                int(self.source_run_id),
                int(self.segment_index),
            ),
        ).fetchone()
        return row is not None

    def _continuation_fence_is_current(
        self,
        conn: sqlite3.Connection,
        fence: GroupRunFence,
    ) -> bool:
        row = conn.execute(
            """SELECT status, version, resolution_outcome
            FROM group_flow_windows
            WHERE profile_id = ? AND instance_id = ? AND window_id = ?
              AND frozen_through_message_id = ? AND lease_token = ?
              AND main_core_task_ref = ?""",
            (
                self.profile_id,
                self.instance_id,
                fence.window_id,
                fence.frozen_through_message_id,
                fence.lease_token,
                fence.main_core_task_ref,
            ),
        ).fetchone()
        if row is None:
            return False
        status = str(row["status"])
        if status == "WAITING_FIRST_ATTEMPT":
            return int(row["version"]) == int(fence.version) + 1
        return bool(
            status == "RESOLVED" and str(row["resolution_outcome"]) == "ADAPTER_CALL_STARTED"
        )

    def _wait_for_first_attempt(self, conn: sqlite3.Connection, fence: GroupRunFence) -> None:
        changed = conn.execute(
            """UPDATE group_flow_windows SET status = 'WAITING_FIRST_ATTEMPT',
            lease_owner = NULL, lease_until = NULL,
            version = version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND window_id = ?
              AND status = 'RUNNING' AND frozen_through_message_id = ?
              AND lease_token = ? AND version = ? AND main_core_task_ref = ?""",
            self._fence_arguments(fence),
        ).rowcount
        if changed != 1:
            raise RuntimeError("group-run ownership changed during Main Core commit")

    def _resolve_without_output(self, conn: sqlite3.Connection, fence: GroupRunFence) -> None:
        row = conn.execute(
            """SELECT last_message_id FROM group_flow_windows
            WHERE profile_id = ? AND instance_id = ? AND window_id = ?""",
            (self.profile_id, self.instance_id, fence.window_id),
        ).fetchone()
        changed = conn.execute(
            """UPDATE group_flow_windows SET status = 'RESOLVED',
            lease_owner = NULL, lease_until = NULL, lease_token = lease_token + 1,
            resolution_outcome = 'MAIN_CORE_COMMITTED_NO_VISIBLE_OUTPUT',
            version = version + 1, updated_at = ?, resolved_at = ?
            WHERE profile_id = ? AND instance_id = ? AND window_id = ?
              AND status = 'RUNNING' AND frozen_through_message_id = ?
              AND lease_token = ? AND version = ? AND main_core_task_ref = ?""",
            (self.now, *self._fence_arguments(fence)),
        ).rowcount
        if changed != 1 or row is None:
            raise RuntimeError("group-run ownership changed during Main Core commit")
        self._record_resolution(conn, int(row["last_message_id"]))
        self._release_knowledge(conn, fence.window_id)

    def _fence_arguments(self, fence: GroupRunFence) -> tuple[object, ...]:
        return (
            self.now,
            self.profile_id,
            self.instance_id,
            fence.window_id,
            fence.frozen_through_message_id,
            fence.lease_token,
            fence.version,
            fence.main_core_task_ref,
        )

    def _record_resolution(self, conn: sqlite3.Connection, message_id: int) -> None:
        conn.execute(
            """INSERT INTO group_flow_instance_state(
                profile_id, instance_id, last_resolved_message_id, updated_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(profile_id, instance_id) DO UPDATE SET
                last_resolved_message_id = excluded.last_resolved_message_id,
                updated_at = excluded.updated_at""",
            (self.profile_id, self.instance_id, message_id, self.now),
        )

    def _release_knowledge(self, conn: sqlite3.Connection, window_id: str) -> None:
        changed = conn.execute(
            """UPDATE instance_messages SET knowledge_eligibility = 'ELIGIBLE',
            knowledge_eligibility_reason = ''
            WHERE profile_id = ? AND instance_id = ?
              AND knowledge_eligibility = 'HELD'
              AND knowledge_eligibility_reason = 'group_flow_pending'
              AND message_id IN (SELECT message_id
                FROM group_flow_window_members WHERE window_id = ?)""",
            (self.profile_id, self.instance_id, window_id),
        ).rowcount
        if changed:
            conn.execute(
                """UPDATE knowledge_processing_state SET
                processing_version = processing_version + 1, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?""",
                (self.now, self.profile_id, self.instance_id),
            )


__all__ = ["GroupFlowCommitSettlement"]
