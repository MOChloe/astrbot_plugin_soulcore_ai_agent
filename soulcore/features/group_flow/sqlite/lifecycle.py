from __future__ import annotations

import sqlite3
from datetime import datetime

from ....storage.sqlite.codec import _dt
from ..service import advance_group_activity_release_boundary
from .due import GroupFlowDueSql
from .orphan_recovery import GroupFlowOrphanRecoverySql


class GroupFlowLifecycleSql(GroupFlowDueSql, GroupFlowOrphanRecoverySql):
    async def mark_first_attempt_started(
        self,
        profile_id: str,
        instance_id: str,
        window_id: str,
        *,
        now: datetime,
    ) -> bool:
        return bool(
            await self.uow.run(
                lambda conn: self._finish_window_sql(
                    conn,
                    profile_id,
                    instance_id,
                    window_id,
                    from_statuses=("WAITING_FIRST_ATTEMPT",),
                    status="RESOLVED",
                    outcome="FIRST_ATTEMPT_STARTED",
                    now=now,
                    first_attempt=True,
                )
            )
        )

    async def resolve_window(
        self,
        profile_id: str,
        instance_id: str,
        window_id: str,
        *,
        outcome: str,
        now: datetime,
    ) -> bool:
        normalized = str(outcome).strip().upper() or "COMPLETED"
        status = (
            "CANCELLED"
            if normalized == "CANCELLED"
            else "FAILED"
            if normalized == "FAILED"
            else "RESOLVED"
        )
        return bool(
            await self.uow.run(
                lambda conn: self._finish_window_sql(
                    conn,
                    profile_id,
                    instance_id,
                    window_id,
                    from_statuses=(
                        "COLLECTING",
                        "JUDGING",
                        "READY",
                        "RUNNING",
                        "WAITING_FIRST_ATTEMPT",
                    ),
                    status=status,
                    outcome=normalized,
                    now=now,
                    first_attempt=False,
                )
            )
        )

    @staticmethod
    def _finish_window_sql(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        window_id: str,
        *,
        from_statuses: tuple[str, ...],
        status: str,
        outcome: str,
        now: datetime,
        first_attempt: bool,
    ) -> bool:
        placeholders = ",".join("?" for _ in from_statuses)
        row = conn.execute(
            f"""SELECT * FROM group_flow_windows WHERE profile_id = ? AND instance_id = ?
            AND window_id = ? AND status IN ({placeholders})""",
            (profile_id, instance_id, window_id, *from_statuses),
        ).fetchone()
        if row is None:
            return False
        now_text = _dt(now)
        assert now_text is not None
        GroupFlowLifecycleSql._update_finished_window(
            conn,
            window_id=window_id,
            status=status,
            outcome=outcome,
            first_attempt=first_attempt,
            now_text=now_text,
        )
        GroupFlowLifecycleSql._update_finished_instance(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            last_message_id=int(row["last_message_id"]),
            first_attempt=first_attempt,
            now_text=now_text,
        )
        advance_group_activity_release_boundary(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            through_message_id=int(row["last_message_id"]),
            now=str(now_text),
        )
        GroupFlowLifecycleSql._release_window_knowledge_hold(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            window_id=window_id,
            now_text=now_text,
        )
        return True

    @staticmethod
    def _update_finished_window(
        conn: sqlite3.Connection,
        *,
        window_id: str,
        status: str,
        outcome: str,
        first_attempt: bool,
        now_text: str,
    ) -> None:
        conn.execute(
            """UPDATE group_flow_windows SET status = ?, first_attempt_started_at =
            CASE WHEN ? THEN ? ELSE first_attempt_started_at END,
            lease_owner = NULL, lease_until = NULL, lease_token = lease_token + 1,
            resolution_outcome = ?, version = version + 1,
            updated_at = ?, resolved_at = ? WHERE window_id = ?""",
            (
                status,
                int(first_attempt),
                now_text,
                outcome[:120],
                now_text,
                now_text,
                window_id,
            ),
        )

    @staticmethod
    def _update_finished_instance(
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        instance_id: str,
        last_message_id: int,
        first_attempt: bool,
        now_text: str,
    ) -> None:
        conn.execute(
            """INSERT INTO group_flow_instance_state(
            profile_id, instance_id, last_visible_assistant_at,
            last_resolved_message_id, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(profile_id, instance_id) DO UPDATE SET
            last_visible_assistant_at = CASE WHEN ?
                THEN excluded.last_visible_assistant_at
                ELSE group_flow_instance_state.last_visible_assistant_at END,
            last_resolved_message_id = excluded.last_resolved_message_id,
            updated_at = excluded.updated_at""",
            (
                profile_id,
                instance_id,
                now_text if first_attempt else None,
                last_message_id,
                now_text,
                int(first_attempt),
            ),
        )

    @staticmethod
    def _release_window_knowledge_hold(
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        instance_id: str,
        window_id: str,
        now_text: str,
    ) -> None:
        changed = conn.execute(
            """UPDATE instance_messages SET knowledge_eligibility = 'ELIGIBLE',
            knowledge_eligibility_reason = '' WHERE profile_id = ? AND instance_id = ?
            AND knowledge_eligibility = 'HELD'
            AND knowledge_eligibility_reason = 'group_flow_pending'
            AND message_id IN (SELECT message_id FROM group_flow_window_members
                WHERE window_id = ?)""",
            (profile_id, instance_id, window_id),
        ).rowcount
        if changed:
            conn.execute(
                """UPDATE knowledge_processing_state SET
                processing_version = processing_version + 1, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?""",
                (now_text, profile_id, instance_id),
            )

    async def recover(self, *, now: datetime) -> int:
        now_text = _dt(now)
        assert now_text is not None

        def operation(conn: sqlite3.Connection) -> int:
            orphans = self._recover_orphaned_messages(
                conn,
                now=now,
                now_text=now_text,
            )
            judgments = self._recover_judgments(conn, now_text)
            runs = conn.execute(
                """UPDATE group_flow_windows SET status = 'READY', ready_at = ?,
                lease_owner = NULL, lease_until = NULL, lease_token = lease_token + 1,
                main_core_task_ref = NULL, error_code = 'run_lease_expired',
                version = version + 1, updated_at = ?
                WHERE status = 'RUNNING' AND lease_until <= ?
                AND EXISTS (SELECT 1 FROM role_profiles profile
                    WHERE profile.profile_id = group_flow_windows.profile_id
                      AND profile.enabled = 1)
                AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                    WHERE chat_policy.profile_id = group_flow_windows.profile_id
                      AND chat_policy.instance_id = group_flow_windows.instance_id
                      AND chat_policy.soulcore_enabled = 0)""",
                (now_text, now_text, now_text),
            ).rowcount
            return orphans + judgments + runs

        return int(await self.uow.run(operation))

    @staticmethod
    def _recover_judgments(conn: sqlite3.Connection, now_text: str) -> int:
        rows = list(
            conn.execute(
                """SELECT * FROM group_flow_windows
                WHERE status = 'JUDGING' AND lease_until <= ?
                AND EXISTS (SELECT 1 FROM role_profiles profile
                    WHERE profile.profile_id = group_flow_windows.profile_id
                      AND profile.enabled = 1)
                AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                    WHERE chat_policy.profile_id = group_flow_windows.profile_id
                      AND chat_policy.instance_id = group_flow_windows.instance_id
                      AND chat_policy.soulcore_enabled = 0)""",
                (now_text,),
            )
        )
        for row in rows:
            judged_through = int(row["judge_through_message_id"] or row["last_message_id"])
            conn.execute(
                """UPDATE group_flow_instance_state SET last_judged_message_id = ?,
                updated_at = ? WHERE profile_id = ? AND instance_id = ?""",
                (
                    judged_through,
                    now_text,
                    row["profile_id"],
                    row["instance_id"],
                ),
            )
            advance_group_activity_release_boundary(
                conn,
                profile_id=str(row["profile_id"]),
                instance_id=str(row["instance_id"]),
                through_message_id=judged_through,
                now=now_text,
            )
            conn.execute(
                """UPDATE group_flow_windows SET status = 'COLLECTING',
                judge_result = 'UNSUITABLE', judge_error_code = 'judge_lease_expired',
                judge_threshold = MIN(4096, message_count + 1), next_judge_at = NULL,
                lease_owner = NULL, lease_until = NULL, lease_token = lease_token + 1,
                version = version + 1, updated_at = ? WHERE window_id = ?""",
                (now_text, row["window_id"]),
            )
        return len(rows)


__all__ = ["GroupFlowLifecycleSql"]
