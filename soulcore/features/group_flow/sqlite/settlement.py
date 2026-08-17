from __future__ import annotations

import sqlite3
from datetime import datetime

from ....contracts.group_flow import GroupFlowPolicy, GroupFlowWindow, GroupRunFence
from ....storage.sqlite.codec import _dt, _parse
from ..policy import GroupSchedule, build_schedule, reply_gap_due_at
from ..service import advance_group_activity_release_boundary
from .lifecycle import GroupFlowLifecycleSql


class GroupFlowSettlementSql(GroupFlowLifecycleSql):
    async def record_judgment(
        self,
        window: GroupFlowWindow,
        *,
        suitable: bool,
        error_code: str,
        now: datetime,
    ) -> GroupFlowWindow | None:
        def operation(conn: sqlite3.Connection) -> bool:
            current = conn.execute(
                """SELECT * FROM group_flow_windows WHERE window_id = ?
                AND status = 'JUDGING' AND version = ? AND lease_token = ?
                AND EXISTS (SELECT 1 FROM role_profiles profile
                    WHERE profile.profile_id = group_flow_windows.profile_id
                      AND profile.enabled = 1)
                AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                    WHERE chat_policy.profile_id = group_flow_windows.profile_id
                      AND chat_policy.instance_id = group_flow_windows.instance_id
                      AND chat_policy.soulcore_enabled = 0)""",
                (window.window_id, window.version, window.lease_token),
            ).fetchone()
            if current is None:
                return False
            if suitable:
                self._accept_judgment(
                    conn,
                    current,
                    error_code=error_code,
                    now=now,
                )
            else:
                self._decline_and_merge(
                    conn,
                    current,
                    error_code=error_code,
                    now=now,
                )
            return True

        if not await self.uow.run(operation):
            return None
        return await self.get_window(window.profile_id, window.instance_id, window.window_id)

    @staticmethod
    def _accept_judgment(
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        error_code: str,
        now: datetime,
    ) -> None:
        now_text = _dt(now)
        conn.execute(
            """UPDATE group_flow_windows SET status = 'READY',
            frozen_through_message_id = judge_through_message_id,
            judge_result = 'SUITABLE', judge_error_code = ?, ready_at = ?,
            lease_owner = NULL, lease_until = NULL, version = version + 1,
            updated_at = ? WHERE window_id = ?""",
            (
                str(error_code).strip()[:120],
                now_text,
                now_text,
                row["window_id"],
            ),
        )
        conn.execute(
            """UPDATE group_flow_instance_state SET last_judged_message_id = ?,
            updated_at = ? WHERE profile_id = ? AND instance_id = ?""",
            (
                row["judge_through_message_id"],
                now_text,
                row["profile_id"],
                row["instance_id"],
            ),
        )

    def _decline_and_merge(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        error_code: str,
        now: datetime,
    ) -> None:
        profile_id = str(row["profile_id"])
        instance_id = str(row["instance_id"])
        members, direct = self._merge_next_members(conn, row)
        state = conn.execute(
            """SELECT * FROM group_flow_instance_state
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        ).fetchone()
        policy_row = conn.execute(
            """SELECT * FROM group_flow_policies
            WHERE profile_id = ? AND scope = 'group'""",
            (profile_id,),
        ).fetchone()
        policy = self._policy(policy_row) if policy_row else GroupFlowPolicy(profile_id)
        first_at = _parse(members[0]["occurred_at"])
        last_at = _parse(members[-1]["occurred_at"])
        assert first_at is not None and last_at is not None
        total = len(members)
        unique_count = len({str(item["normalized_fingerprint"]) for item in members})
        repeat_ratio = (total - unique_count) / total
        last_visible = _parse(state["last_visible_assistant_at"]) if state else None
        schedule = build_schedule(
            policy,
            first_at=first_at,
            last_at=last_at,
            previous_rate=float(state["rate_ewma"]) if state else float(row["rate_ewma"]),
            previous_at=None,
            repeat_ratio=repeat_ratio,
            direct_address=direct,
            last_visible_at=last_visible,
            now=now,
        )
        gap_due = reply_gap_due_at(policy, last_visible_at=last_visible)
        self._persist_declined_window(
            conn,
            row,
            members,
            schedule,
            repeat_ratio,
            direct,
            error_code=error_code,
            gap_due=gap_due,
            now=now,
        )
        conn.execute(
            """UPDATE group_flow_instance_state SET last_judged_message_id = ?,
            updated_at = ? WHERE profile_id = ? AND instance_id = ?""",
            (row["judge_through_message_id"], _dt(now), profile_id, instance_id),
        )
        advance_group_activity_release_boundary(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            through_message_id=int(row["judge_through_message_id"]),
            now=str(_dt(now)),
        )

    @staticmethod
    def _merge_next_members(
        conn: sqlite3.Connection, row: sqlite3.Row
    ) -> tuple[list[sqlite3.Row], bool]:
        next_window = conn.execute(
            """SELECT * FROM group_flow_windows WHERE profile_id = ? AND instance_id = ?
            AND status = 'COLLECTING'""",
            (row["profile_id"], row["instance_id"]),
        ).fetchone()
        moved: list[sqlite3.Row] = []
        direct = bool(row["direct_address"])
        if next_window is not None:
            moved = list(
                conn.execute(
                    """SELECT * FROM group_flow_window_members
                    WHERE window_id = ? ORDER BY ordinal""",
                    (next_window["window_id"],),
                )
            )
            direct = direct or bool(next_window["direct_address"])
            conn.execute(
                "DELETE FROM group_flow_windows WHERE window_id = ?",
                (next_window["window_id"],),
            )
        ordinal = (
            int(
                conn.execute(
                    """SELECT COALESCE(MAX(ordinal), -1) FROM group_flow_window_members
                WHERE window_id = ?""",
                    (row["window_id"],),
                ).fetchone()[0]
            )
            + 1
        )
        for item in moved:
            conn.execute(
                """INSERT INTO group_flow_window_members(
                window_id, profile_id, instance_id, message_id, ordinal,
                normalized_fingerprint, media_cluster_keys_json, sender_id, occurred_at, added_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["window_id"],
                    row["profile_id"],
                    row["instance_id"],
                    item["message_id"],
                    ordinal,
                    item["normalized_fingerprint"],
                    item["media_cluster_keys_json"],
                    item["sender_id"],
                    item["occurred_at"],
                    item["added_at"],
                ),
            )
            ordinal += 1
        return list(
            conn.execute(
                """SELECT * FROM group_flow_window_members
                WHERE window_id = ? ORDER BY ordinal""",
                (row["window_id"],),
            )
        ), direct

    def _persist_declined_window(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        members: list[sqlite3.Row],
        schedule: GroupSchedule,
        repeat_ratio: float,
        direct: bool,
        *,
        error_code: str,
        gap_due: datetime | None,
        now: datetime,
    ) -> None:
        conn.execute(
            """UPDATE group_flow_windows SET status = 'COLLECTING',
            last_message_id = ?, message_count = ?, rate_ewma = ?, repeat_ratio = ?,
            judge_threshold = ?, next_judge_at = ?, quiet_due_at = ?,
            dynamic_due_at = ?, direct_due_at = ?, direct_address = ?,
            judge_result = 'UNSUITABLE', judge_error_code = ?,
            lease_owner = NULL, lease_until = NULL, lease_token = lease_token + 1,
            version = version + 1, updated_at = ? WHERE window_id = ?""",
            (
                members[-1]["message_id"],
                len(members),
                schedule.rate_ewma,
                repeat_ratio,
                schedule.judge_threshold,
                _dt(self._later(schedule.next_judge_at, gap_due)),
                _dt(self._later(schedule.quiet_due_at, gap_due)),
                _dt(self._later(schedule.dynamic_due_at, gap_due)),
                _dt(schedule.direct_due_at),
                int(direct),
                str(error_code).strip()[:120],
                _dt(now),
                row["window_id"],
            ),
        )

    async def mark_waiting_first_attempt(
        self,
        profile_id: str,
        instance_id: str,
        fence: GroupRunFence,
        *,
        now: datetime,
    ) -> bool:
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE group_flow_windows SET status = 'WAITING_FIRST_ATTEMPT',
                lease_owner = NULL, lease_until = NULL, version = version + 1,
                updated_at = ? WHERE profile_id = ? AND instance_id = ?
                AND window_id = ? AND status = 'RUNNING'
                AND frozen_through_message_id = ? AND lease_token = ? AND version = ?
                AND main_core_task_ref = ?""",
                (
                    _dt(now),
                    profile_id,
                    instance_id,
                    fence.window_id,
                    fence.frozen_through_message_id,
                    fence.lease_token,
                    fence.version,
                    fence.main_core_task_ref,
                ),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1

    async def is_first_attempt_protected(
        self, profile_id: str, instance_id: str, window_id: str
    ) -> bool:
        row = await self.db.fetch_one(
            """SELECT 1 FROM group_flow_windows WHERE profile_id = ? AND instance_id = ?
            AND window_id = ? AND status = 'WAITING_FIRST_ATTEMPT'""",
            (profile_id, instance_id, window_id),
        )
        return row is not None

    async def has_protected_run(self, profile_id: str, instance_id: str) -> bool:
        row = await self.db.fetch_one(
            """SELECT 1 FROM group_flow_windows WHERE profile_id = ? AND instance_id = ?
            AND status IN ('RUNNING','WAITING_FIRST_ATTEMPT')""",
            (profile_id, instance_id),
        )
        return row is not None

    async def next_collecting_message_id(self, profile_id: str, instance_id: str) -> int | None:
        row = await self.db.fetch_one(
            """SELECT MIN(first_message_id) AS message_id FROM group_flow_windows
            WHERE profile_id = ? AND instance_id = ? AND status = 'COLLECTING'""",
            (profile_id, instance_id),
        )
        return int(row["message_id"]) if row and row["message_id"] is not None else None

    @staticmethod
    def _later(value: datetime, boundary: datetime | None) -> datetime:
        return value if boundary is None or value >= boundary else boundary


__all__ = ["GroupFlowSettlementSql"]
