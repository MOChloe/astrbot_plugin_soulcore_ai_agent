from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from ....contracts.group_flow import GroupFlowWindow, GroupRunFence
from ....storage.sqlite.codec import _dt
from .judgment_admission import GroupFlowJudgmentAdmissionSql

_PIPELINE = "'JUDGING','READY','RUNNING','WAITING_FIRST_ATTEMPT'"


class GroupFlowClaimSql(GroupFlowJudgmentAdmissionSql):
    async def settle_due_windows(self, *, now: datetime, limit: int = 32) -> int:
        now_text = _dt(now)
        assert now_text is not None
        bounded_limit = max(0, int(limit))

        def operation(conn: sqlite3.Connection) -> int:
            promoted = self._promote_direct_due(
                conn,
                now_text=now_text,
                limit=bounded_limit,
            )
            resolved = self._resolve_declined_due(
                conn,
                now=now,
                now_text=now_text,
                limit=bounded_limit - promoted,
            )
            return promoted + resolved

        return int(await self.uow.run(operation))

    @staticmethod
    def _promote_direct_due(
        conn: sqlite3.Connection,
        *,
        now_text: str,
        limit: int,
    ) -> int:
        rows = list(
            conn.execute(
                f"""SELECT window_id FROM group_flow_windows window
                WHERE status = 'COLLECTING' AND EXISTS (
                    SELECT 1 FROM role_profiles profile
                    WHERE profile.profile_id = window.profile_id AND profile.enabled = 1
                ) AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                    WHERE chat_policy.profile_id = window.profile_id
                      AND chat_policy.instance_id = window.instance_id
                      AND chat_policy.soulcore_enabled = 0)
                AND direct_due_at IS NOT NULL AND direct_due_at <= ?
                AND NOT EXISTS (
                    SELECT 1 FROM group_flow_windows active
                    WHERE active.profile_id = window.profile_id
                      AND active.instance_id = window.instance_id
                      AND active.status IN ({_PIPELINE})
                ) ORDER BY direct_due_at, window_id LIMIT ?""",
                (now_text, max(0, limit)),
            )
        )
        changed = 0
        for row in rows:
            changed += conn.execute(
                """UPDATE group_flow_windows SET status = 'READY',
                frozen_through_message_id = last_message_id, ready_at = ?,
                judge_result = 'SUITABLE', lease_owner = NULL, lease_until = NULL,
                version = version + 1, updated_at = ?
                WHERE window_id = ? AND status = 'COLLECTING'
                  AND direct_due_at IS NOT NULL AND direct_due_at <= ?
                  AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                    WHERE chat_policy.profile_id = group_flow_windows.profile_id
                      AND chat_policy.instance_id = group_flow_windows.instance_id
                      AND chat_policy.soulcore_enabled = 0)""",
                (now_text, now_text, row["window_id"], now_text),
            ).rowcount
        return changed

    def _resolve_declined_due(
        self,
        conn: sqlite3.Connection,
        *,
        now: datetime,
        now_text: str,
        limit: int,
    ) -> int:
        if limit <= 0:
            return 0
        rows = list(
            conn.execute(
                f"""SELECT window_id, profile_id, instance_id
                FROM group_flow_windows window
                WHERE status = 'COLLECTING' AND direct_address = 0
                AND judge_result = 'UNSUITABLE'
                AND ((quiet_due_at IS NOT NULL AND quiet_due_at <= ?) OR
                     (dynamic_due_at IS NOT NULL AND dynamic_due_at <= ?))
                AND EXISTS (
                    SELECT 1 FROM role_profiles profile
                    WHERE profile.profile_id = window.profile_id AND profile.enabled = 1
                )
                AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                    WHERE chat_policy.profile_id = window.profile_id
                      AND chat_policy.instance_id = window.instance_id
                      AND chat_policy.soulcore_enabled = 0)
                AND EXISTS (
                    SELECT 1 FROM group_flow_instance_state judged
                    WHERE judged.profile_id = window.profile_id
                      AND judged.instance_id = window.instance_id
                      AND judged.last_judged_message_id >= window.last_message_id
                )
                AND NOT EXISTS (
                    SELECT 1 FROM group_flow_windows active
                    WHERE active.profile_id = window.profile_id
                      AND active.instance_id = window.instance_id
                      AND active.status IN ({_PIPELINE})
                ) ORDER BY MIN(quiet_due_at, dynamic_due_at), window_id LIMIT ?""",
                (now_text, now_text, limit),
            )
        )
        changed = 0
        for row in rows:
            changed += int(
                self._finish_window_sql(
                    conn,
                    str(row["profile_id"]),
                    str(row["instance_id"]),
                    str(row["window_id"]),
                    from_statuses=("COLLECTING",),
                    status="RESOLVED",
                    outcome="INTERJECTION_UNSUITABLE",
                    now=now,
                    first_attempt=False,
                )
            )
        return changed

    @staticmethod
    def _delay_ordinary_ready_windows(conn: sqlite3.Connection, *, now_text: str) -> None:
        conn.execute(
            """UPDATE group_flow_windows AS window SET ready_at = datetime(
              (SELECT state.last_visible_assistant_at
               FROM group_flow_instance_state state
               WHERE state.profile_id = window.profile_id
                 AND state.instance_id = window.instance_id),
              '+' || (SELECT policy.ordinary_min_reply_gap_seconds
                      FROM group_flow_policies policy
                      WHERE policy.profile_id = window.profile_id
                        AND policy.scope = 'group') || ' seconds'
            ), updated_at = ? WHERE status = 'READY' AND direct_address = 0
            AND EXISTS (SELECT 1 FROM role_profiles profile
              WHERE profile.profile_id = window.profile_id AND profile.enabled = 1)
            AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
              WHERE chat_policy.profile_id = window.profile_id
                AND chat_policy.instance_id = window.instance_id
                AND chat_policy.soulcore_enabled = 0)
            AND EXISTS (
              SELECT 1 FROM group_flow_instance_state state
              JOIN group_flow_policies policy
                ON policy.profile_id = state.profile_id AND policy.scope = 'group'
              WHERE state.profile_id = window.profile_id
                AND state.instance_id = window.instance_id
                AND state.last_visible_assistant_at IS NOT NULL
                AND policy.ordinary_min_reply_gap_seconds > 0
                AND julianday(state.last_visible_assistant_at)
                  + policy.ordinary_min_reply_gap_seconds / 86400.0
                  > julianday(window.ready_at)
            )""",
            (now_text,),
        )

    @staticmethod
    def _claim_ready_identities(
        conn: sqlite3.Connection,
        *,
        now_text: str,
        owner: str,
        lease_until: str,
        limit: int,
    ) -> list[tuple[str, str, str]]:
        rows = list(
            conn.execute(
                """SELECT window_id, profile_id, instance_id
                FROM group_flow_windows window
                WHERE status = 'READY' AND ready_at <= ?
                AND EXISTS (SELECT 1 FROM role_profiles profile
                    WHERE profile.profile_id = window.profile_id AND profile.enabled = 1)
                AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                    WHERE chat_policy.profile_id = window.profile_id
                      AND chat_policy.instance_id = window.instance_id
                      AND chat_policy.soulcore_enabled = 0)
                ORDER BY ready_at, window_id LIMIT ?""",
                (now_text, max(0, int(limit))),
            )
        )
        claimed: list[tuple[str, str, str]] = []
        for row in rows:
            cursor = conn.execute(
                """UPDATE group_flow_windows SET status = 'RUNNING',
                lease_owner = ?, lease_until = ?, lease_token = lease_token + 1,
                version = version + 1, updated_at = ?
                WHERE window_id = ? AND status = 'READY'
                AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                    WHERE chat_policy.profile_id = group_flow_windows.profile_id
                      AND chat_policy.instance_id = group_flow_windows.instance_id
                      AND chat_policy.soulcore_enabled = 0)""",
                (owner, lease_until, now_text, row["window_id"]),
            )
            if cursor.rowcount:
                claimed.append(
                    (str(row["profile_id"]), str(row["instance_id"]), str(row["window_id"]))
                )
        return claimed

    async def claim_ready_windows(
        self,
        *,
        now: datetime,
        worker_id: str,
        limit: int,
        lease_seconds: int,
    ) -> tuple[GroupFlowWindow, ...]:
        owner = str(worker_id).strip()
        if not owner:
            raise ValueError("group-flow worker_id cannot be empty")
        now_text = _dt(now)
        lease_until = _dt(now + timedelta(seconds=max(1, int(lease_seconds))))
        assert now_text is not None and lease_until is not None

        def operation(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
            self._delay_ordinary_ready_windows(conn, now_text=now_text)
            return self._claim_ready_identities(
                conn,
                now_text=now_text,
                owner=owner,
                lease_until=lease_until,
                limit=limit,
            )

        return await self._load_identities(await self.uow.run(operation))

    async def attach_main_core_run(
        self,
        window: GroupFlowWindow,
        *,
        main_core_task_ref: str,
        now: datetime,
    ) -> GroupRunFence | None:
        reference = str(main_core_task_ref).strip()
        if not reference:
            raise ValueError("main_core_task_ref cannot be empty")
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE group_flow_windows SET main_core_task_ref = ?,
                version = version + 1, updated_at = ? WHERE window_id = ?
                AND profile_id = ? AND instance_id = ? AND status = 'RUNNING'
                AND version = ? AND lease_token = ?
                AND EXISTS (SELECT 1 FROM role_profiles profile
                    WHERE profile.profile_id = group_flow_windows.profile_id
                      AND profile.enabled = 1)
                AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                    WHERE chat_policy.profile_id = group_flow_windows.profile_id
                      AND chat_policy.instance_id = group_flow_windows.instance_id
                      AND chat_policy.soulcore_enabled = 0)""",
                (
                    reference,
                    _dt(now),
                    window.window_id,
                    window.profile_id,
                    window.instance_id,
                    window.version,
                    window.lease_token,
                ),
            ),
            transaction=True,
        )
        if cursor.rowcount != 1:
            return None
        current = await self.get_window(window.profile_id, window.instance_id, window.window_id)
        if current is None or current.frozen_through_message_id is None:
            return None
        return GroupRunFence(
            window_id=current.window_id,
            frozen_through_message_id=current.frozen_through_message_id,
            lease_token=current.lease_token,
            version=current.version,
            main_core_task_ref=reference,
        )

    async def release_ready(
        self,
        window: GroupFlowWindow,
        *,
        retry_at: datetime,
        reason: str,
    ) -> bool:
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE group_flow_windows SET status = 'READY', ready_at = ?,
                lease_owner = NULL, lease_until = NULL, lease_token = lease_token + 1,
                main_core_task_ref = NULL, error_code = ?, version = version + 1,
                updated_at = ? WHERE window_id = ? AND profile_id = ? AND instance_id = ?
                AND status = 'RUNNING' AND version = ? AND lease_token = ?
                AND EXISTS (SELECT 1 FROM role_profiles profile
                    WHERE profile.profile_id = group_flow_windows.profile_id
                      AND profile.enabled = 1)""",
                (
                    _dt(retry_at),
                    str(reason).strip()[:120],
                    _dt(datetime.now(retry_at.tzinfo)),
                    window.window_id,
                    window.profile_id,
                    window.instance_id,
                    window.version,
                    window.lease_token,
                ),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1

    async def _load_identities(
        self, identities: list[tuple[str, str, str]]
    ) -> tuple[GroupFlowWindow, ...]:
        result: list[GroupFlowWindow] = []
        for profile_id, instance_id, window_id in identities:
            window = await self.get_window(profile_id, instance_id, window_id)
            if window is not None:
                result.append(window)
        return tuple(result)


__all__ = ["GroupFlowClaimSql"]
