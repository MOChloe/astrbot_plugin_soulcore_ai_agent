from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from ....contracts.group_flow import GroupFlowWindow
from ....storage.sqlite.codec import _dt
from ..policy import (
    group_interjection_check_probability,
    stable_interjection_check_value,
)

_PIPELINE = "'JUDGING','READY','RUNNING','WAITING_FIRST_ATTEMPT'"


class GroupFlowJudgmentAdmissionSql:
    async def release_judgment(self, window: GroupFlowWindow, *, now: datetime) -> bool:
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE group_flow_windows SET status = 'COLLECTING',
                lease_owner = NULL, lease_until = NULL, lease_token = lease_token + 1,
                version = version + 1, updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND window_id = ?
                  AND status = 'JUDGING' AND version = ? AND lease_token = ?""",
                (
                    _dt(now),
                    window.profile_id,
                    window.instance_id,
                    window.window_id,
                    window.version,
                    window.lease_token,
                ),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1

    async def claim_judging_windows(
        self,
        *,
        now: datetime,
        worker_id: str,
        limit: int,
        lease_seconds: int,
    ) -> tuple[GroupFlowWindow, ...]:
        identities = await self._admit_judging_windows(
            now=now,
            worker_id=worker_id,
            limit=limit,
            lease_seconds=lease_seconds,
        )
        return await self._load_identities(identities)

    async def _admit_judging_windows(
        self,
        *,
        now: datetime,
        worker_id: str,
        limit: int,
        lease_seconds: int,
    ) -> list[tuple[str, str, str]]:
        owner = str(worker_id).strip()
        if not owner:
            raise ValueError("group-flow worker_id cannot be empty")
        now_text = _dt(now)
        lease_until = _dt(now + timedelta(seconds=max(1, int(lease_seconds))))
        assert now_text is not None and lease_until is not None

        def operation(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
            rows = self._judgment_candidates(
                conn,
                now_text=now_text,
                limit=max(0, int(limit)),
            )
            return self._apply_admission(
                conn,
                rows,
                owner=owner,
                lease_until=lease_until,
                now=now,
                now_text=now_text,
            )

        return await self.uow.run(operation)

    @staticmethod
    def _judgment_candidates(
        conn: sqlite3.Connection, *, now_text: str, limit: int
    ) -> list[sqlite3.Row]:
        return list(
            conn.execute(
                f"""SELECT window_id, profile_id, instance_id, last_message_id, rate_ewma
                FROM group_flow_windows window WHERE status = 'COLLECTING'
                AND direct_address = 0
                AND EXISTS (SELECT 1 FROM role_profiles profile
                    WHERE profile.profile_id = window.profile_id AND profile.enabled = 1)
                AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                    WHERE chat_policy.profile_id = window.profile_id
                      AND chat_policy.instance_id = window.instance_id
                      AND chat_policy.soulcore_enabled = 0)
                AND ((message_count >= judge_threshold AND next_judge_at <= ?)
                  OR (quiet_due_at IS NOT NULL AND quiet_due_at <= ?)
                  OR (dynamic_due_at IS NOT NULL AND dynamic_due_at <= ?))
                AND NOT EXISTS (
                    SELECT 1 FROM group_flow_instance_state state
                    JOIN group_flow_policies policy
                      ON policy.profile_id = state.profile_id AND policy.scope = 'group'
                    WHERE state.profile_id = window.profile_id
                      AND state.instance_id = window.instance_id
                      AND state.last_visible_assistant_at IS NOT NULL
                      AND julianday(state.last_visible_assistant_at)
                        + policy.ordinary_min_reply_gap_seconds / 86400.0 > julianday(?)
                )
                AND NOT EXISTS (
                    SELECT 1 FROM group_flow_instance_state judged
                    WHERE judged.profile_id = window.profile_id
                      AND judged.instance_id = window.instance_id
                      AND judged.last_judged_message_id >= window.last_message_id
                )
                AND NOT EXISTS (SELECT 1 FROM group_flow_windows active
                    WHERE active.profile_id = window.profile_id
                      AND active.instance_id = window.instance_id
                      AND active.status IN ({_PIPELINE}))
                ORDER BY next_judge_at, window_id LIMIT ?""",
                (now_text, now_text, now_text, now_text, limit),
            )
        )

    def _apply_admission(
        self,
        conn: sqlite3.Connection,
        rows: list[sqlite3.Row],
        *,
        owner: str,
        lease_until: str,
        now: datetime,
        now_text: str,
    ) -> list[tuple[str, str, str]]:
        claimed: list[tuple[str, str, str]] = []
        for row in rows:
            probability = group_interjection_check_probability(float(row["rate_ewma"]))
            sample = stable_interjection_check_value(
                str(row["window_id"]), int(row["last_message_id"])
            )
            if sample >= probability:
                self._silence_skipped_check(conn, row, now=now)
                continue
            if self._claim_judgment(conn, row, owner, lease_until, now_text):
                claimed.append(
                    (str(row["profile_id"]), str(row["instance_id"]), str(row["window_id"]))
                )
        return claimed

    @staticmethod
    def _claim_judgment(
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        owner: str,
        lease_until: str,
        now_text: str,
    ) -> bool:
        return bool(
            conn.execute(
                """UPDATE group_flow_windows SET status = 'JUDGING',
                judge_through_message_id = last_message_id,
                lease_owner = ?, lease_until = ?, lease_token = lease_token + 1,
                version = version + 1, updated_at = ?
                WHERE window_id = ? AND status = 'COLLECTING' AND direct_address = 0
                AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                    WHERE chat_policy.profile_id = group_flow_windows.profile_id
                      AND chat_policy.instance_id = group_flow_windows.instance_id
                      AND chat_policy.soulcore_enabled = 0)""",
                (owner, lease_until, now_text, row["window_id"]),
            ).rowcount
        )

    def _silence_skipped_check(
        self, conn: sqlite3.Connection, row: sqlite3.Row, *, now: datetime
    ) -> None:
        self._finish_window_sql(
            conn,
            str(row["profile_id"]),
            str(row["instance_id"]),
            str(row["window_id"]),
            from_statuses=("COLLECTING",),
            status="RESOLVED",
            outcome="INTERJECTION_CHECK_SKIPPED_BY_RATE",
            now=now,
            first_attempt=False,
        )


__all__ = ["GroupFlowJudgmentAdmissionSql"]
