from __future__ import annotations

import sqlite3
from collections.abc import Mapping

from ....contracts.group_flow import GroupFlowPolicy
from ....storage.sqlite.codec import _dt, _now, _parse

_POLICY_FIELDS = {
    "quiet_seconds",
    "base_message_count",
    "ordinary_min_reply_gap_seconds",
    "judge_token_budget",
}


class GroupFlowPolicySql:
    async def get_group_flow_policy(self, profile_id: str, scope: str = "group") -> GroupFlowPolicy:
        normalized = self._validate_scope(profile_id, scope)
        now = _dt(_now())
        await self.db.call(
            lambda conn: conn.execute(
                """INSERT OR IGNORE INTO group_flow_policies(
                profile_id, scope, created_at, updated_at) VALUES (?, 'group', ?, ?)""",
                (normalized, now, now),
            ),
            transaction=True,
        )
        row = await self.db.fetch_one(
            "SELECT * FROM group_flow_policies WHERE profile_id = ? AND scope = 'group'",
            (normalized,),
        )
        if row is None:
            raise KeyError(normalized)
        return self._policy(row)

    async def update_group_flow_policy(
        self,
        profile_id: str,
        scope: str,
        patch: Mapping[str, object],
        *,
        expected_version: int,
    ) -> GroupFlowPolicy:
        current = await self.get_group_flow_policy(profile_id, scope)
        values = dict(patch)
        unknown = set(values) - _POLICY_FIELDS
        if unknown:
            raise ValueError(f"unknown group flow policy fields: {sorted(unknown)}")
        candidate = GroupFlowPolicy(
            profile_id=current.profile_id,
            scope="group",
            quiet_seconds=int(values.get("quiet_seconds", current.quiet_seconds)),
            base_message_count=int(values.get("base_message_count", current.base_message_count)),
            ordinary_min_reply_gap_seconds=int(
                values.get(
                    "ordinary_min_reply_gap_seconds",
                    current.ordinary_min_reply_gap_seconds,
                )
            ),
            judge_token_budget=int(values.get("judge_token_budget", current.judge_token_budget)),
            version=current.version + 1,
            created_at=current.created_at,
            updated_at=_now(),
        )
        cursor = await self.db.call(
            lambda conn: self._update_policy_sql(
                conn,
                candidate,
                expected_version=int(expected_version),
            ),
            transaction=True,
        )
        if cursor.rowcount != 1:
            raise ValueError("group flow policy version conflict")
        return await self.get_group_flow_policy(profile_id, scope)

    @staticmethod
    def _update_policy_sql(
        conn: sqlite3.Connection,
        value: GroupFlowPolicy,
        *,
        expected_version: int,
    ) -> sqlite3.Cursor:
        return conn.execute(
            """UPDATE group_flow_policies SET quiet_seconds = ?,
            base_message_count = ?, ordinary_min_reply_gap_seconds = ?,
            judge_token_budget = ?, version = version + 1, updated_at = ?
            WHERE profile_id = ? AND scope = 'group' AND version = ?""",
            (
                value.quiet_seconds,
                value.base_message_count,
                value.ordinary_min_reply_gap_seconds,
                value.judge_token_budget,
                _dt(value.updated_at),
                value.profile_id,
                expected_version,
            ),
        )

    @staticmethod
    def _validate_scope(profile_id: str, scope: str) -> str:
        normalized = str(profile_id).strip()
        if not normalized:
            raise ValueError("profile_id cannot be empty")
        if str(scope).strip().lower() != "group":
            raise ValueError("group flow policy only supports group scope")
        return normalized

    @staticmethod
    def _policy(row: sqlite3.Row) -> GroupFlowPolicy:
        return GroupFlowPolicy(
            profile_id=str(row["profile_id"]),
            scope=str(row["scope"]),
            quiet_seconds=int(row["quiet_seconds"]),
            base_message_count=int(row["base_message_count"]),
            ordinary_min_reply_gap_seconds=int(row["ordinary_min_reply_gap_seconds"]),
            judge_token_budget=int(row["judge_token_budget"]),
            version=int(row["version"]),
            created_at=_parse(row["created_at"]),
            updated_at=_parse(row["updated_at"]),
        )


__all__ = ["GroupFlowPolicySql"]
