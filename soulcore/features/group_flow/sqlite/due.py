from __future__ import annotations

from datetime import datetime

from ....storage.sqlite.codec import _parse


class GroupFlowDueSql:
    """Read-only due-time projection shared by the group-flow worker."""

    async def next_due_at(self) -> datetime | None:
        row = await self.db.fetch_one(self._next_due_sql())
        return _parse(row["due_at"]) if row and row["due_at"] else None

    @classmethod
    def _next_due_sql(cls) -> str:
        return (
            "SELECT MIN(value) AS due_at FROM ("
            + cls._ordinary_due_sql()
            + " UNION ALL "
            + cls._relocation_due_sql()
            + ")"
        )

    @staticmethod
    def _ordinary_due_sql() -> str:
        return """SELECT next_judge_at AS value FROM group_flow_windows window
            JOIN role_profiles profile ON profile.profile_id = window.profile_id
            WHERE profile.enabled = 1 AND status = 'COLLECTING'
              AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                WHERE chat_policy.profile_id = window.profile_id
                  AND chat_policy.instance_id = window.instance_id
                  AND chat_policy.soulcore_enabled = 0)
              AND next_judge_at IS NOT NULL
              AND message_count >= judge_threshold
              AND NOT EXISTS (
                SELECT 1 FROM group_flow_instance_state judged
                WHERE judged.profile_id = window.profile_id
                  AND judged.instance_id = window.instance_id
                  AND judged.last_judged_message_id >= window.last_message_id
              )
              AND NOT EXISTS (
                SELECT 1 FROM group_flow_windows active
                WHERE active.profile_id = window.profile_id
                  AND active.instance_id = window.instance_id
                  AND active.status IN ('JUDGING','READY','RUNNING','WAITING_FIRST_ATTEMPT')
              )
        UNION ALL SELECT quiet_due_at FROM group_flow_windows window
            JOIN role_profiles profile ON profile.profile_id = window.profile_id
            WHERE profile.enabled = 1 AND status = 'COLLECTING'
              AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                WHERE chat_policy.profile_id = window.profile_id
                  AND chat_policy.instance_id = window.instance_id
                  AND chat_policy.soulcore_enabled = 0)
              AND direct_address = 0 AND quiet_due_at IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM group_flow_windows active
                WHERE active.profile_id = window.profile_id
                  AND active.instance_id = window.instance_id
                  AND active.status IN ('JUDGING','READY','RUNNING','WAITING_FIRST_ATTEMPT')
              )
        UNION ALL SELECT dynamic_due_at FROM group_flow_windows window
            JOIN role_profiles profile ON profile.profile_id = window.profile_id
            WHERE profile.enabled = 1 AND status = 'COLLECTING'
              AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                WHERE chat_policy.profile_id = window.profile_id
                  AND chat_policy.instance_id = window.instance_id
                  AND chat_policy.soulcore_enabled = 0)
              AND direct_address = 0 AND dynamic_due_at IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM group_flow_windows active
                WHERE active.profile_id = window.profile_id
                  AND active.instance_id = window.instance_id
                  AND active.status IN ('JUDGING','READY','RUNNING','WAITING_FIRST_ATTEMPT')
              )
        UNION ALL SELECT direct_due_at FROM group_flow_windows window
            JOIN role_profiles profile ON profile.profile_id = window.profile_id
            WHERE profile.enabled = 1 AND status = 'COLLECTING'
              AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                WHERE chat_policy.profile_id = window.profile_id
                  AND chat_policy.instance_id = window.instance_id
                  AND chat_policy.soulcore_enabled = 0)
              AND direct_due_at IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM group_flow_windows active
                WHERE active.profile_id = window.profile_id
                  AND active.instance_id = window.instance_id
                  AND active.status IN ('JUDGING','READY','RUNNING','WAITING_FIRST_ATTEMPT')
              )
        UNION ALL SELECT ready_at FROM group_flow_windows window
            JOIN role_profiles profile ON profile.profile_id = window.profile_id
            WHERE profile.enabled = 1 AND status = 'READY' AND ready_at IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                WHERE chat_policy.profile_id = window.profile_id
                  AND chat_policy.instance_id = window.instance_id
                  AND chat_policy.soulcore_enabled = 0)
        UNION ALL SELECT lease_until FROM group_flow_windows window
            JOIN role_profiles profile ON profile.profile_id = window.profile_id
            WHERE profile.enabled = 1 AND status IN ('JUDGING','RUNNING')
              AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                WHERE chat_policy.profile_id = window.profile_id
                  AND chat_policy.instance_id = window.instance_id
                  AND chat_policy.soulcore_enabled = 0)
              AND lease_until IS NOT NULL"""

    @staticmethod
    def _relocation_due_sql() -> str:
        return """SELECT state.candidate_recheck_at
            FROM group_reply_relocation_states state
            JOIN group_flow_windows protected
              ON protected.window_id = state.window_id
            JOIN role_profiles profile
              ON profile.profile_id = state.profile_id
            WHERE profile.enabled = 1
              AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                WHERE chat_policy.profile_id = protected.profile_id
                  AND chat_policy.instance_id = protected.instance_id
                  AND chat_policy.soulcore_enabled = 0)
              AND protected.status IN ('RUNNING','WAITING_FIRST_ATTEMPT')
              AND state.relocation_count = 0
              AND state.check_owner IS NULL
              AND state.candidate_recheck_at IS NOT NULL
        UNION ALL SELECT CASE
              WHEN delta.direct_due_at IS NOT NULL THEN delta.direct_due_at
              WHEN delta.message_count >= delta.judge_threshold
                AND delta.next_judge_at IS NOT NULL
                AND delta.quiet_due_at IS NOT NULL
                THEN MIN(delta.next_judge_at, delta.quiet_due_at)
              WHEN delta.message_count >= delta.judge_threshold
                THEN delta.next_judge_at
              ELSE delta.quiet_due_at
            END
            FROM group_reply_relocation_states state
            JOIN group_flow_windows protected
              ON protected.window_id = state.window_id
            JOIN group_flow_windows delta
              ON delta.profile_id = protected.profile_id
             AND delta.instance_id = protected.instance_id
             AND delta.status = 'COLLECTING'
            JOIN role_profiles profile
              ON profile.profile_id = protected.profile_id
            WHERE profile.enabled = 1
              AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                WHERE chat_policy.profile_id = protected.profile_id
                  AND chat_policy.instance_id = protected.instance_id
                  AND chat_policy.soulcore_enabled = 0)
              AND protected.status IN ('RUNNING','WAITING_FIRST_ATTEMPT')
              AND state.relocation_count = 0
              AND state.check_owner IS NULL
              AND state.candidate_recheck_at IS NULL
              AND delta.last_message_id >
                COALESCE(state.last_checked_message_id, 0)"""


__all__ = ["GroupFlowDueSql"]
