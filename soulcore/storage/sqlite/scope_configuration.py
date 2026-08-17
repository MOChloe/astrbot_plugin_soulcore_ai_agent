from __future__ import annotations

import sqlite3

from ...features.profiles.ports import (
    ScopeConfigurationConflict,
    ScopeConfigurationUpdate,
)
from .codec import _dt, _now
from .repository import SqliteRepository

_CONFLICT_MESSAGES = {
    "scope": "scope configuration changed; reload before saving",
    "contact": "contact policy changed; reload before saving",
    "timezone": "profile timezone changed; reload before saving",
    "delivery": "delivery policy changed; reload before saving",
    "state_gate": "state gate policy changed; reload before saving",
    "group_flow": "group flow policy changed; reload before saving",
}


class ScopeConfigurationCommandRepository(SqliteRepository):
    """Atomically replace one scope template and all of its policy rows."""

    async def save_scope_configuration(self, update: ScopeConfigurationUpdate) -> None:
        if update.scope not in {"private", "group"}:
            raise ValueError("scope must be private or group")
        if update.scope == "group":
            if update.group_flow is None or update.expected_group_flow_version is None:
                raise ValueError("group flow policy and revision are required for group scope")
        elif update.group_flow is not None or update.expected_group_flow_version is not None:
            raise ValueError("private scope cannot include a group flow policy")

        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> None:
            self._update_scope(conn, update, now)
            self._update_contact(conn, update, now)
            self._update_timezone(conn, update, now)
            self._update_delivery(conn, update, now)
            self._update_state_gate(conn, update, now)
            if update.scope == "group":
                self._update_group_flow(conn, update, now)

        await self.uow.run(operation)
        await self.db.publish_backup_after_commit()

    @classmethod
    def _expect_updated(cls, cursor: sqlite3.Cursor, resource: str) -> None:
        if cursor.rowcount != 1:
            raise ScopeConfigurationConflict(_CONFLICT_MESSAGES[resource])

    @classmethod
    def _update_scope(
        cls,
        conn: sqlite3.Connection,
        update: ScopeConfigurationUpdate,
        now: str,
    ) -> None:
        role = update.role
        cursor = conn.execute(
            """UPDATE scope_configs SET
                proactive_enabled = ?, extra_background = ?, world_texture_prompt = ?,
                media_original_retention_days = ?, min_wakeup_minutes = ?,
                max_wakeup_minutes = ?, low_frequency_min_wakeup_minutes = ?,
                low_frequency_max_wakeup_minutes = ?, max_context_tokens = ?,
                target_context_tokens = ?, version = version + 1, updated_at = ?
            WHERE profile_id = ? AND scope = ? AND version = ?""",
            (
                int(bool(role["proactive_enabled"])),
                str(role["extra_background"]),
                str(role["world_texture_prompt"]),
                int(role["media_original_retention_days"]),
                int(role["min_wakeup_minutes"]),
                int(role["max_wakeup_minutes"]),
                int(role["low_frequency_min_wakeup_minutes"]),
                int(role["low_frequency_max_wakeup_minutes"]),
                int(role["max_context_tokens"]),
                int(role["target_context_tokens"]),
                now,
                update.profile_id,
                update.scope,
                int(update.expected_scope_version),
            ),
        )
        cls._expect_updated(cursor, "scope")

    @classmethod
    def _update_contact(
        cls,
        conn: sqlite3.Connection,
        update: ScopeConfigurationUpdate,
        now: str,
    ) -> None:
        contact = update.contact
        cursor = conn.execute(
            """UPDATE contact_policies SET
                proactive_enabled = ?, check_min_minutes = ?, check_max_minutes = ?,
                quiet_enabled = ?, quiet_start = ?, quiet_end = ?,
                min_success_gap_minutes = ?, daily_success_limit = ?,
                max_consecutive_unanswered = ?, failure_mode = ?,
                retry_delay_minutes = ?, retry_max_attempts = ?,
                version = version + 1, updated_at = ?
            WHERE profile_id = ? AND scope = ? AND version = ?""",
            (
                int(bool(contact["proactive_enabled"])),
                int(contact["check_min_minutes"]),
                int(contact["check_max_minutes"]),
                int(bool(contact["quiet_enabled"])),
                str(contact["quiet_start"]),
                str(contact["quiet_end"]),
                int(contact["min_success_gap_minutes"]),
                (
                    None
                    if contact["daily_success_limit"] is None
                    else int(contact["daily_success_limit"])
                ),
                (
                    None
                    if contact["max_consecutive_unanswered"] is None
                    else int(contact["max_consecutive_unanswered"])
                ),
                str(contact["failure_mode"]),
                int(contact["retry_delay_minutes"]),
                int(contact["retry_max_attempts"]),
                now,
                update.profile_id,
                update.scope,
                int(update.expected_contact_version),
            ),
        )
        cls._expect_updated(cursor, "contact")

    @classmethod
    def _update_timezone(
        cls,
        conn: sqlite3.Connection,
        update: ScopeConfigurationUpdate,
        now: str,
    ) -> None:
        cursor = conn.execute(
            """UPDATE profile_runtime_settings SET timezone = ?,
                version = version + 1, updated_at = ?
            WHERE profile_id = ? AND version = ?""",
            (
                str(update.timezone or "").strip(),
                now,
                update.profile_id,
                int(update.expected_timezone_version),
            ),
        )
        cls._expect_updated(cursor, "timezone")

    @classmethod
    def _update_delivery(
        cls,
        conn: sqlite3.Connection,
        update: ScopeConfigurationUpdate,
        now: str,
    ) -> None:
        delivery = update.delivery
        cursor = conn.execute(
            """UPDATE scope_delivery_policies SET
                group_send_qpm_limit = ?, send_qpm_limit = ?,
                version = version + 1, updated_at = ?
            WHERE profile_id = ? AND scope = ? AND version = ?""",
            (
                int(delivery["group_send_qpm_limit"]),
                int(delivery["send_qpm_limit"]),
                now,
                update.profile_id,
                update.scope,
                int(update.expected_delivery_version),
            ),
        )
        cls._expect_updated(cursor, "delivery")

    @classmethod
    def _update_state_gate(
        cls,
        conn: sqlite3.Connection,
        update: ScopeConfigurationUpdate,
        now: str,
    ) -> None:
        state_gate = update.state_gate
        cursor = conn.execute(
            """UPDATE scope_state_gate_policies SET enabled = ?, silent_enabled = ?,
                max_gate_hours = ?, version = version + 1, updated_at = ?
            WHERE profile_id = ? AND scope = ? AND version = ?""",
            (
                int(bool(state_gate["enabled"])),
                int(bool(state_gate["silent_enabled"])),
                int(state_gate["max_gate_hours"]),
                now,
                update.profile_id,
                update.scope,
                int(update.expected_state_gate_version),
            ),
        )
        cls._expect_updated(cursor, "state_gate")

    @classmethod
    def _update_group_flow(
        cls,
        conn: sqlite3.Connection,
        update: ScopeConfigurationUpdate,
        now: str,
    ) -> None:
        assert update.group_flow is not None
        assert update.expected_group_flow_version is not None
        group_flow = update.group_flow
        cursor = conn.execute(
            """UPDATE group_flow_policies SET quiet_seconds = ?,
                base_message_count = ?, ordinary_min_reply_gap_seconds = ?,
                judge_token_budget = ?, version = version + 1, updated_at = ?
            WHERE profile_id = ? AND scope = 'group' AND version = ?""",
            (
                int(group_flow["quiet_seconds"]),
                int(group_flow["base_message_count"]),
                int(group_flow["ordinary_min_reply_gap_seconds"]),
                int(group_flow["judge_token_budget"]),
                now,
                update.profile_id,
                int(update.expected_group_flow_version),
            ),
        )
        cls._expect_updated(cursor, "group_flow")


__all__ = ["ScopeConfigurationCommandRepository"]
