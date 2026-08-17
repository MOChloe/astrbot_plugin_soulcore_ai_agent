from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from ....contracts.group_flow import (
    GroupFlowStatus,
    GroupReplyRelocationCheck,
    GroupReplyRelocationResult,
)
from ....storage.sqlite.codec import _dt
from ....storage.sqlite.expression_batch_lifecycle import (
    cancel_pending_expression_row,
    sync_expression_batch_status,
)


def _load_relocation_rows(
    conn: sqlite3.Connection, check: GroupReplyRelocationCheck
) -> tuple[sqlite3.Row | None, sqlite3.Row | None, sqlite3.Row | None]:
    protected = conn.execute(
        """SELECT * FROM group_flow_windows WHERE window_id = ?
        AND profile_id = ? AND instance_id = ?
        AND status IN ('RUNNING','WAITING_FIRST_ATTEMPT')
        AND version = ? AND lease_token = ?
        AND frozen_through_message_id = ? AND main_core_task_ref = ?
        AND first_attempt_started_at IS NULL""",
        (
            check.fence.window_id,
            check.profile_id,
            check.instance_id,
            check.fence.version,
            check.fence.lease_token,
            check.fence.frozen_through_message_id,
            check.fence.main_core_task_ref,
        ),
    ).fetchone()
    state = conn.execute(
        """SELECT * FROM group_reply_relocation_states WHERE window_id = ?
        AND check_token = ? AND check_owner IS NOT NULL
        AND check_through_message_id = ? AND relocation_count = 0""",
        (check.fence.window_id, check.check_token, check.delta_through_message_id),
    ).fetchone()
    delta = conn.execute(
        """SELECT * FROM group_flow_windows WHERE window_id = ?
        AND profile_id = ? AND instance_id = ? AND status = 'COLLECTING'
        AND last_message_id = ?""",
        (
            check.delta_window_id,
            check.profile_id,
            check.instance_id,
            check.delta_through_message_id,
        ),
    ).fetchone()
    return protected, state, delta


def _runtime_is_enabled(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
) -> bool:
    row = conn.execute(
        """SELECT profile.enabled,
        COALESCE(chat_policy.soulcore_enabled, 1) AS instance_enabled
        FROM role_profiles profile
        LEFT JOIN instance_chat_policies chat_policy
          ON chat_policy.profile_id = profile.profile_id
         AND chat_policy.instance_id = ?
        WHERE profile.profile_id = ?""",
        (instance_id, profile_id),
    ).fetchone()
    return row is not None and bool(row["enabled"]) and bool(row["instance_enabled"])


def _cancel_waiting_first_attempt(
    conn: sqlite3.Connection,
    check: GroupReplyRelocationCheck,
    protected: sqlite3.Row,
    now_text: str,
) -> str:
    if str(protected["status"]) != GroupFlowStatus.WAITING_FIRST_ATTEMPT.value:
        return ""
    active_rows = list(
        conn.execute(
            """SELECT * FROM instance_outbox WHERE profile_id = ? AND instance_id = ?
            AND json_extract(payload_json, '$.group_window_id') = ?
            AND status IN ('PENDING','SENDING')""",
            (check.profile_id, check.instance_id, check.fence.window_id),
        )
    )
    if not active_rows or any(str(row["status"]) != "PENDING" for row in active_rows):
        return "first_platform_attempt_started"
    batch_ids = {
        str(row["expression_batch_id"]) for row in active_rows if row["expression_batch_id"]
    }
    for row in active_rows:
        cancel_pending_expression_row(
            conn,
            row,
            reason="group_reply_relocated_before_first_attempt",
            now=now_text,
        )
    for batch_id in batch_ids:
        conn.execute(
            """UPDATE message_retraction_actions SET status = 'CANCELLED',
            error_code = 'group_reply_relocated_before_first_attempt', updated_at = ?
            WHERE expression_batch_id = ? AND status = 'PENDING'""",
            (now_text, batch_id),
        )
        sync_expression_batch_status(conn, batch_id, now_text)
    return ""


def _relocate_delta_members(
    conn: sqlite3.Connection, check: GroupReplyRelocationCheck
) -> sqlite3.Row | None:
    moved = list(
        conn.execute(
            """SELECT * FROM group_flow_window_members WHERE window_id = ? ORDER BY ordinal""",
            (check.delta_window_id,),
        )
    )
    if not moved:
        return None
    next_ordinal = int(
        conn.execute(
            """SELECT COALESCE(MAX(ordinal), -1) + 1
            FROM group_flow_window_members WHERE window_id = ?""",
            (check.fence.window_id,),
        ).fetchone()[0]
    )
    conn.execute("DELETE FROM group_flow_windows WHERE window_id = ?", (check.delta_window_id,))
    for row in moved:
        conn.execute(
            """INSERT INTO group_flow_window_members(
            window_id, profile_id, instance_id, message_id, ordinal,
            normalized_fingerprint, media_cluster_keys_json, sender_id, occurred_at, added_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                check.fence.window_id,
                check.profile_id,
                check.instance_id,
                row["message_id"],
                next_ordinal,
                row["normalized_fingerprint"],
                row["media_cluster_keys_json"],
                row["sender_id"],
                row["occurred_at"],
                row["added_at"],
            ),
        )
        next_ordinal += 1
    return conn.execute(
        """SELECT COUNT(*) AS total, COUNT(DISTINCT normalized_fingerprint) AS unique_count,
        MIN(message_id) AS first_id, MAX(message_id) AS last_id
        FROM group_flow_window_members WHERE window_id = ?""",
        (check.fence.window_id,),
    ).fetchone()


def _update_relocated_window(
    conn: sqlite3.Connection,
    check: GroupReplyRelocationCheck,
    protected: sqlite3.Row,
    delta: sqlite3.Row,
    stats: sqlite3.Row,
    now: datetime,
    now_text: str,
) -> None:
    total = int(stats["total"])
    repeat_ratio = max(0.0, (total - int(stats["unique_count"])) / total)
    policy = conn.execute(
        "SELECT quiet_seconds FROM group_flow_policies WHERE profile_id = ? AND scope = 'group'",
        (check.profile_id,),
    ).fetchone()
    replacement_check = _dt(now + timedelta(seconds=5))
    replacement_quiet = _dt(now + timedelta(seconds=int(policy["quiet_seconds"]) if policy else 30))
    conn.execute(
        """UPDATE group_flow_windows SET status = 'COLLECTING',
        first_message_id = ?, last_message_id = ?, message_count = ?, rate_ewma = ?,
        repeat_ratio = ?, judge_threshold = ?, judge_through_message_id = NULL,
        frozen_through_message_id = NULL, next_judge_at = ?, quiet_due_at = ?,
        dynamic_due_at = CASE WHEN dynamic_due_at IS NULL OR dynamic_due_at < ?
            THEN ? ELSE dynamic_due_at END, direct_due_at = NULL, direct_address = ?,
        judge_result = '', judge_error_code = '', ready_at = NULL, lease_owner = NULL,
        lease_until = NULL, lease_token = lease_token + 1, main_core_task_ref = NULL,
        error_code = 'group_reply_relocated', resolution_outcome = '', resolved_at = NULL,
        version = version + 1, updated_at = ? WHERE window_id = ?""",
        (
            int(stats["first_id"]),
            int(stats["last_id"]),
            total,
            float(delta["rate_ewma"]),
            repeat_ratio,
            max(int(delta["judge_threshold"]), total),
            replacement_check,
            replacement_quiet,
            replacement_quiet,
            replacement_quiet,
            int(bool(protected["direct_address"]) or bool(delta["direct_address"])),
            now_text,
            check.fence.window_id,
        ),
    )


def _mark_relocation_applied(
    conn: sqlite3.Connection, check: GroupReplyRelocationCheck, now_text: str
) -> None:
    conn.execute(
        """UPDATE group_reply_relocation_states SET relocation_count = 1,
        last_checked_message_id = ?, candidate_through_message_id = NULL,
        candidate_recheck_at = NULL, check_owner = NULL, check_until = NULL,
        check_through_message_id = NULL, check_final = 0, last_error_code = '', updated_at = ?
        WHERE window_id = ?""",
        (check.delta_through_message_id, now_text, check.fence.window_id),
    )


def _apply_relocation(
    conn: sqlite3.Connection, check: GroupReplyRelocationCheck, now: datetime, now_text: str
) -> tuple[bool, str, str]:
    protected, state, delta = _load_relocation_rows(conn, check)
    if protected is None or state is None or delta is None:
        return False, "", "stale_snapshot"
    if not _runtime_is_enabled(conn, check.profile_id, check.instance_id):
        return False, "", "runtime_disabled"
    previous_status = str(protected["status"])
    if reason := _cancel_waiting_first_attempt(conn, check, protected, now_text):
        return False, "", reason
    stats = _relocate_delta_members(conn, check)
    if stats is None:
        return False, "", "delta_empty"
    _update_relocated_window(conn, check, protected, delta, stats, now, now_text)
    _mark_relocation_applied(conn, check, now_text)
    return True, previous_status, ""


class GroupReplyRelocationApplySql:
    async def apply_reply_relocation(
        self, check: GroupReplyRelocationCheck, *, now: datetime
    ) -> GroupReplyRelocationResult:
        now_text = _dt(now)
        assert now_text is not None
        applied, status, reason = await self.uow.run(
            lambda conn: _apply_relocation(conn, check, now, now_text)
        )
        return GroupReplyRelocationResult(
            applied=bool(applied),
            previous_status=GroupFlowStatus(status) if status else None,
            reason=reason,
        )


__all__ = ["GroupReplyRelocationApplySql"]
