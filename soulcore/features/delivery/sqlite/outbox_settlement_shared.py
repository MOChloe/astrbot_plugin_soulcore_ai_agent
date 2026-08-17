"""Atomic Outbox and foreground delivery settlement commands."""

from __future__ import annotations

from ....storage.sqlite.expression_batch_lifecycle import (
    cancel_pending_expression_row,
)
from .expression_interruption_cleanup import restore_cancelled_file_todos
from .group_first_attempt import ResolveUndeliverableGroupWindow
from .support import (
    OutboxInterruptPolicy,
    OutboxStatus,
    _load,
    sqlite3,
)
from .voice_artifacts import schedule_outbox_voice_artifact_cleanup_sql


def _cancel_terminal_expression_suffix(
    conn: sqlite3.Connection,
    current: sqlite3.Row,
    status: OutboxStatus,
    now: str,
) -> None:
    if (
        status not in {OutboxStatus.FAILED, OutboxStatus.CANCELLED}
        or current["expression_batch_id"] is None
        or current["expression_ordinal"] is None
    ):
        return
    reason = (
        "cancelled_after_failed_expression_predecessor"
        if status is OutboxStatus.FAILED
        else "cancelled_after_cancelled_expression_predecessor"
    )
    suffix = list(
        conn.execute(
            """SELECT * FROM instance_outbox
            WHERE expression_batch_id = ? AND expression_ordinal > ?
              AND status = ? AND attempts = 0 AND interrupt_policy = ?""",
            (
                str(current["expression_batch_id"]),
                int(current["expression_ordinal"]),
                OutboxStatus.PENDING.value,
                OutboxInterruptPolicy.CANCEL_ON_PLAYER_MESSAGE.value,
            ),
        )
    )
    cancelled = [
        row for row in suffix if cancel_pending_expression_row(conn, row, reason=reason, now=now)
    ]
    for row in cancelled:
        schedule_outbox_voice_artifact_cleanup_sql(
            conn,
            profile_id=str(row["profile_id"]),
            instance_id=str(row["instance_id"]),
            outbox_id=int(row["outbox_id"]),
            reason="voice_expression_suffix_cancelled",
            now=now,
        )
    restore_cancelled_file_todos(
        conn,
        str(current["profile_id"]),
        str(current["instance_id"]),
        cancelled,
        now,
        reason,
        load_payload=_load,
    )


def _resolve_terminal_group_window(
    conn: sqlite3.Connection,
    current: sqlite3.Row,
    status: OutboxStatus,
    now: str,
) -> bool:
    if status not in {OutboxStatus.FAILED, OutboxStatus.CANCELLED}:
        return False
    payload = _load(current["payload_json"]) or {}
    group_window_id = (
        str(payload.get("group_window_id") or "").strip() if isinstance(payload, dict) else ""
    )
    if not group_window_id:
        return False
    return ResolveUndeliverableGroupWindow(
        str(current["profile_id"]),
        str(current["instance_id"]),
        group_window_id,
        now,
    )(conn)


__all__ = ["_cancel_terminal_expression_suffix", "_resolve_terminal_group_window"]
