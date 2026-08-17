"""Atomic removal of a recalled message from active admission workflows."""

from __future__ import annotations

import json
import sqlite3

from ....storage.sqlite.expression_batch_lifecycle import (
    cancel_pending_expression_row,
    sync_expression_batch_status,
)
from ....storage.sqlite.recall_file_transactions import RecallFileTransactions
from ..domain import InboundRecallHold


def invalidate_active_admission(
    conn: sqlite3.Connection,
    hold: InboundRecallHold,
    now_text: str,
    file_transactions: RecallFileTransactions,
) -> None:
    _invalidate_deferred_batches(conn, hold, now_text)
    _invalidate_turn_buffers(conn, hold, now_text)
    _invalidate_group_windows(conn, hold, now_text, file_transactions)


def _invalidate_deferred_batches(
    conn: sqlite3.Connection,
    hold: InboundRecallHold,
    now_text: str,
) -> None:
    batch_ids = [
        str(row["batch_id"])
        for row in conn.execute(
            """SELECT batch.batch_id FROM deferred_message_batches batch
            JOIN deferred_message_items item ON item.batch_id = batch.batch_id
            WHERE item.profile_id = ? AND item.instance_id = ?
              AND item.message_id = ?
              AND batch.status IN ('PENDING','CLAIMED')""",
            (hold.profile_id, hold.instance_id, hold.ledger_message_id),
        )
    ]
    for batch_id in batch_ids:
        conn.execute(
            """DELETE FROM deferred_message_items
            WHERE batch_id = ? AND message_id = ?""",
            (batch_id, hold.ledger_message_id),
        )
        remaining = int(
            conn.execute(
                "SELECT COUNT(*) FROM deferred_message_items WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()[0]
        )
        if remaining:
            conn.execute(
                """UPDATE deferred_message_batches SET status = 'PENDING',
                lease_until = NULL, lease_token = lease_token + 1,
                version = version + 1, resolution_reason = 'member_recalled',
                updated_at = ? WHERE batch_id = ?""",
                (now_text, batch_id),
            )
        else:
            conn.execute(
                """UPDATE deferred_message_batches SET status = 'CANCELLED',
                lease_until = NULL, lease_token = lease_token + 1,
                version = version + 1, resolution_reason = 'member_recalled',
                resolved_at = ?, updated_at = ? WHERE batch_id = ?""",
                (now_text, now_text, batch_id),
            )


def _invalidate_turn_buffers(
    conn: sqlite3.Connection,
    hold: InboundRecallHold,
    now_text: str,
) -> None:
    batch_ids = [
        str(row["batch_id"])
        for row in conn.execute(
            """SELECT batch.batch_id FROM conversation_turn_buffer_batches batch
            JOIN conversation_turn_buffer_members member
              ON member.batch_id = batch.batch_id
            WHERE member.profile_id = ? AND member.instance_id = ?
              AND member.message_id = ?
              AND batch.status IN ('PENDING','CLASSIFYING','WAITING','CLAIMED')""",
            (hold.profile_id, hold.instance_id, hold.ledger_message_id),
        )
    ]
    for batch_id in batch_ids:
        conn.execute(
            """DELETE FROM conversation_turn_buffer_members
            WHERE batch_id = ? AND message_id = ?""",
            (batch_id, hold.ledger_message_id),
        )
        remaining = int(
            conn.execute(
                """SELECT COUNT(*) FROM conversation_turn_buffer_members
                WHERE batch_id = ?""",
                (batch_id,),
            ).fetchone()[0]
        )
        if remaining:
            conn.execute(
                """UPDATE conversation_turn_buffer_batches SET status = 'PENDING',
                generation = generation + 1, requested_delay_seconds = NULL,
                ai_elapsed_seconds = NULL, remaining_delay_seconds = NULL,
                due_at = NULL, lease_owner = NULL, lease_until = NULL,
                lease_token = lease_token + 1, main_core_task_ref = NULL,
                error_code = 'member_recalled', version = version + 1, updated_at = ?
                WHERE batch_id = ?""",
                (now_text, batch_id),
            )
        else:
            conn.execute(
                """UPDATE conversation_turn_buffer_batches SET status = 'CANCELLED',
                due_at = NULL, lease_owner = NULL, lease_until = NULL,
                lease_token = lease_token + 1, main_core_task_ref = NULL,
                resolution_outcome = 'MEMBER_RECALLED', version = version + 1,
                updated_at = ?, resolved_at = ? WHERE batch_id = ?""",
                (now_text, now_text, batch_id),
            )


def _invalidate_group_windows(
    conn: sqlite3.Connection,
    hold: InboundRecallHold,
    now_text: str,
    file_transactions: RecallFileTransactions,
) -> None:
    windows = [
        (str(row["window_id"]), str(row["status"]))
        for row in conn.execute(
            """SELECT window.window_id, window.status FROM group_flow_windows window
            JOIN group_flow_window_members member
              ON member.window_id = window.window_id
            WHERE member.profile_id = ? AND member.instance_id = ?
              AND member.message_id = ? AND window.status IN (
                'COLLECTING','JUDGING','READY','RUNNING','WAITING_FIRST_ATTEMPT'
              )""",
            (hold.profile_id, hold.instance_id, hold.ledger_message_id),
        )
    ]
    for window_id, status in windows:
        conn.execute(
            """DELETE FROM group_flow_window_members
            WHERE window_id = ? AND message_id = ?""",
            (window_id, hold.ledger_message_id),
        )
        bounds = conn.execute(
            """SELECT MIN(message_id) AS first_id, MAX(message_id) AS last_id,
                COUNT(*) AS count FROM group_flow_window_members WHERE window_id = ?""",
            (window_id,),
        ).fetchone()
        if status == "WAITING_FIRST_ATTEMPT":
            _cancel_unattempted_group_outbox(
                conn,
                hold,
                window_id,
                now_text,
                file_transactions,
            )
        if int(bounds["count"]):
            _reopen_group_window(conn, window_id, bounds, now_text)
        else:
            conn.execute(
                """UPDATE group_flow_windows SET status = 'CANCELLED',
                next_judge_at = NULL, quiet_due_at = NULL, dynamic_due_at = NULL,
                direct_due_at = NULL, lease_owner = NULL, lease_until = NULL,
                lease_token = lease_token + 1, main_core_task_ref = NULL,
                resolution_outcome = 'MEMBER_RECALLED', version = version + 1,
                updated_at = ?, resolved_at = ? WHERE window_id = ?""",
                (now_text, now_text, window_id),
            )


def _cancel_unattempted_group_outbox(
    conn: sqlite3.Connection,
    hold: InboundRecallHold,
    window_id: str,
    now_text: str,
    file_transactions: RecallFileTransactions,
) -> None:
    rows = list(
        conn.execute(
            """SELECT outbox.* FROM instance_outbox outbox
            WHERE outbox.profile_id = ? AND outbox.instance_id = ?
              AND json_extract(outbox.payload_json, '$.group_window_id') = ?
              AND outbox.status IN ('PENDING','SENDING')
              AND NOT EXISTS (
                SELECT 1 FROM platform_send_permits permit
                WHERE permit.origin_kind = 'EXPRESSION_ITEM'
                  AND permit.origin_id = 'expression-outbox:' || outbox.outbox_id
                  AND permit.status IN ('DISPATCHING','ATTEMPTED_UNKNOWN')
              )""",
            (hold.profile_id, hold.instance_id, window_id),
        )
    )
    cancelled: list[sqlite3.Row] = []
    batch_ids: set[str] = set()
    for row in rows:
        if str(row["status"]) == "SENDING":
            changed = conn.execute(
                """UPDATE instance_outbox SET status = 'PENDING', lease_until = NULL,
                lease_token = lease_token + 1, version = version + 1, updated_at = ?
                WHERE outbox_id = ? AND status = 'SENDING' AND NOT EXISTS (
                    SELECT 1 FROM platform_send_permits permit
                    WHERE permit.origin_kind = 'EXPRESSION_ITEM'
                      AND permit.origin_id = 'expression-outbox:' || instance_outbox.outbox_id
                      AND permit.status IN ('DISPATCHING','ATTEMPTED_UNKNOWN')
                )""",
                (now_text, row["outbox_id"]),
            ).rowcount
            if changed != 1:
                continue
            row = conn.execute(
                "SELECT * FROM instance_outbox WHERE outbox_id = ?",
                (row["outbox_id"],),
            ).fetchone()
            if row is None:
                continue
        if cancel_pending_expression_row(
            conn,
            row,
            reason="group_member_recalled_before_first_attempt",
            now=now_text,
        ):
            cancelled.append(row)
            if row["expression_batch_id"]:
                batch_ids.add(str(row["expression_batch_id"]))
    file_transactions.restore(
        conn,
        hold.profile_id,
        hold.instance_id,
        cancelled,
        now_text,
        "group_member_recalled_before_first_attempt",
        load_payload=_load_payload,
    )
    for batch_id in batch_ids:
        sync_expression_batch_status(conn, batch_id, now_text)


def _load_payload(value: object) -> dict[str, object]:
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _reopen_group_window(
    conn: sqlite3.Connection,
    window_id: str,
    bounds: sqlite3.Row,
    now_text: str,
) -> None:
    conn.execute(
        """UPDATE group_flow_windows SET status = 'COLLECTING',
        first_message_id = ?, last_message_id = ?, message_count = ?,
        judge_through_message_id = NULL, frozen_through_message_id = NULL,
        next_judge_at = ?, quiet_due_at = ?, dynamic_due_at = ?,
        judge_result = '', judge_error_code = '', ready_at = NULL,
        lease_owner = NULL, lease_until = NULL, lease_token = lease_token + 1,
        main_core_task_ref = NULL, first_attempt_started_at = NULL,
        error_code = 'member_recalled', version = version + 1, updated_at = ?
        WHERE window_id = ?""",
        (
            int(bounds["first_id"]),
            int(bounds["last_id"]),
            int(bounds["count"]),
            now_text,
            now_text,
            now_text,
            now_text,
            window_id,
        ),
    )


__all__ = ["invalidate_active_admission"]
