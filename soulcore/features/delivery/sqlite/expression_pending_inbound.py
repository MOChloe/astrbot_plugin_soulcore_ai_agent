"""Authoritative pending-inbound fence shared by expression claim and recovery."""

from __future__ import annotations

import sqlite3


def group_first_attempt_is_protected(conn: sqlite3.Connection, expression_batch_id: str) -> bool:
    """Return the delivery-facing protection bit without importing group logic."""

    row = conn.execute(
        """SELECT 1 FROM instance_outbox outbox
        JOIN group_flow_windows window
          ON window.window_id = json_extract(outbox.payload_json, '$.group_window_id')
         AND window.profile_id = outbox.profile_id
         AND window.instance_id = outbox.instance_id
        WHERE outbox.expression_batch_id = ?
          AND window.status = 'WAITING_FIRST_ATTEMPT' LIMIT 1""",
        (str(expression_batch_id),),
    ).fetchone()
    return row is not None


PENDING_INBOUND_FROM_BATCH = """
    message.profile_id = batch.profile_id
    AND message.instance_id = batch.instance_id
    AND message.direction = 'INBOUND'
    AND message.role = 'user'
    AND message.delivery_status = 'RECEIVED'
    AND NOT EXISTS (
        SELECT 1 FROM group_flow_instance_state released_group
        WHERE released_group.profile_id = message.profile_id
          AND released_group.instance_id = message.instance_id
          AND released_group.activity_released_through_message_id IS NOT NULL
          AND message.message_id <= released_group.activity_released_through_message_id
    )
    AND message.created_at > batch.created_at
    AND NOT EXISTS (
        SELECT 1 FROM expression_interruption_events interruption
        WHERE interruption.batch_id = batch.batch_id
          AND interruption.inbound_message_id = message.message_id
    )
"""


def has_pending_expression_inbound(conn: sqlite3.Connection, expression_batch_id: str) -> bool:
    if group_first_attempt_is_protected(conn, expression_batch_id):
        return False
    row = conn.execute(
        f"""SELECT 1 FROM instance_expression_batches batch
        JOIN instance_messages message ON {PENDING_INBOUND_FROM_BATCH}
        WHERE batch.batch_id = ? AND batch.status = 'ACTIVE' LIMIT 1""",
        (str(expression_batch_id),),
    ).fetchone()
    return row is not None


def list_pending_expression_inbound(conn: sqlite3.Connection, *, limit: int) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            f"""SELECT DISTINCT message.profile_id, message.instance_id,
            message.message_id FROM instance_expression_batches batch
            JOIN instance_messages message ON {PENDING_INBOUND_FROM_BATCH}
            WHERE batch.status = 'ACTIVE'
              AND LOWER(COALESCE(
                json_extract(message.metadata_json, '$.scope'), ''
              )) != 'group'
              AND NOT EXISTS (
                SELECT 1 FROM group_flow_window_members held_member
                JOIN group_flow_windows held_window
                  ON held_window.window_id = held_member.window_id
                 AND held_window.profile_id = message.profile_id
                 AND held_window.instance_id = message.instance_id
                LEFT JOIN group_flow_instance_state held_state
                  ON held_state.profile_id = message.profile_id
                 AND held_state.instance_id = message.instance_id
                WHERE held_member.message_id = message.message_id
                  AND held_window.status IN ('COLLECTING', 'JUDGING')
                  AND message.message_id > COALESCE(
                    held_state.activity_released_through_message_id, 0
                  )
              )
              AND NOT EXISTS (
                SELECT 1 FROM instance_outbox protected_outbox
                JOIN group_flow_windows protected_window
                  ON protected_window.window_id = json_extract(
                    protected_outbox.payload_json, '$.group_window_id'
                  )
                 AND protected_window.profile_id = protected_outbox.profile_id
                 AND protected_window.instance_id = protected_outbox.instance_id
                WHERE protected_outbox.expression_batch_id = batch.batch_id
                  AND protected_window.status = 'WAITING_FIRST_ATTEMPT'
              )
            ORDER BY message.message_id LIMIT ?""",
            (max(1, min(int(limit), 1000)),),
        )
    )


__all__ = [
    "group_first_attempt_is_protected",
    "has_pending_expression_inbound",
    "list_pending_expression_inbound",
]
