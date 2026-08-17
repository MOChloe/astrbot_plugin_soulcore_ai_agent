"""Shared SQLite transaction step for confirming an autonomous contact."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import datetime

from ....storage.sqlite.codec import _dt
from ..contact_models import contact_day_bucket_transition
from .support import _contact_day_bucket

KnowledgeRefresh = Callable[..., sqlite3.Row | None]


def mark_latest_contact_attempt_answered_sql(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    *,
    player_message_id: int,
    now: datetime,
    refresh_knowledge_task: KnowledgeRefresh,
) -> sqlite3.Row | None:
    """Confirm at most one delivered contact inside the caller's transaction."""

    point = _dt(now)
    message = conn.execute(
        """SELECT direction, occurred_at FROM instance_messages
        WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
        (profile_id, instance_id, int(player_message_id)),
    ).fetchone()
    if message is None or message["direction"] != "INBOUND":
        raise ValueError("contact answer must reference a real inbound message")
    row = conn.execute(
        """SELECT * FROM contact_attempts WHERE profile_id = ?
        AND instance_id = ? AND status = 'FINALIZED' AND attempted = 1
        AND COALESCE(answered, 0) = 0 AND finalized_at <= ?
        ORDER BY finalized_at DESC, attempt_ref DESC LIMIT 1""",
        (profile_id, instance_id, message["occurred_at"]),
    ).fetchone()
    if row is None:
        return None
    cursor = conn.execute(
        """UPDATE contact_attempts SET answered = 1, success = 1,
        answered_message_id = ?, answered_at = ? WHERE profile_id = ?
        AND instance_id = ? AND attempt_ref = ? AND generation = ?
        AND status = 'FINALIZED' AND attempted = 1
        AND COALESCE(answered, 0) = 0""",
        (
            int(player_message_id),
            point,
            profile_id,
            instance_id,
            row["attempt_ref"],
            int(row["generation"]),
        ),
    )
    if cursor.rowcount != 1:
        return None
    promoted = 0 if int(row["success"] or 0) else 1
    observed_bucket = _contact_day_bucket(conn, profile_id, now)
    state = conn.execute(
        """SELECT daily_bucket, daily_success_count FROM instance_contact_state
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    if state is None:
        raise KeyError((profile_id, instance_id))
    bucket, carry_daily_count = contact_day_bucket_transition(
        state["daily_bucket"], observed_bucket
    )
    daily_count = (int(state["daily_success_count"]) if carry_daily_count else 0) + promoted
    conn.execute(
        """UPDATE instance_contact_state SET consecutive_unanswered = 0,
        cooldown_until = NULL, last_success_at = CASE WHEN ? = 1
            THEN ? ELSE last_success_at END,
        daily_success_count = ?,
        daily_bucket = ?, version = version + 1, updated_at = ?
        WHERE profile_id = ? AND instance_id = ?""",
        (
            promoted,
            point,
            daily_count,
            bucket,
            point,
            profile_id,
            instance_id,
        ),
    )
    released = conn.execute(
        """UPDATE instance_messages SET knowledge_eligibility = 'ELIGIBLE',
        knowledge_eligibility_reason = 'player_response_confirmed_delivery'
        WHERE profile_id = ? AND instance_id = ? AND direction = 'OUTBOUND'
          AND knowledge_eligibility = 'HELD'
          AND knowledge_eligibility_reason = 'delivery_unconfirmed'
          AND json_extract(metadata_json, '$.contact_attempt_ref') = ?""",
        (profile_id, instance_id, row["attempt_ref"]),
    )
    if released.rowcount:
        conn.execute(
            """UPDATE knowledge_processing_state SET
            processing_version = processing_version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?""",
            (point, profile_id, instance_id),
        )
    refresh_knowledge_task(conn, profile_id, instance_id, now_dt=now)
    return conn.execute(
        """SELECT * FROM contact_attempts WHERE profile_id = ?
        AND instance_id = ? AND attempt_ref = ? AND generation = ?""",
        (profile_id, instance_id, row["attempt_ref"], int(row["generation"])),
    ).fetchone()


__all__ = ["mark_latest_contact_attempt_answered_sql"]
