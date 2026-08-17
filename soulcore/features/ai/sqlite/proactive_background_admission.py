"""Foreground-exclusion checks for proactive background-author admission."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from .support import _dt


def proactive_frame_has_foreground_blocker(
    conn: sqlite3.Connection,
    instance: sqlite3.Row,
    *,
    now_text: str,
    requester_task_id: int | None,
) -> bool:
    profile_id = str(instance["profile_id"])
    instance_id = str(instance["instance_id"])
    quiet_cutoff = _dt(datetime.fromisoformat(now_text) - timedelta(seconds=60))
    last_foreground = str(instance["last_foreground_at"] or "")
    if last_foreground and last_foreground > quiet_cutoff:
        return True
    requester = _proactive_main_core_requester(
        conn,
        profile_id,
        instance_id,
        requester_task_id,
    )
    row = conn.execute(
        """SELECT 1 WHERE
          EXISTS (SELECT 1 FROM instance_messages message
            WHERE message.profile_id = ? AND message.instance_id = ?
              AND ((message.direction = 'INBOUND'
                    AND message.knowledge_eligibility = 'HELD')
                OR (message.direction = 'OUTBOUND'
                    AND message.delivery_status = 'PENDING')))
          OR EXISTS (SELECT 1 FROM conversation_turn_buffer_batches batch
            WHERE batch.profile_id = ? AND batch.instance_id = ?
              AND batch.status IN ('PENDING','CLASSIFYING','WAITING','CLAIMED'))
          OR EXISTS (SELECT 1 FROM group_flow_windows window
            WHERE window.profile_id = ? AND window.instance_id = ?
              AND window.status IN ('COLLECTING','JUDGING','READY','RUNNING',
                                    'WAITING_FIRST_ATTEMPT'))
          OR EXISTS (SELECT 1 FROM instance_core_runs run
            WHERE run.profile_id = ? AND run.instance_id = ?
              AND run.status = 'RUNNING')
          OR EXISTS (SELECT 1 FROM instance_main_core_occupancies occupancy
            WHERE occupancy.profile_id = ? AND occupancy.instance_id = ?
              AND occupancy.status = 'ACTIVE')
          OR EXISTS (SELECT 1 FROM instance_expression_batches expression
            WHERE expression.profile_id = ? AND expression.instance_id = ?
              AND expression.status = 'ACTIVE')
          OR EXISTS (SELECT 1 FROM instance_outbox outbox
            WHERE outbox.profile_id = ? AND outbox.instance_id = ?
              AND outbox.status IN ('PENDING','SENDING'))
          OR EXISTS (SELECT 1 FROM message_retraction_actions retraction
            WHERE retraction.profile_id = ? AND retraction.instance_id = ?
              AND retraction.status IN ('PENDING','SENDING'))
          OR EXISTS (SELECT 1 FROM platform_send_permits permit
            WHERE permit.profile_id = ? AND permit.instance_id = ?
              AND permit.status IN ('RESERVED','DISPATCHING')
              AND (permit.status = 'DISPATCHING' OR permit.lease_until > ?))
          OR EXISTS (SELECT 1 FROM ai_tasks main_core
            WHERE main_core.profile_id = ? AND main_core.instance_id = ?
              AND main_core.task_type = 'MAIN_CORE'
              AND main_core.task_id <> ?
              AND (main_core.status IN
                   ('RUNNING','PAUSE_REQUESTED','CANCEL_REQUESTED')
                OR (main_core.status IN ('READY','SCHEDULED','RETRY_WAIT')
                    AND main_core.due_at <= ?)))
        LIMIT 1""",
        (
            profile_id,
            instance_id,
            profile_id,
            instance_id,
            profile_id,
            instance_id,
            profile_id,
            instance_id,
            profile_id,
            instance_id,
            profile_id,
            instance_id,
            profile_id,
            instance_id,
            profile_id,
            instance_id,
            profile_id,
            instance_id,
            now_text,
            profile_id,
            instance_id,
            requester,
            now_text,
        ),
    ).fetchone()
    return row is not None


def _proactive_main_core_requester(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    requester_task_id: int | None,
) -> int:
    requested = int(requester_task_id or 0)
    if requested < 1:
        return -1
    row = conn.execute(
        """SELECT 1 FROM ai_tasks
        WHERE task_id = ? AND profile_id = ? AND instance_id = ?
          AND task_type = 'MAIN_CORE' AND status = 'RUNNING'
          AND json_extract(input_json, '$.payload.source')
              IN ('TIMER','PLUGIN_WAKE')""",
        (requested, profile_id, instance_id),
    ).fetchone()
    return requested if row is not None else -1


__all__ = ["proactive_frame_has_foreground_blocker"]
