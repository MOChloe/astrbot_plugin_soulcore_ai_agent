"""Shared foreground-activity fence used before closing a chat episode."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from .codec import _parse

_FOREGROUND_ACTIVITY_SQL = """
SELECT (
    EXISTS (
        SELECT 1 FROM instance_messages
        WHERE profile_id = :profile_id AND instance_id = :instance_id
          AND (
            :include_unprojected_messages = 1
            OR CAST(json_extract(
                  metadata_json,
                  '$.background_foreground_projection.version'
                ) AS INTEGER) = 1
          )
          AND (
            (direction = 'INBOUND' AND knowledge_eligibility = 'HELD')
            OR (direction = 'OUTBOUND' AND delivery_status = 'PENDING')
          )
    )
    OR EXISTS (
        SELECT 1 FROM conversation_turn_buffer_batches
        WHERE profile_id = :profile_id AND instance_id = :instance_id
          AND status IN ('PENDING','CLASSIFYING','WAITING','CLAIMED')
    )
    OR EXISTS (
        SELECT 1 FROM group_flow_windows
        WHERE profile_id = :profile_id AND instance_id = :instance_id
          AND status IN (
            'COLLECTING','JUDGING','READY','RUNNING','WAITING_FIRST_ATTEMPT'
          )
    )
    OR EXISTS (
        SELECT 1 FROM instance_core_runs
        WHERE profile_id = :profile_id AND instance_id = :instance_id
          AND status = 'RUNNING'
    )
    OR EXISTS (
        SELECT 1 FROM instance_main_core_occupancies
        WHERE profile_id = :profile_id AND instance_id = :instance_id
          AND status = 'ACTIVE'
    )
    OR EXISTS (
        SELECT 1 FROM instance_expression_batches
        WHERE profile_id = :profile_id AND instance_id = :instance_id
          AND status = 'ACTIVE'
    )
    OR EXISTS (
        SELECT 1 FROM instance_outbox
        WHERE profile_id = :profile_id AND instance_id = :instance_id
          AND status IN ('PENDING','SENDING')
    )
    OR EXISTS (
        SELECT 1 FROM message_retraction_actions
        WHERE profile_id = :profile_id AND instance_id = :instance_id
          AND status IN ('PENDING','SENDING')
    )
    OR EXISTS (
        SELECT 1 FROM platform_send_permits
        WHERE profile_id = :profile_id AND instance_id = :instance_id
          AND status IN ('RESERVED','DISPATCHING')
          AND (status = 'DISPATCHING' OR lease_until > :now)
    )
    OR EXISTS (
        SELECT 1 FROM ai_tasks
        WHERE profile_id = :profile_id AND instance_id = :instance_id
          AND task_type = 'MAIN_CORE'
          AND (
            status IN ('RUNNING','PAUSE_REQUESTED','CANCEL_REQUESTED')
            OR (
                status IN ('READY','SCHEDULED','RETRY_WAIT')
                AND due_at <= :now
            )
          )
    )
) AS active
"""


def foreground_activity_is_active_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    now: str,
    include_unprojected_messages: bool = True,
) -> bool:
    """Return whether any durable foreground lifecycle still owns the instance."""

    row = conn.execute(
        _FOREGROUND_ACTIVITY_SQL,
        {
            "profile_id": profile_id,
            "instance_id": instance_id,
            "now": now,
            "include_unprojected_messages": int(include_unprojected_messages),
        },
    ).fetchone()
    if row is None:
        raise RuntimeError("foreground activity fence query returned no row")
    return bool(int(row["active"]))


_FOREGROUND_TERMINAL_WATERMARK_SQL = """
SELECT MAX(moment) AS watermark
FROM (
    SELECT created_at AS moment
    FROM background_instances
    WHERE profile_id = :profile_id AND instance_id = :instance_id
    UNION ALL
    SELECT foreground_lease_until
    FROM background_instances
    WHERE profile_id = :profile_id AND instance_id = :instance_id
    UNION ALL
    SELECT last_foreground_at
    FROM background_instances
    WHERE profile_id = :profile_id AND instance_id = :instance_id
    UNION ALL
    SELECT MAX(finished_at)
    FROM instance_core_runs
    WHERE profile_id = :profile_id AND instance_id = :instance_id
      AND status <> 'RUNNING'
    UNION ALL
    SELECT MAX(COALESCE(finished_at, updated_at))
    FROM ai_tasks
    WHERE profile_id = :profile_id AND instance_id = :instance_id
      AND task_type = 'MAIN_CORE'
      AND status IN ('DEFERRED','SUCCEEDED','FAILED','CANCELLED')
    UNION ALL
    SELECT MAX(COALESCE(settled_at, updated_at))
    FROM instance_expression_batches
    WHERE profile_id = :profile_id AND instance_id = :instance_id
      AND status <> 'ACTIVE'
    UNION ALL
    SELECT MAX(updated_at)
    FROM instance_outbox
    WHERE profile_id = :profile_id AND instance_id = :instance_id
      AND status NOT IN ('PENDING','SENDING')
    UNION ALL
    SELECT MAX(updated_at)
    FROM message_retraction_actions
    WHERE profile_id = :profile_id AND instance_id = :instance_id
      AND status NOT IN ('PENDING','SENDING')
    UNION ALL
    SELECT MAX(
        CASE
          WHEN status = 'RESERVED' THEN lease_until
          ELSE updated_at
        END
    )
    FROM platform_send_permits
    WHERE profile_id = :profile_id AND instance_id = :instance_id
      AND NOT (
        status = 'DISPATCHING'
        OR (status = 'RESERVED' AND lease_until > :now)
      )
    UNION ALL
    SELECT MAX(COALESCE(released_at, updated_at))
    FROM instance_main_core_occupancies
    WHERE profile_id = :profile_id AND instance_id = :instance_id
      AND status <> 'ACTIVE'
    UNION ALL
    SELECT MAX(COALESCE(resolved_at, updated_at))
    FROM conversation_turn_buffer_batches
    WHERE profile_id = :profile_id AND instance_id = :instance_id
      AND status NOT IN ('PENDING','CLASSIFYING','WAITING','CLAIMED')
    UNION ALL
    SELECT MAX(COALESCE(resolved_at, updated_at))
    FROM group_flow_windows
    WHERE profile_id = :profile_id AND instance_id = :instance_id
      AND status NOT IN (
        'COLLECTING','JUDGING','READY','RUNNING','WAITING_FIRST_ATTEMPT'
      )
)
WHERE moment IS NOT NULL
"""


def foreground_terminal_watermark_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    now: str,
) -> datetime:
    """Return the latest trusted local terminal time for this foreground episode."""

    upper_bound = _parse(now)
    if upper_bound is None:
        raise ValueError("foreground terminal watermark bound is invalid")
    row = conn.execute(
        _FOREGROUND_TERMINAL_WATERMARK_SQL,
        {
            "profile_id": profile_id,
            "instance_id": instance_id,
            "now": now,
        },
    ).fetchone()
    watermark = _parse(row["watermark"]) if row is not None else None
    return watermark if watermark is not None else upper_bound


__all__ = [
    "foreground_activity_is_active_sql",
    "foreground_terminal_watermark_sql",
]
