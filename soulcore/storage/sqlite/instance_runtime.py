"""Required singleton runtime rows for one conversation instance."""

from __future__ import annotations

import sqlite3

from .background_projection import ensure_background_instance_sql


def seed_instance_runtime_rows(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    now: str,
) -> None:
    """Restore the same blank runtime invariants used for a new instance."""

    ensure_background_instance_sql(conn, profile_id, instance_id, now)
    conn.execute(
        """INSERT OR IGNORE INTO instance_contact_state(
                profile_id, instance_id, timeline_event_watermark,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)""",
        (profile_id, instance_id, 0, now, now),
    )
    conn.execute(
        """INSERT OR IGNORE INTO instance_state_gate_snapshots(
                profile_id, instance_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?)""",
        (profile_id, instance_id, now, now),
    )
    message_baseline = int(
        conn.execute(
            """SELECT COALESCE(MAX(message_id), 0) FROM instance_messages
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        ).fetchone()[0]
    )
    conn.execute(
        """INSERT OR IGNORE INTO knowledge_processing_state(
                profile_id, instance_id, baseline_message_id,
                committed_through_message_id, desired_through_message_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            profile_id,
            instance_id,
            message_baseline,
            message_baseline,
            message_baseline,
            now,
            now,
        ),
    )


__all__ = ["seed_instance_runtime_rows"]
