"""Deterministic Timer rule completion owned by the persistence state machine."""

from __future__ import annotations

import sqlite3


def complete_one_shot_rule_for_occurrence(
    conn: sqlite3.Connection,
    occurrence: sqlite3.Row,
) -> bool:
    """Soft-complete an active one-shot rule after its occurrence succeeds."""

    changed = conn.execute(
        """UPDATE timer_rules SET status = 'COMPLETED', version = version + 1,
            last_operation_key = ?, last_operation_fingerprint = ''
        WHERE profile_id = ? AND instance_id = ? AND rule_id = ?
          AND schedule_kind IN ('ABSOLUTE', 'RELATIVE') AND status = 'ACTIVE'""",
        (
            f"system:one-shot-completed:{occurrence['occurrence_id']}:{occurrence['generation']}",
            occurrence["profile_id"],
            occurrence["instance_id"],
            occurrence["rule_id"],
        ),
    ).rowcount
    return changed == 1


__all__ = ["complete_one_shot_rule_for_occurrence"]
