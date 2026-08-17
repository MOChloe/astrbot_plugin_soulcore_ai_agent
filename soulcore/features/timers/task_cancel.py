"""Atomic domain settlement for permanent durable-task cancellation."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from ...contracts.ai_task_payload import decode_task_payload
from ...storage.sqlite.codec import encode_datetime
from .domain import IdempotencyKey, TimerOccurrenceStatus
from .record_codec import decode_occurrence
from .transitions import OccurrenceAction, transition_occurrence

_TIMER_CANCELLABLE = {
    TimerOccurrenceStatus.SCHEDULED,
    TimerOccurrenceStatus.WAITING,
    TimerOccurrenceStatus.CLAIMED,
    TimerOccurrenceStatus.RUNNING,
    TimerOccurrenceStatus.PAUSED,
    TimerOccurrenceStatus.RECOVERING,
}


def settle_permanent_task_cancel(
    conn: sqlite3.Connection,
    task: sqlite3.Row,
    now: datetime,
) -> bool:
    """Settle only the Timer occurrence durably bound to ``task``."""

    input_data = _task_input(task)
    if str(task["task_type"] or "").upper() == "TIMER_RUN":
        return _cancel_timer_occurrence(conn, task, input_data, now)
    return False


def has_permanent_domain_cancel(conn: sqlite3.Connection, task: sqlite3.Row) -> bool:
    """Return whether the latest cancel request permanently owns a domain binding."""

    if not _has_domain_binding(task, _task_input(task)):
        return False
    audit = conn.execute(
        """SELECT details_json FROM ai_task_audit WHERE task_id = ?
        AND action = 'REQUEST_CANCEL' ORDER BY audit_id DESC LIMIT 1""",
        (int(task["task_id"]),),
    ).fetchone()
    if audit is None:
        return False
    try:
        details = json.loads(str(audit["details_json"] or "{}"))
    except ValueError:
        return False
    return isinstance(details, dict) and details.get("permanent_domain_cancel") is True


def _cancel_timer_occurrence(
    conn: sqlite3.Connection,
    task: sqlite3.Row,
    input_data: Mapping[str, Any],
    now: datetime,
) -> bool:
    occurrence_id = str(input_data.get("occurrence_id") or "").strip()
    try:
        generation = int(input_data.get("generation"))
    except (TypeError, ValueError):
        return False
    if not occurrence_id:
        return False
    row = conn.execute(
        """SELECT * FROM timer_occurrences WHERE profile_id = ? AND instance_id = ?
        AND occurrence_id = ?""",
        (task["profile_id"], task["instance_id"], occurrence_id),
    ).fetchone()
    if row is None or int(row["generation"]) != generation:
        return False
    occurrence = decode_occurrence(dict(row))
    if occurrence.status not in _TIMER_CANCELLABLE:
        return False
    updated = transition_occurrence(
        occurrence,
        OccurrenceAction.CANCEL,
        expected_version=occurrence.version,
        operation_key=IdempotencyKey(f"ai-task-cancel:{task['task_id']}:{generation}"),
        now=now,
    )
    cursor = conn.execute(
        """UPDATE timer_occurrences SET status = ?, version = ?, generation = ?,
        execution_ref = ?, delivery_ref = ?, recovery_from = ?,
        last_operation_key = ?, last_operation_fingerprint = ?
        WHERE profile_id = ? AND instance_id = ? AND occurrence_id = ?
        AND status = ? AND version = ? AND generation = ?""",
        (
            updated.status.value,
            updated.version,
            updated.generation,
            updated.execution_ref.value if updated.execution_ref else None,
            updated.delivery_ref.value if updated.delivery_ref else None,
            updated.recovery_from.value if updated.recovery_from else None,
            updated.last_operation_key,
            updated.last_operation_fingerprint,
            task["profile_id"],
            task["instance_id"],
            occurrence_id,
            occurrence.status.value,
            occurrence.version,
            occurrence.generation,
        ),
    )
    if cursor.rowcount != 1:
        raise RuntimeError("Timer task cancellation lost its occurrence fence")
    _release_timer_occupancy(conn, task, occurrence_id, generation, now)
    return True


def _release_timer_occupancy(
    conn: sqlite3.Connection,
    task: sqlite3.Row,
    occurrence_id: str,
    generation: int,
    now: datetime,
) -> None:
    occupancy = conn.execute(
        """SELECT * FROM instance_main_core_occupancies WHERE profile_id = ?
        AND instance_id = ? AND resource_ref = ? AND kind = 'TIMER'
        AND status = 'ACTIVE'""",
        (task["profile_id"], task["instance_id"], occurrence_id),
    ).fetchone()
    if occupancy is None:
        return
    if int(occupancy["generation"]) != generation:
        raise RuntimeError("Timer task cancellation found a mismatched occupancy generation")
    stamp = encode_datetime(now)
    cursor = conn.execute(
        """UPDATE instance_main_core_occupancies SET status = 'RELEASED',
        version = version + 1, updated_at = ?, released_at = ?
        WHERE profile_id = ? AND instance_id = ? AND occupancy_id = ?
        AND kind = 'TIMER' AND status = 'ACTIVE' AND version = ? AND generation = ?
        AND lease_owner = ? AND lease_token = ?""",
        (
            stamp,
            stamp,
            occupancy["profile_id"],
            occupancy["instance_id"],
            occupancy["occupancy_id"],
            occupancy["version"],
            occupancy["generation"],
            occupancy["lease_owner"],
            occupancy["lease_token"],
        ),
    )
    if cursor.rowcount != 1:
        raise RuntimeError("Timer task cancellation lost its occupancy fence")


def _has_domain_binding(task: sqlite3.Row, input_data: Mapping[str, Any]) -> bool:
    if str(task["task_type"] or "").upper() == "TIMER_RUN":
        return bool(str(input_data.get("occurrence_id") or "").strip())
    return False


def _task_input(task: sqlite3.Row) -> Mapping[str, Any]:
    try:
        return decode_task_payload("input", task["input_json"])
    except ValueError:
        return {}


__all__ = ["has_permanent_domain_cancel", "settle_permanent_task_cancel"]
