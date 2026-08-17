"""Atomic cleanup side effects for cancelled expression outbox rows."""

from __future__ import annotations

import sqlite3
from typing import Any


def is_file_expression(payload: dict[str, Any]) -> bool:
    if str(payload.get("expression_kind") or "").upper() == "FILE":
        return True
    if str(payload.get("file_delivery_role") or "").upper() == "ARTIFACT":
        return True
    return any(
        isinstance(component, dict) and str(component.get("type") or "").lower() == "file_artifact"
        for component in list(payload.get("components") or [])
    )


def restore_cancelled_file_todos(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    rows: list[sqlite3.Row],
    now: str,
    reason: str,
    *,
    load_payload,
) -> None:
    todo_ids: list[str] = []
    for row in rows:
        payload = load_payload(row["payload_json"]) or {}
        if not is_file_expression(payload):
            continue
        todo_ids.extend(str(value) for value in payload.get("important_todo_ids") or [] if value)
    todo_ids = list(dict.fromkeys(todo_ids))
    if not todo_ids:
        return
    placeholders = ",".join("?" for _ in todo_ids)
    conn.execute(
        f"""UPDATE important_todos SET status = 'PENDING', selected_run_id = NULL,
        selected_activity_epoch = NULL, delivery_outbox_id = NULL,
        resolved_at = NULL,
        version = CASE WHEN status = 'PENDING' THEN version ELSE version + 1 END,
        updated_at = ? WHERE profile_id = ? AND instance_id = ?
        AND todo_id IN ({placeholders})
        AND status IN ('PENDING', 'SELECTED', 'DELIVERY_PENDING')""",
        (now, profile_id, instance_id, *todo_ids),
    )
    conn.execute(
        f"""UPDATE file_assets SET delivery_status = 'NOT_SELECTED', last_error = ?,
        updated_at = ? WHERE asset_id IN (SELECT file_asset_id FROM important_todos
        WHERE profile_id = ? AND instance_id = ? AND todo_id IN ({placeholders}))""",
        (reason[:600], now, profile_id, instance_id, *todo_ids),
    )


__all__ = [
    "is_file_expression",
    "restore_cancelled_file_todos",
]
