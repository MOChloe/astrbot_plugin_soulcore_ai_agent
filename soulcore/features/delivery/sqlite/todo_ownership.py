"""Atomic ownership binding between durable outbox rows and file todos."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

_ACTIVE_OUTBOX_STATUSES = frozenset({"PENDING", "SENDING"})
_BINDABLE_TODO_STATUSES = frozenset({"SELECTED", "DELIVERY_PENDING"})


def bind_outbox_todos(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    outbox_id: int,
    todo_ids: Iterable[str],
    selected_run_id: int | None,
) -> None:
    """Bind selected todos to one live outbox, replacing only terminal owners."""

    normalized = _normalized_todo_ids(todo_ids)
    if not normalized:
        return
    _require_live_outbox(conn, profile_id, instance_id, outbox_id)
    rows = _todo_ownership_rows(conn, profile_id, instance_id, normalized)
    _validate_todo_ownership_rows(rows, normalized, outbox_id, selected_run_id)
    _bind_todo_rows(conn, profile_id, instance_id, outbox_id, normalized)


def _normalized_todo_ids(todo_ids: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value).strip() for value in todo_ids if str(value).strip()))


def _require_live_outbox(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    outbox_id: int,
) -> None:
    owner = conn.execute(
        """SELECT status FROM instance_outbox
        WHERE profile_id = ? AND instance_id = ? AND outbox_id = ?""",
        (profile_id, instance_id, int(outbox_id)),
    ).fetchone()
    if owner is None:
        raise KeyError((profile_id, instance_id, int(outbox_id)))
    if str(owner["status"]).upper() not in _ACTIVE_OUTBOX_STATUSES:
        raise RuntimeError("file todo owner outbox is already terminal")


def _todo_ownership_rows(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    todo_ids: tuple[str, ...],
) -> list[sqlite3.Row]:
    placeholders = ",".join("?" for _ in todo_ids)
    return list(
        conn.execute(
            f"""SELECT todo.todo_id, todo.status, todo.selected_run_id,
                todo.delivery_outbox_id, previous.status AS previous_outbox_status
            FROM important_todos AS todo
            LEFT JOIN instance_outbox AS previous
              ON previous.outbox_id = todo.delivery_outbox_id
            WHERE todo.profile_id = ? AND todo.instance_id = ?
              AND todo.todo_id IN ({placeholders})""",
            (profile_id, instance_id, *todo_ids),
        )
    )


def _validate_todo_ownership_rows(
    rows: list[sqlite3.Row],
    todo_ids: tuple[str, ...],
    outbox_id: int,
    selected_run_id: int | None,
) -> None:
    if {str(row["todo_id"]) for row in rows} != set(todo_ids):
        raise KeyError("file todo ownership changed before outbox binding")
    for row in rows:
        _validate_todo_ownership_row(row, outbox_id, selected_run_id)


def _validate_todo_ownership_row(
    row: sqlite3.Row,
    outbox_id: int,
    selected_run_id: int | None,
) -> None:
    if str(row["status"]).upper() not in _BINDABLE_TODO_STATUSES:
        raise RuntimeError("file todo is not selected for durable delivery")
    if selected_run_id is not None and int(row["selected_run_id"] or 0) != int(selected_run_id):
        raise RuntimeError("file todo belongs to another core run")
    previous_id = int(row["delivery_outbox_id"] or 0)
    previous_is_live = str(row["previous_outbox_status"] or "").upper() in _ACTIVE_OUTBOX_STATUSES
    if previous_id and previous_id != int(outbox_id) and previous_is_live:
        raise RuntimeError("file todo already belongs to another live outbox")


def _bind_todo_rows(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    outbox_id: int,
    todo_ids: tuple[str, ...],
) -> None:
    placeholders = ",".join("?" for _ in todo_ids)
    cursor = conn.execute(
        f"""UPDATE important_todos SET delivery_outbox_id = ?
        WHERE profile_id = ? AND instance_id = ?
          AND todo_id IN ({placeholders})""",
        (int(outbox_id), profile_id, instance_id, *todo_ids),
    )
    if cursor.rowcount != len(todo_ids):
        raise RuntimeError("file todo outbox binding was not atomic")


__all__ = ["bind_outbox_todos"]
