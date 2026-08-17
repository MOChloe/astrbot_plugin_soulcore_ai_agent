"""Atomic SQLite transitions shared by expression delivery and group relocation."""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from typing import Any

from ...contracts.models import ExpressionBatchStatus, OutboxStatus
from .codec import _dt, _parse

_ATTEMPTED_OUTBOX_STATUSES = frozenset(
    {
        OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED.value,
        OutboxStatus.UNKNOWN_AFTER_CRASH.value,
    }
)


def _resolved_batch_status(
    statuses: list[str], now: str
) -> tuple[ExpressionBatchStatus, str | None]:
    if any(status in {"PENDING", "SENDING"} for status in statuses):
        return ExpressionBatchStatus.ACTIVE, None
    if all(status == OutboxStatus.CANCELLED.value for status in statuses):
        return ExpressionBatchStatus.CANCELLED, now
    if all(status == OutboxStatus.FAILED.value for status in statuses):
        return ExpressionBatchStatus.FAILED, now
    if all(status in {*_ATTEMPTED_OUTBOX_STATUSES, "RETRACTED"} for status in statuses):
        return ExpressionBatchStatus.SETTLED, now
    return ExpressionBatchStatus.PARTIALLY_SETTLED, now


def sync_expression_batch_status(conn: sqlite3.Connection, batch_id: str | None, now: str) -> None:
    """Set batch settlement and release a terminal timer occupancy atomically."""

    if not batch_id:
        return
    outbox_rows = list(
        conn.execute(
            """SELECT status, COALESCE(expression_step_ordinal, expression_ordinal) AS step_ordinal
            FROM instance_outbox WHERE expression_batch_id = ?
            ORDER BY COALESCE(expression_step_ordinal, expression_ordinal)""",
            (batch_id,),
        )
    )
    action_rows = list(
        conn.execute(
            """SELECT status, step_ordinal FROM message_retraction_actions
            WHERE expression_batch_id = ? ORDER BY step_ordinal""",
            (batch_id,),
        )
    )
    rows = sorted([*outbox_rows, *action_rows], key=lambda item: int(item["step_ordinal"]))
    if not rows:
        return
    status, settled_at = _resolved_batch_status([str(row["status"]) for row in rows], now)
    conn.execute(
        """UPDATE instance_expression_batches SET status = ?, settled_at = ?,
        updated_at = ? WHERE batch_id = ?""",
        (status.value, settled_at, now, batch_id),
    )
    if status is not ExpressionBatchStatus.ACTIVE:
        _settle_terminal_timer_occupancy(conn, batch_id, status, now)


def terminal_timer_occurrence_status(batch_status: str | ExpressionBatchStatus) -> str:
    status = (
        batch_status
        if isinstance(batch_status, ExpressionBatchStatus)
        else ExpressionBatchStatus(str(batch_status))
    )
    if status is ExpressionBatchStatus.SETTLED:
        return "COMPLETED"
    if status is ExpressionBatchStatus.CANCELLED:
        return "CANCELLED"
    if status in {
        ExpressionBatchStatus.FAILED,
        ExpressionBatchStatus.PARTIALLY_SETTLED,
    }:
        return "FAILED"
    raise ValueError("active expression batch cannot settle Timer occurrence")


def _settle_terminal_timer_occupancy(
    conn: sqlite3.Connection,
    batch_id: str,
    batch_status: ExpressionBatchStatus,
    now: str,
) -> None:
    occupancy = conn.execute(
        """SELECT * FROM instance_main_core_occupancies
        WHERE kind = 'EXPRESSION' AND resource_ref = ? AND status = 'ACTIVE'""",
        (batch_id,),
    ).fetchone()
    if occupancy is None:
        return
    occurrence = conn.execute(
        """SELECT * FROM timer_occurrences WHERE profile_id = ? AND instance_id = ?
        AND delivery_ref = ? AND status = 'WAITING_DELIVERY' AND generation = ?""",
        (occupancy["profile_id"], occupancy["instance_id"], batch_id, occupancy["generation"]),
    ).fetchone()
    if occurrence is None:
        return
    changed = conn.execute(
        """UPDATE timer_occurrences SET status = ?, version = version + 1
        WHERE profile_id = ? AND instance_id = ? AND occurrence_id = ?
        AND version = ? AND generation = ? AND status = 'WAITING_DELIVERY'""",
        (
            terminal_timer_occurrence_status(batch_status),
            occurrence["profile_id"],
            occurrence["instance_id"],
            occurrence["occurrence_id"],
            occurrence["version"],
            occurrence["generation"],
        ),
    ).rowcount
    if changed != 1:
        return
    if terminal_timer_occurrence_status(batch_status) == "COMPLETED":
        from ...features.timers.sqlite.rule_completion import (
            complete_one_shot_rule_for_occurrence,
        )

        complete_one_shot_rule_for_occurrence(conn, occurrence)
    released = conn.execute(
        """UPDATE instance_main_core_occupancies SET status = 'RELEASED',
        version = version + 1, updated_at = ?, released_at = ?
        WHERE profile_id = ? AND instance_id = ? AND occupancy_id = ?
        AND version = ? AND generation = ? AND kind = 'EXPRESSION'
        AND resource_ref = ? AND status = 'ACTIVE'""",
        (
            now,
            now,
            occupancy["profile_id"],
            occupancy["instance_id"],
            occupancy["occupancy_id"],
            occupancy["version"],
            occupancy["generation"],
            batch_id,
        ),
    ).rowcount
    if released != 1:
        raise RuntimeError("Timer expression occupancy release lost")


def cancel_pending_expression_row(
    conn: sqlite3.Connection, row: sqlite3.Row, *, reason: str, now: str
) -> bool:
    """Cancel a not-yet-dispatched expression and settle all dependent SQL state."""

    return settle_pending_outbox_row(
        conn,
        row,
        status=OutboxStatus.CANCELLED,
        reason=reason,
        now=now,
    )


def settle_pending_outbox_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    status: OutboxStatus,
    reason: str,
    now: str,
    error_code: str = "",
) -> bool:
    """Settle a definitely unattempted row and release its selected resources."""

    if status not in {OutboxStatus.CANCELLED, OutboxStatus.FAILED}:
        raise ValueError("pending outbox can only settle as cancelled or failed")

    cursor = conn.execute(
        """UPDATE instance_outbox SET status = ?, last_error_code = ?,
        last_error = ?, updated_at = ?
        WHERE outbox_id = ? AND status = ?""",
        (
            status.value,
            str(error_code or ""),
            reason,
            now,
            int(row["outbox_id"]),
            OutboxStatus.PENDING.value,
        ),
    )
    if cursor.rowcount != 1:
        return False
    _settle_linked_ledger(conn, row, status)
    _fail_reserved_permit(conn, row, reason, now)
    _fail_selected_image_assets(conn, row, reason, now)
    return True


def _settle_linked_ledger(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    status: OutboxStatus,
) -> None:
    if row["context_message_id"] is None:
        return
    conn.execute(
        """UPDATE instance_messages SET delivery_status = ?
        WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
        (
            status.value,
            str(row["profile_id"]),
            str(row["instance_id"]),
            int(row["context_message_id"]),
        ),
    )


def _fail_reserved_permit(
    conn: sqlite3.Connection, row: sqlite3.Row, reason: str, now: str
) -> None:
    conn.execute(
        """UPDATE platform_send_permits SET status = 'FAILED_BEFORE_DISPATCH',
        detail = ?, updated_at = ? WHERE profile_id = ? AND instance_id = ?
        AND origin_kind = 'EXPRESSION_ITEM' AND origin_id = ? AND status = 'RESERVED'""",
        (
            reason[:600],
            now,
            str(row["profile_id"]),
            str(row["instance_id"]),
            f"expression-outbox:{int(row['outbox_id'])}",
        ),
    )


def _fail_selected_image_assets(
    conn: sqlite3.Connection, row: sqlite3.Row, reason: str, now: str
) -> None:
    payload = _load_payload(row["payload_json"])
    asset_ids = list(
        dict.fromkeys(
            str(component.get("asset_id") or "").strip()
            for component in list(payload.get("components") or [])
            if isinstance(component, dict)
            and str(component.get("type") or "").lower() == "image_asset"
            and str(component.get("asset_id") or "").strip()
        )
    )
    if not asset_ids:
        return
    current = _parse(now)
    assert current is not None
    placeholders = ",".join("?" for _ in asset_ids)
    conn.execute(
        f"""UPDATE media_assets SET delivery_status = 'FAILED',
        expires_at = COALESCE(expires_at, ?), last_error = ?, updated_at = ?
        WHERE profile_id = ? AND instance_id = ? AND delivery_status = 'SELECTED'
        AND asset_id IN ({placeholders})""",
        (
            _dt(current + timedelta(hours=24)),
            reason,
            now,
            str(row["profile_id"]),
            str(row["instance_id"]),
            *asset_ids,
        ),
    )


def _load_payload(value: object) -> dict[str, Any]:
    try:
        payload = json.loads(str(value or "{}"))
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


__all__ = [
    "cancel_pending_expression_row",
    "settle_pending_outbox_row",
    "sync_expression_batch_status",
    "terminal_timer_occurrence_status",
]
