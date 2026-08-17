"""Operational expression-interruption events attached to their inbound trigger."""

from __future__ import annotations

import sqlite3
from typing import Any

from .expression_outbox import ATTEMPTED_OUTBOX_STATUSES
from .support import OutboxInterruptPolicy, OutboxStatus, _dump, _load


def _output_text(payload: dict[str, Any]) -> str:
    content = str(payload.get("content") or "").strip()
    if content:
        return f"文本：{content}"
    labels = {"IMAGE": "一张图片", "STICKER": "一个表情包", "FILE": "一个文件"}
    kind = str(payload.get("expression_kind") or "").strip().upper()
    return f"原计划发送{labels[kind]}" if kind in labels else "一项非文字表达"


def _continuing_reason(row: sqlite3.Row, explicit: dict[int, str]) -> str:
    if int(row["outbox_id"]) in explicit:
        reason = explicit[int(row["outbox_id"])]
        return {
            "NEAR_DUE_RUNTIME_KEEP": "临近发送时间，保持原计划",
            "AI_PRESERVE": "明确保留",
        }.get(reason, reason)
    runtime = (_load(row["payload_json"]) or {}).get("interrupt_runtime")
    if isinstance(runtime, dict) and runtime.get("decision") == "KEEP":
        return "临近发送时间，保持原计划"
    return "明确保留"


def _event_note(
    delivered: list[sqlite3.Row],
    continuing: list[tuple[sqlite3.Row, str]],
    cancelled: list[sqlite3.Row],
) -> str:
    sections = ["这条新消息打断了本人此前已经提交的表达批次。"]
    if delivered:
        lines = [
            f"{index}. {_output_text(_load(row['payload_json']) or {})}"
            for index, row in enumerate(delivered, start=1)
        ]
        sections.append("以下内容已经真实发起发送，可能已被对方看见：\n" + "\n".join(lines))
    if continuing:
        lines = [
            f"{index}. {_output_text(_load(row['payload_json']) or {})}（{reason}）"
            for index, (row, reason) in enumerate(continuing, start=1)
        ]
        sections.append("以下内容仍会按原顺序继续发送：\n" + "\n".join(lines))
    if cancelled:
        lines = [
            f"{index}. {_output_text(_load(row['payload_json']) or {})}"
            for index, row in enumerate(cancelled, start=1)
        ]
        sections.append("以下内容当时尚未发送，现已取消，对方没有看见：\n" + "\n".join(lines))
    return "\n".join(sections)


def interruption_event_exists(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    batch_id: str,
    inbound_message_id: int,
) -> bool:
    return bool(
        conn.execute(
            """SELECT 1 FROM expression_interruption_events
            WHERE profile_id = ? AND instance_id = ? AND batch_id = ?
              AND inbound_message_id = ?""",
            (profile_id, instance_id, batch_id, inbound_message_id),
        ).fetchone()
    )


def _continuing_rows(
    rows: list[sqlite3.Row], explicit: dict[int, str]
) -> list[tuple[sqlite3.Row, str]]:
    return [
        (row, _continuing_reason(row, explicit))
        for row in rows
        if str(row["status"]) in {OutboxStatus.PENDING.value, OutboxStatus.SENDING.value}
        and str(row["interrupt_policy"]) == OutboxInterruptPolicy.PRESERVE.value
    ]


def _runtime_details(rows: list[sqlite3.Row], inbound_message_id: int) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for row in rows:
        runtime = (_load(row["payload_json"]) or {}).get("interrupt_runtime")
        if isinstance(runtime, dict) and int(runtime.get("message_id") or 0) == inbound_message_id:
            details.append({"ordinal": int(row["expression_ordinal"]), **runtime})
    return sorted(details, key=lambda item: int(item["ordinal"]))


def _event_metadata(
    conn: sqlite3.Connection,
    *,
    batch_id: str,
    inbound_message_id: int,
    all_rows: list[sqlite3.Row],
    delivered: list[sqlite3.Row],
    continuing: list[tuple[sqlite3.Row, str]],
    cancelled: list[sqlite3.Row],
) -> dict[str, Any]:
    batch = conn.execute(
        "SELECT status FROM instance_expression_batches WHERE batch_id = ?", (batch_id,)
    ).fetchone()
    statuses = [str(row["status"]) for row in all_rows]
    delivered_prefix_count = 0
    for status in statuses:
        if status not in ATTEMPTED_OUTBOX_STATUSES:
            break
        delivered_prefix_count += 1
    return {
        "event_kind": "expression_interruption",
        "interrupted_by_message_id": inbound_message_id,
        "batch_status": str(batch["status"] if batch else "CANCELLED"),
        "delivered_prefix_count": delivered_prefix_count,
        "attempted_count": len(delivered),
        "attempted_ordinals": [int(row["expression_ordinal"]) for row in delivered],
        "continuing_ordinals": [int(row["expression_ordinal"]) for row, _ in continuing],
        "continuing_reasons": [
            {"ordinal": int(row["expression_ordinal"]), "reason": reason}
            for row, reason in continuing
        ],
        "cancelled_ordinals": [int(row["expression_ordinal"]) for row in cancelled],
        "interrupt_runtime": _runtime_details(
            [row for row, _ in continuing] + cancelled, inbound_message_id
        ),
        "all_statuses": statuses,
    }


def append_interruption_event(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    batch_id: str,
    inbound_message_id: int,
    continuing_reasons: dict[int, str],
    cancelled_rows: list[sqlite3.Row],
    now: str,
) -> bool:
    if interruption_event_exists(conn, profile_id, instance_id, batch_id, inbound_message_id):
        return False
    all_rows = list(
        conn.execute(
            """SELECT * FROM instance_outbox WHERE expression_batch_id = ?
            ORDER BY expression_ordinal""",
            (batch_id,),
        )
    )
    continuing = _continuing_rows(all_rows, continuing_reasons)
    delivered = [row for row in all_rows if str(row["status"]) in ATTEMPTED_OUTBOX_STATUSES]
    if not delivered and not continuing and not cancelled_rows:
        return False
    note = _event_note(delivered, continuing, cancelled_rows)
    metadata = _event_metadata(
        conn,
        batch_id=batch_id,
        inbound_message_id=inbound_message_id,
        all_rows=all_rows,
        delivered=delivered,
        continuing=continuing,
        cancelled=cancelled_rows,
    )
    inserted = conn.execute(
        """INSERT INTO expression_interruption_events(
            batch_id, profile_id, instance_id, inbound_message_id,
            context_note, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(batch_id, inbound_message_id) DO NOTHING""",
        (batch_id, profile_id, instance_id, inbound_message_id, note, _dump(metadata), now),
    ).rowcount
    return inserted == 1


__all__ = ["append_interruption_event", "interruption_event_exists"]
