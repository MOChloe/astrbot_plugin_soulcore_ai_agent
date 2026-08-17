"""Row decoding and stable identifiers for persisted recall state."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from typing import Any

from ....storage.sqlite.codec import _dt, _parse
from ..domain import InboundRecallHold, OneBotRecallNotice


def hold_from_row(row: sqlite3.Row) -> InboundRecallHold:
    return InboundRecallHold(
        profile_id=str(row["profile_id"]),
        instance_id=str(row["instance_id"]),
        ledger_message_id=int(row["ledger_message_id"]),
        platform_instance_id=str(row["platform_instance_id"]),
        route_umo=str(row["route_umo"]),
        platform_message_id=str(row["platform_message_id"]),
        scope=str(row["scope"]),
        direct_address=bool(row["direct_address"]),
        received_at=aware_datetime(row["received_at"]),
        grace_until=aware_datetime(row["grace_until"]),
        previous_activity_at=_parse(row["previous_activity_at"]),
        status=str(row["status"]),
        lease_token=int(row["lease_token"]),
        lease_until=_parse(row["lease_until"]),
        committed_full_at=_parse(row["committed_full_at"]),
        original_plain_text=str(row["original_plain_text"] or ""),
        original_components_json=str(row["original_components_json"] or "[]"),
    )


def notice_from_row(row: sqlite3.Row) -> OneBotRecallNotice:
    return OneBotRecallNotice(
        notice_type=str(row["notice_type"]),
        platform_message_id=str(row["platform_message_id"]),
        sender_id=str(row["sender_id"]),
        operator_id=str(row["operator_id"]),
        received_at=aware_datetime(row["received_at"]),
        platform_occurred_at=_parse(row["platform_occurred_at"]),
    )


def recall_receipt_id(
    profile_id: str,
    instance_id: str,
    platform_instance_id: str,
    route_umo: str,
    platform_message_id: str,
) -> str:
    raw = "\0".join((profile_id, instance_id, platform_instance_id, route_umo, platform_message_id))
    return "recall:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def required_datetime(value: datetime) -> str:
    result = _dt(value)
    if result is None:
        raise ValueError("recall timestamps must be timezone-aware")
    return result


def aware_datetime(value: Any) -> datetime:
    parsed = _parse(str(value))
    if parsed is None:
        raise ValueError("stored recall timestamp is invalid")
    return parsed.astimezone(UTC)


def normalize_scope(value: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in {"private", "group", "guild"} else "private"


__all__ = [
    "aware_datetime",
    "hold_from_row",
    "normalize_scope",
    "notice_from_row",
    "recall_receipt_id",
    "required_datetime",
]
