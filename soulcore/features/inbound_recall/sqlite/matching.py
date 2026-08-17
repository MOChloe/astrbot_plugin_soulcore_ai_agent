"""Transactional matching between OneBot recall receipts and ledger messages."""

from __future__ import annotations

import sqlite3
from datetime import timedelta

from ....storage.sqlite.codec import _dt
from ....storage.sqlite.dialogue_turns import context_eligible_sql
from ..algorithm import INBOUND_RECALL_ALGORITHM_VERSION
from ..domain import InboundRecallHold, InboundRecallTarget, OneBotRecallNotice
from .records import hold_from_row, normalize_scope, recall_receipt_id, required_datetime


def begin_notice(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    platform_instance_id: str,
    route_umo: str,
    notice: OneBotRecallNotice,
    insert: bool,
) -> InboundRecallTarget | None:
    receipt_id = recall_receipt_id(
        profile_id,
        instance_id,
        platform_instance_id,
        route_umo,
        notice.platform_message_id,
    )
    received_text = required_datetime(notice.received_at)
    if insert:
        _insert_receipt(
            conn,
            receipt_id=receipt_id,
            profile_id=profile_id,
            instance_id=instance_id,
            platform_instance_id=platform_instance_id,
            route_umo=route_umo,
            notice=notice,
            received_text=received_text,
        )
    receipt = _find_receipt(
        conn,
        profile_id=profile_id,
        instance_id=instance_id,
        platform_instance_id=platform_instance_id,
        route_umo=route_umo,
        platform_message_id=notice.platform_message_id,
    )
    if receipt is None or str(receipt["status"]) != "UNMATCHED":
        return None
    hold = find_or_create_hold(
        conn,
        profile_id=profile_id,
        instance_id=instance_id,
        platform_instance_id=platform_instance_id,
        route_umo=route_umo,
        platform_message_id=notice.platform_message_id,
        now_text=received_text,
    )
    if hold is None:
        return None
    cursor = conn.execute(
        """UPDATE inbound_recall_receipts SET status = 'PROCESSING',
        matched_ledger_message_id = ?, updated_at = ?
        WHERE receipt_id = ? AND status = 'UNMATCHED'""",
        (hold.ledger_message_id, received_text, str(receipt["receipt_id"])),
    )
    if cursor.rowcount != 1:
        return None
    return InboundRecallTarget(str(receipt["receipt_id"]), notice, hold)


def _insert_receipt(
    conn: sqlite3.Connection,
    *,
    receipt_id: str,
    profile_id: str,
    instance_id: str,
    platform_instance_id: str,
    route_umo: str,
    notice: OneBotRecallNotice,
    received_text: str,
) -> None:
    conn.execute(
        """INSERT INTO inbound_recall_receipts(
            receipt_id, profile_id, instance_id, platform_instance_id, route_umo,
            platform_message_id, notice_type, sender_id, operator_id, received_at,
            platform_occurred_at, status, expires_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'UNMATCHED', ?, ?, ?)
        ON CONFLICT(profile_id, instance_id, platform_instance_id, route_umo,
            platform_message_id) DO NOTHING""",
        (
            receipt_id,
            profile_id,
            instance_id,
            platform_instance_id,
            route_umo,
            notice.platform_message_id,
            notice.notice_type,
            notice.sender_id,
            notice.operator_id,
            received_text,
            _dt(notice.platform_occurred_at),
            required_datetime(notice.received_at + timedelta(hours=24)),
            received_text,
            received_text,
        ),
    )


def _find_receipt(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    platform_instance_id: str,
    route_umo: str,
    platform_message_id: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """SELECT * FROM inbound_recall_receipts
        WHERE profile_id = ? AND instance_id = ? AND platform_instance_id = ?
          AND route_umo = ? AND platform_message_id = ?""",
        (profile_id, instance_id, platform_instance_id, route_umo, platform_message_id),
    ).fetchone()


def find_or_create_hold(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    platform_instance_id: str,
    route_umo: str,
    platform_message_id: str,
    now_text: str,
) -> InboundRecallHold | None:
    row = conn.execute(
        """SELECT * FROM inbound_message_recall_states
        WHERE profile_id = ? AND instance_id = ? AND platform_instance_id = ?
          AND route_umo = ? AND platform_message_id = ?""",
        (profile_id, instance_id, platform_instance_id, route_umo, platform_message_id),
    ).fetchone()
    if row is not None:
        return hold_from_row(row)
    selected = _find_current_platform_message(
        conn,
        profile_id=profile_id,
        instance_id=instance_id,
        platform_instance_id=platform_instance_id,
        route_umo=route_umo,
        platform_message_id=platform_message_id,
    )
    if selected is None or str(selected["delivery_status"]) == "PENDING_RECALL_GRACE":
        return None
    previous = _previous_activity(
        conn,
        profile_id=profile_id,
        instance_id=instance_id,
        message_id=int(selected["message_id"]),
        occurred_at=str(selected["occurred_at"]),
    )
    conn.execute(
        """INSERT INTO inbound_message_recall_states(
            profile_id, instance_id, ledger_message_id, platform_instance_id,
            route_umo, platform_message_id, scope, direct_address,
            received_at, grace_until, previous_activity_at, status,
            committed_full_at, algorithm_version, original_plain_text,
            original_components_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, 'DISPATCHED', ?, ?, ?, ?, ?, ?)""",
        (
            profile_id,
            instance_id,
            int(selected["message_id"]),
            platform_instance_id,
            str(selected["route_umo"]),
            platform_message_id,
            normalize_scope(str(selected["scope"])),
            str(selected["occurred_at"]),
            str(selected["occurred_at"]),
            str(previous["occurred_at"]) if previous is not None else None,
            str(selected["occurred_at"]),
            INBOUND_RECALL_ALGORITHM_VERSION,
            str(selected["plain_text"] or ""),
            str(selected["components_json"] or "[]"),
            now_text,
            now_text,
        ),
    )
    created = conn.execute(
        """SELECT * FROM inbound_message_recall_states
        WHERE profile_id = ? AND instance_id = ? AND ledger_message_id = ?""",
        (profile_id, instance_id, int(selected["message_id"])),
    ).fetchone()
    return hold_from_row(created) if created is not None else None


def _find_current_platform_message(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    platform_instance_id: str,
    route_umo: str,
    platform_message_id: str,
) -> sqlite3.Row | None:
    """Resolve a recall notice through the current platform-message ledger.

    A normally dispatched inbound message has no recall-grace state.  When a
    later OneBot recall notice arrives, its durable platform fragment is the
    authoritative link to that message; a recall state is then materialized
    from the current ledger for settlement.  Do not fall back to a loose
    message-id or route lookup: the exact platform locator is the boundary.
    """

    return conn.execute(
        """SELECT message.*, fragment.platform_instance_id, fragment.route_umo,
            fragment.platform_message_id, instance.scope
        FROM instance_message_fragments fragment
        JOIN instance_messages message
          ON message.profile_id = fragment.profile_id
         AND message.instance_id = fragment.instance_id
         AND message.message_id = fragment.ledger_message_id
        JOIN character_instances instance
          ON instance.profile_id = message.profile_id
         AND instance.instance_id = message.instance_id
        WHERE fragment.profile_id = ? AND fragment.instance_id = ?
          AND fragment.platform_instance_id = ? AND fragment.route_umo = ?
          AND fragment.platform_message_id = ? AND fragment.direction = 'INBOUND'""",
        (profile_id, instance_id, platform_instance_id, route_umo, platform_message_id),
    ).fetchone()


def _previous_activity(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    message_id: int,
    occurred_at: str,
) -> sqlite3.Row | None:
    return conn.execute(
        f"""SELECT occurred_at FROM instance_messages
        WHERE profile_id = ? AND instance_id = ? AND message_id <> ?
          AND occurred_at <= ? AND {context_eligible_sql()}
          AND (direction = 'OUTBOUND' OR role = 'user')
        ORDER BY occurred_at DESC, message_id DESC LIMIT 1""",
        (profile_id, instance_id, message_id, occurred_at),
    ).fetchone()


__all__ = ["begin_notice", "find_or_create_hold"]
