"""Idempotent durable admission for one inbound ledger message."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from ....storage.sqlite.background_projection import (
    release_inbound_foreground_lease_sql,
    renew_inbound_foreground_lease_sql,
)
from ....storage.sqlite.codec import _dt, _parse
from .expression_interruptions import _AdvanceActivityAndInterrupt

KnowledgeRefresh = Callable[..., sqlite3.Row | None]
ContactAnswer = Callable[..., sqlite3.Row | None]
ADMISSION_METADATA_KEY = "inbound_admission"


@dataclass(frozen=True, slots=True)
class InboundAdmissionResult:
    applied: bool
    activity_epoch: int
    activity_advanced: bool
    group_activity_held: bool
    interruption_changed: bool = False
    ownership_valid: bool = True


def renew_inbound_admission_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    message_id: int,
    lease_owner: str,
    lease_token: int,
    now: datetime,
    lease_seconds: int,
) -> bool:
    row = _admission_row(conn, profile_id, instance_id, message_id)
    metadata, marker = _metadata_and_marker(row)
    lease_until = _parse(str(marker.get("lease_until") or ""))
    if (
        marker.get("status") != "ADMITTING"
        or str(marker.get("lease_owner") or "") != lease_owner
        or int(marker.get("lease_token") or 0) != int(lease_token)
        or lease_until is None
        or lease_until <= now
    ):
        return False
    next_lease_until = _dt(now + timedelta(seconds=max(1, int(lease_seconds))))
    marker["lease_until"] = next_lease_until
    persisted = _persist_marker(
        conn,
        profile_id,
        instance_id,
        message_id,
        metadata,
        expected_metadata=str(row["metadata_json"]),
    )
    if persisted:
        renew_inbound_foreground_lease_sql(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            message_id=int(message_id),
            metadata=metadata,
            lease_until=next_lease_until,
            now=_dt(now),
        )
    return persisted


def claim_expired_inbound_admission_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    message_id: int,
    lease_owner: str,
    now: datetime,
    lease_seconds: int,
) -> int | None:
    row = _admission_row(conn, profile_id, instance_id, message_id)
    metadata, marker = _metadata_and_marker(row)
    lease_until = _parse(str(marker.get("lease_until") or ""))
    if marker.get("status") != "ADMITTING" or lease_until is None or lease_until > now:
        return None
    token = int(marker.get("lease_token") or 0) + 1
    marker.update(
        lease_owner=str(lease_owner),
        lease_token=token,
        lease_until=_dt(now + timedelta(seconds=max(1, int(lease_seconds)))),
    )
    if not _persist_marker(
        conn,
        profile_id,
        instance_id,
        message_id,
        metadata,
        expected_metadata=str(row["metadata_json"]),
    ):
        return None
    return token


def complete_inbound_admission_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    message_id: int,
    lease_owner: str,
    lease_token: int,
    now: datetime,
    status: str = "APPLIED",
) -> bool:
    row = _admission_row(conn, profile_id, instance_id, message_id)
    metadata, marker = _metadata_and_marker(row)
    if not _owns_marker(marker, lease_owner, lease_token, now=now):
        return False
    marker.update(status=str(status), lease_until=None)
    persisted = _persist_marker(
        conn,
        profile_id,
        instance_id,
        message_id,
        metadata,
        expected_metadata=str(row["metadata_json"]),
    )
    if persisted:
        release_inbound_foreground_lease_sql(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            message_id=int(message_id),
            metadata=metadata,
            released_at=_dt(now),
        )
    return persisted


def apply_inbound_admission_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    message_id: int,
    now: datetime,
    group_scope: bool,
    refresh_knowledge_task: KnowledgeRefresh,
    mark_contact_answered: ContactAnswer,
    lease_owner: str | None = None,
    lease_token: int | None = None,
) -> InboundAdmissionResult:
    """Apply every durable activity side effect once, marked on the ledger row."""

    row = _admission_row(conn, profile_id, instance_id, message_id)
    metadata, marker = _metadata_and_marker(row)
    owned = lease_owner is None or (
        lease_token is not None and _owns_marker(marker, lease_owner, lease_token, now=now)
    )
    if marker.get("status") == "APPLIED" or marker.get("activity_applied"):
        return InboundAdmissionResult(
            applied=False,
            activity_epoch=int(marker.get("activity_epoch") or 0),
            activity_advanced=bool(marker.get("activity_advanced")),
            group_activity_held=bool(marker.get("group_activity_held")),
            ownership_valid=owned,
        )
    if not owned:
        return InboundAdmissionResult(False, 0, False, False, ownership_valid=False)

    now_text = _dt(now)
    platform_message_id = str(metadata.get("platform_message_id") or "").strip()
    if platform_message_id:
        conn.execute(
            """INSERT INTO instance_delivery_state(
                profile_id, instance_id, inbound_message_id,
                inbound_received_at, passive_reply_uses,
                wakeup_periods_json, last_status, updated_at
            ) VALUES (?, ?, ?, ?, 0, '{}', 'INBOUND_REFRESHED', ?)
            ON CONFLICT(profile_id, instance_id) DO UPDATE SET
                inbound_message_id = excluded.inbound_message_id,
                inbound_received_at = excluded.inbound_received_at,
                passive_reply_uses = 0, wakeup_periods_json = '{}',
                last_mode = NULL, last_status = 'INBOUND_REFRESHED',
                last_error = NULL, updated_at = excluded.updated_at""",
            (profile_id, instance_id, platform_message_id, now_text, now_text),
        )

    mark_contact_answered(
        conn,
        profile_id,
        instance_id,
        player_message_id=int(message_id),
        now=now,
        refresh_knowledge_task=refresh_knowledge_task,
    )
    # Group activity is admitted in two distinct phases.  Appending the
    # durable window establishes the judge hold; only a later suitable/direct
    # settlement may advance activity and interrupt the old expression
    # suffix.  Treat every window-owned group message as held here, including
    # recovery after the window was silently resolved, so a recovered handler
    # can never manufacture a late interruption.
    group_activity_held = bool(
        group_scope and _belongs_to_group_window(conn, profile_id, instance_id, int(message_id))
    )
    interruption_changed = False
    if group_activity_held:
        epoch = _current_activity_epoch(conn, profile_id, instance_id)
    else:
        epoch, interruption = _AdvanceActivityAndInterrupt(
            profile_id, instance_id, int(message_id), now_text
        )(conn)
        interruption_changed = bool(interruption.changed)
        _invalidate_contact_clock_sql(
            conn,
            profile_id,
            instance_id,
            activity_epoch=int(epoch),
            defer_until=now + timedelta(seconds=5),
            now_text=now_text,
        )

    marker.update(
        activity_applied=True,
        activity_epoch=int(epoch),
        activity_advanced=not group_activity_held,
        group_activity_held=group_activity_held,
        applied_at=now_text,
    )
    if lease_owner is None:
        marker.update(status="APPLIED", lease_until=None)
    if not _persist_marker(
        conn,
        profile_id,
        instance_id,
        message_id,
        metadata,
        expected_metadata=str(row["metadata_json"]),
    ):
        raise RuntimeError("inbound admission marker was not persisted")
    return InboundAdmissionResult(
        applied=True,
        activity_epoch=int(epoch),
        activity_advanced=not group_activity_held,
        group_activity_held=group_activity_held,
        interruption_changed=interruption_changed,
    )


def _belongs_to_group_window(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    message_id: int,
) -> bool:
    return (
        conn.execute(
            """SELECT 1 FROM group_flow_window_members
            WHERE profile_id = ? AND instance_id = ? AND message_id = ?
            LIMIT 1""",
            (profile_id, instance_id, int(message_id)),
        ).fetchone()
        is not None
    )


def _admission_row(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    message_id: int,
) -> sqlite3.Row:
    row = conn.execute(
        """SELECT direction, metadata_json FROM instance_messages
        WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
        (profile_id, instance_id, int(message_id)),
    ).fetchone()
    if row is None or str(row["direction"]) != "INBOUND":
        raise ValueError("inbound admission must reference a real inbound message")
    return row


def _metadata_and_marker(row: sqlite3.Row) -> tuple[dict[str, object], dict[str, object]]:
    metadata = _json_object(row["metadata_json"])
    raw_marker = metadata.get(ADMISSION_METADATA_KEY)
    marker = dict(raw_marker) if isinstance(raw_marker, dict) else {}
    metadata[ADMISSION_METADATA_KEY] = marker
    return metadata, marker


def _owns_marker(
    marker: dict[str, object],
    lease_owner: str,
    lease_token: int,
    *,
    now: datetime,
) -> bool:
    lease_until = _parse(str(marker.get("lease_until") or ""))
    return bool(
        marker.get("status") == "ADMITTING"
        and str(marker.get("lease_owner") or "") == str(lease_owner)
        and int(marker.get("lease_token") or 0) == int(lease_token)
        and lease_until is not None
        and lease_until > now
    )


def _persist_marker(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    message_id: int,
    metadata: dict[str, object],
    *,
    expected_metadata: str,
) -> bool:
    cursor = conn.execute(
        """UPDATE instance_messages SET metadata_json = ?
        WHERE profile_id = ? AND instance_id = ? AND message_id = ?
          AND metadata_json = ?""",
        (
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
            profile_id,
            instance_id,
            int(message_id),
            expected_metadata,
        ),
    )
    return cursor.rowcount == 1


def _current_activity_epoch(conn: sqlite3.Connection, profile_id: str, instance_id: str) -> int:
    row = conn.execute(
        """SELECT activity_epoch FROM instance_core_state
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    if row is None:
        raise KeyError((profile_id, instance_id))
    return int(row["activity_epoch"])


def _invalidate_contact_clock_sql(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    *,
    activity_epoch: int,
    defer_until: datetime,
    now_text: str,
) -> None:
    conn.execute(
        """UPDATE instance_contact_state SET
        next_check_at = CASE WHEN next_check_at IS NULL OR next_check_at < ?
            THEN ? ELSE next_check_at END,
        lease_until = NULL, lease_token = lease_token + 1,
        generation = generation + 1, version = version + 1,
        activity_epoch_snapshot = ?, evidence_snapshot_json = '{}',
        last_result = 'INVALIDATED', last_reason = 'foreground-activity',
        updated_at = ? WHERE profile_id = ? AND instance_id = ?""",
        (
            _dt(defer_until),
            _dt(defer_until),
            int(activity_epoch),
            now_text,
            profile_id,
            instance_id,
        ),
    )


def _json_object(raw: object) -> dict[str, object]:
    try:
        parsed = json.loads(str(raw or "{}"))
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


__all__ = [
    "ADMISSION_METADATA_KEY",
    "InboundAdmissionResult",
    "apply_inbound_admission_sql",
    "claim_expired_inbound_admission_sql",
    "complete_inbound_admission_sql",
    "renew_inbound_admission_sql",
]
