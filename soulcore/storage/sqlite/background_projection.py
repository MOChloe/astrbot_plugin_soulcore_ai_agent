"""Foreground-to-background continuity and exclusion projection.

The conversation ledger remains the only copy of foreground speech.  This
module only advances the background continuity fence, owns the short-lived
foreground lease, and invalidates background author work that was snapshotted
before foreground activity began.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from ...contracts.delivery_visibility import DeliveryVisibility, delivery_visibility
from ...features.background.domain import (
    DEFAULT_ORDINARY_MAX_MINUTES,
    DEFAULT_ORDINARY_MIN_MINUTES,
)
from .codec import _dump, _load
from .foreground_activity import (
    foreground_activity_is_active_sql,
)
from .foreground_lease_registry import (
    dump_foreground_lease_holders as _dump_foreground_lease_holders,
)
from .foreground_lease_registry import (
    foreground_lease_holders as _foreground_lease_holders,
)
from .foreground_lease_registry import (
    release_foreground_lease_sql,
)
from .foreground_lease_registry import (
    required_foreground_lease_identity as _required_foreground_lease_identity,
)

_AUTHOR_KINDS = ("WORLD", "LIFE_DIRECTION", "STORY_SOURCE", "KEYFRAME", "ORDINARY")
_PROJECTION_METADATA_KEY = "background_foreground_projection"
_PROJECTION_VERSION = 1
_QUEUED_BACKGROUND_STATUSES = (
    "SCHEDULED",
    "READY",
    "PAUSED",
    "RETRY_WAIT",
    "RECOVERY_REQUIRED",
)
_LEASED_BACKGROUND_STATUSES = ("RUNNING", "PAUSE_REQUESTED", "CANCEL_REQUESTED")


def ensure_background_instance_sql(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    now: str,
) -> None:
    """Create one independent world line and all five author slots if absent."""

    seed = conn.execute(
        """SELECT world.life_direction
        FROM role_profiles profile
        LEFT JOIN world_definitions world ON world.profile_id = profile.profile_id
        WHERE profile.profile_id = ?""",
        (profile_id,),
    ).fetchone()
    if seed is None:
        raise KeyError(profile_id)
    initial_direction = str(seed["life_direction"] or "")
    conn.execute(
        """INSERT INTO background_instances(
            profile_id, instance_id, initial_life_direction,
            ordinary_min_minutes, ordinary_max_minutes, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(profile_id, instance_id) DO NOTHING""",
        (
            profile_id,
            instance_id,
            initial_direction,
            DEFAULT_ORDINARY_MIN_MINUTES,
            DEFAULT_ORDINARY_MAX_MINUTES,
            now,
            now,
        ),
    )
    for author_kind in _AUTHOR_KINDS:
        conn.execute(
            """INSERT INTO background_author_states(
                profile_id, instance_id, author_kind, next_due_at, hard_due_at,
                created_at, updated_at
            ) VALUES (?, ?, ?, NULL, NULL, ?, ?)
            ON CONFLICT(profile_id, instance_id, author_kind) DO NOTHING""",
            (profile_id, instance_id, author_kind, now, now),
        )
    conn.execute(
        """INSERT INTO background_role_current_views(
            profile_id, instance_id, as_of, source, created_at, updated_at
        ) VALUES (?, ?, ?, 'INITIALIZATION', ?, ?)
        ON CONFLICT(profile_id, instance_id) DO NOTHING""",
        (profile_id, instance_id, now, now, now),
    )


def _advance_foreground_clock_sql(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    *,
    through_at: str,
    now: str,
) -> None:
    """Fence background snapshots without turning chat activity into a frame signal."""

    conn.execute(
        """UPDATE background_instances
        SET continuity_version = continuity_version + 1,
            last_foreground_at = CASE
                WHEN last_foreground_at IS NULL OR last_foreground_at < ? THEN ?
                ELSE last_foreground_at END,
            updated_at = ?
        WHERE profile_id = ? AND instance_id = ?""",
        (through_at, through_at, now, profile_id, instance_id),
    )


def project_foreground_message_continuity_sql(
    conn: sqlite3.Connection,
    row: sqlite3.Row | Mapping[str, Any],
    *,
    settled_at: str,
) -> int | None:
    """Project one visible ledger row into a continuity fence.

    No second ledger is created: Ordinary/Keyframe later consume the
    authoritative ``instance_messages`` rows by cursor.
    """

    direction = str(row["direction"] or "").upper()
    status = str(row["delivery_status"] or "").upper()
    visibility = delivery_visibility(direction, status)
    if visibility is DeliveryVisibility.NOT_VISIBLE:
        return None

    profile_id = str(row["profile_id"])
    instance_id = str(row["instance_id"])
    message_id = int(row["message_id"])
    metadata = _json_object(row["metadata_json"])
    marker = metadata.get(_PROJECTION_METADATA_KEY)
    if isinstance(marker, Mapping) and marker.get("version") == _PROJECTION_VERSION:
        return message_id

    ensure_background_instance_sql(conn, profile_id, instance_id, settled_at)
    if direction == "INBOUND":
        _acquire_initial_inbound_lease(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            message_id=message_id,
            metadata=metadata,
            now=settled_at,
        )
    _advance_foreground_clock_sql(
        conn,
        profile_id,
        instance_id,
        # Platform event timestamps are evidence metadata, not a trusted
        # simulation clock.  A future-skewed timestamp must not place the
        # foreground boundary after the locally observed settlement.
        through_at=settled_at,
        now=settled_at,
    )
    _invalidate_background_tasks_sql(
        conn,
        profile_id,
        instance_id,
        reason=f"foreground_message:{message_id}",
        now=settled_at,
    )
    projection_marker = metadata.get(_PROJECTION_METADATA_KEY)
    projection_marker = dict(projection_marker) if isinstance(projection_marker, Mapping) else {}
    projection_marker.update(
        version=_PROJECTION_VERSION,
        projected_at=settled_at,
    )
    metadata[_PROJECTION_METADATA_KEY] = projection_marker
    conn.execute(
        """UPDATE instance_messages SET metadata_json = ?
        WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
        (_dump(metadata), profile_id, instance_id, message_id),
    )
    return message_id


def project_foreground_retraction_continuity_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    source_message_id: int,
    retraction_key: str,
    settled_at: str,
    metadata: Mapping[str, Any] | None = None,
) -> int | None:
    """Fence a foreground retraction without manufacturing a correction fact."""

    del metadata
    ensure_background_instance_sql(conn, profile_id, instance_id, settled_at)
    _advance_foreground_clock_sql(
        conn,
        profile_id,
        instance_id,
        through_at=settled_at,
        now=settled_at,
    )
    _invalidate_background_tasks_sql(
        conn,
        profile_id,
        instance_id,
        reason=f"foreground_retraction:{str(retraction_key).strip()}",
        now=settled_at,
    )
    return int(source_message_id)


def _shares_foreground_generation(
    conn: sqlite3.Connection,
    *,
    current_token: str,
    current_until: str,
    current_count: int,
    profile_id: str,
    instance_id: str,
    now: str,
) -> bool:
    if not current_token or current_count <= 0:
        return False
    if current_until > now:
        return True
    return foreground_activity_is_active_sql(
        conn,
        profile_id=profile_id,
        instance_id=instance_id,
        now=now,
        include_unprojected_messages=False,
    )


def acquire_foreground_lease_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    owner: str,
    token: str,
    lease_until: str,
    now: str,
) -> str:
    """Acquire one reference in the current foreground lease generation."""

    normalized_owner, normalized_token = _required_foreground_lease_identity(
        owner,
        token,
    )
    ensure_background_instance_sql(conn, profile_id, instance_id, now)
    current = conn.execute(
        """SELECT foreground_lease_owner, foreground_lease_token,
            foreground_lease_until, foreground_lease_holders_json,
            foreground_lease_count
        FROM background_instances
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    if current is None:
        raise RuntimeError("background instance disappeared during foreground lease acquisition")
    current_token = str(current["foreground_lease_token"] or "").strip()
    current_until = str(current["foreground_lease_until"] or "")
    current_count = int(current["foreground_lease_count"] or 0)
    current_holders = _foreground_lease_holders(
        current["foreground_lease_holders_json"],
        expected_count=current_count,
    )
    shares_generation = _shares_foreground_generation(
        conn,
        current_token=current_token,
        current_until=current_until,
        current_count=current_count,
        profile_id=profile_id,
        instance_id=instance_id,
        now=now,
    )
    effective_token = current_token if shares_generation else normalized_token
    holders = dict(current_holders) if shares_generation else {}
    holders[normalized_owner] = 1
    holder_count = sum(holders.values())
    effective_owner = (
        str(current["foreground_lease_owner"]) if shares_generation else normalized_owner
    )
    changed = conn.execute(
        """UPDATE background_instances
        SET foreground_lease_owner = ?,
            foreground_lease_token = ?,
            foreground_lease_until = CASE
                WHEN ? = 1 AND foreground_lease_until > ? THEN foreground_lease_until
                ELSE ? END,
            foreground_lease_holders_json = ?,
            foreground_lease_count = ?,
            continuity_version = continuity_version + 1,
            last_foreground_at = CASE
                WHEN last_foreground_at IS NULL OR last_foreground_at < ? THEN ?
                ELSE last_foreground_at END,
            updated_at = ?
        WHERE profile_id = ? AND instance_id = ?""",
        (
            effective_owner,
            effective_token,
            int(shares_generation),
            lease_until,
            lease_until,
            _dump_foreground_lease_holders(holders),
            holder_count,
            now,
            now,
            now,
            profile_id,
            instance_id,
        ),
    ).rowcount
    if changed != 1:
        raise RuntimeError("foreground/background exclusion lease could not be acquired")
    _invalidate_background_tasks_sql(
        conn,
        profile_id,
        instance_id,
        reason=f"foreground_lease:{normalized_owner}",
        now=now,
    )
    return effective_token


def renew_foreground_lease_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    owner: str,
    token: str,
    lease_until: str,
    now: str,
) -> bool:
    """Extend a live generation only while this owner still holds it."""

    normalized_owner, normalized_token = _required_foreground_lease_identity(owner, token)
    current = conn.execute(
        """SELECT foreground_lease_token, foreground_lease_until,
            foreground_lease_holders_json, foreground_lease_count
        FROM background_instances
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    if current is None:
        return False
    current_count = int(current["foreground_lease_count"] or 0)
    holders = _foreground_lease_holders(
        current["foreground_lease_holders_json"],
        expected_count=current_count,
    )
    if (
        str(current["foreground_lease_token"] or "") != normalized_token
        or normalized_owner not in holders
        or current_count <= 0
        or str(current["foreground_lease_until"] or "") <= now
    ):
        return False
    return (
        conn.execute(
            """UPDATE background_instances SET
                foreground_lease_until = CASE
                    WHEN foreground_lease_until > ? THEN foreground_lease_until
                    ELSE ? END,
                updated_at = ?
            WHERE profile_id = ? AND instance_id = ?
              AND foreground_lease_token = ?
              AND foreground_lease_holders_json = ?
              AND foreground_lease_count = ?
              AND foreground_lease_until > ?""",
            (
                lease_until,
                lease_until,
                now,
                profile_id,
                instance_id,
                normalized_token,
                str(current["foreground_lease_holders_json"]),
                current_count,
                now,
            ),
        ).rowcount
        == 1
    )


def renew_inbound_foreground_lease_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    message_id: int,
    metadata: Mapping[str, Any],
    lease_until: str,
    now: str,
) -> bool:
    """Renew the initial ledger lease while inbound admission still owns it."""

    lease = _inbound_projection_lease(metadata, int(message_id))
    if lease is None:
        return False
    owner, token = lease
    return renew_foreground_lease_sql(
        conn,
        profile_id=profile_id,
        instance_id=instance_id,
        owner=owner,
        token=token,
        lease_until=lease_until,
        now=now,
    )


def release_inbound_foreground_lease_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    message_id: int,
    metadata: Mapping[str, Any],
    released_at: str,
) -> bool:
    """Release the initial ledger lease after its durable handoff completed."""

    lease = _inbound_projection_lease(metadata, int(message_id))
    if lease is None:
        return False
    owner, token = lease
    return release_foreground_lease_sql(
        conn,
        profile_id=profile_id,
        instance_id=instance_id,
        owner=owner,
        token=token,
        released_at=released_at,
    )


def _acquire_initial_inbound_lease(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    message_id: int,
    metadata: dict[str, Any],
    now: str,
) -> None:
    admission = metadata.get("inbound_admission")
    if not isinstance(admission, Mapping):
        return
    lease_until = str(admission.get("lease_until") or "").strip()
    if not lease_until:
        return
    owner = f"foreground-inbound:{message_id}"
    token = uuid.uuid4().hex
    effective_token = acquire_foreground_lease_sql(
        conn,
        profile_id=profile_id,
        instance_id=instance_id,
        owner=owner,
        token=token,
        lease_until=lease_until,
        now=now,
    )
    metadata[_PROJECTION_METADATA_KEY] = {
        "version": _PROJECTION_VERSION,
        "projected_at": now,
        "lease_owner": owner,
        "lease_token": effective_token,
    }


def _inbound_projection_lease(
    metadata: Mapping[str, Any],
    message_id: int,
) -> tuple[str, str] | None:
    marker = metadata.get(_PROJECTION_METADATA_KEY)
    if not isinstance(marker, Mapping) or marker.get("version") != _PROJECTION_VERSION:
        return None
    owner = str(marker.get("lease_owner") or "").strip()
    token = str(marker.get("lease_token") or "").strip()
    if owner != f"foreground-inbound:{int(message_id)}" or not token:
        return None
    return owner, token


def _invalidate_background_tasks_sql(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    *,
    reason: str,
    now: str,
) -> int:
    rows = list(
        conn.execute(
            """SELECT * FROM ai_tasks
            WHERE profile_id = ? AND instance_id = ?
              AND task_type = 'BACKGROUND_AUTHOR'
              AND status NOT IN ('DEFERRED','SUCCEEDED','FAILED','CANCELLED')
            ORDER BY task_id""",
            (profile_id, instance_id),
        )
    )
    changed = 0
    for row in rows:
        previous_status = str(row["status"])
        if previous_status in _LEASED_BACKGROUND_STATUSES:
            target = "CANCEL_REQUESTED"
            finished_at = None
        elif previous_status in _QUEUED_BACKGROUND_STATUSES:
            target = "CANCELLED"
            finished_at = now
        else:
            target = "CANCELLED"
            finished_at = now
        if target == previous_status:
            continue
        updated = conn.execute(
            """UPDATE ai_tasks SET status = ?, last_error = ?, updated_at = ?,
                finished_at = COALESCE(?, finished_at), version = version + 1,
                lease_owner = CASE WHEN ? = 'CANCELLED' THEN NULL ELSE lease_owner END,
                lease_until = CASE WHEN ? = 'CANCELLED' THEN NULL ELSE lease_until END
            WHERE task_id = ? AND version = ? AND status = ?""",
            (
                target,
                reason,
                now,
                finished_at,
                target,
                target,
                int(row["task_id"]),
                int(row["version"]),
                previous_status,
            ),
        ).rowcount
        if updated != 1:
            continue
        changed += 1
        conn.execute(
            """INSERT INTO ai_task_audit(
                task_id, profile_id, instance_id, actor_type, actor_id,
                action, from_status, to_status, details_json, created_at
            ) VALUES (?, ?, ?, 'SYSTEM', 'foreground_projection',
                'FOREGROUND_INVALIDATE', ?, ?, ?, ?)""",
            (
                int(row["task_id"]),
                profile_id,
                instance_id,
                previous_status,
                target,
                _dump({"reason": reason}),
                now,
            ),
        )
        if target == "CANCELLED" and row["workflow_id"] is not None:
            conn.execute(
                """UPDATE ai_workflows SET status = 'CANCELLED',
                    final_error_code = 'FOREGROUND_INVALIDATED',
                    final_message = ?, finished_at = ?, updated_at = ?,
                    version = version + 1
                WHERE workflow_id = ? AND status = 'RUNNING'""",
                (reason, now, now, int(row["workflow_id"])),
            )
    if rows:
        task_ids = tuple(int(row["task_id"]) for row in rows)
        placeholders = ",".join("?" for _ in task_ids)
        conn.execute(
            f"""UPDATE background_author_states
            SET status = 'IDLE', active_task_id = NULL,
                last_error = ?, schedule_version = schedule_version + 1,
                updated_at = ?
            WHERE profile_id = ? AND instance_id = ?
              AND active_task_id IN ({placeholders})""",
            (
                reason,
                now,
                profile_id,
                instance_id,
                *task_ids,
            ),
        )
    return changed


def _json_object(value: Any) -> dict[str, Any]:
    loaded = _load(value) if value else {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def lease_deadline(now: datetime, lease_seconds: int) -> str:
    """Return an ISO deadline for repository lease adapters."""

    return (now + timedelta(seconds=max(1, int(lease_seconds)))).isoformat()


__all__ = [
    "acquire_foreground_lease_sql",
    "ensure_background_instance_sql",
    "lease_deadline",
    "project_foreground_message_continuity_sql",
    "project_foreground_retraction_continuity_sql",
    "release_foreground_lease_sql",
    "release_inbound_foreground_lease_sql",
    "renew_foreground_lease_sql",
    "renew_inbound_foreground_lease_sql",
]
