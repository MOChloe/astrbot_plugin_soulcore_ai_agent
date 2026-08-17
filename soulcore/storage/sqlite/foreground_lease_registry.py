"""Durable foreground lease holder registry and release transition."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from .codec import _dump, _load, _parse
from .foreground_activity import (
    foreground_activity_is_active_sql,
    foreground_terminal_watermark_sql,
)

FOREGROUND_DRAIN_HOLDER = "__soulcore_foreground_drain__"


def required_foreground_lease_identity(owner: str, token: str) -> tuple[str, str]:
    """Normalize one real holder identity and reject the internal drain owner."""

    normalized_owner = str(owner or "").strip()
    normalized_token = str(token or "").strip()
    if not normalized_owner or not normalized_token or normalized_owner == FOREGROUND_DRAIN_HOLDER:
        raise ValueError("foreground lease owner and token are required")
    return normalized_owner, normalized_token


def foreground_lease_holders(
    value: object,
    *,
    expected_count: int,
) -> dict[str, int]:
    """Load and validate the durable holder registry."""

    try:
        loaded = _load(str(value)) if isinstance(value, str) else value
    except (TypeError, ValueError) as exc:
        raise RuntimeError("foreground lease holder registry is invalid") from exc
    if not isinstance(loaded, Mapping):
        raise RuntimeError("foreground lease holder registry must be a JSON object")
    holders: dict[str, int] = {}
    for raw_owner, raw_count in loaded.items():
        owner = str(raw_owner or "").strip()
        if (
            not isinstance(raw_owner, str)
            or owner != raw_owner
            or not owner
            or type(raw_count) is not int
            or raw_count <= 0
        ):
            raise RuntimeError("foreground lease holder registry entry is invalid")
        holders[owner] = raw_count
    if sum(holders.values()) != int(expected_count):
        raise RuntimeError("foreground lease holder registry count is inconsistent")
    return holders


def dump_foreground_lease_holders(holders: Mapping[str, int]) -> str:
    """Serialize the registry deterministically for compare-and-swap updates."""

    return _dump({owner: int(holders[owner]) for owner in sorted(holders)})


@dataclass(frozen=True, slots=True)
class _ReleaseProjection:
    holders: dict[str, int]
    owner: str | None
    token: str | None
    lease_until: str | None
    logical_released_at: str

    @property
    def count(self) -> int:
        return sum(self.holders.values())


def _foreground_end(current: sqlite3.Row, released_at: str) -> datetime:
    released = _parse(released_at)
    if released is None:
        raise ValueError("foreground lease release time is invalid")
    last_foreground = _parse(current["last_foreground_at"])
    return max(value for value in (released, last_foreground) if value is not None)


def _remove_owner_reference(
    holders: Mapping[str, int],
    owner: str,
) -> dict[str, int] | None:
    owner_count = int(holders.get(owner, 0))
    if owner_count <= 0:
        return None
    remaining = dict(holders)
    if owner_count == 1:
        remaining.pop(owner)
    else:
        remaining[owner] = owner_count - 1
    return remaining


def _release_projection(
    conn: sqlite3.Connection,
    current: sqlite3.Row,
    *,
    profile_id: str,
    instance_id: str,
    token: str,
    released_at: str,
    holders: dict[str, int],
) -> _ReleaseProjection:
    foreground_end = _foreground_end(current, released_at)
    logical_released_at = foreground_end.isoformat()
    only_drain_remains = sum(holders.values()) == 1 and holders.get(FOREGROUND_DRAIN_HOLDER) == 1
    closing_generation = not holders or only_drain_remains
    if closing_generation and foreground_activity_is_active_sql(
        conn,
        profile_id=profile_id,
        instance_id=instance_id,
        now=released_at,
    ):
        return _ReleaseProjection(
            holders={FOREGROUND_DRAIN_HOLDER: 1},
            owner=FOREGROUND_DRAIN_HOLDER,
            token=token,
            lease_until=released_at,
            logical_released_at=logical_released_at,
        )
    if closing_generation:
        return _ReleaseProjection(
            holders={},
            owner=None,
            token=None,
            lease_until=None,
            logical_released_at=logical_released_at,
        )
    return _ReleaseProjection(
        holders=holders,
        owner=sorted(holders)[0],
        token=token,
        lease_until=str(current["foreground_lease_until"]),
        logical_released_at=logical_released_at,
    )


def release_foreground_lease_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    owner: str,
    token: str,
    released_at: str,
) -> bool:
    """Release this owner's reference and settle the foreground boundary."""

    normalized_owner, normalized_token = required_foreground_lease_identity(owner, token)
    current = conn.execute(
        """SELECT foreground_lease_owner, foreground_lease_token,
            foreground_lease_until, foreground_lease_holders_json,
            foreground_lease_count, last_foreground_at
        FROM background_instances
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    if current is None or str(current["foreground_lease_token"] or "") != normalized_token:
        return False
    current_count = int(current["foreground_lease_count"] or 0)
    current_holders_json = str(current["foreground_lease_holders_json"])
    holders = foreground_lease_holders(
        current_holders_json,
        expected_count=current_count,
    )
    remaining = _remove_owner_reference(holders, normalized_owner)
    if current_count <= 0 or remaining is None:
        return False
    projection = _release_projection(
        conn,
        current,
        profile_id=profile_id,
        instance_id=instance_id,
        token=normalized_token,
        released_at=released_at,
        holders=remaining,
    )
    return (
        conn.execute(
            """UPDATE background_instances
            SET foreground_lease_count = ?,
                foreground_lease_holders_json = ?,
                foreground_lease_owner = ?,
                foreground_lease_token = ?,
                foreground_lease_until = ?,
                continuity_version = continuity_version + 1,
                last_foreground_at = CASE
                    WHEN last_foreground_at IS NULL OR last_foreground_at < ? THEN ?
                    ELSE last_foreground_at END,
                updated_at = ?
            WHERE profile_id = ? AND instance_id = ?
              AND foreground_lease_token = ?
              AND foreground_lease_holders_json = ?
              AND foreground_lease_count = ?""",
            (
                projection.count,
                dump_foreground_lease_holders(projection.holders),
                projection.owner,
                projection.token,
                projection.lease_until,
                projection.logical_released_at,
                projection.logical_released_at,
                projection.logical_released_at,
                profile_id,
                instance_id,
                normalized_token,
                current_holders_json,
                current_count,
            ),
        ).rowcount
        == 1
    )


def recover_expired_foreground_leases_sql(
    conn: sqlite3.Connection,
    *,
    now: str,
) -> int:
    """Release crashed foreground generations without scheduling background work."""

    if _parse(now) is None:
        raise ValueError("expired foreground recovery time is invalid")
    rows = conn.execute(
        """SELECT profile_id, instance_id, foreground_lease_token,
            foreground_lease_until, foreground_lease_holders_json,
            foreground_lease_count, last_foreground_at
        FROM background_instances
        WHERE foreground_lease_count > 0
          AND (foreground_lease_until IS NULL OR foreground_lease_until <= ?)""",
        (now,),
    ).fetchall()
    recovered = 0
    for row in rows:
        profile_id = str(row["profile_id"])
        instance_id = str(row["instance_id"])
        if foreground_activity_is_active_sql(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            now=now,
        ):
            continue
        recovery_at = foreground_terminal_watermark_sql(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            now=now,
        ).isoformat()
        changed = conn.execute(
            """UPDATE background_instances
            SET foreground_lease_owner = NULL,
                foreground_lease_token = NULL,
                foreground_lease_until = NULL,
                foreground_lease_holders_json = '{}',
                foreground_lease_count = 0,
                continuity_version = continuity_version + 1,
                last_foreground_at = CASE
                    WHEN last_foreground_at IS NULL OR last_foreground_at < ? THEN ?
                    ELSE last_foreground_at END,
                updated_at = ?
            WHERE profile_id = ? AND instance_id = ?
              AND foreground_lease_count = ?
              AND foreground_lease_holders_json = ?
              AND foreground_lease_token IS ?
              AND foreground_lease_until IS ?""",
            (
                recovery_at,
                recovery_at,
                now,
                profile_id,
                instance_id,
                int(row["foreground_lease_count"]),
                row["foreground_lease_holders_json"],
                row["foreground_lease_token"],
                row["foreground_lease_until"],
            ),
        ).rowcount
        if changed != 1:
            raise RuntimeError("expired foreground lease changed during recovery")
        recovered += 1
    return recovered


__all__ = [
    "dump_foreground_lease_holders",
    "foreground_lease_holders",
    "recover_expired_foreground_leases_sql",
    "release_foreground_lease_sql",
    "required_foreground_lease_identity",
]
