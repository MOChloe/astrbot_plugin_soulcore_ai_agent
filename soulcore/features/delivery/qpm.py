"""Atomic rolling-window quota reservations for SoulCore-owned sends.

The application layer deliberately depends on :class:`QPMStorage` instead of
SQLite.  The in-memory implementation is deterministic and useful for tests;
the repository adapter can implement the same protocol with one transaction
covering every bucket attached to a reservation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

ROLLING_WINDOW_SECONDS = 60
DEFAULT_RESERVATION_TTL_SECONDS = 120


def utc_now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class QPMBucketScope(StrEnum):
    GROUP = "group"
    QQ_ACCOUNT_PROACTIVE = "qq_account_proactive"


@dataclass(frozen=True, slots=True)
class QPMBucketKey:
    scope: QPMBucketScope
    identity: str


@dataclass(frozen=True, slots=True)
class QPMBucketLimit:
    key: QPMBucketKey
    limit: int

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("QPM limit must be positive")


class QPMAttemptState(StrEnum):
    DISPATCHING = "dispatching"
    ATTEMPTED_UNKNOWN = "attempted_unknown"


@dataclass(frozen=True, slots=True)
class QPMReservation:
    reservation_id: str
    profile_id: str
    instance_id: str
    origin_kind: str
    buckets: tuple[QPMBucketLimit, ...]
    units: int
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class QPMReservationDecision:
    allowed: bool
    reservation: QPMReservation | None = None
    blocked_bucket: QPMBucketKey | None = None
    available_units: int = 0
    reason: str = ""


@dataclass(frozen=True, slots=True)
class QPMDispatchDecision:
    allowed: bool
    attempt_id: str | None = None
    already_started: bool = False
    reason: str = ""


@dataclass(frozen=True, slots=True)
class QPMDispatchFence:
    """Persistent business fence joined to the exact dispatch transition."""

    profile_id: str
    instance_id: str
    group_window_id: str
    outbox_id: int


@dataclass(frozen=True, slots=True)
class QPMDispatchPreparation:
    allowed: bool
    payload: dict[str, Any] | None = None
    already_started: bool = False
    reason: str = ""


@dataclass(frozen=True, slots=True)
class QPMBucketSnapshot:
    key: QPMBucketKey
    limit: int
    attempted: int
    reserved: int
    remaining: int
    next_available_at: datetime | None
    window_seconds: int = ROLLING_WINDOW_SECONDS
    coverage: str = "soulcore_owned_sends_only"


class QPMStorage(Protocol):
    """Persistence boundary for atomic multi-bucket QPM accounting.

    A durable implementation must make ``reserve`` and ``resize`` atomic
    across all supplied buckets.  ``begin_dispatch`` reserves the exact
    physical attempt immediately before the sender boundary.  A live caller
    that is cancelled before entering the sender must close that attempt with
    ``fail_before_platform_call``; a process crash still recovers a persisted
    DISPATCHING attempt as unknown.
    """

    async def reserve(
        self,
        *,
        profile_id: str,
        instance_id: str = "",
        origin_kind: str = "soulcore_send",
        origin_id: str | None = None,
        buckets: Sequence[QPMBucketLimit],
        units: int,
        now: datetime | None = None,
        ttl_seconds: int = DEFAULT_RESERVATION_TTL_SECONDS,
    ) -> QPMReservationDecision: ...

    async def resize(
        self,
        reservation_id: str,
        units: int,
        *,
        now: datetime | None = None,
    ) -> QPMReservationDecision: ...

    async def renew(
        self,
        reservation_id: str,
        *,
        now: datetime | None = None,
        ttl_seconds: int = DEFAULT_RESERVATION_TTL_SECONDS,
    ) -> QPMReservation | None: ...

    async def prepare_dispatch(
        self,
        reservation_id: str,
        chunk_index: int,
        *,
        fence: QPMDispatchFence | None = None,
        now: datetime | None = None,
    ) -> QPMDispatchPreparation: ...

    async def begin_dispatch(
        self,
        reservation_id: str,
        chunk_index: int,
        *,
        fence: QPMDispatchFence | None = None,
        now: datetime | None = None,
    ) -> QPMDispatchDecision: ...

    async def mark_attempted_unknown(
        self,
        attempt_id: str,
        *,
        detail: str = "",
        now: datetime | None = None,
        fence: QPMDispatchFence | None = None,
    ) -> None: ...

    async def fail_before_platform_call(
        self,
        attempt_id: str,
        *,
        detail: str = "",
        now: datetime | None = None,
    ) -> bool: ...

    async def release(
        self,
        reservation_id: str,
        *,
        now: datetime | None = None,
    ) -> None: ...

    async def snapshots(
        self,
        buckets: Sequence[QPMBucketLimit],
        *,
        now: datetime | None = None,
    ) -> tuple[QPMBucketSnapshot, ...]: ...


__all__ = [
    "QPMStorage",
    "QPMBucketScope",
    "QPMBucketKey",
    "QPMBucketLimit",
    "QPMAttemptState",
    "QPMReservation",
    "QPMReservationDecision",
    "QPMDispatchDecision",
    "QPMDispatchFence",
    "QPMDispatchPreparation",
    "QPMBucketSnapshot",
]
