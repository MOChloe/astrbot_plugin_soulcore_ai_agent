"""Typed persistence commands for Timer storage and instance occupancy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from .domain import (
    DeliveryAssociationRef,
    ExecutionEnvelopeRef,
    IdempotencyKey,
    TimerOccurrence,
    TimerOccurrenceId,
    TimerRule,
    TimerScope,
    require_aware,
)
from .errors import TimerErrorCode, fail
from .transitions import OccurrenceAction

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")


def _token(value: str) -> str:
    if not isinstance(value, str) or _SAFE_TOKEN.fullmatch(value) is None:
        raise fail(TimerErrorCode.INVALID_REFERENCE)
    return value


class InstanceOccupancyKind(StrEnum):
    PLAYER = "PLAYER"
    TIMER = "TIMER"
    EXPRESSION = "EXPRESSION"


class InstanceOccupancyStatus(StrEnum):
    ACTIVE = "ACTIVE"
    RELEASED = "RELEASED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True, slots=True)
class InstanceOccupancy:
    occupancy_id: str
    scope: TimerScope
    kind: InstanceOccupancyKind
    resource_ref: str
    status: InstanceOccupancyStatus
    version: int
    generation: int
    lease_owner: str
    lease_token: str
    lease_expires_at: datetime
    created_at: datetime
    updated_at: datetime
    released_at: datetime | None = None

    def __post_init__(self) -> None:
        for value in (
            self.occupancy_id,
            self.resource_ref,
            self.lease_owner,
            self.lease_token,
        ):
            _token(value)
        for name in ("lease_expires_at", "created_at", "updated_at"):
            object.__setattr__(self, name, require_aware(getattr(self, name)))
        if self.released_at is not None:
            object.__setattr__(self, "released_at", require_aware(self.released_at))
        if self.version < 1 or self.generation < 0:
            raise fail(TimerErrorCode.INVALID_STATE)
        if (self.status is InstanceOccupancyStatus.ACTIVE) != (self.released_at is None):
            raise fail(TimerErrorCode.INVALID_STATE)


@dataclass(frozen=True, slots=True)
class RulePage:
    items: tuple[TimerRule, ...]
    next_created_sequence: int | None


@dataclass(frozen=True, slots=True)
class OccurrencePage:
    items: tuple[TimerOccurrence, ...]
    next_cursor: tuple[datetime, int, str] | None


@dataclass(frozen=True, slots=True)
class AdvanceOccurrenceCommand:
    scope: TimerScope
    occurrence_id: TimerOccurrenceId
    action: OccurrenceAction
    expected_version: int
    expected_generation: int
    now: datetime
    idempotency_key: IdempotencyKey

    def __post_init__(self) -> None:
        object.__setattr__(self, "now", require_aware(self.now))
        if self.action not in {
            OccurrenceAction.MARK_DUE,
            OccurrenceAction.MARK_MISSED_COALESCED,
        }:
            raise fail(TimerErrorCode.INVALID_STATE)
        if self.expected_version < 1 or self.expected_generation < 0:
            raise fail(TimerErrorCode.VERSION_CONFLICT)


@dataclass(frozen=True, slots=True)
class MutateClaimedOccurrenceCommand:
    scope: TimerScope
    occurrence_id: TimerOccurrenceId
    action: OccurrenceAction
    expected_version: int
    expected_generation: int
    occupancy_id: str
    expected_occupancy_version: int
    lease_owner: str
    lease_token: str
    now: datetime
    idempotency_key: IdempotencyKey
    execution_ref: ExecutionEnvelopeRef | None = None
    delivery_ref: DeliveryAssociationRef | None = None

    def __post_init__(self) -> None:
        for value in (self.occupancy_id, self.lease_owner, self.lease_token):
            _token(value)
        object.__setattr__(self, "now", require_aware(self.now))
        if (
            self.expected_version < 1
            or self.expected_generation < 0
            or self.expected_occupancy_version < 1
        ):
            raise fail(TimerErrorCode.VERSION_CONFLICT)


@dataclass(frozen=True, slots=True)
class OccurrenceMutationResult:
    occurrence: TimerOccurrence
    occupancy: InstanceOccupancy
    replayed: bool = False


__all__ = [
    "AdvanceOccurrenceCommand",
    "InstanceOccupancy",
    "InstanceOccupancyKind",
    "InstanceOccupancyStatus",
    "MutateClaimedOccurrenceCommand",
    "OccurrenceMutationResult",
    "OccurrencePage",
    "RulePage",
]
