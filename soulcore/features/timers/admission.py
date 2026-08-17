"""Public contracts for persistent Timer admission."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from .domain import (
    DeliveryAssociationRef,
    ExecutionEnvelopeRef,
    TimerOccurrence,
    TimerOccurrenceId,
    TimerScope,
    require_aware,
)
from .repository import InstanceOccupancy

_SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")


def _reference(value: str) -> str:
    normalized = str(value).strip()
    if _SAFE_REF.fullmatch(normalized) is None:
        raise ValueError("invalid admission reference")
    return normalized


class TimerClaimOutcome(StrEnum):
    CLAIMED = "CLAIMED"
    PLAYER_WAITING = "PLAYER_WAITING"
    OCCUPIED = "OCCUPIED"
    EMPTY = "EMPTY"


class TimerProviderOutcome(StrEnum):
    STARTED = "STARTED"
    SUPERSEDED = "SUPERSEDED"


class TimerAdmissionFenceError(RuntimeError):
    """Raised when a stale or expired executor attempts a state transition."""


@dataclass(frozen=True, slots=True)
class TimerRunFence:
    scope: TimerScope
    occurrence_id: TimerOccurrenceId
    occurrence_version: int
    generation: int
    occupancy_id: str
    occupancy_version: int
    lease_owner: str
    lease_token: str

    def __post_init__(self) -> None:
        for name in ("occupancy_id", "lease_owner", "lease_token"):
            object.__setattr__(self, name, _reference(getattr(self, name)))
        if self.occurrence_version < 1 or self.occupancy_version < 1 or self.generation < 0:
            raise ValueError("invalid admission fence version")

    def as_metadata(self) -> dict[str, object]:
        return {
            "profile_id": self.scope.profile_id,
            "instance_id": self.scope.instance_id,
            "occurrence_id": self.occurrence_id.value,
            "occurrence_version": self.occurrence_version,
            "generation": self.generation,
            "occupancy_id": self.occupancy_id,
            "occupancy_version": self.occupancy_version,
            "lease_owner": self.lease_owner,
            "lease_token": self.lease_token,
        }

    @classmethod
    def from_metadata(cls, value: object) -> TimerRunFence | None:
        if not isinstance(value, dict):
            return None
        try:
            return cls(
                TimerScope(str(value["profile_id"]), str(value["instance_id"])),
                TimerOccurrenceId(str(value["occurrence_id"])),
                int(value["occurrence_version"]),
                int(value["generation"]),
                str(value["occupancy_id"]),
                int(value["occupancy_version"]),
                str(value["lease_owner"]),
                str(value["lease_token"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True, slots=True)
class ClaimNextTimerCommand:
    scope: TimerScope
    occupancy_id: str
    lease_owner: str
    lease_token: str
    lease_expires_at: datetime
    now: datetime

    def __post_init__(self) -> None:
        for name in ("occupancy_id", "lease_owner", "lease_token"):
            object.__setattr__(self, name, _reference(getattr(self, name)))
        object.__setattr__(self, "lease_expires_at", require_aware(self.lease_expires_at))
        object.__setattr__(self, "now", require_aware(self.now))
        if self.lease_expires_at <= self.now:
            raise ValueError("admission lease must expire after now")


@dataclass(frozen=True, slots=True)
class TimerAdmissionResult:
    outcome: TimerClaimOutcome
    occurrence: TimerOccurrence | None = None
    occupancy: InstanceOccupancy | None = None
    fence: TimerRunFence | None = None
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class BeginTimerProviderCommand:
    fence: TimerRunFence
    execution_ref: ExecutionEnvelopeRef
    now: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "now", require_aware(self.now))


@dataclass(frozen=True, slots=True)
class TimerProviderResult:
    outcome: TimerProviderOutcome
    occurrence: TimerOccurrence
    occupancy: InstanceOccupancy
    fence: TimerRunFence | None


@dataclass(frozen=True, slots=True)
class CompleteTimerNoOpCommand:
    fence: TimerRunFence
    now: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "now", require_aware(self.now))


@dataclass(frozen=True, slots=True)
class SupersedeTimerRunCommand:
    fence: TimerRunFence
    now: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "now", require_aware(self.now))


@dataclass(frozen=True, slots=True)
class RetryTimerRunCommand:
    """Return a failed provider run to WAITING without changing its generation."""

    fence: TimerRunFence
    now: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "now", require_aware(self.now))


@dataclass(frozen=True, slots=True)
class HandoffTimerExpressionCommand:
    fence: TimerRunFence
    delivery_ref: DeliveryAssociationRef
    source_run_id: int
    now: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "now", require_aware(self.now))
        if self.source_run_id < 1:
            raise ValueError("source_run_id must be positive")


@dataclass(frozen=True, slots=True)
class TimerSettlementResult:
    occurrence: TimerOccurrence
    occupancy: InstanceOccupancy


class TimerAdmissionPort(Protocol):
    async def is_instance_occupied(self, scope: TimerScope) -> bool: ...

    async def claim_next_timer(self, command: ClaimNextTimerCommand) -> TimerAdmissionResult: ...

    async def begin_timer_provider(
        self, command: BeginTimerProviderCommand
    ) -> TimerProviderResult: ...

    async def complete_timer_noop(
        self, command: CompleteTimerNoOpCommand
    ) -> TimerSettlementResult: ...

    async def supersede_timer_run(
        self, command: SupersedeTimerRunCommand
    ) -> TimerSettlementResult: ...

    async def retry_timer_run(self, command: RetryTimerRunCommand) -> TimerSettlementResult: ...

    async def handoff_timer_expression(
        self, command: HandoffTimerExpressionCommand
    ) -> TimerSettlementResult: ...

    async def reconcile_timer_occupancy(self, scope: TimerScope, *, now: datetime) -> int: ...


__all__ = [
    "BeginTimerProviderCommand",
    "ClaimNextTimerCommand",
    "CompleteTimerNoOpCommand",
    "HandoffTimerExpressionCommand",
    "RetryTimerRunCommand",
    "SupersedeTimerRunCommand",
    "TimerAdmissionFenceError",
    "TimerAdmissionPort",
    "TimerAdmissionResult",
    "TimerClaimOutcome",
    "TimerProviderOutcome",
    "TimerProviderResult",
    "TimerRunFence",
    "TimerSettlementResult",
]
