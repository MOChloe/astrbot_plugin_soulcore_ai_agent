"""Commands and results shared by later Timer adapters and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from .constants import MAX_SOURCE_REFS
from .domain import (
    IdempotencyKey,
    NormalizedTimerRule,
    OpaqueTimerRef,
    SourceMessageRef,
    SourceRunRef,
    TimerOccurrence,
    TimerRule,
    TimerRuleId,
    TimerScope,
    normalize_prompt,
    require_aware,
)
from .errors import TimerErrorCode, fail
from .projection import TimerRefTarget


class ManageTimerAction(StrEnum):
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    CANCEL = "CANCEL"


class CreateTimerOutcome(StrEnum):
    CREATED = "CREATED"
    ALREADY_EXISTS = "ALREADY_EXISTS"


class ManageTimerOutcome(StrEnum):
    APPLIED = "APPLIED"
    REPLAYED = "REPLAYED"
    TOO_LATE_OR_UNKNOWN = "TOO_LATE_OR_UNKNOWN"


@dataclass(frozen=True, slots=True)
class CreateTimerCommand:
    scope: TimerScope
    schedule: NormalizedTimerRule
    prompt: str = field(repr=False)
    fingerprint: str
    source_run_ref: SourceRunRef
    idempotency_key: IdempotencyKey
    source_message_refs: tuple[SourceMessageRef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", normalize_prompt(self.prompt))
        if len(self.fingerprint) != 64:
            raise fail(TimerErrorCode.INVALID_RULE)
        if len(self.source_message_refs) > MAX_SOURCE_REFS:
            raise fail(TimerErrorCode.LIMIT_EXCEEDED)


@dataclass(frozen=True, slots=True)
class ManageTimerCommand:
    scope: TimerScope
    source_run_ref: SourceRunRef
    opaque_ref: OpaqueTimerRef
    target: TimerRefTarget
    action: ManageTimerAction
    expected_version: int
    idempotency_key: IdempotencyKey

    def __post_init__(self) -> None:
        if self.expected_version < 1:
            raise fail(TimerErrorCode.VERSION_CONFLICT)


@dataclass(frozen=True, slots=True)
class RollOccurrenceCommand:
    scope: TimerScope
    rule_id: TimerRuleId
    last_materialized_due_at: datetime
    through: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "last_materialized_due_at", require_aware(self.last_materialized_due_at)
        )
        object.__setattr__(self, "through", require_aware(self.through))
        if self.through < self.last_materialized_due_at:
            raise fail(TimerErrorCode.INVALID_RULE)


@dataclass(frozen=True, slots=True)
class PreparedTimerCreation:
    rule: TimerRule
    first_occurrence: TimerOccurrence

    def __post_init__(self) -> None:
        if (
            self.rule.scope != self.first_occurrence.scope
            or self.rule.rule_id != self.first_occurrence.rule_id
        ):
            raise fail(TimerErrorCode.SCOPE_MISMATCH)


@dataclass(frozen=True, slots=True)
class CreateTimerResult:
    outcome: CreateTimerOutcome
    opaque_ref: OpaqueTimerRef

    def __post_init__(self) -> None:
        if not isinstance(self.opaque_ref, OpaqueTimerRef):
            raise fail(TimerErrorCode.INVALID_STATE)


@dataclass(frozen=True, slots=True)
class ManageTimerResult:
    outcome: ManageTimerOutcome
    opaque_ref: OpaqueTimerRef
    status: str
    version: int

    def __post_init__(self) -> None:
        if self.version < 1 or len(self.status) > 32:
            raise fail(TimerErrorCode.INVALID_STATE)


@dataclass(frozen=True, slots=True)
class ReviseTimerCommand:
    scope: TimerScope
    source_run_ref: SourceRunRef
    opaque_ref: OpaqueTimerRef
    expected_version: int
    idempotency_key: IdempotencyKey
    changed_at: datetime
    schedule: NormalizedTimerRule | None = None
    prompt: str | None = field(default=None, repr=False)
    time_expression: str = ""
    timezone: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed_at", require_aware(self.changed_at))
        if self.expected_version < 1 or (self.schedule is None) == (self.prompt is None):
            raise fail(TimerErrorCode.INVALID_RULE)
        if self.prompt is not None:
            object.__setattr__(self, "prompt", normalize_prompt(self.prompt))
        if len(self.time_expression) > 200 or len(self.timezone) > 128:
            raise fail(TimerErrorCode.INVALID_RULE)


@dataclass(frozen=True, slots=True)
class ReviseTimerResult:
    outcome: ManageTimerOutcome
    opaque_ref: OpaqueTimerRef
    version: int

    def __post_init__(self) -> None:
        if self.version < 1:
            raise fail(TimerErrorCode.INVALID_STATE)


__all__ = [
    "CreateTimerCommand",
    "CreateTimerOutcome",
    "CreateTimerResult",
    "ManageTimerAction",
    "ManageTimerCommand",
    "ManageTimerOutcome",
    "ManageTimerResult",
    "PreparedTimerCreation",
    "ReviseTimerCommand",
    "ReviseTimerResult",
    "RollOccurrenceCommand",
]
