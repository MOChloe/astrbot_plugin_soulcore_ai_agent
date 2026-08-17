"""Immutable values and aggregates for the standalone Timer domain."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from typing import TypeAlias

from .constants import (
    MAX_IDEMPOTENCY_KEY_CHARS,
    MAX_INSTANCE_ID_CHARS,
    MAX_INTERNAL_REF_CHARS,
    MAX_PROFILE_ID_CHARS,
    MAX_PROMPT_CHARS,
    MAX_RELATIVE_DELAY_SECONDS,
    MAX_SOURCE_REFS,
    MIN_RELATIVE_DELAY_SECONDS,
)
from .errors import TimerErrorCode, fail

_SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_OPAQUE_REF = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


def require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise fail(TimerErrorCode.INVALID_RULE)
    return value.astimezone(UTC)


def _bounded_text(value: str, maximum: int, code: TimerErrorCode) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise fail(code)
    if not _SAFE_REF.fullmatch(value):
        raise fail(code)
    return value


@dataclass(frozen=True, slots=True)
class TimerScope:
    """Trusted profile and role-instance ownership supplied by the current Run."""

    profile_id: str
    instance_id: str

    def __post_init__(self) -> None:
        _bounded_text(self.profile_id, MAX_PROFILE_ID_CHARS, TimerErrorCode.INVALID_SCOPE)
        _bounded_text(self.instance_id, MAX_INSTANCE_ID_CHARS, TimerErrorCode.INVALID_SCOPE)

    @property
    def fingerprint_parts(self) -> tuple[str, str]:
        return self.profile_id, self.instance_id


@dataclass(frozen=True, slots=True)
class _InternalRef:
    value: str

    def __post_init__(self) -> None:
        _bounded_text(self.value, MAX_INTERNAL_REF_CHARS, TimerErrorCode.INVALID_REFERENCE)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class TimerRuleId(_InternalRef):
    """Persistence identity; never accepted from or projected to a model."""


@dataclass(frozen=True, slots=True)
class TimerOccurrenceId(_InternalRef):
    """Persistence identity; never accepted from or projected to a model."""


@dataclass(frozen=True, slots=True)
class OccurrenceStableRef(_InternalRef):
    """Stable internal tie-break reference, separate from short-lived model refs."""


@dataclass(frozen=True, slots=True)
class ExecutionEnvelopeRef(_InternalRef):
    """Narrow association to a later Main Core execution envelope."""


@dataclass(frozen=True, slots=True)
class DeliveryAssociationRef(_InternalRef):
    """Narrow association to a later expression/delivery lifecycle."""


@dataclass(frozen=True, slots=True)
class SourceRunRef(_InternalRef):
    """Opaque provenance for the Run that requested creation."""


@dataclass(frozen=True, slots=True)
class SourceMessageRef(_InternalRef):
    """Opaque provenance for a controlled source message."""


@dataclass(frozen=True, slots=True)
class OpaqueTimerRef:
    """Short-lived allowlisted ref exposed to a model instead of internal IDs."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not _OPAQUE_REF.fullmatch(self.value):
            raise fail(TimerErrorCode.INVALID_REFERENCE)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class IdempotencyKey:
    value: str

    def __post_init__(self) -> None:
        _bounded_text(
            self.value,
            MAX_IDEMPOTENCY_KEY_CHARS,
            TimerErrorCode.INVALID_REFERENCE,
        )


def normalize_prompt(value: str) -> str:
    if not isinstance(value, str):
        raise fail(TimerErrorCode.INVALID_PROMPT)
    normalized = unicodedata.normalize("NFC", value.strip())
    if not normalized or len(normalized) > MAX_PROMPT_CHARS:
        raise fail(TimerErrorCode.INVALID_PROMPT)
    if any(unicodedata.category(char) in {"Cc", "Cs"} for char in normalized):
        raise fail(TimerErrorCode.INVALID_PROMPT)
    return normalized


class TimerRuleKind(StrEnum):
    ABSOLUTE = "ABSOLUTE"
    RELATIVE = "RELATIVE"
    WEEKLY = "WEEKLY"
    YEARLY = "YEARLY"


@dataclass(frozen=True, slots=True)
class AbsoluteTimerRule:
    due_at: datetime
    kind: TimerRuleKind = TimerRuleKind.ABSOLUTE

    def __post_init__(self) -> None:
        object.__setattr__(self, "due_at", require_aware(self.due_at))


@dataclass(frozen=True, slots=True)
class RelativeTimerRule:
    delay_seconds: int
    anchored_at: datetime
    due_at: datetime
    kind: TimerRuleKind = TimerRuleKind.RELATIVE

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchored_at", require_aware(self.anchored_at))
        object.__setattr__(self, "due_at", require_aware(self.due_at))
        if not MIN_RELATIVE_DELAY_SECONDS <= self.delay_seconds <= MAX_RELATIVE_DELAY_SECONDS:
            raise fail(TimerErrorCode.OUT_OF_RANGE)
        if self.due_at != self.anchored_at + timedelta(seconds=self.delay_seconds):
            raise fail(TimerErrorCode.INVALID_RULE)


@dataclass(frozen=True, slots=True)
class WeeklyTimerRule:
    iso_weekday: int
    wall_time: time
    timezone: str
    kind: TimerRuleKind = TimerRuleKind.WEEKLY

    def __post_init__(self) -> None:
        if not 1 <= self.iso_weekday <= 7 or not self.timezone:
            raise fail(TimerErrorCode.INVALID_RULE)
        if self.wall_time.tzinfo is not None or self.wall_time.second or self.wall_time.microsecond:
            raise fail(TimerErrorCode.INVALID_RULE)


@dataclass(frozen=True, slots=True)
class YearlyTimerRule:
    month: int
    day: int
    wall_time: time
    timezone: str
    kind: TimerRuleKind = TimerRuleKind.YEARLY

    def __post_init__(self) -> None:
        try:
            date(2000, self.month, self.day)
        except ValueError:
            raise fail(TimerErrorCode.OUT_OF_RANGE) from None
        if not self.timezone:
            raise fail(TimerErrorCode.INVALID_RULE)
        if self.wall_time.tzinfo is not None or self.wall_time.second or self.wall_time.microsecond:
            raise fail(TimerErrorCode.INVALID_RULE)


NormalizedTimerRule: TypeAlias = (
    AbsoluteTimerRule | RelativeTimerRule | WeeklyTimerRule | YearlyTimerRule
)


class TimerRuleStatus(StrEnum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"


class TimerOccurrenceStatus(StrEnum):
    SCHEDULED = "SCHEDULED"
    WAITING = "WAITING"
    CLAIMED = "CLAIMED"
    RUNNING = "RUNNING"
    WAITING_DELIVERY = "WAITING_DELIVERY"
    PAUSED = "PAUSED"
    RECOVERING = "RECOVERING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    MISSED_COALESCED = "MISSED_COALESCED"

    @property
    def terminal(self) -> bool:
        return self in {
            TimerOccurrenceStatus.COMPLETED,
            TimerOccurrenceStatus.CANCELLED,
            TimerOccurrenceStatus.FAILED,
            TimerOccurrenceStatus.MISSED_COALESCED,
        }


@dataclass(frozen=True, slots=True)
class TimerRuleRevision:
    """One durable user-intent version stored inside the existing schedule JSON."""

    version: int
    changed_at: datetime
    schedule: NormalizedTimerRule
    prompt: str = field(repr=False)
    time_expression: str = ""
    timezone: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed_at", require_aware(self.changed_at))
        object.__setattr__(self, "prompt", normalize_prompt(self.prompt))
        expression = unicodedata.normalize("NFC", str(self.time_expression or "").strip())
        timezone = str(self.timezone or "").strip()
        if self.version < 1 or len(expression) > 200 or len(timezone) > 128:
            raise fail(TimerErrorCode.INVALID_RULE)
        object.__setattr__(self, "time_expression", expression)
        object.__setattr__(self, "timezone", timezone)


@dataclass(frozen=True, slots=True)
class TimerRule:
    rule_id: TimerRuleId
    scope: TimerScope
    schedule: NormalizedTimerRule
    prompt: str = field(repr=False)
    fingerprint: str
    status: TimerRuleStatus
    version: int
    created_sequence: int
    created_at: datetime
    source_run_ref: SourceRunRef
    source_message_refs: tuple[SourceMessageRef, ...] = ()
    last_operation_key: str = ""
    last_operation_fingerprint: str = ""
    time_expression: str = ""
    timezone: str = ""
    revisions: tuple[TimerRuleRevision, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", normalize_prompt(self.prompt))
        object.__setattr__(self, "created_at", require_aware(self.created_at))
        time_expression, timezone = _validated_rule_metadata(self)
        object.__setattr__(self, "time_expression", time_expression)
        object.__setattr__(self, "timezone", timezone)
        _validate_source_message_refs(self.source_message_refs)
        _validate_rule_revisions(self)


def _validated_rule_metadata(rule: TimerRule) -> tuple[str, str]:
    time_expression = unicodedata.normalize("NFC", str(rule.time_expression or "").strip())
    timezone = str(rule.timezone or "").strip()
    if (
        rule.version < 1
        or rule.created_sequence < 1
        or re.fullmatch(r"[0-9a-f]{64}", rule.fingerprint) is None
        or len(time_expression) > 200
        or len(timezone) > 128
    ):
        raise fail(TimerErrorCode.INVALID_RULE)
    return time_expression, timezone


def _validate_source_message_refs(refs: tuple[SourceMessageRef, ...]) -> None:
    if len(refs) > MAX_SOURCE_REFS:
        raise fail(TimerErrorCode.LIMIT_EXCEEDED)
    if len(set(refs)) != len(refs):
        raise fail(TimerErrorCode.INVALID_REFERENCE)


def _validate_rule_revisions(rule: TimerRule) -> None:
    if not rule.revisions:
        return
    versions = tuple(item.version for item in rule.revisions)
    if versions != tuple(sorted(set(versions))) or versions[-1] > rule.version:
        raise fail(TimerErrorCode.INVALID_RULE)
    latest = rule.revisions[-1]
    if (
        latest.schedule != rule.schedule
        or latest.prompt != rule.prompt
        or latest.time_expression != rule.time_expression
        or latest.timezone != rule.timezone
    ):
        raise fail(TimerErrorCode.INVALID_RULE)


@dataclass(frozen=True, slots=True)
class TimerOccurrence:
    occurrence_id: TimerOccurrenceId
    stable_ref: OccurrenceStableRef
    rule_id: TimerRuleId
    scope: TimerScope
    original_due_at: datetime
    status: TimerOccurrenceStatus
    version: int
    generation: int
    created_sequence: int
    created_at: datetime
    execution_ref: ExecutionEnvelopeRef | None = None
    delivery_ref: DeliveryAssociationRef | None = None
    recovery_from: TimerOccurrenceStatus | None = None
    last_operation_key: str = ""
    last_operation_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_due_at", require_aware(self.original_due_at))
        object.__setattr__(self, "created_at", require_aware(self.created_at))
        if self.version < 1 or self.generation < 0 or self.created_sequence < 1:
            raise fail(TimerErrorCode.INVALID_RULE)
        if self.status is TimerOccurrenceStatus.RECOVERING and self.recovery_from is None:
            raise fail(TimerErrorCode.INVALID_STATE)
        if self.status is not TimerOccurrenceStatus.RECOVERING and self.recovery_from is not None:
            raise fail(TimerErrorCode.INVALID_STATE)
        if self.status is TimerOccurrenceStatus.RUNNING and self.execution_ref is None:
            raise fail(TimerErrorCode.INVALID_STATE)
        if self.status is TimerOccurrenceStatus.WAITING_DELIVERY and self.delivery_ref is None:
            raise fail(TimerErrorCode.INVALID_STATE)


__all__ = [
    "AbsoluteTimerRule",
    "DeliveryAssociationRef",
    "ExecutionEnvelopeRef",
    "IdempotencyKey",
    "NormalizedTimerRule",
    "OccurrenceStableRef",
    "OpaqueTimerRef",
    "RelativeTimerRule",
    "SourceMessageRef",
    "SourceRunRef",
    "TimerOccurrence",
    "TimerOccurrenceId",
    "TimerOccurrenceStatus",
    "TimerRule",
    "TimerRuleId",
    "TimerRuleKind",
    "TimerRuleRevision",
    "TimerRuleStatus",
    "TimerScope",
    "WeeklyTimerRule",
    "YearlyTimerRule",
    "normalize_prompt",
    "require_aware",
]
