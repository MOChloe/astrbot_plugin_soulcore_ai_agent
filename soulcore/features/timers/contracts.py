"Commands and results shared by later Timer adapters and persistence."

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from .domain import (
    MAX_CANDIDATE_PREVIEW_CHARS,
    MAX_SEMANTIC_CANDIDATES,
    MAX_SOURCE_REFS,
    IdempotencyKey,
    NormalizedTimerRule,
    OpaqueTimerRef,
    SourceMessageRef,
    SourceRunRef,
    TimerErrorCode,
    TimerOccurrence,
    TimerRule,
    TimerRuleId,
    TimerScope,
    fail,
    normalize_prompt,
    require_aware,
)


class TimerRefTarget(StrEnum):
    SERIES = "SERIES"
    OCCURRENCE = "OCCURRENCE"


@dataclass(frozen=True, slots=True)
class TimerProjectionSource:
    opaque_ref: OpaqueTimerRef
    target: TimerRefTarget
    rule: TimerRule
    occurrence: TimerOccurrence | None = None
    next_due_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.occurrence is not None and (
            self.occurrence.scope != self.rule.scope or self.occurrence.rule_id != self.rule.rule_id
        ):
            raise fail(TimerErrorCode.SCOPE_MISMATCH)
        if self.target is TimerRefTarget.OCCURRENCE and self.occurrence is None:
            raise fail(TimerErrorCode.INVALID_REFERENCE)
        if self.next_due_at is not None:
            object.__setattr__(self, "next_due_at", require_aware(self.next_due_at))


@dataclass(frozen=True, slots=True)
class TimerCandidateProjection:
    opaque_ref: OpaqueTimerRef
    target: TimerRefTarget
    rule_kind: str
    status: str
    original_or_next_due_at: datetime | None
    prompt_preview: str


def prompt_preview(prompt: str) -> str:
    """Return an informative prefix while always withholding at least one character."""

    if len(prompt) <= 1:
        return "…"
    visible = min(len(prompt) - 1, MAX_CANDIDATE_PREVIEW_CHARS - 1)
    return f"{prompt[:visible]}…"


def project_candidates(
    sources: tuple[TimerProjectionSource, ...],
) -> tuple[TimerCandidateProjection, ...]:
    if len(sources) > MAX_SEMANTIC_CANDIDATES:
        raise fail(TimerErrorCode.LIMIT_EXCEEDED)
    seen: set[str] = set()
    result: list[TimerCandidateProjection] = []
    for source in sources:
        if source.opaque_ref.value in seen:
            raise fail(TimerErrorCode.INVALID_REFERENCE)
        seen.add(source.opaque_ref.value)
        occurrence = source.occurrence
        result.append(
            TimerCandidateProjection(
                opaque_ref=source.opaque_ref,
                target=source.target,
                rule_kind=source.rule.schedule.kind.value,
                status=(occurrence.status.value if occurrence else source.rule.status.value),
                original_or_next_due_at=(
                    occurrence.original_due_at if occurrence else source.next_due_at
                ),
                prompt_preview=prompt_preview(source.rule.prompt),
            )
        )
    return tuple(result)


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
    "TimerCandidateProjection",
    "TimerProjectionSource",
    "TimerRefTarget",
    "project_candidates",
    "prompt_preview",
]
