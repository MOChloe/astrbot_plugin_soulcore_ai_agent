"""Bounded model-facing projections which never expose Timer internal IDs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from .constants import (
    MAX_CANDIDATE_PREVIEW_CHARS,
    MAX_SEMANTIC_CANDIDATES,
)
from .domain import OpaqueTimerRef, TimerOccurrence, TimerRule, require_aware
from .errors import TimerErrorCode, fail


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


__all__ = [
    "TimerCandidateProjection",
    "TimerProjectionSource",
    "TimerRefTarget",
    "project_candidates",
    "prompt_preview",
]
