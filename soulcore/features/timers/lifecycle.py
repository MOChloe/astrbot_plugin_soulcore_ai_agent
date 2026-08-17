"""Domain contracts for conservative Timer lifecycle review."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from .domain import TimerOccurrenceId, TimerRuleId, TimerScope, require_aware

TIMER_LIFECYCLE_REVIEW_CAPABILITY = "conversation.timer_lifecycle_review"


class TimerLifecycleDecision(StrEnum):
    KEEP_ONGOING = "KEEP_ONGOING"
    KEEP_UNCERTAIN = "KEEP_UNCERTAIN"
    COMPLETE_FULFILLED = "COMPLETE_FULFILLED"
    COMPLETE_ENDED = "COMPLETE_ENDED"

    @property
    def completes_rule(self) -> bool:
        return self in {
            TimerLifecycleDecision.COMPLETE_FULFILLED,
            TimerLifecycleDecision.COMPLETE_ENDED,
        }


class TimerLifecycleReviewStatus(StrEnum):
    PENDING = "PENDING"
    KEPT = "KEPT"
    COMPLETED = "COMPLETED"
    STALE = "STALE"
    ERROR_KEEP = "ERROR_KEEP"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True)
class TimerLifecycleEvidence:
    timer_description: str
    source_messages: tuple[str, ...] = ()
    working_text: str = field(default="", repr=False)
    decision_kind: str = ""
    output_status: str = ""


@dataclass(frozen=True, slots=True)
class TimerLifecycleReview:
    review_id: str
    scope: TimerScope
    rule_id: TimerRuleId
    occurrence_id: TimerOccurrenceId
    occurrence_generation: int
    main_core_run_id: int
    expected_rule_version: int
    expected_activity_epoch: int
    evidence: TimerLifecycleEvidence = field(repr=False)
    status: TimerLifecycleReviewStatus = TimerLifecycleReviewStatus.PENDING
    decision: TimerLifecycleDecision | None = None
    error_code: str = ""
    task_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.occurrence_generation < 0 or self.main_core_run_id < 1:
            raise ValueError("invalid Timer lifecycle review fence")
        if self.expected_rule_version < 1 or self.expected_activity_epoch < 0:
            raise ValueError("invalid Timer lifecycle review version")
        for name in ("created_at", "updated_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_aware(value))


@dataclass(frozen=True, slots=True)
class TimerLifecycleModelResult:
    decision: TimerLifecycleDecision | None = None
    backend_id: str = ""
    error_code: str = ""


__all__ = [
    "TIMER_LIFECYCLE_REVIEW_CAPABILITY",
    "TimerLifecycleDecision",
    "TimerLifecycleEvidence",
    "TimerLifecycleModelResult",
    "TimerLifecycleReview",
    "TimerLifecycleReviewStatus",
]
