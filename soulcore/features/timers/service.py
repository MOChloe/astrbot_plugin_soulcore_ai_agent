"""Public Timer service surface for cross-feature Main Core consumers."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .constants import MAX_SEMANTIC_CANDIDATES
from .domain import SourceMessageRef
from .errors import TimerDomainError
from .lifecycle import (
    TIMER_LIFECYCLE_REVIEW_CAPABILITY,
    TimerLifecycleDecision,
    TimerLifecycleEvidence,
    TimerLifecycleModelResult,
)
from .main_core import (
    PendingTimerCreation,
    PendingTimerManagement,
    PendingTimerRevision,
    TimerMainCoreReader,
    TimerMainCoreService,
    TimerRunToolContext,
    build_timer_wake_request,
)
from .main_core_views import validated_timezone
from .natural_time import NaturalTimeResolution, NaturalTimeStatus, interpret_natural_time
from .task_cancel import has_permanent_domain_cancel, settle_permanent_task_cancel

MIN_TEMPORARY_ABSENCE = timedelta(minutes=1)
MAX_TEMPORARY_ABSENCE = timedelta(hours=24)


@dataclass(frozen=True, slots=True)
class TemporaryAbsenceCommandContext:
    """Resolve one model-visible decision without mutating persistent state."""

    checked_at: datetime
    timezone: str
    max_duration: timedelta = MAX_TEMPORARY_ABSENCE

    def resolve(self, *, reason: str, time_expression: str) -> dict[str, Any]:
        normalized_reason = str(reason or "").strip()
        if not normalized_reason:
            raise ValueError("暂离原因不能为空")
        if len(normalized_reason) > 1000:
            raise ValueError("暂离原因不能超过 1000 个字符")
        expression = str(time_expression or "").strip()
        timezone = _temporary_absence_timezone(self.timezone)
        resolution = _interpret_temporary_absence_time(
            expression,
            now=self.checked_at,
            timezone=timezone,
        )
        if resolution.status is not NaturalTimeStatus.UNIQUE or resolution.unique is None:
            message = str(resolution.message or "").strip()
            raise ValueError(message or "大概多久必须能确定到唯一的未来时刻")
        rule = dict(resolution.unique.rule or {})
        kind = str(rule.get("kind") or "").upper()
        if kind not in {"RELATIVE", "ABSOLUTE"}:
            raise ValueError("暂离只接受一次性的相对时间或明确时刻")
        due_at = _resolved_due_at(rule, now=self.checked_at)
        duration = due_at - self.checked_at
        maximum = min(self.max_duration, MAX_TEMPORARY_ABSENCE)
        if duration < MIN_TEMPORARY_ABSENCE or duration > maximum:
            maximum_hours = max(1, int(maximum.total_seconds() // 3600))
            raise ValueError(f"暂离时间必须至少一分钟，且不能超过 {maximum_hours} 小时")
        return {
            "reason": normalized_reason,
            "time_expression": expression,
            "timezone": resolution.unique.timezone,
            "summary": resolution.unique.summary,
            "rule": rule,
        }


def _temporary_absence_timezone(value: str) -> str:
    timezone = str(value or "").strip()
    if timezone:
        return timezone
    try:
        return validated_timezone(timezone)
    except ValueError:
        raise ValueError("时区无效；请使用会话中的 IANA 时区") from None


def _interpret_temporary_absence_time(
    expression: str,
    *,
    now: datetime,
    timezone: str,
) -> NaturalTimeResolution:
    resolution = interpret_natural_time(expression, now=now, timezone=timezone)
    if resolution.status is not NaturalTimeStatus.INVALID:
        return resolution
    duration_resolution = interpret_natural_time(f"{expression}后", now=now, timezone=timezone)
    candidate = duration_resolution.unique
    if candidate is None:
        return resolution
    rule = dict(candidate.rule or {})
    if str(rule.get("kind") or "").upper() != "RELATIVE":
        return resolution
    return duration_resolution


def _resolved_due_at(rule: Mapping[str, object], *, now: datetime) -> datetime:
    kind = str(rule.get("kind") or "").upper()
    if kind == "RELATIVE":
        return now + timedelta(seconds=int(rule.get("delay_seconds") or 0))
    if kind == "ABSOLUTE":
        value = datetime.fromisoformat(str(rule.get("at") or ""))
        if value.tzinfo is None:
            raise ValueError("暂离结束时刻必须包含明确时区")
        return value
    raise ValueError("暂离时间规则无效")


__all__ = [
    "PendingTimerCreation",
    "PendingTimerManagement",
    "PendingTimerRevision",
    "MAX_SEMANTIC_CANDIDATES",
    "SourceMessageRef",
    "TIMER_LIFECYCLE_REVIEW_CAPABILITY",
    "TimerDomainError",
    "TimerLifecycleDecision",
    "TimerLifecycleEvidence",
    "TimerLifecycleModelResult",
    "TimerMainCoreReader",
    "TimerMainCoreService",
    "TimerRunToolContext",
    "TemporaryAbsenceCommandContext",
    "build_timer_wake_request",
    "has_permanent_domain_cancel",
    "settle_permanent_task_cancel",
]
