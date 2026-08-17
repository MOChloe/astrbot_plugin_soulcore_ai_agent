"""Presentation helpers for the run-scoped Timer Main Core adapter."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tzlocal import get_localzone_name

from .domain import OpaqueTimerRef, TimerRule
from .errors import TimerErrorCode, fail
from .projection import TimerCandidateProjection, TimerProjectionSource, TimerRefTarget
from .rules import canonical_rule, next_occurrence


def candidate_payload(candidate: TimerCandidateProjection) -> dict[str, Any]:
    return {
        "timer_ref": candidate.opaque_ref.value,
        "target": candidate.target.value,
        "rule_kind": candidate.rule_kind,
        "status": candidate.status,
        "due_at": (
            candidate.original_or_next_due_at.isoformat()
            if candidate.original_or_next_due_at is not None
            else None
        ),
        "prompt_preview": candidate.prompt_preview,
    }


def pending_creation_payload(intent: Any, *, idempotent: bool) -> dict[str, Any]:
    payload = {
        "status": "pending_final_commit",
        "kind": "create",
        "prompt": intent.prompt,
        "idempotent": idempotent,
    }
    if intent.time_expression:
        payload.update(
            time_expression=intent.time_expression,
            timezone=intent.timezone,
        )
    return payload


def pending_management_payload(intent: Any, *, idempotent: bool) -> dict[str, Any]:
    return {
        "status": "pending_final_commit",
        "kind": "manage",
        "timer_ref": intent.opaque_ref.value,
        "target": intent.target.value,
        "action": intent.action.value,
        "idempotent": idempotent,
    }


def pending_revision_payload(intent: Any, *, idempotent: bool) -> dict[str, Any]:
    return {
        "status": "pending_final_commit",
        "kind": "adjust",
        "arrangement_ref": intent.opaque_ref.value,
        "idempotent": idempotent,
    }


def arrangement_payload(source: Any, timezone: str) -> dict[str, Any]:
    due = source.next_due_at
    when = schedule_summary(source.rule.schedule, due, timezone, source.rule.timezone)
    status = {
        "ACTIVE": "进行中",
        "PAUSED": "已暂停",
        "CANCELLED": "已取消",
        "COMPLETED": "已完成",
    }.get(source.rule.status.value, source.rule.status.value)
    action = short_quote(source.rule.prompt)
    return {
        "arrangement_ref": source.opaque_ref.value,
        "summary": f"{when}；到时候：{action}",
        "status": status,
        "when": when,
        "action_summary": action,
    }


def validated_timezone(value: str) -> str:
    timezone = str(value or "").strip()
    if not timezone:
        try:
            timezone = str(get_localzone_name() or "").strip()
        except (OSError, RuntimeError, ValueError, ZoneInfoNotFoundError):
            raise fail(TimerErrorCode.INVALID_TIMEZONE) from None
    try:
        return ZoneInfo(timezone).key
    except (ValueError, ZoneInfoNotFoundError):
        raise fail(TimerErrorCode.INVALID_TIMEZONE) from None


def source_from_resolved(
    opaque_ref: OpaqueTimerRef,
    rule: TimerRule,
    checked_at: datetime,
) -> TimerProjectionSource:
    return TimerProjectionSource(
        opaque_ref,
        TimerRefTarget.SERIES,
        rule,
        next_due_at=next_occurrence(rule.schedule, after=checked_at),
    )


def short_quote(value: str, limit: int = 80) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        text = f"{text[: limit - 1]}…"
    return f"“{text}”"


def explicit_entire_series(value: str) -> bool:
    text = str(value or "")
    return any(marker in text for marker in ("整个安排", "整个系列", "以后都", "每次"))


def arrangement_range_kind(query: str) -> str:
    compact = str(query or "").strip().replace(" ", "")
    if compact in {"最近", "近期", "接下来", "近期安排", "最近安排"}:
        return "upcoming"
    if compact in {"下周", "下个星期", "下星期", "下周安排"}:
        return "next_week"
    return ""


def arrangement_in_range(
    due_at: datetime | None,
    *,
    range_kind: str,
    checked_at: datetime,
    timezone: str,
) -> bool:
    if due_at is None:
        return False
    zone = _projection_zone(timezone, "")
    local_due = due_at.astimezone(zone)
    local_checked = checked_at.astimezone(zone)
    if range_kind == "upcoming":
        return local_checked <= local_due <= local_checked + timedelta(days=30)
    if range_kind == "next_week":
        next_monday = local_checked.date() + timedelta(days=7 - local_checked.date().weekday())
        return next_monday <= local_due.date() < next_monday + timedelta(days=7)
    return True


def _schedule_summary(schedule: Any, due: datetime | None, zone: ZoneInfo) -> str:
    canonical = canonical_rule(schedule)
    kind = str(canonical["kind"])
    if kind in {"ABSOLUTE", "RELATIVE"}:
        value = due or schedule.due_at
        local = value.astimezone(zone)
        return f"{local.year}年{local.month}月{local.day}日 {local.hour:02d}:{local.minute:02d}"
    if kind == "WEEKLY":
        names = ("一", "二", "三", "四", "五", "六", "日")
        return f"每周{names[int(str(canonical['iso_weekday'])) - 1]} {canonical['wall_time']}"
    return f"每年{canonical['month']}月{canonical['day']}日 {canonical['wall_time']}"


def schedule_summary(
    schedule: Any,
    due: datetime | None,
    timezone: str,
    fallback_timezone: str = "",
) -> str:
    """Render one resolved schedule in the same stable form used by timer views."""

    return _schedule_summary(schedule, due, _projection_zone(timezone, fallback_timezone))


def _projection_zone(primary: str, secondary: str) -> ZoneInfo:
    for value in (primary, secondary, "UTC"):
        try:
            if value:
                return ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError):
            continue
    return ZoneInfo("UTC")


__all__ = [
    "arrangement_in_range",
    "arrangement_payload",
    "arrangement_range_kind",
    "candidate_payload",
    "explicit_entire_series",
    "pending_creation_payload",
    "pending_management_payload",
    "pending_revision_payload",
    "schedule_summary",
    "short_quote",
    "source_from_resolved",
    "validated_timezone",
]
