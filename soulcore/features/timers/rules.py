"""Deterministic Timer rule parsing, normalization, recurrence and recovery."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from .constants import (
    MAX_ABSOLUTE_HORIZON_SECONDS,
    MAX_RELATIVE_DELAY_SECONDS,
    MIN_ABSOLUTE_LEAD_SECONDS,
    MIN_RELATIVE_DELAY_SECONDS,
)
from .domain import (
    AbsoluteTimerRule,
    NormalizedTimerRule,
    RelativeTimerRule,
    TimerRuleKind,
    TimerScope,
    WeeklyTimerRule,
    YearlyTimerRule,
    normalize_prompt,
    require_aware,
)
from .errors import TimerErrorCode, fail

_RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")
_WALL_TIME = re.compile(r"^(?P<hour>[01]\d|2[0-3]):(?P<minute>[0-5]\d)$")
_MAX_TIMEZONE_CHARS = 128
_RULE_FIELDS: dict[TimerRuleKind, frozenset[str]] = {
    TimerRuleKind.ABSOLUTE: frozenset({"kind", "at"}),
    TimerRuleKind.RELATIVE: frozenset({"kind", "delay_seconds"}),
    TimerRuleKind.WEEKLY: frozenset({"kind", "iso_weekday", "wall_time", "timezone"}),
    TimerRuleKind.YEARLY: frozenset({"kind", "month", "day", "wall_time", "timezone"}),
}


@dataclass(frozen=True, slots=True)
class OccurrenceRollPlan:
    """Bounded recovery decision without enumerating every missed occurrence."""

    latest_missed_due_at: datetime | None
    coalesced_count: int
    next_future_due_at: datetime | None


def _strict_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise fail(TimerErrorCode.INVALID_RULE)
    return value


def _parse_wall_time(value: object) -> time:
    if not isinstance(value, str):
        raise fail(TimerErrorCode.INVALID_RULE)
    match = _WALL_TIME.fullmatch(value)
    if match is None:
        raise fail(TimerErrorCode.INVALID_RULE)
    return time(int(match.group("hour")), int(match.group("minute")))


def _timezone(value: object) -> ZoneInfo:
    if not isinstance(value, str) or not value or len(value) > _MAX_TIMEZONE_CHARS:
        raise fail(TimerErrorCode.INVALID_TIMEZONE)
    if value not in available_timezones():
        raise fail(TimerErrorCode.INVALID_TIMEZONE)
    try:
        return ZoneInfo(value)
    except (ValueError, ZoneInfoNotFoundError):
        raise fail(TimerErrorCode.INVALID_TIMEZONE) from None


def _parse_rfc3339(value: object) -> datetime:
    if not isinstance(value, str) or _RFC3339.fullmatch(value) is None:
        raise fail(TimerErrorCode.INVALID_RULE)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise fail(TimerErrorCode.INVALID_RULE) from None
    return require_aware(parsed)


def _rule_kind(payload: Mapping[str, object]) -> TimerRuleKind:
    if len(payload) > 5 or any(not isinstance(key, str) for key in payload):
        raise fail(TimerErrorCode.INVALID_RULE)
    raw_kind = payload.get("kind")
    if not isinstance(raw_kind, str):
        raise fail(TimerErrorCode.INVALID_RULE)
    try:
        kind = TimerRuleKind(raw_kind)
    except ValueError:
        raise fail(TimerErrorCode.UNSUPPORTED_RULE) from None
    if frozenset(payload) != _RULE_FIELDS[kind]:
        raise fail(TimerErrorCode.INVALID_RULE)
    return kind


def _parse_absolute_rule(
    payload: Mapping[str, object], committed_at: datetime
) -> AbsoluteTimerRule:
    due_at = _parse_rfc3339(payload["at"])
    lead = (due_at - committed_at).total_seconds()
    if lead < MIN_ABSOLUTE_LEAD_SECONDS or lead > MAX_ABSOLUTE_HORIZON_SECONDS:
        raise fail(TimerErrorCode.OUT_OF_RANGE)
    return AbsoluteTimerRule(due_at=due_at)


def _parse_relative_rule(
    payload: Mapping[str, object], committed_at: datetime
) -> RelativeTimerRule:
    delay = _strict_int(payload["delay_seconds"])
    if not MIN_RELATIVE_DELAY_SECONDS <= delay <= MAX_RELATIVE_DELAY_SECONDS:
        raise fail(TimerErrorCode.OUT_OF_RANGE)
    return RelativeTimerRule(
        delay_seconds=delay,
        anchored_at=committed_at,
        due_at=committed_at + timedelta(seconds=delay),
    )


def _parse_weekly_rule(payload: Mapping[str, object]) -> WeeklyTimerRule:
    wall_time = _parse_wall_time(payload["wall_time"])
    zone = _timezone(payload["timezone"])
    weekday = _strict_int(payload["iso_weekday"])
    if not 1 <= weekday <= 7:
        raise fail(TimerErrorCode.OUT_OF_RANGE)
    return WeeklyTimerRule(weekday, wall_time, zone.key)


def _parse_yearly_rule(payload: Mapping[str, object]) -> YearlyTimerRule:
    wall_time = _parse_wall_time(payload["wall_time"])
    zone = _timezone(payload["timezone"])
    month = _strict_int(payload["month"])
    day = _strict_int(payload["day"])
    try:
        date(2000, month, day)
    except ValueError:
        raise fail(TimerErrorCode.OUT_OF_RANGE) from None
    return YearlyTimerRule(month, day, wall_time, zone.key)


def parse_and_normalize_rule(
    payload: Mapping[str, object], *, committed_at: datetime
) -> NormalizedTimerRule:
    """Parse the four-field union and anchor relative rules at final commit time."""

    committed_at = require_aware(committed_at)
    kind = _rule_kind(payload)
    if kind is TimerRuleKind.ABSOLUTE:
        return _parse_absolute_rule(payload, committed_at)
    if kind is TimerRuleKind.RELATIVE:
        return _parse_relative_rule(payload, committed_at)
    if kind is TimerRuleKind.WEEKLY:
        return _parse_weekly_rule(payload)
    return _parse_yearly_rule(payload)


def _wall_candidates(naive: datetime, zone: ZoneInfo) -> tuple[datetime, ...]:
    candidates: set[datetime] = set()
    for fold in (0, 1):
        local = naive.replace(tzinfo=zone, fold=fold)
        instant = local.astimezone(UTC)
        round_trip = instant.astimezone(zone)
        if round_trip.replace(tzinfo=None) == naive and round_trip.fold == fold:
            candidates.add(instant)
    return tuple(sorted(candidates))


def resolve_wall_time(local_date: date, wall_time: time, timezone: str) -> datetime:
    """Resolve a wall minute; gaps move forward and overlaps choose the later instant."""

    zone = _timezone(timezone)
    naive = datetime.combine(local_date, wall_time)
    for minute_offset in range(24 * 60):
        candidate_naive = naive + timedelta(minutes=minute_offset)
        if candidate_naive.date() != local_date:
            break
        candidates = _wall_candidates(candidate_naive, zone)
        if candidates:
            return candidates[-1]
    raise fail(TimerErrorCode.INVALID_RULE)


def next_occurrence(schedule: NormalizedTimerRule, *, after: datetime) -> datetime | None:
    """Return the first occurrence strictly after an aware instant."""

    after = require_aware(after)
    if isinstance(schedule, (AbsoluteTimerRule, RelativeTimerRule)):
        return schedule.due_at if schedule.due_at > after else None
    zone = _timezone(schedule.timezone)
    local_after = after.astimezone(zone)
    if isinstance(schedule, WeeklyTimerRule):
        for days_ahead in range(8):
            local_date = local_after.date() + timedelta(days=days_ahead)
            if local_date.isoweekday() != schedule.iso_weekday:
                continue
            candidate = resolve_wall_time(local_date, schedule.wall_time, schedule.timezone)
            if candidate > after:
                return candidate
        raise fail(TimerErrorCode.INVALID_RULE)

    for year in range(local_after.year, 10_000):
        try:
            local_date = date(year, schedule.month, schedule.day)
        except ValueError:
            continue
        candidate = resolve_wall_time(local_date, schedule.wall_time, schedule.timezone)
        if candidate > after:
            return candidate
    return None


def _previous_occurrence(
    schedule: WeeklyTimerRule | YearlyTimerRule, *, at_or_before: datetime
) -> datetime | None:
    at_or_before = require_aware(at_or_before)
    zone = _timezone(schedule.timezone)
    local = at_or_before.astimezone(zone)
    if isinstance(schedule, WeeklyTimerRule):
        for days_back in range(8):
            local_date = local.date() - timedelta(days=days_back)
            if local_date.isoweekday() != schedule.iso_weekday:
                continue
            candidate = resolve_wall_time(local_date, schedule.wall_time, schedule.timezone)
            if candidate <= at_or_before:
                return candidate
        return None

    for year in range(local.year, max(0, local.year - 8), -1):
        try:
            local_date = date(year, schedule.month, schedule.day)
        except ValueError:
            continue
        candidate = resolve_wall_time(local_date, schedule.wall_time, schedule.timezone)
        if candidate <= at_or_before:
            return candidate
    return None


def plan_occurrence_roll(
    schedule: NormalizedTimerRule,
    *,
    last_materialized_due_at: datetime,
    recovered_at: datetime,
) -> OccurrenceRollPlan:
    """Coalesce missed periodic occurrences while retaining the most recent one."""

    last_due = require_aware(last_materialized_due_at)
    recovered = require_aware(recovered_at)
    if recovered < last_due:
        raise fail(TimerErrorCode.INVALID_RULE)
    next_future = next_occurrence(schedule, after=recovered)
    if isinstance(schedule, (AbsoluteTimerRule, RelativeTimerRule)):
        missed = schedule.due_at if last_due < schedule.due_at <= recovered else None
        return OccurrenceRollPlan(missed, 0, next_future)

    latest = _previous_occurrence(schedule, at_or_before=recovered)
    first = next_occurrence(schedule, after=last_due)
    if latest is None or first is None or latest < first:
        return OccurrenceRollPlan(None, 0, next_future)

    zone = _timezone(schedule.timezone)
    if isinstance(schedule, WeeklyTimerRule):
        span_days = (latest.astimezone(zone).date() - first.astimezone(zone).date()).days
        missed_count = span_days // 7 + 1
    else:
        first_year = first.astimezone(zone).year
        latest_year = latest.astimezone(zone).year
        missed_count = 0
        for year in range(first_year, latest_year + 1):
            try:
                date(year, schedule.month, schedule.day)
            except ValueError:
                continue
            missed_count += 1
    return OccurrenceRollPlan(latest, max(0, missed_count - 1), next_future)


def canonical_rule(schedule: NormalizedTimerRule) -> dict[str, object]:
    if isinstance(schedule, AbsoluteTimerRule):
        return {"kind": schedule.kind.value, "due_at": schedule.due_at.isoformat()}
    if isinstance(schedule, RelativeTimerRule):
        return {
            "kind": schedule.kind.value,
            "delay_seconds": schedule.delay_seconds,
            "anchored_at": schedule.anchored_at.isoformat(),
            "due_at": schedule.due_at.isoformat(),
        }
    if isinstance(schedule, WeeklyTimerRule):
        return {
            "kind": schedule.kind.value,
            "iso_weekday": schedule.iso_weekday,
            "wall_time": schedule.wall_time.strftime("%H:%M"),
            "timezone": schedule.timezone,
        }
    yearly = schedule
    return {
        "kind": yearly.kind.value,
        "month": yearly.month,
        "day": yearly.day,
        "wall_time": yearly.wall_time.strftime("%H:%M"),
        "timezone": yearly.timezone,
    }


def exact_timer_fingerprint(scope: TimerScope, schedule: NormalizedTimerRule, prompt: str) -> str:
    payload = {
        "scope": list(scope.fingerprint_parts),
        "rule": canonical_rule(schedule),
        "prompt": normalize_prompt(prompt),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = [
    "OccurrenceRollPlan",
    "canonical_rule",
    "exact_timer_fingerprint",
    "next_occurrence",
    "parse_and_normalize_rule",
    "plan_occurrence_roll",
    "resolve_wall_time",
]
