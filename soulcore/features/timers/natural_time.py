"""Deterministic Chinese-first natural time and arrangement-change interpretation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .constants import (
    MAX_ABSOLUTE_HORIZON_SECONDS,
    MAX_RELATIVE_DELAY_SECONDS,
    MIN_ABSOLUTE_LEAD_SECONDS,
    MIN_RELATIVE_DELAY_SECONDS,
)
from .domain import require_aware
from .natural_time_parsing import (
    WEEKDAYS,
    Clock,
    parse_clocks,
    parse_date,
    parse_number,
)

MAX_NATURAL_TIME_CANDIDATES = 4
_MAX_EXPRESSION_CHARS = 200
_IANA_ZONE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_+-]+/[A-Za-z0-9_+./-]+)")
_RFC3339 = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})")
_RELATIVE = re.compile(
    r"(?P<number>\d+|半|[零〇一二两三四五六七八九十百千]+)\s*"
    r"(?P<unit>秒钟?|分钟?|刻钟|个?小时|个?钟头|天|日|周|星期)\s*(?:以后|之后|后)"
)
_ENGLISH_RELATIVE = re.compile(
    r"\bin\s+(?P<number>\d+)\s+(?P<unit>seconds?|minutes?|hours?|days?|weeks?)\b",
    re.IGNORECASE,
)
_ZONE_ALIASES = {
    "北京时间": "Asia/Shanghai",
    "中国时间": "Asia/Shanghai",
    "香港时间": "Asia/Hong_Kong",
    "东京时间": "Asia/Tokyo",
    "日本时间": "Asia/Tokyo",
    "纽约时间": "America/New_York",
    "伦敦时间": "Europe/London",
    "协调世界时": "UTC",
}
_VAGUE_TERMS = (
    "过几天",
    "这几天",
    "哪天",
    "某天",
    "晚些时候",
    "稍后",
    "有空时",
    "有空的时候",
    "下班后",
    "晚上",
    "白天",
    "周末",
)


class NaturalTimeStatus(StrEnum):
    UNIQUE = "UNIQUE"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class NaturalTimeCandidate:
    time_expression: str
    summary: str
    timezone: str
    rule: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class NaturalTimeResolution:
    status: NaturalTimeStatus
    candidates: tuple[NaturalTimeCandidate, ...] = ()
    message: str = ""

    @property
    def unique(self) -> NaturalTimeCandidate | None:
        if self.status is NaturalTimeStatus.UNIQUE and len(self.candidates) == 1:
            return self.candidates[0]
        return None


class ArrangementChangeKind(StrEnum):
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    CANCEL = "CANCEL"
    RESCHEDULE = "RESCHEDULE"
    REWRITE = "REWRITE"


@dataclass(frozen=True, slots=True)
class ArrangementChangeResolution:
    status: NaturalTimeStatus
    kind: ArrangementChangeKind | None = None
    time_resolution: NaturalTimeResolution | None = None
    action_text: str = ""
    message: str = ""


def interpret_natural_time(
    expression: str,
    *,
    now: datetime,
    timezone: str,
) -> NaturalTimeResolution:
    """Resolve only deterministic supported expressions; never choose among meanings."""

    raw = _normalize_expression(expression)
    now = require_aware(now)
    if not raw:
        return _invalid("时间表达不能为空")
    try:
        zone, body = _expression_zone(raw, timezone)
    except (ValueError, ZoneInfoNotFoundError):
        return _invalid("时区无效；请使用会话中的 IANA 时区")
    local_now = now.astimezone(zone)

    rfc3339 = _RFC3339.fullmatch(body)
    if rfc3339 is not None:
        try:
            due = datetime.fromisoformat(body.replace("Z", "+00:00"))
        except ValueError:
            return _invalid("日期或时间无效")
        return _absolute_resolution(raw, due, now=now, zone=zone)

    relative = _relative_seconds(body)
    if relative is not None:
        if not MIN_RELATIVE_DELAY_SECONDS <= relative <= MAX_RELATIVE_DELAY_SECONDS:
            return _invalid("相对时间必须至少一分钟，且不能超过一年")
        due = now + timedelta(seconds=relative)
        return NaturalTimeResolution(
            NaturalTimeStatus.UNIQUE,
            (
                NaturalTimeCandidate(
                    raw,
                    _absolute_summary(due, zone),
                    zone.key,
                    {"kind": "RELATIVE", "delay_seconds": relative},
                ),
            ),
        )

    recurring = _recurring_resolution(raw, body, local_now, zone)
    if recurring is not None:
        return recurring

    absolute = _calendar_resolution(raw, body, local_now, now, zone)
    if absolute is not None:
        return absolute

    if any(term in body for term in _VAGUE_TERMS):
        return _ambiguous(
            "这个时间还不能确定到唯一执行时刻；请补充具体日期和钟点",
            _clarification_candidates(raw, zone.key),
        )
    return _invalid("无法把这个表达确定为受支持的未来时间")


def interpret_arrangement_change(
    change: str,
    *,
    now: datetime,
    timezone: str,
) -> ArrangementChangeResolution:
    """Classify one arrangement change without resolving its target."""

    text = _normalize_expression(change)
    if not text:
        return ArrangementChangeResolution(NaturalTimeStatus.INVALID, message="调整内容不能为空")
    compact = re.sub(r"[\s，,。.!！]+", "", text)
    if compact in {"暂停", "先暂停", "停一下", "暂时停下", "先停一下"}:
        return ArrangementChangeResolution(NaturalTimeStatus.UNIQUE, ArrangementChangeKind.PAUSE)
    if compact in {"继续", "恢复", "恢复安排", "继续安排", "重新开始"}:
        return ArrangementChangeResolution(NaturalTimeStatus.UNIQUE, ArrangementChangeKind.RESUME)
    if compact in {"取消", "取消安排", "不要了", "删掉", "作废"}:
        return ArrangementChangeResolution(NaturalTimeStatus.UNIQUE, ArrangementChangeKind.CANCEL)

    content_match = re.fullmatch(
        r"(?:把)?(?:到时候做的事|到时候做什么|到时做什么|行动内容|内容)"
        r"(?:改成|改为|换成)\s*(?P<content>.+)",
        text,
    )
    if content_match is None:
        content_match = re.fullmatch(r"到时(?:改成|改为|换成)\s*(?P<content>.+)", text)
    if content_match is not None:
        content = content_match.group("content").strip()
        if not content:
            return ArrangementChangeResolution(
                NaturalTimeStatus.INVALID, message="新的行动内容不能为空"
            )
        return ArrangementChangeResolution(
            NaturalTimeStatus.UNIQUE,
            ArrangementChangeKind.REWRITE,
            action_text=content,
        )

    time_match = re.fullmatch(
        r"(?:把)?(?:整个安排|整个系列|以后都|每次|下一次)?"
        r"(?:时间)?(?:改到|改成|改为|换到|挪到)\s*(?P<time>.+)",
        text,
    )
    if time_match is not None:
        time_text = time_match.group("time").strip()
        resolution = interpret_natural_time(time_text, now=now, timezone=timezone)
        return ArrangementChangeResolution(
            resolution.status,
            ArrangementChangeKind.RESCHEDULE,
            time_resolution=resolution,
            message=resolution.message,
        )

    if compact.startswith(("延后", "推迟", "往后挪")):
        return ArrangementChangeResolution(
            NaturalTimeStatus.AMBIGUOUS,
            ArrangementChangeKind.RESCHEDULE,
            message="“延后”需要说明相对原安排延后多久，或直接写新的具体时间",
        )
    return ArrangementChangeResolution(
        NaturalTimeStatus.INVALID,
        message="只支持暂停、继续、取消、改时间或改到时候做的事",
    )


def natural_time_candidate_payload(candidate: NaturalTimeCandidate) -> dict[str, object]:
    return {
        "time_expression": candidate.time_expression,
        "summary": candidate.summary,
        "timezone": candidate.timezone,
    }


def _normalize_expression(value: str) -> str:
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value).strip()
    if len(normalized) > _MAX_EXPRESSION_CHARS:
        return ""
    return re.sub(r"\s+", " ", normalized)


def _expression_zone(raw: str, default_timezone: str) -> tuple[ZoneInfo, str]:
    default = str(default_timezone or "").strip()
    if not default:
        raise ValueError("missing timezone")
    zone_name = default
    body = raw
    iana = _IANA_ZONE.search(body)
    if iana is not None:
        zone_name = iana.group(1)
        body = f"{body[: iana.start()]} {body[iana.end() :]}".strip()
    else:
        for alias, mapped in _ZONE_ALIASES.items():
            if alias in body:
                zone_name = mapped
                body = body.replace(alias, " ").strip()
                break
        if re.search(r"(?:UTC|GMT)\s*[+-]\s*\d", body, re.IGNORECASE):
            raise ValueError("numeric offset is not an IANA timezone")
        if re.search(r"[A-Za-z]+/[A-Za-z0-9_+./-]+", body):
            raise ValueError("invalid IANA timezone")
    return ZoneInfo(zone_name), re.sub(r"\s+", " ", body).strip()


def _relative_seconds(text: str) -> int | None:
    match = _RELATIVE.fullmatch(text)
    if match is not None:
        number_text = match.group("number")
        unit = match.group("unit")
        if number_text == "半":
            if unit not in {"小时", "个钟头"}:
                return None
            return 30 * 60
        number = parse_number(number_text)
        if number is None:
            return None
        multiplier = {
            "秒": 1,
            "秒钟": 1,
            "分": 60,
            "分钟": 60,
            "刻钟": 15 * 60,
            "小时": 60 * 60,
            "个小时": 60 * 60,
            "个钟头": 60 * 60,
            "钟头": 60 * 60,
            "天": 24 * 60 * 60,
            "日": 24 * 60 * 60,
            "周": 7 * 24 * 60 * 60,
            "星期": 7 * 24 * 60 * 60,
        }[unit]
        return number * multiplier
    english = _ENGLISH_RELATIVE.fullmatch(text)
    if english is None:
        return None
    number = int(english.group("number"))
    unit = english.group("unit").lower()
    if unit.startswith("second"):
        return number
    if unit.startswith("minute"):
        return number * 60
    if unit.startswith("hour"):
        return number * 60 * 60
    if unit.startswith("day"):
        return number * 24 * 60 * 60
    return number * 7 * 24 * 60 * 60


def _recurring_resolution(
    raw: str,
    text: str,
    local_now: datetime,
    zone: ZoneInfo,
) -> NaturalTimeResolution | None:
    weekly = re.fullmatch(
        r"每(?:个)?(?:周|星期)(?P<weekday>[一二三四五六日天1-7])\s*(?P<clock>.*)",
        text,
    )
    if weekly is not None:
        clock_text = weekly.group("clock").strip()
        clocks = parse_clocks(clock_text)
        if not clock_text or clocks is None:
            return _ambiguous(
                "重复安排需要唯一的执行钟点",
                tuple(
                    NaturalTimeCandidate(
                        f"每周{weekly.group('weekday')} {hour:02d}:00",
                        f"每周{weekly.group('weekday')} {hour:02d}:00",
                        zone.key,
                    )
                    for hour in (9, 18, 21)
                ),
            )
        if len(clocks) != 1:
            return _clock_ambiguity(raw, clocks, local_now.date(), zone, recurring="weekly")
        clock = clocks[0]
        return NaturalTimeResolution(
            NaturalTimeStatus.UNIQUE,
            (
                NaturalTimeCandidate(
                    raw,
                    f"每周{weekly.group('weekday')} {clock.hour:02d}:{clock.minute:02d}",
                    zone.key,
                    {
                        "kind": "WEEKLY",
                        "iso_weekday": WEEKDAYS[weekly.group("weekday")],
                        "wall_time": f"{clock.hour:02d}:{clock.minute:02d}",
                        "timezone": zone.key,
                    },
                ),
            ),
        )

    yearly = re.fullmatch(
        r"每年\s*(?P<month>\d{1,2}|[零〇一二两三四五六七八九十]+)月"
        r"\s*(?P<day>\d{1,2}|[零〇一二两三四五六七八九十]+)(?:日|号)"
        r"\s*(?P<clock>.*)",
        text,
    )
    if yearly is None:
        return None
    month = parse_number(yearly.group("month"))
    day = parse_number(yearly.group("day"))
    if month is None or day is None:
        return _invalid("重复日期无效")
    try:
        date(2000, month, day)
    except ValueError:
        return _invalid("重复日期无效")
    clock_text = yearly.group("clock").strip()
    clocks = parse_clocks(clock_text)
    if not clock_text or clocks is None:
        return _ambiguous(
            "重复安排需要唯一的执行钟点",
            tuple(
                NaturalTimeCandidate(
                    f"每年{month}月{day}日 {hour:02d}:00",
                    f"每年{month}月{day}日 {hour:02d}:00",
                    zone.key,
                )
                for hour in (9, 18, 21)
            ),
        )
    if len(clocks) != 1:
        return _clock_ambiguity(raw, clocks, date(2000, month, day), zone, recurring="yearly")
    clock = clocks[0]
    return NaturalTimeResolution(
        NaturalTimeStatus.UNIQUE,
        (
            NaturalTimeCandidate(
                raw,
                f"每年{month}月{day}日 {clock.hour:02d}:{clock.minute:02d}",
                zone.key,
                {
                    "kind": "YEARLY",
                    "month": month,
                    "day": day,
                    "wall_time": f"{clock.hour:02d}:{clock.minute:02d}",
                    "timezone": zone.key,
                },
            ),
        ),
    )


def _calendar_resolution(
    raw: str,
    text: str,
    local_now: datetime,
    now: datetime,
    zone: ZoneInfo,
) -> NaturalTimeResolution | None:
    parsed_date, remainder, date_was_explicit, invalid_date = parse_date(text, local_now)
    if invalid_date:
        return _invalid("日期无效")
    if parsed_date is None:
        return None
    clocks = parse_clocks(remainder)
    if clocks is None or not remainder.strip():
        return _ambiguous(
            "日期已经确定，但还缺少唯一的执行钟点",
            tuple(
                NaturalTimeCandidate(
                    f"{parsed_date.isoformat()} {hour:02d}:00",
                    f"{parsed_date.year}年{parsed_date.month}月{parsed_date.day}日 {hour:02d}:00",
                    zone.key,
                )
                for hour in (9, 18, 21)
            ),
        )
    if len(clocks) > 1:
        return _clock_ambiguity(raw, clocks, parsed_date, zone)
    clock = clocks[0]
    candidates = _local_datetime_candidates(parsed_date, time(clock.hour, clock.minute), zone)
    if not candidates:
        return _invalid("这个本地时间不存在，可能落在夏令时跳时区间")
    if len(candidates) > 1:
        return _ambiguous(
            "这个本地时间在夏令时切换时出现两次；请写明明确偏移或改用其他时刻",
            tuple(
                NaturalTimeCandidate(
                    candidate.isoformat(),
                    _absolute_summary(candidate, zone),
                    zone.key,
                    {"kind": "ABSOLUTE", "at": candidate.isoformat()},
                )
                for candidate in candidates[:MAX_NATURAL_TIME_CANDIDATES]
            ),
        )
    due = candidates[0]
    if not date_was_explicit and due <= local_now:
        due = _local_datetime_candidates(
            parsed_date + timedelta(days=1), time(clock.hour, clock.minute), zone
        )[0]
    return _absolute_resolution(raw, due, now=now, zone=zone)


def _absolute_resolution(
    raw: str,
    due: datetime,
    *,
    now: datetime,
    zone: ZoneInfo,
) -> NaturalTimeResolution:
    try:
        due = require_aware(due)
    except Exception:
        return _invalid("执行时间必须包含明确时区")
    lead = (due - now).total_seconds()
    if lead < MIN_ABSOLUTE_LEAD_SECONDS:
        return _invalid("执行时间已经过去，或距离现在不足一分钟")
    if lead > MAX_ABSOLUTE_HORIZON_SECONDS:
        return _invalid("执行时间不能超过十年")
    local_due = due.astimezone(zone)
    return NaturalTimeResolution(
        NaturalTimeStatus.UNIQUE,
        (
            NaturalTimeCandidate(
                raw,
                _absolute_summary(local_due, zone),
                zone.key,
                {"kind": "ABSOLUTE", "at": local_due.isoformat()},
            ),
        ),
    )


def _clock_ambiguity(
    raw: str,
    clocks: tuple[Clock, ...],
    local_date: date,
    zone: ZoneInfo,
    *,
    recurring: str = "",
) -> NaturalTimeResolution:
    candidates: list[NaturalTimeCandidate] = []
    for clock in clocks[:MAX_NATURAL_TIME_CANDIDATES]:
        if recurring == "weekly":
            summary = f"每周该日 {clock.hour:02d}:{clock.minute:02d}"
        elif recurring == "yearly":
            summary = f"每年该日 {clock.hour:02d}:{clock.minute:02d}"
        else:
            summary = (
                f"{local_date.year}年{local_date.month}月{local_date.day}日 "
                f"{clock.hour:02d}:{clock.minute:02d}"
            )
        candidates.append(
            NaturalTimeCandidate(
                f"{raw}（{clock.hour:02d}:{clock.minute:02d}）",
                summary,
                zone.key,
            )
        )
    return _ambiguous("钟点存在上午/下午歧义；请写明时段或使用 24 小时制", tuple(candidates))


def _local_datetime_candidates(
    local_date: date,
    wall_time: time,
    zone: ZoneInfo,
) -> tuple[datetime, ...]:
    naive = datetime.combine(local_date, wall_time)
    found: dict[datetime, datetime] = {}
    for fold in (0, 1):
        local = naive.replace(tzinfo=zone, fold=fold)
        instant = local.astimezone(UTC)
        round_trip = instant.astimezone(zone)
        if round_trip.replace(tzinfo=None) == naive and round_trip.fold == fold:
            found[instant] = local
    return tuple(found[key] for key in sorted(found))


def _absolute_summary(value: datetime, zone: ZoneInfo) -> str:
    local = value.astimezone(zone)
    return f"{local.year}年{local.month}月{local.day}日 {local.hour:02d}:{local.minute:02d}"


def _clarification_candidates(raw: str, timezone: str) -> tuple[NaturalTimeCandidate, ...]:
    return (
        NaturalTimeCandidate(raw, "补充具体日期，例如“8月3日晚上九点”", timezone),
        NaturalTimeCandidate(raw, "补充准确间隔，例如“三天后 20:00”", timezone),
        NaturalTimeCandidate(raw, "重复事项写明周期，例如“每周五 21:00”", timezone),
    )


def _ambiguous(
    message: str,
    candidates: tuple[NaturalTimeCandidate, ...],
) -> NaturalTimeResolution:
    return NaturalTimeResolution(
        NaturalTimeStatus.AMBIGUOUS,
        candidates[:MAX_NATURAL_TIME_CANDIDATES],
        message,
    )


def _invalid(message: str) -> NaturalTimeResolution:
    return NaturalTimeResolution(NaturalTimeStatus.INVALID, message=message)


__all__ = [
    "ArrangementChangeKind",
    "ArrangementChangeResolution",
    "MAX_NATURAL_TIME_CANDIDATES",
    "NaturalTimeCandidate",
    "NaturalTimeResolution",
    "NaturalTimeStatus",
    "interpret_arrangement_change",
    "interpret_natural_time",
    "natural_time_candidate_payload",
]
