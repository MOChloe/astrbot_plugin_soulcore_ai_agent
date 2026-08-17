"""Calendar and clock parsing primitives for deterministic natural time."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

WEEKDAYS = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "日": 7,
    "天": 7,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
}
_PERIODS = (
    "凌晨",
    "清晨",
    "早上",
    "上午",
    "中午",
    "下午",
    "傍晚",
    "晚上",
    "今晚",
    "夜里",
    "晚",
)


@dataclass(frozen=True, slots=True)
class Clock:
    hour: int
    minute: int
    explicit_24_hour: bool


def parse_date(
    text: str,
    local_now: datetime,
) -> tuple[date | None, str, bool, bool]:
    body = text.strip()
    for parser in (
        _named_relative_date,
        _offset_date,
        _explicit_calendar_date,
        _weekday_date,
    ):
        parsed = parser(body, local_now)
        if parsed is not None:
            return parsed
    clocks = parse_clocks(body)
    if clocks is not None:
        return local_now.date(), body, False, False
    return None, body, False, False


def _named_relative_date(
    body: str,
    local_now: datetime,
) -> tuple[date | None, str, bool, bool] | None:
    for marker, offset, period in (
        ("大后天晚上", 3, "晚上"),
        ("后天晚上", 2, "晚上"),
        ("明天晚上", 1, "晚上"),
        ("明晚", 1, "晚上"),
        ("今晚", 0, "晚上"),
    ):
        if body.startswith(marker):
            return (
                local_now.date() + timedelta(days=offset),
                f"{period}{body[len(marker) :].strip()}",
                True,
                False,
            )
    for marker, offset in (
        ("大后天", 3),
        ("后天", 2),
        ("明天", 1),
        ("明日", 1),
        ("今天", 0),
        ("今日", 0),
    ):
        if body.startswith(marker):
            return (
                local_now.date() + timedelta(days=offset),
                body[len(marker) :].strip(),
                True,
                False,
            )
    english_markers = (("day after tomorrow", 2), ("tomorrow", 1), ("today", 0))
    lowered = body.lower()
    for marker, offset in english_markers:
        if lowered.startswith(marker):
            remainder = body[len(marker) :].strip()
            remainder = re.sub(r"^at\s+", "", remainder, flags=re.IGNORECASE)
            return local_now.date() + timedelta(days=offset), remainder, True, False
    return None


def _offset_date(
    body: str,
    local_now: datetime,
) -> tuple[date | None, str, bool, bool] | None:
    day_offset = re.match(
        r"(?P<number>\d+|[零〇一二两三四五六七八九十百千]+)\s*(?:天|日)(?:以后|之后|后)",
        body,
    )
    if day_offset is None:
        return None
    number = parse_number(day_offset.group("number"))
    if number is None:
        return None, body, False, True
    return (
        local_now.date() + timedelta(days=number),
        body[day_offset.end() :].strip(),
        True,
        False,
    )


def _explicit_calendar_date(
    body: str,
    local_now: datetime,
) -> tuple[date | None, str, bool, bool] | None:
    full = re.match(
        r"(?P<year>\d{4})\s*(?:年|[-/.])\s*(?P<month>\d{1,2})"
        r"\s*(?:月|[-/.])\s*(?P<day>\d{1,2})\s*(?:日|号)?",
        body,
    )
    if full is not None:
        parsed = _safe_date(
            int(full.group("year")), int(full.group("month")), int(full.group("day"))
        )
        return parsed, body[full.end() :].strip(), True, parsed is None

    month_day = re.match(
        r"(?P<month>\d{1,2}|[零〇一二两三四五六七八九十]+)月"
        r"\s*(?P<day>\d{1,2}|[零〇一二两三四五六七八九十]+)(?:日|号)",
        body,
    )
    if month_day is None:
        return None
    month = parse_number(month_day.group("month"))
    day = parse_number(month_day.group("day"))
    if month is None or day is None:
        return None, body, False, True
    parsed = _safe_date(local_now.year, month, day)
    if parsed is None:
        return None, body, False, True
    remainder = body[month_day.end() :].strip()
    clocks = parse_clocks(remainder)
    if parsed < local_now.date() or (
        parsed == local_now.date()
        and clocks
        and len(clocks) == 1
        and time(clocks[0].hour, clocks[0].minute) <= local_now.time()
    ):
        parsed = _safe_date(local_now.year + 1, month, day)
        if parsed is None:
            return None, body, False, True
    return parsed, remainder, True, False


def _weekday_date(
    body: str,
    local_now: datetime,
) -> tuple[date | None, str, bool, bool] | None:
    weekday = re.match(
        r"(?P<next>下周|下星期|下礼拜|本周|这周)?"
        r"(?:周|星期|礼拜)(?P<weekday>[一二三四五六日天1-7])",
        body,
    )
    if weekday is None:
        return None
    wanted = WEEKDAYS[weekday.group("weekday")]
    prefix = weekday.group("next") or ""
    if prefix in {"下周", "下星期", "下礼拜"}:
        days = 7 - local_now.isoweekday() + wanted
    elif prefix in {"本周", "这周"}:
        days = wanted - local_now.isoweekday()
        if days < 0:
            return None, body, False, True
    else:
        days = (wanted - local_now.isoweekday()) % 7
    return (
        local_now.date() + timedelta(days=days),
        body[weekday.end() :].strip(),
        True,
        False,
    )


def parse_clocks(text: str) -> tuple[Clock, ...] | None:
    value = text.strip()
    if not value:
        return None
    value = re.sub(r"^(?:在|at)\s+", "", value, flags=re.IGNORECASE)
    numeric_matched, numeric = _numeric_clock(value)
    if numeric_matched:
        return numeric
    return _spoken_clock(value)


def _numeric_clock(value: str) -> tuple[bool, tuple[Clock, ...] | None]:
    english = re.fullmatch(
        r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>am|pm)",
        value,
        re.IGNORECASE,
    )
    if english is not None:
        hour = int(english.group("hour"))
        minute = int(english.group("minute") or 0)
        if not 1 <= hour <= 12 or not 0 <= minute <= 59:
            return True, None
        if english.group("ampm").lower() == "pm" and hour != 12:
            hour += 12
        if english.group("ampm").lower() == "am" and hour == 12:
            hour = 0
        return True, (Clock(hour, minute, True),)

    colon = re.fullmatch(r"(?P<hour>\d{1,2}):(?P<minute>\d{2})", value)
    if colon is not None:
        hour = int(colon.group("hour"))
        minute = int(colon.group("minute"))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return True, (Clock(hour, minute, True),)
        return True, None
    return False, None


def _spoken_clock(value: str) -> tuple[Clock, ...] | None:
    period, body = _split_period(value)
    match = re.fullmatch(
        r"(?P<hour>\d{1,2}|[零〇一二两三四五六七八九十]+)"
        r"(?:点|时)"
        r"(?:(?P<minute>\d{1,2}|[零〇一二两三四五六七八九十]+)分?"
        r"|(?P<quarter>半|一刻|三刻))?",
        body,
    )
    if match is None:
        return None
    hour = parse_number(match.group("hour"))
    minute = _spoken_minute(match.group("minute"), match.group("quarter"))
    if hour is None or minute is None or not 0 <= minute <= 59 or not 0 <= hour <= 24:
        return None
    normalized_hour = 0 if hour == 24 else hour
    return _clock_options(period, normalized_hour, minute)


def _split_period(value: str) -> tuple[str, str]:
    for period in _PERIODS:
        if value.startswith(period):
            return period, value[len(period) :].strip()
    return "", value


def _spoken_minute(minute_text: str | None, quarter: str | None) -> int | None:
    if quarter == "半":
        return 30
    if quarter == "一刻":
        return 15
    if quarter == "三刻":
        return 45
    return parse_number(minute_text) if minute_text else 0


def _clock_options(period: str, hour: int, minute: int) -> tuple[Clock, ...] | None:
    if period:
        converted = _period_hour(period, hour)
        return (Clock(converted, minute, True),) if converted is not None else None
    if hour >= 13 or hour == 0:
        return (Clock(hour, minute, True),)
    if hour == 12:
        return (
            Clock(0, minute, False),
            Clock(12, minute, False),
        )
    return (
        Clock(hour, minute, False),
        Clock(hour + 12, minute, False),
    )


def _period_hour(period: str, hour: int) -> int | None:
    if hour > 12:
        return None
    if period in {"凌晨", "清晨", "早上", "上午"}:
        return 0 if hour == 12 else hour
    if period == "中午":
        return hour if hour >= 11 else hour + 12
    if period in {"下午", "傍晚", "晚上", "今晚", "夜里", "晚"}:
        return 0 if hour == 12 else hour + 12
    return hour


def parse_number(value: str | None) -> int | None:
    if value is None or not value:
        return None
    if value.isdigit():
        return int(value)
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    units = {"十": 10, "百": 100, "千": 1000}
    total = 0
    section = 0
    current = 0
    for char in value:
        if char in digits:
            current = digits[char]
            continue
        unit = units.get(char)
        if unit is None:
            return None
        section += (current or 1) * unit
        current = 0
    total += section + current
    return total


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


__all__ = ["Clock", "WEEKDAYS", "parse_clocks", "parse_date", "parse_number"]
