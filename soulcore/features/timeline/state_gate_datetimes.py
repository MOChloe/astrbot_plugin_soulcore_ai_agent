from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("state gate datetimes must be timezone-aware")
    return value.astimezone(UTC)


def parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return as_utc(value)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return as_utc(datetime.fromisoformat(text))
    except ValueError as exc:
        raise ValueError("invalid state gate datetime") from exc


def required_datetime(value: Any) -> datetime:
    parsed = parse_datetime(value)
    if parsed is None:
        raise ValueError("required state gate datetime is missing")
    return parsed


__all__ = ["as_utc", "parse_datetime", "required_datetime"]
