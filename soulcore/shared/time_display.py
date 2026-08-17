"""Human-readable datetime projection for model-facing text."""

from __future__ import annotations

from datetime import UTC, datetime, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def resolve_timezone(value: Any) -> tzinfo:
    """Resolve a configured IANA timezone without making host timezone implicit."""

    name = str(value or "").strip()
    if not name:
        return UTC
    try:
        return ZoneInfo(name)
    except (ValueError, ZoneInfoNotFoundError):
        return UTC


def model_datetime(
    value: Any,
    *,
    localize: bool = True,
    timezone_name: str = "",
) -> str:
    """Render one datetime without leaking ISO protocol syntax into prompts.

    Formatting is deliberately separate from storage and scheduling. Model-facing
    text is local by default because persisted datetimes are UTC. Callers that
    explicitly need the encoded wall-clock fields can opt out of localization.
    """

    parsed = _datetime_value(value)
    if parsed is None:
        return str(value or "").strip()
    if timezone_name and parsed.utcoffset() is not None:
        parsed = parsed.astimezone(resolve_timezone(timezone_name))
    elif localize and parsed.utcoffset() is not None:
        parsed = parsed.astimezone()
    return (
        f"{parsed.year}/{parsed.month}/{parsed.day} "
        f"{parsed.hour:02d}:{parsed.minute:02d}:{parsed.second:02d}"
    )


def model_iso_datetime(value: Any, *, timezone_name: str = "UTC") -> str:
    """Render an exact model-writeable timestamp in one explicit timezone."""

    parsed = _datetime_value(value)
    if parsed is None:
        return str(value or "").strip()
    zone = resolve_timezone(timezone_name)
    parsed = parsed.replace(tzinfo=zone) if parsed.tzinfo is None else parsed.astimezone(zone)
    return parsed.isoformat(timespec="seconds")


def _datetime_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


__all__ = ["model_datetime", "model_iso_datetime", "resolve_timezone"]
