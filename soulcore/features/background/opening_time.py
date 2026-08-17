"""Deterministic story-time seam for the two opening life frames."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

from ...shared.time_display import resolve_timezone

OPENING_HANDOFF_LOCAL_HOUR = 16


def opening_handoff_at(anchor_at: datetime, *, timezone_name: str) -> datetime:
    """Return 16:00 on the prior local calendar day, encoded as UTC.

    A calendar boundary is intentional here.  A fixed relative offset can land
    the opening Keyframe in the middle of the night and leave Ordinary too
    little elapsed life to account naturally for sleep.
    """

    anchor = anchor_at if anchor_at.tzinfo is not None else anchor_at.replace(tzinfo=UTC)
    zone = resolve_timezone(timezone_name)
    local_anchor = anchor.astimezone(zone)
    handoff_date = local_anchor.date() - timedelta(days=1)
    local_handoff = datetime.combine(
        handoff_date,
        time(hour=OPENING_HANDOFF_LOCAL_HOUR),
        tzinfo=zone,
    )
    return local_handoff.astimezone(UTC)


__all__ = ["OPENING_HANDOFF_LOCAL_HOUR", "opening_handoff_at"]
