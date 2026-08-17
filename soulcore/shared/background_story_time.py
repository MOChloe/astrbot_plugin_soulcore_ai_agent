"""Stable story-time lag shared by background scheduling boundaries."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

BACKGROUND_STORY_LAG_MINUTES = (5, 20)


def background_story_lag_minutes(stable_ref: str) -> int:
    """Return one restart-stable pseudo-random lag in the inclusive range."""

    reference = str(stable_ref or "").strip()
    if not reference:
        raise ValueError("background story-time reference cannot be empty")
    minimum, maximum = BACKGROUND_STORY_LAG_MINUTES
    digest = hashlib.sha256(reference.encode("utf-8")).digest()
    return minimum + int.from_bytes(digest[:8], "big") % (maximum - minimum + 1)


def background_story_cutoff_at(anchor_at: datetime, *, stable_ref: str) -> datetime:
    """Move one real-time anchor behind by its stable creative handoff lag."""

    if anchor_at.tzinfo is None or anchor_at.utcoffset() is None:
        raise ValueError("background story-time anchor must be timezone-aware")
    anchor = anchor_at.astimezone(UTC)
    return anchor - timedelta(minutes=background_story_lag_minutes(stable_ref))


__all__ = [
    "BACKGROUND_STORY_LAG_MINUTES",
    "background_story_cutoff_at",
    "background_story_lag_minutes",
]
