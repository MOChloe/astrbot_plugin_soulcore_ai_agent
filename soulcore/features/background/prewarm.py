"""Deterministic orchestration rules for proactive role-frame prewarming.

This module deliberately contains no model-visible text.  It only chooses the
owned story-time boundary and the durable execution deadline used by an
ordinary background author task.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

PROACTIVE_FRAME_FRESH_MINUTES = 60
PROACTIVE_FRAME_COOLDOWN_MINUTES = 60
PROACTIVE_FRAME_TIMEOUT_MINUTES = 20
PROACTIVE_FRAME_GAP_MINUTES = (12, 20, 35)


class ProactiveFrameSourceKind(StrEnum):
    TIMER = "TIMER"
    WAKEUP = "WAKEUP"
    CONTACT = "CONTACT"
    AI_TASK = "AI_TASK"
    JUST_IN_TIME = "JUST_IN_TIME"


@dataclass(frozen=True, slots=True)
class ProactiveFrameCandidate:
    profile_id: str
    instance_id: str
    source_kind: ProactiveFrameSourceKind
    source_ref: str
    planned_main_core_at: datetime
    seed_planned_main_core_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ProactiveFrameTiming:
    source_ref: str
    planned_main_core_at: datetime
    gap_minutes: int
    frame_end_at: datetime
    deadline_at: datetime


def deterministic_gap_minutes(source_ref: str, planned_main_core_at: datetime) -> int:
    """Map one stable proactive occurrence to a triangular 12/20/35 minute gap."""

    reference = str(source_ref or "").strip()
    if not reference:
        raise ValueError("proactive frame source_ref cannot be empty")
    planned = _utc_datetime(planned_main_core_at, "planned MainCore time")
    canonical = f"{reference}\n{planned.isoformat()}".encode()
    digest = hashlib.sha256(canonical).digest()
    unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
    minimum, mode, maximum = PROACTIVE_FRAME_GAP_MINUTES
    split = (mode - minimum) / (maximum - minimum)
    if unit < split:
        sampled = minimum + math.sqrt(unit * (maximum - minimum) * (mode - minimum))
    else:
        sampled = maximum - math.sqrt((1.0 - unit) * (maximum - minimum) * (maximum - mode))
    return max(minimum, min(maximum, math.floor(sampled + 0.5)))


def proactive_frame_timing(
    source_ref: str,
    planned_main_core_at: datetime,
    *,
    admitted_at: datetime | None = None,
    effective_main_core_at: datetime | None = None,
) -> ProactiveFrameTiming:
    seed_planned = _utc_datetime(planned_main_core_at, "planned MainCore time")
    effective_planned = (
        _utc_datetime(effective_main_core_at, "effective MainCore time")
        if effective_main_core_at is not None
        else seed_planned
    )
    gap = deterministic_gap_minutes(source_ref, seed_planned)
    frame_end_at = effective_planned - timedelta(minutes=gap)
    start = (
        _utc_datetime(admitted_at, "proactive frame admission time")
        if admitted_at is not None
        else frame_end_at
    )
    return ProactiveFrameTiming(
        source_ref=str(source_ref).strip(),
        planned_main_core_at=effective_planned,
        gap_minutes=gap,
        frame_end_at=frame_end_at,
        deadline_at=start + timedelta(minutes=PROACTIVE_FRAME_TIMEOUT_MINUTES),
    )


def role_story_is_fresh(
    simulated_through_at: datetime | None,
    *,
    main_core_at: datetime,
) -> bool:
    main_core = _utc_datetime(main_core_at, "MainCore freshness time")
    if simulated_through_at is None:
        return False
    simulated = _utc_datetime(simulated_through_at, "simulated-through time")
    return simulated > main_core - timedelta(minutes=PROACTIVE_FRAME_FRESH_MINUTES)


def proactive_frame_cooldown_active(
    last_attempt_at: datetime | None,
    *,
    now: datetime,
) -> bool:
    current = _utc_datetime(now, "proactive frame cooldown time")
    if last_attempt_at is None:
        return False
    attempted = _utc_datetime(last_attempt_at, "last proactive frame attempt time")
    return attempted > current - timedelta(minutes=PROACTIVE_FRAME_COOLDOWN_MINUTES)


def _utc_datetime(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


__all__ = [
    "PROACTIVE_FRAME_COOLDOWN_MINUTES",
    "PROACTIVE_FRAME_FRESH_MINUTES",
    "PROACTIVE_FRAME_GAP_MINUTES",
    "PROACTIVE_FRAME_TIMEOUT_MINUTES",
    "ProactiveFrameCandidate",
    "ProactiveFrameSourceKind",
    "ProactiveFrameTiming",
    "deterministic_gap_minutes",
    "proactive_frame_cooldown_active",
    "proactive_frame_timing",
    "role_story_is_fresh",
]
