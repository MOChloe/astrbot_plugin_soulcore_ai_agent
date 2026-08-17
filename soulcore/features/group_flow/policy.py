"""Pure, replayable group-flow scheduling rules."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from ...contracts.group_flow import GroupFlowPolicy

RATE_HALF_LIFE_SECONDS = 10.0
JUDGE_SAMPLE_SECONDS = 5.0
MAX_JUDGE_MESSAGE_COUNT = 4096
ABSOLUTE_WINDOW_SECONDS = 600
FULL_INTERJECTION_CHECK_RATE = 0.125
INTERJECTION_CHECK_RATE_SCALE = 0.30


@dataclass(frozen=True, slots=True)
class GroupSchedule:
    rate_ewma: float
    judge_threshold: int
    next_judge_at: datetime
    quiet_due_at: datetime
    dynamic_due_at: datetime
    direct_due_at: datetime | None


def update_message_rate(
    previous_rate: float,
    previous_at: datetime | None,
    occurred_at: datetime,
) -> float:
    if previous_at is None:
        return max(0.0, float(previous_rate))
    elapsed = max(0.001, (occurred_at - previous_at).total_seconds())
    instantaneous = 1.0 / elapsed
    decay = math.exp(-math.log(2.0) * elapsed / RATE_HALF_LIFE_SECONDS)
    return max(0.0, float(previous_rate)) * decay + instantaneous * (1.0 - decay)


def judge_message_threshold(
    policy: GroupFlowPolicy,
    *,
    rate_ewma: float,
    repeat_ratio: float,
) -> int:
    threshold = max(policy.base_message_count, math.ceil(max(0.0, rate_ewma) * 5.0))
    if repeat_ratio >= 0.85:
        threshold *= 4
    elif repeat_ratio >= 0.60:
        threshold *= 2
    return min(MAX_JUDGE_MESSAGE_COUNT, threshold)


def dynamic_wait_seconds(
    *,
    rate_ewma: float,
    repeat_ratio: float,
    seconds_since_last_visible: float | None = None,
) -> float:
    wait = 60.0 + max(0.0, rate_ewma) * 30.0
    if seconds_since_last_visible is not None:
        age = max(0.0, seconds_since_last_visible)
        if age < 60.0:
            wait += 180.0 * (1.0 - age / 60.0)
        elif age < 300.0:
            wait += 60.0 * (1.0 - (age - 60.0) / 240.0)
    wait = min(300.0, max(60.0, wait))
    if repeat_ratio >= 0.85:
        wait *= 0.5
    elif repeat_ratio >= 0.60:
        wait *= 0.75
    return min(300.0, max(60.0, wait))


def build_schedule(
    policy: GroupFlowPolicy,
    *,
    first_at: datetime,
    last_at: datetime,
    previous_rate: float,
    previous_at: datetime | None,
    repeat_ratio: float,
    direct_address: bool,
    last_visible_at: datetime | None = None,
    now: datetime | None = None,
) -> GroupSchedule:
    rate = update_message_rate(previous_rate, previous_at, last_at)
    threshold = judge_message_threshold(
        policy,
        rate_ewma=rate,
        repeat_ratio=repeat_ratio,
    )
    dynamic_due = first_at + timedelta(
        seconds=min(
            ABSOLUTE_WINDOW_SECONDS,
            dynamic_wait_seconds(
                rate_ewma=rate,
                repeat_ratio=repeat_ratio,
                seconds_since_last_visible=(
                    None
                    if last_visible_at is None
                    else ((now or last_at) - last_visible_at).total_seconds()
                ),
            ),
        )
    )
    return GroupSchedule(
        rate_ewma=rate,
        judge_threshold=threshold,
        next_judge_at=last_at + timedelta(seconds=JUDGE_SAMPLE_SECONDS),
        quiet_due_at=last_at + timedelta(seconds=policy.quiet_seconds),
        dynamic_due_at=dynamic_due,
        direct_due_at=last_at + timedelta(seconds=2) if direct_address else None,
    )


def reply_gap_due_at(
    policy: GroupFlowPolicy,
    *,
    last_visible_at: datetime | None,
) -> datetime | None:
    if last_visible_at is None or policy.ordinary_min_reply_gap_seconds == 0:
        return None
    return last_visible_at + timedelta(seconds=policy.ordinary_min_reply_gap_seconds)


def group_interjection_check_probability(rate_ewma: float) -> float:
    """Return one admission chance for a settled ordinary group window."""
    rate = max(0.0, float(rate_ewma))
    if rate <= FULL_INTERJECTION_CHECK_RATE:
        return 1.0
    excess = (rate - FULL_INTERJECTION_CHECK_RATE) / INTERJECTION_CHECK_RATE_SCALE
    return 1.0 / (1.0 + excess * excess)


def stable_interjection_check_value(window_id: str, last_message_id: int) -> float:
    material = f"group-interjection-check:{window_id}:{int(last_message_id)}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / 2**64


__all__ = [
    "ABSOLUTE_WINDOW_SECONDS",
    "GroupSchedule",
    "build_schedule",
    "dynamic_wait_seconds",
    "group_interjection_check_probability",
    "judge_message_threshold",
    "reply_gap_due_at",
    "stable_interjection_check_value",
    "update_message_rate",
]
