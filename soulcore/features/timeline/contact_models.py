"""ContactClock policy, evidence, claim, and repository contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class ContactEvidenceKind(StrEnum):
    ROLE_TIMELINE_EVENT = "ROLE_TIMELINE_EVENT"
    ACTION_RESULT = "ACTION_RESULT"


class ContactOutcome(StrEnum):
    READY = "READY"
    NO_EVIDENCE = "NO_EVIDENCE"
    DISABLED_CONSUMED = "DISABLED_CONSUMED"
    QUIET_HOURS = "QUIET_HOURS"
    MIN_INTERVAL = "MIN_INTERVAL"
    DAILY_LIMIT = "DAILY_LIMIT"
    UNANSWERED_LIMIT = "UNANSWERED_LIMIT"
    SUPERSEDED_BY_FOREGROUND = "SUPERSEDED_BY_FOREGROUND"
    SUPERSEDED_BY_CORE_STATE = "SUPERSEDED_BY_CORE_STATE"


def contact_day_bucket_transition(stored: object, observed: object) -> tuple[str, bool]:
    """Keep daily quotas monotonic when the host clock/date moves backwards.

    The boolean tells callers whether the existing count belongs to the
    effective bucket and must be carried forward.
    """

    previous = str(stored or "").strip()
    current = str(observed or "").strip()
    if not current:
        raise ValueError("observed contact day bucket cannot be empty")
    if previous and previous >= current:
        return previous, True
    return current, False


@dataclass(frozen=True, slots=True)
class ContactPolicy:
    """Platform-independent defaults confirmed for the first batch."""

    enabled: bool = True
    check_min_minutes: int = 180
    check_max_minutes: int = 480
    quiet_enabled: bool = True
    quiet_start_minute: int = 23 * 60
    quiet_end_minute: int = 8 * 60
    min_contact_interval_minutes: int = 120
    daily_limit: int | None = 6
    unanswered_limit: int | None = None
    timezone_name: str | None = None
    failure_mode: str = "SKIP"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ContactPolicy:
        daily_mode = str(value.get("daily_limit_mode", "LIMITED")).upper()
        unanswered_mode = str(value.get("unanswered_limit_mode", "UNLIMITED")).upper()
        return cls(
            enabled=bool(value.get("proactive_enabled", True)),
            check_min_minutes=int(value.get("check_min_minutes", 180)),
            check_max_minutes=int(value.get("check_max_minutes", 480)),
            quiet_enabled=bool(value.get("quiet_enabled", True)),
            quiet_start_minute=_minute_of_day(value.get("quiet_start", "23:00")),
            quiet_end_minute=_minute_of_day(value.get("quiet_end", "08:00")),
            min_contact_interval_minutes=int(value.get("min_success_gap_minutes", 120)),
            daily_limit=(
                None if daily_mode == "UNLIMITED" else int(value.get("daily_success_limit") or 6)
            ),
            unanswered_limit=(
                None
                if unanswered_mode == "UNLIMITED"
                else int(value.get("max_consecutive_unanswered") or 1)
            ),
            timezone_name=(str(value.get("timezone") or "").strip() or None),
            failure_mode=str(value.get("failure_mode") or "SKIP").upper(),
        )

    def __post_init__(self) -> None:
        if self.check_min_minutes < 1:
            raise ValueError("contact check minimum must be positive")
        if self.check_max_minutes < self.check_min_minutes:
            raise ValueError("contact check maximum cannot be below minimum")
        if not 0 <= self.quiet_start_minute < 24 * 60:
            raise ValueError("quiet_start_minute must be within one day")
        if not 0 <= self.quiet_end_minute < 24 * 60:
            raise ValueError("quiet_end_minute must be within one day")
        if self.min_contact_interval_minutes < 0:
            raise ValueError("minimum contact interval cannot be negative")
        if (self.daily_limit is not None and self.daily_limit < 1) or (
            self.unanswered_limit is not None and self.unanswered_limit < 1
        ):
            raise ValueError("contact limits must be positive or unlimited")
        if self.failure_mode not in {"SKIP", "RETRY_BACKOFF"}:
            raise ValueError("contact failure_mode must be SKIP or RETRY_BACKOFF")


@dataclass(frozen=True, slots=True)
class TimelineEvidence:
    evidence_id: str
    summary: str
    occurred_at: datetime
    important: bool = False
    evidence_kind: ContactEvidenceKind = ContactEvidenceKind.ROLE_TIMELINE_EVENT
    importance: float = 0.0
    reason: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return self.evidence_kind.value, self.evidence_id


@dataclass(frozen=True, slots=True)
class ContactClaim:
    profile_id: str
    instance_id: str
    generation: int
    activity_epoch: int
    evidence: tuple[TimelineEvidence, ...]
    state_epoch: int = 0
    last_contact_at: datetime | None = None
    contacts_today: int = 0
    unanswered_count: int = 0
    route_umo: str = ""
    version: int = 0
    lease_token: int = 0
    timeline_event_watermark: int = 0
    timeline_event_through: int = 0
    action_event_through: int = 0
    reroll_count: int = 0

    @property
    def latest_timeline_event_id(self) -> str | None:
        timeline = [
            item
            for item in self.evidence
            if item.evidence_kind is ContactEvidenceKind.ROLE_TIMELINE_EVENT
        ]
        return timeline[-1].evidence_id if timeline else None


@dataclass(frozen=True, slots=True)
class ContactEvaluation:
    outcome: ContactOutcome
    next_check_at: datetime
    consume_through_evidence_id: str | None
    evidence_snapshot: tuple[TimelineEvidence, ...] = ()
    retry_not_before: datetime | None = None


@dataclass(frozen=True, slots=True)
class ContactOpportunity:
    """A committed contact attempt that the integration layer may wake later."""

    profile_id: str
    instance_id: str
    generation: int
    activity_epoch: int
    route_umo: str
    evidence: tuple[TimelineEvidence, ...]
    state_epoch: int = 0
    attempt_ref: str = ""
    failure_mode: str = "SKIP"
    reroll_count: int = 0
    proactive_frame_planned_at: datetime | None = None
    proactive_frame_source_ref: str = ""


@runtime_checkable
class ContactClockRepository(Protocol):
    async def resolve_contact_policy(self, *args: object, **kwargs: object) -> Any: ...
    async def list_role_timeline_events(self, *args: object, **kwargs: object) -> Any: ...
    async def list_contact_action_results(self, *args: object, **kwargs: object) -> Any: ...
    async def reserve_contact_evidence(self, *args: object, **kwargs: object) -> Any: ...
    async def settle_contact_evidence(self, *args: object, **kwargs: object) -> Any: ...
    async def finalize_contact_attempt(self, *args: object, **kwargs: object) -> Any: ...
    async def release_contact_clock(self, *args: object, **kwargs: object) -> Any: ...

    async def claim_contact_clock(
        self,
        *,
        now: datetime,
        limit: int = 10,
        profile_id: str | None = None,
        instance_id: str | None = None,
    ) -> Sequence[object]: ...

    async def commit_contact_clock(
        self,
        profile_id: str,
        instance_id: str,
        *,
        expected_version: int,
        lease_token: int,
        expected_generation: int,
        expected_state_epoch: int,
        expected_activity_epoch: int,
        next_check_at: datetime,
        result: str,
        reason: str,
        timeline_event_watermark: int | None,
        deferred_evidence: Mapping[str, object] | None,
        attempt_ref: str | None,
        now: datetime,
    ) -> Any | None: ...

    async def expedite_contact_clock(
        self,
        profile_id: str,
        instance_id: str,
        *,
        event_id: str,
        due_at: datetime,
        now: datetime | None = None,
    ) -> bool: ...

    async def invalidate_contact_clock_for_foreground(
        self,
        profile_id: str,
        instance_id: str,
        *,
        activity_epoch: int,
        defer_until: datetime,
    ) -> bool: ...


@runtime_checkable
class ContactProfileRepository(Protocol):
    async def get_instance_state(self, *args: object, **kwargs: object) -> Any: ...
    async def get_character_instance(self, *args: object, **kwargs: object) -> Any: ...


def _minute_of_day(value: Any) -> int:
    text = str(value).strip()
    if ":" in text:
        hour, minute = text.split(":", 1)
        return int(hour) * 60 + int(minute)
    return int(text) * 60
