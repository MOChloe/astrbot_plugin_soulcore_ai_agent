"""Mapping adapters for persisted ContactClock claims."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from .contact_models import (
    ContactClaim,
    ContactEvidenceKind,
    ContactPolicy,
    TimelineEvidence,
)


def _quiet_until(now: datetime, policy: ContactPolicy) -> datetime | None:
    start = policy.quiet_start_minute
    end = policy.quiet_end_minute
    current = now.hour * 60 + now.minute
    if start == end:
        return None
    quiet = start <= current < end if start < end else current >= start or current < end
    if not quiet:
        return None
    target = now.replace(hour=end // 60, minute=end % 60, second=0, microsecond=0)
    if (start > end and current >= start) or target <= now:
        target += timedelta(days=1)
    return target


def _field(value: Mapping[str, Any], name: str, default: Any = None) -> Any:
    return value.get(name, default)


def _contact_claim(value: Mapping[str, Any], *, profile_id: str, instance_id: str) -> ContactClaim:
    return ContactClaim(
        profile_id=str(_field(value, "profile_id", profile_id)),
        instance_id=str(_field(value, "instance_id", instance_id)),
        generation=_integer_field(value, "generation"),
        activity_epoch=_integer_field(value, "activity_epoch_snapshot"),
        evidence=_claim_evidence(value),
        state_epoch=_integer_field(value, "state_epoch"),
        last_contact_at=_datetime_field(value, "last_success_at"),
        contacts_today=_integer_field(value, "daily_success_count"),
        unanswered_count=_integer_field(value, "consecutive_unanswered"),
        route_umo=str(_field(value, "route_umo", "")),
        version=_integer_field(value, "version"),
        lease_token=_integer_field(value, "lease_token"),
        timeline_event_watermark=_integer_field(value, "timeline_event_watermark"),
        timeline_event_through=_snapshot_integer(value, "timeline_event_through"),
        action_event_through=_snapshot_integer(value, "action_event_through"),
        reroll_count=_claim_reroll_count(value),
    )


def _integer_field(value: Mapping[str, Any], name: str) -> int:
    return int(_field(value, name, 0) or 0)


def _snapshot_integer(value: Mapping[str, Any], name: str) -> int:
    snapshot = _field(value, "evidence_snapshot", {}) or {}
    return int(snapshot.get(name, 0) or 0) if isinstance(snapshot, Mapping) else 0


def _claim_evidence(value: Mapping[str, Any]) -> tuple[TimelineEvidence, ...]:
    return tuple(_contact_evidence(raw) for raw in _field(value, "evidence", ()) or ())


def _claim_reroll_count(value: Mapping[str, Any]) -> int:
    deferred = _field(value, "deferred_evidence", {}) or {}
    if not isinstance(deferred, Mapping):
        return 0
    return max(0, int(deferred.get("reroll_count") or 0))


def _contact_evidence(raw: Any) -> TimelineEvidence:
    if not isinstance(raw, Mapping):
        raise ValueError("contact evidence must be an object")
    occurred = _datetime_value(_field(raw, "occurred_at"))
    if occurred is None:
        raise ValueError("contact evidence occurred_at must be datetime")
    kind = str(_field(raw, "evidence_kind", ContactEvidenceKind.ROLE_TIMELINE_EVENT.value)).upper()
    return TimelineEvidence(
        evidence_id=str(_field(raw, "evidence_id", "")),
        summary=str(_field(raw, "summary", "")),
        occurred_at=occurred,
        important=bool(_field(raw, "important", False)),
        evidence_kind=ContactEvidenceKind(kind),
        importance=float(_field(raw, "importance", 0.0) or 0.0),
        reason=str(_field(raw, "reason", "")),
    )


def _datetime_value(raw: Any) -> datetime | None:
    return raw if isinstance(raw, datetime) else _parse_datetime(raw)


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _datetime_field(value: Mapping[str, Any], name: str) -> datetime | None:
    raw = value.get(name)
    if isinstance(raw, datetime):
        return raw
    return _parse_datetime(raw)
