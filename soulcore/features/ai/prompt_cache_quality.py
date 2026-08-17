"""Deterministic prompt-cache quality tracking stored in capability evidence JSON."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

QUALITY_REJECTION_KIND = "CACHE_QUALITY"
MARKER_REJECTION_KIND = "MARKER_UNSUPPORTED"
MIN_REUSABLE_PREFIX_TOKENS = 4096
MAX_TRACKED_FAMILIES = 64


@dataclass(frozen=True, slots=True)
class PromptCacheQualitySample:
    observation_id: str
    predecessor_id: str
    cache_family: str
    request_started_at: datetime
    observed_at: datetime
    ttl_seconds: int
    read_tokens: int
    write_tokens: int
    read_fields: tuple[str, ...]
    write_fields: tuple[str, ...]
    breakpoints: tuple[Mapping[str, Any], ...]
    wire_mode: str
    cache_applied: bool
    observed_state: str


@dataclass(frozen=True, slots=True)
class PromptCacheQualitySettlement:
    evidence: dict[str, Any]
    rejection: dict[str, Any] | None
    state: str
    next_probe_at: datetime | None
    decision: str


def prompt_cache_quality_predecessor(evidence: Any, cache_family: str) -> str:
    quality = _quality_record(evidence)
    families = quality.get("families")
    if not isinstance(families, Mapping):
        return ""
    family = families.get(str(cache_family or ""))
    return str(family.get("sample_id") or "") if isinstance(family, Mapping) else ""


def settle_prompt_cache_quality(
    *,
    existing_evidence: Any,
    existing_rejection: Any,
    current_state: str,
    current_next_probe_at: datetime | None,
    sample: PromptCacheQualitySample,
) -> PromptCacheQualitySettlement:
    """Settle one response against exactly the predecessor claimed before sending it."""

    evidence = _evidence_record(existing_evidence)
    rejection = _mapping(existing_rejection) or None
    quality_rejected = str((rejection or {}).get("kind") or "") == QUALITY_REJECTION_KIND
    _record_quality_sample(evidence, sample)
    early = _pre_tracking_settlement(
        evidence,
        rejection,
        quality_rejected=quality_rejected,
        current_state=current_state,
        current_next_probe_at=current_next_probe_at,
        sample=sample,
    )
    if early is not None:
        return early

    observed_state = (
        sample.observed_state
        if sample.observed_state in {"CONFIRMED", "ACCEPTED_UNVERIFIED"}
        else current_state
    )
    if not _trackable_sample(sample):
        probe_state = (
            "PROBING" if quality_rejected and current_state == "PROBING" else observed_state
        )
        return _settlement(evidence, rejection, probe_state, None, "")
    return _settle_tracked_quality(
        evidence,
        rejection,
        quality_rejected=quality_rejected,
        current_state=current_state,
        current_next_probe_at=current_next_probe_at,
        observed_state=observed_state,
        sample=sample,
    )


def _record_quality_sample(evidence: dict[str, Any], sample: PromptCacheQualitySample) -> None:
    quality = evidence["quality"]
    quality["last_read_tokens"] = sample.read_tokens
    quality["last_write_tokens"] = sample.write_tokens
    evidence["latest"] = {
        "read_fields": list(sample.read_fields),
        "write_fields": list(sample.write_fields),
        "cache_read_tokens": sample.read_tokens,
        "cache_write_tokens": sample.write_tokens,
        "cache_family": sample.cache_family,
        "breakpoints": [dict(item) for item in sample.breakpoints],
    }


def _pre_tracking_settlement(
    evidence: dict[str, Any],
    rejection: dict[str, Any] | None,
    *,
    quality_rejected: bool,
    current_state: str,
    current_next_probe_at: datetime | None,
    sample: PromptCacheQualitySample,
) -> PromptCacheQualitySettlement | None:
    quality = evidence["quality"]
    # A response sent before suspension may add counters, but cannot weaken it.
    if quality_rejected and current_state == "REJECTED" and sample.cache_applied:
        quality["status"] = "SUSPENDED"
        return _settlement(
            evidence, rejection, current_state, current_next_probe_at, "QUALITY_SUSPENDED"
        )
    if sample.cache_applied:
        return None
    if quality_rejected and (sample.read_tokens > 0 or sample.write_tokens > 0):
        quality["status"] = "UNCONTROLLABLE_WARNING"
        quality["reason"] = "服务商在未发送显式缓存标记时仍报告缓存读写"
        return _settlement(
            evidence,
            rejection,
            current_state,
            current_next_probe_at,
            "CACHE_REPORTED_WHILE_DISABLED",
        )
    return _settlement(evidence, rejection, current_state, current_next_probe_at, "")


def _settle_tracked_quality(
    evidence: dict[str, Any],
    rejection: dict[str, Any] | None,
    *,
    quality_rejected: bool,
    current_state: str,
    current_next_probe_at: datetime | None,
    observed_state: str,
    sample: PromptCacheQualitySample,
) -> PromptCacheQualitySettlement:
    quality = evidence["quality"]

    families = quality["families"]
    previous = families.get(sample.cache_family)
    previous_id = str(previous.get("sample_id") or "") if isinstance(previous, Mapping) else ""
    if sample.predecessor_id != previous_id:
        return _settlement(evidence, rejection, current_state, current_next_probe_at, "")

    comparison = _compare(previous, sample)
    streak, decision = _apply_quality_comparison(quality, previous, comparison)
    _store_quality_family(families, sample, streak)
    reprobe = _reprobe_settlement(
        evidence,
        rejection,
        quality_rejected=quality_rejected,
        current_state=current_state,
        comparison=comparison,
        decision=decision,
        sample=sample,
    )
    if reprobe is not None:
        return reprobe
    repeated = _repeated_anomaly_settlement(
        evidence,
        rejection,
        comparison=comparison,
        streak=streak,
        observed_state=observed_state,
        sample=sample,
    )
    if repeated is not None:
        return repeated
    return _settlement(evidence, rejection, observed_state, None, decision)


def _apply_quality_comparison(
    quality: dict[str, Any], previous: Any, comparison: str
) -> tuple[int, str]:
    streak = int((previous or {}).get("anomaly_streak") or 0)
    if comparison == "SEVERE":
        streak += 1
        quality["status"] = "ANOMALY"
        quality["reason"] = "同前缀缓存读取极低且发生大量重写"
        quality["anomaly_count"] = streak
        return streak, "QUALITY_ANOMALY"
    elif comparison in {"HEALTHY", "NORMAL"}:
        streak = 0
        quality["anomaly_count"] = 0
        quality["status"] = "HEALTHY"
        quality["reason"] = ""
    elif comparison == "INCOMPARABLE":
        # An expired or changed prefix becomes a fresh baseline. It must not
        # carry an anomaly streak from the older prefix into a later response.
        streak = 0
        quality["anomaly_count"] = 0
        if quality.get("status") == "ANOMALY":
            quality["status"] = "OBSERVING"
            quality["reason"] = ""
    return streak, ""


def _store_quality_family(
    families: dict[str, Any], sample: PromptCacheQualitySample, streak: int
) -> None:
    family_record = {
        "sample_id": sample.observation_id,
        "observed_at": _iso(sample.observed_at),
        "request_started_at": _iso(sample.request_started_at),
        "breakpoints": [_compact_breakpoint(item) for item in sample.breakpoints],
        "anomaly_streak": streak,
        "last_read_tokens": sample.read_tokens,
        "last_write_tokens": sample.write_tokens,
    }
    families[sample.cache_family] = family_record
    _trim_families(families)


def _reprobe_settlement(
    evidence: dict[str, Any],
    rejection: dict[str, Any] | None,
    *,
    quality_rejected: bool,
    current_state: str,
    comparison: str,
    decision: str,
    sample: PromptCacheQualitySample,
) -> PromptCacheQualitySettlement | None:
    if quality_rejected and current_state == "PROBING":
        quality = evidence["quality"]
        quality["status"] = "REPROBING"
        if comparison == "HEALTHY":
            quality.update(
                {
                    "status": "HEALTHY",
                    "reason": "",
                    "anomaly_count": 0,
                    "trip_count": 0,
                }
            )
            return _settlement(evidence, None, "CONFIRMED", None, "QUALITY_RECOVERED")
        if comparison == "SEVERE":
            return _suspend(evidence, rejection, sample, immediate=True)
        return _settlement(evidence, rejection, "PROBING", None, decision)
    return None


def _repeated_anomaly_settlement(
    evidence: dict[str, Any],
    rejection: dict[str, Any] | None,
    *,
    comparison: str,
    streak: int,
    observed_state: str,
    sample: PromptCacheQualitySample,
) -> PromptCacheQualitySettlement | None:
    explicit = sample.wire_mode in {"OPENAI_EXPLICIT", "ANTHROPIC_EPHEMERAL"}
    if explicit and comparison == "SEVERE" and streak >= 2:
        return _suspend(evidence, rejection, sample, immediate=False)

    if sample.wire_mode == "OPENAI_AUTO" and comparison == "SEVERE" and streak >= 2:
        quality = evidence["quality"]
        quality["status"] = "AUTO_WARNING"
        quality["reason"] = "服务商自动缓存连续出现极低读取和大量重写"
        return _settlement(evidence, rejection, observed_state, None, "AUTO_QUALITY_WARNING")
    return None


def quality_retry_ready(evidence: Any) -> dict[str, Any]:
    updated = _evidence_record(evidence)
    quality = updated["quality"]
    quality["status"] = "REPROBE_READY"
    quality["reason"] = "已提前解除等待，将由下一次符合条件的真实请求复探"
    return updated


def quality_probe_started(evidence: Any) -> dict[str, Any]:
    updated = _evidence_record(evidence)
    quality = updated["quality"]
    quality["status"] = "REPROBING"
    quality["reason"] = "正在随符合条件的真实请求复探"
    return updated


def _compare(previous: Any, sample: PromptCacheQualitySample) -> str:
    if not isinstance(previous, Mapping):
        return "BASELINE"
    if not sample.read_fields or not sample.write_fields:
        return "INCOMPARABLE"
    observed_at = _datetime(previous.get("observed_at"))
    if observed_at is None:
        return "INCOMPARABLE"
    elapsed = (sample.request_started_at - observed_at).total_seconds()
    if elapsed < 0 or elapsed > max(1, int(sample.ttl_seconds)):
        return "INCOMPARABLE"
    reusable = _reusable_prefix_tokens(previous.get("breakpoints"), sample.breakpoints)
    if reusable < MIN_REUSABLE_PREFIX_TOKENS:
        return "INCOMPARABLE"
    if sample.read_tokens * 10 < reusable and sample.write_tokens * 2 >= reusable:
        return "SEVERE"
    if sample.read_tokens * 2 >= reusable and sample.write_tokens * 4 <= reusable:
        return "HEALTHY"
    return "NORMAL"


def _trackable_sample(sample: PromptCacheQualitySample) -> bool:
    maximum_prefix = max(
        (max(0, int(item.get("prefix_tokens") or 0)) for item in sample.breakpoints),
        default=0,
    )
    return bool(sample.read_fields and sample.write_fields) and (
        maximum_prefix >= MIN_REUSABLE_PREFIX_TOKENS
    )


def _reusable_prefix_tokens(previous: Any, current: Sequence[Mapping[str, Any]]) -> int:
    if not isinstance(previous, Sequence) or isinstance(previous, (str, bytes)):
        return 0
    previous_hashes = {
        str(item.get("prefix_hash") or ""): max(0, int(item.get("prefix_tokens") or 0))
        for item in previous
        if isinstance(item, Mapping) and str(item.get("prefix_hash") or "")
    }
    return max(
        (
            min(
                previous_hashes.get(str(item.get("prefix_hash") or ""), 0),
                max(0, int(item.get("prefix_tokens") or 0)),
            )
            for item in current
            if str(item.get("prefix_hash") or "") in previous_hashes
        ),
        default=0,
    )


def _suspend(
    evidence: dict[str, Any],
    previous_rejection: Mapping[str, Any] | None,
    sample: PromptCacheQualitySample,
    *,
    immediate: bool,
) -> PromptCacheQualitySettlement:
    prior_trip = int((previous_rejection or {}).get("trip_count") or 0)
    trip = prior_trip + 1
    cooldown_hours = (1, 6, 24)[min(trip - 1, 2)]
    next_probe = sample.observed_at + timedelta(hours=cooldown_hours)
    quality = evidence["quality"]
    quality.update(
        {
            "status": "SUSPENDED",
            "reason": "同前缀缓存连续极低读取且大量重写，已暂停显式缓存",
            "trip_count": trip,
        }
    )
    rejection = {
        "kind": QUALITY_REJECTION_KIND,
        "reason_code": "EXTREME_LOW_READ_HIGH_WRITE",
        "trip_count": trip,
        "cooldown_hours": cooldown_hours,
        "suspended_at": _iso(sample.observed_at),
        "cache_read_tokens": sample.read_tokens,
        "cache_write_tokens": sample.write_tokens,
        "reprobe_failure": bool(immediate),
    }
    return _settlement(evidence, rejection, "REJECTED", next_probe, "QUALITY_SUSPENDED")


def _evidence_record(value: Any) -> dict[str, Any]:
    existing = _current_evidence(value)
    if not existing:
        return {
            "schema_version": 2,
            "latest": {},
            "quality": _quality_record(existing),
        }
    evidence = dict(existing)
    evidence["latest"] = dict(existing["latest"])
    evidence["quality"] = _quality_record(existing)
    return evidence


def _quality_record(evidence: Any) -> dict[str, Any]:
    source = _current_evidence(evidence)
    quality = dict(source["quality"]) if source else {}
    families = quality.get("families")
    quality["families"] = {
        str(key): dict(value)
        for key, value in (families.items() if isinstance(families, Mapping) else ())
        if isinstance(value, Mapping)
    }
    quality.setdefault("status", "OBSERVING")
    quality.setdefault("reason", "")
    quality.setdefault("anomaly_count", 0)
    quality.setdefault("last_read_tokens", 0)
    quality.setdefault("last_write_tokens", 0)
    return quality


def _current_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("prompt-cache evidence must use the current schema")
    evidence = dict(value)
    if not evidence:
        return {}
    if (
        evidence.get("schema_version") != 2
        or not isinstance(evidence.get("latest"), Mapping)
        or not isinstance(evidence.get("quality"), Mapping)
    ):
        raise ValueError("prompt-cache evidence must use the current schema")
    return evidence


def _trim_families(families: dict[str, Any]) -> None:
    if len(families) <= MAX_TRACKED_FAMILIES:
        return
    oldest = sorted(
        families,
        key=lambda key: str((families.get(key) or {}).get("observed_at") or ""),
    )[: len(families) - MAX_TRACKED_FAMILIES]
    for key in oldest:
        families.pop(key, None)


def _compact_breakpoint(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prefix_hash": str(value.get("prefix_hash") or ""),
        "prefix_tokens": max(0, int(value.get("prefix_tokens") or 0)),
    }


def _settlement(
    evidence: dict[str, Any],
    rejection: dict[str, Any] | None,
    state: str,
    next_probe_at: datetime | None,
    decision: str,
) -> PromptCacheQualitySettlement:
    return PromptCacheQualitySettlement(evidence, rejection, state, next_probe_at, decision)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        result = value
    elif value:
        try:
            result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)


def _iso(value: datetime) -> str:
    normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return normalized.isoformat()


__all__ = [
    "MARKER_REJECTION_KIND",
    "MAX_TRACKED_FAMILIES",
    "MIN_REUSABLE_PREFIX_TOKENS",
    "PromptCacheQualitySample",
    "PromptCacheQualitySettlement",
    "QUALITY_REJECTION_KIND",
    "prompt_cache_quality_predecessor",
    "quality_probe_started",
    "quality_retry_ready",
    "settle_prompt_cache_quality",
]
