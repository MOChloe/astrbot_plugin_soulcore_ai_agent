"""Run-scoped validation for Main Core's explicit temporary absence decision."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

TEMPORARY_ABSENCE_REASON_CODE = "TEMPORARY_ABSENCE"
TEMPORARY_ABSENCE_EXPIRY_KEY = "temporary_absence_expiry"
TEMPORARY_ABSENCE_EXPIRY_IDEMPOTENCY_PREFIX = "temporary-absence-expiry:"


@dataclass(frozen=True, slots=True)
class TemporaryAbsenceExpiryWake:
    gate_generation: int
    activity_epoch: int
    source_run_id: int

    def __post_init__(self) -> None:
        if self.gate_generation < 1 or self.activity_epoch < 0 or self.source_run_id < 0:
            raise ValueError("temporary absence expiry wake metadata is invalid")

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any] | None,
    ) -> TemporaryAbsenceExpiryWake | None:
        if not isinstance(metadata, Mapping) or TEMPORARY_ABSENCE_EXPIRY_KEY not in metadata:
            return None
        value = metadata.get(TEMPORARY_ABSENCE_EXPIRY_KEY)
        if not isinstance(value, Mapping):
            raise ValueError("temporary absence expiry wake metadata is invalid")
        try:
            gate_generation = int(value.get("gate_generation") or 0)
            activity_epoch = int(value.get("activity_epoch") or 0)
            source_run_id = int(value.get("source_run_id") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("temporary absence expiry wake metadata is invalid") from exc
        if gate_generation < 1 or activity_epoch < 0 or source_run_id < 0:
            raise ValueError("temporary absence expiry wake metadata is invalid")
        return cls(
            gate_generation=gate_generation,
            activity_epoch=activity_epoch,
            source_run_id=source_run_id,
        )

    def as_metadata(self) -> dict[str, int]:
        return {
            "gate_generation": self.gate_generation,
            "activity_epoch": self.activity_epoch,
            "source_run_id": self.source_run_id,
        }

    @property
    def idempotency_key(self) -> str:
        owner = (
            f"run:{self.source_run_id}"
            if self.source_run_id > 0
            else f"gate:{self.gate_generation}"
        )
        return f"{TEMPORARY_ABSENCE_EXPIRY_IDEMPOTENCY_PREFIX}{owner}"


def temporary_absence_metadata(
    *,
    reason: str,
    started_at: datetime | None,
    planned_until: datetime | None,
    ended_at: datetime,
    end_reason: str,
) -> dict[str, Any]:
    """Build the bounded Main Core view of a just-ended temporary absence."""

    if not str(reason or "").strip() or started_at is None or planned_until is None:
        return {}
    elapsed = max(0, int((ended_at - started_at).total_seconds()))
    return {
        "status": "ENDED",
        "reason": str(reason).strip()[:1000],
        "started_at": started_at.isoformat(),
        "planned_until": planned_until.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": elapsed,
        "end_reason": str(end_reason or "NATURAL_EXPIRY").strip().upper(),
    }


def temporary_absence_expiry_payload(
    *,
    reason: str,
    started_at: datetime,
    planned_until: datetime,
    gate_generation: int,
    activity_epoch: int,
    source_run_id: int,
) -> dict[str, Any]:
    marker = TemporaryAbsenceExpiryWake(
        gate_generation=int(gate_generation),
        activity_epoch=int(activity_epoch),
        source_run_id=int(source_run_id),
    )
    return {
        TEMPORARY_ABSENCE_EXPIRY_KEY: marker.as_metadata(),
        "temporary_absence": temporary_absence_metadata(
            reason=reason,
            started_at=started_at,
            planned_until=planned_until,
            ended_at=planned_until,
            end_reason="NATURAL_EXPIRY",
        ),
    }


__all__ = [
    "TEMPORARY_ABSENCE_EXPIRY_IDEMPOTENCY_PREFIX",
    "TEMPORARY_ABSENCE_EXPIRY_KEY",
    "TEMPORARY_ABSENCE_REASON_CODE",
    "TemporaryAbsenceExpiryWake",
    "temporary_absence_expiry_payload",
    "temporary_absence_metadata",
]
