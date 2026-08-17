"""Internal records for file-backed Main Core work recovery.

These records are persistence fences only.  They are never prompt, conversation,
memo, expression, event, or player-output payloads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from .work_checkpoint import WorkScope
from .work_continuity import MainCoreWorkSession
from .work_recovery import MainCoreWorkRecoveryEnvelope

_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,159}$")


def _token(value: str, label: str) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{label} is not a bounded token")
    return value


def _aware(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


class FileWorkBindingStatus(StrEnum):
    PENDING = "PENDING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RecoveryWakeStatus(StrEnum):
    READY = "READY"
    CLAIMED = "CLAIMED"
    EXPIRED = "EXPIRED"
    SUPERSEDED = "SUPERSEDED"


@dataclass(frozen=True, slots=True)
class FileWorkBindingSpec:
    job_id: str
    request_ref: str
    slot_id: str

    def __post_init__(self) -> None:
        _token(self.job_id, "job_id")
        _token(self.request_ref, "request_ref")
        _token(self.slot_id, "slot_id")


@dataclass(frozen=True, slots=True)
class FileWorkBindingRecord:
    scope: WorkScope
    work_ref: str
    job_id: str
    request_ref: str
    slot_id: str
    checkpoint_version: int
    run_generation: int
    callback_sequence: int
    status: FileWorkBindingStatus
    resource_ref: str
    result_kind: str
    result_summary: str
    todo_id: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    def __post_init__(self) -> None:
        for value, label in (
            (self.work_ref, "work_ref"),
            (self.job_id, "job_id"),
            (self.request_ref, "request_ref"),
            (self.slot_id, "slot_id"),
        ):
            _token(value, label)
        if min(self.checkpoint_version, self.run_generation, self.callback_sequence) <= 0:
            raise ValueError("file work fences must be positive")
        _aware(self.created_at, "created_at")
        _aware(self.updated_at, "updated_at")
        if self.completed_at is not None:
            _aware(self.completed_at, "completed_at")


@dataclass(frozen=True, slots=True)
class RecoveryWakeRecord:
    scope: WorkScope
    work_ref: str
    checkpoint_version: int
    wakeup_id: int
    status: RecoveryWakeStatus
    claimed_run_id: int | None
    created_at: datetime
    updated_at: datetime
    claimed_at: datetime | None
    terminal_reason: str

    def __post_init__(self) -> None:
        _token(self.work_ref, "work_ref")
        if self.checkpoint_version <= 0 or self.wakeup_id <= 0:
            raise ValueError("recovery wake fences must be positive")
        _aware(self.created_at, "created_at")
        _aware(self.updated_at, "updated_at")
        if self.claimed_at is not None:
            _aware(self.claimed_at, "claimed_at")


@dataclass(frozen=True, slots=True)
class RecoveryWakeClaim:
    wake: RecoveryWakeRecord
    envelope: MainCoreWorkRecoveryEnvelope
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class WorkRecoveryRunStart:
    run_id: int
    session: MainCoreWorkSession
    envelope: MainCoreWorkRecoveryEnvelope
    controlled_resource_refs: frozenset[str]
    expected_state_epoch: int
    expected_activity_epoch: int
    replayed: bool = False


__all__ = [
    "FileWorkBindingRecord",
    "FileWorkBindingSpec",
    "FileWorkBindingStatus",
    "RecoveryWakeClaim",
    "RecoveryWakeRecord",
    "RecoveryWakeStatus",
    "WorkRecoveryRunStart",
]
