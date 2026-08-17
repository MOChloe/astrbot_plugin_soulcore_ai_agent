"""Typed persistence commands and internal records for work checkpoints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from .work_checkpoint import (
    MainCoreWorkCheckpoint,
    WorkCallbackEnvelope,
    WorkCallbackLease,
    WorkRecoveryRuntime,
    WorkScope,
)
from .work_recovery import MainCoreWorkRecoveryEnvelope, WorkRecoveryDecision

MAX_RUNTIME_CONTROLLED_REFS = 256
MAX_EVENT_PAGE_SIZE = 256
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,159}$")


def _token(value: str, label: str) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{label} is not a bounded token")
    return value


def _aware(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


class WorkCheckpointEventKind(StrEnum):
    CREATED = "CREATED"
    CALLBACK_ACCEPTED = "CALLBACK_ACCEPTED"
    CALLBACK_REJECTED = "CALLBACK_REJECTED"
    LEASE_CLAIMED = "LEASE_CLAIMED"
    LEASE_RENEWED = "LEASE_RENEWED"
    LEASE_RELEASED = "LEASE_RELEASED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    SUPERSEDED = "SUPERSEDED"


class WorkCheckpointTerminalAction(StrEnum):
    CANCEL = "CANCEL"
    EXPIRE = "EXPIRE"
    SUPERSEDE = "SUPERSEDE"


@dataclass(frozen=True, slots=True)
class FreezeWorkCheckpointCommand:
    checkpoint: MainCoreWorkCheckpoint
    idempotency_key: str
    now: datetime

    def __post_init__(self) -> None:
        _token(self.idempotency_key, "idempotency_key")
        object.__setattr__(self, "now", _aware(self.now, "now"))


@dataclass(frozen=True, slots=True)
class ApplyWorkCallbackCommand:
    callback: WorkCallbackEnvelope
    runtime: WorkRecoveryRuntime

    def __post_init__(self) -> None:
        if len(self.runtime.controlled_resource_refs) > MAX_RUNTIME_CONTROLLED_REFS:
            raise ValueError("runtime controlled refs exceed their hard limit")


@dataclass(frozen=True, slots=True)
class ClaimWorkCheckpointLeaseCommand:
    scope: WorkScope
    work_ref: str
    expected_version: int
    lease: WorkCallbackLease
    now: datetime
    idempotency_key: str

    def __post_init__(self) -> None:
        _token(self.work_ref, "work_ref")
        _token(self.idempotency_key, "idempotency_key")
        object.__setattr__(self, "now", _aware(self.now, "now"))
        if self.expected_version <= 0:
            raise ValueError("expected_version must be positive")


@dataclass(frozen=True, slots=True)
class RenewWorkCheckpointLeaseCommand:
    scope: WorkScope
    work_ref: str
    expected_version: int
    expected_lease_owner: str
    expected_lease_token: int
    lease: WorkCallbackLease
    now: datetime
    idempotency_key: str

    def __post_init__(self) -> None:
        _token(self.work_ref, "work_ref")
        _token(self.expected_lease_owner, "expected_lease_owner")
        _token(self.idempotency_key, "idempotency_key")
        object.__setattr__(self, "now", _aware(self.now, "now"))
        if self.expected_version <= 0 or self.expected_lease_token <= 0:
            raise ValueError("lease fences must be positive")


@dataclass(frozen=True, slots=True)
class ReleaseWorkCheckpointLeaseCommand:
    scope: WorkScope
    work_ref: str
    expected_version: int
    expected_lease_owner: str
    expected_lease_token: int
    now: datetime
    idempotency_key: str

    def __post_init__(self) -> None:
        _token(self.work_ref, "work_ref")
        _token(self.expected_lease_owner, "expected_lease_owner")
        _token(self.idempotency_key, "idempotency_key")
        object.__setattr__(self, "now", _aware(self.now, "now"))
        if self.expected_version <= 0 or self.expected_lease_token <= 0:
            raise ValueError("lease fences must be positive")


@dataclass(frozen=True, slots=True)
class TransitionWorkCheckpointCommand:
    scope: WorkScope
    work_ref: str
    action: WorkCheckpointTerminalAction
    expected_version: int
    reason: str
    now: datetime
    idempotency_key: str

    def __post_init__(self) -> None:
        _token(self.work_ref, "work_ref")
        _token(self.idempotency_key, "idempotency_key")
        object.__setattr__(self, "now", _aware(self.now, "now"))
        if not isinstance(self.action, WorkCheckpointTerminalAction):
            raise ValueError("terminal action is invalid")
        if self.expected_version <= 0:
            raise ValueError("expected_version must be positive")


@dataclass(frozen=True, slots=True)
class WorkCheckpointMutationResult:
    checkpoint: MainCoreWorkCheckpoint
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class WorkCallbackStorageResult:
    decision: WorkRecoveryDecision
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class RecoveryReadyRecord:
    checkpoint: MainCoreWorkCheckpoint
    envelope: MainCoreWorkRecoveryEnvelope

    def __post_init__(self) -> None:
        if self.checkpoint.scope != self.envelope.scope:
            raise ValueError("ready record scope mismatch")
        if self.checkpoint.work_ref != self.envelope.work_ref:
            raise ValueError("ready record work_ref mismatch")
        if self.checkpoint.checkpoint_version != self.envelope.checkpoint_version:
            raise ValueError("ready record checkpoint version mismatch")


@dataclass(frozen=True, slots=True)
class WorkCheckpointEvent:
    scope: WorkScope
    work_ref: str
    event_sequence: int
    kind: WorkCheckpointEventKind
    checkpoint_version: int
    run_generation: int
    callback_sequence: int
    checkpoint_status: str
    idempotency_key: str
    request_fingerprint: str
    decision_code: str
    created_at: datetime

    def __post_init__(self) -> None:
        _token(self.work_ref, "work_ref")
        _token(self.idempotency_key, "idempotency_key")
        object.__setattr__(self, "created_at", _aware(self.created_at, "created_at"))
        if not isinstance(self.kind, WorkCheckpointEventKind):
            raise ValueError("event kind is invalid")
        if (
            min(
                self.event_sequence,
                self.checkpoint_version,
                self.run_generation,
            )
            <= 0
            or self.callback_sequence < 0
        ):
            raise ValueError("event fences are invalid")
        if len(self.request_fingerprint) != 64:
            raise ValueError("event fingerprint is invalid")
        if len(self.decision_code) > 64:
            raise ValueError("event decision code is too long")


__all__ = [
    "ApplyWorkCallbackCommand",
    "ClaimWorkCheckpointLeaseCommand",
    "FreezeWorkCheckpointCommand",
    "RecoveryReadyRecord",
    "ReleaseWorkCheckpointLeaseCommand",
    "RenewWorkCheckpointLeaseCommand",
    "TransitionWorkCheckpointCommand",
    "WorkCallbackStorageResult",
    "WorkCheckpointEvent",
    "WorkCheckpointEventKind",
    "WorkCheckpointMutationResult",
    "WorkCheckpointTerminalAction",
]
