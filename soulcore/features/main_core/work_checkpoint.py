"""Pure domain fences for durable Main Core work callbacks.

This module does not persist, dispatch, render, or execute anything.  It only
decides whether a callback may create an internal recovery envelope for a new
Main Core run.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum

from .work_continuity import MainCoreWorkSnapshot, valid_work_resource_ref

MAX_CALLBACK_RESULTS = 24
MAX_CHECKPOINT_RESULTS = 60
MAX_RESULT_KIND_CHARS = 64
MAX_CALLBACK_SUMMARY_CHARS = 400
MAX_TERMINAL_REASON_CHARS = 400
MAX_IDEMPOTENCY_KEY_CHARS = 160
MAX_LEASE_OWNER_CHARS = 120
MAX_CHECKPOINT_LIFETIME_SECONDS = 7 * 24 * 60 * 60

_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,159}$")


class WorkCheckpointError(ValueError):
    """A checkpoint value or transition violated the recovery contract."""


class WorkCheckpointStatus(StrEnum):
    WAITING = "WAITING"
    RECOVERY_READY = "RECOVERY_READY"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    SUPERSEDED = "SUPERSEDED"


class WorkCallbackOutcome(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class WorkRecoveryAction(StrEnum):
    REASSESS_PLAN = "REASSESS_PLAN"
    UPDATE_WORK = "UPDATE_WORK"
    COMPLETE_WORK = "COMPLETE_WORK"
    CANCEL_WORK = "CANCEL_WORK"


class WorkReevaluationFlag(StrEnum):
    CALLBACK_RESULT = "CALLBACK_RESULT"
    PROCESS_RESTARTED = "PROCESS_RESTARTED"
    ROLE_STATE_CHANGED = "ROLE_STATE_CHANGED"
    PERMISSIONS_CHANGED = "PERMISSIONS_CHANGED"
    BUDGET_CHANGED = "BUDGET_CHANGED"


class WorkCallbackRejection(StrEnum):
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    WORK_REF_MISMATCH = "WORK_REF_MISMATCH"
    DUPLICATE_CALLBACK = "DUPLICATE_CALLBACK"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    CHECKPOINT_TERMINAL = "CHECKPOINT_TERMINAL"
    CHECKPOINT_EXPIRED = "CHECKPOINT_EXPIRED"
    CHECKPOINT_VERSION_MISMATCH = "CHECKPOINT_VERSION_MISMATCH"
    RUN_GENERATION_MISMATCH = "RUN_GENERATION_MISMATCH"
    CALLBACK_SEQUENCE_OLD = "CALLBACK_SEQUENCE_OLD"
    CALLBACK_SEQUENCE_GAP = "CALLBACK_SEQUENCE_GAP"
    LEASE_MISMATCH = "LEASE_MISMATCH"
    LEASE_EXPIRED = "LEASE_EXPIRED"
    SUPERSEDED_BY_PLAYER_ACTIVITY = "SUPERSEDED_BY_PLAYER_ACTIVITY"
    UNKNOWN_RESULT_SLOT = "UNKNOWN_RESULT_SLOT"
    UNCONTROLLED_RESULT_REF = "UNCONTROLLED_RESULT_REF"
    DUPLICATE_RESULT_REF = "DUPLICATE_RESULT_REF"
    RESULT_LIMIT_EXCEEDED = "RESULT_LIMIT_EXCEEDED"


@dataclass(frozen=True, slots=True)
class WorkScope:
    profile_id: str
    instance_id: str

    def __post_init__(self) -> None:
        _required_token(self.profile_id, "profile_id")
        _required_token(self.instance_id, "instance_id")


@dataclass(frozen=True, slots=True)
class WorkRecoveryBaseline:
    activity_generation: int
    role_state_revision: int
    permission_revision: int
    budget_revision: int

    def __post_init__(self) -> None:
        for label, value in (
            ("activity_generation", self.activity_generation),
            ("role_state_revision", self.role_state_revision),
            ("permission_revision", self.permission_revision),
            ("budget_revision", self.budget_revision),
        ):
            if int(value) < 0:
                raise WorkCheckpointError(f"{label} must be non-negative")


@dataclass(frozen=True, slots=True)
class WorkCallbackLease:
    owner: str
    token: int
    expires_at: datetime

    def __post_init__(self) -> None:
        _bounded_required(self.owner, "lease owner", MAX_LEASE_OWNER_CHARS)
        if int(self.token) <= 0:
            raise WorkCheckpointError("lease token must be positive")
        _aware_utc(self.expires_at, "lease expires_at")


@dataclass(frozen=True, slots=True)
class ControlledWorkResult:
    slot_id: str
    resource_ref: str
    result_kind: str
    source_callback_sequence: int

    def __post_init__(self) -> None:
        _required_token(self.slot_id, "result slot_id")
        if not valid_work_resource_ref(self.resource_ref):
            raise WorkCheckpointError("controlled result resource_ref is invalid")
        _bounded_required(self.result_kind, "result kind", MAX_RESULT_KIND_CHARS)
        if int(self.source_callback_sequence) < 0:
            raise WorkCheckpointError("source callback sequence must be non-negative")


@dataclass(frozen=True, slots=True)
class WorkCallbackEnvelope:
    scope: WorkScope
    work_ref: str
    checkpoint_version: int
    run_generation: int
    callback_sequence: int
    lease_owner: str
    lease_token: int
    idempotency_key: str
    outcome: WorkCallbackOutcome
    result_summary: str = ""
    results: tuple[ControlledWorkResult, ...] = ()

    def __post_init__(self) -> None:
        _required_token(self.work_ref, "work_ref")
        if int(self.checkpoint_version) <= 0:
            raise WorkCheckpointError("checkpoint_version must be positive")
        if int(self.run_generation) <= 0:
            raise WorkCheckpointError("run_generation must be positive")
        if int(self.callback_sequence) <= 0:
            raise WorkCheckpointError("callback_sequence must be positive")
        _bounded_required(self.lease_owner, "lease owner", MAX_LEASE_OWNER_CHARS)
        if int(self.lease_token) <= 0:
            raise WorkCheckpointError("lease token must be positive")
        if not isinstance(self.outcome, WorkCallbackOutcome):
            raise WorkCheckpointError("callback outcome is invalid")
        _bounded_required(
            self.idempotency_key,
            "idempotency key",
            MAX_IDEMPOTENCY_KEY_CHARS,
        )
        _bounded_optional(
            self.result_summary,
            "callback result summary",
            MAX_CALLBACK_SUMMARY_CHARS,
        )
        if len(self.results) > MAX_CALLBACK_RESULTS:
            raise WorkCheckpointError("callback contains too many controlled results")
        if any(item.source_callback_sequence != self.callback_sequence for item in self.results):
            raise WorkCheckpointError("callback results must carry their callback sequence")


@dataclass(frozen=True, slots=True)
class WorkRecoveryRuntime:
    scope: WorkScope
    now: datetime
    baseline: WorkRecoveryBaseline
    controlled_resource_refs: frozenset[str]
    process_restarted: bool = False

    def __post_init__(self) -> None:
        _aware_utc(self.now, "recovery now")
        if any(not valid_work_resource_ref(item) for item in self.controlled_resource_refs):
            raise WorkCheckpointError("runtime controlled resource ref is invalid")


@dataclass(frozen=True, slots=True)
class MainCoreWorkCheckpoint:
    scope: WorkScope
    snapshot: MainCoreWorkSnapshot
    checkpoint_version: int
    run_generation: int
    callback_sequence: int
    baseline: WorkRecoveryBaseline
    allowed_actions: tuple[WorkRecoveryAction, ...]
    controlled_results: tuple[ControlledWorkResult, ...]
    lease: WorkCallbackLease | None
    created_at: datetime
    expires_at: datetime
    status: WorkCheckpointStatus = WorkCheckpointStatus.WAITING
    terminal_reason: str = ""
    last_idempotency_key: str = ""
    last_callback_fingerprint: str = ""

    @property
    def work_ref(self) -> str:
        return self.snapshot.work_ref

    def __post_init__(self) -> None:
        _validate_checkpoint_identity(self)
        _validate_checkpoint_actions(self.allowed_actions)
        _validate_checkpoint_results(self.controlled_results)
        _validate_checkpoint_lifetime(self)
        _validate_checkpoint_status(self)


@dataclass(frozen=True, slots=True)
class WorkCallbackDecision:
    accepted: bool
    checkpoint: MainCoreWorkCheckpoint
    rejection: WorkCallbackRejection | None = None
    triggering_results: tuple[ControlledWorkResult, ...] = ()
    reevaluation_flags: tuple[WorkReevaluationFlag, ...] = ()


def decide_callback(
    checkpoint: MainCoreWorkCheckpoint,
    callback: WorkCallbackEnvelope,
    runtime: WorkRecoveryRuntime,
) -> WorkCallbackDecision:
    """Accept exactly one fenced callback or return a deterministic rejection."""

    fingerprint = callback_fingerprint(callback)
    rejection = _identity_rejection(checkpoint, callback, runtime, fingerprint)
    if rejection is not None:
        return _rejected(checkpoint, rejection)
    now = _aware_utc(runtime.now, "recovery now")
    if now >= _aware_utc(checkpoint.expires_at, "checkpoint expires_at"):
        expired = _terminal(
            checkpoint,
            status=WorkCheckpointStatus.EXPIRED,
            reason="checkpoint lifetime elapsed",
        )
        return _rejected(expired, WorkCallbackRejection.CHECKPOINT_EXPIRED)
    assert checkpoint.lease is not None
    if (
        callback.lease_owner != checkpoint.lease.owner
        or callback.lease_token != checkpoint.lease.token
    ):
        return _rejected(checkpoint, WorkCallbackRejection.LEASE_MISMATCH)
    if now >= _aware_utc(checkpoint.lease.expires_at, "lease expires_at"):
        return _rejected(checkpoint, WorkCallbackRejection.LEASE_EXPIRED)
    if runtime.baseline.activity_generation != checkpoint.baseline.activity_generation:
        superseded = _terminal(
            checkpoint,
            status=WorkCheckpointStatus.SUPERSEDED,
            reason="new player activity superseded the checkpoint",
        )
        return _rejected(
            superseded,
            WorkCallbackRejection.SUPERSEDED_BY_PLAYER_ACTIVITY,
        )
    result_rejection = _result_rejection(checkpoint, callback, runtime)
    if result_rejection is not None:
        return _rejected(checkpoint, result_rejection)
    flags = _reevaluation_flags(checkpoint, runtime)
    accepted = replace(
        checkpoint,
        checkpoint_version=checkpoint.checkpoint_version + 1,
        run_generation=checkpoint.run_generation + 1,
        callback_sequence=callback.callback_sequence,
        controlled_results=(*checkpoint.controlled_results, *callback.results),
        lease=None,
        status=WorkCheckpointStatus.RECOVERY_READY,
        last_idempotency_key=callback.idempotency_key,
        last_callback_fingerprint=fingerprint,
    )
    return WorkCallbackDecision(
        accepted=True,
        checkpoint=accepted,
        triggering_results=callback.results,
        reevaluation_flags=flags,
    )


def renew_callback_lease(
    checkpoint: MainCoreWorkCheckpoint,
    *,
    expected_version: int,
    lease: WorkCallbackLease,
    now: datetime,
) -> MainCoreWorkCheckpoint:
    """Replace a waiting lease with a strictly newer fencing token."""

    _require_waiting_version(checkpoint, expected_version)
    current = checkpoint.lease
    assert current is not None
    current_time = _aware_utc(now, "lease renewal now")
    if current_time >= _aware_utc(checkpoint.expires_at, "checkpoint expires_at"):
        raise WorkCheckpointError("cannot renew an expired checkpoint")
    if lease.token <= current.token:
        raise WorkCheckpointError("renewed lease token must increase")
    if lease.expires_at <= current_time or lease.expires_at > checkpoint.expires_at:
        raise WorkCheckpointError("renewed lease expiry is outside checkpoint lifetime")
    return replace(
        checkpoint,
        checkpoint_version=checkpoint.checkpoint_version + 1,
        lease=lease,
    )


def release_callback_lease(
    checkpoint: MainCoreWorkCheckpoint,
    *,
    expected_version: int,
    expected_owner: str,
    expected_token: int,
    now: datetime,
) -> MainCoreWorkCheckpoint:
    """Fence off a waiting lease while preserving its monotonic token floor."""

    _require_waiting_version(checkpoint, expected_version)
    current = checkpoint.lease
    assert current is not None
    if current.owner != expected_owner or current.token != int(expected_token):
        raise WorkCheckpointError("lease owner or token mismatch")
    released_at = _aware_utc(now, "lease release now")
    created_at = _aware_utc(checkpoint.created_at, "checkpoint created_at")
    if released_at <= created_at or released_at >= checkpoint.expires_at:
        raise WorkCheckpointError("lease release is outside checkpoint lifetime")
    if released_at >= current.expires_at:
        raise WorkCheckpointError("lease is already expired")
    return replace(
        checkpoint,
        checkpoint_version=checkpoint.checkpoint_version + 1,
        lease=replace(current, expires_at=released_at),
    )


def cancel_checkpoint(
    checkpoint: MainCoreWorkCheckpoint, *, expected_version: int, reason: str
) -> MainCoreWorkCheckpoint:
    _require_waiting_version(checkpoint, expected_version)
    return _terminal(
        checkpoint,
        status=WorkCheckpointStatus.CANCELLED,
        reason=reason,
    )


def expire_checkpoint(
    checkpoint: MainCoreWorkCheckpoint,
    *,
    expected_version: int,
    now: datetime,
    reason: str = "checkpoint lifetime elapsed",
) -> MainCoreWorkCheckpoint:
    _require_waiting_version(checkpoint, expected_version)
    if _aware_utc(now, "expiry now") < _aware_utc(checkpoint.expires_at, "checkpoint expires_at"):
        raise WorkCheckpointError("checkpoint cannot expire before expires_at")
    return _terminal(checkpoint, status=WorkCheckpointStatus.EXPIRED, reason=reason)


def supersede_checkpoint(
    checkpoint: MainCoreWorkCheckpoint, *, expected_version: int, reason: str
) -> MainCoreWorkCheckpoint:
    _require_waiting_version(checkpoint, expected_version)
    return _terminal(
        checkpoint,
        status=WorkCheckpointStatus.SUPERSEDED,
        reason=reason,
    )


def callback_fingerprint(callback: WorkCallbackEnvelope) -> str:
    encoded = json.dumps(
        {
            "scope": [callback.scope.profile_id, callback.scope.instance_id],
            "work_ref": callback.work_ref,
            "checkpoint_version": callback.checkpoint_version,
            "run_generation": callback.run_generation,
            "callback_sequence": callback.callback_sequence,
            "lease_owner": callback.lease_owner,
            "lease_token": callback.lease_token,
            "outcome": callback.outcome,
            "result_summary": callback.result_summary,
            "results": [
                [item.slot_id, item.resource_ref, item.result_kind] for item in callback.results
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _identity_rejection(
    checkpoint: MainCoreWorkCheckpoint,
    callback: WorkCallbackEnvelope,
    runtime: WorkRecoveryRuntime,
    fingerprint: str,
) -> WorkCallbackRejection | None:
    if callback.scope != checkpoint.scope or runtime.scope != checkpoint.scope:
        return WorkCallbackRejection.SCOPE_MISMATCH
    if callback.work_ref != checkpoint.work_ref:
        return WorkCallbackRejection.WORK_REF_MISMATCH
    if callback.idempotency_key == checkpoint.last_idempotency_key:
        if fingerprint == checkpoint.last_callback_fingerprint:
            return WorkCallbackRejection.DUPLICATE_CALLBACK
        return WorkCallbackRejection.IDEMPOTENCY_CONFLICT
    if checkpoint.status != WorkCheckpointStatus.WAITING:
        return WorkCallbackRejection.CHECKPOINT_TERMINAL
    if callback.checkpoint_version != checkpoint.checkpoint_version:
        return WorkCallbackRejection.CHECKPOINT_VERSION_MISMATCH
    if callback.run_generation != checkpoint.run_generation:
        return WorkCallbackRejection.RUN_GENERATION_MISMATCH
    if callback.callback_sequence <= checkpoint.callback_sequence:
        return WorkCallbackRejection.CALLBACK_SEQUENCE_OLD
    if callback.callback_sequence != checkpoint.callback_sequence + 1:
        return WorkCallbackRejection.CALLBACK_SEQUENCE_GAP
    return None


def _result_rejection(
    checkpoint: MainCoreWorkCheckpoint,
    callback: WorkCallbackEnvelope,
    runtime: WorkRecoveryRuntime,
) -> WorkCallbackRejection | None:
    slot_ids = {item.slot_id for item in checkpoint.snapshot.result_slots}
    if any(item.slot_id not in slot_ids for item in callback.results):
        return WorkCallbackRejection.UNKNOWN_RESULT_SLOT
    if any(item.resource_ref not in runtime.controlled_resource_refs for item in callback.results):
        return WorkCallbackRejection.UNCONTROLLED_RESULT_REF
    identities = {(item.slot_id, item.resource_ref) for item in callback.results}
    if len(identities) != len(callback.results):
        return WorkCallbackRejection.DUPLICATE_RESULT_REF
    existing = {(item.slot_id, item.resource_ref) for item in checkpoint.controlled_results}
    if identities & existing:
        return WorkCallbackRejection.DUPLICATE_RESULT_REF
    if len(checkpoint.controlled_results) + len(callback.results) > MAX_CHECKPOINT_RESULTS:
        return WorkCallbackRejection.RESULT_LIMIT_EXCEEDED
    return None


def _reevaluation_flags(
    checkpoint: MainCoreWorkCheckpoint,
    runtime: WorkRecoveryRuntime,
) -> tuple[WorkReevaluationFlag, ...]:
    flags = [WorkReevaluationFlag.CALLBACK_RESULT]
    if runtime.process_restarted:
        flags.append(WorkReevaluationFlag.PROCESS_RESTARTED)
    if runtime.baseline.role_state_revision != checkpoint.baseline.role_state_revision:
        flags.append(WorkReevaluationFlag.ROLE_STATE_CHANGED)
    if runtime.baseline.permission_revision != checkpoint.baseline.permission_revision:
        flags.append(WorkReevaluationFlag.PERMISSIONS_CHANGED)
    if runtime.baseline.budget_revision != checkpoint.baseline.budget_revision:
        flags.append(WorkReevaluationFlag.BUDGET_CHANGED)
    return tuple(flags)


def _terminal(
    checkpoint: MainCoreWorkCheckpoint,
    *,
    status: WorkCheckpointStatus,
    reason: str,
) -> MainCoreWorkCheckpoint:
    value = _bounded_required(reason, "checkpoint terminal reason", MAX_TERMINAL_REASON_CHARS)
    return replace(
        checkpoint,
        checkpoint_version=checkpoint.checkpoint_version + 1,
        lease=None,
        status=status,
        terminal_reason=value,
    )


def _require_waiting_version(checkpoint: MainCoreWorkCheckpoint, expected_version: int) -> None:
    if checkpoint.status != WorkCheckpointStatus.WAITING:
        raise WorkCheckpointError("only a waiting checkpoint may change")
    if int(expected_version) != checkpoint.checkpoint_version:
        raise WorkCheckpointError("stale checkpoint version")


def _rejected(
    checkpoint: MainCoreWorkCheckpoint, rejection: WorkCallbackRejection
) -> WorkCallbackDecision:
    return WorkCallbackDecision(
        accepted=False,
        checkpoint=checkpoint,
        rejection=rejection,
    )


def _validate_result_uniqueness(values: tuple[ControlledWorkResult, ...]) -> None:
    identities = {(item.slot_id, item.resource_ref) for item in values}
    if len(identities) != len(values):
        raise WorkCheckpointError("controlled result bindings must be unique")


def _validate_checkpoint_identity(checkpoint: MainCoreWorkCheckpoint) -> None:
    if checkpoint.snapshot.status != "ACTIVE":
        raise WorkCheckpointError("only active Main Core work may be checkpointed")
    if not isinstance(checkpoint.status, WorkCheckpointStatus):
        raise WorkCheckpointError("checkpoint status is invalid")
    _required_token(checkpoint.work_ref, "work_ref")
    if int(checkpoint.checkpoint_version) <= 0:
        raise WorkCheckpointError("checkpoint_version must be positive")
    if int(checkpoint.run_generation) <= 0:
        raise WorkCheckpointError("run_generation must be positive")
    if int(checkpoint.callback_sequence) < 0:
        raise WorkCheckpointError("callback_sequence must be non-negative")


def _validate_checkpoint_actions(actions: tuple[WorkRecoveryAction, ...]) -> None:
    if any(not isinstance(item, WorkRecoveryAction) for item in actions):
        raise WorkCheckpointError("recovery actions contain an unknown value")
    if not actions or WorkRecoveryAction.REASSESS_PLAN not in actions:
        raise WorkCheckpointError("recovery actions must include REASSESS_PLAN")
    if len(set(actions)) != len(actions):
        raise WorkCheckpointError("recovery actions must be unique")


def _validate_checkpoint_results(results: tuple[ControlledWorkResult, ...]) -> None:
    if len(results) > MAX_CHECKPOINT_RESULTS:
        raise WorkCheckpointError("checkpoint contains too many controlled results")
    _validate_result_uniqueness(results)


def _validate_checkpoint_lifetime(checkpoint: MainCoreWorkCheckpoint) -> None:
    created = _aware_utc(checkpoint.created_at, "checkpoint created_at")
    expires = _aware_utc(checkpoint.expires_at, "checkpoint expires_at")
    lifetime = (expires - created).total_seconds()
    if lifetime <= 0 or lifetime > MAX_CHECKPOINT_LIFETIME_SECONDS:
        raise WorkCheckpointError("checkpoint lifetime is outside its hard bounds")
    if checkpoint.status != WorkCheckpointStatus.WAITING:
        return
    if checkpoint.lease is None:
        raise WorkCheckpointError("a waiting checkpoint requires a lease")
    lease_expiry = _aware_utc(checkpoint.lease.expires_at, "lease expires_at")
    if lease_expiry <= created or lease_expiry > expires:
        raise WorkCheckpointError("lease expiry must be inside checkpoint lifetime")


def _validate_checkpoint_status(checkpoint: MainCoreWorkCheckpoint) -> None:
    if checkpoint.status != WorkCheckpointStatus.WAITING and checkpoint.lease is not None:
        raise WorkCheckpointError("a terminal or recovery-ready checkpoint cannot hold a lease")
    _bounded_optional(
        checkpoint.terminal_reason,
        "checkpoint terminal reason",
        MAX_TERMINAL_REASON_CHARS,
    )
    terminal = {
        WorkCheckpointStatus.CANCELLED,
        WorkCheckpointStatus.EXPIRED,
        WorkCheckpointStatus.SUPERSEDED,
    }
    if checkpoint.status in terminal and not checkpoint.terminal_reason:
        raise WorkCheckpointError("terminal checkpoint requires a reason")


def _required_token(value: str, label: str) -> str:
    text = str(value or "").strip()
    if not _TOKEN_PATTERN.fullmatch(text):
        raise WorkCheckpointError(f"{label} is not a valid bounded token")
    return text


def _bounded_required(value: str, label: str, maximum: int) -> str:
    text = str(value or "").strip()
    if not text or len(text) > maximum:
        raise WorkCheckpointError(f"{label} is empty or exceeds its hard limit")
    return text


def _bounded_optional(value: str, label: str, maximum: int) -> str:
    text = str(value or "").strip()
    if len(text) > maximum:
        raise WorkCheckpointError(f"{label} exceeds its hard limit")
    return text


def _aware_utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise WorkCheckpointError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


__all__ = [
    "ControlledWorkResult",
    "MainCoreWorkCheckpoint",
    "WorkCallbackDecision",
    "WorkCallbackEnvelope",
    "WorkCallbackLease",
    "WorkCallbackOutcome",
    "WorkCallbackRejection",
    "WorkCheckpointError",
    "WorkCheckpointStatus",
    "WorkRecoveryAction",
    "WorkRecoveryBaseline",
    "WorkRecoveryRuntime",
    "WorkReevaluationFlag",
    "WorkScope",
    "callback_fingerprint",
    "cancel_checkpoint",
    "decide_callback",
    "expire_checkpoint",
    "release_callback_lease",
    "renew_callback_lease",
    "supersede_checkpoint",
]
