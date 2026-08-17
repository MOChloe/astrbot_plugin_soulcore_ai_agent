"""Deterministic player-message gating driven by explicit structured state.

This module deliberately knows nothing about ``current_state`` or any other
natural-language narrative. Administrators independently decide whether an
explicit snapshot is allowed to affect player messages. Durable storage is
expressed as a Protocol so the message entry point and repository can integrate
it without coupling policy to SQLite details.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from .ports import StateGateRepositoryPort
from .state_gate_datetimes import as_utc, parse_datetime
from .state_gate_datetimes import (
    as_utc as _as_utc,
)
from .state_gate_datetimes import (
    parse_datetime as _parse_datetime,
)
from .state_gate_datetimes import (
    required_datetime as _required_datetime,
)
from .temporary_absence import (
    TEMPORARY_ABSENCE_REASON_CODE,
    TemporaryAbsenceExpiryWake,
    temporary_absence_metadata,
)


class TemporaryAbsenceExpiryGateMixin:
    """Service boundary for claiming and settling one natural-expiry wake."""

    async def ensure_expired_temporary_absence_wakeups(
        self,
        *,
        now: datetime,
        limit: int = 100,
    ) -> int:
        return int(
            await self.repository.ensure_expired_temporary_absence_wakeups(
                now=as_utc(now),
                limit=max(1, int(limit)),
            )
        )

    async def prepare_temporary_absence_expiry(
        self,
        wakeup: Any,
        marker: TemporaryAbsenceExpiryWake,
        *,
        now: datetime,
    ) -> dict[str, Any]:
        raw = await self.repository.prepare_temporary_absence_expiry(
            str(wakeup.profile_id),
            str(wakeup.instance_id),
            wakeup_id=int(wakeup.wakeup_id),
            expected_wakeup_generation=int(wakeup.generation),
            wakeup_lease_token=int(wakeup.lease_token),
            expected_wakeup_version=int(wakeup.version),
            expected_wakeup_idempotency_key=marker.idempotency_key,
            gate_generation=marker.gate_generation,
            expected_activity_epoch=marker.activity_epoch,
            now=as_utc(now),
        )
        result = dict(raw or {})
        if result.get("retry_at"):
            retry_at = parse_datetime(result["retry_at"])
            if retry_at is not None:
                result["retry_at"] = retry_at
        return result

    async def finalize_temporary_absence_expiry(
        self,
        profile_id: str,
        instance_id: str,
        *,
        gate_generation: int,
        now: datetime,
    ) -> bool:
        return bool(
            await self.repository.finalize_temporary_absence_expiry(
                profile_id,
                instance_id,
                gate_generation=int(gate_generation),
                now=as_utc(now),
            )
        )


MAX_NON_OPEN_DURATION = timedelta(hours=24)


class StateGateMode(StrEnum):
    OPEN = "OPEN"
    DECLINE = "DECLINE"
    SILENT = "SILENT"
    DEFER = "DEFER"


class StateGateDisposition(StrEnum):
    PROCEED = "PROCEED"
    RESTRICTED_DECLINE = "RESTRICTED_DECLINE"
    SILENT = "SILENT"
    DEFER = "DEFER"


class DeferredGateStatus(StrEnum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    RESOLVED = "RESOLVED"
    CANCELLED = "CANCELLED"


class DeferredGateLeaseDisposition(StrEnum):
    CURRENT = "CURRENT"
    COMMITTED = "COMMITTED"
    LOST = "LOST"


@dataclass(frozen=True, slots=True)
class DeferredGateLeaseProbe:
    disposition: DeferredGateLeaseDisposition
    resolution_run_id: int | None = None


@dataclass(frozen=True, slots=True)
class StateGatePolicy:
    """Administrator-owned policy; disabled is the product default."""

    enabled: bool = False
    silent_enabled: bool = False
    max_non_open_duration: timedelta = MAX_NON_OPEN_DURATION

    def __post_init__(self) -> None:
        if self.max_non_open_duration <= timedelta(0):
            raise ValueError("state gate duration must be positive")
        if self.max_non_open_duration > MAX_NON_OPEN_DURATION:
            raise ValueError("state gate duration cannot exceed 24 hours")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> StateGatePolicy:
        row = value or {}
        hours = float(row.get("max_non_open_hours", 24) or 24)
        return cls(
            enabled=bool(row.get("state_message_gate_enabled", False)),
            silent_enabled=bool(row.get("state_message_silent_enabled", False)),
            max_non_open_duration=timedelta(hours=hours),
        )


@dataclass(frozen=True, slots=True)
class StateGateSnapshot:
    """Explicit structured state; never inferred from prose."""

    mode: StateGateMode = StateGateMode.OPEN
    generation: int = 0
    effective_at: datetime | None = None
    expires_at: datetime | None = None
    reason_code: str = ""
    expression_context: str = ""
    source_run_id: int | None = None
    version: int = 0

    def __post_init__(self) -> None:
        if self.generation < 0 or self.version < 0:
            raise ValueError("state gate generation and version cannot be negative")
        if self.mode is StateGateMode.OPEN:
            return
        if self.effective_at is None or self.expires_at is None:
            raise ValueError("non-open state gate requires effective_at and expires_at")
        effective = _as_utc(self.effective_at)
        expires = _as_utc(self.expires_at)
        if expires <= effective:
            raise ValueError("state gate expiry must be after its effective time")
        if expires - effective > MAX_NON_OPEN_DURATION:
            raise ValueError("non-open state gate cannot exceed 24 hours")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> StateGateSnapshot:
        row = value or {}
        return cls(
            mode=StateGateMode(str(row.get("mode", "OPEN")).strip().upper()),
            generation=max(0, int(row.get("generation", 0) or 0)),
            effective_at=_parse_datetime(row.get("effective_at")),
            expires_at=_parse_datetime(row.get("expires_at")),
            reason_code=str(row.get("reason_code", "") or "").strip()[:80],
            expression_context=str(row.get("expression_context", "") or "").strip()[:1000],
            source_run_id=(
                int(row["source_run_id"]) if row.get("source_run_id") is not None else None
            ),
            version=max(0, int(row.get("version", 0) or 0)),
        )

    def is_active(self, now: datetime) -> bool:
        if self.mode is StateGateMode.OPEN:
            return False
        current = _as_utc(now)
        assert self.effective_at is not None and self.expires_at is not None
        return _as_utc(self.effective_at) <= current < _as_utc(self.expires_at)


@dataclass(frozen=True, slots=True)
class StateGateBypass:
    """Only trusted, already-classified callers may set these flags."""

    administrator_command: bool = False

    @property
    def reason(self) -> str:
        if self.administrator_command:
            return "administrator_command"
        return ""


@dataclass(frozen=True, slots=True)
class StateGateDecision:
    disposition: StateGateDisposition
    reason: str
    snapshot_generation: int = 0
    due_at: datetime | None = None
    expression_context: str = ""
    reason_code: str = ""
    effective_at: datetime | None = None
    expires_at: datetime | None = None

    @property
    def invoke_main_core(self) -> bool:
        return self.disposition in {
            StateGateDisposition.PROCEED,
            StateGateDisposition.RESTRICTED_DECLINE,
        }

    @property
    def restricted_main_core(self) -> bool:
        return self.disposition is StateGateDisposition.RESTRICTED_DECLINE

    @property
    def allow_commands(self) -> bool:
        return self.disposition is StateGateDisposition.PROCEED

    @property
    def allow_schedule_mutations(self) -> bool:
        return self.disposition is StateGateDisposition.PROCEED

    @property
    def allow_intent_mutations(self) -> bool:
        return self.disposition is StateGateDisposition.PROCEED

    @property
    def knowledge_eligible_now(self) -> bool:
        return self.disposition is not StateGateDisposition.DEFER

    @property
    def should_enqueue_deferred(self) -> bool:
        return self.disposition is StateGateDisposition.DEFER

    def ended_temporary_absence_metadata(
        self,
        *,
        ended_at: datetime,
        end_reason: str = "NATURAL_EXPIRY",
    ) -> dict[str, Any]:
        if self.reason_code != TEMPORARY_ABSENCE_REASON_CODE:
            return {}
        normalized_end_reason = str(end_reason or "NATURAL_EXPIRY").strip().upper()
        actual_end = (
            self.expires_at
            if normalized_end_reason == "NATURAL_EXPIRY" and self.expires_at is not None
            else ended_at
        )
        return temporary_absence_metadata(
            reason=self.expression_context,
            started_at=self.effective_at,
            planned_until=self.expires_at,
            ended_at=_as_utc(actual_end),
            end_reason=normalized_end_reason,
        )


@dataclass(frozen=True, slots=True)
class DeferredGateMessage:
    message_ref: str
    ledger_entry_id: int
    activity_epoch: int
    received_at: datetime
    idempotency_key: str = ""

    def __post_init__(self) -> None:
        if not self.message_ref.strip():
            raise ValueError("deferred message requires a stable message_ref")
        if self.ledger_entry_id < 1:
            raise ValueError("deferred message requires a persisted ledger entry")
        if self.activity_epoch < 0:
            raise ValueError("deferred message activity epoch cannot be negative")

    @property
    def stable_key(self) -> str:
        return self.idempotency_key.strip() or self.message_ref.strip()


@dataclass(frozen=True, slots=True)
class DeferredGateBatch:
    batch_ref: str
    profile_id: str
    instance_id: str
    gate_generation: int
    activity_epoch: int
    due_at: datetime
    messages: tuple[DeferredGateMessage, ...]
    status: DeferredGateStatus = DeferredGateStatus.PENDING
    version: int = 0
    lease_token: int = 0

    def __post_init__(self) -> None:
        if not self.batch_ref or not self.profile_id or not self.instance_id:
            raise ValueError("deferred gate batch identity is incomplete")
        if self.gate_generation < 0 or self.activity_epoch < 0:
            raise ValueError("deferred gate batch generations cannot be negative")
        if self.version < 0 or self.lease_token < 0:
            raise ValueError("deferred gate batch fences cannot be negative")
        if not self.messages:
            raise ValueError("deferred gate batch cannot be empty")
        keys = [item.stable_key for item in self.messages]
        if len(keys) != len(set(keys)):
            raise ValueError("deferred gate batch contains duplicate messages")


@dataclass(frozen=True, slots=True)
class TemporaryAbsenceInterruption:
    reason: str
    started_at: datetime
    planned_until: datetime
    ended_at: datetime
    batch: DeferredGateBatch | None = None

    def prompt_metadata(self) -> dict[str, Any]:
        return temporary_absence_metadata(
            reason=self.reason,
            started_at=self.started_at,
            planned_until=self.planned_until,
            ended_at=self.ended_at,
            end_reason="TIMER",
        )


@runtime_checkable
class StateMessageGateRepository(Protocol):
    async def get_deferred_message_batch(
        self, profile_id: str, instance_id: str, batch_ref: str
    ) -> Mapping[str, Any] | None: ...

    async def get_state_message_gate_policy(
        self, profile_id: str, instance_id: str
    ) -> Mapping[str, Any] | StateGatePolicy: ...

    async def get_state_message_gate_snapshot(
        self, profile_id: str, instance_id: str
    ) -> Mapping[str, Any] | StateGateSnapshot | None: ...

    async def append_or_merge_deferred_gate_message(
        self,
        profile_id: str,
        instance_id: str,
        *,
        message_ref: str,
        ledger_entry_id: int,
        idempotency_key: str,
        gate_generation: int,
        activity_epoch: int,
        received_at: datetime,
        due_at: datetime,
    ) -> Mapping[str, Any] | DeferredGateBatch: ...

    async def claim_due_deferred_gate_batches(
        self,
        *,
        now: datetime,
        limit: int = 10,
        lease_seconds: int = 120,
    ) -> Sequence[Mapping[str, Any] | DeferredGateBatch]: ...

    async def claim_deferred_gate_batch_for_foreground(
        self,
        profile_id: str,
        instance_id: str,
        *,
        expected_activity_epoch: int,
        now: datetime,
        lease_seconds: int = 120,
    ) -> Mapping[str, Any] | DeferredGateBatch | None: ...

    async def resolve_deferred_gate_batch(
        self,
        profile_id: str,
        instance_id: str,
        batch_ref: str,
        *,
        expected_version: int,
        lease_token: int,
        expected_gate_generation: int,
        expected_activity_epoch: int,
        outcome: str,
        resolved_at: datetime,
    ) -> bool: ...

    async def renew_deferred_gate_batch_lease(
        self,
        profile_id: str,
        instance_id: str,
        batch_ref: str,
        *,
        expected_version: int,
        lease_token: int,
        now: datetime,
        lease_seconds: int = 120,
    ) -> bool: ...

    async def release_deferred_gate_batch(
        self,
        profile_id: str,
        instance_id: str,
        batch_ref: str,
        *,
        expected_version: int,
        lease_token: int,
        retry_at: datetime,
        reason: str,
    ) -> bool: ...

    async def ensure_expired_temporary_absence_wakeups(
        self, *, now: datetime, limit: int = 100
    ) -> int: ...

    async def prepare_temporary_absence_expiry(
        self, *args: object, **kwargs: object
    ) -> Mapping[str, Any]: ...

    async def finalize_temporary_absence_expiry(self, *args: object, **kwargs: object) -> bool: ...

    async def interrupt_temporary_absence_for_timer(
        self,
        profile_id: str,
        instance_id: str,
        *,
        now: datetime,
        lease_seconds: int = 120,
    ) -> Mapping[str, Any] | None: ...


def _active_state_gate_decision(
    snapshot: StateGateSnapshot,
    policy: StateGatePolicy,
    *,
    policy_expiry: datetime,
    common: dict[str, Any],
) -> StateGateDecision:
    if snapshot.mode is StateGateMode.DECLINE:
        return StateGateDecision(
            StateGateDisposition.RESTRICTED_DECLINE,
            snapshot.reason_code or "state_decline",
            **common,
        )
    if snapshot.mode is StateGateMode.SILENT:
        if policy.silent_enabled:
            return StateGateDecision(
                StateGateDisposition.SILENT,
                snapshot.reason_code or "state_silent",
                **common,
            )
        return StateGateDecision(
            StateGateDisposition.RESTRICTED_DECLINE,
            "silent_disabled_fallback_decline",
            **common,
        )
    assert snapshot.expires_at is not None
    return StateGateDecision(
        StateGateDisposition.DEFER,
        snapshot.reason_code or "state_defer",
        due_at=min(_as_utc(snapshot.expires_at), policy_expiry),
        **common,
    )


class StateMessageGate(TemporaryAbsenceExpiryGateMixin):
    """Policy evaluation and durable deferral orchestration."""

    def __init__(self, repository: StateMessageGateRepository | StateGateRepositoryPort) -> None:
        self.repository = repository

    @staticmethod
    def evaluate(
        snapshot: StateGateSnapshot | None,
        policy: StateGatePolicy,
        *,
        now: datetime,
        bypass: StateGateBypass | None = None,
    ) -> StateGateDecision:
        if not policy.enabled:
            return StateGateDecision(StateGateDisposition.PROCEED, "policy_disabled")
        trusted_bypass = bypass or StateGateBypass()
        if trusted_bypass.reason:
            return StateGateDecision(
                StateGateDisposition.PROCEED,
                f"bypass:{trusted_bypass.reason}",
                snapshot_generation=(snapshot.generation if snapshot else 0),
            )
        if snapshot is None:
            return StateGateDecision(StateGateDisposition.PROCEED, "no_snapshot")
        common = {
            "snapshot_generation": snapshot.generation,
            "expression_context": snapshot.expression_context,
            "reason_code": snapshot.reason_code,
            "effective_at": snapshot.effective_at,
            "expires_at": snapshot.expires_at,
        }
        if snapshot.mode is StateGateMode.OPEN:
            return StateGateDecision(
                StateGateDisposition.PROCEED,
                (
                    "temporary_absence_ended"
                    if snapshot.reason_code == TEMPORARY_ABSENCE_REASON_CODE
                    else "snapshot_open"
                ),
                **common,
            )
        if not snapshot.is_active(now):
            return StateGateDecision(
                StateGateDisposition.PROCEED,
                (
                    "temporary_absence_ended"
                    if snapshot.reason_code == TEMPORARY_ABSENCE_REASON_CODE
                    else "snapshot_inactive"
                ),
                **common,
            )

        # The persisted snapshot is globally capped at 24 hours.  A scope may
        # choose a shorter administrative cap, which must also shorten DEFER's
        # due time rather than silently keeping an overlong batch alive.
        assert snapshot.effective_at is not None
        policy_expiry = _as_utc(snapshot.effective_at) + policy.max_non_open_duration
        if _as_utc(now) >= policy_expiry:
            return StateGateDecision(
                StateGateDisposition.PROCEED,
                "policy_duration_elapsed",
                **common,
            )
        return _active_state_gate_decision(
            snapshot,
            policy,
            policy_expiry=policy_expiry,
            common=common,
        )

    async def decision_for_message(
        self,
        profile_id: str,
        instance_id: str,
        *,
        now: datetime,
        bypass: StateGateBypass | None = None,
    ) -> StateGateDecision:
        raw_policy = await self.repository.get_state_message_gate_policy(profile_id, instance_id)
        policy = (
            raw_policy
            if isinstance(raw_policy, StateGatePolicy)
            else StateGatePolicy.from_mapping(raw_policy)
        )
        if not policy.enabled:
            return self.evaluate(None, policy, now=now, bypass=bypass)
        trusted_bypass = bypass or StateGateBypass()
        if trusted_bypass.reason:
            return self.evaluate(None, policy, now=now, bypass=trusted_bypass)
        raw_snapshot = await self.repository.get_state_message_gate_snapshot(
            profile_id, instance_id
        )
        try:
            snapshot = (
                raw_snapshot
                if isinstance(raw_snapshot, StateGateSnapshot)
                else StateGateSnapshot.from_mapping(raw_snapshot)
                if raw_snapshot is not None
                else None
            )
        except (TypeError, ValueError):
            # Invalid persisted state must never make ordinary chat vanish.
            return StateGateDecision(StateGateDisposition.PROCEED, "invalid_snapshot")
        return self.evaluate(snapshot, policy, now=now, bypass=bypass)

    async def defer_message(
        self,
        profile_id: str,
        instance_id: str,
        *,
        decision: StateGateDecision,
        message: DeferredGateMessage,
    ) -> DeferredGateBatch:
        if decision.disposition is not StateGateDisposition.DEFER:
            raise ValueError("only a DEFER decision may enqueue a deferred message")
        if decision.due_at is None:
            raise ValueError("DEFER decision requires due_at")
        raw = await self.repository.append_or_merge_deferred_gate_message(
            profile_id,
            instance_id,
            message_ref=message.message_ref,
            ledger_entry_id=message.ledger_entry_id,
            idempotency_key=message.stable_key,
            gate_generation=decision.snapshot_generation,
            activity_epoch=message.activity_epoch,
            received_at=_as_utc(message.received_at),
            due_at=_as_utc(decision.due_at),
        )
        return _deferred_batch(raw)

    async def claim_due(
        self,
        *,
        now: datetime,
        limit: int = 10,
        lease_seconds: int = 120,
    ) -> tuple[DeferredGateBatch, ...]:
        rows = await self.repository.claim_due_deferred_gate_batches(
            now=_as_utc(now),
            limit=max(1, int(limit)),
            lease_seconds=max(1, int(lease_seconds)),
        )
        return tuple(_deferred_batch(row) for row in rows)

    async def claim_for_foreground(
        self,
        profile_id: str,
        instance_id: str,
        *,
        activity_epoch: int,
        now: datetime,
        lease_seconds: int = 120,
    ) -> DeferredGateBatch | None:
        raw = await self.repository.claim_deferred_gate_batch_for_foreground(
            profile_id,
            instance_id,
            expected_activity_epoch=activity_epoch,
            now=_as_utc(now),
            lease_seconds=max(1, int(lease_seconds)),
        )
        return _deferred_batch(raw) if raw is not None else None

    async def interrupt_for_timer(
        self,
        profile_id: str,
        instance_id: str,
        *,
        now: datetime,
        lease_seconds: int = 120,
    ) -> TemporaryAbsenceInterruption | None:
        raw = await self.repository.interrupt_temporary_absence_for_timer(
            profile_id,
            instance_id,
            now=_as_utc(now),
            lease_seconds=max(1, int(lease_seconds)),
        )
        if raw is None:
            return None
        started_at = _parse_datetime(raw.get("started_at"))
        planned_until = _parse_datetime(raw.get("planned_until"))
        ended_at = _parse_datetime(raw.get("ended_at"))
        if started_at is None or planned_until is None or ended_at is None:
            raise ValueError("temporary absence interruption returned invalid timestamps")
        batch_raw = raw.get("batch")
        return TemporaryAbsenceInterruption(
            reason=str(raw.get("reason") or "").strip()[:1000],
            started_at=started_at,
            planned_until=planned_until,
            ended_at=ended_at,
            batch=_deferred_batch(batch_raw) if isinstance(batch_raw, Mapping) else None,
        )

    async def resolve(
        self,
        batch: DeferredGateBatch,
        *,
        outcome: str,
        now: datetime,
    ) -> bool:
        if batch.status is not DeferredGateStatus.CLAIMED:
            raise ValueError("only a claimed deferred gate batch may be resolved")
        return bool(
            await self.repository.resolve_deferred_gate_batch(
                batch.profile_id,
                batch.instance_id,
                batch.batch_ref,
                expected_version=batch.version,
                lease_token=batch.lease_token,
                expected_gate_generation=batch.gate_generation,
                expected_activity_epoch=batch.activity_epoch,
                outcome=str(outcome).strip() or "completed",
                resolved_at=_as_utc(now),
            )
        )

    async def renew(
        self,
        batch: DeferredGateBatch,
        *,
        now: datetime,
        lease_seconds: int = 120,
    ) -> bool:
        if batch.status is not DeferredGateStatus.CLAIMED:
            raise ValueError("only a claimed deferred gate batch lease may be renewed")
        return bool(
            await self.repository.renew_deferred_gate_batch_lease(
                batch.profile_id,
                batch.instance_id,
                batch.batch_ref,
                expected_version=batch.version,
                lease_token=batch.lease_token,
                now=_as_utc(now),
                lease_seconds=max(1, int(lease_seconds)),
            )
        )

    async def probe_claim(self, batch: DeferredGateBatch) -> DeferredGateLeaseProbe:
        raw = await self.repository.get_deferred_message_batch(
            batch.profile_id,
            batch.instance_id,
            batch.batch_ref,
        )
        if raw is None:
            return DeferredGateLeaseProbe(DeferredGateLeaseDisposition.LOST)
        status = str(raw.get("status", "")).strip().upper()
        if (
            status == DeferredGateStatus.CLAIMED.value
            and raw.get("gate_generation") == batch.gate_generation
            and raw.get("activity_epoch") == batch.activity_epoch
            and raw.get("version") == batch.version
            and raw.get("lease_token") == batch.lease_token
        ):
            return DeferredGateLeaseProbe(DeferredGateLeaseDisposition.CURRENT)
        raw_run_id = raw.get("resolution_run_id")
        resolution_run_id = (
            raw_run_id if isinstance(raw_run_id, int) and not isinstance(raw_run_id, bool) else 0
        )
        if (
            status == DeferredGateStatus.RESOLVED.value
            and str(raw.get("resolution_reason", "")).strip() == "merged_into_foreground"
            and resolution_run_id > 0
        ):
            return DeferredGateLeaseProbe(
                DeferredGateLeaseDisposition.COMMITTED,
                resolution_run_id=resolution_run_id,
            )
        return DeferredGateLeaseProbe(DeferredGateLeaseDisposition.LOST)

    async def release(
        self,
        batch: DeferredGateBatch,
        *,
        retry_at: datetime,
        reason: str,
    ) -> bool:
        if batch.status is not DeferredGateStatus.CLAIMED:
            raise ValueError("only a claimed deferred gate batch may be released")
        return bool(
            await self.repository.release_deferred_gate_batch(
                batch.profile_id,
                batch.instance_id,
                batch.batch_ref,
                expected_version=batch.version,
                lease_token=batch.lease_token,
                retry_at=_as_utc(retry_at),
                reason=str(reason).strip() or "retry",
            )
        )


def _deferred_batch(value: Mapping[str, Any] | DeferredGateBatch) -> DeferredGateBatch:
    if isinstance(value, DeferredGateBatch):
        return value
    messages = tuple(
        item
        if isinstance(item, DeferredGateMessage)
        else DeferredGateMessage(
            message_ref=str(item.get("message_ref", "")),
            ledger_entry_id=int(item.get("ledger_entry_id", 0) or 0),
            activity_epoch=int(item.get("activity_epoch", 0) or 0),
            received_at=_required_datetime(item.get("received_at")),
            idempotency_key=str(item.get("idempotency_key", "") or ""),
        )
        for item in value.get("messages", ()) or ()
    )
    return DeferredGateBatch(
        batch_ref=str(value.get("batch_ref", "")),
        profile_id=str(value.get("profile_id", "")),
        instance_id=str(value.get("instance_id", "")),
        gate_generation=int(value.get("gate_generation", 0) or 0),
        activity_epoch=int(value.get("activity_epoch", 0) or 0),
        due_at=_required_datetime(value.get("due_at")),
        messages=messages,
        status=DeferredGateStatus(str(value.get("status", "PENDING")).strip().upper()),
        version=int(value.get("version", 0) or 0),
        lease_token=int(value.get("lease_token", 0) or 0),
    )


__all__ = [
    "DeferredGateBatch",
    "DeferredGateLeaseDisposition",
    "DeferredGateLeaseProbe",
    "DeferredGateMessage",
    "DeferredGateStatus",
    "MAX_NON_OPEN_DURATION",
    "StateGateBypass",
    "StateGateDecision",
    "StateGateDisposition",
    "StateGateMode",
    "StateGatePolicy",
    "StateGateSnapshot",
    "StateMessageGate",
    "StateMessageGateRepository",
    "TemporaryAbsenceInterruption",
]
