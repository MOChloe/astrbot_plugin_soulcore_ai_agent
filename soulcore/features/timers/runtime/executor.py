"""First durable Timer runtime closure: discover, admit, and wake Main Core."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Protocol

from ....contracts.models import CharacterInstance, CoreRunResult, CoreWakeRequest
from ..admission import (
    ClaimNextTimerCommand,
    TimerAdmissionResult,
    TimerClaimOutcome,
)
from ..domain import (
    IdempotencyKey,
    TimerOccurrence,
    TimerScope,
    require_aware,
)
from ..occupancy import TimerOccupancyCoordinator
from ..ports import TimerOccurrenceMutationWriter, TimerOccurrenceRollReader, TimerPageReader
from ..repository import AdvanceOccurrenceCommand
from ..service import SourceMessageRef, build_timer_wake_request
from ..transitions import OccurrenceAction


class TimerProfileReader(Protocol):
    async def get_character_instance(
        self, profile_id: str, instance_id: str
    ) -> CharacterInstance | None: ...


class TimerMainCoreRunner(Protocol):
    async def handle(self, request: CoreWakeRequest) -> CoreRunResult: ...


class TimerDeferredMessage(Protocol):
    ledger_entry_id: int


class TimerDeferredBatch(Protocol):
    messages: Sequence[TimerDeferredMessage]
    activity_epoch: int
    batch_ref: str
    version: int
    lease_token: int
    gate_generation: int
    due_at: datetime


class TimerTemporaryAbsenceInterruption(Protocol):
    batch: TimerDeferredBatch | None

    def prompt_metadata(self) -> dict[str, object]: ...


class TimerStateMessageGate(Protocol):
    async def interrupt_for_timer(
        self,
        profile_id: str,
        instance_id: str,
        *,
        now: datetime,
        lease_seconds: int,
    ) -> TimerTemporaryAbsenceInterruption | None: ...

    async def release(
        self,
        batch: TimerDeferredBatch,
        *,
        retry_at: datetime,
        reason: str,
    ) -> bool: ...

    async def renew(
        self,
        batch: TimerDeferredBatch,
        *,
        now: datetime,
        lease_seconds: int,
    ) -> bool: ...


class TimerRuntimeRepository(
    TimerPageReader,
    TimerOccurrenceMutationWriter,
    TimerOccurrenceRollReader,
    Protocol,
):
    """Persistence intersection required by one Timer runtime executor."""


class TimerRuntimeExecutor:
    """Move due rows into the persistent queue and execute only its admitted head."""

    def __init__(
        self,
        *,
        timers: TimerRuntimeRepository,
        profiles: TimerProfileReader,
        occupancy: TimerOccupancyCoordinator,
        runner: TimerMainCoreRunner,
        worker_id: str,
        lease_seconds: int = 300,
        scan_limit: int = 256,
        state_message_gate: TimerStateMessageGate | None = None,
    ) -> None:
        self._timers = timers
        self._profiles = profiles
        self._occupancy = occupancy
        self._runner = runner
        self._worker_id = _safe_runtime_ref(worker_id)
        self._lease_seconds = max(30, int(lease_seconds))
        self._scan_limit = max(1, min(int(scan_limit), 256))
        self._state_message_gate = state_message_gate

    async def claim_scope(self, scope: TimerScope, *, now: datetime) -> TimerAdmissionResult:
        now = require_aware(now)
        await self._occupancy.reconcile(scope, now=now)
        marker = uuid.uuid4().hex
        return await self._occupancy.claim(
            ClaimNextTimerCommand(
                scope=scope,
                occupancy_id=f"timer-occ:{marker}",
                lease_owner=self._worker_id,
                lease_token=f"lease:{marker}",
                lease_expires_at=now + timedelta(seconds=self._lease_seconds),
                now=now,
            )
        )

    async def reconcile_scope(self, scope: TimerScope, *, now: datetime) -> int:
        return await self._occupancy.reconcile(scope, now=require_aware(now))

    async def mark_due(self, scope: TimerScope, *, now: datetime) -> int:
        now = require_aware(now)
        marked = 0
        while marked < self._scan_limit:
            due = await self._timers.list_due_scheduled_occurrences(
                scope, through=now, limit=min(64, self._scan_limit - marked)
            )
            if not due:
                break
            for occurrence in due:
                await self._timers.advance_occurrence(
                    AdvanceOccurrenceCommand(
                        scope=scope,
                        occurrence_id=occurrence.occurrence_id,
                        action=OccurrenceAction.MARK_DUE,
                        expected_version=occurrence.version,
                        expected_generation=occurrence.generation,
                        now=now,
                        idempotency_key=_operation_key("due", occurrence),
                    )
                )
                marked += 1
            if len(due) < min(64, self._scan_limit - marked + len(due)):
                break
        return marked

    async def execute_claimed(
        self,
        admission: TimerAdmissionResult,
        *,
        requested_at: datetime,
        ai_task_id: int | None = None,
    ) -> CoreRunResult:
        if (
            admission.outcome is not TimerClaimOutcome.CLAIMED
            or admission.occurrence is None
            or admission.fence is None
        ):
            raise ValueError("Timer execution requires a claimed occurrence and fence")
        occurrence = admission.occurrence
        rule = await self._timers.get_rule(occurrence.scope, occurrence.rule_id)
        if rule is None:
            raise RuntimeError("claimed Timer rule no longer exists")
        instance = await self._profiles.get_character_instance(
            occurrence.scope.profile_id,
            occurrence.scope.instance_id,
        )
        if instance is None:
            raise RuntimeError("claimed Timer instance no longer exists")
        interruption = await self._interrupt_temporary_absence(
            occurrence.scope,
            now=requested_at,
        )
        try:
            held_refs = (
                tuple(
                    SourceMessageRef(f"ledger-message:{item.ledger_entry_id}")
                    for item in interruption.batch.messages
                )
                if interruption is not None and interruption.batch is not None
                else ()
            )
            request = build_timer_wake_request(
                profile_id=occurrence.scope.profile_id,
                instance_id=occurrence.scope.instance_id,
                route_umo=str(instance.route_umo),
                prompt=rule.prompt,
                admission_fence=admission.fence.as_metadata(),
                source_message_refs=tuple(dict.fromkeys((*rule.source_message_refs, *held_refs))),
                caused_by_run_ref=rule.source_run_ref.value,
                ai_task_id=ai_task_id,
                requested_at=require_aware(requested_at),
            )
            if interruption is not None:
                request.metadata["temporary_absence"] = interruption.prompt_metadata()
                if interruption.batch is not None:
                    batch = interruption.batch
                    request.expected_activity_epoch = batch.activity_epoch
                    request.metadata["deferred_gate_fence"] = {
                        "batch_ref": batch.batch_ref,
                        "version": batch.version,
                        "lease_token": batch.lease_token,
                        "gate_generation": batch.gate_generation,
                        "activity_epoch": batch.activity_epoch,
                    }
        except BaseException:
            if interruption is not None and interruption.batch is not None:
                await self._release_deferred_batch_safely(
                    interruption.batch,
                    reason="timer_request_preparation_failed",
                )
            raise
        return await self._run_with_deferred_lease(request, interruption)

    async def _interrupt_temporary_absence(
        self,
        scope: TimerScope,
        *,
        now: datetime,
    ) -> TimerTemporaryAbsenceInterruption | None:
        if self._state_message_gate is None:
            return None
        return await self._state_message_gate.interrupt_for_timer(
            scope.profile_id,
            scope.instance_id,
            now=require_aware(now),
            lease_seconds=self._lease_seconds,
        )

    async def _run_with_deferred_lease(
        self,
        request: CoreWakeRequest,
        interruption: TimerTemporaryAbsenceInterruption | None,
    ) -> CoreRunResult:
        batch = interruption.batch if interruption is not None else None
        if batch is None or self._state_message_gate is None:
            return await self._runner.handle(request)
        run = asyncio.create_task(self._runner.handle(request))
        heartbeat = asyncio.create_task(self._maintain_deferred_lease(batch))
        try:
            done, _ = await asyncio.wait({run, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                await heartbeat
                raise RuntimeError("temporary absence message ownership changed")
            result = await run
            if result.status.value != "COMPLETED":
                await self._state_message_gate.release(
                    batch,
                    retry_at=datetime.now(batch.due_at.tzinfo) + timedelta(minutes=1),
                    reason=result.error or "timer_main_core_not_committed",
                )
            return result
        except BaseException:
            if not run.done():
                run.cancel("temporary_absence_message_lease_lost")
            await self._release_deferred_batch_safely(
                batch,
                reason="timer_main_core_failed",
            )
            raise
        finally:
            heartbeat.cancel()
            await asyncio.gather(run, heartbeat, return_exceptions=True)

    async def _maintain_deferred_lease(self, batch: TimerDeferredBatch) -> None:
        assert self._state_message_gate is not None
        while True:
            await asyncio.sleep(self._lease_seconds / 3)
            if not await self._state_message_gate.renew(
                batch,
                now=datetime.now(batch.due_at.tzinfo),
                lease_seconds=self._lease_seconds,
            ):
                return

    async def _release_deferred_batch_safely(
        self,
        batch: TimerDeferredBatch,
        *,
        reason: str,
    ) -> bool:
        assert self._state_message_gate is not None
        release = asyncio.create_task(
            self._state_message_gate.release(
                batch,
                retry_at=datetime.now(batch.due_at.tzinfo) + timedelta(minutes=1),
                reason=reason,
            ),
            name="soulcore-timer-temporary-absence-release",
        )
        current = asyncio.current_task()
        assert current is not None
        observed_cancellations = current.cancelling()
        while True:
            try:
                return bool(await asyncio.shield(release))
            except asyncio.CancelledError as exc:
                current_cancellations = current.cancelling()
                if current_cancellations > observed_cancellations:
                    observed_cancellations = current_cancellations
                    continue
                raise RuntimeError("temporary absence release task was cancelled") from exc


def _operation_key(kind: str, occurrence: TimerOccurrence) -> IdempotencyKey:
    payload = (
        f"{kind}:{occurrence.scope.profile_id}:{occurrence.scope.instance_id}:"
        f"{occurrence.occurrence_id.value}:{occurrence.version}:{occurrence.generation}"
    )
    return IdempotencyKey(f"timer-runtime:{hashlib.sha256(payload.encode()).hexdigest()}")


def _safe_runtime_ref(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or len(normalized) > 160:
        raise ValueError("invalid Timer runtime worker id")
    return normalized


__all__ = ["TimerRuntimeExecutor"]
