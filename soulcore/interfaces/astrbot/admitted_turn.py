"""Shared admitted-turn orchestration kept outside the inbound adapter."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from astrbot.api.event import AstrMessageEvent

from ...contracts.deferred_gate import DeferredGateCommitFence
from ...contracts.models import CoreRunResult, RunStatus
from ...features.conversation.ports import TurnBufferBatch
from ...features.main_core.foreground_coordinator import ForegroundLeaseLost
from ...features.timeline.state_gate import DeferredGateLeaseDisposition
from .foreground import ForegroundTurn
from .passive_feedback import (
    send_ephemeral_passive_notice,
    state_gate_no_reply_notice,
)
from .umo import CapturedUMO

_DEFERRED_FOREGROUND_LEASE_SECONDS = 180
_BACKGROUND_FOREGROUND_LEASE_SECONDS = 180
_FOREGROUND_CANCELLATION_TIMEOUT_SECONDS = 5.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _BackgroundForegroundLease:
    owner: str
    token: str
    heartbeat: asyncio.Task[Any]


class _AdmissionState:
    def __init__(self) -> None:
        self.gate: Any = None
        self.admission: Any = None
        self.deferred_batch: Any = None
        self.deferred_enqueued = False
        self.knowledge_released = False
        self.buffer_transferred = False


@dataclass(slots=True)
class _AdmittedTurnExecution:
    event: AstrMessageEvent
    profile_id: str
    instance: Any
    scope_config: Any
    captured: CapturedUMO
    message_text: str
    payload: dict[str, Any]
    ledger: Any
    epoch: int
    message_id: str
    current_ledgers: list[Any]
    turn_buffer_batch: TurnBufferBatch | None
    group_window: Any | None
    state: _AdmissionState


class AdmittedTurnMixin:
    async def _run_admitted(
        self,
        event: AstrMessageEvent,
        profile_id: str,
        instance: Any,
        scope_config: Any,
        captured: CapturedUMO,
        message_text: str,
        payload: dict[str, Any],
        ledger: Any,
        epoch: int,
        message_id: str,
        *,
        ledgers: list[Any] | None = None,
        turn_buffer_batch: TurnBufferBatch | None = None,
        group_window: Any | None = None,
    ) -> Any:
        turn = _AdmittedTurnExecution(
            event=event,
            profile_id=profile_id,
            instance=instance,
            scope_config=scope_config,
            captured=captured,
            message_text=message_text,
            payload=payload,
            ledger=ledger,
            epoch=epoch,
            message_id=message_id,
            current_ledgers=list(ledgers) if ledgers is not None else [ledger],
            turn_buffer_batch=turn_buffer_batch,
            group_window=group_window,
            state=_AdmissionState(),
        )
        foreground_lease = await self._start_background_foreground_lease(
            profile_id,
            instance.instance_id,
        )

        try:
            return await self._run_with_background_foreground_lease(
                foreground_lease.heartbeat,
                instance.instance_id,
                lambda: self._execute_admitted_turn(turn),
            )
        finally:
            try:
                await self._release_admission(
                    turn.state,
                    profile_id,
                    instance.instance_id,
                    turn.current_ledgers,
                    buffered=turn_buffer_batch is not None or group_window is not None,
                )
            finally:
                await self._finish_background_foreground_lease(
                    profile_id,
                    instance.instance_id,
                    foreground_lease,
                )

    async def _execute_admitted_turn(self, turn: _AdmittedTurnExecution) -> Any:
        await self.ai_tasks.interrupt_background_tasks(
            turn.profile_id,
            turn.instance.instance_id,
        )
        now = datetime.now(UTC)
        recall_fences = await self._recall_commit_fences(
            turn.profile_id,
            turn.instance.instance_id,
            turn.current_ledgers,
            turn.epoch,
        )
        if await self._admission_gate_stops_turn(
            turn.state,
            turn.profile_id,
            turn.instance,
            turn.message_text,
            turn.current_ledgers,
            turn.epoch,
            now,
            turn.turn_buffer_batch,
            turn.group_window,
            turn.captured,
            turn.event,
        ):
            await self._settle_recall_handoff(
                turn.profile_id,
                turn.instance.instance_id,
                recall_fences,
            )
            return None
        turn.message_text, deferred_ledgers = await self._claim_deferred(
            turn.state,
            turn.profile_id,
            turn.instance.instance_id,
            turn.epoch,
            turn.message_text,
            now,
        )
        turn.current_ledgers = self._merge_turn_ledgers(
            deferred_ledgers,
            turn.current_ledgers,
        )
        await self._reconstruct_durable_media_payload(
            turn.profile_id,
            turn.instance.instance_id,
            turn.current_ledgers,
            turn.payload,
        )
        recall_fences = await self._recall_commit_fences(
            turn.profile_id,
            turn.instance.instance_id,
            turn.current_ledgers,
            turn.epoch,
        )
        evidence, qpm = await self._reserve_turn(
            turn.state,
            turn.profile_id,
            turn.instance,
            turn.captured,
            turn.message_id,
            turn.ledger,
            turn.epoch,
        )
        if self._admission_rejected(turn.state.admission):
            await self._settle_rejected_admission(
                turn.state,
                turn.profile_id,
                turn.instance,
                turn.captured,
                turn.turn_buffer_batch,
                turn.group_window,
            )
            await self._settle_recall_handoff(
                turn.profile_id,
                turn.instance.instance_id,
                recall_fences,
            )
            return None
        group_fence = await self._attach_group_fence(turn.group_window)
        if turn.group_window is not None and group_fence is None:
            return False

        async def execute_foreground() -> Any:
            return await self._execute_admitted_foreground(
                turn.event,
                turn.profile_id,
                turn.instance,
                turn.scope_config,
                turn.captured,
                turn.message_text,
                turn.epoch,
                turn.ledger,
                turn.payload,
                turn.state,
                qpm,
                evidence,
                turn.current_ledgers,
                turn.turn_buffer_batch,
                group_fence,
                recall_fences,
                self._deferred_gate_commit_fence(turn.state.deferred_batch),
            )

        result = await self._run_with_deferred_lease(
            turn.state.deferred_batch,
            execute_foreground,
        )
        if turn.state.deferred_batch is not None and result.status is RunStatus.COMPLETED:
            turn.state.deferred_batch = None
        await self._resolve_result_buffer(turn.turn_buffer_batch, result)
        await self._resolve_result_group(turn.group_window, result)
        if turn.group_window is not None:
            await self.release_group_first_attempt_activity(
                turn.profile_id,
                turn.instance.instance_id,
                turn.group_window.window_id,
                turn.captured.raw,
            )
        return result

    async def _start_background_foreground_lease(
        self,
        profile_id: str,
        instance_id: str,
    ) -> _BackgroundForegroundLease:
        owner = f"main-core:{uuid.uuid4().hex}"
        token = await self.delivery_repository.acquire_foreground_lease(
            profile_id,
            instance_id,
            owner=owner,
            lease_seconds=_BACKGROUND_FOREGROUND_LEASE_SECONDS,
        )
        heartbeat = asyncio.create_task(
            self._maintain_background_foreground_lease(
                profile_id,
                instance_id,
                owner,
                token,
            ),
            name=f"soulcore-background-foreground-lease:{instance_id}",
        )
        return _BackgroundForegroundLease(owner=owner, token=token, heartbeat=heartbeat)

    @staticmethod
    async def _run_with_background_foreground_lease(
        heartbeat: asyncio.Task[Any],
        instance_id: str,
        operation: Any,
    ) -> Any:
        run = asyncio.create_task(
            operation(),
            name=f"soulcore-admitted-main-core:{instance_id}",
        )
        try:
            done, _ = await asyncio.wait(
                {run, heartbeat},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat in done:
                await heartbeat
                raise ForegroundLeaseLost(
                    f"foreground lease heartbeat stopped for admitted turn {instance_id}"
                )
            return await run
        finally:
            if not run.done():
                run.cancel("foreground_lease_lost")
            await asyncio.gather(run, return_exceptions=True)

    async def _finish_background_foreground_lease(
        self,
        profile_id: str,
        instance_id: str,
        lease: _BackgroundForegroundLease,
    ) -> None:
        await self._stop_background_foreground_heartbeat(lease.heartbeat)
        await self._release_background_foreground_lease(
            profile_id,
            instance_id,
            lease.owner,
            lease.token,
        )

    async def _maintain_background_foreground_lease(
        self,
        profile_id: str,
        instance_id: str,
        owner: str,
        token: str,
    ) -> None:
        while True:
            await asyncio.sleep(_BACKGROUND_FOREGROUND_LEASE_SECONDS / 3)
            renewed = await self.delivery_repository.renew_foreground_lease(
                profile_id,
                instance_id,
                owner=owner,
                token=token,
                lease_seconds=_BACKGROUND_FOREGROUND_LEASE_SECONDS,
            )
            if not renewed:
                raise ForegroundLeaseLost(
                    f"foreground lease lost for admitted turn {profile_id}/{instance_id}"
                )

    @staticmethod
    async def _stop_background_foreground_heartbeat(heartbeat: asyncio.Task[Any]) -> None:
        if not heartbeat.done():
            heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("foreground/background lease heartbeat failed while stopping")

    async def _release_background_foreground_lease(
        self,
        profile_id: str,
        instance_id: str,
        owner: str,
        token: str,
    ) -> None:
        release = asyncio.create_task(
            self.delivery_repository.release_foreground_lease(
                profile_id,
                instance_id,
                owner=owner,
                token=token,
            )
        )
        cancelled = False
        while not release.done():
            try:
                await asyncio.shield(release)
            except asyncio.CancelledError:
                cancelled = True
        await release
        if cancelled:
            raise asyncio.CancelledError

    async def _recall_commit_fences(
        self,
        profile_id: str,
        instance_id: str,
        ledgers: list[Any],
        activity_epoch: int,
    ) -> tuple[Any, ...]:
        repository = self.inbound_recall
        if repository is None:
            return ()
        return await repository.list_release_commit_fences(
            profile_id,
            instance_id,
            [int(item.message_id) for item in ledgers],
            activity_epoch=activity_epoch,
        )

    async def _settle_recall_handoff(
        self,
        profile_id: str,
        instance_id: str,
        fences: tuple[Any, ...],
    ) -> None:
        repository = self.inbound_recall
        if repository is None or not fences:
            return
        settled = await repository.mark_release_handoffs_dispatched(
            profile_id,
            instance_id,
            fences,
            now=datetime.now(UTC),
        )
        if not settled:
            raise RuntimeError("inbound-recall durable handoff ownership changed")

    @staticmethod
    def _merge_turn_ledgers(deferred: list[Any], current: list[Any]) -> list[Any]:
        merged: dict[int, Any] = {}
        for row in (*deferred, *current):
            message_id = int(row.message_id)
            if message_id:
                merged[message_id] = row
        return list(merged.values())

    @staticmethod
    def _deferred_gate_commit_fence(batch: Any | None) -> DeferredGateCommitFence | None:
        if batch is None:
            return None
        return DeferredGateCommitFence(
            batch_ref=batch.batch_ref,
            gate_generation=batch.gate_generation,
            activity_epoch=batch.activity_epoch,
            version=batch.version,
            lease_token=batch.lease_token,
        )

    async def _run_with_deferred_lease(self, batch: Any | None, operation: Any) -> Any:
        if batch is None:
            return await operation()
        renewed = await self.state_message_gate.renew(
            batch,
            now=datetime.now(UTC),
            lease_seconds=_DEFERRED_FOREGROUND_LEASE_SECONDS,
        )
        if not renewed:
            raise RuntimeError("deferred foreground lease changed before Main Core")
        run = asyncio.create_task(
            operation(),
            name=f"soulcore-foreground-deferred:{batch.batch_ref}",
        )
        heartbeat = asyncio.create_task(
            self._maintain_deferred_foreground_lease(batch),
            name=f"soulcore-foreground-deferred-lease:{batch.batch_ref}",
        )
        try:
            done, _pending = await asyncio.wait(
                {run, heartbeat},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if run in done:
                result = await run
                if (
                    heartbeat in done
                    and not heartbeat.cancelled()
                    and heartbeat.exception() is None
                ):
                    committed_run_id = heartbeat.result()
                    self._validate_deferred_commit_result(result, committed_run_id)
                return result
            committed_run_id = await heartbeat
            result = await run
            self._validate_deferred_commit_result(result, committed_run_id)
            return result
        finally:
            await self._cancel_foreground_task(run)
            await self._stop_deferred_heartbeat(heartbeat)

    @staticmethod
    async def _cancel_foreground_task(run: asyncio.Task[Any]) -> None:
        if run.done():
            AdmittedTurnMixin._record_stopped_foreground_result(run)
            return
        run.cancel()
        done, _pending = await asyncio.wait(
            {run},
            timeout=_FOREGROUND_CANCELLATION_TIMEOUT_SECONDS,
        )
        if run not in done:
            run.cancel()
            run.add_done_callback(AdmittedTurnMixin._record_stopped_foreground_result)
            return
        try:
            run.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("foreground Main Core failed while stopping")

    @staticmethod
    def _record_stopped_foreground_result(run: asyncio.Task[Any]) -> None:
        if run.cancelled():
            return
        error = run.exception()
        if error is not None:
            logger.error(
                "foreground Main Core failed while stopping",
                exc_info=(type(error), error, error.__traceback__),
            )

    @staticmethod
    async def _stop_deferred_heartbeat(heartbeat: asyncio.Task[Any]) -> None:
        if not heartbeat.done():
            heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("deferred foreground lease heartbeat failed while stopping")

    @staticmethod
    def _validate_deferred_commit_result(
        result: CoreRunResult, committed_run_id: int | None
    ) -> None:
        if committed_run_id is None:
            raise RuntimeError("deferred foreground lease changed during Main Core")
        if result.run_id != committed_run_id:
            raise RuntimeError("deferred foreground commit belongs to a different Main Core run")

    async def _maintain_deferred_foreground_lease(self, batch: Any) -> int | None:
        while True:
            await asyncio.sleep(_DEFERRED_FOREGROUND_LEASE_SECONDS / 3)
            renewed = await self.state_message_gate.renew(
                batch,
                now=datetime.now(UTC),
                lease_seconds=_DEFERRED_FOREGROUND_LEASE_SECONDS,
            )
            if not renewed:
                probe = await self.state_message_gate.probe_claim(batch)
                if probe.disposition is DeferredGateLeaseDisposition.COMMITTED:
                    return probe.resolution_run_id
                raise RuntimeError("deferred foreground lease changed during Main Core")

    async def _admission_gate_stops_turn(
        self,
        state: _AdmissionState,
        profile_id: str,
        instance: Any,
        message_text: str,
        ledgers: list[Any],
        epoch: int,
        now: datetime,
        turn_buffer_batch: TurnBufferBatch | None,
        group_window: Any | None,
        captured: CapturedUMO,
        event: AstrMessageEvent,
    ) -> bool:
        state.gate = await self.state_message_gate.decision_for_message(
            profile_id,
            instance.instance_id,
            now=now,
        )
        stopped = await self._short_circuit_gate(
            state,
            profile_id,
            instance.instance_id,
            ledgers,
            epoch,
            now,
            turn_buffer_batch,
        )
        if not stopped:
            return False
        await self._resolve_gate_buffer(turn_buffer_batch, state)
        notice = state_gate_no_reply_notice(state.gate)
        if notice:
            policy = await self.timeline.get_delivery_policy(profile_id, instance.scope)
            await send_ephemeral_passive_notice(
                profiles=self.profiles,
                delivery=self.delivery,
                event=event,
                captured=captured,
                profile_id=profile_id,
                instance_id=instance.instance_id,
                configured_group_limit=self._group_qpm(policy),
                text=notice,
            )
        if group_window is not None:
            outcome = "STATE_GATE_DEFERRED" if state.deferred_enqueued else "STATE_GATE_SILENT"
            await self.group_flow.resolve(
                group_window.profile_id,
                group_window.instance_id,
                group_window.window_id,
                outcome=outcome,
            )
            await self.release_group_first_attempt_activity(
                profile_id, instance.instance_id, group_window.window_id, captured.raw
            )
        return True

    async def _reserve_turn(
        self,
        state: _AdmissionState,
        profile_id: str,
        instance: Any,
        captured: CapturedUMO,
        message_id: str,
        ledger: Any,
        epoch: int,
    ) -> tuple[dict[str, Any] | None, int]:
        evidence = await self.timeline.claim_contact_evidence_for_foreground(
            profile_id, instance.instance_id, activity_epoch=epoch, limit=12
        )
        policy = await self.timeline.get_delivery_policy(profile_id, instance.scope)
        qpm = self._group_qpm(policy)
        state.admission = await self.delivery.reserve_main_core(
            captured,
            profile_id=profile_id,
            instance_id=instance.instance_id,
            origin_id=self._origin_id(message_id, ledger),
            configured_group_limit=qpm,
            proactive=False,
        )
        return evidence, qpm

    async def _settle_rejected_admission(
        self,
        state: _AdmissionState,
        profile_id: str,
        instance: Any,
        captured: CapturedUMO,
        turn_buffer_batch: TurnBufferBatch | None,
        group_window: Any | None,
    ) -> None:
        await self._resolve_optional_buffer(turn_buffer_batch, "QPM_SKIPPED")
        if group_window is None:
            return
        await self.group_flow.release_ready(
            group_window,
            retry_at=datetime.now(UTC) + timedelta(seconds=5),
            reason="group_delivery_quota_waiting",
        )
        await self.release_group_first_attempt_activity(
            profile_id, instance.instance_id, group_window.window_id, captured.raw
        )

    async def _attach_group_fence(self, group_window: Any | None) -> Any | None:
        if group_window is None:
            return None
        return await self.group_flow.attach_main_core_run(
            group_window,
            main_core_task_ref=f"group-flow:{group_window.window_id}:v{group_window.version}",
        )

    async def _execute_admitted_foreground(
        self,
        event: AstrMessageEvent,
        profile_id: str,
        instance: Any,
        scope_config: Any,
        captured: CapturedUMO,
        message_text: str,
        epoch: int,
        ledger: Any,
        payload: dict[str, Any],
        state: _AdmissionState,
        qpm: int,
        evidence: dict[str, Any] | None,
        ledgers: list[Any],
        batch: TurnBufferBatch | None,
        group_fence: Any | None,
        recall_fences: tuple[Any, ...],
        deferred_gate_fence: DeferredGateCommitFence | None,
    ) -> Any:
        return await self.foreground.run(
            ForegroundTurn(
                event,
                profile_id,
                instance,
                scope_config,
                captured,
                message_text,
                epoch,
                ledger,
                payload,
                state.admission,
                qpm,
                gate_decision=state.gate,
                contact_evidence=self._evidence_items(evidence),
                ledger_messages=ledgers,
                turn_buffer_fence=self._turn_buffer_commit_fence(batch),
                group_run_fence=group_fence,
                inbound_recall_fences=recall_fences,
                deferred_gate_fence=deferred_gate_fence,
            )
        )


__all__ = ["AdmittedTurnMixin"]
