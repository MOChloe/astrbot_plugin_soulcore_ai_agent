"""Small orchestration surface for Timer occupancy consumers."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime

from .admission import (
    BeginTimerProviderCommand,
    ClaimNextTimerCommand,
    CompleteTimerNoOpCommand,
    HandoffTimerExpressionCommand,
    RetryTimerRunCommand,
    SupersedeTimerRunCommand,
    TimerAdmissionPort,
    TimerAdmissionResult,
    TimerProviderResult,
    TimerRunFence,
    TimerSettlementResult,
)
from .domain import DeliveryAssociationRef, ExecutionEnvelopeRef, TimerScope


class TimerOccupancyCoordinator:
    """Keep runtime consumers dependent on a narrow persistent port."""

    def __init__(self, admission: TimerAdmissionPort) -> None:
        self._admission = admission

    async def claim(self, command: ClaimNextTimerCommand) -> TimerAdmissionResult:
        return await self._admission.claim_next_timer(command)

    async def begin_provider(self, command: BeginTimerProviderCommand) -> TimerProviderResult:
        return await self._admission.begin_timer_provider(command)

    async def complete_noop(self, command: CompleteTimerNoOpCommand) -> TimerSettlementResult:
        return await self._admission.complete_timer_noop(command)

    async def supersede(self, command: SupersedeTimerRunCommand) -> TimerSettlementResult:
        return await self._admission.supersede_timer_run(command)

    async def retry(self, command: RetryTimerRunCommand) -> TimerSettlementResult:
        return await self._admission.retry_timer_run(command)

    async def handoff_expression(
        self, command: HandoffTimerExpressionCommand
    ) -> TimerSettlementResult:
        return await self._admission.handoff_timer_expression(command)

    async def reconcile(self, scope: TimerScope, *, now: datetime) -> int:
        return await self._admission.reconcile_timer_occupancy(scope, now=now)

    async def wait_for_instance_available(
        self,
        profile_id: str,
        instance_id: str,
        *,
        poll_seconds: float = 0.05,
    ) -> None:
        scope = TimerScope(profile_id, instance_id)
        while True:
            now = datetime.now(UTC)
            await self.reconcile(scope, now=now)
            if not await self._admission.is_instance_occupied(scope):
                return
            await asyncio.sleep(max(0.01, min(float(poll_seconds), 1.0)))

    async def begin_provider_metadata(
        self,
        metadata: Mapping[str, object],
        *,
        profile_id: str,
        instance_id: str,
        execution_ref: str,
        now: datetime,
    ) -> dict[str, object] | None:
        fence = self._metadata_fence(metadata, profile_id, instance_id)
        result = await self.begin_provider(
            BeginTimerProviderCommand(fence, ExecutionEnvelopeRef(execution_ref), now)
        )
        return result.fence.as_metadata() if result.fence is not None else None

    async def complete_noop_metadata(
        self, metadata: Mapping[str, object], *, now: datetime
    ) -> None:
        fence = self._metadata_fence(metadata)
        await self.complete_noop(CompleteTimerNoOpCommand(fence, now))

    async def supersede_metadata(self, metadata: Mapping[str, object], *, now: datetime) -> None:
        fence = self._metadata_fence(metadata)
        await self.supersede(SupersedeTimerRunCommand(fence, now))

    async def retry_metadata(self, metadata: Mapping[str, object], *, now: datetime) -> None:
        fence = self._metadata_fence(metadata)
        await self.retry(RetryTimerRunCommand(fence, now))

    async def handoff_expression_metadata(
        self,
        metadata: Mapping[str, object],
        *,
        delivery_ref: str,
        source_run_id: int,
        now: datetime,
    ) -> None:
        fence = self._metadata_fence(metadata)
        await self.handoff_expression(
            HandoffTimerExpressionCommand(
                fence, DeliveryAssociationRef(delivery_ref), source_run_id, now
            )
        )

    @staticmethod
    def _metadata_fence(
        metadata: Mapping[str, object],
        profile_id: str | None = None,
        instance_id: str | None = None,
    ) -> TimerRunFence:
        fence = TimerRunFence.from_metadata(dict(metadata))
        if fence is None:
            raise ValueError("invalid Timer admission fence metadata")
        if profile_id is not None and fence.scope.profile_id != profile_id:
            raise ValueError("Timer admission fence profile does not match request")
        if instance_id is not None and fence.scope.instance_id != instance_id:
            raise ValueError("Timer admission fence instance does not match request")
        return fence


__all__ = ["TimerOccupancyCoordinator"]
