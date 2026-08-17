"""Narrow Timer occupancy bridge for Main Core provider and settlement boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from ...contracts.ai_models import AICapabilityRequest, AIModelRequest
from ...contracts.models import CoreRunResult, CoreWakeRequest, RunStatus
from ...shared.time import utcnow
from ..profiles.service import ProfileRuntimeDisabled

TIMER_ADMISSION_FENCE_METADATA = "timer_admission_fence"
_FOREGROUND_CANCEL_REASON = "superseded_by_newer_foreground_activity"


class TimerOccupancyBridge(Protocol):
    async def wait_for_instance_available(
        self, profile_id: str, instance_id: str, *, poll_seconds: float = 0.05
    ) -> None: ...

    async def begin_provider_metadata(
        self,
        metadata: Mapping[str, object],
        *,
        profile_id: str,
        instance_id: str,
        execution_ref: str,
        now: datetime,
    ) -> dict[str, object] | None: ...

    async def complete_noop_metadata(
        self, metadata: Mapping[str, object], *, now: datetime
    ) -> None: ...

    async def supersede_metadata(
        self, metadata: Mapping[str, object], *, now: datetime
    ) -> None: ...

    async def retry_metadata(self, metadata: Mapping[str, object], *, now: datetime) -> None: ...

    async def handoff_expression_metadata(
        self,
        metadata: Mapping[str, object],
        *,
        delivery_ref: str,
        source_run_id: int,
        now: datetime,
    ) -> None: ...


@dataclass(slots=True)
class _TimerProviderState:
    fence: Mapping[str, object]
    provider_started: bool = False


class _TimerSupersededBeforeProvider(RuntimeError):
    pass


class _TimerAwareModelGateway:
    def __init__(self, runner: Any, delegate: Any) -> None:
        self._runner = runner
        self._delegate = delegate

    async def invoke_model(self, request: AIModelRequest) -> Any:
        await self._runner._begin_timer_provider_call(request)
        return await self._delegate.invoke_model(request)

    async def invoke_capability(self, request: AICapabilityRequest) -> Any:
        """Forward non-text capabilities through the MainCore gateway boundary."""

        return await self._delegate.invoke_capability(request)

    async def annotate_model_exchange(
        self,
        invocation_id: str,
        *,
        round_no: int,
        processing: dict[str, Any],
    ) -> None:
        await self._delegate.annotate_model_exchange(
            invocation_id,
            round_no=round_no,
            processing=processing,
        )

    async def resolve_backend_hint(self, **values: Any) -> Any:
        return await self._delegate.resolve_backend_hint(**values)

    async def is_capability_available(
        self,
        capability: str,
        profile_id: str,
        preferred_backend_id: str = "",
    ) -> bool:
        return bool(
            await self._delegate.is_capability_available(
                capability,
                profile_id,
                preferred_backend_id=preferred_backend_id,
            )
        )

    async def start_ai_workflow(self, **values: Any) -> Any:
        return await self._delegate.start_ai_workflow(**values)

    def bind_ai_workflow(self, trace: Any) -> Any:
        return self._delegate.bind_ai_workflow(trace)

    async def finish_ai_workflow(self, workflow_id: int, **values: Any) -> None:
        await self._delegate.finish_ai_workflow(workflow_id, **values)

    async def record_ai_work_event(self, **values: Any) -> None:
        await self._delegate.record_ai_work_event(**values)

    async def start_ai_work_node(self, **values: Any) -> Any:
        return await self._delegate.start_ai_work_node(**values)

    async def finish_ai_work_node(self, node_id: int, **values: Any) -> None:
        await self._delegate.finish_ai_work_node(node_id, **values)

    async def ai_work_node_context(self, invocation_id: str) -> Any:
        return await self._delegate.ai_work_node_context(invocation_id)

    async def project_model_visible_message_ids(
        self,
        run_id: int,
        node_id: int,
        message_ids: tuple[int, ...],
        *,
        summary_ids: tuple[int, ...] = (),
        summary_coverage: tuple[tuple[int, int, int], ...] = (),
    ) -> Any:
        return await self._delegate.project_model_visible_message_ids(
            run_id,
            node_id,
            message_ids,
            summary_ids=summary_ids,
            summary_coverage=summary_coverage,
        )


class TimerOccupancyMixin:
    def _initialize_timer_occupancy(
        self: Any,
        model_gateway: Any,
        occupancy: TimerOccupancyBridge,
    ) -> Any:
        self.timer_occupancy = occupancy
        self._timer_provider_states: dict[asyncio.Task[Any], _TimerProviderState] = {}
        self._active_run_ids: dict[asyncio.Task[Any], int] = {}
        return _TimerAwareModelGateway(self, model_gateway)

    def _clear_timer_occupancy_states(self: Any) -> None:
        self._timer_provider_states.clear()
        self._active_run_ids.clear()

    def _bind_active_run_id(self: Any, run_id: int) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._active_run_ids[task] = int(run_id)

    def _active_run_id(self: Any, task: asyncio.Task[Any]) -> int:
        return int(self._active_run_ids.get(task, 0))

    async def _run_locked_main_core(
        self: Any,
        request: CoreWakeRequest,
        role: Any,
        profile_config: Any,
        event: Any,
    ) -> CoreRunResult:
        current_task = asyncio.current_task()
        assert current_task is not None
        timer_state = None
        try:
            timer_state = await self._prepare_timer_state(request, current_task)
            result = await self._execute_locked_main_core(request, role, profile_config, event)
            if timer_state is not None and not timer_state.provider_started:
                await self._supersede_timer_state(timer_state)
            return result
        except _TimerSupersededBeforeProvider:
            return await self._timer_superseded_result(request, self._active_run_id(current_task))
        except asyncio.CancelledError as exc:
            run_id = self._active_run_id(current_task)
            if _FOREGROUND_CANCEL_REASON in {str(item) for item in exc.args}:
                await asyncio.shield(self._supersede_timer_state(timer_state))
            return await self._handle_core_cancellation(request, run_id, exc)
        except ProfileRuntimeDisabled:
            run_id = self._active_run_id(current_task)
            if run_id:
                await self._finish_run(
                    request, run_id, RunStatus.SUPERSEDED, error="profile_disabled"
                )
            raise
        except Exception as exc:
            run_id = self._active_run_id(current_task)
            result = await self._handle_core_failure(request, run_id, exc)
            await self._settle_timer_result(run_id, result)
            return result
        finally:
            self._timer_provider_states.pop(current_task, None)
            self._active_run_ids.pop(current_task, None)

    async def _prepare_timer_state(
        self: Any, request: CoreWakeRequest, current_task: asyncio.Task[Any]
    ) -> _TimerProviderState | None:
        timer_state = self._timer_provider_state(request)
        if timer_state is not None:
            self._timer_provider_states[current_task] = timer_state
        else:
            await self.timer_occupancy.wait_for_instance_available(
                request.profile_id, request.instance_id
            )
        return timer_state

    async def _timer_superseded_result(
        self: Any, request: CoreWakeRequest, run_id: int
    ) -> CoreRunResult:
        if run_id:
            await self._finish_run(
                request,
                run_id,
                RunStatus.SUPERSEDED,
                error="timer_superseded_before_provider",
            )
        return CoreRunResult(
            run_id,
            RunStatus.SUPERSEDED,
            superseded=True,
            error="timer_superseded_before_provider",
        )

    def _timer_provider_state(self: Any, request: CoreWakeRequest) -> _TimerProviderState | None:
        fence = request.metadata.get(TIMER_ADMISSION_FENCE_METADATA)
        if fence is None:
            return None
        if not isinstance(fence, Mapping):
            raise RuntimeError("Timer admission fence metadata is invalid")
        if self.timer_occupancy is None:
            raise RuntimeError("Timer admission fence requires a bound occupancy coordinator")
        return _TimerProviderState(fence)

    async def _begin_timer_provider_call(self: Any, provider_request: AIModelRequest) -> None:
        current_task = asyncio.current_task()
        state = self._timer_provider_states.get(current_task) if current_task is not None else None
        if state is None or state.provider_started:
            return
        assert self.timer_occupancy is not None
        run_id = int(provider_request.metadata["run_id"])
        fence = await self.timer_occupancy.begin_provider_metadata(
            state.fence,
            profile_id=provider_request.profile_id,
            instance_id=provider_request.instance_id,
            execution_ref=f"core-run:{run_id}",
            now=utcnow(),
        )
        if fence is None:
            raise _TimerSupersededBeforeProvider
        state.fence = fence
        state.provider_started = True

    async def _settle_timer_result(self: Any, run_id: int, result: CoreRunResult) -> None:
        current_task = asyncio.current_task()
        state = self._timer_provider_states.get(current_task) if current_task is not None else None
        if state is None or not state.provider_started or self.timer_occupancy is None:
            return
        if result.status is RunStatus.FAILED:
            await self.timer_occupancy.retry_metadata(state.fence, now=utcnow())
            return
        if result.status is not RunStatus.COMPLETED:
            await self._supersede_timer_state(state)
            return
        now = utcnow()
        if result.expression_batch_id:
            await self.timer_occupancy.handoff_expression_metadata(
                state.fence,
                delivery_ref=result.expression_batch_id,
                source_run_id=run_id,
                now=now,
            )
            return
        await self.timer_occupancy.complete_noop_metadata(state.fence, now=now)

    async def _supersede_timer_state(self: Any, state: _TimerProviderState | None) -> None:
        if state is None or self.timer_occupancy is None:
            return
        with suppress(Exception):
            await self.timer_occupancy.supersede_metadata(state.fence, now=utcnow())


__all__ = ["TimerOccupancyBridge", "TimerOccupancyMixin"]
