"""Durable ``TIMER_RUN`` executor with profile/instance/route gates."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast

from ....contracts.ai_models import AIErrorCode, AIErrorInfo, AIInvocationError
from ....contracts.models import (
    CharacterInstance,
    CoreRunResult,
    RoleProfile,
    RouteReadiness,
    RunStatus,
    WakeSource,
)
from ..admission import TimerAdmissionFenceError, TimerAdmissionResult, TimerClaimOutcome
from ..domain import TimerOccurrenceId, TimerScope
from ..errors import TimerDomainError
from .executor import TimerProfileReader, TimerRuntimeExecutor
from .retry import TimerClaimRetrySettler

TIMER_RUN_TASK_TYPE = "TIMER_RUN"
TIMER_OCCUPANCY_BACKOFF_SECONDS = 30
logger = logging.getLogger(__name__)


class TimerTaskLeaseRepository(Protocol):
    async def release_ai_task(
        self,
        task_id: int,
        lease_token: int,
        worker_id: str,
        *,
        reason: str,
        due_at: datetime | None = None,
    ) -> object: ...


class TimerTaskControl(Protocol):
    repository: TimerTaskLeaseRepository
    task_id: int
    lease_token: int
    worker_id: str

    async def check_control(self) -> None: ...


class TimerTaskProfileReader(TimerProfileReader, Protocol):
    async def get_profile(self, profile_id: str) -> RoleProfile | None: ...


class TimerProactiveFramePrewarmer(Protocol):
    async def prewarm(
        self,
        *,
        profile_id: str,
        instance_id: str,
        source: WakeSource,
        source_ref: str,
        planned_main_core_at: datetime,
    ) -> object: ...


class TimerLifecycleCapture(Protocol):
    async def capture_after_main_core(
        self,
        *,
        scope: TimerScope,
        occurrence_id: TimerOccurrenceId,
        occurrence_generation: int,
        result: CoreRunResult,
        now: datetime,
    ) -> object: ...


class TimerRunTaskExecutor:
    """Execute one queued Timer while keeping generic task retries auditable."""

    def __init__(
        self,
        *,
        runtime: TimerRuntimeExecutor,
        retry_settler: TimerClaimRetrySettler,
        profiles: TimerTaskProfileReader,
        proactive_frame_prewarmer: TimerProactiveFramePrewarmer | None = None,
        lifecycle: TimerLifecycleCapture | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._runtime = runtime
        self._retry_settler = retry_settler
        self._profiles = profiles
        self._proactive_frame_prewarmer = proactive_frame_prewarmer
        self._lifecycle = lifecycle
        self._clock = clock or (lambda: datetime.now(UTC))

    async def execute(
        self,
        task: dict[str, object],
        raw_control: object,
    ) -> dict[str, object]:
        control = cast(TimerTaskControl, raw_control)
        await control.check_control()
        data = _mapping(task.get("input"))
        scope = TimerScope(
            str(data.get("profile_id") or task.get("profile_id") or ""),
            str(data.get("instance_id") or task.get("instance_id") or ""),
        )
        expected_id = TimerOccurrenceId(str(data.get("occurrence_id") or ""))
        expected_generation = int(str(data.get("generation") or 0))
        attempt = max(1, int(str(task.get("attempts") or 1)))
        maximum = max(1, int(str(task.get("max_attempts") or 4)))
        now = _aware_datetime(self._clock(), label="Timer execution clock")
        now = await self._prepare_proactive_frame(data, scope, expected_id, control, now=now)
        admission = await self._claim_expected(
            scope,
            expected_id,
            expected_generation,
            control=control,
            now=now,
        )
        if isinstance(admission, dict):
            return admission
        gate_error = await self._readiness_gate_error(scope)
        if gate_error:
            await self._settle_pre_provider_failure(
                admission,
                now=now,
                attempt=attempt,
                maximum=maximum,
                message=gate_error,
            )
        try:
            result = await self._runtime.execute_claimed(
                admission,
                requested_at=now,
                ai_task_id=control.task_id,
            )
        except (TimerAdmissionFenceError, TimerDomainError):
            return {"settlement": "stale_fence_noop"}
        except Exception as exc:
            await self._settle_pre_provider_failure(
                admission,
                now=now,
                attempt=attempt,
                maximum=maximum,
                message=f"{type(exc).__name__}: {exc}",
            )
            raise AssertionError("unreachable") from exc
        return await self._settle_run_result(
            result,
            scope=scope,
            expected_id=expected_id,
            expected_generation=expected_generation,
            control=control,
            now=now,
            attempt=attempt,
            maximum=maximum,
        )

    async def _prepare_proactive_frame(
        self,
        data: Mapping[str, object],
        scope: TimerScope,
        expected_id: TimerOccurrenceId,
        control: TimerTaskControl,
        *,
        now: datetime,
    ) -> datetime:
        if self._proactive_frame_prewarmer is None:
            return now
        try:
            planned = _aware_datetime(data.get("original_due_at"), fallback=now)
            stable_ref = str(data.get("stable_ref") or expected_id.value).strip()
            await self._proactive_frame_prewarmer.prewarm(
                profile_id=scope.profile_id,
                instance_id=scope.instance_id,
                source=WakeSource.TIMER,
                source_ref=f"timer-occurrence:{stable_ref}",
                planned_main_core_at=planned,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Timer proactive frame preparation failed for %s/%s: %s: %s",
                scope.profile_id,
                scope.instance_id,
                type(exc).__name__,
                exc,
            )
        # Re-observe cancellation and wall time before claiming Timer state.
        await control.check_control()
        return _aware_datetime(self._clock(), label="Timer post-prewarm clock")

    async def _claim_expected(
        self,
        scope: TimerScope,
        expected_id: TimerOccurrenceId,
        expected_generation: int,
        *,
        control: TimerTaskControl,
        now: datetime,
    ) -> TimerAdmissionResult | dict[str, object]:
        admission = await self._runtime.claim_scope(scope, now=now)
        if admission.outcome in {
            TimerClaimOutcome.PLAYER_WAITING,
            TimerClaimOutcome.OCCUPIED,
        }:
            return await self._release_task(
                control,
                admission.outcome.value.lower(),
                due_at=now + timedelta(seconds=TIMER_OCCUPANCY_BACKOFF_SECONDS),
            )
        if admission.outcome is TimerClaimOutcome.EMPTY:
            return {"settlement": "stale_task_noop"}
        if admission.occurrence is None:
            raise RuntimeError("Timer admission returned no occurrence")
        if (
            admission.occurrence.occurrence_id != expected_id
            or admission.occurrence.generation != expected_generation
        ):
            await self._retry_settler.settle(
                admission,
                now=now,
                attempt=1,
                max_attempts=2,
                retryable=True,
            )
            return await self._release_task(control, "timer_head_changed")
        return admission

    async def _readiness_gate_error(self, scope: TimerScope) -> str:
        profile = await self._profiles.get_profile(scope.profile_id)
        instance = await self._profiles.get_character_instance(scope.profile_id, scope.instance_id)
        return _readiness_gate_error(profile, instance)

    async def _settle_run_result(
        self,
        result: CoreRunResult,
        *,
        scope: TimerScope,
        expected_id: TimerOccurrenceId,
        expected_generation: int,
        control: TimerTaskControl,
        now: datetime,
        attempt: int,
        maximum: int,
    ) -> dict[str, object]:
        if result.status is RunStatus.COMPLETED:
            if self._lifecycle is not None:
                try:
                    await self._lifecycle.capture_after_main_core(
                        scope=scope,
                        occurrence_id=expected_id,
                        occurrence_generation=expected_generation,
                        result=result,
                        now=_aware_datetime(self._clock(), label="Timer lifecycle clock"),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Timer lifecycle candidate capture failed for %s/%s/%s",
                        scope.profile_id,
                        scope.instance_id,
                        expected_id.value,
                    )
            return {"settlement": "main_core_completed", "run_id": result.run_id}
        if result.status is RunStatus.SUPERSEDED:
            return {"settlement": "foreground_superseded", "run_id": result.run_id}
        if attempt >= maximum:
            if await self._settle_exhausted_after_run(scope, expected_id, now=now):
                raise _execution_error("Timer Main Core retry budget exhausted")
            return await self._release_task(control, "timer_exhaustion_waits_for_admission")
        raise _execution_error(result.error or "Timer Main Core invocation failed")

    async def _settle_pre_provider_failure(
        self,
        admission: TimerAdmissionResult,
        *,
        now: datetime,
        attempt: int,
        maximum: int,
        message: str,
    ) -> None:
        settled = await self._retry_settler.settle(
            admission,
            now=now,
            attempt=attempt,
            max_attempts=maximum,
            retryable=True,
        )
        if settled.exhausted:
            raise _execution_error(f"Timer runtime exhausted: {message}")
        raise _execution_error(f"Timer runtime retry: {message}")

    async def _settle_exhausted_after_run(
        self,
        scope: TimerScope,
        expected_id: TimerOccurrenceId,
        *,
        now: datetime,
    ) -> bool:
        admission = await self._runtime.claim_scope(scope, now=now)
        if (
            admission.outcome is not TimerClaimOutcome.CLAIMED
            or admission.occurrence is None
            or admission.occurrence.occurrence_id != expected_id
        ):
            return False
        await self._retry_settler.settle(
            admission,
            now=now,
            attempt=1,
            max_attempts=1,
            retryable=False,
        )
        return True

    @staticmethod
    async def _release_task(
        control: TimerTaskControl,
        reason: str,
        *,
        due_at: datetime | None = None,
    ) -> dict[str, object]:
        released = await control.repository.release_ai_task(
            control.task_id,
            control.lease_token,
            control.worker_id,
            reason=reason,
            due_at=due_at,
        )
        return {"settlement": "released", "reason": reason, "released": bool(released)}


def _readiness_gate_error(
    profile: RoleProfile | None,
    instance: CharacterInstance | None,
) -> str:
    if profile is None:
        return "profile_deleted"
    if not profile.enabled:
        return "profile_disabled"
    if instance is None:
        return "instance_deleted"
    if instance.readiness is not RouteReadiness.READY:
        return "route_not_ready"
    return ""


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _aware_datetime(
    value: object,
    *,
    fallback: datetime | None = None,
    label: str = "Timer proactive frame timestamp",
) -> datetime:
    if value in (None, ""):
        if fallback is None:
            raise ValueError(f"{label} is required")
        parsed = fallback
    else:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed.astimezone(UTC)


def _execution_error(message: str) -> AIInvocationError:
    return AIInvocationError(
        AIErrorInfo(
            AIErrorCode.NETWORK,
            message,
            retryable=True,
        )
    )


__all__ = [
    "TIMER_OCCUPANCY_BACKOFF_SECONDS",
    "TIMER_RUN_TASK_TYPE",
    "TimerProactiveFramePrewarmer",
    "TimerRunTaskExecutor",
]
