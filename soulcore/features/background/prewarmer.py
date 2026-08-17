"""Internal life-frame preparation before proactive Main Core runs.

The service deliberately accepts scheduling facts as data and never mutates a
``CoreWakeRequest``.  Consequently neither the source reference nor the frame
cutoff can be projected into the Main Core prompt.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from typing import Any, Protocol

from ...contracts.durable_task_context import current_durable_ai_task_id
from ...contracts.models import CoreWakeRequest, WakeSource
from .prewarm import (
    PROACTIVE_FRAME_TIMEOUT_MINUTES,
    ProactiveFrameTiming,
    proactive_frame_timing,
)

logger = logging.getLogger(__name__)

_PROACTIVE_SOURCES = {WakeSource.TIMER, WakeSource.PLUGIN_WAKE}
_TERMINAL_TASK_STATUSES = {"DEFERRED", "SUCCEEDED", "FAILED", "CANCELLED"}


class ProactiveFrameRepositoryPort(Protocol):
    async def materialize_proactive_frame_task(
        self,
        profile_id: str,
        instance_id: str,
        *,
        source_ref: str,
        planned_main_core_at: datetime,
        frame_end_at: datetime,
        deadline_at: datetime,
        now: datetime | None = None,
        requester_task_id: int | None = None,
    ) -> dict[str, Any]: ...

    async def get_ai_task(self, task_id: int) -> dict[str, Any] | None: ...


class ProactiveFrameTaskManagerPort(Protocol):
    async def execute_prerequisite_task(
        self,
        task_id: int,
    ) -> object: ...

    async def run_once(self, *, wait: bool = True) -> int: ...


@dataclass(frozen=True, slots=True)
class ProactiveFramePrewarmResult:
    outcome: str
    task_id: int | None = None
    task_status: str = ""


@dataclass(frozen=True, slots=True)
class _AdmittedFrameTask:
    outcome: str
    task_id: int
    task_status: str
    deadline: datetime


class ProactiveFramePrewarmer:
    """Reserve, execute, or reuse one bounded frame before proactive speech."""

    def __init__(
        self,
        repository: ProactiveFrameRepositoryPort,
        task_manager: ProactiveFrameTaskManagerPort,
        *,
        clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        poll_seconds: float = 0.05,
    ) -> None:
        self._repository = repository
        self._tasks = task_manager
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic_clock or monotonic
        self._poll_seconds = max(0.01, float(poll_seconds))

    async def prewarm_request(
        self,
        request: CoreWakeRequest,
    ) -> ProactiveFramePrewarmResult:
        """Prepare only proactive sources; inbound and deferred paths are no-ops."""

        if request.source not in _PROACTIVE_SOURCES:
            return ProactiveFramePrewarmResult("SOURCE_SKIPPED")
        # Durable Timer execution prepares before claiming its occupancy.  A
        # fenced request has already crossed that boundary and must not wait
        # behind (or attempt to bypass) the occupancy it owns.
        if request.source is WakeSource.TIMER and isinstance(
            request.metadata.get("timer_admission_fence"), dict
        ):
            return ProactiveFramePrewarmResult("TIMER_ALREADY_PREPARED")
        return await self.prewarm(
            profile_id=request.profile_id,
            instance_id=str(request.instance_id or ""),
            source=request.source,
            source_ref=self._request_source_ref(request),
            planned_main_core_at=(request.proactive_frame_planned_at or request.requested_at),
        )

    async def prewarm(
        self,
        *,
        profile_id: str,
        instance_id: str,
        source: WakeSource,
        source_ref: str,
        planned_main_core_at: datetime,
    ) -> ProactiveFramePrewarmResult:
        if source not in _PROACTIVE_SOURCES or not str(instance_id or "").strip():
            return ProactiveFramePrewarmResult("SOURCE_SKIPPED")
        now = self._aware(self._clock())
        planned = self._aware(planned_main_core_at)
        timing = proactive_frame_timing(
            source_ref=source_ref,
            planned_main_core_at=planned,
            admitted_at=now,
            # A delayed/recovered proactive source still needs a fresh frame
            # relative to the real opening time.  The original planned time
            # remains the deterministic random seed, so retry never re-rolls.
            effective_main_core_at=max(planned, now),
        )
        requester_task_id = current_durable_ai_task_id()
        admission = await self._materialize(
            profile_id,
            instance_id,
            source_ref=source_ref,
            timing=timing,
            now=now,
            requester_task_id=requester_task_id,
        )
        if isinstance(admission, ProactiveFramePrewarmResult):
            return admission
        admitted = self._admitted_task(admission, timing)
        if isinstance(admitted, ProactiveFramePrewarmResult):
            return admitted
        current = self._aware(self._clock())
        if admitted.deadline <= current:
            return ProactiveFramePrewarmResult("DEADLINE_ELAPSED", task_id=admitted.task_id)
        # Persisted UTC is authoritative across restarts.  Within one process,
        # also cap the wait with a monotonic deadline so an NTP/host-clock
        # rollback cannot stretch the twenty-minute budget indefinitely.
        persisted_remaining = (admitted.deadline - current).total_seconds()
        monotonic_deadline = self._monotonic() + min(
            persisted_remaining,
            float(PROACTIVE_FRAME_TIMEOUT_MINUTES * 60),
        )

        # A durable proactive caller can execute its exact child inside the
        # current worker slot.  Direct/manual wakeups have no trusted caller;
        # they ask the ordinary manager loop to claim the task instead.
        if requester_task_id is not None:
            result = await self._execute_exact_prerequisite(
                admitted,
                monotonic_deadline=monotonic_deadline,
            )
            if result is not None:
                return result
        else:
            await self._nudge_worker(admitted.task_id)

        return await self._wait_for_terminal(
            admitted.task_id,
            deadline=admitted.deadline,
            monotonic_deadline=monotonic_deadline,
            initial=admitted.outcome,
        )

    async def _materialize(
        self,
        profile_id: str,
        instance_id: str,
        *,
        source_ref: str,
        timing: ProactiveFrameTiming,
        now: datetime,
        requester_task_id: int | None,
    ) -> dict[str, Any] | ProactiveFramePrewarmResult:
        try:
            return await self._repository.materialize_proactive_frame_task(
                profile_id,
                instance_id,
                source_ref=source_ref,
                planned_main_core_at=timing.planned_main_core_at,
                frame_end_at=timing.frame_end_at,
                deadline_at=timing.deadline_at,
                now=now,
                requester_task_id=requester_task_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "proactive frame admission failed for %s/%s: %s: %s",
                profile_id,
                instance_id,
                type(exc).__name__,
                exc,
            )
            return ProactiveFramePrewarmResult("ADMISSION_FAILED")

    def _admitted_task(
        self,
        admission: dict[str, Any],
        timing: ProactiveFrameTiming,
    ) -> _AdmittedFrameTask | ProactiveFramePrewarmResult:
        outcome = str(admission.get("outcome") or "INELIGIBLE").upper()
        task = admission.get("task")
        if not isinstance(task, dict):
            return ProactiveFramePrewarmResult(outcome)
        task_id = int(task.get("task_id") or 0)
        if task_id < 1:
            return ProactiveFramePrewarmResult("INVALID_TASK")
        return _AdmittedFrameTask(
            outcome=outcome,
            task_id=task_id,
            task_status=str(task.get("status") or "").upper(),
            deadline=self._task_deadline(task) or timing.deadline_at,
        )

    async def _execute_exact_prerequisite(
        self,
        admitted: _AdmittedFrameTask,
        *,
        monotonic_deadline: float,
    ) -> ProactiveFramePrewarmResult | None:
        exact = asyncio.create_task(
            self._tasks.execute_prerequisite_task(admitted.task_id),
            name=f"soulcore-proactive-frame-prerequisite:{admitted.task_id}",
        )
        remaining = self._remaining_budget(admitted.deadline, monotonic_deadline)
        done, _ = await asyncio.wait({exact}, timeout=remaining)
        if exact not in done:
            exact.cancel("proactive_frame_deadline_elapsed")
            await asyncio.gather(exact, return_exceptions=True)
            return ProactiveFramePrewarmResult(
                "DEADLINE_ELAPSED",
                task_id=admitted.task_id,
                task_status=admitted.task_status,
            )
        try:
            exact_result = await exact
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "proactive frame exact execution failed for task %s: %s: %s",
                admitted.task_id,
                type(exc).__name__,
                exc,
            )
            return None
        exact_outcome = str(getattr(exact_result, "outcome", "")).upper()
        if not exact_outcome.endswith("NOT_CLAIMABLE"):
            return None
        row = await self._repository.get_ai_task(admitted.task_id)
        return ProactiveFramePrewarmResult(
            "BLOCKED",
            task_id=admitted.task_id,
            task_status=str((row or {}).get("status") or "").upper(),
        )

    async def _nudge_worker(self, task_id: int) -> None:
        try:
            await self._tasks.run_once(wait=False)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "proactive frame worker nudge failed for task %s: %s: %s",
                task_id,
                type(exc).__name__,
                exc,
            )

    async def _wait_for_terminal(
        self,
        task_id: int,
        *,
        deadline: datetime,
        monotonic_deadline: float,
        initial: str,
    ) -> ProactiveFramePrewarmResult:
        while True:
            row = await self._repository.get_ai_task(task_id)
            if row is None:
                return ProactiveFramePrewarmResult("TASK_DISAPPEARED", task_id=task_id)
            status = str(row.get("status") or "").upper()
            if status in _TERMINAL_TASK_STATUSES:
                return ProactiveFramePrewarmResult(
                    initial,
                    task_id=task_id,
                    task_status=status,
                )
            remaining = self._remaining_budget(deadline, monotonic_deadline)
            if remaining <= 0:
                return ProactiveFramePrewarmResult(
                    "DEADLINE_ELAPSED",
                    task_id=task_id,
                    task_status=status,
                )
            await asyncio.sleep(min(self._poll_seconds, remaining))

    def _remaining_budget(self, deadline: datetime, monotonic_deadline: float) -> float:
        wall_remaining = (deadline - self._aware(self._clock())).total_seconds()
        monotonic_remaining = monotonic_deadline - self._monotonic()
        return max(0.0, min(wall_remaining, monotonic_remaining))

    @staticmethod
    def _task_deadline(task: dict[str, Any]) -> datetime | None:
        data = dict(task.get("input") or {})
        metadata = data.get("proactive_frame")
        if not isinstance(metadata, dict) or not metadata.get("deadline_at"):
            return None
        return ProactiveFramePrewarmer._aware(datetime.fromisoformat(str(metadata["deadline_at"])))

    @staticmethod
    def _request_source_ref(request: CoreWakeRequest) -> str:
        persisted = str(request.proactive_frame_source_ref or "").strip()
        if persisted:
            return persisted
        if request.wakeup_id is not None:
            return f"instance-wakeup:{int(request.wakeup_id)}"
        contact_ref = str(request.metadata.get("contact_attempt_ref") or "").strip()
        if contact_ref:
            return f"contact-attempt:{contact_ref}"
        ai_task_id = int(request.metadata.get("ai_task_id") or 0)
        if ai_task_id > 0:
            return f"ai-task:{ai_task_id}"
        digest = hashlib.sha256(
            "|".join(
                (
                    str(request.profile_id),
                    str(request.instance_id or ""),
                    request.source.value,
                    ProactiveFramePrewarmer._aware(request.requested_at).isoformat(),
                )
            ).encode("utf-8")
        ).hexdigest()
        return f"jit:{digest}"

    @staticmethod
    def _aware(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("proactive frame datetime must be timezone-aware")
        return value.astimezone(UTC)


__all__ = [
    "ProactiveFramePrewarmResult",
    "ProactiveFramePrewarmer",
    "ProactiveFrameRepositoryPort",
    "ProactiveFrameTaskManagerPort",
]
