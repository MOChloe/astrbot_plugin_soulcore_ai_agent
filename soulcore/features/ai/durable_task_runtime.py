"Internal runtime primitives for fenced durable AI task execution."

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from ...contracts.durable_task_context import _current_task_id, current_durable_ai_task_id

TaskRecord = dict[str, Any]


class PrerequisiteTaskClaimOutcome(StrEnum):
    """Result of trying to claim one exact durable prerequisite task."""

    CLAIMED = "CLAIMED"
    ACTIVE = "ACTIVE"
    NOT_CLAIMABLE = "NOT_CLAIMABLE"


@dataclass(frozen=True, slots=True)
class PrerequisiteTaskClaim:
    outcome: PrerequisiteTaskClaimOutcome
    task: TaskRecord | None = None


class DurableTaskRuntimeRepository(Protocol):
    """Exact persistence surface used by the fenced runtime worker."""

    async def heartbeat_ai_task(
        self,
        task_id: int,
        lease_token: int,
        worker_id: str,
        *,
        lease_seconds: int,
        checkpoint: TaskRecord | None,
        progress: TaskRecord | None,
    ) -> TaskRecord | None: ...

    async def recover_expired_ai_tasks(
        self, *, current_worker_id: str | None = None
    ) -> list[TaskRecord]: ...

    async def claim_ai_tasks(
        self,
        worker_id: str,
        *,
        limit: int,
        lease_seconds: int,
        task_types: tuple[str, ...],
    ) -> list[TaskRecord]: ...

    async def claim_ai_task_prerequisite(
        self,
        worker_id: str,
        task_id: int,
        requester_task_id: int,
        requester_lease_token: int,
        *,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> PrerequisiteTaskClaim: ...

    async def release_ai_task(
        self,
        task_id: int,
        lease_token: int,
        worker_id: str,
        *,
        reason: str,
        due_at: datetime | None = None,
    ) -> bool: ...

    async def acknowledge_pause_ai_task(
        self,
        task_id: int,
        lease_token: int,
        worker_id: str,
        *,
        checkpoint: TaskRecord | None = None,
    ) -> bool: ...

    async def acknowledge_cancel_ai_task(
        self,
        task_id: int,
        lease_token: int,
        worker_id: str,
        *,
        recovery_required: bool = False,
    ) -> bool: ...

    async def defer_ai_task(
        self,
        task_id: int,
        lease_token: int,
        worker_id: str,
        *,
        result: TaskRecord | None,
        reason: str,
    ) -> bool: ...

    async def complete_ai_task(
        self,
        task_id: int,
        lease_token: int,
        worker_id: str,
        *,
        result: TaskRecord | None = None,
    ) -> bool: ...

    async def fail_ai_task(
        self,
        task_id: int,
        lease_token: int,
        worker_id: str,
        error: str,
        *,
        retryable: bool,
        recovery_required: bool,
    ) -> TaskRecord | None: ...

    async def get_ai_task(self, task_id: int) -> TaskRecord | None: ...
    async def record_ai_backend_success(self, backend_id: str) -> object: ...
    async def record_ai_backend_failure(self, backend_id: str, error: str) -> object: ...
    async def cleanup_ai_task_history(self) -> TaskRecord: ...
    async def create_ai_task(
        self, profile_id: str, task_type: str, **values: object
    ) -> TaskRecord: ...
    async def request_pause_ai_task(self, task_id: int, **values: object) -> TaskRecord | None: ...
    async def resume_ai_task(self, task_id: int, **values: object) -> TaskRecord | None: ...
    async def request_cancel_ai_task(self, task_id: int, **values: object) -> TaskRecord | None: ...
    async def manual_retry_ai_task(self, task_id: int, **values: object) -> TaskRecord | None: ...
    async def ensure_ai_task_workflow(self, task_id: int) -> TaskRecord | None: ...


class TaskControlSignal(Protocol):
    task_id: int
    lease_token: int
    worker_id: str
    requested_status: str
    control_event: asyncio.Event


class ActiveTaskRuntime:
    """In-process cancellation bridge for one already-claimed durable task."""

    def __init__(self, task: dict[str, Any]) -> None:
        self.task = task
        self.control: TaskControlSignal | None = None
        self.foreground_preempted = False
        self.stopped = asyncio.Event()

    def attach(self, control: TaskControlSignal) -> None:
        self.control = control
        if self.foreground_preempted:
            self._signal_control(control)

    def preempt_for_foreground(self) -> None:
        self.foreground_preempted = True
        if self.control is not None:
            self._signal_control(self.control)

    @staticmethod
    def _signal_control(control: TaskControlSignal) -> None:
        control.requested_status = "CANCEL_REQUESTED"
        control.control_event.set()


async def stop_runtime_watchers(
    runtime: ActiveTaskRuntime,
    heartbeat: asyncio.Task[Any] | None,
    gate_watch: asyncio.Task[Any] | None,
) -> None:
    """Cancel lifecycle watchers and always publish the local stop fence."""

    try:
        watchers = tuple(item for item in (heartbeat, gate_watch) if item is not None)
        for watcher in watchers:
            watcher.cancel()
        if watchers:
            await asyncio.gather(*watchers, return_exceptions=True)
    finally:
        runtime.stopped.set()


class AITaskLeaseLost(RuntimeError):
    pass


class AITaskPauseRequested(RuntimeError):
    pass


class AITaskCancelRequested(RuntimeError):
    pass


class AITaskControl:
    def __init__(
        self,
        repository: DurableTaskRuntimeRepository,
        task: dict[str, Any],
        worker_id: str,
        lease_seconds: int,
    ) -> None:
        self.repository = repository
        self.task_id = int(task["task_id"])
        self.lease_token = int(task["lease_token"])
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.checkpoint = dict(task.get("checkpoint") or {})
        self.progress = dict(task.get("progress") or {})
        self.requested_status = str(task.get("status") or "RUNNING")
        self.control_event = asyncio.Event()
        self.runtime_disabled = False

    async def heartbeat(
        self,
        *,
        checkpoint: dict[str, Any] | None = None,
        progress: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if checkpoint is not None:
            self.checkpoint = dict(checkpoint)
        if progress is not None:
            self.progress = dict(progress)
        row = await self.repository.heartbeat_ai_task(
            self.task_id,
            self.lease_token,
            self.worker_id,
            lease_seconds=self.lease_seconds,
            checkpoint=self.checkpoint if checkpoint is not None else None,
            progress=self.progress if progress is not None else None,
        )
        if row is None:
            raise AITaskLeaseLost(f"AI task lease lost: {self.task_id}")
        self.requested_status = str(row["status"])
        if self.requested_status == "PAUSE_REQUESTED":
            self.control_event.set()
            raise AITaskPauseRequested()
        if self.requested_status == "CANCEL_REQUESTED":
            self.control_event.set()
            raise AITaskCancelRequested()
        return row

    async def check_control(self) -> None:
        await self.heartbeat()

    async def pause(self, reason: str) -> None:
        row = await self.repository.request_pause_ai_task(
            self.task_id,
            actor_id=self.worker_id,
            reason=str(reason or ""),
        )
        if row is None:
            raise AITaskLeaseLost(f"AI task lease lost: {self.task_id}")
        self.requested_status = str(row["status"])
        self.control_event.set()
        raise AITaskPauseRequested()


__all__ = [
    "AITaskCancelRequested",
    "AITaskControl",
    "AITaskLeaseLost",
    "AITaskPauseRequested",
    "_current_task_id",
    "current_durable_ai_task_id",
]
