"""Internal runtime primitives for fenced durable AI task execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

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
