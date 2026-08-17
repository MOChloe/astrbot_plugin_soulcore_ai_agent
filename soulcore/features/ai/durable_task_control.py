"""Lease control and task-local identity for durable AI execution."""

from __future__ import annotations

import asyncio
from typing import Any

from ...contracts.durable_task_context import _current_task_id, current_durable_ai_task_id
from .durable_task_runtime import DurableTaskRuntimeRepository


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
