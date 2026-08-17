"""Idempotent Timer scanner that enqueues one durable head task per instance."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from typing import Protocol

from ....contracts.models import CharacterInstance, RoleProfile
from ..domain import TimerOccurrence, TimerScope, require_aware
from ..ports import TimerPageReader
from ..task_identity import timer_run_task_idempotency_key
from .executor import TimerRuntimeExecutor
from .recovery import TimerRuntimeRecovery
from .tasks import TIMER_RUN_TASK_TYPE


class TimerRuntimeProfiles(Protocol):
    async def list_profiles(self, *, include_orphaned: bool = True) -> Sequence[RoleProfile]: ...

    async def list_character_instances(
        self,
        profile_id: str,
        scope: str | None = None,
    ) -> Sequence[CharacterInstance]: ...


class TimerTaskCreator(Protocol):
    async def create_ai_task(
        self,
        profile_id: str,
        task_type: str,
        **values: object,
    ) -> Mapping[str, object]: ...


class TimerLifecycleRecovery(Protocol):
    async def recover_if_due(self, *, now: datetime, force: bool = False) -> int: ...


class TimerRuntimeWorker:
    def __init__(
        self,
        *,
        profiles: TimerRuntimeProfiles,
        timers: TimerPageReader,
        tasks: TimerTaskCreator,
        executor: TimerRuntimeExecutor,
        recovery: TimerRuntimeRecovery,
        lifecycle: TimerLifecycleRecovery | None = None,
        poll_seconds: float = 5.0,
    ) -> None:
        self._profiles = profiles
        self._timers = timers
        self._tasks = tasks
        self._executor = executor
        self._recovery = recovery
        self._lifecycle = lifecycle
        self._poll_seconds = max(0.1, float(poll_seconds))
        self._loop_task: asyncio.Task[None] | None = None
        self._closed = False
        self.last_error = ""

    @property
    def running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    def start(self) -> None:
        if self.running:
            return
        self._closed = False
        self._loop_task = asyncio.create_task(
            self._loop(),
            name="soulcore-timer-runtime",
        )

    async def stop(self) -> None:
        self._closed = True
        loop, self._loop_task = self._loop_task, None
        if loop is None:
            return
        loop.cancel()
        with suppress(asyncio.CancelledError):
            await loop

    async def scan_once(self, *, now: datetime | None = None) -> int:
        current = require_aware(now or datetime.now(UTC))
        enqueued = 0
        errors: list[str] = []
        if self._lifecycle is not None:
            try:
                await self._lifecycle.recover_if_due(now=current)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                errors.append(f"lifecycle: {type(exc).__name__}: {exc}")
        for profile in await self._profiles.list_profiles(include_orphaned=False):
            if not profile.enabled:
                continue
            try:
                instances = await self._profiles.list_character_instances(profile.profile_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                errors.append(f"{profile.profile_id}: {type(exc).__name__}: {exc}")
                continue
            for instance in instances:
                scope = TimerScope(profile.profile_id, instance.instance_id)
                try:
                    await self._executor.reconcile_scope(scope, now=current)
                    await self._recovery.reconcile_scope(scope, now=current)
                    await self._executor.mark_due(scope, now=current)
                    head = await self._waiting_head(scope)
                    if head is None:
                        continue
                    await self._enqueue(head)
                    enqueued += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    errors.append(
                        f"{scope.profile_id}/{scope.instance_id}: {type(exc).__name__}: {exc}"
                    )
                    continue
        self.last_error = "; ".join(errors)
        return enqueued

    async def _loop(self) -> None:
        while not self._closed:
            try:
                await self.scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(self._poll_seconds)

    async def _waiting_head(self, scope: TimerScope) -> TimerOccurrence | None:
        return await self._timers.first_waiting_occurrence(scope)

    async def _enqueue(self, occurrence: TimerOccurrence) -> None:
        await self._tasks.create_ai_task(
            occurrence.scope.profile_id,
            TIMER_RUN_TASK_TYPE,
            instance_id=occurrence.scope.instance_id,
            task_class="BACKGROUND",
            capability="conversation.timer_run",
            due_at=occurrence.original_due_at,
            priority=0,
            mutex_key="main-core-runtime",
            idempotency_key=timer_run_task_idempotency_key(
                occurrence.scope.profile_id,
                occurrence.scope.instance_id,
                occurrence.occurrence_id.value,
                occurrence.generation,
            ),
            generation=occurrence.generation + 1,
            input_data={
                "profile_id": occurrence.scope.profile_id,
                "instance_id": occurrence.scope.instance_id,
                "occurrence_id": occurrence.occurrence_id.value,
                "stable_ref": occurrence.stable_ref.value,
                "generation": occurrence.generation,
                "original_due_at": occurrence.original_due_at.isoformat(),
            },
            recovery_policy="RESTART_SAFE",
            retry_policy={"delays_hours": [1 / 60, 5 / 60, 15 / 60, 1]},
            max_attempts=4,
            actor_type="SYSTEM",
            actor_id="timer-runtime",
        )


__all__ = ["TimerRuntimeWorker"]
