"""Durable wake-slot materializer for the five background authors."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import datetime
from typing import Any

from ...shared.time import utcnow
from .ports import (
    BackgroundSchedulerRepositoryPort,
    BackgroundTaskRepositoryPort,
)
from .prewarm import ProactiveFrameSourceKind, proactive_frame_timing
from .proactive_sources import (
    PREDICTABLE_PROACTIVE_SOURCE_LIMIT,
    PredictableProactiveSource,
)

_PROACTIVE_MATERIALIZE_LIMIT = 12
ContactPolicyEligibility = Callable[[PredictableProactiveSource], Awaitable[bool]]


class BackgroundSchedulerWorker:
    def __init__(
        self,
        *,
        repository: BackgroundSchedulerRepositoryPort,
        ai_tasks: BackgroundTaskRepositoryPort,
        contact_policy_eligibility: ContactPolicyEligibility | None = None,
        poll_seconds: float = 15.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.ai_tasks = ai_tasks
        self.contact_policy_eligibility = contact_policy_eligibility
        self.poll_seconds = max(1.0, float(poll_seconds))
        self.clock = clock or utcnow
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._startup_task: asyncio.Task[None] | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._generation = 0
        self.last_error: str | None = None

    @property
    def running(self) -> bool:
        loop_running = self._task is not None and not self._task.done()
        startup_running = self._startup_task is not None and not self._startup_task.done()
        return loop_running or startup_running

    def start(self) -> None:
        if self.running:
            return
        self._generation += 1
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self._task = asyncio.create_task(
            self._loop(),
            name="soulcore-background-scheduler",
        )

    async def start_ready(self) -> None:
        """Complete one durable materialization pass before publishing readiness."""

        async with self._lifecycle_lock:
            if self._task is not None and not self._task.done():
                return
            startup = self._startup_task
            if startup is None or startup.done():
                self._generation += 1
                generation = self._generation
                self._stop = asyncio.Event()
                self._wake = asyncio.Event()
                startup = asyncio.create_task(
                    self._start_sequence(generation),
                    name="soulcore-background-scheduler-startup",
                )
                self._startup_task = startup
                startup.add_done_callback(self._clear_startup_task)
        await startup

    async def _start_sequence(self, generation: int) -> None:
        try:
            await self.run_once()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise
        async with self._lifecycle_lock:
            if generation != self._generation or self._stop.is_set():
                raise asyncio.CancelledError
            self.last_error = None
            self._task = asyncio.create_task(
                self._loop(wait_before_first=True),
                name="soulcore-background-scheduler",
            )

    def _clear_startup_task(self, task: asyncio.Task[None]) -> None:
        if self._startup_task is task:
            self._startup_task = None

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            self._generation += 1
            self._stop.set()
            self._wake.set()
            startup = self._startup_task
            task = self._task
            self._task = None
            for owned in (startup, task):
                if owned is not None and not owned.done():
                    owned.cancel()
        owned_tasks = tuple(
            owned
            for owned in (startup, task)
            if owned is not None and owned is not asyncio.current_task()
        )
        if owned_tasks:
            await asyncio.gather(*owned_tasks, return_exceptions=True)

    def notify(self) -> None:
        self._wake.set()

    async def run_once(self) -> int:
        await self.repository.ensure_all_instances()
        proactive = await self._materialize_predictable_proactive_frames(now=self.clock())
        tasks = await self.ai_tasks.materialize_due_background_tasks(limit=12)
        return proactive + len(tasks)

    async def _materialize_predictable_proactive_frames(self, *, now: datetime) -> int:
        list_sources = getattr(
            self.repository,
            "list_predictable_proactive_sources",
            None,
        )
        materialize = getattr(
            self.ai_tasks,
            "materialize_proactive_frame_task",
            None,
        )
        if not callable(list_sources) or not callable(materialize):
            return 0
        sources = await list_sources(
            now=now,
            limit=PREDICTABLE_PROACTIVE_SOURCE_LIMIT,
        )
        due: list[tuple[datetime, PredictableProactiveSource, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for source in sources:
            candidate = source.candidate
            key = (
                candidate.profile_id,
                candidate.instance_id,
                candidate.source_ref,
            )
            if key in seen:
                continue
            seen.add(key)
            timing = proactive_frame_timing(
                candidate.source_ref,
                candidate.seed_planned_main_core_at or candidate.planned_main_core_at,
                admitted_at=now,
                effective_main_core_at=candidate.planned_main_core_at,
            )
            if timing.frame_end_at <= now:
                due.append((timing.frame_end_at, source, timing))
        due.sort(
            key=lambda item: (
                item[0],
                item[1].candidate.planned_main_core_at,
                item[1].candidate.source_ref,
            )
        )
        created = 0
        for _frame_end_at, source, timing in due:
            if created >= _PROACTIVE_MATERIALIZE_LIMIT:
                break
            candidate = source.candidate
            if candidate.source_kind is ProactiveFrameSourceKind.CONTACT:
                if self.contact_policy_eligibility is None:
                    continue
                if not await self.contact_policy_eligibility(source):
                    continue
            result = await materialize(
                candidate.profile_id,
                candidate.instance_id,
                source_ref=candidate.source_ref,
                planned_main_core_at=candidate.planned_main_core_at,
                frame_end_at=timing.frame_end_at,
                deadline_at=timing.deadline_at,
                now=now,
            )
            if str(result.get("outcome") or "").upper() == "CREATED":
                created += 1
        return created

    async def _loop(self, *, wait_before_first: bool = False) -> None:
        if wait_before_first and not self._stop.is_set():
            with suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=self.poll_seconds)
            self._wake.clear()
        while not self._stop.is_set():
            try:
                await self.run_once()
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            with suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=self.poll_seconds)
            self._wake.clear()


__all__ = ["BackgroundSchedulerWorker"]
