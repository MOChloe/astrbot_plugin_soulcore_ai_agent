from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

T = TypeVar("T")
TaskFactory = Callable[[], Awaitable[object]]


class TaskSupervisor:
    """Own named runtime tasks and provide idempotent start/stop operations."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[object]] = set()
        self._closing = False

    @property
    def closing(self) -> bool:
        return self._closing

    @property
    def active_names(self) -> frozenset[str]:
        return frozenset(task.get_name() for task in self._tasks if not task.done())

    def create(self, awaitable: Awaitable[T], *, name: str) -> asyncio.Task[T]:
        if self._closing:
            raise RuntimeError("task supervisor is closing")
        task = asyncio.create_task(awaitable, name=name)
        self._tasks.add(task)  # type: ignore[arg-type]
        task.add_done_callback(self._discard)
        return task

    def ensure(self, factories: Iterable[tuple[str, TaskFactory]]) -> None:
        active = self.active_names
        for name, factory in factories:
            if name not in active:
                self.create(factory(), name=name)

    async def close(self) -> None:
        self._closing = True
        current = asyncio.current_task()
        tasks = [task for task in tuple(self._tasks) if task is not current and not task.done()]
        await self._cancel(tasks)
        self._tasks.clear()

    async def _cancel(self, tasks: Iterable[asyncio.Task[object]]) -> None:
        pending = list(tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def _discard(self, task: asyncio.Task[object]) -> None:
        self._tasks.discard(task)
