"""Runtime lifecycle with ordered startup and reverse rollback."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, NoReturn

from .supervisor import TaskSupervisor

AsyncCleanup = Callable[[], Awaitable[object] | object]


@dataclass(slots=True)
class RuntimeLifecycle:
    """Own cleanup order independently from the AstrBot plugin instance."""

    supervisor: TaskSupervisor = field(default_factory=TaskSupervisor)
    _cleanups: list[AsyncCleanup] = field(default_factory=list)
    _close_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    started: bool = False
    closing: bool = False

    def add_cleanup(self, cleanup: AsyncCleanup) -> None:
        self._cleanups.append(cleanup)

    async def rollback(self) -> None:
        await self._close(run_all=True)

    async def close(self) -> None:
        await self._close(run_all=True)

    async def _close(self, *, run_all: bool) -> None:
        async with self._close_lock:
            self.closing = True
            failures: list[Exception] = []
            try:
                try:
                    await self.supervisor.close()
                except Exception as exc:
                    failures.append(exc)
                for index in range(len(self._cleanups) - 1, -1, -1):
                    cleanup = self._cleanups[index]
                    try:
                        result = cleanup()
                        if inspect.isawaitable(result):
                            await result
                    except Exception as exc:
                        failures.append(exc)
                        if not run_all:
                            break
                    else:
                        self._cleanups.pop(index)
                self.started = False
                if failures:
                    raise ExceptionGroup("SoulCore runtime cleanup failed", failures)
            finally:
                self.closing = False


@dataclass(slots=True)
class BootstrapRollbackOwner:
    """Retain failed assembly resources until a later lifecycle retry succeeds."""

    container: Any
    runtime_fence: Any

    async def close(self) -> None:
        await self.container.close()
        self.runtime_fence.close()


class BootstrapRollbackError(RuntimeError):
    def __init__(
        self,
        startup_error: Exception,
        cleanup_error: Exception,
        owner: BootstrapRollbackOwner,
    ) -> None:
        super().__init__(
            "SoulCore bootstrap failed and rollback remains incomplete: "
            f"{type(startup_error).__name__}; cleanup={type(cleanup_error).__name__}"
        )
        self.startup_error = startup_error
        self.cleanup_error = cleanup_error
        self.owner = owner


async def rollback_bootstrap(
    owner: BootstrapRollbackOwner,
    startup_error: BaseException,
) -> NoReturn:
    """Close an assembly owner or attach it to the propagated failure for retry."""

    cleanup = asyncio.create_task(
        owner.close(),
        name="soulcore-bootstrap-rollback",
    )
    current = asyncio.current_task()
    assert current is not None
    observed_cancellations = current.cancelling()
    deferred_cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(cleanup)
            break
        except asyncio.CancelledError as exc:
            current_cancellations = current.cancelling()
            if current_cancellations > observed_cancellations:
                deferred_cancellation = deferred_cancellation or exc
                observed_cancellations = current_cancellations
                continue
            _attach_rollback_owner(exc, owner)
            raise exc from startup_error
        except Exception as cleanup_error:
            propagated = deferred_cancellation or startup_error
            _attach_rollback_owner(propagated, owner)
            if isinstance(propagated, asyncio.CancelledError):
                raise propagated from cleanup_error
            if isinstance(startup_error, Exception):
                raise BootstrapRollbackError(
                    startup_error,
                    cleanup_error,
                    owner,
                ) from startup_error
            raise propagated from cleanup_error
    if deferred_cancellation is not None:
        raise deferred_cancellation
    raise startup_error


def _attach_rollback_owner(
    error: BaseException,
    owner: BootstrapRollbackOwner,
) -> None:
    error.rollback_owner = owner  # type: ignore[attr-defined]


__all__ = [
    "BootstrapRollbackError",
    "BootstrapRollbackOwner",
    "RuntimeLifecycle",
    "rollback_bootstrap",
]
