"""Precise lifecycle worker for durable Main Core expression output."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress

from ...contracts.models import MessageRetractionAction, OutboxItem
from ...shared.time import utcnow
from .ports import OutboxRepositoryPort

ExpressionDispatch = Callable[[OutboxItem], Awaitable[bool]]
RetractionDispatch = Callable[[MessageRetractionAction], Awaitable[bool]]


class ExpressionOutboxWorker:
    """Dispatch expression items at their SQLite-owned due time.

    The single global wake event only accelerates re-querying.  SQLite remains
    authoritative for ordering, eligibility, and the next due timestamp.
    """

    def __init__(
        self,
        repository: OutboxRepositoryPort,
        dispatch: ExpressionDispatch,
        dispatch_retraction: RetractionDispatch | None = None,
        *,
        batch_limit: int = 50,
        idle_poll_seconds: float = 5.0,
        unavailable_retry_seconds: float = 0.5,
    ) -> None:
        self.repository = repository
        self.dispatch = dispatch
        self.dispatch_retraction = dispatch_retraction
        self.batch_limit = max(1, min(200, int(batch_limit)))
        self.idle_poll_seconds = max(0.25, float(idle_poll_seconds))
        self.unavailable_retry_seconds = max(0.25, min(1.0, float(unavailable_retry_seconds)))
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self._ready = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._made_progress = False
        self._recovery_complete = False
        self.last_error = ""

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._recovery_complete = False
        self._ready.clear()
        self._spawn_loop()

    async def start_ready(self) -> None:
        """Recover durable interruption state before admitting other workers."""

        if self.running:
            await self._ready.wait()
            return
        self._stop.clear()
        self._recovery_complete = False
        self._ready.clear()
        try:
            await self._ensure_recovered()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise
        self.last_error = ""
        self._spawn_loop()

    def _spawn_loop(self) -> None:
        self._wake.set()
        self._task = asyncio.create_task(self._loop(), name="soulcore-expression-outbox")

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        task, self._task = self._task, None
        if task is None or task is asyncio.current_task():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def notify(self) -> None:
        self._wake.set()

    async def run_once(self) -> int:
        """Dispatch one ordered snapshot without letting one item stop peers."""

        recovered_now = await self._ensure_recovered()
        if not recovered_now:
            await self._recover_pages()
        items = list(
            await self.repository.list_due_expression_outbox(now=utcnow(), limit=self.batch_limit)
        )
        retractions = (
            list(
                await self.repository.list_due_retraction_actions(
                    now=utcnow(), limit=self.batch_limit
                )
            )
            if self.dispatch_retraction is not None
            else []
        )
        self.last_error = ""
        self._made_progress = False
        for action in retractions:
            try:
                progressed = bool(await self.dispatch_retraction(action))
                self._made_progress = progressed or self._made_progress
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
        for item in items:
            try:
                progressed = bool(await self.dispatch(item))
                self._made_progress = progressed or self._made_progress
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
        return len(items) + len(retractions)

    async def _ensure_recovered(self) -> bool:
        if self._recovery_complete:
            return False
        await self._recover_pages()
        self._recovery_complete = True
        self._ready.set()
        return True

    async def _recover_pages(self) -> None:
        while await self.repository.recover_pending_expression_interruptions() >= 100:
            pass
        await self.repository.recover_sending_retraction_actions()

    async def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.clear()
            try:
                await self.run_once()
                wait_seconds = await self._next_wait_seconds(made_progress=self._made_progress)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                wait_seconds = self.unavailable_retry_seconds
            if self._stop.is_set():
                break
            if self._wake.is_set() or wait_seconds <= 0:
                continue
            with suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=wait_seconds)

    async def _next_wait_seconds(self, *, made_progress: bool = False) -> float:
        due_at = await self.repository.next_expression_outbox_due_at()
        retraction_due = await self.repository.next_retraction_action_due_at()
        if retraction_due is not None and (due_at is None or retraction_due < due_at):
            due_at = retraction_due
        if due_at is None:
            return self.idle_poll_seconds
        remaining = (due_at - utcnow()).total_seconds()
        if remaining > 0:
            return remaining
        if made_progress:
            return 0.0
        return self.unavailable_retry_seconds


__all__ = ["ExpressionDispatch", "ExpressionOutboxWorker", "RetractionDispatch"]
