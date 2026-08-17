"""Lifecycle-managed release worker for the durable inbound recall grace."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta

from .domain import InboundRecallHold, InboundRecallTarget

InboundRecallDispatch = Callable[[InboundRecallHold], Awaitable[object | None]]
InboundRecallSettlementDispatch = Callable[[InboundRecallTarget], Awaitable[object | None]]


class InboundRecallGraceWorker:
    def __init__(
        self,
        repository,
        *,
        worker_id: str | None = None,
        maximum_parallel: int = 16,
    ) -> None:
        self.repository = repository
        self.worker_id = str(worker_id or f"inbound-recall:{uuid.uuid4().hex}")
        self.maximum_parallel = max(1, min(32, int(maximum_parallel)))
        self._dispatch: InboundRecallDispatch | None = None
        self._recall_dispatch: InboundRecallSettlementDispatch | None = None
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._startup_error: Exception | None = None
        self.last_error = ""

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def bind_dispatch(self, dispatch: InboundRecallDispatch) -> None:
        if self._dispatch is not None:
            raise RuntimeError("inbound recall dispatch is already bound")
        self._dispatch = dispatch

    def bind_recall_dispatch(self, dispatch: InboundRecallSettlementDispatch) -> None:
        if self._recall_dispatch is not None:
            raise RuntimeError("inbound recall settlement dispatch is already bound")
        self._recall_dispatch = dispatch

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._wake.set()
        self._ready.clear()
        self._startup_error = None
        self._task = asyncio.create_task(self._loop(), name="soulcore-inbound-recall-grace")

    async def start_ready(self) -> None:
        """Finish recovery of current leased holds before admission opens."""

        self.start()
        await self._ready.wait()
        if self._startup_error is None:
            return
        task, self._task = self._task, None
        if task is not None:
            await task
        raise self._startup_error

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        task, self._task = self._task, None
        if not self._ready.is_set():
            self._startup_error = RuntimeError("inbound recall worker stopped during startup")
            self._ready.set()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def notify(self) -> None:
        self._wake.set()

    async def run_once(self) -> int:
        now = datetime.now(UTC)
        recovered = await self.repository.recover(now=now)
        expired = await self.repository.cleanup_expired_receipts(now=now)
        targets = await self.repository.claim_matching_receipts(limit=self.maximum_parallel)
        if targets:
            if self._recall_dispatch is None:
                raise RuntimeError("inbound recall settlement dispatch is unavailable")
            await asyncio.gather(*(self._dispatch_target(target) for target in targets))
        holds = await self.repository.claim_due(
            now=now,
            worker_id=self.worker_id,
            limit=self.maximum_parallel,
            lease_seconds=90,
        )
        if holds:
            await asyncio.gather(*(self._dispatch_hold(hold) for hold in holds))
        return int(recovered) + int(expired) + len(targets) + len(holds)

    async def _dispatch_target(self, target: InboundRecallTarget) -> None:
        try:
            assert self._recall_dispatch is not None
            await self._recall_dispatch(target)
        except asyncio.CancelledError:
            if self._stop.is_set():
                await asyncio.shield(
                    self.repository.retry_receipt(target.receipt_id, now=datetime.now(UTC))
                )
            raise
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            await self.repository.retry_receipt(target.receipt_id, now=datetime.now(UTC))
            self.notify()

    async def _dispatch_hold(self, hold: InboundRecallHold) -> None:
        try:
            if self._dispatch is None:
                raise RuntimeError("inbound recall grace dispatch is unavailable")
            await self._dispatch(hold)
        except asyncio.CancelledError:
            if self._stop.is_set():
                await asyncio.shield(
                    self.repository.defer_claim_for_shutdown(hold, now=datetime.now(UTC))
                )
            raise
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            await self.repository.retry_claim(
                hold,
                retry_at=datetime.now(UTC) + timedelta(seconds=5),
                error=type(exc).__name__,
            )
            self.notify()

    async def _loop(self) -> None:
        try:
            await self.repository.recover(
                now=datetime.now(UTC),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._startup_error = exc
            raise
        finally:
            self._ready.set()
        while not self._stop.is_set():
            self._wake.clear()
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            timeout = await self._next_wait_seconds()
            if self._wake.is_set():
                continue
            with suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=timeout)

    async def _next_wait_seconds(self) -> float:
        due = await self.repository.next_due_at()
        if due is None:
            return 5.0
        return max(0.05, min(5.0, (due - datetime.now(UTC)).total_seconds()))


__all__ = [
    "InboundRecallDispatch",
    "InboundRecallGraceWorker",
    "InboundRecallSettlementDispatch",
]
