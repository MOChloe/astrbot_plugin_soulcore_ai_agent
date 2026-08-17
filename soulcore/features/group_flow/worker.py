"""Single lifecycle-managed worker over SQLite-owned group windows."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta

from ...contracts.group_flow import GroupFlowWindow, GroupReplyRelocationCheck
from ..profiles.service import ProfileRuntimeGate
from .ports import GroupFlowRepository, GroupInterjectionJudge, GroupReplyRelocationJudge
from .relocation import GroupReplyRelocationAction, GroupReplyRelocationDecision
from .service import GroupInterjectionDecision

GroupFlowDispatch = Callable[[GroupFlowWindow], Awaitable[object | None]]
GroupReplyRelocation = Callable[[GroupReplyRelocationCheck], Awaitable[bool]]


class GroupFlowWorker:
    def __init__(
        self,
        repository: GroupFlowRepository,
        judge: GroupInterjectionJudge,
        runtime_gate: ProfileRuntimeGate,
        *,
        relocation_judge: GroupReplyRelocationJudge | None = None,
        worker_id: str | None = None,
        maximum_parallel: int = 8,
    ) -> None:
        self.repository = repository
        self.judge = judge
        self.runtime_gate = runtime_gate
        self.relocation_judge = relocation_judge
        self.worker_id = str(worker_id or f"group-flow:{uuid.uuid4().hex}")
        self.maximum_parallel = max(1, min(32, int(maximum_parallel)))
        self._dispatch: GroupFlowDispatch | None = None
        self._relocate: GroupReplyRelocation | None = None
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._startup_error: Exception | None = None
        self.last_error = ""

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def bind_dispatch(self, dispatch: GroupFlowDispatch) -> None:
        if self._dispatch is not None:
            raise RuntimeError("group-flow dispatch is already bound")
        self._dispatch = dispatch

    def bind_relocation(self, relocate: GroupReplyRelocation) -> None:
        if self._relocate is not None:
            raise RuntimeError("group reply relocation is already bound")
        self._relocate = relocate

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._wake.set()
        self._ready.clear()
        self._startup_error = None
        self._task = asyncio.create_task(self._loop(), name="soulcore-group-flow")

    async def start_ready(self) -> None:
        """Finish recovery of current leased windows before admission opens."""

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
            self._startup_error = RuntimeError("group flow worker stopped during startup")
            self._ready.set()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def notify(self) -> None:
        self._wake.set()

    async def run_once(self) -> int:
        now = datetime.now(UTC)
        recovered = await self.repository.recover(now=now)
        relocations = await self.repository.claim_reply_relocation_checks(
            now=now,
            worker_id=self.worker_id,
            limit=self.maximum_parallel,
            lease_seconds=45,
        )
        if relocations:
            await asyncio.gather(*(self._check_relocation(check) for check in relocations))
        settled = await self.repository.settle_due_windows(now=now)
        judgments = await self.repository.claim_judging_windows(
            now=now,
            worker_id=self.worker_id,
            limit=self.maximum_parallel,
            lease_seconds=45,
        )
        if judgments:
            await asyncio.gather(*(self._judge(window) for window in judgments))
        ready = await self.repository.claim_ready_windows(
            now=datetime.now(UTC),
            worker_id=self.worker_id,
            limit=self.maximum_parallel,
            lease_seconds=300,
        )
        if ready:
            await asyncio.gather(*(self._dispatch_ready(window) for window in ready))
        return recovered + settled + len(relocations) + len(judgments) + len(ready)

    async def _check_relocation(self, check: GroupReplyRelocationCheck) -> None:
        try:
            if not await self._relocation_enabled(check):
                await self.repository.release_reply_relocation_check(check, now=datetime.now(UTC))
                return
            decision = await self._judge_relocation(check)
            await self._settle_relocation(check, decision)
        except asyncio.CancelledError:
            if self._stop.is_set():
                await asyncio.shield(
                    self.repository.release_reply_relocation_check(check, now=datetime.now(UTC))
                )
            raise
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            with suppress(Exception):
                await self.repository.release_reply_relocation_check(check, now=datetime.now(UTC))

    async def _relocation_enabled(self, check: GroupReplyRelocationCheck) -> bool:
        return self.relocation_judge is not None and await self.runtime_gate.is_enabled(
            check.profile_id, check.instance_id
        )

    async def _judge_relocation(
        self, check: GroupReplyRelocationCheck
    ) -> GroupReplyRelocationDecision:
        original, later = await asyncio.gather(
            self.repository.load_window_messages(
                check.profile_id, check.instance_id, check.fence.window_id
            ),
            self.repository.load_window_messages(
                check.profile_id, check.instance_id, check.delta_window_id
            ),
        )
        policy = await self.repository.get_group_flow_policy(check.profile_id, "group")
        assert self.relocation_judge is not None
        raw = await self.relocation_judge.judge(
            profile_id=check.profile_id,
            instance_id=check.instance_id,
            original_messages=original,
            later_messages=later,
            pending_first_text=check.pending_first_text,
            token_budget=policy.judge_token_budget,
            owner_id=check.fence.window_id,
            idempotency_key=self._relocation_key(check),
            allow_wait=not check.final_recheck,
        )
        return self._relocation_decision(raw)

    async def _settle_relocation(
        self, check: GroupReplyRelocationCheck, decision: GroupReplyRelocationDecision
    ) -> None:
        if decision.action is GroupReplyRelocationAction.RELOCATE:
            applied = self._relocate is not None and await self._relocate(check)
        else:
            applied = await self.repository.record_reply_relocation_decision(
                check,
                recheck_after_seconds=(
                    decision.wait_seconds
                    if decision.action is GroupReplyRelocationAction.WAIT
                    else 0
                ),
                error_code=decision.error_code,
                now=datetime.now(UTC),
            )
        if not applied:
            await self.repository.release_reply_relocation_check(check, now=datetime.now(UTC))

    @staticmethod
    def _relocation_key(check: GroupReplyRelocationCheck) -> str:
        stage = "final" if check.final_recheck else "initial"
        return f"group-reply-relocation:{check.fence.window_id}:{check.delta_through_message_id}:{stage}"

    async def _judge(self, window: GroupFlowWindow) -> None:
        try:
            if not await self.runtime_gate.is_enabled(window.profile_id, window.instance_id):
                return
            policy = await self.repository.get_group_flow_policy(window.profile_id, "group")
            messages = await self.repository.load_judge_messages(
                window.profile_id,
                window.instance_id,
                window.window_id,
            )
            if not await self.runtime_gate.is_enabled(window.profile_id, window.instance_id):
                return
            raw = await self.judge.judge(
                profile_id=window.profile_id,
                instance_id=window.instance_id,
                messages=messages,
                token_budget=policy.judge_token_budget,
                owner_id=window.window_id,
                idempotency_key=f"group-judge:{window.window_id}:v{window.version}",
            )
            if not await self.runtime_gate.is_enabled(window.profile_id, window.instance_id):
                return
            decision = self._decision(raw)
            await self.repository.record_judgment(
                window,
                suitable=decision.suitable and not decision.error_code,
                error_code=decision.error_code,
                now=datetime.now(UTC),
            )
        except asyncio.CancelledError:
            if self._stop.is_set():
                await asyncio.shield(
                    self.repository.release_judgment(window, now=datetime.now(UTC))
                )
            raise
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"

    async def _dispatch_ready(self, window: GroupFlowWindow) -> None:
        try:
            if not await self.runtime_gate.is_enabled(window.profile_id, window.instance_id):
                return
            if self._dispatch is None:
                raise RuntimeError("group-flow dispatch is unavailable")
            result = await self._dispatch(window)
            if result is False:
                if not await self.runtime_gate.is_enabled(window.profile_id, window.instance_id):
                    return
                await self.repository.release_ready(
                    window,
                    retry_at=datetime.now(UTC) + timedelta(minutes=1),
                    reason="dispatch_deferred",
                )
        except asyncio.CancelledError:
            if self._stop.is_set():
                await asyncio.shield(
                    self.repository.release_ready(
                        window,
                        retry_at=datetime.now(UTC),
                        reason="dispatch_cancelled_for_shutdown",
                    )
                )
            raise
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            if not await self.runtime_gate.is_enabled(window.profile_id, window.instance_id):
                return
            await self.repository.release_ready(
                window,
                retry_at=datetime.now(UTC) + timedelta(minutes=1),
                reason=f"dispatch_failed:{type(exc).__name__}",
            )

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

    @staticmethod
    def _decision(value: GroupInterjectionDecision) -> GroupInterjectionDecision:
        return GroupInterjectionDecision(
            suitable=value.suitable,
            backend_id=value.backend_id,
            error_code=value.error_code,
        )

    @staticmethod
    def _relocation_decision(value: object) -> GroupReplyRelocationDecision:
        if isinstance(value, GroupReplyRelocationDecision):
            return value
        return GroupReplyRelocationDecision(error_code="INVALID_DECISION")


__all__ = ["GroupFlowDispatch", "GroupFlowWorker", "GroupReplyRelocation"]
