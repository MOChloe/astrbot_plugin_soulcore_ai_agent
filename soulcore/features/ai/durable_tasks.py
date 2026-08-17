"""Persistent, fenced AI task manager for long-running background work."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, Protocol

from ...contracts.ai_models import AIErrorCode, AIInvocationError
from ..profiles.service import ProfileRuntimeDisabled, ProfileRuntimeGate
from .diagnostics import classify_generic_error
from .durable_task_control import (
    AITaskCancelRequested,
    AITaskControl,
    AITaskLeaseLost,
    AITaskPauseRequested,
    _current_task_id,
    current_durable_ai_task_id,
)
from .durable_task_runtime import (
    ActiveTaskRuntime,
    DurableTaskRuntimeRepository,
    PrerequisiteTaskClaim,
    PrerequisiteTaskClaimOutcome,
    stop_runtime_watchers,
)
from .work_taxonomy import durable_task_owns_workflow
from .workflow_context import AIWorkContext, bind_ai_work_context


@dataclass(frozen=True, slots=True)
class _PrerequisiteAcquisition:
    claim: PrerequisiteTaskClaim
    runtime: ActiveTaskRuntime | None = None


async def execute_prerequisite_task(manager: Any, task_id: int) -> PrerequisiteTaskClaim:
    requester_task_id = current_durable_ai_task_id()
    if requester_task_id is None or manager._closed:
        return _not_claimable()
    task_id = int(task_id)
    if task_id < 1 or task_id == int(requester_task_id):
        return _not_claimable()

    acquired = await _acquire_prerequisite(manager, task_id, int(requester_task_id))
    claim = acquired.claim
    if claim.outcome is PrerequisiteTaskClaimOutcome.ACTIVE:
        return await _reuse_active_prerequisite(manager, task_id, acquired)
    if claim.outcome is not PrerequisiteTaskClaimOutcome.CLAIMED:
        return claim
    return await _run_claimed_prerequisite(manager, task_id, acquired)


async def _acquire_prerequisite(
    manager: Any,
    task_id: int,
    requester_task_id: int,
) -> _PrerequisiteAcquisition:
    async with manager._active_runtime_lock:
        requester_runtime = manager._active_runtimes.get(requester_task_id)
        if not _requester_is_eligible(manager, requester_runtime, requester_task_id):
            return _PrerequisiteAcquisition(_not_claimable())
        requester_scope = (
            str(requester_runtime.task.get("profile_id") or ""),
            str(requester_runtime.task.get("instance_id") or ""),
        )
        if requester_scope in manager._blocked_instance_scopes:
            return _PrerequisiteAcquisition(_not_claimable())

        control = requester_runtime.control
        claim = await manager.repository.claim_ai_task_prerequisite(
            manager.worker_id,
            task_id,
            requester_task_id,
            int(control.lease_token),
            lease_seconds=manager.lease_seconds,
        )
        child_runtime: ActiveTaskRuntime | None = None
        if claim.outcome is PrerequisiteTaskClaimOutcome.CLAIMED:
            if claim.task is None:
                raise RuntimeError("claimed prerequisite task row is missing")
            child_runtime = ActiveTaskRuntime(claim.task)
            manager._active_runtimes[task_id] = child_runtime
        elif claim.outcome is PrerequisiteTaskClaimOutcome.ACTIVE:
            child_runtime = manager._active_runtimes.get(task_id)
        return _PrerequisiteAcquisition(claim, child_runtime)


def _requester_is_eligible(manager: Any, runtime: Any, requester_task_id: int) -> bool:
    if runtime is None:
        return False
    control = runtime.control
    if control is None:
        return False
    if runtime.stopped.is_set():
        return False
    if runtime.foreground_preempted:
        return False
    if control.task_id != requester_task_id:
        return False
    if control.worker_id != manager.worker_id:
        return False
    return control.requested_status == "RUNNING"


async def _reuse_active_prerequisite(
    manager: Any,
    task_id: int,
    acquired: _PrerequisiteAcquisition,
) -> PrerequisiteTaskClaim:
    runtime = acquired.runtime
    if runtime is not None and not runtime.stopped.is_set():
        await runtime.stopped.wait()
    current = await manager.repository.get_ai_task(task_id)
    return PrerequisiteTaskClaim(acquired.claim.outcome, current or acquired.claim.task)


async def _run_claimed_prerequisite(
    manager: Any,
    task_id: int,
    acquired: _PrerequisiteAcquisition,
) -> PrerequisiteTaskClaim:
    runtime = acquired.runtime
    task = acquired.claim.task
    assert runtime is not None
    assert task is not None
    try:
        await manager._execute(task, runtime)
    finally:
        async with manager._active_runtime_lock:
            runtime.stopped.set()
            if manager._active_runtimes.get(task_id) is runtime:
                manager._active_runtimes.pop(task_id, None)
    current = await manager.repository.get_ai_task(task_id)
    return PrerequisiteTaskClaim(acquired.claim.outcome, current or task)


def _not_claimable() -> PrerequisiteTaskClaim:
    return PrerequisiteTaskClaim(PrerequisiteTaskClaimOutcome.NOT_CLAIMABLE)


AITaskExecutor = Callable[[dict[str, Any], "AITaskControl"], Awaitable[dict[str, Any] | None]]


class FileTaskReconciler(Protocol):
    async def reconcile_terminal_file_jobs(self) -> object: ...


class DurableAITaskManager:
    def __init__(
        self,
        repository: DurableTaskRuntimeRepository,
        *,
        worker_id: str | None = None,
        executors: dict[str, AITaskExecutor] | None = None,
        lease_seconds: int = 300,
        poll_seconds: int = 5,
        concurrency: int = 1,
        runtime_gate: ProfileRuntimeGate,
        file_reconciler: FileTaskReconciler | None = None,
    ) -> None:
        self.repository = repository
        self.worker_id = worker_id or f"embedded:{uuid.uuid4().hex}"
        self.executors = {
            str(name).upper(): executor for name, executor in (executors or {}).items()
        }
        self.lease_seconds = max(30, int(lease_seconds))
        self.poll_seconds = max(1, int(poll_seconds))
        self.concurrency = max(1, int(concurrency))
        self.runtime_gate = runtime_gate
        self.file_reconciler = file_reconciler
        self._loop_task: asyncio.Task[Any] | None = None
        self._ready = asyncio.Event()
        self._startup_error: Exception | None = None
        self._active: set[asyncio.Task[Any]] = set()
        self._active_runtimes: dict[int, ActiveTaskRuntime] = {}
        self._active_runtime_lock = asyncio.Lock()
        self._blocked_instance_scopes: set[tuple[str, str]] = set()
        self._closed = False
        self.last_error = ""
        self._last_cleanup_monotonic = 0.0

    @property
    def running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    def register_executor(self, task_type: str, executor: AITaskExecutor) -> None:
        self.executors[str(task_type).upper()] = executor

    def start(self) -> None:
        if self.running:
            return
        self._closed = False
        self._ready.clear()
        self._startup_error = None
        self._loop_task = asyncio.create_task(self._loop(), name="soulcore-durable-ai-tasks")

    async def start_ready(self) -> None:
        """Complete startup recovery before the application publishes readiness."""

        self.start()
        await self._ready.wait()
        if self._startup_error is None:
            return
        loop, self._loop_task = self._loop_task, None
        if loop is not None:
            await loop
        raise self._startup_error

    async def stop(self) -> None:
        self._closed = True
        loop, self._loop_task = self._loop_task, None
        if not self._ready.is_set():
            self._startup_error = RuntimeError("durable AI task worker stopped during startup")
            self._ready.set()
        if loop is not None:
            loop.cancel()
        for task in tuple(self._active):
            task.cancel()
        if loop is not None:
            with suppress(asyncio.CancelledError):
                await loop
        if self._active:
            await asyncio.gather(*tuple(self._active), return_exceptions=True)

    async def _loop(self) -> None:
        try:
            await self.repository.recover_expired_ai_tasks(current_worker_id=self.worker_id)
            if self.file_reconciler is not None:
                await self.file_reconciler.reconcile_terminal_file_jobs()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._startup_error = exc
            raise
        finally:
            self._ready.set()
        while not self._closed:
            try:
                # Recovery is not a startup-only migration. A local executor can
                # lose its heartbeat while the manager loop remains alive, and
                # a new boot owns every lease left by the previous boot under
                # the process-wide runtime fence.
                await self.repository.recover_expired_ai_tasks(current_worker_id=self.worker_id)
                # The service loop must keep polling so a newly freed slot is
                # replenished even while another long task is still running.
                await self.run_once(wait=False)
                if self.file_reconciler is not None:
                    await self.file_reconciler.reconcile_terminal_file_jobs()
                self.last_error = ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(self.poll_seconds)

    async def run_once(self, *, wait: bool = True) -> int:
        claimed_count = 0
        if not self.executors:
            return 0
        try:
            while True:
                claimed: list[dict[str, Any]] = []
                # Capacity calculation, durable claim, and in-process
                # registration are one critical section.  The service loop and
                # prewarmer nudge may call run_once concurrently.
                async with self._active_runtime_lock:
                    self._active = {task for task in self._active if not task.done()}
                    available = max(0, self.concurrency - len(self._active))
                    if available > 0:
                        claimed = await self.repository.claim_ai_tasks(
                            self.worker_id,
                            limit=available,
                            lease_seconds=self.lease_seconds,
                            task_types=tuple(self.executors),
                        )
                        claimed_count += len(claimed)
                        for row in claimed:
                            runtime = ActiveTaskRuntime(row)
                            task_id = int(row["task_id"])
                            task = asyncio.create_task(
                                self._execute(row, runtime),
                                name=(f"soulcore-ai-task:{task_id}:{row['lease_token']}"),
                            )
                            self._active.add(task)
                            self._active_runtimes[task_id] = runtime
                            task.add_done_callback(
                                lambda finished, claimed_task_id=task_id, claimed_runtime=runtime: (
                                    self._discard_active_runtime(
                                        claimed_task_id,
                                        claimed_runtime,
                                        finished,
                                    )
                                )
                            )
                if not wait or not self._active:
                    break
                # Refill immediately when one slot is released instead of idling
                # behind the slowest sibling in an all-task gather.
                await asyncio.wait(tuple(self._active), return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            # Preserve the previous lease-release guarantee: cancelling a
            # waiting run_once also cancels its claimed workers and lets each
            # _execute() release its fenced lease before control returns.
            active = tuple(self._active)
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
            raise
        await self._cleanup_if_due()
        return claimed_count

    async def execute_prerequisite_task(self, task_id: int) -> PrerequisiteTaskClaim:
        """Run one exact proactive-frame task inside its durable requester's slot.

        A running ``MAIN_CORE``/``TIMER_RUN`` already occupies one manager slot.
        Registering the prerequisite for interruption without adding another
        top-level ``_active`` task lets that slot execute the two operations in
        sequence, including when configured concurrency is one.
        """

        return await execute_prerequisite_task(self, task_id)

    def _discard_active_runtime(
        self,
        task_id: int,
        runtime: ActiveTaskRuntime,
        finished: asyncio.Task[Any],
    ) -> None:
        runtime.stopped.set()
        self._active.discard(finished)
        if self._active_runtimes.get(task_id) is runtime:
            self._active_runtimes.pop(task_id, None)

    async def interrupt_background_tasks(self, profile_id: str, instance_id: str) -> int:
        """Stop local background executors after their durable cancel was committed.

        The caller first establishes the SQLite foreground fence.  This method
        closes the remaining same-process heartbeat window and does not return
        until every matching provider/executor coroutine has stopped.
        """

        targets = await self._interrupt_instance_runtimes(
            profile_id,
            instance_id,
            task_types={"BACKGROUND_AUTHOR"},
        )
        if targets:
            await asyncio.gather(*(runtime.stopped.wait() for runtime in targets))
        return len(targets)

    async def interrupt_instance_tasks(self, profile_id: str, instance_id: str) -> int:
        """Stop every claimed durable AI task owned by one resetting instance."""

        targets = await self._interrupt_instance_runtimes(profile_id, instance_id)
        if targets:
            await asyncio.gather(*(runtime.stopped.wait() for runtime in targets))
        return len(targets)

    @asynccontextmanager
    async def quiesce_instance(
        self,
        profile_id: str,
        instance_id: str,
    ) -> AsyncIterator[None]:
        """Keep one instance from starting durable AI work during its reset."""

        scope = (str(profile_id), str(instance_id))
        self._blocked_instance_scopes.add(scope)
        await self.interrupt_instance_tasks(*scope)
        try:
            yield
        finally:
            self._blocked_instance_scopes.discard(scope)

    async def _interrupt_instance_runtimes(
        self,
        profile_id: str,
        instance_id: str,
        *,
        task_types: set[str] | None = None,
    ) -> list[ActiveTaskRuntime]:
        async with self._active_runtime_lock:
            targets = [
                runtime
                for runtime in tuple(self._active_runtimes.values())
                if (
                    task_types is None
                    or str(runtime.task.get("task_type") or "").upper() in task_types
                )
                and str(runtime.task.get("profile_id") or "") == str(profile_id)
                and str(runtime.task.get("instance_id") or "") == str(instance_id)
                and not runtime.stopped.is_set()
            ]
            for runtime in targets:
                runtime.preempt_for_foreground()
        return targets

    async def _prepare_claimed_task(
        self,
        task: dict[str, Any],
        control: AITaskControl,
    ) -> tuple[dict[str, Any], AITaskControl, AITaskExecutor] | None:
        profile_id = str(task.get("profile_id") or "")
        instance_id = str(task.get("instance_id") or "")
        if (profile_id, instance_id) in self._blocked_instance_scopes:
            await self.repository.release_ai_task(
                control.task_id,
                control.lease_token,
                self.worker_id,
                reason="instance_reset_in_progress",
            )
            return None
        if not await self.runtime_gate.is_enabled(profile_id, instance_id):
            await self.repository.release_ai_task(
                control.task_id,
                control.lease_token,
                self.worker_id,
                reason="profile_disabled_waiting",
            )
            return None
        task_type = str(task.get("task_type") or "").upper()
        if durable_task_owns_workflow(task_type):
            task = {**task, "workflow_id": None}
        else:
            workflow = await self.repository.ensure_ai_task_workflow(int(task["task_id"]))
            if workflow is not None:
                task = {**task, "workflow_id": int(workflow["workflow_id"])}
                control = AITaskControl(self.repository, task, self.worker_id, self.lease_seconds)
        executor = self.executors.get(task_type)
        if executor is None:
            raise RuntimeError(f"no executor registered for {task['task_type']}")
        return task, control, executor

    async def _execute(
        self,
        task: dict[str, Any],
        runtime: ActiveTaskRuntime | None = None,
    ) -> None:
        runtime = runtime or ActiveTaskRuntime(task)
        control = AITaskControl(self.repository, task, self.worker_id, self.lease_seconds)
        # A foreground interrupter that arrived while claim I/O was in flight
        # is queued on this same fair lock before a newly registered executor.
        # It can therefore mark the runtime before provider preparation begins.
        async with self._active_runtime_lock:
            runtime.attach(control)
        profile_id = str(task.get("profile_id") or "")
        instance_id = str(task.get("instance_id") or "")
        executor: AITaskExecutor | None = None
        heartbeat: asyncio.Task[Any] | None = None
        gate_watch: asyncio.Task[Any] | None = None
        execution: asyncio.Task[Any] | None = None
        try:
            if runtime.foreground_preempted:
                await self._settle_control_interrupt(task, control)
                return
            prepared = await self._prepare_claimed_task(task, control)
            if prepared is None:
                return
            task, control, executor = prepared
            runtime.attach(control)
            if runtime.foreground_preempted:
                await self._settle_control_interrupt(task, control)
                return
            heartbeat = asyncio.create_task(
                self._heartbeat_loop(control),
                name=f"soulcore-ai-heartbeat:{task['task_id']}",
            )
            gate_watch = asyncio.create_task(
                self._runtime_gate_loop(control, profile_id, instance_id),
                name=f"soulcore-ai-profile-gate:{task['task_id']}",
            )
            execution = asyncio.create_task(self._execute_scoped(task, control, executor))
            interrupted = await self._wait_for_execution(
                task,
                control,
                execution,
                heartbeat,
                gate_watch,
            )
            if interrupted:
                return
            await self._complete_execution(task, control, execution.result())
        except AITaskPauseRequested:
            await self.repository.acknowledge_pause_ai_task(
                control.task_id,
                control.lease_token,
                self.worker_id,
                checkpoint=control.checkpoint,
            )
            return
        except AITaskCancelRequested:
            # The executor observed cancellation at an explicit safe point.
            await self.repository.acknowledge_cancel_ai_task(
                control.task_id, control.lease_token, self.worker_id
            )
            return
        except AITaskLeaseLost:
            return
        except ProfileRuntimeDisabled:
            await self.repository.release_ai_task(
                control.task_id,
                control.lease_token,
                self.worker_id,
                reason="profile_disabled_waiting",
            )
            return
        except asyncio.CancelledError:
            await self._release_cancelled_execution(control, execution)
            raise
        except Exception as exc:
            await self._fail_execution(task, control, exc, executor_exists=executor is not None)
        finally:
            await stop_runtime_watchers(runtime, heartbeat, gate_watch)

    async def _release_cancelled_execution(
        self,
        control: AITaskControl,
        execution: asyncio.Task[Any] | None,
    ) -> None:
        if execution is not None:
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        with suppress(Exception):
            await asyncio.shield(
                self.repository.release_ai_task(
                    control.task_id,
                    control.lease_token,
                    self.worker_id,
                    reason="durable_ai_task_manager_stopped",
                )
            )

    async def _execute_scoped(
        self,
        task: dict[str, Any],
        control: AITaskControl,
        executor: AITaskExecutor,
    ) -> dict[str, Any] | None:
        token = _current_task_id.set(int(task["task_id"]))
        try:
            workflow_id = int(task.get("workflow_id") or 0)
            trace = AIWorkContext(workflow_id) if workflow_id > 0 else None
            with bind_ai_work_context(trace):
                return await executor(task, control)
        finally:
            _current_task_id.reset(token)

    async def _wait_for_execution(
        self,
        task: dict[str, Any],
        control: AITaskControl,
        execution: asyncio.Task[Any],
        heartbeat: asyncio.Task[Any],
        gate_watch: asyncio.Task[Any],
    ) -> bool:
        watchers = {heartbeat, gate_watch}
        while not execution.done():
            control_wait = asyncio.create_task(control.control_event.wait())
            done, _ = await asyncio.wait(
                {execution, control_wait, *watchers},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if control_wait not in done:
                control_wait.cancel()
                await asyncio.gather(control_wait, return_exceptions=True)
            for watcher in watchers.intersection(done):
                if watcher.cancelled():
                    continue
                error = watcher.exception()
                if error is not None:
                    execution.cancel()
                    await asyncio.gather(execution, return_exceptions=True)
                    raise error
            if control.control_event.is_set() and not execution.done():
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
                await self._settle_control_interrupt(task, control)
                return True
            stopped = [watcher for watcher in watchers if watcher.done()]
            if stopped and not execution.done():
                execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
                raise RuntimeError("durable AI task watcher stopped unexpectedly")
        return False

    async def _settle_control_interrupt(self, task: dict[str, Any], control: AITaskControl) -> None:
        if control.runtime_disabled:
            await self.repository.release_ai_task(
                control.task_id,
                control.lease_token,
                self.worker_id,
                reason="profile_disabled_waiting",
            )
            return
        if control.requested_status == "PAUSE_REQUESTED":
            await self.repository.acknowledge_pause_ai_task(
                control.task_id,
                control.lease_token,
                self.worker_id,
                checkpoint=control.checkpoint,
            )
            return
        task_type = str(task.get("task_type") or "").upper()
        recovery_required = task_type not in {
            "BACKGROUND_AUTHOR",
            "FILE_ARTIFACT_GENERATION",
        }
        settled = await self.repository.acknowledge_cancel_ai_task(
            control.task_id,
            control.lease_token,
            self.worker_id,
            recovery_required=recovery_required,
        )
        if settled:
            return

    async def _complete_execution(
        self,
        task: dict[str, Any],
        control: AITaskControl,
        result: dict[str, Any] | None,
    ) -> None:
        if task.get("backend_id"):
            await self.repository.record_ai_backend_success(task["backend_id"])
        result_data = dict(result or {})
        default_status = "CANCELLED" if bool(result_data.get("cancelled")) else "SUCCEEDED"
        requested = str(result_data.pop("_task_status", default_status) or default_status).upper()
        if requested == "DEFERRED":
            completed = await self.repository.defer_ai_task(
                control.task_id,
                control.lease_token,
                self.worker_id,
                result=result_data,
                reason=str(result_data.get("deferred_reason") or "待依赖恢复"),
            )
        elif requested == "SUCCEEDED":
            completed = await self.repository.complete_ai_task(
                control.task_id,
                control.lease_token,
                self.worker_id,
                result=result_data,
            )
        elif requested == "CANCELLED":
            await self.repository.request_cancel_ai_task(
                control.task_id,
                actor_id="executor",
                reason=str(result_data.get("reason") or "executor_cancelled"),
                settle_domain=False,
            )
            completed = await self.repository.acknowledge_cancel_ai_task(
                control.task_id,
                control.lease_token,
                self.worker_id,
            )
        else:
            raise RuntimeError(f"executor requested unsupported terminal status {requested}")
        if not completed:
            await self._reconcile_terminal_request(control)

    async def _reconcile_terminal_request(
        self,
        control: AITaskControl,
    ) -> None:
        current = await self.repository.get_ai_task(control.task_id)
        if current and current["status"] == "PAUSE_REQUESTED":
            await self.repository.acknowledge_pause_ai_task(
                control.task_id,
                control.lease_token,
                self.worker_id,
                checkpoint=control.checkpoint,
            )
        elif current and current["status"] == "CANCEL_REQUESTED":
            settled = await self.repository.acknowledge_cancel_ai_task(
                control.task_id, control.lease_token, self.worker_id
            )
            if settled:
                return

    async def _fail_execution(
        self,
        task: dict[str, Any],
        control: AITaskControl,
        exc: Exception,
        *,
        executor_exists: bool,
    ) -> None:
        if task.get("backend_id"):
            with suppress(Exception):
                await self.repository.record_ai_backend_failure(
                    task["backend_id"], f"{type(exc).__name__}: {exc}"
                )
        retryable, recovery_required = self._failure_policy(exc, executor_exists)
        failed = await self.repository.fail_ai_task(
            control.task_id,
            control.lease_token,
            self.worker_id,
            f"{type(exc).__name__}: {exc}",
            retryable=retryable,
            recovery_required=recovery_required,
        )
        if failed is None:
            await self._reconcile_terminal_request(control)
            return

    @staticmethod
    def _failure_policy(exc: Exception, executor_exists: bool) -> tuple[bool, bool]:
        """Apply the producer's error classification without widening retries.

        Retry budgets are finite and survive worker restarts.  The old
        ``executor_exists`` fallback was still unsafe: a local, deterministic
        context construction failure would consume that budget needlessly.
        The invocation layer is the authority on whether repeating the same
        input can help; unwrapped exceptions are classified through that same
        policy before the durable state is changed.
        """

        del executor_exists
        info = exc.info if isinstance(exc, AIInvocationError) else classify_generic_error(exc)
        # Backend selection happens inside one invocation.  Once it returns a
        # context-budget error, every eligible backend has rejected the same
        # immutable request; durable replay cannot change its size.
        retryable = bool(info.retryable) and info.code is not AIErrorCode.CONTEXT_BUDGET
        details = info.details
        recovery_required = bool(
            details.get("external_side_effect_unknown") or details.get("recovery_required")
        )
        return retryable, recovery_required

    async def _heartbeat_loop(self, control: AITaskControl) -> None:
        interval = max(5, min(30, self.lease_seconds // 3))
        while True:
            await asyncio.sleep(interval)
            try:
                await control.heartbeat()
            except (AITaskPauseRequested, AITaskCancelRequested):
                return
            except AITaskLeaseLost:
                control.control_event.set()
                return

    async def _runtime_gate_loop(
        self,
        control: AITaskControl,
        profile_id: str,
        instance_id: str,
    ) -> None:
        while True:
            await asyncio.sleep(max(1, min(5, self.poll_seconds)))
            if not await self.runtime_gate.is_enabled(profile_id, instance_id):
                control.runtime_disabled = True
                control.control_event.set()
                return

    async def _cleanup_if_due(self) -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()
        if now - self._last_cleanup_monotonic < 3600:
            return
        await self.repository.cleanup_ai_task_history()
        self._last_cleanup_monotonic = now

    async def create_task(self, profile_id: str, task_type: str, **kwargs: Any) -> dict[str, Any]:
        await self.runtime_gate.require_enabled(
            profile_id,
            str(kwargs.get("instance_id") or ""),
        )
        return await self.repository.create_ai_task(profile_id, task_type, **kwargs)

    async def pause(
        self,
        task_id: int,
        *,
        actor_id: str = "admin",
        reason: str = "",
        expected_version: int | None = None,
    ) -> Any:
        return await self.repository.request_pause_ai_task(
            task_id,
            actor_id=actor_id,
            reason=reason,
            expected_version=expected_version,
        )

    async def resume(
        self,
        task_id: int,
        *,
        actor_id: str = "admin",
        expected_version: int | None = None,
    ) -> Any:
        return await self.repository.resume_ai_task(
            task_id, actor_id=actor_id, expected_version=expected_version
        )

    async def cancel(
        self,
        task_id: int,
        *,
        actor_id: str = "admin",
        reason: str = "",
        expected_version: int | None = None,
        permanent: bool = True,
    ) -> Any:
        return await self.repository.request_cancel_ai_task(
            task_id,
            actor_id=actor_id,
            reason=reason,
            expected_version=expected_version,
            settle_domain=permanent,
        )

    async def manual_retry(
        self,
        task_id: int,
        *,
        actor_id: str = "admin",
        expected_version: int | None = None,
    ) -> Any:
        return await self.repository.manual_retry_ai_task(
            task_id, actor_id=actor_id, expected_version=expected_version
        )


__all__ = [
    "AITaskCancelRequested",
    "AITaskControl",
    "AITaskExecutor",
    "AITaskLeaseLost",
    "AITaskPauseRequested",
    "DurableAITaskManager",
    "PrerequisiteTaskClaim",
    "PrerequisiteTaskClaimOutcome",
    "current_durable_ai_task_id",
]
