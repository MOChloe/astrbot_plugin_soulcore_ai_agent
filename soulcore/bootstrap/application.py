"""Owned SoulCore application lifetime after composition is complete."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ..features.files.release_recovery import drain_file_artifact_releases
from ..features.files.runtime_cleanup import drain_runtime_file_cleanup
from ..features.media.visual_cache import (
    VISUAL_OBSERVATION_CONTRACT_VERSION,
    VisualCachePolicy,
)
from ..shared.event_log import record_event
from ..storage import RepositoryBundle


class SoulCoreApplication:
    _CLEANUP_STEP_TIMEOUT_SECONDS = 30.0
    _GLOBAL_LOOPS = {
        "soulcore-media-cleanup",
        "soulcore-media-recovery",
        "soulcore-sticker-triggers",
    }

    def __init__(
        self,
        *,
        repositories: RepositoryBundle,
        runner: Any,
        scheduler: Any,
        background_scheduler: Any,
        expression_outbox_worker: Any,
        ai_tasks: Any,
        visual_service: Any,
        media_storage: Any,
        file_artifacts: Any,
        voice_artifacts: Any,
        sticker_triggers: Any,
        turn_buffer_worker: Any,
        group_flow_worker: Any,
        recall_index_worker: Any,
        route_readiness: Any,
        plugin_version: str,
        timer_runtime: Any,
        inbound_recall_worker: Any,
        runtime_fence: Any,
        activation_gate: Any,
    ) -> None:
        self.repositories = repositories
        self.runner = runner
        self.scheduler = scheduler
        self.background_scheduler = background_scheduler
        self.expression_outbox_worker = expression_outbox_worker
        self.ai_tasks = ai_tasks
        self.visual_service = visual_service
        self.media_storage = media_storage
        self.file_artifacts = file_artifacts
        self.voice_artifacts = voice_artifacts
        self.sticker_triggers = sticker_triggers
        self.turn_buffer_worker = turn_buffer_worker
        self.group_flow_worker = group_flow_worker
        self.recall_index_worker = recall_index_worker
        self.inbound_recall_worker = inbound_recall_worker
        self.timer_runtime = timer_runtime
        self.route_readiness = route_readiness
        self.runtime_fence = runtime_fence
        self.activation_gate = activation_gate
        self.plugin_version = plugin_version
        self.commands: Any = None
        self.page: Any = None
        self.inbound: Any = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._close_lock = asyncio.Lock()
        self._closed_cleanup_steps: set[str] = set()
        self.terminating = False
        self.started = False

    @property
    def tracked_tasks(self) -> set[asyncio.Task[Any]]:
        return self._tasks

    def bind_interfaces(self, *, commands: Any, page: Any, inbound: Any) -> None:
        if self.commands is not None:
            raise RuntimeError("application interfaces are already bound")
        self.commands, self.page, self.inbound = commands, page, inbound

    async def start(self) -> None:
        if self.started:
            return
        if self.terminating:
            raise RuntimeError("application is terminating")
        self.terminating = False
        await self._prepare_expired_file_artifact_releases()
        await self._drain_file_artifact_releases()
        await self._drain_runtime_file_cleanup()
        await self.expression_outbox_worker.start_ready()
        await self.recall_index_worker.start_ready()
        self.scheduler.start()
        await self.background_scheduler.start_ready()
        await self.ai_tasks.start_ready()
        self.timer_runtime.start()
        await self.inbound_recall_worker.start_ready()
        await self.turn_buffer_worker.start_ready()
        await self.group_flow_worker.start_ready()
        self.start_global_loops()
        await self._assert_critical_workers_running()
        await self._record_started()
        await self._assert_critical_workers_running()
        if self.inbound is not None:
            await self.inbound.open_admission()
        self.started = True
        self.activation_gate.commit()
        logger.info(f"[SoulCore] {self.plugin_version} initialized")

    async def close(self) -> None:
        async with self._close_lock:
            self.terminating = True
            self.started = False
            self.activation_gate.abort()
            failures: list[Exception] = []
            for name, cleanup in self._cleanup_steps():
                if name in self._closed_cleanup_steps:
                    continue
                try:
                    await asyncio.wait_for(cleanup(), timeout=self._CLEANUP_STEP_TIMEOUT_SECONDS)
                except TimeoutError:
                    failures.append(RuntimeError(f"cleanup step {name} exceeded its deadline"))
                except Exception as exc:
                    failures.append(exc)
                else:
                    self._closed_cleanup_steps.add(name)
            if not failures and "runtime_fence" not in self._closed_cleanup_steps:
                try:
                    await self._close_runtime_fence()
                except Exception as exc:
                    failures.append(exc)
                else:
                    self._closed_cleanup_steps.add("runtime_fence")
            if failures:
                raise ExceptionGroup("SoulCore application cleanup failed", failures)

    async def on_message(self, event: AstrMessageEvent) -> Any:
        if self.inbound is None:
            raise RuntimeError("inbound interface is unavailable")
        return await self.inbound.handle(event)

    async def call_page(self, method: str, payload: Any) -> dict[str, Any]:
        if self.page is None:
            raise RuntimeError("advanced settings interface is unavailable")
        return await self.page.call(method, payload)

    async def call_page_download(self, method: str, payload: Any) -> Any:
        if self.page is None:
            raise RuntimeError("advanced settings interface is unavailable")
        return await self.page.download(method, payload)

    def create_task(self, coroutine: Any, *, name: str) -> asyncio.Task[Any]:
        if self.terminating:
            close = getattr(coroutine, "close", None)
            if callable(close):
                close()
            raise RuntimeError("application is terminating")
        task = asyncio.create_task(coroutine, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._task_finished)
        return task

    def _task_finished(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            failure = task.exception()
        except asyncio.CancelledError:
            return
        if failure is not None:
            logger.error(
                "[SoulCore] application task %s failed: %s: %s",
                task.get_name(),
                type(failure).__name__,
                failure,
            )

    def start_global_loops(self) -> None:
        if self.terminating:
            return
        active = {task.get_name() for task in self._tasks if not task.done()}
        factories = (
            ("soulcore-media-cleanup", self._media_cleanup_loop),
            ("soulcore-media-recovery", self._recover_pending_media),
            ("soulcore-sticker-triggers", self._sticker_trigger_loop),
        )
        for name, factory in factories:
            if name not in active:
                self.create_task(factory(), name=name)

    async def stop_global_loops(self) -> None:
        current = asyncio.current_task()
        tasks = [
            task
            for task in tuple(self._tasks)
            if task is not current and not task.done() and task.get_name() in self._GLOBAL_LOOPS
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _media_cleanup_loop(self) -> None:
        while not self.terminating:
            try:
                await self.repositories.media.recover_interrupted_media_delivery(
                    stale_before=datetime.now(UTC) - timedelta(minutes=10)
                )
                await self._prepare_expired_file_artifact_releases()
                await self._drain_file_artifact_releases()
                await self._drain_runtime_file_cleanup()
                await self.media_storage.cleanup_expired(limit=100)
                await self.media_storage.release_pending(limit=100)
                await self.repositories.media.prune_visual_observation_cache(
                    contract_version=VISUAL_OBSERVATION_CONTRACT_VERSION
                )
                await self.repositories.web.cleanup_expired_web_research()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"[SoulCore] media cleanup failed: {type(exc).__name__}: {exc}")
            await asyncio.sleep(60)

    async def _drain_runtime_file_cleanup(self) -> None:
        result = await drain_runtime_file_cleanup(
            self.repositories.operations.runtime_cleanup,
            media_store=self.media_storage.store,
            file_artifacts=self.file_artifacts,
            voice_artifacts=self.voice_artifacts,
            limit=100,
            raise_on_failure=False,
        )
        if result.failed:
            logger.warning(
                "[SoulCore] durable runtime file cleanup retained "
                f"{result.failed} failed intent(s) for retry"
            )

    async def _drain_file_artifact_releases(self) -> None:
        result = await drain_file_artifact_releases(
            self.repositories.files,
            self.file_artifacts,
            limit=100,
            raise_on_failure=False,
        )
        if result.failed:
            logger.warning(
                "[SoulCore] file artifact release recovery retained "
                f"{result.failed} failed artifact(s) for retry"
            )

    async def _prepare_expired_file_artifact_releases(self) -> None:
        prepared = await self.repositories.files.prepare_expired_file_artifact_releases(limit=100)
        if prepared:
            logger.info(
                f"[SoulCore] prepared {prepared} expired file artifact(s) for durable release"
            )

    async def _recover_pending_media(self) -> None:
        for profile in await self.repositories.profiles.list_profiles(include_orphaned=False):
            instances = await self.repositories.profiles.list_character_instances(
                profile.profile_id
            )
            for instance in instances:
                await self._recover_instance_media(profile.profile_id, instance.instance_id)

    async def _recover_instance_media(self, profile_id: str, instance_id: str) -> None:
        offset = 0
        while True:
            assets = await self.repositories.media.list_media_assets(
                profile_id,
                instance_id,
                limit=1000,
                offset=offset,
            )
            pending = [
                asset.asset_id
                for asset in assets
                if asset.inspection_status.value in {"PENDING", "RUNNING"}
                and await self.repositories.media.asset_is_model_visible(
                    profile_id, instance_id, asset.asset_id
                )
            ]
            if pending:
                self.visual_service.describe_in_background(
                    profile_id=profile_id,
                    instance_id=instance_id,
                    asset_ids=pending,
                    cache_policy=VisualCachePolicy.USE,
                )
            if len(assets) < 1000:
                return
            offset += len(assets)

    async def _sticker_trigger_loop(self) -> None:
        await self._poll(self.sticker_triggers.scan_once, 30)

    async def _poll(self, action: Any, seconds: int) -> None:
        while not self.terminating:
            try:
                await action()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"[SoulCore] background scan failed: {type(exc).__name__}: {exc}")
            await asyncio.sleep(seconds)

    async def _cancel_tasks(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failures = [result for result in results if isinstance(result, Exception)]
            self._tasks.clear()
            if failures:
                raise ExceptionGroup("SoulCore application task cleanup failed", failures)
        else:
            self._tasks.clear()

    async def _assert_critical_workers_running(self) -> None:
        await asyncio.sleep(0)
        workers = [
            ("expression_outbox", self.expression_outbox_worker),
            ("scheduler", self.scheduler),
            ("background_scheduler", self.background_scheduler),
            ("ai_tasks", self.ai_tasks),
            ("turn_buffer", self.turn_buffer_worker),
            ("group_flow", self.group_flow_worker),
            ("recall_index", self.recall_index_worker),
        ]
        workers.append(("timer_runtime", self.timer_runtime))
        workers.append(("inbound_recall", self.inbound_recall_worker))
        stopped = [name for name, worker in workers if not bool(worker.running)]
        if stopped:
            raise RuntimeError(
                "critical workers exited during startup: " + ", ".join(sorted(stopped))
            )

    async def _close_inbound_admission(self) -> None:
        if self.inbound is not None:
            await self.inbound.close_admission_and_drain()

    def _cleanup_steps(self) -> tuple[tuple[str, Any], ...]:
        steps: list[tuple[str, Any]] = [
            ("inbound_admission", self._close_inbound_admission),
            ("application_tasks", self._cancel_tasks),
        ]
        steps.append(("inbound_recall", self.inbound_recall_worker.stop))
        steps.extend(
            (
                ("group_flow", self.group_flow_worker.stop),
                ("recall_index", self.recall_index_worker.stop),
                ("timer_runtime", self._stop_timer_runtime),
                ("turn_buffer", self.turn_buffer_worker.stop),
                ("expression_outbox", self.expression_outbox_worker.stop),
                ("scheduler", self.scheduler.stop),
                ("background_scheduler", self.background_scheduler.stop),
                ("ai_tasks", self.ai_tasks.stop),
                ("runner", self.runner.shutdown),
                ("visual_service", self.visual_service.close),
                ("route_readiness", self._clear_route_readiness),
                ("repositories", self.repositories.close),
            )
        )
        return tuple(steps)

    async def _stop_timer_runtime(self) -> None:
        await self.timer_runtime.stop()

    async def _clear_route_readiness(self) -> None:
        self.route_readiness.clear()

    async def _close_runtime_fence(self) -> None:
        self.runtime_fence.close()

    async def _record_started(self) -> None:
        for profile in await self.repositories.profiles.list_profiles(include_orphaned=False):
            await record_event(
                self.repositories.delivery,
                profile_id=profile.profile_id,
                level="INFO",
                category="lifecycle",
                message=f"SoulCore {self.plugin_version} 初始化完成",
                details={"scheduler_running": self.scheduler.running},
            )


__all__ = ["SoulCoreApplication"]
