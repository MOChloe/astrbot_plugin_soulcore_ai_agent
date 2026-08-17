"""Destructive runtime operations and explicit scheduler ticks."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

from ....contracts.models import CoreWakeRequest, RunStatus, WakeSource
from ....contracts.runtime_cleanup import (
    FILE_ARTIFACT_RELEASE_PATHS_KEY,
    MEDIA_RELEASE_PATHS_KEY,
    STICKER_RELEASE_PATHS_KEY,
    VOICE_ARTIFACT_RELEASE_PATHS_KEY,
)
from ....features.ai.durable_tasks import DurableAITaskManager
from ....features.background.domain import AUTHOR_ORDER
from ....features.files.runtime_cleanup import drain_runtime_file_cleanup
from ....features.files.service import FileArtifactService
from ....features.main_core.ports import MainCoreHandlePort
from ....features.main_core.service import MainCoreRunner
from ....features.media.storage import MediaStorageCoordinator
from ....features.profiles.ports import ProfilesRepositoryPort
from ....features.timeline.ports import TimelineRepositoryPort
from ....features.timeline.scheduler import DurableSchedulerWorker
from ....shared.contact_runtime import contact_policy_enabled
from ..presentation import jsonable
from .profiles import ProfilesAdminController

AsyncAction = Callable[[], Awaitable[None]]
StartAction = Callable[[], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class _WorkersRunning:
    scheduler: bool
    background_scheduler: bool
    ai_tasks: bool
    expression_outbox: bool
    turn_buffer: bool
    group_flow: bool
    timer_runtime: bool


class RuntimeOperationsController:
    def __init__(
        self,
        *,
        repository: ProfilesRepositoryPort,
        timeline_repository: TimelineRepositoryPort,
        profiles: ProfilesAdminController,
        runner: MainCoreRunner,
        scheduler: DurableSchedulerWorker,
        background_repository: Any,
        background_scheduler: Any,
        ai_manager: Any,
        ai_tasks: DurableAITaskManager,
        expression_outbox_worker: Any,
        turn_buffer_worker: Any,
        group_flow_worker: Any,
        timer_runtime: Any,
        inbound: Any,
        media_storage: MediaStorageCoordinator,
        file_artifacts: FileArtifactService,
        visual_service: Any,
        tracked_tasks: set[asyncio.Task[Any]],
        stop_global_loops: AsyncAction,
        start_global_loops: Callable[[], None],
        create_task: Callable[..., asyncio.Task[Any]],
        main_core: MainCoreHandlePort,
    ) -> None:
        self.repository = repository
        self.timeline_repository = timeline_repository
        self.profiles = profiles
        self.runner = runner
        self.main_core = main_core
        self.scheduler = scheduler
        self.background_repository = background_repository
        self.background_scheduler = background_scheduler
        self.ai_manager = ai_manager
        self.ai_tasks = ai_tasks
        self.expression_outbox_worker = expression_outbox_worker
        self.turn_buffer_worker = turn_buffer_worker
        self.group_flow_worker = group_flow_worker
        self.timer_runtime = timer_runtime
        self.inbound = inbound
        self.media_storage = media_storage
        self.file_artifacts = file_artifacts
        self.visual_service = visual_service
        self.tracked_tasks = tracked_tasks
        self.stop_global_loops = stop_global_loops
        self.start_global_loops = start_global_loops
        self.create_task = create_task

    async def clear_profile_runtime(self, profile_id: str) -> dict[str, Any]:
        async with self.inbound.quiesce_profile(profile_id):
            running = await self._quiesce_workers()
            try:
                await self._cancel_profile_tasks(profile_id)
                await self.visual_service.cancel_profile_background(profile_id)
                deleted = await self.runner.clear_profile_runtime(
                    profile_id,
                    after_database_clear=self._delete_runtime_files,
                )
                return {
                    "ok": True,
                    "profile_id": profile_id,
                    "configuration_preserved": True,
                    "runtime_cleared": True,
                    "deleted": deleted,
                }
            finally:
                await self._resume_workers(running)

    async def reset_character_instance(
        self,
        profile_id: str,
        instance_id: str,
        *,
        preserve_stickers: bool = True,
    ) -> dict[str, Any]:
        """Quiesce one conversation's writers, reset it and start initialization."""

        await self.profiles.require_role_instance(profile_id, instance_id)
        async with (
            self.inbound.quiesce_instance(profile_id, instance_id),
            self.ai_manager.quiesce_instance(profile_id, instance_id),
            self.ai_tasks.quiesce_instance(profile_id, instance_id),
            self.visual_service.quiesce_instance_background(profile_id, instance_id),
        ):
            await self._cancel_profile_tasks(profile_id, instance_id)
            reset = await self.runner.reset_instance_runtime(
                profile_id,
                instance_id,
                preserve_stickers=bool(preserve_stickers),
                after_database_reset=self._delete_runtime_files,
            )
            return {
                "ok": True,
                "profile_id": profile_id,
                "instance_id": instance_id,
                "configuration_preserved": True,
                "conversation_binding_preserved": True,
                "stickers_preserved": bool(preserve_stickers),
                "reset_mode": "KEEP_STICKERS" if preserve_stickers else "ALL",
                "initialization_started": bool(reset["initialization_started"]),
                "initialization_state": str(reset["initialization_state"]),
                "deleted": reset["deleted"],
            }

    async def _quiesce_workers(self) -> _WorkersRunning:
        running = _WorkersRunning(
            scheduler=self.scheduler.running,
            background_scheduler=self.background_scheduler.running,
            ai_tasks=self.ai_tasks.running,
            expression_outbox=self.expression_outbox_worker.running,
            turn_buffer=self.turn_buffer_worker.running,
            group_flow=self.group_flow_worker.running,
            timer_runtime=bool(self.timer_runtime.running),
        )
        await self.stop_global_loops()
        if running.group_flow:
            await self.group_flow_worker.stop()
        if running.turn_buffer:
            await self.turn_buffer_worker.stop()
        if running.timer_runtime:
            await self.timer_runtime.stop()
        if running.expression_outbox:
            await self.expression_outbox_worker.stop()
        if running.scheduler:
            await self.scheduler.stop()
        if running.background_scheduler:
            await self.background_scheduler.stop()
        if running.ai_tasks:
            await self.ai_tasks.stop()
        return running

    async def _resume_workers(self, running: _WorkersRunning) -> None:
        async with self.inbound.worker_resume_fence() as generation:
            if generation is None:
                return
            started: list[AsyncAction] = []
            try:
                for start, stop in self._resume_steps(running):
                    current = await self._start_resumed_component(
                        generation,
                        start=start,
                        stop=stop,
                        started=started,
                    )
                    if not current:
                        await self._rollback_resumed_components(started)
                        return
            except BaseException as exc:
                try:
                    await self._rollback_resumed_components(started)
                except Exception as cleanup_error:
                    raise exc from cleanup_error
                raise

    def _resume_steps(
        self,
        running: _WorkersRunning,
    ) -> tuple[tuple[StartAction, AsyncAction], ...]:
        steps: list[tuple[StartAction, AsyncAction]] = []
        if running.expression_outbox:
            steps.append(
                (
                    self.expression_outbox_worker.start_ready,
                    self.expression_outbox_worker.stop,
                )
            )
        if running.scheduler:
            steps.append((self.scheduler.start, self.scheduler.stop))
        if running.background_scheduler:
            steps.append(
                (
                    self.background_scheduler.start_ready,
                    self.background_scheduler.stop,
                )
            )
        if running.ai_tasks:
            steps.append((self.ai_tasks.start_ready, self.ai_tasks.stop))
        if running.timer_runtime:
            steps.append((self.timer_runtime.start, self.timer_runtime.stop))
        if running.turn_buffer:
            steps.append((self.turn_buffer_worker.start_ready, self.turn_buffer_worker.stop))
        if running.group_flow:
            steps.append((self.group_flow_worker.start, self.group_flow_worker.stop))
        steps.append((self.start_global_loops, self.stop_global_loops))
        return tuple(steps)

    async def _start_resumed_component(
        self,
        generation: int,
        *,
        start: StartAction,
        stop: AsyncAction,
        started: list[AsyncAction],
    ) -> bool:
        if not await self.inbound.resume_generation_is_current(generation):
            return False
        started.append(stop)
        result = start()
        if inspect.isawaitable(result):
            await result
        return await self.inbound.resume_generation_is_current(generation)

    @staticmethod
    async def _rollback_resumed_components(started: list[AsyncAction]) -> None:
        failures: list[Exception] = []
        for stop in reversed(started):
            try:
                await stop()
            except Exception as exc:
                failures.append(exc)
        started.clear()
        if failures:
            raise ExceptionGroup("SoulCore reset worker resume rollback failed", failures)

    async def _cancel_profile_tasks(self, profile_id: str, instance_id: str = "") -> None:
        current = asyncio.current_task()
        profile_marker = f":{profile_id}"
        pending = [
            task
            for task in tuple(self.tracked_tasks)
            if task is not current
            and not task.done()
            and profile_marker in task.get_name()
            and (not instance_id or instance_id in task.get_name())
        ]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _delete_runtime_files(self, deleted: dict[str, Any]) -> None:
        media_paths = tuple(
            dict.fromkeys(
                str(path)
                for key in (MEDIA_RELEASE_PATHS_KEY, STICKER_RELEASE_PATHS_KEY)
                for path in deleted.pop(key, ())
                if str(path)
            )
        )
        artifact_paths = tuple(
            dict.fromkeys(
                str(path) for path in deleted.pop(FILE_ARTIFACT_RELEASE_PATHS_KEY, ()) if str(path)
            )
        )
        voice_paths = tuple(
            dict.fromkeys(
                str(path) for path in deleted.pop(VOICE_ARTIFACT_RELEASE_PATHS_KEY, ()) if str(path)
            )
        )
        await drain_runtime_file_cleanup(
            self.runner.runtime_cleanup,
            media_store=self.media_storage.store,
            file_artifacts=self.file_artifacts,
            voice_artifacts=getattr(self.runner, "voice_artifact_service", None),
            targets=(
                *(("MEDIA", relative_path) for relative_path in media_paths),
                *(("FILE_ARTIFACT", relative_path) for relative_path in artifact_paths),
                *(("VOICE_ARTIFACT", relative_path) for relative_path in voice_paths),
            ),
        )

    async def _reconcile_media_files(self) -> None:
        """Mark missing media files before any worker can consume them."""

        for profile in await self.repository.list_profiles():
            for instance in await self.repository.list_character_instances(profile.profile_id):
                await self.media_storage.reconcile(profile.profile_id, instance.instance_id)

    async def trigger_instance_tick(
        self,
        profile_id: str,
        instance_id: str,
        *,
        commit: bool = True,
        wait: bool = True,
        event: AstrMessageEvent | None = None,
        force_proactive_delivery: bool = False,
    ) -> dict[str, Any]:
        assert self.repository is not None
        assert self.runner is not None
        if not await self.repository.get_profile_soulcore_enabled(profile_id):
            return {"ok": False, "message": "SoulCore is disabled for this profile"}
        instance = await self.repository.get_character_instance(profile_id, instance_id)
        if instance is None:
            return {"ok": False, "message": "character instance is unavailable"}
        scope_config = await self.repository.get_scope_config(profile_id, instance.scope)
        if scope_config is None:
            return {"ok": False, "message": "character instance configuration is unavailable"}
        if force_proactive_delivery and not await contact_policy_enabled(
            self.timeline_repository, profile_id, instance_id
        ):
            return {
                "ok": False,
                "message": "proactive contact is disabled for this instance",
            }
        if not commit:
            return {
                "ok": True,
                "dry_run": True,
                "profile_id": profile_id,
                "instance_id": instance_id,
                "route": instance.route_umo,
            }
        if not force_proactive_delivery:
            workspace = await self.background_repository.load_background_workspace(
                profile_id, instance_id
            )
            background_instance = (
                workspace.get("instance") if isinstance(workspace, Mapping) else None
            )
            if not isinstance(background_instance, Mapping):
                raise RuntimeError("background workspace instance is unavailable")
            forced = await self.background_repository.force_background_authors(
                profile_id,
                instance_id,
                author_kinds=AUTHOR_ORDER,
                expected_version=int(background_instance["config_version"]),
            )
            for task_id in forced["active_task_ids"]:
                await self.ai_tasks.repository.expedite_ai_task(
                    int(task_id),
                    actor_id="background-manual-wake",
                )
            self.background_scheduler.notify()
            return {
                "ok": True,
                "queued": True,
                "profile_id": profile_id,
                "instance_id": instance_id,
                "config_version": int(forced["config_version"]),
            }
        source = WakeSource.PLUGIN_WAKE
        request = CoreWakeRequest(
            profile_id=profile_id,
            instance_id=instance_id,
            source=source,
            reason=(
                "从高级设置手动唤醒主 Core 验证当前实例表达"
                if force_proactive_delivery
                else "从高级设置手动唤醒主 Core"
            ),
            route_umo=instance.route_umo,
            metadata=(
                {
                    "proactive_contact_required": True,
                    "forced_proactive": True,
                    "required_proactive_umo": instance.route_umo,
                }
                if force_proactive_delivery
                else {}
            ),
        )

        async def execute() -> dict[str, Any]:
            result = await self.main_core.handle(request, event=event)
            return {
                "ok": result.status is RunStatus.COMPLETED,
                "forced_proactive": force_proactive_delivery,
                "profile_id": profile_id,
                "instance_id": instance_id,
                "target_umo": instance.route_umo,
                "result": jsonable(result),
            }

        if wait:
            return await execute()
        self.create_task(execute(), name=f"soulcore-instance-tick:{profile_id}:{instance_id}")
        return {
            "ok": True,
            "queued": True,
            "profile_id": profile_id,
            "instance_id": instance_id,
        }
