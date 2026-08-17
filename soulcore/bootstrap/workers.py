"""Assemble durable workers after the explicit Core collaborators exist."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from ..contracts.runtime_limits import AI_OPERATION_TIMEOUT_SECONDS, SCHEDULER_POLL_SECONDS
from ..features.ai.timer_lifecycle_reviewer import TimerLifecycleReviewer
from ..features.background.prewarmer import ProactiveFramePrewarmer
from ..features.background.runner import BackgroundAuthorRunner
from ..features.background.scheduler import BackgroundSchedulerWorker
from ..features.conversation.summary import DialogueSummaryExecutor
from ..features.delivery.expression_worker import ExpressionOutboxWorker
from ..features.files.executor import FileArtifactTaskExecutor
from ..features.knowledge.service import KnowledgeFormationPlugin
from ..features.main_core.foreground_coordinator import ForegroundMainCoreCoordinator
from ..features.main_core.service import MainCoreRunner
from ..features.stickers.service import StickerCollectorPlugin
from ..features.stickers.trigger import StickerTaskExecutor, StickerTriggerService
from ..features.timeline.contact_clock import ContactClock
from ..features.timeline.scheduler import DurableSchedulerWorker
from ..features.timeline.service import contact_source_is_policy_eligible
from ..features.timers.occupancy import TimerOccupancyCoordinator
from ..features.timers.runtime import (
    TimerClaimRetrySettler,
    TimerLifecycleCoordinator,
    TimerLifecycleReviewTaskExecutor,
    TimerRunTaskExecutor,
    TimerRuntimeExecutor,
    TimerRuntimeRecovery,
    TimerRuntimeWorker,
)
from .executors import DurableExecutors, register_durable_executors
from .task_executors import MainCoreTaskExecutor

if TYPE_CHECKING:
    from .foundation import Foundation


@dataclass(frozen=True, slots=True)
class CoreRuntimeParts:
    """Narrow handoff supplied by the Main Core/Sticker refactor."""

    runner: MainCoreRunner
    sticker_collector: StickerCollectorPlugin
    timer_occupancy: TimerOccupancyCoordinator


@dataclass(slots=True)
class WorkerParts:
    foreground_main_core: ForegroundMainCoreCoordinator
    background_author_runner: BackgroundAuthorRunner
    background_scheduler: BackgroundSchedulerWorker
    summary_executor: DialogueSummaryExecutor
    knowledge_plugin: KnowledgeFormationPlugin
    scheduler: DurableSchedulerWorker
    expression_outbox_worker: ExpressionOutboxWorker
    sticker_triggers: StickerTriggerService
    file_executor: FileArtifactTaskExecutor
    timer_runtime: TimerRuntimeWorker


def _assemble_background_runtime(
    foundation: Foundation,
    core: CoreRuntimeParts,
) -> tuple[ProactiveFramePrewarmer, ForegroundMainCoreCoordinator, BackgroundAuthorRunner]:
    repos = foundation.repositories
    prewarmer = ProactiveFramePrewarmer(repos.ai, foundation.ai_tasks)
    foreground = ForegroundMainCoreCoordinator(
        core.runner,
        repos.delivery,
        foundation.ai_tasks,
        proactive_frame_prewarmer=prewarmer,
    )
    author = BackgroundAuthorRunner(
        repository=repos.background,
        model_gateway=foundation.ai_manager,
        character_models=core.runner.character_models,
        identity=foundation.identity,
        operation_timeout_seconds=AI_OPERATION_TIMEOUT_SECONDS,
    )
    return prewarmer, foreground, author


def _assemble_timer_runtime(
    foundation: Foundation,
    core: CoreRuntimeParts,
    foreground_main_core: ForegroundMainCoreCoordinator,
    proactive_frame_prewarmer: ProactiveFramePrewarmer,
) -> tuple[TimerRuntimeWorker, TimerRunTaskExecutor, TimerLifecycleReviewTaskExecutor]:
    repos = foundation.repositories
    executor = TimerRuntimeExecutor(
        timers=repos.timers,
        profiles=repos.profiles,
        occupancy=core.timer_occupancy,
        runner=foreground_main_core,
        worker_id=f"soulcore:{foundation.boot_epoch}:timer",
        state_message_gate=foundation.state_message_gate,
    )
    lifecycle = TimerLifecycleCoordinator(repos.timers, repos.ai)
    task_executor = TimerRunTaskExecutor(
        runtime=executor,
        retry_settler=TimerClaimRetrySettler(repos.timers),
        profiles=repos.profiles,
        proactive_frame_prewarmer=proactive_frame_prewarmer,
        lifecycle=lifecycle,
    )
    return (
        TimerRuntimeWorker(
            profiles=repos.profiles,
            timers=repos.timers,
            tasks=repos.ai,
            executor=executor,
            recovery=TimerRuntimeRecovery(repos.timers),
            lifecycle=lifecycle,
            poll_seconds=SCHEDULER_POLL_SECONDS,
        ),
        task_executor,
        TimerLifecycleReviewTaskExecutor(
            repository=repos.timers,
            reviewer=TimerLifecycleReviewer(foundation.ai_manager),
        ),
    )


def assemble_workers(foundation: Foundation, core: CoreRuntimeParts) -> WorkerParts:
    repos = foundation.repositories
    proactive_frame_prewarmer, foreground_main_core, background_author = (
        _assemble_background_runtime(foundation, core)
    )
    summary = DialogueSummaryExecutor(
        repository=repos.conversation,
        profiles_repository=repos.profiles,
        media_repository=repos.media,
        model_gateway=foundation.ai_manager,
        context_service=foundation.context_service,
        runtime_gate=foundation.runtime_gate,
    )
    knowledge = KnowledgeFormationPlugin(
        context=foundation.context,
        repository=repos.knowledge,
        profiles=repos.profiles,
        event_log=repos.delivery,
        model_gateway=foundation.ai_manager,
        runtime_gate=foundation.runtime_gate,
        identity=foundation.identity,
    )
    file_executor = FileArtifactTaskExecutor(
        file_repository=repos.files,
        profiles_repository=repos.profiles,
        media_repository=repos.media,
        model_gateway=foundation.ai_manager,
        service=foundation.file_artifacts,
        visual_service=foundation.visual_service,
        identity=foundation.identity,
    )
    sticker_executor = StickerTaskExecutor(repos.stickers, core.sticker_collector)
    timer_runtime, timer_task_executor, timer_lifecycle_review_executor = _assemble_timer_runtime(
        foundation,
        core,
        foreground_main_core,
        proactive_frame_prewarmer,
    )
    register_durable_executors(
        foundation.ai_tasks,
        DurableExecutors(
            main_core=MainCoreTaskExecutor(
                core.runner,
                repos.timeline,
                repos.conversation,
                main_core=foreground_main_core,
            ).execute,
            background_author=background_author.execute_task,
            dialogue_summary=summary.execute_ai_task,
            knowledge_formation=knowledge.execute_ai_task,
            file_artifact_generation=file_executor.execute,
            vision_description=foundation.visual_service.execute_description_task,
            sticker_collection=sticker_executor.execute,
            sticker_check=sticker_executor.execute,
            sticker_intake=sticker_executor.execute,
            timer_run=timer_task_executor.execute,
            timer_lifecycle_review=timer_lifecycle_review_executor.execute,
        ),
    )
    scheduler = DurableSchedulerWorker(
        repos.timeline,
        core.runner,
        instance_wakeup_repository=repos.delivery,
        event_log=repos.delivery,
        profiles_repository=repos.profiles,
        conversation_repository=repos.conversation,
        media_repository=repos.media,
        contact_clock=ContactClock(repos.timeline, profiles=repos.profiles),
        ai_tasks=foundation.ai_tasks,
        poll_seconds=SCHEDULER_POLL_SECONDS,
        runtime_gate=foundation.runtime_gate,
        state_message_gate=foundation.state_message_gate,
    )
    expression_outbox_worker = ExpressionOutboxWorker(
        repos.delivery,
        core.runner.dispatch_due_expression_item,
        core.runner.dispatch_due_retraction_action,
    )

    def notify_expression_runtime() -> None:
        expression_outbox_worker.notify()
        foundation.turn_buffer_worker.notify()
        foundation.group_flow_worker.notify()

    core.runner.bind_expression_outbox_notifier(notify_expression_runtime)
    background_scheduler = BackgroundSchedulerWorker(
        repository=repos.background,
        ai_tasks=repos.ai,
        contact_policy_eligibility=partial(
            contact_source_is_policy_eligible,
            resolver=repos.timeline,
        ),
        poll_seconds=SCHEDULER_POLL_SECONDS,
    )
    return WorkerParts(
        foreground_main_core=foreground_main_core,
        background_author_runner=background_author,
        background_scheduler=background_scheduler,
        summary_executor=summary,
        knowledge_plugin=knowledge,
        scheduler=scheduler,
        expression_outbox_worker=expression_outbox_worker,
        sticker_triggers=StickerTriggerService(
            repos.stickers,
            repos.profiles,
            repos.ai,
            repos.conversation,
        ),
        file_executor=file_executor,
        timer_runtime=timer_runtime,
    )


__all__ = ["CoreRuntimeParts", "WorkerParts", "assemble_workers"]
