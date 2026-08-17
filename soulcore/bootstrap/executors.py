"""Durable task executor registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..contracts.ai import TaskExecutor


class DurableExecutorManager(Protocol):
    def register_executor(self, task_type: str, executor: TaskExecutor) -> None: ...


@dataclass(frozen=True, slots=True)
class DurableExecutors:
    main_core: TaskExecutor
    background_author: TaskExecutor
    dialogue_summary: TaskExecutor
    knowledge_formation: TaskExecutor
    file_artifact_generation: TaskExecutor
    vision_description: TaskExecutor
    sticker_collection: TaskExecutor
    sticker_check: TaskExecutor
    sticker_intake: TaskExecutor
    timer_run: TaskExecutor
    timer_lifecycle_review: TaskExecutor


def register_durable_executors(
    manager: DurableExecutorManager,
    executors: DurableExecutors,
) -> None:
    """Register the frozen task surface in one auditable location."""

    manager.register_executor("MAIN_CORE", executors.main_core)
    manager.register_executor("BACKGROUND_AUTHOR", executors.background_author)
    manager.register_executor("DIALOGUE_SUMMARY", executors.dialogue_summary)
    manager.register_executor("KNOWLEDGE_FORMATION", executors.knowledge_formation)
    manager.register_executor("FILE_ARTIFACT_GENERATION", executors.file_artifact_generation)
    manager.register_executor("VISION_DESCRIPTION", executors.vision_description)
    manager.register_executor("STICKER_COLLECTION", executors.sticker_collection)
    manager.register_executor("STICKER_CHECK", executors.sticker_check)
    manager.register_executor("STICKER_INTAKE", executors.sticker_intake)
    manager.register_executor("TIMER_RUN", executors.timer_run)
    manager.register_executor("TIMER_LIFECYCLE_REVIEW", executors.timer_lifecycle_review)
