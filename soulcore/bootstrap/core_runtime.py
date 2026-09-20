"Explicit Main Core and Sticker composition."

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..contracts.runtime_limits import AI_OPERATION_TIMEOUT_SECONDS
from ..features.main_core.service import MainCoreRunner, RunnerSettings
from ..features.stickers.service import StickerCollectorPlugin
from ..features.timers.occupancy import TimerOccupancyCoordinator
from ..features.timers.service import TimerMainCoreService
from .workers import CoreRuntimeParts

PLUGIN_NAME = "astrbot_plugin_soulcore_ai_agent"
DEFAULT_COMMAND_PARALLEL_CALLS = 8
MIN_COMMAND_PARALLEL_CALLS = 1
MAX_COMMAND_PARALLEL_CALLS = 32


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    scheduler_poll_seconds: int = 5
    command_parallel_calls: int = DEFAULT_COMMAND_PARALLEL_CALLS
    ai_operation_timeout_seconds: int = 300
    image_generation_timeout_seconds: int = 600
    file_artifact_pdf_timeout_seconds: int = 3000
    ai_background_concurrency: int = 2

    def operation_timeout(self, operation: str) -> int:
        if operation == "image.generate":
            return self.image_generation_timeout_seconds
        if operation == "file.pdf":
            return self.file_artifact_pdf_timeout_seconds
        return self.ai_operation_timeout_seconds


DEFAULT_RUNTIME_SETTINGS = RuntimeSettings()


def runtime_settings_from_config(config: Any) -> RuntimeSettings:
    """Build the remaining bounded settings owned by AstrBot configuration."""

    getter = getattr(config, "get", None)
    parallel_raw = (
        getter("command_parallel_calls", DEFAULT_COMMAND_PARALLEL_CALLS)
        if callable(getter)
        else None
    )
    try:
        parallel = (
            DEFAULT_COMMAND_PARALLEL_CALLS if isinstance(parallel_raw, bool) else int(parallel_raw)
        )
    except (TypeError, ValueError):
        parallel = DEFAULT_COMMAND_PARALLEL_CALLS
    parallel = max(
        MIN_COMMAND_PARALLEL_CALLS,
        min(MAX_COMMAND_PARALLEL_CALLS, parallel),
    )
    return RuntimeSettings(
        command_parallel_calls=parallel,
    )


if TYPE_CHECKING:
    from .foundation import Foundation


def build_core_runtime(foundation: Foundation) -> CoreRuntimeParts:
    """Build the three mutually dependent runtime collaborators from narrow ports."""

    runtime_settings = runtime_settings_from_config(foundation.config)
    repositories = foundation.repositories
    operations = repositories.operations
    timer_occupancy = TimerOccupancyCoordinator(repositories.timer_admission)
    foundation.visual_service.bind_character_models(foundation.character_models)
    runner = MainCoreRunner(
        profiles=repositories.profiles,
        conversation=repositories.conversation,
        timeline=repositories.timeline,
        knowledge=repositories.knowledge,
        stickers=repositories.stickers,
        outbox=repositories.delivery,
        delivery_policy=repositories.timeline,
        files=repositories.files,
        media=repositories.media,
        background=repositories.background,
        core_results=operations.core_results,
        outbox_settlement=operations.outbox_settlement,
        runtime_cleanup=operations.runtime_cleanup,
        event_log=repositories.delivery,
        model_gateway=foundation.ai_manager,
        delivery=foundation.delivery,
        settings=RunnerSettings(
            command_parallel_calls=runtime_settings.command_parallel_calls,
            command_timeout_seconds=AI_OPERATION_TIMEOUT_SECONDS,
        ),
        context_service=foundation.context_service,
        recall_service=foundation.recall,
        visual_service=foundation.visual_service,
        web_research=foundation.web_research,
        file_artifact_service=foundation.file_artifacts,
        voice_artifact_service=foundation.voice_artifacts,
        runtime_gate=foundation.runtime_gate,
        character_models=foundation.character_models,
        timer_occupancy=timer_occupancy,
        timer_commands=TimerMainCoreService(repositories.timers),
    )
    sticker_collector = StickerCollectorPlugin(
        stickers=repositories.stickers,
        profiles=repositories.profiles,
        media=repositories.media,
        model_gateway=foundation.ai_manager,
        visual_service=foundation.visual_service,
        web_research=foundation.web_research,
        media_storage=foundation.media_storage,
        identity=foundation.identity,
        character_models=foundation.character_models,
        operation_timeout_seconds=AI_OPERATION_TIMEOUT_SECONDS,
    )
    return CoreRuntimeParts(
        runner,
        sticker_collector,
        timer_occupancy,
    )


__all__ = ["build_core_runtime"]
