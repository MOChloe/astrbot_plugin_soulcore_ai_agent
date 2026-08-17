"""Compose administrator and AstrBot adapters from explicit domain ports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from ..features.character_model.ports import CharacterModelAdminPort
from ..interfaces.admin.controllers.ai import AIAdminController
from ..interfaces.admin.controllers.ai_configuration import AIConfigurationController
from ..interfaces.admin.controllers.ai_probes import AIProbeController
from ..interfaces.admin.controllers.background import BackgroundAdminController
from ..interfaces.admin.controllers.character_import import QuickSetupCharacterImportController
from ..interfaces.admin.controllers.character_models import CharacterModelsAdminController
from ..interfaces.admin.controllers.diagnostics import DiagnosticsAdminController
from ..interfaces.admin.controllers.knowledge import KnowledgeAdminController
from ..interfaces.admin.controllers.media import MediaAdminController
from ..interfaces.admin.controllers.operations import RuntimeOperationsController
from ..interfaces.admin.controllers.player_profiles import PlayerProfilesAdminController
from ..interfaces.admin.controllers.profile_settings import ProfileSettingsController
from ..interfaces.admin.controllers.profiles import ProfilesAdminController
from ..interfaces.admin.controllers.role_packages import RolePackageController
from ..interfaces.admin.controllers.sticker_references import StickerReferenceController
from ..interfaces.admin.controllers.stickers import StickersAdminController
from ..interfaces.admin.controllers.thinking import ThinkingSettingsController
from ..interfaces.admin.controllers.timeline import TimelineAdminController
from ..interfaces.admin.controllers.web import WebAdminController
from ..interfaces.admin.page_controller import AdminPageController
from ..interfaces.astrbot.command_probes import CommandProbeController
from ..interfaces.astrbot.commands import CommandController
from ..interfaces.astrbot.foreground import ForegroundCoreController
from ..interfaces.astrbot.inbound import InboundEventController
from ..interfaces.astrbot.inbound_voice_repository import (
    SqliteInboundVoiceAdmissionRepository,
)
from ..interfaces.astrbot.outbound import ForegroundOutboundController
from ..interfaces.astrbot.persona_import import AstrBotPersonaImportAdapter
from ..storage.sqlite.uow import SqliteUnitOfWork
from .application import SoulCoreApplication
from .foundation import Foundation
from .workers import CoreRuntimeParts, WorkerParts


class InstanceRunQueryPort(Protocol):
    async def list_instance_runs(
        self, profile_id: str, instance_id: str, limit: int = 20
    ) -> list[dict[str, object]]: ...


class MainCoreQueries:
    """Public Main Core run read model backed by its owning store."""

    def __init__(self, timeline: InstanceRunQueryPort) -> None:
        self.timeline = timeline

    async def list_instance_runs(
        self, profile_id: str, instance_id: str, limit: int = 20
    ) -> list[dict[str, object]]:
        return await self.timeline.list_instance_runs(profile_id, instance_id, limit)


@dataclass(frozen=True, slots=True)
class InterfaceParts:
    page: AdminPageController
    commands: CommandController
    inbound: InboundEventController


@dataclass(frozen=True, slots=True)
class AstrBotInterfaceParts:
    outbound: ForegroundOutboundController
    inbound: InboundEventController


@dataclass(frozen=True, slots=True)
class BaseAdminControllers:
    profiles: ProfilesAdminController
    profile_settings: ProfileSettingsController
    knowledge: KnowledgeAdminController
    media: MediaAdminController
    references: StickerReferenceController
    stickers: StickersAdminController
    web: WebAdminController
    player_profiles: PlayerProfilesAdminController
    thinking: ThinkingSettingsController


@dataclass(frozen=True, slots=True)
class AdminControllers:
    base: BaseAdminControllers
    timeline: TimelineAdminController
    ai: AIAdminController
    operations: RuntimeOperationsController
    diagnostics: DiagnosticsAdminController
    background: BackgroundAdminController


def assemble_interfaces(
    foundation: Foundation,
    core: CoreRuntimeParts,
    workers: WorkerParts,
    application: SoulCoreApplication,
    *,
    plugin_version: str,
) -> InterfaceParts:
    repos = foundation.repositories
    queries = MainCoreQueries(repos.timeline)
    astrbot = _assemble_astrbot_interfaces(foundation, core)
    base_admin = _assemble_base_admin(foundation, core, workers, application)
    admin = _assemble_admin_controllers(
        foundation,
        core,
        workers,
        application,
        astrbot.inbound,
        base_admin,
        queries,
        plugin_version,
    )
    commands = _assemble_commands(foundation, core, admin, astrbot.outbound, queries)
    page = _assemble_page(
        foundation,
        application,
        admin,
        workers,
        plugin_version=plugin_version,
    )
    return InterfaceParts(page, commands, astrbot.inbound)


def _assemble_astrbot_interfaces(
    foundation: Foundation, core: CoreRuntimeParts
) -> AstrBotInterfaceParts:
    repos = foundation.repositories
    outbound = ForegroundOutboundController(
        profiles_repository=repos.profiles,
        conversation_repository=repos.conversation,
        media_repository=repos.media,
        sticker_repository=repos.stickers,
        event_log=repos.delivery,
        delivery=foundation.delivery,
        visual_service=foundation.visual_service,
        context_service=foundation.context_service,
        settlement_repository=repos.operations.outbox_settlement,
    )
    foreground = ForegroundCoreController(
        profiles_repository=repos.profiles,
        timeline_repository=repos.timeline,
        event_log=repos.delivery,
        ai_manager=foundation.ai_manager,
        visual_service=foundation.visual_service,
        runner=core.runner,
        outbound=outbound,
        delivery=foundation.delivery,
        context_service=foundation.context_service,
        inbound_recall_repository=repos.inbound_recall,
    )
    inbound = InboundEventController(
        boot_epoch=foundation.boot_epoch,
        profiles_repository=repos.profiles,
        conversation_repository=repos.conversation,
        timeline_repository=repos.timeline,
        delivery_repository=repos.delivery,
        ai_tasks=foundation.ai_tasks,
        media_repository=repos.media,
        event_log=repos.delivery,
        profile_resolver=foundation.profile_resolver,
        route_readiness=foundation.route_readiness,
        delivery=foundation.delivery,
        visual_service=foundation.visual_service,
        media_storage=foundation.media_storage,
        state_message_gate=foundation.state_message_gate,
        foreground=foreground,
        runner=core.runner,
        turn_buffer_repository=repos.turn_buffer,
        turn_buffer_worker=foundation.turn_buffer_worker,
        turn_buffer_gate_transfer=repos.operations.turn_buffer_gate_transfer,
        synthetic_event_factory=foundation.synthetic_event_factory,
        group_flow_service=foundation.group_flow_service,
        group_flow_repository=repos.group_flow,
        group_flow_worker=foundation.group_flow_worker,
        identity=foundation.identity,
        inbound_recall_repository=repos.inbound_recall,
        inbound_recall_worker=foundation.inbound_recall_worker,
        ai_manager=foundation.ai_manager,
        runtime_gate=foundation.runtime_gate,
        inbound_voice_repository=SqliteInboundVoiceAdmissionRepository(
            SqliteUnitOfWork(repos.engine)
        ),
    )
    foundation.inbound_recall_worker.bind_dispatch(inbound.dispatch_recall_grace_hold)
    foundation.inbound_recall_worker.bind_recall_dispatch(inbound.dispatch_recovered_recall_target)
    foundation.turn_buffer_worker.bind_dispatch(inbound.dispatch_buffered_batch)
    foundation.group_flow_worker.bind_dispatch(inbound.dispatch_group_window)
    foundation.group_flow_worker.bind_relocation(inbound.relocate_group_reply)
    core.runner.bind_group_first_attempt_callback(inbound.release_group_first_attempt_activity)
    return AstrBotInterfaceParts(outbound, inbound)


def _assemble_base_admin(
    foundation: Foundation,
    core: CoreRuntimeParts,
    workers: WorkerParts,
    application: SoulCoreApplication,
) -> BaseAdminControllers:
    repos = foundation.repositories
    profiles = ProfilesAdminController(
        repos.profiles,
        repos.timeline,
        repos.files,
        repos.delivery,
        foundation.context,
        foundation.profile_resolver,
        repos.group_flow,
        foundation.turn_buffer_worker,
    )
    profile_settings = ProfileSettingsController(
        repos.profiles,
        repos.timeline,
        foundation.context,
        profiles,
        repos.operations.scope_configuration,
        repos.group_flow,
    )
    knowledge = KnowledgeAdminController(
        repos.knowledge,
        workers.knowledge_plugin,
        foundation.recall,
        foundation.identity,
    )
    media = MediaAdminController(
        repos.media,
        repos.files,
        repos.delivery,
        foundation.file_artifacts,
        foundation.media_storage,
    )
    references = StickerReferenceController(
        repos.stickers,
        repos.profiles,
        repos.media,
        foundation.media_storage,
        foundation.visual_service,
        core.sticker_collector,
    )
    stickers = StickersAdminController(
        repos.stickers,
        repos.profiles,
        repos.conversation,
        repos.ai,
        foundation.ai_tasks,
        foundation.media_storage,
        core.sticker_collector,
        references,
        lambda: application.terminating,
    )
    web = WebAdminController(
        repos.web,
        repos.profiles,
        foundation.ai_manager,
        foundation.credential_vault,
        foundation.ai_runtime.reload,
    )
    player_profiles = PlayerProfilesAdminController(repos.player_profiles, foundation.identity)
    thinking = ThinkingSettingsController(repos.profiles)
    return BaseAdminControllers(
        profiles,
        profile_settings,
        knowledge,
        media,
        references,
        stickers,
        web,
        player_profiles,
        thinking,
    )


def _assemble_admin_controllers(
    foundation: Foundation,
    core: CoreRuntimeParts,
    workers: WorkerParts,
    application: SoulCoreApplication,
    inbound: InboundEventController,
    base: BaseAdminControllers,
    queries: MainCoreQueries,
    plugin_version: str,
) -> AdminControllers:
    repos = foundation.repositories
    timeline = TimelineAdminController(
        repos.profiles,
        repos.timeline,
        repos.conversation,
        repos.ai,
        repos.delivery,
        base.profiles,
        foundation.delivery,
        core.runner,
        foundation.context_service,
    )
    ai_configuration = AIConfigurationController(
        repos.ai,
        foundation.credential_vault,
        foundation.ai_runtime.reload,
        foundation.context,
    )
    ai_probes = AIProbeController(repos.ai, foundation.ai_manager)
    ai = AIAdminController(
        repository=repos.ai,
        main_core_queries=queries,
        delivery_repository=repos.delivery,
        ai_tasks=foundation.ai_tasks,
        ai_manager=foundation.ai_manager,
        credential_vault=foundation.credential_vault,
        configuration=ai_configuration,
        probes=ai_probes,
        reload_backends=foundation.ai_runtime.reload,
    )
    operations = RuntimeOperationsController(
        repository=repos.profiles,
        timeline_repository=repos.timeline,
        profiles=base.profiles,
        runner=core.runner,
        scheduler=workers.scheduler,
        background_repository=repos.background,
        background_scheduler=workers.background_scheduler,
        ai_manager=foundation.ai_manager,
        ai_tasks=foundation.ai_tasks,
        expression_outbox_worker=workers.expression_outbox_worker,
        turn_buffer_worker=foundation.turn_buffer_worker,
        group_flow_worker=foundation.group_flow_worker,
        timer_runtime=workers.timer_runtime,
        inbound=inbound,
        media_storage=foundation.media_storage,
        file_artifacts=foundation.file_artifacts,
        visual_service=foundation.visual_service,
        tracked_tasks=application.tracked_tasks,
        stop_global_loops=application.stop_global_loops,
        start_global_loops=application.start_global_loops,
        create_task=application.create_task,
        main_core=workers.foreground_main_core,
    )
    diagnostics = _diagnostics(
        foundation,
        workers,
        base.profiles,
        base.knowledge,
        ai,
        queries,
        plugin_version,
    )
    background = BackgroundAdminController(
        repos.background,
        repos.timeline,
        base.profiles,
        workers.background_scheduler,
        repos.ai,
    )
    return AdminControllers(base, timeline, ai, operations, diagnostics, background)


def _assemble_commands(
    foundation: Foundation,
    core: CoreRuntimeParts,
    admin: AdminControllers,
    outbound: ForegroundOutboundController,
    queries: MainCoreQueries,
) -> CommandController:
    repos = foundation.repositories
    base = admin.base
    command_probes = CommandProbeController(
        profiles_repository=repos.profiles,
        timeline_repository=repos.timeline,
        media_repository=repos.media,
        delivery_repository=repos.delivery,
        diagnostics=admin.diagnostics,
        operations=admin.operations,
        timeline=admin.timeline,
        knowledge=base.knowledge,
        visual_service=foundation.visual_service,
        runner=core.runner,
        route_readiness=foundation.route_readiness,
        boot_epoch=foundation.boot_epoch,
    )
    commands = CommandController(
        profiles_repository=repos.profiles,
        timeline_repository=repos.timeline,
        main_core_queries=queries,
        knowledge_repository=repos.knowledge,
        delivery_repository=repos.delivery,
        ai_repository=repos.ai,
        sticker_repository=repos.stickers,
        web_repository=repos.web,
        profile_resolver=foundation.profile_resolver,
        profiles=base.profiles,
        timeline=admin.timeline,
        knowledge=base.knowledge,
        ai=admin.ai,
        media=base.media,
        stickers=base.stickers,
        web=base.web,
        diagnostics=admin.diagnostics,
        operations=admin.operations,
        outbound=outbound,
        probes=command_probes,
        web_research=foundation.web_research,
    )
    return commands


def _assemble_page(
    foundation: Foundation,
    application: SoulCoreApplication,
    admin: AdminControllers,
    workers: WorkerParts,
    *,
    plugin_version: str,
) -> AdminPageController:
    repos = foundation.repositories
    base = admin.base
    page = AdminPageController(
        profiles_repository=repos.profiles,
        timeline_repository=repos.timeline,
        sticker_repository=repos.stickers,
        timer_repository=repos.timers,
        model_gateway=foundation.ai_manager,
        require_ready=lambda: _require_ready(application),
        profiles=base.profiles,
        character_models=CharacterModelsAdminController(
            cast(CharacterModelAdminPort, foundation.character_models)
        ),
        profile_settings=base.profile_settings,
        ai=admin.ai,
        timeline=admin.timeline,
        knowledge=base.knowledge,
        media=base.media,
        web=base.web,
        stickers=base.stickers,
        sticker_references=base.references,
        diagnostics=admin.diagnostics,
        operations=admin.operations,
        identity=foundation.identity,
        player_profiles=base.player_profiles,
        thinking=base.thinking,
        background=admin.background,
        character_import=QuickSetupCharacterImportController(
            AstrBotPersonaImportAdapter(foundation.context),
            foundation.ai_manager,
        ),
        role_packages=RolePackageController(
            repos.role_packages,
            character_models=repos.character_models,
            media_repository=repos.media,
            media_storage=foundation.media_storage,
            file_artifacts=foundation.file_artifacts,
            data_dir=foundation.data_dir,
            plugin_version=plugin_version,
            notify_background=workers.background_scheduler.notify,
        ),
    )
    return page


def _diagnostics(
    foundation: Foundation,
    workers: WorkerParts,
    profiles: ProfilesAdminController,
    knowledge: KnowledgeAdminController,
    ai: AIAdminController,
    queries: MainCoreQueries,
    version: str,
) -> DiagnosticsAdminController:
    repos = foundation.repositories
    return DiagnosticsAdminController(
        profiles_repository=repos.profiles,
        timeline_repository=repos.timeline,
        background_repository=repos.background,
        main_core_queries=queries,
        delivery_repository=repos.delivery,
        conversation_repository=repos.conversation,
        ai_repository=repos.ai,
        web_repository=repos.web,
        media_repository=repos.media,
        sticker_repository=repos.stickers,
        event_log=repos.delivery,
        profiles=profiles,
        knowledge=knowledge,
        ai=ai,
        scheduler=workers.scheduler,
        knowledge_plugin=workers.knowledge_plugin,
        recall=foundation.recall,
        context_service=foundation.context_service,
        credential_vault=foundation.credential_vault,
        plugin_version=version,
        group_flow_repository=repos.group_flow,
    )


def _require_ready(application: SoulCoreApplication) -> None:
    if not application.started:
        raise RuntimeError("SoulCore is not initialized")


__all__ = ["InterfaceParts", "MainCoreQueries", "assemble_interfaces"]
