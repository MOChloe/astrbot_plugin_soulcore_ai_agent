"""Assemble resources that do not depend on Main Core or Sticker internals."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrbot.api.star import Context

from ..contracts.activation import RuntimeActivationGate
from ..contracts.runtime_limits import (
    AI_BACKGROUND_CONCURRENCY,
    AI_OPERATION_TIMEOUT_SECONDS,
    IMAGE_GENERATION_TIMEOUT_SECONDS,
    SCHEDULER_POLL_SECONDS,
)
from ..features.ai.durable_tasks import DurableAITaskManager
from ..features.ai.service import AIManager, BackendPool
from ..features.character_model.ports import CharacterModelReadPort
from ..features.character_model.service import CharacterModelService
from ..features.conversation.service import ConversationContextService
from ..features.delivery.qpm_repository import RepositoryQPMStorage
from ..features.delivery.voice_artifacts import VoiceArtifactService
from ..features.files.service import FileArtifactService
from ..features.group_flow.relocation import GroupReplyRelocationJudge
from ..features.group_flow.service import GroupFlowService, GroupInterjectionJudge
from ..features.group_flow.worker import GroupFlowWorker
from ..features.identity import IdentityService
from ..features.inbound_recall.worker import InboundRecallGraceWorker
from ..features.media.image_service import VisualExpressionService
from ..features.media.storage import MediaStorageCoordinator
from ..features.player_profiles.ports import PlayerProfileReader
from ..features.profiles.credentials import CredentialVault
from ..features.profiles.service import ProfileRuntimeGate
from ..features.recall import (
    AstrBotRecallProviderRegistry,
    RecallIndexWorker,
    RecallService,
    configure_tokenizer_cache,
)
from ..features.stickers.service import StickerService
from ..features.timeline.state_gate import StateMessageGate
from ..features.turn_buffer.service import TurnBufferClassifier
from ..features.turn_buffer.worker import TurnBufferWorker
from ..features.web.service import WebResearchService
from ..interfaces.astrbot.delivery import DeliveryTransport
from ..interfaces.astrbot.profile import ProfileResolver
from ..interfaces.astrbot.send_coordinator import UnifiedSendCoordinator
from ..interfaces.astrbot.support import AstrBotSyntheticEventFactory
from ..interfaces.astrbot.umo import RouteReadinessTracker
from ..storage import RepositoryBundle
from .ai_runtime import AIRuntimeLoader
from .container import SoulCoreContainer
from .runtime import (
    BootstrapRollbackError,
    BootstrapRollbackOwner,
    rollback_bootstrap,
)
from .runtime_fence import RuntimeOwnershipFence

_AI_CAPABILITY_TIMEOUTS = {"image.generate": IMAGE_GENERATION_TIMEOUT_SECONDS}


@dataclass(slots=True)
class Foundation:
    context: Context
    config: Any
    data_dir: Path
    boot_epoch: str
    runtime_fence: RuntimeOwnershipFence
    activation_gate: RuntimeActivationGate
    container: SoulCoreContainer
    repositories: RepositoryBundle
    profile_resolver: ProfileResolver
    route_readiness: RouteReadinessTracker
    delivery: DeliveryTransport
    credential_vault: CredentialVault
    runtime_gate: ProfileRuntimeGate
    character_models: CharacterModelReadPort
    identity: IdentityService
    player_profiles: PlayerProfileReader
    ai_manager: AIManager
    ai_runtime: AIRuntimeLoader
    context_service: ConversationContextService
    state_message_gate: StateMessageGate
    recall: RecallService
    recall_index_worker: RecallIndexWorker
    media_storage: MediaStorageCoordinator
    visual_service: VisualExpressionService
    file_artifacts: FileArtifactService
    voice_artifacts: VoiceArtifactService
    web_research: WebResearchService
    ai_tasks: DurableAITaskManager
    turn_buffer_worker: TurnBufferWorker
    group_flow_service: GroupFlowService
    group_flow_worker: GroupFlowWorker
    inbound_recall_worker: InboundRecallGraceWorker
    synthetic_event_factory: AstrBotSyntheticEventFactory


@dataclass(slots=True)
class _CoreRuntime:
    resolver: ProfileResolver
    route_readiness: RouteReadinessTracker
    activation_gate: RuntimeActivationGate
    runtime_gate: ProfileRuntimeGate
    delivery: DeliveryTransport
    synthetic_event_factory: AstrBotSyntheticEventFactory
    vault: CredentialVault
    manager: AIManager
    ai_runtime: AIRuntimeLoader


async def assemble_foundation(context: Context, config: Any, data_dir: Path) -> Foundation:
    runtime_fence = RuntimeOwnershipFence.acquire(data_dir)
    try:
        container = await SoulCoreContainer.create(data_dir)
    except BaseException as exc:
        rollback_container = getattr(exc, "rollback_container", None)
        cleanup_error = getattr(exc, "rollback_cleanup_error", None)
        if rollback_container is None or not isinstance(cleanup_error, BaseException):
            runtime_fence.close()
            raise
        owner = BootstrapRollbackOwner(rollback_container, runtime_fence)
        if isinstance(exc, asyncio.CancelledError):
            exc.rollback_owner = owner  # type: ignore[attr-defined]
            raise exc from cleanup_error
        if isinstance(exc, Exception):
            assert isinstance(cleanup_error, Exception)
            raise BootstrapRollbackError(exc, cleanup_error, owner) from exc
        raise
    try:
        return await _build_foundation(context, config, data_dir, runtime_fence, container)
    except BaseException as exc:
        return await rollback_bootstrap(
            BootstrapRollbackOwner(container, runtime_fence),
            exc,
        )


async def _build_foundation(
    context: Context,
    config: Any,
    data_dir: Path,
    runtime_fence: RuntimeOwnershipFence,
    container: SoulCoreContainer,
) -> Foundation:
    repositories, boot_epoch = container.repositories, uuid.uuid4().hex
    core = await _build_core_runtime(context, data_dir, repositories, boot_epoch)
    character_models = CharacterModelService(repositories.character_models)
    identity = IdentityService(repositories.profiles, character_models)
    recall, recall_index_worker, context_service = _build_recall_runtime(
        repositories,
        context,
        config,
        data_dir,
        boot_epoch,
        core.manager,
        identity,
        core.runtime_gate,
        core.activation_gate,
    )
    media_storage = MediaStorageCoordinator.for_repository(
        repositories.media, repositories.stickers
    )
    visual_service = VisualExpressionService(
        media_repository=repositories.media,
        profiles_repository=repositories.profiles,
        world_repository=repositories.timeline,
        event_log=repositories.delivery,
        ai_manager=core.manager,
        identity=identity,
        file_store=media_storage.store,
        runtime_gate=core.runtime_gate,
        recall=recall,
        sticker_repository=repositories.stickers,
    )
    ai_tasks = DurableAITaskManager(
        repositories.ai,
        worker_id=f"soulcore:{boot_epoch}",
        poll_seconds=SCHEDULER_POLL_SECONDS,
        concurrency=AI_BACKGROUND_CONCURRENCY,
        runtime_gate=core.runtime_gate,
        file_reconciler=repositories.files,
    )
    turn_buffer_worker, group_flow_service, group_flow_worker, inbound_recall_worker = (
        _build_message_flow_workers(
            repositories,
            core.manager,
            core.runtime_gate,
            character_models,
            boot_epoch,
        )
    )
    result = Foundation(
        context,
        config if config is not None else {},
        data_dir,
        boot_epoch,
        runtime_fence,
        core.activation_gate,
        container,
        repositories,
        core.resolver,
        core.route_readiness,
        core.delivery,
        core.vault,
        core.runtime_gate,
        character_models,
        identity,
        repositories.player_profiles,
        core.manager,
        core.ai_runtime,
        context_service,
        StateMessageGate(repositories.timeline),
        recall,
        recall_index_worker,
        media_storage,
        visual_service,
        FileArtifactService(data_dir / "file_artifacts"),
        VoiceArtifactService(data_dir / "voice_artifacts"),
        WebResearchService(core.manager, repositories.web, repositories.profiles),
        ai_tasks,
        turn_buffer_worker,
        group_flow_service,
        group_flow_worker,
        inbound_recall_worker,
        core.synthetic_event_factory,
    )
    await _reconcile_media(result)
    return result


async def _build_core_runtime(
    context: Context,
    data_dir: Path,
    repositories: RepositoryBundle,
    boot_epoch: str,
) -> _CoreRuntime:
    resolver = ProfileResolver(context)
    await _sync_profiles(resolver, repositories)
    route_readiness = RouteReadinessTracker()
    activation_gate = RuntimeActivationGate()
    runtime_gate = ProfileRuntimeGate(repositories.profiles)
    delivery = DeliveryTransport(
        context,
        route_readiness,
        runtime_gate=runtime_gate,
        activation_gate=activation_gate,
    )
    delivery.policy_store = repositories.delivery
    delivery.send_coordinator = UnifiedSendCoordinator(RepositoryQPMStorage(repositories.delivery))
    synthetic_event_factory = AstrBotSyntheticEventFactory(context)
    vault = CredentialVault(data_dir)
    manager = AIManager(
        BackendPool(),
        repository=repositories.ai,
        credential_vault=vault,
        operation_timeout_seconds=AI_OPERATION_TIMEOUT_SECONDS,
        operation_timeout_seconds_by_capability=_AI_CAPABILITY_TIMEOUTS,
        activation_gate=activation_gate,
    )
    ai_runtime = AIRuntimeLoader(
        ai_repository=repositories.ai,
        profiles_repository=repositories.profiles,
        manager=manager,
        web_repository=repositories.web,
        credential_vault=vault,
    )
    await _recover_storage(repositories, boot_epoch)
    await ai_runtime.reload()
    return _CoreRuntime(
        resolver,
        route_readiness,
        activation_gate,
        runtime_gate,
        delivery,
        synthetic_event_factory,
        vault,
        manager,
        ai_runtime,
    )


def _build_recall_runtime(
    repositories: RepositoryBundle,
    context: Context,
    config: Any,
    data_dir: Path,
    boot_epoch: str,
    manager: AIManager,
    identity: IdentityService,
    runtime_gate: ProfileRuntimeGate,
    activation_gate: RuntimeActivationGate,
) -> tuple[RecallService, RecallIndexWorker, ConversationContextService]:
    configure_tokenizer_cache(data_dir / "recall" / "jieba")
    providers = AstrBotRecallProviderRegistry(context, config)
    recall = RecallService(repositories.recall, providers, ai_manager=manager)
    worker = RecallIndexWorker(
        repositories.recall,
        recall,
        providers,
        worker_id=f"soulcore:{boot_epoch}:recall-index",
        activation_gate=activation_gate,
    )
    context_service = _build_context_service(
        repositories,
        runtime_gate,
        identity,
    )
    return recall, worker, context_service


def _build_message_flow_workers(
    repositories: RepositoryBundle,
    manager: AIManager,
    runtime_gate: ProfileRuntimeGate,
    character_models: CharacterModelService,
    boot_epoch: str,
) -> tuple[TurnBufferWorker, GroupFlowService, GroupFlowWorker, InboundRecallGraceWorker]:
    turn_buffer_worker = TurnBufferWorker(
        repositories.turn_buffer,
        repositories.conversation,
        repositories.profiles,
        TurnBufferClassifier(manager),
        admission_barrier=repositories.delivery,
        worker_id=f"soulcore:{boot_epoch}:turn-buffer",
    )
    group_flow_service = GroupFlowService(repositories.group_flow)
    group_flow_worker = GroupFlowWorker(
        repositories.group_flow,
        GroupInterjectionJudge(manager, character_models),
        runtime_gate,
        relocation_judge=GroupReplyRelocationJudge(manager, character_models),
        worker_id=f"soulcore:{boot_epoch}:group-flow",
    )
    recall_worker = InboundRecallGraceWorker(
        repositories.inbound_recall,
        worker_id=f"soulcore:{boot_epoch}:inbound-recall",
    )
    return turn_buffer_worker, group_flow_service, group_flow_worker, recall_worker


def _build_context_service(
    repositories: RepositoryBundle,
    runtime_gate: ProfileRuntimeGate,
    identity: IdentityService,
) -> ConversationContextService:
    return ConversationContextService(
        repositories.conversation,
        media_repository=repositories.media,
        profiles=repositories.profiles,
        ai_tasks=repositories.ai,
        runtime_gate=runtime_gate,
        knowledge=repositories.knowledge,
        stickers=StickerService(repositories.stickers, profiles=repositories.profiles),
        platform_messages=repositories.delivery,
        player_profiles=repositories.player_profiles,
        identity=identity,
    )


async def _sync_profiles(resolver: ProfileResolver, repositories: RepositoryBundle) -> None:
    profiles = await resolver.list_profiles()
    await repositories.profiles.sync_profiles(
        [{"id": item.id, "name": item.name} for item in profiles]
    )


async def _recover_storage(repositories: RepositoryBundle, boot_epoch: str) -> None:
    await repositories.ai.recover_orphaned_ai_invocations()
    await repositories.delivery.recover_sending_instance_outbox()
    await repositories.media.recover_interrupted_media_delivery()
    await repositories.ai.recover_expired_ai_tasks(current_worker_id=f"soulcore:{boot_epoch}")
    for profile in await repositories.profiles.list_profiles(include_orphaned=False):
        await repositories.profiles.reset_instance_readiness(profile.profile_id)


async def _reconcile_media(foundation: Foundation) -> None:
    for profile in await foundation.repositories.profiles.list_profiles(include_orphaned=False):
        instances = await foundation.repositories.profiles.list_character_instances(
            profile.profile_id
        )
        for instance in instances:
            await foundation.media_storage.reconcile(profile.profile_id, instance.instance_id)


__all__ = ["Foundation", "assemble_foundation"]
