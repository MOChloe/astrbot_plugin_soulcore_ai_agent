"""Inbound AstrBot event orchestration for one foreground SoulCore turn."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from astrbot.api.event import AstrMessageEvent

from ...contracts.initialization import INSTANCE_INITIALIZATION_STARTED_NOTICE
from ...contracts.message_reference import with_inbound_reply_projection
from ...contracts.models import (
    InstanceInitializationState,
    stable_instance_id,
)
from ...contracts.turn_buffer import TurnBufferGateTransferPort
from ...features.conversation.ports import (
    ConversationRepositoryPort,
    TurnBufferRepositoryPort,
)
from ...features.delivery.ports import DeliveryRepositoryPort
from ...features.media.image_service import VisualExpressionService
from ...features.media.ports import MediaRepositoryPort
from ...features.media.service import GroupMediaProjectionService
from ...features.media.storage import MediaStorageCoordinator
from ...features.profiles.ports import ProfilesRepositoryPort
from ...features.profiles.service import ProfileRuntimeGate
from ...features.timeline.ports import TimelineRepositoryPort
from ...features.timeline.state_gate import (
    StateMessageGate,
)
from ...features.turn_buffer.worker import TurnBufferWorker
from ...shared.event_log import EventLogPort, record_event
from .admitted_turn import AdmittedTurnMixin
from .buffered_inbound import BufferedInboundMixin, BufferedLiveHandoff
from .context_message import event_context_payload
from .delivery import DeliveryTransport
from .event_ids import event_message_id, event_reference_message_id
from .foreground import ForegroundCoreController
from .group_inbound import INBOUND_ADMISSION_LEASE_SECONDS, GroupInboundMixin
from .inbound_ledger import append_inbound_ledger
from .inbound_lifecycle import INSTANCE_RESET_CANCEL_REASON, InboundLifecycleMixin
from .inbound_recall import InboundRecallMixin, onebot_recall_notice
from .inbound_references import InboundReferenceMixin
from .inbound_runtime import InboundRuntimeMixin
from .inbound_voice import (
    VOICE_TRANSCRIPTION_FAILURE_NOTICE,
    InboundOrderToken,
    InboundRouteOrderToken,
    InboundVoiceCoordinator,
    InboundVoiceTranscriptionError,
)
from .inbound_voice_repository import InboundVoiceAdmissionPort
from .initialization_inbound import hold_initialization_trigger
from .profile import ProfileResolver
from .support import has_trusted_astrbot_command_marker
from .umo import CapturedUMO, RouteReadinessTracker, physical_event_route


class InboundMediaSupportMixin:
    visual_service: VisualExpressionService
    event_log: EventLogPort

    async def _asset_paths(
        self, profile_id: str, instance_id: str, asset_ids: list[str]
    ) -> list[str]:
        result = []
        for asset_id in asset_ids:
            path = await self.visual_service.asset_file_path(
                profile_id=profile_id, instance_id=instance_id, asset_id=asset_id
            )
            if path:
                result.append(str(path))
        return result

    async def _media_error(
        self, profile_id: str, instance_id: str, message: str, exc: Exception
    ) -> None:
        await record_event(
            self.event_log,
            profile_id=profile_id,
            instance_id=instance_id,
            level="ERROR",
            category="media.ingest",
            message=message,
            details={"error": f"{type(exc).__name__}: {exc}"},
        )

    async def _log_duplicate(self, profile_id: str, instance_id: str, message_id: str) -> None:
        await record_event(
            self.event_log,
            profile_id=profile_id,
            instance_id=instance_id,
            level="WARN",
            category="foreground",
            message="平台重投消息的媒体已完成幂等补扫，不重复运行主 Core",
            details={"message_id": message_id},
        )

    @staticmethod
    def _safe_component(component: Mapping[str, Any]) -> dict[str, Any]:
        if str(component.get("type") or "").lower() != "image":
            return dict(component)
        return {key: value for key, value in component.items() if key not in {"url", "file"}}


class InboundEventController(
    InboundLifecycleMixin,
    InboundRuntimeMixin,
    InboundRecallMixin,
    InboundMediaSupportMixin,
    InboundReferenceMixin,
    AdmittedTurnMixin,
    GroupInboundMixin,
    BufferedInboundMixin,
):
    def __init__(
        self,
        *,
        boot_epoch: str,
        profiles_repository: ProfilesRepositoryPort,
        conversation_repository: ConversationRepositoryPort,
        timeline_repository: TimelineRepositoryPort,
        delivery_repository: DeliveryRepositoryPort,
        ai_tasks: Any,
        media_repository: MediaRepositoryPort,
        event_log: EventLogPort,
        profile_resolver: ProfileResolver,
        route_readiness: RouteReadinessTracker,
        delivery: DeliveryTransport,
        visual_service: VisualExpressionService,
        media_storage: MediaStorageCoordinator,
        state_message_gate: StateMessageGate,
        foreground: ForegroundCoreController,
        runner: Any,
        turn_buffer_repository: TurnBufferRepositoryPort,
        turn_buffer_worker: TurnBufferWorker,
        turn_buffer_gate_transfer: TurnBufferGateTransferPort,
        synthetic_event_factory: Any,
        group_flow_service: Any,
        group_flow_repository: Any,
        group_flow_worker: Any,
        identity: Any,
        inbound_recall_repository: Any,
        inbound_recall_worker: Any,
        ai_manager: Any,
        inbound_voice_repository: InboundVoiceAdmissionPort,
        runtime_gate: ProfileRuntimeGate,
    ) -> None:
        self.boot_epoch = boot_epoch
        self.profiles = profiles_repository
        self.conversation = conversation_repository
        self.timeline = timeline_repository
        self.delivery_repository = delivery_repository
        self.ai_tasks = ai_tasks
        self.media = media_repository
        self.event_log = event_log
        self.profile_resolver = profile_resolver
        self.route_readiness = route_readiness
        self.delivery = delivery
        self.visual_service = visual_service
        self.media_storage = media_storage
        self.state_message_gate = state_message_gate
        self.foreground = foreground
        self.runner = runner
        self.turn_buffer_repository = turn_buffer_repository
        self.turn_buffer_worker = turn_buffer_worker
        self.turn_buffer_gate_transfer = turn_buffer_gate_transfer
        self.synthetic_event_factory = synthetic_event_factory
        self.group_flow = group_flow_service
        self.group_flow_repository = group_flow_repository
        self.group_flow_worker = group_flow_worker
        self.identity = identity
        self.inbound_recall = inbound_recall_repository
        self.inbound_recall_worker = inbound_recall_worker
        self.inbound_voice_coordinator = InboundVoiceCoordinator(
            ai_manager=ai_manager,
            repository=inbound_voice_repository,
        )
        self.runtime_gate = runtime_gate
        self.group_media = GroupMediaProjectionService(media_repository, visual_service)
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._recall_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._inbound_registry_lock = asyncio.Lock()
        self._worker_lifecycle_lock = asyncio.Lock()
        self._accepting_inbound = True
        self._admission_generation = 0
        self._active_inbound_handlers: dict[asyncio.Task[Any], int] = {}
        self._inbound_handler_sequence = 0
        self._active_inbound_handler_sequences: dict[asyncio.Task[Any], int] = {}
        self._inflight_inbound: dict[tuple[str, str], set[asyncio.Task[Any]]] = {}
        self._profile_inbound_cutoffs: dict[str, int] = {}
        self._instance_inbound_cutoffs: dict[tuple[str, str], int] = {}
        self._resetting_instances: set[tuple[str, str]] = set()
        self._resetting_profiles: set[str] = set()
        self._instance_reset_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._profile_reset_locks: dict[str, asyncio.Lock] = {}
        self._recall_worker_quiesce_lock = asyncio.Lock()
        self._inbound_order_tokens: dict[asyncio.Task[Any], InboundOrderToken] = {}
        self._inbound_route_order_tokens: dict[asyncio.Task[Any], InboundRouteOrderToken] = {}

    async def handle(self, event: AstrMessageEvent) -> Any:
        # AstrBot has already selected the command handler at this point.
        # Commands are control-plane events and must never resolve a character,
        # touch initialization gates, enter a turn buffer, or wake MainCore.
        if has_trusted_astrbot_command_marker(event):
            return None
        if not await self._register_inbound_handler():
            return None
        route_token: InboundRouteOrderToken | None = None
        try:
            route_token = await self._begin_inbound_route_order(physical_event_route(event).raw)
            return await self._handle_admitted(event)
        finally:
            self._release_inbound_route_order(route_token)
            await self._unregister_inbound_handler()

    async def _handle_admitted(self, event: AstrMessageEvent) -> Any:
        arrived_at = datetime.now(UTC)
        profile = await self.profile_resolver.resolve_event(event)
        try:
            enabled = await self.profiles.get_profile_soulcore_enabled(profile.id)
        except KeyError:
            enabled = False
        if not enabled:
            return None
        captured = self.route_readiness.note_inbound(physical_event_route(event))
        if not captured.is_valid:
            # The profile is enabled but the route cannot be scoped to an
            # instance.  Preserve the existing fail-closed behavior for this
            # malformed SoulCore route.
            event.should_call_llm(False)
            event.stop_event()
            return None
        instance_id = stable_instance_id(
            captured.raw,
            platform_id=captured.platform_id,
            message_type=captured.message_type,
            target_id=captured.target_id,
        )
        if not await self.runtime_gate.is_enabled(profile.id, instance_id):
            return None
        # SoulCore only owns the event after both the profile and this exact
        # conversation route are enabled.  A disabled friend/group therefore
        # remains available to later AstrBot handlers without any SoulCore
        # history, media, or foreground side effects.
        instance_key = (profile.id, instance_id)
        if not await self._register_inbound(instance_key):
            return None
        try:
            recall_enabled = self._recall_enabled(captured)
            consumed, result = await self._consume_recall_notice(
                event,
                profile_id=profile.id,
                captured=captured,
                arrived_at=arrived_at,
                recall_enabled=recall_enabled,
            )
            if consumed:
                self._claim_event(event)
                return result
            instance = await self.profiles.get_character_instance(profile.id, instance_id)
            instance_refreshed = bool(
                instance is None
                or instance.initialization_state is InstanceInitializationState.UNINITIALIZED
            )
            if instance_refreshed:
                instance = await self._ensure_instance(profile.id, captured)
            result = await self._handle_registered_inbound(
                event,
                profile_id=profile.id,
                instance=instance,
                captured=captured,
                arrived_at=arrived_at,
                recall_enabled=recall_enabled,
                instance_refreshed=instance_refreshed,
            )
            self._claim_event(event)
            return result
        except asyncio.CancelledError as exc:
            if INSTANCE_RESET_CANCEL_REASON in {str(item) for item in exc.args}:
                return None
            raise
        finally:
            await self._unregister_inbound(instance_key)

    @staticmethod
    def _claim_event(event: AstrMessageEvent) -> None:
        event.should_call_llm(False)
        event.stop_event()

    async def _handle_registered_inbound(
        self,
        event: AstrMessageEvent,
        *,
        profile_id: str,
        instance: Any,
        captured: CapturedUMO,
        arrived_at: datetime,
        recall_enabled: bool,
        instance_refreshed: bool,
    ) -> Any:
        if (
            instance.initialization_state is not InstanceInitializationState.UNINITIALIZED
            and await self._gate_instance_initialization(
                event,
                profile_id,
                instance,
                captured,
                arrived_at,
                payload={},
                message_text="",
                has_content=False,
            )
        ):
            return None
        payload = event_context_payload(event)
        message_text = str(payload["plain_text"] or "").strip()
        has_content = bool(
            message_text or payload["components"] or event_reference_message_id(event)
        )
        if (
            instance.initialization_state is InstanceInitializationState.UNINITIALIZED
            and await self._gate_instance_initialization(
                event,
                profile_id,
                instance,
                captured,
                arrived_at,
                payload=payload,
                message_text=message_text,
                has_content=has_content,
            )
        ):
            return None
        if not instance_refreshed:
            instance = await self._ensure_instance(profile_id, captured)
        if not has_content:
            return None
        scope_config = await self.profiles.get_scope_config(profile_id, instance.scope)
        if scope_config is None:
            return None
        coordinator = self._voice_coordinator()
        order_token = await self._begin_inbound_order(profile_id, instance.instance_id)
        try:
            try:
                await coordinator.transcribe_payload(
                    profile_id=profile_id,
                    instance_id=instance.instance_id,
                    platform_message_id=event_message_id(event),
                    payload=payload,
                )
            except InboundVoiceTranscriptionError as exc:
                await record_event(
                    self.event_log,
                    profile_id=profile_id,
                    instance_id=instance.instance_id,
                    level="WARN",
                    category="conversation.inbound_voice",
                    message="语音转写失败，消息未写入会话账本",
                    details={"error": type(exc.__cause__ or exc).__name__},
                )
                event.should_call_llm(False)
                event.stop_event()
                await event.send(event.plain_result(VOICE_TRANSCRIPTION_FAILURE_NOTICE))
                return None
            message_text = str(payload.get("plain_text") or "").strip()
            return await self._accept_ready_event(
                event,
                profile_id=profile_id,
                instance=instance,
                scope_config=scope_config,
                captured=captured,
                message_text=message_text,
                payload=payload,
                recall_enabled=recall_enabled,
            )
        finally:
            self._release_inbound_order(order_token)

    async def _accept_ready_event(
        self,
        event: AstrMessageEvent,
        *,
        profile_id: str,
        instance: Any,
        scope_config: Any,
        captured: CapturedUMO,
        message_text: str,
        payload: dict[str, Any],
        recall_enabled: bool,
    ) -> Any:
        if recall_enabled and event_message_id(event):
            return await self._hold_recall_grace(
                event,
                profile_id,
                instance,
                scope_config,
                captured,
                message_text,
                payload,
            )
        if instance.scope == "group":
            return await self._accept_group_message(
                event,
                profile_id,
                instance,
                scope_config,
                captured,
                message_text,
                payload,
                passive=False,
            )
        return await self._accept_message(
            event,
            profile_id,
            instance,
            scope_config,
            captured,
            message_text,
            payload,
        )

    def _recall_enabled(self, captured: CapturedUMO) -> bool:
        return bool(self.delivery.capability_for(captured).inbound_recall_notice)

    async def _consume_recall_notice(
        self,
        event: AstrMessageEvent,
        *,
        profile_id: str,
        captured: CapturedUMO,
        arrived_at: datetime,
        recall_enabled: bool,
    ) -> tuple[bool, Any]:
        is_recall, notice = onebot_recall_notice(event, received_at=arrived_at)
        if not is_recall or not recall_enabled:
            return False, None
        instance_id = stable_instance_id(
            captured.raw,
            platform_id=captured.platform_id,
            message_type=captured.message_type,
            target_id=captured.target_id,
        )
        if notice is None:
            await record_event(
                self.event_log,
                profile_id=profile_id,
                instance_id=instance_id,
                level="WARN",
                category="conversation.inbound_recall",
                message="忽略缺少原始目标消息号的 OneBot 撤回通知",
                details={"notice_source": "raw_message.message_id"},
            )
            return True, None
        return (
            True,
            await self._handle_inbound_recall(profile_id, instance_id, captured, notice),
        )

    async def _ensure_instance(self, profile_id: str, captured: CapturedUMO) -> Any:
        values = {
            "platform_id": captured.platform_id,
            "message_type": captured.message_type,
            "target_id": captured.target_id,
            "session_kind": captured.kind.value,
        }
        return await self.profiles.ensure_character_instance(
            profile_id, captured.raw, ready=True, **values
        )

    async def _gate_instance_initialization(
        self,
        event: AstrMessageEvent,
        profile_id: str,
        instance: Any,
        captured: CapturedUMO,
        arrived_at: datetime,
        *,
        payload: dict[str, Any],
        message_text: str,
        has_content: bool,
    ) -> bool:
        decision = await self.profiles.begin_instance_initialization(
            profile_id,
            instance.instance_id,
            arrived_at,
            captured.raw,
        )
        if decision.accepts_messages:
            return False
        if decision.started:
            if has_content and str(instance.scope).lower() != "group":
                self._voice_coordinator().settle_without_transcription(payload)
                message_text = str(payload.get("plain_text") or "").strip()
                await self._hold_initialization_trigger(
                    event,
                    profile_id=profile_id,
                    instance=instance,
                    captured=captured,
                    arrived_at=arrived_at,
                    message_text=message_text,
                    payload=payload,
                )
            try:
                await event.send(event.plain_result(INSTANCE_INITIALIZATION_STARTED_NOTICE))
            except Exception as exc:
                await record_event(
                    self.event_log,
                    profile_id=profile_id,
                    instance_id=instance.instance_id,
                    level="ERROR",
                    category="initialization",
                    message="首次会话初始化提示发送失败",
                    details={"error": type(exc).__name__},
                )
        return True

    async def _hold_initialization_trigger(
        self,
        event: AstrMessageEvent,
        *,
        profile_id: str,
        instance: Any,
        captured: CapturedUMO,
        arrived_at: datetime,
        message_text: str,
        payload: dict[str, Any],
    ) -> None:
        await hold_initialization_trigger(
            self,
            event,
            profile_id=profile_id,
            instance=instance,
            captured=captured,
            arrived_at=arrived_at,
            message_text=message_text,
            payload=payload,
        )

    async def _accept_message(
        self,
        event: AstrMessageEvent,
        profile_id: str,
        instance: Any,
        scope_config: Any,
        captured: CapturedUMO,
        message_text: str,
        payload: dict[str, Any],
    ) -> Any:
        message_id = event_message_id(event)
        turn_buffer_enabled = await self.profiles.get_profile_turn_buffer_enabled(profile_id)
        ledger, inserted, lease, finish_ledger = await self._append_ledger(
            event,
            profile_id,
            instance,
            captured,
            message_text,
            payload,
            message_id,
            turn_buffer_enabled=turn_buffer_enabled,
        )
        message_text = str(payload.get("plain_text") or "").strip()
        if not inserted:
            await finish_ledger(None)
            await self._log_duplicate(profile_id, instance.instance_id, message_id)
            self.turn_buffer_worker.notify()
            return None
        assert lease is not None

        async def admit() -> Any:
            if not await finish_ledger(lease):
                return None
            if not await self._renew_owned_inbound_lease(ledger, lease):
                return None
            await self._ingest_media(
                event, profile_id, instance, captured, ledger, payload, message_id
            )
            activity = await self._mark_owned_activity(
                profile_id, instance.instance_id, captured, ledger, lease
            )
            if activity is None:
                return None
            activity_epoch, expression_barrier = activity
            projected = with_inbound_reply_projection(message_text, payload["components"])
            result = await self._dispatch_after_activity(
                turn_buffer_enabled=turn_buffer_enabled,
                event=event,
                profile_id=profile_id,
                instance=instance,
                scope_config=scope_config,
                captured=captured,
                message_text=projected,
                payload=payload,
                ledger=ledger,
                activity_epoch=activity_epoch,
                platform_message_id=message_id,
                force_durable_wait=expression_barrier,
                admission_lease=lease,
                durable_handoff_only=True,
            )
            if isinstance(result, BufferedLiveHandoff):
                return result
            if not await self._complete_inbound_admission(ledger, lease):
                return None
            return result

        result = await self._run_with_inbound_lease(ledger, lease, admit)
        if isinstance(result, BufferedLiveHandoff):
            return await self.turn_buffer_worker.wait_for_live_turn(
                result.batch,
                result.context,
            )
        return result

    async def _append_ledger(
        self,
        event: AstrMessageEvent,
        profile_id: str,
        instance: Any,
        captured: CapturedUMO,
        message_text: str,
        payload: dict[str, Any],
        message_id: str,
        *,
        turn_buffer_enabled: bool,
        knowledge_reason: str | None = None,
        delivery_status: str = "RECEIVED",
        direct_address: bool = False,
        project_foreground: bool = True,
        interrupt_background_author: bool = True,
    ) -> tuple[
        Any,
        bool,
        tuple[str, int] | None,
        Callable[[tuple[str, int] | None], Awaitable[bool]],
    ]:
        try:
            return await append_inbound_ledger(
                self,
                event,
                profile_id,
                instance,
                captured,
                message_text,
                payload,
                message_id,
                turn_buffer_enabled=turn_buffer_enabled,
                knowledge_reason=knowledge_reason,
                delivery_status=delivery_status,
                direct_address=direct_address,
                project_foreground=project_foreground,
                interrupt_background_author=interrupt_background_author,
                lease_seconds=INBOUND_ADMISSION_LEASE_SECONDS,
            )
        finally:
            self._release_inbound_order()

    async def _begin_inbound_order(self, profile_id: str, instance_id: str) -> InboundOrderToken:
        task = asyncio.current_task()
        assert task is not None
        sequence = int(self._active_inbound_handler_sequences[task])
        token = await self._voice_coordinator().acquire(profile_id, instance_id, sequence)
        self._inbound_order_tokens[task] = token
        return token

    async def _begin_inbound_route_order(self, route_key: str) -> InboundRouteOrderToken:
        task = asyncio.current_task()
        assert task is not None
        sequence = int(self._active_inbound_handler_sequences[task])
        coordinator = self._voice_coordinator()
        coordinator.register_route(route_key, sequence)
        token = await coordinator.acquire_route(route_key, sequence)
        self._inbound_route_order_tokens[task] = token
        return token

    def _release_inbound_order(self, token: InboundOrderToken | None = None) -> None:
        task = asyncio.current_task()
        active = self._inbound_order_tokens.pop(task, None) if task is not None else None
        owned = active or token
        if owned is not None:
            owned.release()

    def _release_inbound_route_order(self, token: InboundRouteOrderToken | None = None) -> None:
        task = asyncio.current_task()
        active = self._inbound_route_order_tokens.pop(task, None) if task is not None else None
        owned = active or token
        if owned is not None:
            owned.release()

    def _voice_coordinator(self) -> InboundVoiceCoordinator:
        return self.inbound_voice_coordinator

    async def _observe_inbound_identity(
        self,
        event: AstrMessageEvent,
        profile_id: str,
        instance: Any,
        sender_id: str,
        sender_name: str,
        message_id: int,
    ) -> None:
        await self.identity.observe_participant(
            profile_id,
            instance.instance_id,
            participant_id=sender_id,
            display_name=sender_name,
            source="OBSERVED",
            message_id=message_id,
        )
        if str(instance.scope).lower() == "group":
            await self.identity.refresh_group_directory(profile_id, instance.instance_id, event)
