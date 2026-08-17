"""Implementation of the administrator probe command."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from astrbot.api.event import AstrMessageEvent

from ...contracts.models import OutboxStatus
from ...features.delivery.ports import DeliveryRepositoryPort
from ...features.main_core.service import MainCoreRunner
from ...features.media.image_service import VisualExpressionService
from ...features.media.ports import MediaRepositoryPort
from ...features.media.visual_cache import VisualCachePolicy
from ...features.profiles.ports import ProfilesRepositoryPort
from ...features.timeline.ports import TimelineRepositoryPort
from ..admin.controllers.diagnostics import DiagnosticsAdminController
from ..admin.controllers.knowledge import KnowledgeAdminController
from ..admin.controllers.operations import RuntimeOperationsController
from ..admin.controllers.timeline import TimelineAdminController
from ..admin.presentation import jsonable
from .umo import CapturedUMO, RouteReadinessTracker


class CommandProbeController:
    def __init__(
        self,
        *,
        profiles_repository: ProfilesRepositoryPort,
        timeline_repository: TimelineRepositoryPort,
        media_repository: MediaRepositoryPort,
        delivery_repository: DeliveryRepositoryPort,
        diagnostics: DiagnosticsAdminController,
        operations: RuntimeOperationsController,
        timeline: TimelineAdminController,
        knowledge: KnowledgeAdminController,
        visual_service: VisualExpressionService,
        runner: MainCoreRunner,
        route_readiness: RouteReadinessTracker,
        boot_epoch: str,
    ) -> None:
        self.profiles_repository = profiles_repository
        self.timeline_repository = timeline_repository
        self.media_repository = media_repository
        self.delivery_repository = delivery_repository
        self.diagnostics = diagnostics
        self.operations = operations
        self.timeline = timeline
        self.knowledge = knowledge
        self.visual_service = visual_service
        self.runner = runner
        self.route_readiness = route_readiness
        self.boot_epoch = boot_epoch

    async def probe(self, event: AstrMessageEvent, profile_id: str, component: str) -> Any:
        diagnostics = await self.diagnostics.diagnostics(profile_id)
        instance = await self._current_instance(event, profile_id)
        handlers: dict[str, Callable[[], Awaitable[Any]]] = {
            "database": lambda: self._database(profile_id, instance),
            "provider": lambda: self._provider(diagnostics),
            "main_core": lambda: self._main_core(profile_id, instance),
            "knowledge": lambda: self._knowledge(profile_id, instance),
            "vision": lambda: self._vision(profile_id, instance),
            "image": self._image,
            "delivery": lambda: self._delivery(event, profile_id, instance),
            "scheduler": lambda: self._scheduler(diagnostics),
            "context": lambda: self._context(profile_id, instance),
        }
        handler = handlers.get(str(component).lower())
        if handler is None:
            return {"ok": False, "error": "unknown probe component"}
        return await handler()

    async def _current_instance(self, event: AstrMessageEvent, profile_id: str) -> Any:
        captured = CapturedUMO.parse(str(event.unified_msg_origin))
        if not captured.is_valid:
            return None
        return await self.profiles_repository.ensure_character_instance(
            profile_id,
            captured.raw,
            platform_id=captured.platform_id,
            message_type=captured.message_type,
            target_id=captured.target_id,
            session_kind=captured.kind.value,
        )

    async def _database(self, profile_id: str, instance: Any) -> dict[str, Any]:
        return {
            "state": (
                await self.profiles_repository.get_instance_state(profile_id, instance.instance_id)
                if instance is not None
                else None
            ),
            "instance_id": instance.instance_id if instance else None,
            "ok": instance is not None,
        }

    @staticmethod
    async def _provider(diagnostics: dict[str, Any]) -> Any:
        return next(
            (item for item in diagnostics["doctor"] if item.get("name") == "direct_text_backend"),
            {"name": "direct_text_backend", "status": "error", "message": "not configured"},
        )

    async def _main_core(self, profile_id: str, instance: Any) -> Any:
        if instance is None:
            return {"ok": False, "error": "invalid_current_instance_route"}
        return await self.operations.trigger_instance_tick(
            profile_id, instance.instance_id, commit=False
        )

    async def _knowledge(self, profile_id: str, instance: Any) -> Any:
        if instance is None:
            return {"ok": False, "error": "invalid_current_instance_route"}
        return await self.knowledge.knowledge_snapshot(profile_id, instance.instance_id)

    async def _vision(self, profile_id: str, instance: Any) -> dict[str, Any]:
        assets = (
            await self.media_repository.list_media_assets(
                profile_id,
                instance.instance_id,
                file_status="AVAILABLE",
                limit=1000,
            )
            if instance is not None
            else []
        )
        asset = next(
            (item for item in assets if str(item.mime_type).casefold().startswith("image/")),
            None,
        )
        if asset is None:
            return {"ok": False, "error": "no_image_asset_for_vision_probe"}
        try:
            description = await self.visual_service.describe_asset(
                profile_id=profile_id,
                instance_id=instance.instance_id,
                asset_id=asset.asset_id,
                foreground=True,
                cache_policy=VisualCachePolicy.BYPASS,
            )
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {
            "ok": True,
            "asset_id": asset.asset_id,
            "description": jsonable(description),
        }

    @staticmethod
    async def _image() -> dict[str, Any]:
        return {
            "ok": False,
            "requires_confirmation": True,
            "warning": "真实生图探针会产生模型调用和费用。请在高级设置中确认后执行。",
            "capability": "image.generate",
        }

    async def _delivery(
        self, event: AstrMessageEvent, profile_id: str, instance: Any
    ) -> dict[str, Any]:
        captured = self.route_readiness.note_inbound(str(event.unified_msg_origin))
        if instance is None:
            return {
                "ok": False,
                "sends_message": False,
                "error": "invalid_current_instance_route",
            }
        if captured.is_valid:
            await self.profiles_repository.mark_instance_ready(
                profile_id, instance.instance_id, True
            )
        state = await self.profiles_repository.get_instance_state(profile_id, instance.instance_id)
        item = await self.delivery_repository.enqueue_instance_outbox(
            profile_id,
            instance.instance_id,
            {
                "content": "【SoulCore 投递测试】如果你看到这条消息，说明主动发送链路工作正常。",
                "context_record": False,
            },
            f"delivery-probe:{uuid.uuid4().hex}",
            activity_epoch=state.activity_epoch,
            origin_kind="DELIVERY_PROBE",
        )
        await self.runner.flush_instance_outbox(profile_id, instance.instance_id)
        item = await self.delivery_repository.get_instance_outbox(
            profile_id, instance.instance_id, item.outbox_id
        )
        accepted = bool(
            item
            and item.status is OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED
            and item.last_diagnostic_code
            in {
                "qq_passive_reply_api_accepted_unconfirmed",
                "qq_wakeup_api_accepted_unconfirmed",
                "adapter_acknowledged_message_id_change",
            }
        )
        return {
            "ok": accepted,
            "dispatch_attempted": bool(
                item
                and item.status
                in {
                    OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED,
                    OutboxStatus.PARTIALLY_ATTEMPTED,
                }
            ),
            "api_accepted_unconfirmed": accepted,
            "delivered": None,
            "delivery_note": "QQ/AstrBot 没有端到端送达回执；请以客户端是否实际显示为准。",
            "instance_id": instance.instance_id,
            "route": instance.route_umo,
            "outbox": jsonable(item),
        }

    @staticmethod
    async def _scheduler(diagnostics: dict[str, Any]) -> dict[str, Any]:
        return {"wakeups": diagnostics["wakeups"], "worker": "active"}

    async def _context(self, profile_id: str, instance: Any) -> Any:
        if instance is None:
            return {"ok": False, "error": "invalid_current_instance_route"}
        return await self.timeline.context_dry_run(profile_id, instance.instance_id)
