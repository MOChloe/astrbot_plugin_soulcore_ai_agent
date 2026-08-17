"""Foreground platform dispatch and durable ledger settlement."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from astrbot.api.event import AstrMessageEvent, MessageChain

from ...contracts.delivery_visibility import (
    FOREGROUND_DELIVERY_BOUNDARY_PREPARED,
    foreground_delivery_metadata,
)
from ...features.conversation.ports import ConversationRepositoryPort
from ...features.conversation.service import ConversationContextService
from ...features.delivery.ports import OutboxSettlementPort
from ...features.delivery.transport import DeliveryResult, DeliveryStatus
from ...features.media.image_service import VisualExpressionService
from ...features.media.ports import MediaRepositoryPort
from ...features.profiles.ports import ProfilesRepositoryPort
from ...features.stickers.policy import load_sticker_runtime_policy
from ...features.stickers.ports import StickerRepositoryPort
from ...shared.event_log import EventLogPort, record_event
from .delivery import DeliveryTransport
from .foreground_dispatch import (
    ForegroundDispatchCommand,
    ForegroundPlatformBoundaryState,
    ForegroundSendOutcome,
    ResolvedForegroundAssets,
    drain_cancelled_result,
    drain_cancelled_settlement,
)
from .umo import CapturedUMO

logger = logging.getLogger(__name__)
_UNKNOWN_SETTLEMENT_RETRY_DELAYS = (0.0, 0.05, 0.2)


class ForegroundOutboundController:
    def __init__(
        self,
        *,
        profiles_repository: ProfilesRepositoryPort,
        conversation_repository: ConversationRepositoryPort,
        media_repository: MediaRepositoryPort,
        sticker_repository: StickerRepositoryPort,
        event_log: EventLogPort,
        delivery: DeliveryTransport,
        visual_service: VisualExpressionService,
        context_service: ConversationContextService,
        settlement_repository: OutboxSettlementPort,
    ) -> None:
        self.profiles_repository = profiles_repository
        self.conversation_repository = conversation_repository
        self.media_repository = media_repository
        self.sticker_repository = sticker_repository
        self.event_log = event_log
        self.delivery = delivery
        self.visual_service = visual_service
        self.context_service = context_service
        self.settlement_repository = settlement_repository

    async def send_and_record_foreground(
        self,
        *,
        event: AstrMessageEvent,
        profile_id: str,
        instance_id: str,
        text: str,
        internal_memo: str = "",
        media_asset_ids: list[str],
        sticker_ref_ids: list[str],
        file_asset_ids: list[str],
        important_todo_ids: list[str],
        run_id: int,
        idempotency_key: str,
        metadata: dict[str, Any],
        captured: CapturedUMO | None = None,
        qpm_reservation: Any | None = None,
        configured_group_limit: int = 20,
    ) -> bool:
        if file_asset_ids or important_todo_ids:
            raise RuntimeError("file artifacts require the durable MainCore expression outbox")
        identity_template = str(text or "").strip()
        identity_context = await self.context_service.identity.context(profile_id, instance_id)
        text = self.context_service.identity.render(identity_template, identity_context).strip()
        assets, sticker_ref_ids = await self._resolve_allowed_assets(
            profile_id,
            instance_id,
            run_id,
            sticker_ref_ids,
        )
        if not text and not media_asset_ids and not assets.sticker_items:
            return False
        entry = await self._append_pending(
            profile_id,
            instance_id,
            text,
            identity_template,
            internal_memo,
            media_asset_ids,
            assets,
            idempotency_key,
            metadata,
        )
        return (
            await self._dispatch_pending_foreground(
                event=event,
                captured=captured,
                profile_id=profile_id,
                instance_id=instance_id,
                entry=entry,
                text=text,
                assets=assets,
                media_asset_ids=media_asset_ids,
                sticker_ref_ids=sticker_ref_ids,
                run_id=run_id,
                qpm_reservation=qpm_reservation,
                configured_group_limit=configured_group_limit,
            )
            is ForegroundSendOutcome.SENT
        )

    async def _dispatch_pending_foreground(
        self,
        *,
        event: AstrMessageEvent,
        captured: CapturedUMO | None,
        profile_id: str,
        instance_id: str,
        entry: Any,
        text: str,
        assets: ResolvedForegroundAssets,
        media_asset_ids: list[str],
        sticker_ref_ids: list[str],
        run_id: int,
        qpm_reservation: Any | None,
        configured_group_limit: int,
    ) -> ForegroundSendOutcome:
        return await ForegroundDispatchCommand(
            controller=self,
            event=event,
            captured=captured,
            profile_id=profile_id,
            instance_id=instance_id,
            entry=entry,
            text=text,
            assets=assets,
            media_asset_ids=media_asset_ids,
            sticker_ref_ids=sticker_ref_ids,
            run_id=run_id,
            qpm_reservation=qpm_reservation,
            configured_group_limit=configured_group_limit,
        ).run()

    async def _begin_foreground_platform_delivery(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
        boundary: ForegroundPlatformBoundaryState,
    ) -> bool:
        transition = asyncio.create_task(
            self.conversation_repository.begin_foreground_platform_delivery(
                profile_id,
                instance_id,
                message_id,
            )
        )
        try:
            entered = bool(await asyncio.shield(transition))
        except asyncio.CancelledError:
            entered = bool(await drain_cancelled_result(transition))
            boundary.entered = boundary.entered or entered
            raise
        boundary.entered = boundary.entered or entered
        return boundary.entered

    async def _stickers_enabled(self, profile_id: str, instance_id: str) -> bool:
        policy = await load_sticker_runtime_policy(
            self.sticker_repository,
            self.profiles_repository,
            profile_id,
            instance_id=instance_id,
        )
        return policy.enabled

    @staticmethod
    def _without_stickers(assets: ResolvedForegroundAssets) -> ResolvedForegroundAssets:
        return ResolvedForegroundAssets((), ())

    async def _resolve_allowed_assets(
        self,
        profile_id: str,
        instance_id: str,
        run_id: int,
        sticker_ref_ids: list[str],
    ) -> tuple[ResolvedForegroundAssets, list[str]]:
        allowed_sticker_refs = list(sticker_ref_ids)
        if allowed_sticker_refs and not await self._stickers_enabled(profile_id, instance_id):
            allowed_sticker_refs = []
        assets = await self._resolve_assets(
            profile_id,
            instance_id,
            run_id,
            allowed_sticker_refs,
        )
        if assets.sticker_items and not await self._stickers_enabled(profile_id, instance_id):
            return self._without_stickers(assets), []
        return assets, allowed_sticker_refs

    async def _pending_delivery_rejection(
        self,
        profile_id: str,
        instance_id: str,
        assets: ResolvedForegroundAssets,
    ) -> str:
        if not await self.profiles_repository.get_profile_soulcore_enabled(profile_id):
            return "profile_disabled_before_dispatch"
        if assets.sticker_items and not await self._stickers_enabled(profile_id, instance_id):
            return "sticker_disabled_before_dispatch"
        return ""

    async def _resolve_assets(
        self,
        profile_id: str,
        instance_id: str,
        run_id: int,
        sticker_ref_ids: list[str],
    ) -> ResolvedForegroundAssets:
        sticker_items = (
            await self.sticker_repository.resolve_sticker_run_refs(
                profile_id, instance_id, run_id, sticker_ref_ids
            )
            if sticker_ref_ids
            else []
        )
        return ResolvedForegroundAssets(
            tuple(sticker_items),
            tuple(item.asset_id for item in sticker_items),
        )

    async def _append_pending(
        self,
        profile_id: str,
        instance_id: str,
        text: str,
        identity_template: str,
        internal_memo: str,
        media_asset_ids: list[str],
        assets: ResolvedForegroundAssets,
        idempotency_key: str,
        metadata: dict[str, Any],
    ) -> Any:
        projections = [f"[表情包] {item.compact_description}" for item in assets.sticker_items]
        ledger_text = "\n".join(part for part in [str(text or "").strip(), *projections] if part)
        components = [{"type": "image_asset", "asset_id": asset_id} for asset_id in media_asset_ids]
        components.extend(
            {"type": "sticker", "projection": projection} for projection in projections
        )
        return await self.conversation_repository.append_instance_message(
            profile_id,
            instance_id,
            direction="OUTBOUND",
            role="assistant",
            sender_id="soulcore",
            sender_name="SoulCore",
            plain_text=ledger_text,
            identity_template=identity_template,
            internal_memo=str(internal_memo or "").strip(),
            components=components,
            delivery_status="PENDING",
            idempotency_key=idempotency_key,
            metadata={
                **metadata,
                **foreground_delivery_metadata(FOREGROUND_DELIVERY_BOUNDARY_PREPARED),
            },
        )

    async def _link_assets(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
        media_asset_ids: list[str],
        sticker_asset_ids: tuple[str, ...],
    ) -> None:
        for ordinal, asset_id in enumerate(media_asset_ids):
            await self.media_repository.link_media_to_message(
                profile_id,
                instance_id,
                asset_id,
                message_id,
                relation="GENERATED_OUTPUT",
                ordinal=ordinal,
            )

    async def _message_chain(
        self,
        profile_id: str,
        instance_id: str,
        text: str,
        media_asset_ids: list[str],
        assets: ResolvedForegroundAssets,
    ) -> Any:
        if not media_asset_ids and not assets.sticker_asset_ids:
            return text
        chain = MessageChain()
        if text:
            chain.message(text)
        for asset_id in [*media_asset_ids, *assets.sticker_asset_ids]:
            path = await self.visual_service.asset_file_path(
                profile_id=profile_id,
                instance_id=instance_id,
                asset_id=asset_id,
            )
            if not path:
                raise RuntimeError(f"selected image asset is unavailable: {asset_id}")
            chain.file_image(str(path))
        return chain

    async def _dispatch(
        self,
        event: AstrMessageEvent,
        chain: Any,
        captured: CapturedUMO | None,
        profile_id: str,
        instance_id: str,
        group_limit: int,
        reservation: Any | None,
        before_platform_call: Any,
    ) -> DeliveryResult | None:
        payload = event.plain_result(chain) if isinstance(chain, str) else chain
        if captured is None:
            if not await before_platform_call():
                return None
            await event.send(payload)
            return DeliveryResult.accepted_unconfirmed("foreground_event_send_accepted_unconfirmed")
        getter = getattr(event, "get_extra", None)
        if callable(getter) and bool(getter("soulcore_synthetic", False)):
            result = await self.delivery.send(
                captured,
                payload,
                profile_id=profile_id,
                instance_id=instance_id,
                configured_group_limit=group_limit,
                proactive=False,
                qpm_reservation=reservation,
                before_platform_call=before_platform_call,
            )
            if result.status not in {
                DeliveryStatus.ATTEMPTED_UNKNOWN,
                DeliveryStatus.PARTIALLY_ATTEMPTED,
            }:
                raise RuntimeError(
                    result.error_message or result.error_code or result.diagnostic_code
                )
            return result
        result = await self.delivery.send_event(
            event,
            captured,
            payload,
            profile_id=profile_id,
            instance_id=instance_id,
            configured_group_limit=group_limit,
            proactive=False,
            qpm_reservation=reservation,
            before_platform_call=before_platform_call,
        )
        if result.status not in {
            DeliveryStatus.ATTEMPTED_UNKNOWN,
            DeliveryStatus.PARTIALLY_ATTEMPTED,
        }:
            raise RuntimeError(result.error_message or result.error_code or result.diagnostic_code)
        return result

    async def _settle_failure(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
        media_asset_ids: list[str],
        exc: BaseException,
    ) -> None:
        error = f"{type(exc).__name__}: {exc}"
        await self.settlement_repository.finalize_foreground_delivery(
            profile_id,
            instance_id,
            message_id,
            media_asset_ids=tuple(media_asset_ids),
            todo_ids=(),
            status="FAILED",
            error=error,
        )
        await self._publish_settlement_backup(profile_id, instance_id, message_id)

    async def _settle_success(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
        run_id: int,
        media_asset_ids: list[str],
        sticker_ref_ids: list[str],
        *,
        sticker_items: tuple[Any, ...] = (),
        receipts: tuple[Any, ...] = (),
        route_umo: str = "",
    ) -> None:
        if len(sticker_ref_ids) != len(sticker_items):
            raise RuntimeError("foreground sticker delivery ownership is incomplete")
        sticker_deliveries = tuple(
            {
                "run_id": run_id,
                "sticker_ref": sticker_ref,
                "item_id": item.item_id,
                "projection": f"[表情包] {item.compact_description}",
            }
            for sticker_ref, item in zip(sticker_ref_ids, sticker_items, strict=True)
        )
        await self.settlement_repository.finalize_foreground_delivery(
            profile_id,
            instance_id,
            message_id,
            media_asset_ids=tuple(media_asset_ids),
            todo_ids=(),
            receipts=receipts,
            sticker_deliveries=sticker_deliveries,
            route_umo=route_umo,
        )
        if media_asset_ids:
            try:
                await self._record_post_settlement_event(
                    profile_id,
                    instance_id,
                    level="INFO",
                    category="image.delivery",
                    message="前台图片消息链已交给平台适配器",
                    details={"asset_count": len(media_asset_ids)},
                )
            except Exception:
                logger.exception(
                    "foreground image delivery event recording failed",
                    extra={
                        "profile_id": profile_id,
                        "instance_id": instance_id,
                        "message_id": message_id,
                    },
                )
        await self._publish_settlement_backup(profile_id, instance_id, message_id)

    async def _settle_partial(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
        media_asset_ids: list[str],
        *,
        diagnostic_code: str,
        receipts: tuple[Any, ...] = (),
        route_umo: str = "",
    ) -> None:
        await self.settlement_repository.finalize_foreground_delivery(
            profile_id,
            instance_id,
            message_id,
            media_asset_ids=tuple(media_asset_ids),
            todo_ids=(),
            status="PARTIALLY_ATTEMPTED",
            error=str(diagnostic_code or "partially_attempted"),
            receipts=receipts,
            route_umo=route_umo,
        )
        try:
            await self._record_post_settlement_event(
                profile_id,
                instance_id,
                level="WARNING",
                category="delivery.partial",
                message="前台消息仅有部分分片进入平台调用，其余分片未发送",
                details={"message_id": message_id},
            )
        except Exception:
            logger.exception(
                "foreground partial-delivery event recording failed",
                extra={
                    "profile_id": profile_id,
                    "instance_id": instance_id,
                    "message_id": message_id,
                },
            )
        await self._publish_settlement_backup(profile_id, instance_id, message_id)

    async def _settle_foreground_cancellation(
        self,
        exc: asyncio.CancelledError,
        *,
        boundary_entered: bool,
        profile_id: str,
        instance_id: str,
        message_id: int,
        media_asset_ids: list[str],
        route_umo: str = "",
    ) -> None:
        if not boundary_entered:
            await drain_cancelled_settlement(
                self._settle_failure(
                    profile_id,
                    instance_id,
                    message_id,
                    media_asset_ids,
                    asyncio.CancelledError(),
                )
            )
            return
        if self._is_partial_cancellation(exc):
            await drain_cancelled_settlement(
                self._settle_partial_with_retry(
                    profile_id,
                    instance_id,
                    message_id,
                    media_asset_ids,
                    diagnostic_code=(
                        "foreground_dispatch_cancelled_after_partial_platform_boundary"
                    ),
                    receipts=tuple(getattr(exc, "receipts", ()) or ()),
                    route_umo=route_umo,
                )
            )
            return
        await drain_cancelled_settlement(
            self._settle_unknown_with_retry(
                profile_id,
                instance_id,
                message_id,
                media_asset_ids,
                error="foreground_dispatch_cancelled_after_platform_boundary",
            )
        )

    @staticmethod
    def _is_partial_cancellation(exc: asyncio.CancelledError) -> bool:
        chunks = max(0, int(getattr(exc, "chunks", 0) or 0))
        attempted = max(0, int(getattr(exc, "attempted_chunks", 0) or 0))
        return 0 < attempted < chunks

    async def _settle_unknown(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
        media_asset_ids: list[str],
        *,
        error: str,
    ) -> None:
        await self.settlement_repository.finalize_foreground_delivery(
            profile_id,
            instance_id,
            message_id,
            media_asset_ids=tuple(media_asset_ids),
            todo_ids=(),
            status="UNKNOWN_AFTER_CRASH",
            error=error,
        )
        try:
            await self._record_post_settlement_event(
                profile_id,
                instance_id,
                level="WARNING",
                category="delivery.unknown",
                message="前台消息已进入平台调用边界，但取消使投递结果无法确认",
                details={"message_id": message_id},
            )
        except Exception:
            logger.exception(
                "foreground unknown delivery event recording failed",
                extra={
                    "profile_id": profile_id,
                    "instance_id": instance_id,
                    "message_id": message_id,
                },
            )
        await self._publish_settlement_backup(profile_id, instance_id, message_id)

    async def _settle_unknown_with_retry(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
        media_asset_ids: list[str],
        *,
        error: str,
    ) -> None:
        last_error: Exception | None = None
        for attempt, delay in enumerate(_UNKNOWN_SETTLEMENT_RETRY_DELAYS, start=1):
            if delay:
                await asyncio.sleep(delay)
            try:
                await self._settle_unknown(
                    profile_id,
                    instance_id,
                    message_id,
                    media_asset_ids,
                    error=error,
                )
                return
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "foreground unknown delivery settlement attempt failed",
                    extra={
                        "profile_id": profile_id,
                        "instance_id": instance_id,
                        "message_id": message_id,
                        "attempt": attempt,
                    },
                )
        assert last_error is not None
        raise last_error

    async def _settle_partial_with_retry(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
        media_asset_ids: list[str],
        *,
        diagnostic_code: str,
        receipts: tuple[Any, ...] = (),
        route_umo: str = "",
    ) -> None:
        last_error: Exception | None = None
        for attempt, delay in enumerate(_UNKNOWN_SETTLEMENT_RETRY_DELAYS, start=1):
            if delay:
                await asyncio.sleep(delay)
            try:
                await self._settle_partial(
                    profile_id,
                    instance_id,
                    message_id,
                    media_asset_ids,
                    diagnostic_code=diagnostic_code,
                    receipts=receipts,
                    route_umo=route_umo,
                )
                return
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "foreground partial delivery settlement attempt failed",
                    extra={
                        "profile_id": profile_id,
                        "instance_id": instance_id,
                        "message_id": message_id,
                        "attempt": attempt,
                    },
                )
        assert last_error is not None
        raise last_error

    async def _settle_partial_result_cancel_safely(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
        media_asset_ids: list[str],
        *,
        diagnostic_code: str,
        receipts: tuple[Any, ...] = (),
        route_umo: str = "",
    ) -> None:
        kwargs = {
            "diagnostic_code": diagnostic_code,
            "receipts": receipts,
            "route_umo": route_umo,
        }
        try:
            await self._settle_partial_with_retry(
                profile_id,
                instance_id,
                message_id,
                media_asset_ids,
                **kwargs,
            )
        except asyncio.CancelledError:
            await drain_cancelled_settlement(
                self._settle_partial_with_retry(
                    profile_id,
                    instance_id,
                    message_id,
                    media_asset_ids,
                    **kwargs,
                )
            )
            raise

    async def _publish_settlement_backup(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
    ) -> None:
        try:
            await self.settlement_repository.publish_context_backup()
        except Exception as exc:
            logger.exception(
                "foreground delivery backup publication failed",
                extra={
                    "profile_id": profile_id,
                    "instance_id": instance_id,
                    "message_id": message_id,
                },
            )
            try:
                await self._record_post_settlement_event(
                    profile_id,
                    instance_id,
                    level="ERROR",
                    category="delivery.backup",
                    message="前台消息投递结算已提交，但上下文备份发布失败",
                    details={"message_id": message_id, "error": type(exc).__name__},
                )
            except Exception:
                logger.exception(
                    "foreground delivery backup failure event recording failed",
                    extra={
                        "profile_id": profile_id,
                        "instance_id": instance_id,
                        "message_id": message_id,
                    },
                )

    async def _record_post_settlement_event(
        self,
        profile_id: str,
        instance_id: str,
        *,
        level: str,
        category: str,
        message: str,
        details: dict[str, Any],
    ) -> None:
        try:
            await record_event(
                self.event_log,
                profile_id=profile_id,
                instance_id=instance_id,
                level=level,
                category=category,
                message=message,
                details=details,
            )
        except Exception:
            logger.exception(
                "foreground post-settlement event persistence failed",
                extra={
                    "profile_id": profile_id,
                    "instance_id": instance_id,
                    "category": category,
                },
            )
