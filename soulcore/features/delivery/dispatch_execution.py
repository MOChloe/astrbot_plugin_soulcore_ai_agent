"""Message construction, transport invocation and Outbox settlement."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from ...contracts.models import OutboxItem, OutboxStatus
from ...shared.contact_runtime import (
    CONTACT_POLICY_DISABLED_REASON,
    contact_policy_enabled,
    is_autonomous_contact,
    supersede_contact_attempt,
)
from ...shared.event_log import record_event
from ..files.service import verify_artifact
from .addressing import addressing_prefixes
from .dispatch_context import OutboxDispatchContext, OutboxDispatchPreparationMixin
from .dispatch_settlement import OutboxDispatchSettlementMixin
from .dispatch_transport import OutboxPreparedTransportMixin, VoiceArtifactDispatchError
from .qpm import QPMDispatchFence
from .sticker_delivery import prepare_sticker_delivery_image
from .transport import DeliveryMessage, DeliveryResult, DeliveryStatus
from .voice_delivery import VoiceDeliveryMixin

logger = logging.getLogger(__name__)

INSTANCE_RUNTIME_DISABLED_REASON = "instance_disabled"
INSTANCE_IMAGE_SEND_DISABLED_REASON = "instance_image_send_disabled"


class OutboxDispatchExecutionMixin(
    OutboxPreparedTransportMixin,
    VoiceDeliveryMixin,
    OutboxDispatchSettlementMixin,
    OutboxDispatchPreparationMixin,
):
    @staticmethod
    def _is_image_outbox(item: OutboxItem) -> bool:
        payload = item.payload
        if str(payload.get("expression_kind") or "").upper() == "IMAGE":
            return True
        return any(
            isinstance(component, dict)
            and str(component.get("type") or "").lower() == "image_asset"
            for component in list(payload.get("components") or [])
        )

    async def _outbox_runtime_rejection(
        self,
        profile_id: str,
        instance_id: str,
        item: OutboxItem,
    ) -> str:
        decision = await self.runtime_gate.decision(profile_id, instance_id)
        if not decision.enabled:
            return str(decision.reason or "runtime_disabled")
        if self._is_image_outbox(item) and not await self.runtime_gate.image_send_enabled(
            profile_id,
            instance_id,
        ):
            return INSTANCE_IMAGE_SEND_DISABLED_REASON
        return ""

    async def _cancel_instance_policy_outbox(
        self,
        profile_id: str,
        instance_id: str,
        item: OutboxItem,
        *,
        reason: str,
    ) -> bool:
        if reason not in {
            INSTANCE_RUNTIME_DISABLED_REASON,
            INSTANCE_IMAGE_SEND_DISABLED_REASON,
        }:
            return False
        changed = await self._transition_scoped_outbox(
            profile_id,
            instance_id,
            item.outbox_id,
            OutboxStatus.CANCELLED,
            error=reason,
        )
        if changed and is_autonomous_contact(item.payload):
            await supersede_contact_attempt(
                self.timeline,
                profile_id,
                instance_id,
                item.payload,
                task_id=int(item.payload.get("ai_task_id") or 0) or None,
            )
        return changed

    async def _runtime_policy_allows_outbox(
        self,
        profile_id: str,
        instance_id: str,
        item: OutboxItem,
    ) -> tuple[bool, str]:
        reason = await self._outbox_runtime_rejection(profile_id, instance_id, item)
        if not reason:
            return True, ""
        if reason in {
            INSTANCE_RUNTIME_DISABLED_REASON,
            INSTANCE_IMAGE_SEND_DISABLED_REASON,
        }:
            await self._cancel_instance_policy_outbox(
                profile_id,
                instance_id,
                item,
                reason=reason,
            )
        return False, reason

    async def _drain_cancelled_dispatch_settlement(
        self,
        ctx: OutboxDispatchContext,
    ) -> None:
        settlement = asyncio.create_task(
            self._settle_cancelled_dispatch(ctx),
            name="soulcore-cancelled-outbox-dispatch-settlement",
        )
        current = asyncio.current_task()
        assert current is not None
        observed_cancellations = current.cancelling()
        while True:
            try:
                await asyncio.shield(settlement)
                break
            except asyncio.CancelledError:
                current_cancellations = current.cancelling()
                if current_cancellations > observed_cancellations:
                    observed_cancellations = current_cancellations
                    continue
                logger.error("cancelled Outbox settlement task was cancelled")
                break
            except Exception:
                logger.exception("cancelled dispatch settlement failed")
                break

    async def _settle_delivery_exception(
        self,
        ctx: OutboxDispatchContext,
        exc: Exception,
    ) -> Any | None:
        if ctx.authoritative_delivery_result is not None:
            logger.exception(
                "delivery authority settlement failed after an authoritative result",
                extra={
                    "profile_id": ctx.profile_id,
                    "instance_id": ctx.instance_id,
                    "outbox_id": ctx.item.outbox_id,
                },
            )
            return ctx.authoritative_delivery_result
        if ctx.expression_send_permit_started:
            await self._mark_expression_permit_attempted(
                ctx, f"delivery_exception:{type(exc).__name__}"
            )
        else:
            await self._release_reserved_expression_permit(
                ctx, f"delivery_exception:{type(exc).__name__}"
            )
        await self._handle_delivery_exception(ctx, exc)
        return None

    def _attach_expression_send_authority(
        self,
        ctx: OutboxDispatchContext,
        kwargs: dict[str, Any],
    ) -> None:
        if ctx.expression_qpm_reservation is not None:
            kwargs["qpm_reservation"] = ctx.expression_qpm_reservation
            kwargs["qpm_dispatch_fence"] = ctx.expression_qpm_dispatch_fence
            return
        if ctx.expression_send_permit_id is None:
            return

        async def begin_platform_call() -> bool:
            return await self._begin_expression_send_permit(ctx)

        kwargs["before_platform_call"] = begin_platform_call

    async def _settle_expression_send_authority(
        self,
        ctx: OutboxDispatchContext,
        result: DeliveryResult,
    ) -> None:
        if ctx.expression_qpm_reservation is not None:
            await self.delivery.cancel_expression_outbox(ctx.expression_qpm_reservation)
            ctx.expression_qpm_reservation = None
            ctx.expression_qpm_dispatch_fence = None
            return
        if bool(getattr(result, "platform_called", False)):
            await self._mark_expression_permit_attempted(
                ctx,
                self._delivery_result_reason(result),
            )
            return
        await self._release_reserved_expression_permit(
            ctx,
            self._delivery_result_reason(result),
        )

    @staticmethod
    def _note_qpm_cancellation(
        ctx: OutboxDispatchContext,
        exc: asyncio.CancelledError,
    ) -> None:
        chunks = max(0, int(getattr(exc, "chunks", 0) or 0))
        attempted_chunks = max(
            0,
            int(getattr(exc, "attempted_chunks", 0) or 0),
        )
        if chunks:
            ctx.delivery_chunks = chunks
            ctx.delivery_attempted_chunks = min(chunks, attempted_chunks)
            ctx.delivery_receipts = tuple(getattr(exc, "receipts", ()) or ())
        platform_attempted = bool(getattr(exc, "platform_attempted", False))
        if platform_attempted:
            ctx.expression_send_permit_started = True
        elif (
            ctx.expression_qpm_reservation is not None or ctx.expression_send_permit_id is not None
        ):
            ctx.expression_send_permit_started = False

    async def _settle_cancelled_dispatch(self, ctx: OutboxDispatchContext) -> None:
        if ctx.authoritative_delivery_result is not None:
            try:
                await self._settle_expression_send_authority(
                    ctx,
                    ctx.authoritative_delivery_result,
                )
            except asyncio.CancelledError:
                logger.warning(
                    "delivery authority retry was cancelled after result capture",
                    extra={
                        "profile_id": ctx.profile_id,
                        "instance_id": ctx.instance_id,
                        "outbox_id": ctx.item.outbox_id,
                    },
                )
            except Exception:
                logger.exception(
                    "delivery authority cancellation settlement failed after result capture",
                    extra={
                        "profile_id": ctx.profile_id,
                        "instance_id": ctx.instance_id,
                        "outbox_id": ctx.item.outbox_id,
                    },
                )
            await self._finalize_dispatch(ctx, ctx.authoritative_delivery_result)
            return
        if ctx.expression_send_permit_started:
            await self._mark_expression_permit_attempted(
                ctx, "delivery_cancelled_after_platform_boundary"
            )
            result = self._cancelled_delivery_result(
                ctx,
                "delivery_cancelled_after_platform_boundary",
            )
            await self._finalize_dispatch(
                ctx,
                result,
            )
            return
        released = await self._release_reserved_expression_permit(
            ctx, "delivery_cancelled_before_platform_boundary"
        )
        if not released:
            ctx.expression_send_permit_started = True
            await self._mark_expression_permit_attempted(
                ctx, "delivery_cancelled_permit_rollback_failed"
            )
            await self._finalize_dispatch(
                ctx,
                self._cancelled_delivery_result(
                    ctx,
                    "delivery_cancelled_permit_rollback_failed",
                    rollback_failed=True,
                ),
            )
            return
        await self._transition_scoped_outbox(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.outbox_id,
            OutboxStatus.PENDING,
            error="delivery_cancelled_before_platform_boundary",
        )

    @staticmethod
    def _cancelled_delivery_result(
        ctx: OutboxDispatchContext,
        diagnostic_code: str,
        *,
        rollback_failed: bool = False,
    ) -> DeliveryResult:
        chunks = int(ctx.delivery_chunks)
        attempted = int(ctx.delivery_attempted_chunks)
        if rollback_failed:
            if chunks == 0:
                chunks = 1
            attempted = min(chunks, attempted + 1)
        if 0 < attempted < chunks:
            return DeliveryResult.partially_attempted(
                diagnostic_code,
                receipts=ctx.delivery_receipts,
            )
        return DeliveryResult.accepted_unconfirmed(
            diagnostic_code,
            receipts=ctx.delivery_receipts,
        )

    async def _prepare_group_expression_dispatch(self, ctx: OutboxDispatchContext) -> bool:
        if ctx.expression_qpm_reservation is None and ctx.expression_send_permit_id is None:
            return True
        group_window_id = str(ctx.item.payload.get("group_window_id") or "").strip()
        if not group_window_id:
            return True
        if ctx.expression_qpm_reservation is not None:
            fence = QPMDispatchFence(
                ctx.profile_id,
                ctx.instance_id,
                group_window_id,
                ctx.item.outbox_id,
            )
            result = await self.delivery.prepare_expression_outbox(
                ctx.expression_qpm_reservation,
                fence,
            )
            prepared = result.allowed
            reason = str(result.reason or "unknown").upper()
            payload = result.payload
            already_started = result.already_started
            ctx.expression_qpm_dispatch_fence = fence
        else:
            result = await self.outbox.prepare_group_expression_dispatch(
                ctx.expression_send_permit_id,
                profile_id=ctx.profile_id,
                instance_id=ctx.instance_id,
                group_window_id=group_window_id,
                outbox_id=ctx.item.outbox_id,
            )
            prepared = bool(result.get("prepared"))
            reason = str(result.get("reason") or "UNKNOWN")
            payload = result.get("payload")
            already_started = reason == "ALREADY_STARTED"
        if prepared and isinstance(payload, dict):
            ctx.item = replace(ctx.item, payload=dict(payload))
            ctx.current = replace(ctx.current, payload=dict(payload))
            return True
        if prepared:
            return True
        if already_started:
            await self._mark_expression_dispatch_unknown(ctx)
            return False
        if reason == "INVALID_IDENTITY":
            await self.outbox_settlement.finalize_instance_outbox_delivery(
                ctx.profile_id,
                ctx.instance_id,
                ctx.item.outbox_id,
                OutboxStatus.FAILED,
                error_code="EXPRESSION_PERMIT_IDENTITY_MISMATCH",
                error="group_dispatch_prepare:invalid_identity",
            )
            await self._release_reserved_expression_permit(
                ctx,
                "group_dispatch_prepare:invalid_identity",
            )
            return False
        await self._release_reserved_expression_permit(
            ctx, f"group_dispatch_prepare:{reason.lower()}"
        )
        await self._transition_scoped_outbox(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.outbox_id,
            OutboxStatus.PENDING,
            error=f"group_dispatch_prepare:{reason.lower()}",
        )
        return prepared

    async def _cancel_disabled_contact_outbox(
        self,
        profile_id: str,
        instance_id: str,
        item: OutboxItem,
    ) -> bool:
        if not is_autonomous_contact(item.payload):
            return False
        if await contact_policy_enabled(self.timeline, profile_id, instance_id):
            return False
        await self._transition_scoped_outbox(
            profile_id,
            instance_id,
            item.outbox_id,
            OutboxStatus.CANCELLED,
            error=CONTACT_POLICY_DISABLED_REASON,
        )
        await supersede_contact_attempt(
            self.timeline,
            profile_id,
            instance_id,
            item.payload,
            task_id=int(item.payload.get("ai_task_id") or 0) or None,
        )
        return True

    async def _begin_expression_send_permit(self, ctx: OutboxDispatchContext) -> bool:
        if ctx.expression_send_permit_id is None:
            return True

        async def begin() -> bool:
            group_window_id = str(ctx.item.payload.get("group_window_id") or "").strip()
            if group_window_id:
                result = await self.outbox.begin_group_expression_send_permit(
                    ctx.expression_send_permit_id,
                    profile_id=ctx.profile_id,
                    instance_id=ctx.instance_id,
                    group_window_id=group_window_id,
                    outbox_id=ctx.item.outbox_id,
                )
                return bool(result.get("started"))
            return bool(
                await self.outbox.begin_dispatch_platform_send_permit(
                    ctx.expression_send_permit_id,
                    profile_id=ctx.profile_id,
                    instance_id=ctx.instance_id,
                    outbox_id=ctx.item.outbox_id,
                )
            )

        task = asyncio.create_task(begin())
        try:
            started = await asyncio.shield(task)
        except asyncio.CancelledError:
            started = await asyncio.shield(task)
            ctx.expression_send_permit_started = started
            raise
        ctx.expression_send_permit_started = started
        return started

    async def _mark_expression_permit_attempted(
        self, ctx: OutboxDispatchContext, detail: str
    ) -> bool:
        if ctx.expression_send_permit_id is None:
            return False
        group_window_id = str(ctx.item.payload.get("group_window_id") or "").strip()
        return bool(
            await self.outbox.mark_platform_send_permit_attempted_unknown(
                ctx.expression_send_permit_id,
                detail=str(detail or ""),
                profile_id=ctx.profile_id if group_window_id else "",
                instance_id=ctx.instance_id if group_window_id else "",
                group_window_id=group_window_id,
                outbox_id=ctx.item.outbox_id if group_window_id else None,
            )
        )

    async def _release_reserved_expression_permit(
        self, ctx: OutboxDispatchContext, detail: str
    ) -> bool:
        if ctx.expression_qpm_reservation is not None:
            await self.delivery.cancel_expression_outbox(ctx.expression_qpm_reservation)
            ctx.expression_qpm_reservation = None
            ctx.expression_qpm_dispatch_fence = None
            return True
        if ctx.expression_send_permit_id is None:
            return True
        released = bool(
            await self.outbox.fail_platform_send_permit_before_dispatch(
                ctx.expression_send_permit_id, detail=str(detail or "")
            )
        )
        if released:
            ctx.expression_send_permit_id = None
        return released

    async def _mark_expression_dispatch_unknown(self, ctx: OutboxDispatchContext) -> None:
        await self.outbox_settlement.finalize_instance_outbox_delivery(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.outbox_id,
            OutboxStatus.UNKNOWN_AFTER_CRASH,
            diagnostic_code="expression_send_permit_already_started",
        )

    @staticmethod
    def _has_sticker_components(ctx: OutboxDispatchContext) -> bool:
        return any(
            str(component.get("type") or "") == "sticker_ref" for component in ctx.components
        )

    async def _build_dispatch_body(
        self,
        ctx: OutboxDispatchContext,
        *,
        temporary_root: Path | None = None,
        include_voice: bool = True,
    ) -> DeliveryMessage:
        components: list[dict[str, object]] = []
        for component in ctx.components:
            await self._append_dispatch_component(
                ctx,
                components,
                component,
                temporary_root=temporary_root,
            )
        if ctx.voice_artifact is not None and include_voice:
            try:
                path = await asyncio.to_thread(
                    self.voice_artifact_service.resolve,
                    ctx.voice_artifact,
                    touch=True,
                )
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                raise VoiceArtifactDispatchError(
                    "controlled voice artifact is unavailable"
                ) from exc
            components.append(
                {
                    "type": "audio_record",
                    "path": str(path),
                    "mime_type": ctx.voice_artifact.mime_type,
                    "filename": ctx.voice_artifact.filename,
                }
            )
            ctx.voice_delivered = True
            return DeliveryMessage(content="", components=tuple(components))
        if not include_voice:
            ctx.voice_delivered = False
        return DeliveryMessage(content=ctx.content, components=tuple(components))

    @staticmethod
    def _voice_result_allows_text_fallback(
        ctx: OutboxDispatchContext,
        result: DeliveryResult,
    ) -> bool:
        return bool(
            ctx.voice_delivered
            and result.status is DeliveryStatus.FAILED
            and not result.platform_called
            and not ctx.expression_send_permit_started
        )

    def _apply_dispatch_addressing(
        self,
        ctx: OutboxDispatchContext,
        body: DeliveryMessage,
    ) -> DeliveryMessage:
        capability = self._delivery_addressing_capability(ctx.item.umo)
        prefix_components, fallback_prefix = addressing_prefixes(
            ctx.item.payload,
            native_reply_supported=bool(capability and capability.quote),
            native_mention_supported=bool(capability and capability.mention),
        )
        content = "\n".join([*fallback_prefix, body.content]).strip()
        return DeliveryMessage(
            content=content,
            components=(*prefix_components, *body.components),
        )

    def _delivery_addressing_capability(self, umo: str) -> Any | None:
        return self.delivery.capability_for(umo)

    async def _append_dispatch_component(
        self,
        ctx: OutboxDispatchContext,
        output: list[dict[str, object]],
        component: dict[str, Any],
        *,
        temporary_root: Path | None = None,
    ) -> None:
        component_type = str(component.get("type") or "")
        if component_type == "file_artifact":
            self._append_file_component(ctx, output, component)
            return
        if component_type not in {"image_asset", "sticker_ref"}:
            return
        asset_id = self._component_asset_id(ctx, component)
        path = await self.visual_service.asset_file_path(
            profile_id=ctx.profile_id,
            instance_id=ctx.instance_id,
            asset_id=asset_id,
        )
        if not path:
            raise RuntimeError("selected image asset is unavailable")
        if component_type == "sticker_ref":
            if temporary_root is None:
                raise RuntimeError("sticker delivery workspace is unavailable")
            path = await asyncio.to_thread(
                prepare_sticker_delivery_image,
                path,
                temporary_root,
                idempotency_key=ctx.item.idempotency_key,
                asset_id=asset_id,
            )
        output.append({"type": "image_file", "path": str(path)})

    def _append_file_component(
        self,
        ctx: OutboxDispatchContext,
        output: list[dict[str, object]],
        component: dict[str, Any],
    ) -> None:
        row = next(
            item
            for item in ctx.file_deliveries
            if str(item.get("asset_id") or "") == str(component.get("asset_id") or "")
        )
        path = self.file_artifact_service.resolve_path(str(row.get("storage_relpath") or ""))
        if not verify_artifact(
            path,
            int(row.get("byte_size") or 0),
            str(row.get("sha256") or ""),
        ):
            raise RuntimeError("controlled file integrity check failed")
        output.append(
            {
                "type": "file",
                "path": str(path),
                "name": str(row.get("display_name") or path.name),
            }
        )

    @staticmethod
    def _component_asset_id(ctx: OutboxDispatchContext, component: dict[str, Any]) -> str:
        if str(component.get("type") or "") != "sticker_ref":
            return str(component.get("asset_id") or "")
        sticker_ref = str(component.get("sticker_ref") or "")
        delivery = next(row for row in ctx.sticker_deliveries if row["sticker_ref"] == sticker_ref)
        return str(delivery["asset_id"])

    async def _delivery_send_kwargs(self, ctx: OutboxDispatchContext) -> dict[str, Any]:
        origin_kind = str(ctx.item.payload.get("origin_kind") or "").upper()
        proactive = origin_kind not in {"FOREGROUND_MESSAGE", "DEFERRED_MESSAGE"}
        kwargs: dict[str, Any] = {
            "profile_id": ctx.profile_id,
            "proactive": proactive,
        }
        self._apply_reply_target_kwargs(kwargs, ctx.item.payload)
        kwargs["instance_id"] = ctx.instance_id
        if ctx.instance_id is None:
            return kwargs
        instance = await self.profiles.get_character_instance(ctx.profile_id, ctx.instance_id)
        if instance is None:
            return kwargs
        policy = await self.delivery_policy.get_delivery_policy(ctx.profile_id, instance.scope)
        kwargs["configured_group_limit"] = int((policy or {}).get("group_send_qpm_limit") or 20)
        return kwargs

    @staticmethod
    def _apply_reply_target_kwargs(
        kwargs: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        kwargs.pop("reply_to_platform_message_id", None)
        kwargs.pop("reply_to_platform_reference_id", None)
        reply_target = payload.get("reply_target")
        if isinstance(reply_target, dict):
            platform_message_id = str(reply_target.get("platform_message_id") or "").strip()
            platform_reference_id = str(reply_target.get("platform_reference_id") or "").strip()
            if platform_message_id:
                # Keep native addressing bound to the persistent target even if
                # an AstrBot MessageChain implementation drops an unknown Reply
                # component while copying or normalising the chain.
                kwargs["reply_to_platform_message_id"] = platform_message_id
            if platform_reference_id:
                kwargs["reply_to_platform_reference_id"] = platform_reference_id

    async def _maybe_enqueue_delivery_summary(
        self, ctx: OutboxDispatchContext, ledger_message: Any | None
    ) -> None:
        if ledger_message is None or self.context_service is None:
            return
        instance = await self.profiles.get_character_instance(ctx.profile_id, ctx.instance_id)
        if instance is None:
            return
        config = await self.profiles.get_scope_config(ctx.profile_id, instance.scope)
        if config is not None:
            await self.context_service.maybe_enqueue_summary(
                ctx.profile_id, ctx.instance_id, config
            )

    async def _record_delivery_result(
        self, ctx: OutboxDispatchContext, result: DeliveryResult, status: OutboxStatus
    ) -> None:
        level = (
            "ERROR"
            if result.status is DeliveryStatus.FAILED
            else "WARN"
            if result.status is DeliveryStatus.PARTIALLY_ATTEMPTED
            else "INFO"
        )
        details = {
            "outbox_id": ctx.item.outbox_id,
            "status": status.value,
            "transport_status": result.status.value,
            "diagnostic_code": result.diagnostic_code,
            "error_code": result.error_code,
            "error_message": result.error_message,
            "physical_receipt_count": len(result.receipts),
            "reference_receipt_count": sum(
                1 for receipt in result.receipts if str(receipt.platform_reference_id or "").strip()
            ),
        }
        await record_event(
            self.event_log,
            profile_id=ctx.profile_id,
            instance_id=ctx.instance_id,
            level=level,
            category="delivery",
            message="Outbox 投递状态已更新",
            details=details,
        )
        if ctx.voice_requested and ctx.voice_artifact is None and ctx.voice_fallback_reason:
            await record_event(
                self.event_log,
                profile_id=ctx.profile_id,
                instance_id=ctx.instance_id,
                level="WARN",
                category="voice.delivery",
                message="语音呈现已降级为普通文本投递",
                details={
                    "outbox_id": ctx.item.outbox_id,
                    "reason": ctx.voice_fallback_reason,
                },
            )
        if any(str(item.get("type") or "") == "image_asset" for item in ctx.components):
            await record_event(
                self.event_log,
                profile_id=ctx.profile_id,
                instance_id=ctx.instance_id,
                level=level,
                category="image.delivery",
                message="图片消息链投递状态已更新",
                details={
                    "outbox_id": ctx.item.outbox_id,
                    "status": status.value,
                    "asset_count": len(ctx.components),
                },
            )

    def _clear_outbox_wait_markers(self, ctx: OutboxDispatchContext) -> None:
        self._logged_outbox_waits = {
            key
            for key in self._logged_outbox_waits
            if not (
                key[0] == ctx.profile_id
                and key[1] == ctx.instance_id
                and key[2] == ctx.item.outbox_id
            )
        }


__all__ = ["OutboxDispatchExecutionMixin"]
