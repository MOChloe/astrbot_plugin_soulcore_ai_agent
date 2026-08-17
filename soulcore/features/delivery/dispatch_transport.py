"""Final transport stage for a prepared Outbox dispatch."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ...contracts.models import OutboxStatus
from ...shared.contact_runtime import CONTACT_POLICY_DISABLED_REASON
from ..files.service import FILE_ARTIFACTS_DISABLED_REASON
from .dispatch_context import OutboxDispatchContext


class VoiceArtifactDispatchError(RuntimeError):
    """The short-lived voice artifact disappeared before transport."""


class OutboxPreparedTransportMixin:
    async def _send_prepared_dispatch(self, ctx: OutboxDispatchContext) -> Any | None:
        try:
            return await self._run_prepared_dispatch(ctx)
        except asyncio.CancelledError as exc:
            self._note_qpm_cancellation(ctx, exc)
            await self._drain_cancelled_dispatch_settlement(ctx)
            raise
        except Exception as exc:
            return await self._settle_delivery_exception(ctx, exc)

    async def _run_prepared_dispatch(self, ctx: OutboxDispatchContext) -> Any | None:
        if not await self._prepared_dispatch_requirements_met(ctx):
            return None
        workspace = (
            TemporaryDirectory(prefix="soulcore-sticker-delivery-")
            if self._has_sticker_components(ctx)
            else nullcontext(None)
        )
        with workspace as temporary_directory:
            temporary_root = Path(temporary_directory) if temporary_directory else None
            body = await self._build_body_with_voice_fallback(ctx, temporary_root)
            kwargs = await self._delivery_send_kwargs(ctx)
            if not await self._transport_still_allowed(ctx):
                return None
            self._attach_expression_send_authority(ctx, kwargs)
            self._attach_runtime_policy_send_fence(ctx, kwargs)
            chain = self._apply_dispatch_addressing(ctx, body)
            self._apply_reply_target_kwargs(kwargs, ctx.item.payload)
            result = await self._send_with_voice_text_fallback(
                ctx,
                chain=chain,
                kwargs=kwargs,
                temporary_root=temporary_root,
            )
            if ctx.runtime_policy_rejection_reason:
                return None
            ctx.authoritative_delivery_result = result
            await self._settle_expression_send_authority(ctx, result)
            return result

    async def _prepared_dispatch_requirements_met(self, ctx: OutboxDispatchContext) -> bool:
        if not await self._sticker_components_enabled(ctx):
            self._remove_sticker_components(ctx)
            if not ctx.content and not ctx.components:
                await self._settle_sticker_disabled_dispatch(ctx)
                return False
        if not await self._prepare_group_expression_dispatch(ctx):
            return False
        if self._is_file_expression(ctx.current.payload) and not (
            await self.files.get_profile_file_artifacts_enabled(ctx.profile_id)
        ):
            await self._defer_for_disabled_file_artifacts(ctx)
            return False
        return True

    async def _settle_sticker_disabled_dispatch(self, ctx: OutboxDispatchContext) -> None:
        await self.outbox_settlement.finalize_instance_outbox_delivery(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.outbox_id,
            OutboxStatus.CANCELLED,
            error="sticker_disabled",
        )
        await self._settle_contact_outbox(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.payload,
            delivered=False,
            reason="sticker_disabled",
        )
        await self._release_reserved_expression_permit(ctx, "sticker_disabled")

    async def _defer_for_disabled_file_artifacts(self, ctx: OutboxDispatchContext) -> None:
        await self._transition_scoped_outbox(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.outbox_id,
            OutboxStatus.PENDING,
            error=FILE_ARTIFACTS_DISABLED_REASON,
        )
        await self._release_reserved_expression_permit(ctx, FILE_ARTIFACTS_DISABLED_REASON)
        if ctx.important_todo_ids:
            await self.files.settle_file_todos(
                ctx.profile_id,
                ctx.instance_id,
                ctx.important_todo_ids,
                status="PENDING",
                error=FILE_ARTIFACTS_DISABLED_REASON,
            )
        await self._record_outbox_wait_once(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.outbox_id,
            FILE_ARTIFACTS_DISABLED_REASON,
        )

    async def _build_body_with_voice_fallback(
        self,
        ctx: OutboxDispatchContext,
        temporary_root: Path | None,
    ) -> Any:
        try:
            return await self._build_dispatch_body(ctx, temporary_root=temporary_root)
        except VoiceArtifactDispatchError:
            if not await self._persist_voice_text_fallback(
                ctx,
                reason="AUDIO_ARTIFACT_INVALID",
            ):
                raise
            return await self._build_dispatch_body(
                ctx,
                temporary_root=temporary_root,
                include_voice=False,
            )

    async def _transport_still_allowed(self, ctx: OutboxDispatchContext) -> bool:
        allowed, reason = await self._runtime_policy_allows_outbox(
            ctx.profile_id,
            ctx.instance_id,
            ctx.current,
        )
        if not allowed:
            ctx.runtime_policy_rejection_reason = reason
            if reason in {"instance_disabled", "instance_image_send_disabled"}:
                await self._release_reserved_expression_permit(ctx, reason)
                return False
            await self._transition_scoped_outbox(
                ctx.profile_id,
                ctx.instance_id,
                ctx.item.outbox_id,
                OutboxStatus.PENDING,
                error="profile_disabled_waiting",
            )
            await self._release_reserved_expression_permit(ctx, "profile_disabled_waiting")
            return False
        if await self._cancel_disabled_contact_outbox(
            ctx.profile_id,
            ctx.instance_id,
            ctx.current,
        ):
            await self._release_reserved_expression_permit(ctx, CONTACT_POLICY_DISABLED_REASON)
            return False
        return True

    def _attach_runtime_policy_send_fence(
        self,
        ctx: OutboxDispatchContext,
        kwargs: dict[str, Any],
    ) -> None:
        previous = kwargs.get("before_platform_call")

        async def before_platform_call() -> bool:
            if not await self._transport_still_allowed(ctx):
                return False
            if previous is None:
                return True
            return bool(await previous())

        kwargs["before_platform_call"] = before_platform_call

    async def _send_with_voice_text_fallback(
        self,
        ctx: OutboxDispatchContext,
        *,
        chain: Any,
        kwargs: dict[str, Any],
        temporary_root: Path | None,
    ) -> Any:
        result = await self.delivery.send(ctx.item.umo, chain, **kwargs)
        if ctx.runtime_policy_rejection_reason:
            return result
        should_fallback = self._voice_result_allows_text_fallback(ctx, result)
        if not should_fallback or not await self._persist_voice_text_fallback(
            ctx,
            reason="PLATFORM_DELIVERY_FAILED",
        ):
            return result
        fallback_body = await self._build_dispatch_body(
            ctx,
            temporary_root=temporary_root,
            include_voice=False,
        )
        fallback_kwargs = self._voice_text_fallback_kwargs(ctx, kwargs)
        fallback_chain = self._apply_dispatch_addressing(ctx, fallback_body)
        return await self.delivery.send(ctx.item.umo, fallback_chain, **fallback_kwargs)

    @staticmethod
    def _voice_text_fallback_kwargs(
        ctx: OutboxDispatchContext,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        fallback_kwargs = dict(kwargs)
        if ctx.expression_qpm_reservation is not None:
            # The definite preparation failure released this reservation.  The
            # text attempt must acquire a fresh physical-send reservation.
            fallback_kwargs.pop("qpm_reservation", None)
            fallback_kwargs.pop("qpm_dispatch_fence", None)
        return fallback_kwargs
