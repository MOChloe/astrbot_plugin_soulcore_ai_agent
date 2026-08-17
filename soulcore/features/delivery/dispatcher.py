"""Serialized Outbox dispatch with dependency and generation fences."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from ...contracts.models import OutboxItem, OutboxStatus
from ...shared.time import utcnow
from .dispatch_context import OutboxDispatchContext
from .dispatch_execution import OutboxDispatchExecutionMixin
from .routes import CapturedUMO, RouteKind


class OutboxDispatcherMixin(OutboxDispatchExecutionMixin):
    async def flush_instance_outbox(self, profile_id: str, instance_id: str) -> None:
        decision = await self.runtime_gate.decision(profile_id, instance_id)
        if not decision.enabled:
            if decision.reason == "instance_disabled":
                await self._cancel_all_pending_instance_outbox(profile_id, instance_id)
            return
        pending = await self.outbox.list_instance_outbox(
            profile_id,
            instance_id,
            status=OutboxStatus.PENDING,
            limit=10_000,
        )
        for item in reversed(pending):
            allowed, _reason = await self._runtime_policy_allows_outbox(
                profile_id,
                instance_id,
                item,
            )
            if not allowed:
                continue
            if self._is_expression_outbox(item):
                continue
            async with self._delivery_lock(profile_id, item.umo):
                await self._dispatch_one_outbox(profile_id, instance_id, item)

    async def _cancel_all_pending_instance_outbox(
        self,
        profile_id: str,
        instance_id: str,
    ) -> None:
        while True:
            pending = await self.outbox.list_instance_outbox(
                profile_id,
                instance_id,
                status=OutboxStatus.PENDING,
                limit=100,
            )
            if not pending:
                return
            changed = 0
            for item in pending:
                if await self._cancel_instance_policy_outbox(
                    profile_id,
                    instance_id,
                    item,
                    reason="instance_disabled",
                ):
                    changed += 1
            if changed == 0:
                return

    async def _dispatch_one_outbox(
        self,
        profile_id: str,
        instance_id: str,
        item: OutboxItem,
    ) -> None:
        allowed, _reason = await self._runtime_policy_allows_outbox(
            profile_id,
            instance_id,
            item,
        )
        if not allowed:
            return
        if await self._cancel_disabled_contact_outbox(profile_id, instance_id, item):
            return
        ctx = await self._prepare_dispatch_context(profile_id, instance_id, item)
        if ctx is None or not await self._dependency_allows_dispatch(ctx):
            return
        cleanup_reason = "voice_dispatch_finished"
        try:
            ready, cleanup_reason = await self._prepare_claimed_dispatch(ctx)
            if not ready:
                return
            await self._deliver_claimed_dispatch(ctx)
        except asyncio.CancelledError:
            cleanup_reason = "voice_dispatch_cancelled"
            ctx.retain_voice_artifact = False
            raise
        finally:
            if ctx.voice_artifact is not None and not ctx.retain_voice_artifact:
                await self._drain_voice_artifact_cleanup(ctx, reason=cleanup_reason)

    async def _prepare_claimed_dispatch(
        self,
        ctx: OutboxDispatchContext,
    ) -> tuple[bool, str]:
        cleanup_reason = "voice_dispatch_finished"
        if not await self._preclaim_dispatch_ready(ctx):
            return False, cleanup_reason
        await self._prepare_voice_delivery(ctx)
        # TTS may be much slower than route preparation.  Recheck the activity
        # generation before reserving the 120-second send permit.
        if ctx.voice_artifact is not None and not await self._activity_epoch_matches(
            ctx,
            after_claim=False,
        ):
            return False, "voice_interrupted_after_synthesis"
        if not await self._reserve_expression_send_permit(ctx):
            ctx.retain_voice_artifact = ctx.voice_artifact is not None
            return False, cleanup_reason
        if not await self._resolve_dispatch_components(ctx):
            await self._release_expression_send_permit(ctx, "component_resolution_failed")
            return False, cleanup_reason
        if not await self._dispatch_has_content(ctx):
            await self._release_expression_send_permit(ctx, "empty_content")
            return False, cleanup_reason
        if not await self._claim_dispatch(ctx):
            await self._release_expression_send_permit(ctx, "dispatch_claim_lost")
            await self._release_unclaimed_file_todos(ctx)
            return False, cleanup_reason
        if not await self._activity_epoch_matches(ctx, after_claim=True):
            await self._release_expression_send_permit(ctx, "activity_epoch_changed")
            return False, "voice_interrupted_after_claim"
        return True, cleanup_reason

    async def _deliver_claimed_dispatch(self, ctx: OutboxDispatchContext) -> None:
        result = await self._send_prepared_dispatch(ctx)
        if result is not None:
            ctx.retain_voice_artifact = (
                self._outbox_status_for_result(ctx, result) is OutboxStatus.PENDING
            )
            await self._finalize_dispatch_with_cancel_drain(ctx, result)
            return
        if ctx.voice_artifact is None:
            return
        current = await self.outbox.get_instance_outbox(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.outbox_id,
        )
        ctx.retain_voice_artifact = bool(
            current is not None and current.status is OutboxStatus.PENDING
        )

    async def _release_unclaimed_file_todos(self, ctx: OutboxDispatchContext) -> None:
        if ctx.important_todo_ids:
            await self.files.settle_file_todos(
                ctx.profile_id,
                ctx.instance_id,
                ctx.important_todo_ids,
                status="PENDING",
                error="dispatch_claim_lost",
            )

    async def _reserve_expression_send_permit(self, ctx: OutboxDispatchContext) -> bool:
        batch_id = str(ctx.current.expression_batch_id or "").strip()
        if not batch_id:
            return True
        captured = CapturedUMO.parse(ctx.item.umo)
        origin_kind = str(ctx.item.payload.get("origin_kind") or "").upper()
        proactive = origin_kind not in {"FOREGROUND_MESSAGE", "DEFERRED_MESSAGE"}
        if captured.kind is RouteKind.GROUP:
            return await self._reserve_group_expression_send(ctx, captured, proactive)
        return await self._reserve_direct_expression_send(ctx, captured, proactive)

    async def _reserve_group_expression_send(
        self, ctx: OutboxDispatchContext, captured: CapturedUMO, proactive: bool
    ) -> bool:
        kwargs = await self._delivery_send_kwargs(ctx)
        admission = await self.delivery.reserve_expression_outbox(
            captured,
            profile_id=ctx.profile_id,
            instance_id=ctx.instance_id,
            origin_id=f"expression-outbox:{ctx.item.outbox_id}",
            configured_group_limit=int(kwargs.get("configured_group_limit") or 20),
            proactive=proactive,
        )
        if admission is not None and admission.admitted and admission.prepared is not None:
            ctx.expression_qpm_reservation = admission.prepared
            return True
        await self._defer_unclaimed_expression_suffix(
            ctx,
            admission.next_available_at if admission is not None else None,
        )
        return False

    async def _reserve_direct_expression_send(
        self, ctx: OutboxDispatchContext, captured: CapturedUMO, proactive: bool
    ) -> bool:
        capability = self.delivery.capability_for(captured)
        account_bucket = capability.proactive_account_bucket() if proactive and capability else None
        result = await self.outbox.reserve_expression_send_permit(
            ctx.profile_id,
            ctx.instance_id,
            platform_instance_id=captured.platform_id,
            target_id=captured.target_id,
            origin_id=f"expression-outbox:{ctx.item.outbox_id}",
            account_key=account_bucket.key.identity if account_bucket is not None else "",
            account_limit=account_bucket.limit if account_bucket is not None else None,
        )
        if bool(result.get("reserved")):
            permit = result.get("permit") or {}
            ctx.expression_send_permit_id = int(permit.get("permit_id") or 0) or None
            return True
        await self._defer_unclaimed_expression_suffix(ctx, result.get("next_available_at"))
        return False

    async def _defer_unclaimed_expression_suffix(
        self, ctx: OutboxDispatchContext, next_available_at: object
    ) -> None:
        batch_id = str(ctx.current.expression_batch_id or "").strip()
        ordinal = int(ctx.current.expression_ordinal or 0)
        due_at = self._coerce_defer_time(next_available_at)
        await self.outbox.defer_expression_batch_suffix(
            ctx.profile_id,
            ctx.instance_id,
            batch_id,
            from_ordinal=ordinal,
            not_before_at=due_at,
            error="expression_send_qpm_limited",
        )

    @staticmethod
    def _coerce_defer_time(value: object) -> datetime:
        if isinstance(value, datetime):
            parsed = value
        else:
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                parsed = utcnow() + timedelta(seconds=5)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return max(parsed.astimezone(UTC), utcnow() + timedelta(seconds=1))

    async def _release_expression_send_permit(
        self, ctx: OutboxDispatchContext, reason: str
    ) -> None:
        if ctx.expression_qpm_reservation is not None:
            await self.delivery.cancel_expression_outbox(ctx.expression_qpm_reservation)
            ctx.expression_qpm_reservation = None
            ctx.expression_qpm_dispatch_fence = None
            if reason in {"component_resolution_failed", "empty_content"}:
                await self._resolve_undeliverable_group_window(ctx)
            return
        if ctx.expression_send_permit_id is None:
            return
        await self.outbox.fail_platform_send_permit_before_dispatch(
            ctx.expression_send_permit_id, detail=reason
        )
        ctx.expression_send_permit_id = None
        if reason in {"component_resolution_failed", "empty_content"}:
            await self._resolve_undeliverable_group_window(ctx)

    async def _resolve_undeliverable_group_window(self, ctx: OutboxDispatchContext) -> None:
        window_id = str(ctx.item.payload.get("group_window_id") or "").strip()
        if not window_id:
            return
        await self.outbox.resolve_group_window_if_no_deliverable(
            ctx.profile_id, ctx.instance_id, window_id
        )


__all__ = ["OutboxDispatchContext", "OutboxDispatcherMixin"]
