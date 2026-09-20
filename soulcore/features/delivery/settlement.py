"Outbox state transitions, contact settlement and retry backoff."

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any

from ...contracts.models import OutboxItem, OutboxStatus
from ...shared.event_log import record_event
from ...shared.time import utcnow
from .dispatch_context import OutboxDispatchContext
from .transport import DeliveryResult, DeliveryStatus


class OutboxSettlementMixin:
    async def _record_outbox_wait_once(
        self,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        key = (profile_id, instance_id, outbox_id, reason)
        if key in self._logged_outbox_waits:
            return
        self._logged_outbox_waits.add(key)
        await record_event(
            self.event_log,
            profile_id=profile_id,
            instance_id=instance_id,
            level="WARN",
            category="delivery",
            message="Outbox 暂未投递，等待路由恢复",
            details={"outbox_id": outbox_id, "reason": reason, **(details or {})},
        )

    async def _transition_scoped_outbox(
        self,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        status: OutboxStatus,
        *,
        error: str | None = None,
    ) -> bool:
        return await self.outbox.transition_instance_outbox(
            profile_id, instance_id, outbox_id, status, error=error
        )

    async def _settle_contact_outbox(
        self,
        profile_id: str,
        instance_id: str,
        payload: dict[str, Any],
        *,
        delivered: bool,
        reason: str,
        superseded: bool = False,
        attempted_unknown: bool = False,
    ) -> bool:
        """Settle autonomous contact only after the adapter outcome is known.

        Main Core producing text is intentionally insufficient: the attempt,
        daily quota and one-shot evidence remain untouched until this delivery
        boundary is reached.
        """

        attempt_ref = str(payload.get("contact_attempt_ref") or "").strip()
        if not attempt_ref:
            return False
        generation = int(payload.get("contact_generation") or 0)
        task_id = int(payload.get("ai_task_id") or 0) or None
        failure_mode = str(payload.get("contact_failure_mode") or "SKIP").upper()
        outcome = self._contact_outcome(
            delivered=delivered,
            attempted_unknown=attempted_unknown,
            superseded=superseded,
            failure_mode=failure_mode,
        )
        settled = await self._settle_contact_evidence(
            profile_id,
            instance_id,
            attempt_ref=attempt_ref,
            generation=generation,
            outcome=outcome,
        )
        finalized = await self._finalize_contact_attempt(
            profile_id,
            instance_id,
            attempt_ref=attempt_ref,
            generation=generation,
            task_id=task_id,
            delivered=delivered,
            attempted_unknown=attempted_unknown,
        )
        return settled or finalized

    @staticmethod
    def _contact_outcome(
        *,
        delivered: bool,
        attempted_unknown: bool,
        superseded: bool,
        failure_mode: str,
    ) -> str:
        if delivered:
            return "DELIVERED"
        if attempted_unknown:
            return "ATTEMPTED_UNKNOWN"
        # Foreground activity releases the reservation. The latest foreground
        # run may consume the projection without counting it as delivery.
        if superseded:
            return "SUPPRESSED"
        return "SUPERSEDED" if failure_mode == "SKIP" else "FAILED"

    async def _settle_contact_evidence(
        self,
        profile_id: str,
        instance_id: str,
        *,
        attempt_ref: str,
        generation: int,
        outcome: str,
    ) -> bool:
        return bool(
            await self.timeline.settle_contact_evidence(
                profile_id,
                instance_id,
                attempt_ref=attempt_ref,
                generation=generation,
                outcome=outcome,
            )
        )

    async def _finalize_contact_attempt(
        self,
        profile_id: str,
        instance_id: str,
        *,
        attempt_ref: str,
        generation: int,
        task_id: int | None,
        delivered: bool,
        attempted_unknown: bool,
    ) -> bool:
        try:
            return bool(
                await self.timeline.finalize_contact_attempt(
                    profile_id,
                    instance_id,
                    attempt_ref,
                    generation=generation,
                    attempted=bool(delivered or attempted_unknown),
                    success=bool(delivered),
                    answered=False,
                    task_id=task_id,
                )
            )
        except KeyError:
            return False

    async def _outbox_belongs_to_bound_instance(
        self, profile_id: str, instance_id: str, item: Any
    ) -> bool:
        """Reject an outbox row that drifted away from its owning instance."""

        if str(item.instance_id or "") != instance_id:
            return False
        instance = await self.profiles.get_character_instance(profile_id, instance_id)
        if instance is None:
            return False
        return str(instance.route_umo) == str(item.umo)


logger = logging.getLogger(__name__)


class OutboxDispatchSettlementMixin:
    async def _handle_delivery_exception(self, ctx: OutboxDispatchContext, exc: Exception) -> None:
        error = f"delivery_exception:{type(exc).__name__}:{exc}"
        if ctx.expression_send_permit_started:
            await self._finalize_dispatch(
                ctx,
                DeliveryResult.accepted_unconfirmed(
                    f"delivery_exception_after_platform_boundary:{type(exc).__name__}"
                ),
            )
            return
        await self._settle_contact_outbox(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.payload,
            delivered=False,
            reason=error,
        )
        await self.outbox_settlement.finalize_instance_outbox_delivery(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.outbox_id,
            OutboxStatus.FAILED,
            error_code="DELIVERY_EXCEPTION",
            error=error,
        )
        await record_event(
            self.event_log,
            profile_id=ctx.profile_id,
            instance_id=ctx.instance_id,
            level="ERROR",
            category="delivery",
            message="Outbox 投递发生异常",
            details={"outbox_id": ctx.item.outbox_id, "error": f"{type(exc).__name__}: {exc}"},
        )

    async def _finalize_dispatch(self, ctx: OutboxDispatchContext, result: Any) -> None:
        status = self._outbox_status_for_result(ctx, result)
        ledger_message = await self._finalize_outbox_record(ctx, result, status)
        await self._enforce_next_relative_delay(ctx, status)
        await self._defer_rate_limited_expression_suffix(ctx, result, status)
        if status is OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED:
            await self._maybe_enqueue_delivery_summary(ctx, ledger_message)
        await self._record_delivery_result(ctx, result, status)
        self._clear_outbox_wait_markers(ctx)

    async def _finalize_dispatch_with_cancel_drain(
        self,
        ctx: OutboxDispatchContext,
        result: Any,
    ) -> None:
        settlement = asyncio.create_task(
            self._finalize_dispatch(ctx, result),
            name="soulcore-outbox-dispatch-finalization",
        )
        try:
            await asyncio.shield(settlement)
        except asyncio.CancelledError as original:
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
                    original.add_note("outbox finalization task was cancelled")
                    break
                except Exception as settlement_error:
                    original.add_note(
                        "outbox finalization failed after platform result capture: "
                        f"{type(settlement_error).__name__}: {settlement_error}"
                    )
                    logger.exception("outbox finalization failed while draining cancellation")
                    break
            raise

    async def _enforce_next_relative_delay(
        self, ctx: OutboxDispatchContext, status: OutboxStatus
    ) -> None:
        if status is not OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED or not ctx.instance_id:
            return
        batch_id, ordinal = self._expression_identity(ctx)
        if not batch_id:
            return
        pending = await self.outbox.list_instance_outbox(
            ctx.profile_id,
            ctx.instance_id,
            status=OutboxStatus.PENDING,
            limit=100,
        )
        following = self._following_expression_item(pending, batch_id, ordinal + 1)
        if following is None:
            return
        delay = max(
            0,
            int(following.payload.get("delay_after_previous_seconds") or 0),
        )
        if delay == 0:
            return
        await self.outbox.defer_expression_batch_suffix(
            ctx.profile_id,
            ctx.instance_id,
            batch_id,
            ordinal + 1,
            utcnow() + timedelta(seconds=delay),
        )

    @staticmethod
    def _expression_identity(ctx: OutboxDispatchContext) -> tuple[str, int]:
        batch_id = str(ctx.current.expression_batch_id or "").strip()
        ordinal = int(ctx.current.expression_ordinal or 0)
        return batch_id, ordinal

    @staticmethod
    def _following_expression_item(
        pending: list[OutboxItem], batch_id: str, ordinal: int
    ) -> OutboxItem | None:
        for item in pending:
            if str(item.expression_batch_id or "") != batch_id:
                continue
            if int(item.expression_ordinal or 0) == ordinal:
                return item
        return None

    @staticmethod
    def _outbox_status_for_result(ctx: OutboxDispatchContext, result: Any) -> OutboxStatus:
        if result.status is DeliveryStatus.PARTIALLY_ATTEMPTED:
            return OutboxStatus.PARTIALLY_ATTEMPTED
        if bool(getattr(result, "platform_called", False)) or (
            result.status is DeliveryStatus.ATTEMPTED_UNKNOWN
        ):
            return OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED
        if result.status is DeliveryStatus.ROUTE_NOT_READY:
            return OutboxStatus.PENDING
        is_retryable_file = bool(ctx.important_todo_ids) or (
            str(ctx.item.payload.get("file_delivery_role") or "") == "ANNOUNCEMENT"
        )
        is_expression = bool(str(ctx.current.expression_batch_id or "").strip())
        if result.status is DeliveryStatus.RATE_LIMITED and (is_retryable_file or is_expression):
            return OutboxStatus.PENDING
        return OutboxStatus.FAILED

    @staticmethod
    def _delivery_result_reason(result: DeliveryResult) -> str:
        return str(
            result.error_message
            or result.error_code
            or result.diagnostic_code
            or result.status.value
        )

    async def _defer_rate_limited_expression_suffix(
        self,
        ctx: OutboxDispatchContext,
        result: Any,
        status: OutboxStatus,
    ) -> None:
        if status is not OutboxStatus.PENDING or result.status is not DeliveryStatus.RATE_LIMITED:
            return
        batch_id = str(ctx.current.expression_batch_id or "").strip()
        if not batch_id:
            return
        ordinal = int(ctx.current.expression_ordinal or 0)
        await self.outbox.defer_expression_batch_suffix(
            ctx.profile_id,
            ctx.instance_id,
            batch_id,
            from_ordinal=ordinal,
            not_before_at=utcnow() + timedelta(seconds=5),
            error="platform_rate_limited",
        )

    async def _finalize_outbox_record(
        self,
        ctx: OutboxDispatchContext,
        result: DeliveryResult,
        status: OutboxStatus,
    ) -> Any | None:
        _, message = await self.outbox_settlement.finalize_instance_outbox_delivery(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.outbox_id,
            status,
            error_code=result.error_code if status is OutboxStatus.FAILED else "",
            error=result.error_message if status is OutboxStatus.FAILED else None,
            diagnostic_code=result.diagnostic_code,
            context_message=self._context_message(ctx, result, status),
            receipts=result.receipts,
            sticker_deliveries=tuple(ctx.sticker_deliveries),
        )
        return message

    def _context_message(
        self, ctx: OutboxDispatchContext, result: Any, status: OutboxStatus
    ) -> dict[str, Any] | None:
        should_record = (
            self.context_service is not None
            and status is OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED
            and bool(ctx.item.payload.get("context_record", True))
        )
        if not should_record:
            return None
        plain_text = "\n".join(
            value
            for value in (
                self._history_delivery_text(ctx),
                *(row["projection"] for row in ctx.sticker_deliveries),
            )
            if value
        )
        components = [
            component
            for component in ctx.components
            if str(component.get("type") or "") != "sticker_ref"
        ] + [{"type": "sticker", "projection": row["projection"]} for row in ctx.sticker_deliveries]
        return {
            "role": "assistant",
            "sender_id": "soulcore",
            "sender_name": "SoulCore",
            "plain_text": plain_text,
            "identity_template": ctx.identity_template,
            "internal_memo": str(ctx.item.payload.get("internal_memo") or "").strip(),
            "components": components,
            "idempotency_key": f"outbox:{ctx.item.outbox_id}",
            "metadata": self._context_metadata(ctx, result),
            "knowledge_eligibility": "HELD",
            "knowledge_eligibility_reason": "delivery_unconfirmed",
        }

    @staticmethod
    def _context_metadata(ctx: OutboxDispatchContext, result: DeliveryResult) -> dict[str, Any]:
        payload = ctx.item.payload
        return {
            "outbox_id": ctx.item.outbox_id,
            "delivery_diagnostic_code": result.diagnostic_code,
            "delivery_error_code": result.error_code,
            "origin_kind": str(payload.get("origin_kind") or "CORE_RUN"),
            "contact_attempt_ref": str(payload.get("contact_attempt_ref") or ""),
            "contact_generation": int(payload.get("contact_generation") or 0),
            "contact_evidence": list(payload.get("contact_evidence") or [])[:12],
            "expression_batch_id": str(payload.get("expression_batch_id") or ""),
            "expression_ordinal": int(payload.get("expression_ordinal") or 0),
        }


__all__ = ["OutboxDispatchSettlementMixin", "OutboxSettlementMixin"]
