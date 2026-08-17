"""Foreground platform-boundary dispatch command."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ...features.delivery.transport import DeliveryStatus
from .umo import CapturedUMO

logger = logging.getLogger(__name__)


async def drain_cancelled_settlement(pending: Awaitable[None]) -> None:
    """Finish one durable settlement before propagating owner cancellation."""

    owner = asyncio.create_task(pending)
    while True:
        try:
            await asyncio.shield(owner)
            return
        except asyncio.CancelledError:
            if owner.done():
                owner.result()
                return


async def drain_cancelled_result(owner: asyncio.Task[Any]) -> Any:
    """Recover a shielded operation's result before propagating cancellation."""

    while True:
        try:
            return await asyncio.shield(owner)
        except asyncio.CancelledError:
            if owner.done():
                return owner.result()


@dataclass(slots=True)
class ForegroundPlatformBoundaryState:
    entered: bool = False


@dataclass(frozen=True, slots=True)
class ResolvedForegroundAssets:
    sticker_items: tuple[Any, ...]
    sticker_asset_ids: tuple[str, ...]


class ForegroundDeliveryNotAttempted(RuntimeError):
    """A controlled gate rejection before any platform call."""


class ForegroundSendOutcome(StrEnum):
    SENT = "SENT"
    PARTIALLY_ATTEMPTED = "PARTIALLY_ATTEMPTED"
    DEFINITELY_NOT_ATTEMPTED = "DEFINITELY_NOT_ATTEMPTED"
    OWNERSHIP_LOST = "OWNERSHIP_LOST"


@dataclass(slots=True)
class ForegroundDispatchCommand:
    controller: Any
    event: Any
    captured: CapturedUMO | None
    profile_id: str
    instance_id: str
    entry: Any
    text: str
    assets: Any
    media_asset_ids: list[str]
    sticker_ref_ids: list[str]
    run_id: int
    qpm_reservation: Any | None
    configured_group_limit: int
    boundary: ForegroundPlatformBoundaryState = field(
        default_factory=ForegroundPlatformBoundaryState
    )

    async def run(self) -> ForegroundSendOutcome:
        try:
            delivery_result = await self._prepare_and_dispatch()
        except asyncio.CancelledError as original:
            await self._settle_cancellation(original)
            raise
        except ForegroundDeliveryNotAttempted as exc:
            await self.controller._settle_failure(
                self.profile_id,
                self.instance_id,
                self.entry.message_id,
                self.media_asset_ids,
                exc,
            )
            return ForegroundSendOutcome.DEFINITELY_NOT_ATTEMPTED
        except Exception as exc:
            await self._settle_dispatch_exception(exc)
            raise
        if not self.boundary.entered:
            return ForegroundSendOutcome.OWNERSHIP_LOST
        return await self._settle_result(delivery_result)

    async def _prepare_and_dispatch(self) -> Any:
        if not await self.controller.conversation_repository.claim_foreground_delivery_preparation(
            self.profile_id,
            self.instance_id,
            self.entry.message_id,
        ):
            return None
        await self.controller._link_assets(
            self.profile_id,
            self.instance_id,
            self.entry.message_id,
            self.media_asset_ids,
            self.assets.sticker_asset_ids,
        )
        await self._require_delivery_allowed()
        chain = await self.controller._message_chain(
            self.profile_id,
            self.instance_id,
            self.text,
            self.media_asset_ids,
            self.assets,
        )
        await self._require_delivery_allowed()
        return await self.controller._dispatch(
            self.event,
            chain,
            self.captured,
            self.profile_id,
            self.instance_id,
            self.configured_group_limit,
            self.qpm_reservation,
            self._begin_platform_call,
        )

    async def _require_delivery_allowed(self) -> None:
        rejection = await self.controller._pending_delivery_rejection(
            self.profile_id,
            self.instance_id,
            self.assets,
        )
        if rejection:
            raise ForegroundDeliveryNotAttempted(rejection)

    async def _begin_platform_call(self) -> bool:
        return await self.controller._begin_foreground_platform_delivery(
            self.profile_id,
            self.instance_id,
            self.entry.message_id,
            self.boundary,
        )

    async def _settle_cancellation(self, original: asyncio.CancelledError) -> None:
        try:
            await self.controller._settle_foreground_cancellation(
                original,
                boundary_entered=self.boundary.entered,
                profile_id=self.profile_id,
                instance_id=self.instance_id,
                message_id=self.entry.message_id,
                media_asset_ids=self.media_asset_ids,
                route_umo=self._route_umo(),
            )
        except Exception as settlement_error:
            original.add_note(
                "cancelled foreground delivery settlement failed after retries: "
                f"{type(settlement_error).__name__}: {settlement_error}"
            )
            logger.exception(
                "cancelled foreground delivery settlement failed",
                extra=self._log_context(platform_called=self.boundary.entered),
            )

    async def _settle_dispatch_exception(self, exc: Exception) -> None:
        if self.boundary.entered:
            await self.controller._settle_unknown_with_retry(
                self.profile_id,
                self.instance_id,
                self.entry.message_id,
                self.media_asset_ids,
                error=(
                    f"foreground_dispatch_exception_after_platform_boundary:{type(exc).__name__}"
                ),
            )
            return
        await self.controller._settle_failure(
            self.profile_id,
            self.instance_id,
            self.entry.message_id,
            self.media_asset_ids,
            exc,
        )

    async def _settle_result(self, delivery_result: Any) -> ForegroundSendOutcome:
        if (
            delivery_result is not None
            and delivery_result.status is DeliveryStatus.PARTIALLY_ATTEMPTED
        ):
            await self.controller._settle_partial_result_cancel_safely(
                self.profile_id,
                self.instance_id,
                self.entry.message_id,
                self.media_asset_ids,
                diagnostic_code=delivery_result.diagnostic_code,
                receipts=tuple(delivery_result.receipts),
                route_umo=self._route_umo(),
            )
            return ForegroundSendOutcome.PARTIALLY_ATTEMPTED
        await self._settle_success_cancel_safely(delivery_result)
        return ForegroundSendOutcome.SENT

    async def _settle_success_cancel_safely(self, delivery_result: Any) -> None:
        receipts = tuple(getattr(delivery_result, "receipts", ()) or ())
        settlement = {
            "receipts": receipts,
            "route_umo": self._route_umo(),
        }
        try:
            await self._settle_success(**settlement)
        except asyncio.CancelledError:
            await drain_cancelled_settlement(self._settle_success(**settlement))
            raise
        except Exception as exc:
            await self._mark_success_settlement_unknown(exc)
            raise

    async def _settle_success(
        self,
        *,
        receipts: tuple[Any, ...],
        route_umo: str,
    ) -> None:
        await self.controller._settle_success(
            self.profile_id,
            self.instance_id,
            self.entry.message_id,
            self.run_id,
            self.media_asset_ids,
            self.sticker_ref_ids,
            sticker_items=tuple(self.assets.sticker_items),
            receipts=receipts,
            route_umo=route_umo,
        )

    async def _mark_success_settlement_unknown(self, exc: Exception) -> None:
        try:
            await self.controller._settle_unknown_with_retry(
                self.profile_id,
                self.instance_id,
                self.entry.message_id,
                self.media_asset_ids,
                error=f"foreground_success_settlement_failed:{type(exc).__name__}",
            )
        except Exception as settlement_error:
            exc.add_note(
                "foreground uncertainty settlement failed after retries: "
                f"{type(settlement_error).__name__}: {settlement_error}"
            )
            logger.exception(
                "foreground delivery uncertainty settlement failed",
                extra=self._log_context(),
            )

    def _route_umo(self) -> str:
        return str(getattr(self.captured, "raw", "") or "")

    def _log_context(self, **extra: Any) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "instance_id": self.instance_id,
            "message_id": self.entry.message_id,
            **extra,
        }
