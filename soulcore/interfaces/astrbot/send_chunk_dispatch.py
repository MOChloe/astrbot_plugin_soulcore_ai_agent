"""Cancellation-safe physical chunk dispatch over one QPM reservation."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from ...features.delivery.capabilities import DeliveryCapability, PhysicalDeliveryReceipt
from ...features.delivery.qpm import QPMDispatchFence
from .onebot_transport import receipts_from_sender_result
from .qq_outbound_receipts import capture_qq_outbound_receipt


async def send_chunk_with_receipt(
    sender: Callable[[Any], Any], chunk: Any, platform: Any, capability: DeliveryCapability
) -> tuple[Any, Any | None]:
    if capability.qq_official:
        with capture_qq_outbound_receipt(platform) as receipt:
            value = sender(chunk)
            if inspect.isawaitable(value):
                value = await value
        return value, receipt
    value = sender(chunk)
    if inspect.isawaitable(value):
        value = await value
    return value, None


def merge_qq_receipt(
    receipts: tuple[PhysicalDeliveryReceipt, ...],
    qq_receipt: Any | None,
    index: int,
    capability: DeliveryCapability,
) -> tuple[PhysicalDeliveryReceipt, ...]:
    if qq_receipt is None or not qq_receipt.platform_reference_id:
        return receipts
    if receipts:
        first, *rest = receipts
        if not first.platform_reference_id:
            first = capability.receipt(
                first.platform_message_id,
                first.fragment_ordinal,
                accepted_unconfirmed=first.accepted_unconfirmed,
                platform_reference_id=qq_receipt.platform_reference_id,
            )
        return (first, *rest)
    if not qq_receipt.message_id:
        return receipts
    return (
        capability.receipt(
            qq_receipt.message_id,
            index,
            platform_reference_id=qq_receipt.platform_reference_id,
        ),
    )


logger = logging.getLogger(__name__)
PreparedPlatformCall = Callable[[], Awaitable[Any]]
PlatformCallPreparer = Callable[[Any], Awaitable[PreparedPlatformCall]]


class CoordinatedSendStatus(StrEnum):
    ATTEMPTED_UNKNOWN = "attempted_unknown"
    PARTIALLY_ATTEMPTED = "partially_attempted"
    RATE_LIMITED = "rate_limited"
    FAILED_BEFORE_DISPATCH = "failed_before_dispatch"


@dataclass(frozen=True, slots=True)
class CoordinatedSendResult:
    status: CoordinatedSendStatus
    detail: str
    chunks: int
    attempted_chunks: int
    reservation_id: str = ""
    receipts: tuple[PhysicalDeliveryReceipt, ...] = ()

    @property
    def attempted(self) -> bool:
        return self.attempted_chunks > 0


def status_after_attempted_chunks(
    chunks: int,
    attempted_chunks: int,
) -> CoordinatedSendStatus:
    if attempted_chunks <= 0:
        return CoordinatedSendStatus.FAILED_BEFORE_DISPATCH
    if attempted_chunks < chunks:
        return CoordinatedSendStatus.PARTIALLY_ATTEMPTED
    return CoordinatedSendStatus.ATTEMPTED_UNKNOWN


class CoordinatedSendCancelled(asyncio.CancelledError):
    """Cancellation carrying the exact physical-fragment attempt boundary."""

    def __init__(
        self,
        detail: str,
        *,
        chunks: int,
        attempted_chunks: int,
        receipts: tuple[PhysicalDeliveryReceipt, ...] = (),
    ) -> None:
        super().__init__(detail)
        self.chunks = max(0, int(chunks))
        self.attempted_chunks = min(self.chunks, max(0, int(attempted_chunks)))
        self.receipts = tuple(receipts)

    @property
    def platform_attempted(self) -> bool:
        return self.attempted_chunks > 0


@dataclass(slots=True)
class ChunkDispatchCommand:
    owner: Any
    prepare_platform_call: PlatformCallPreparer
    platform: Any
    chunks: Sequence[Any]
    reservation: Any
    capability: DeliveryCapability
    before_platform_call: Callable[[], Awaitable[bool]] | None
    dispatch_fence: QPMDispatchFence | None
    now: datetime | None
    attempted: int = 0
    receipts: list[PhysicalDeliveryReceipt] = field(default_factory=list)

    async def run(self) -> CoordinatedSendResult:
        for index, chunk in enumerate(self.chunks):
            prepared_call, rejection = await self._prepare_chunk(chunk)
            if rejection is not None:
                return rejection
            dispatch, rejection = await self._start_dispatch(index)
            if rejection is not None:
                return rejection
            if not dispatch.allowed or not dispatch.attempt_id:
                return await self._dispatch_not_started(dispatch)
            fence_rejection = await self._apply_platform_call_fence(dispatch.attempt_id)
            if fence_rejection is not None:
                return fence_rejection
            self.attempted += 1
            call_rejection = await self._invoke_chunk(
                index,
                chunk,
                prepared_call,
                dispatch.attempt_id,
            )
            if call_rejection is not None:
                return call_rejection
        await self.owner.storage.release(self.reservation_id, now=self.now)
        return self._all_attempted_result()

    async def _prepare_chunk(
        self,
        chunk: Any,
    ) -> tuple[PreparedPlatformCall | None, CoordinatedSendResult | None]:
        try:
            return await self.prepare_platform_call(chunk), None
        except asyncio.CancelledError:
            await self._release_after_cancellation("platform_call_preparation_cancelled")
            raise self._cancelled("platform_call_preparation_cancelled") from None
        except Exception as exc:
            await self.owner.storage.release(self.reservation_id, now=self.now)
            return None, self._result(
                status_after_attempted_chunks(self.total, self.attempted),
                f"platform_call_preparation_exception:{type(exc).__name__}",
            )

    async def _start_dispatch(
        self,
        index: int,
    ) -> tuple[Any | None, CoordinatedSendResult | None]:
        try:
            dispatch = await self._begin_dispatch(index)
            return dispatch, None
        except CoordinatedSendCancelled:
            raise
        except Exception as exc:
            await self.owner.storage.release(self.reservation_id, now=self.now)
            return None, self._result(
                status_after_attempted_chunks(self.total, self.attempted),
                f"dispatch_begin_exception:{type(exc).__name__}",
            )

    async def _dispatch_not_started(self, dispatch: Any) -> CoordinatedSendResult:
        await self.owner.storage.release(self.reservation_id, now=self.now)
        status = status_after_attempted_chunks(self.total, self.attempted)
        if dispatch.already_started and not self.attempted:
            status = CoordinatedSendStatus.ATTEMPTED_UNKNOWN
        return self._result(status, f"dispatch_not_started:{dispatch.reason}")

    async def _invoke_chunk(
        self,
        index: int,
        chunk: Any,
        prepared_call: PreparedPlatformCall | None,
        attempt_id: str,
    ) -> CoordinatedSendResult | None:
        assert prepared_call is not None

        async def invoke_prepared(_chunk: Any) -> Any:
            return await prepared_call()

        try:
            value, qq_receipt = await send_chunk_with_receipt(
                invoke_prepared,
                chunk,
                self.platform,
                self.capability,
            )
        except asyncio.CancelledError:
            await self._settle_cancelled_platform_call(attempt_id)
            raise self._cancelled("platform_call_cancelled_unknown") from None
        except Exception as exc:
            return await self._settle_failed_platform_call(attempt_id, exc)
        self._capture_receipts(value, qq_receipt, index)
        await self._settle_returned_platform_call(attempt_id, index)
        return None

    async def _settle_cancelled_platform_call(self, attempt_id: str) -> None:
        settlement = asyncio.create_task(
            self._settle_platform_call_attempt(
                attempt_id,
                detail="platform_call_cancelled_unknown",
            ),
            name="soulcore-cancelled-platform-call-settlement",
        )
        try:
            await self._drain_platform_settlement(
                settlement,
                already_cancelled=True,
                detail="cancelled platform-call settlement",
            )
        except Exception:
            logger.exception("cancelled platform-call settlement failed")

    async def _settle_failed_platform_call(
        self,
        attempt_id: str,
        exc: Exception,
    ) -> CoordinatedSendResult:
        detail = f"platform_call_unknown:{type(exc).__name__}:{exc}"
        settlement = asyncio.create_task(
            self._settle_platform_call_attempt(attempt_id, detail=detail),
            name="soulcore-failed-platform-call-settlement",
        )
        cancelled = await self._drain_platform_settlement(
            settlement,
            already_cancelled=False,
            detail="failed platform-call settlement",
        )
        if cancelled:
            raise self._cancelled("platform_call_settlement_cancelled_unknown") from None
        return self._result(
            status_after_attempted_chunks(self.total, self.attempted),
            detail,
        )

    def _capture_receipts(self, value: Any, qq_receipt: Any, index: int) -> None:
        chunk_receipts = receipts_from_sender_result(value, index, self.capability)
        chunk_receipts = merge_qq_receipt(
            chunk_receipts,
            qq_receipt,
            index,
            self.capability,
        )
        self.receipts.extend(chunk_receipts)

    async def _settle_returned_platform_call(self, attempt_id: str, index: int) -> None:
        detail = (
            "platform_call_returned_with_acceptance_receipt"
            if self.receipts and self.receipts[-1].fragment_ordinal == index
            else "platform_call_returned_without_delivery_receipt"
        )
        settlement = asyncio.create_task(
            self.owner.storage.mark_attempted_unknown(
                attempt_id,
                detail=detail,
                now=self.now,
                fence=self.dispatch_fence,
            ),
            name="soulcore-returned-platform-call-settlement",
        )
        cancelled = await self._drain_platform_settlement(
            settlement,
            already_cancelled=False,
            detail="returned platform-call settlement",
        )
        if cancelled:
            await self._release_after_cancellation("returned_platform_call_cancelled_release")
            raise self._cancelled("platform_call_settlement_cancelled_unknown") from None

    async def _apply_platform_call_fence(
        self,
        attempt_id: str,
    ) -> CoordinatedSendResult | None:
        if self.before_platform_call is None:
            return None
        try:
            platform_call_allowed = await self.before_platform_call()
        except asyncio.CancelledError:
            rolled_back, _ = await self._settle_platform_fence_rollback(
                attempt_id,
                detail="platform_call_fence_cancelled",
                task_name="soulcore-cancelled-platform-fence-rollback",
                already_cancelled=True,
            )
            raise self._fence_cancelled("platform_call_fence_cancelled", rolled_back) from None
        except Exception as exc:
            return await self._settle_failed_platform_fence(attempt_id, exc)
        if platform_call_allowed:
            return None
        return await self._settle_rejected_platform_fence(attempt_id)

    async def _settle_failed_platform_fence(
        self,
        attempt_id: str,
        exc: Exception,
    ) -> CoordinatedSendResult:
        detail = f"platform_call_fence_exception:{type(exc).__name__}"
        rolled_back, cancelled = await self._settle_platform_fence_rollback(
            attempt_id,
            detail=detail,
            task_name="soulcore-failed-platform-fence-rollback",
            already_cancelled=False,
        )
        if cancelled:
            raise self._fence_cancelled(
                "platform_call_fence_settlement_cancelled",
                rolled_back,
            ) from None
        effective = self._attempted_after_rollback(rolled_back)
        status = (
            CoordinatedSendStatus.ATTEMPTED_UNKNOWN
            if not rolled_back and not self.attempted
            else status_after_attempted_chunks(self.total, effective)
        )
        return self._result(status, detail, attempted=effective)

    async def _settle_rejected_platform_fence(
        self,
        attempt_id: str,
    ) -> CoordinatedSendResult:
        rolled_back, cancelled = await self._settle_platform_fence_rollback(
            attempt_id,
            detail="platform_call_fence_already_started",
            task_name="soulcore-rejected-platform-fence-rollback",
            already_cancelled=False,
        )
        if cancelled:
            raise self._fence_cancelled(
                "platform_call_fence_settlement_cancelled",
                rolled_back,
            ) from None
        effective = self._attempted_after_rollback(rolled_back)
        status = (
            CoordinatedSendStatus.ATTEMPTED_UNKNOWN
            if not self.attempted
            else status_after_attempted_chunks(self.total, effective)
        )
        detail = (
            "platform_call_fence_already_started"
            if rolled_back
            else "platform_call_fence_rollback_unknown"
        )
        return self._result(status, detail, attempted=effective)

    async def _begin_dispatch(self, chunk_index: int) -> Any:
        task = asyncio.create_task(
            self.owner.storage.begin_dispatch(
                self.reservation_id,
                chunk_index,
                fence=self.dispatch_fence,
                now=self.now,
            )
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            attempted = await self._drain_cancelled_dispatch_begin(task)
            raise CoordinatedSendCancelled(
                "dispatch_begin_cancelled",
                chunks=self.total,
                attempted_chunks=attempted,
                receipts=tuple(self.receipts),
            ) from None

    async def _drain_cancelled_dispatch_begin(self, begin_task: asyncio.Task[Any]) -> int:
        cleanup = asyncio.create_task(
            self._settle_cancelled_dispatch_begin(begin_task),
            name="soulcore-cancelled-dispatch-begin-settlement",
        )
        current = asyncio.current_task()
        assert current is not None
        observed_cancellations = current.cancelling()
        attempted_chunks = max(0, int(self.attempted))
        while True:
            try:
                return await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                current_cancellations = current.cancelling()
                if current_cancellations > observed_cancellations:
                    observed_cancellations = current_cancellations
                    continue
                logger.error("cancelled dispatch-begin settlement task was cancelled")
                return min(self.total, attempted_chunks + 1)

    async def _settle_cancelled_dispatch_begin(
        self,
        begin_task: asyncio.Task[Any],
    ) -> int:
        started = False
        failed_before_call = False
        settlement_uncertain = False
        try:
            decision = await begin_task
            attempt_id = str(decision.attempt_id or "")
            started = bool(decision.allowed and attempt_id)
            if started:
                failed_before_call = bool(
                    await self.owner.storage.fail_before_platform_call(
                        attempt_id,
                        detail="dispatch_begin_cancelled_before_platform_call",
                        now=self.now,
                    )
                )
        except asyncio.CancelledError:
            settlement_uncertain = True
            logger.error("dispatch-begin task cancelled before settlement")
        except Exception:
            settlement_uncertain = True
            logger.exception("cancelled dispatch-begin settlement failed")
        await self._release_after_cancellation("dispatch_begin_cancelled")
        return self.attempted + int(settlement_uncertain or (started and not failed_before_call))

    async def _rollback_dispatch_before_platform_call(
        self,
        attempt_id: str,
        *,
        detail: str,
    ) -> bool:
        try:
            return bool(
                await self.owner.storage.fail_before_platform_call(
                    attempt_id,
                    detail=detail,
                    now=self.now,
                )
            )
        finally:
            await self._release_after_cancellation("before_platform_call_rollback_release")

    async def _settle_platform_fence_rollback(
        self,
        attempt_id: str,
        *,
        detail: str,
        task_name: str,
        already_cancelled: bool,
    ) -> tuple[bool, bool]:
        rollback = asyncio.create_task(
            self._rollback_dispatch_before_platform_call(
                attempt_id,
                detail=detail,
            ),
            name=task_name,
        )
        try:
            return await self._drain_boolean_settlement(
                rollback,
                already_cancelled=already_cancelled,
                detail=f"{detail} rollback",
            )
        except Exception:
            logger.exception(
                "platform-call fence rollback failed",
                extra={"attempt_id": attempt_id, "detail": detail},
            )
            return False, bool(already_cancelled)

    @staticmethod
    async def _drain_boolean_settlement(
        settlement: asyncio.Task[bool],
        *,
        already_cancelled: bool,
        detail: str,
    ) -> tuple[bool, bool]:
        current = asyncio.current_task()
        assert current is not None
        observed_cancellations = current.cancelling()
        cancelled = bool(already_cancelled)
        while True:
            try:
                return bool(await asyncio.shield(settlement)), cancelled
            except asyncio.CancelledError as exc:
                current_cancellations = current.cancelling()
                if current_cancellations > observed_cancellations:
                    cancelled = True
                    observed_cancellations = current_cancellations
                    continue
                raise RuntimeError(f"{detail} task was cancelled") from exc

    async def _settle_platform_call_attempt(
        self,
        attempt_id: str,
        *,
        detail: str,
    ) -> None:
        try:
            await self.owner.storage.mark_attempted_unknown(
                attempt_id,
                detail=detail,
                now=self.now,
                fence=self.dispatch_fence,
            )
        finally:
            await self._release_after_cancellation("attempted_platform_call_release")

    @staticmethod
    async def _drain_platform_settlement(
        settlement: asyncio.Task[Any],
        *,
        already_cancelled: bool,
        detail: str,
    ) -> bool:
        current = asyncio.current_task()
        assert current is not None
        observed_cancellations = current.cancelling()
        cancelled = bool(already_cancelled)
        while True:
            try:
                await asyncio.shield(settlement)
                return cancelled
            except asyncio.CancelledError as exc:
                current_cancellations = current.cancelling()
                if current_cancellations > observed_cancellations:
                    cancelled = True
                    observed_cancellations = current_cancellations
                    continue
                raise RuntimeError(f"{detail} task was cancelled") from exc

    async def _release_after_cancellation(self, detail: str) -> None:
        release = asyncio.create_task(
            self.owner.storage.release(self.reservation_id, now=self.now),
            name="soulcore-cancelled-qpm-release",
        )
        current = asyncio.current_task()
        assert current is not None
        observed_cancellations = current.cancelling()
        while True:
            try:
                await asyncio.shield(release)
                return
            except asyncio.CancelledError:
                current_cancellations = current.cancelling()
                if current_cancellations > observed_cancellations:
                    observed_cancellations = current_cancellations
                    continue
                logger.error("%s settlement task was cancelled", detail)
                return
            except Exception:
                logger.exception("%s settlement failed", detail)
                return

    @property
    def total(self) -> int:
        return len(self.chunks)

    @property
    def reservation_id(self) -> str:
        return str(self.reservation.reservation_id)

    def _attempted_after_rollback(self, rolled_back: bool) -> int:
        return min(self.total, self.attempted + int(not rolled_back))

    def _cancelled(self, detail: str) -> CoordinatedSendCancelled:
        return CoordinatedSendCancelled(
            detail,
            chunks=self.total,
            attempted_chunks=self.attempted,
            receipts=tuple(self.receipts),
        )

    def _fence_cancelled(
        self,
        detail: str,
        rolled_back: bool,
    ) -> CoordinatedSendCancelled:
        return CoordinatedSendCancelled(
            detail,
            chunks=self.total,
            attempted_chunks=self._attempted_after_rollback(rolled_back),
            receipts=tuple(self.receipts),
        )

    def _result(
        self,
        status: CoordinatedSendStatus,
        detail: str,
        *,
        attempted: int | None = None,
    ) -> CoordinatedSendResult:
        return CoordinatedSendResult(
            status,
            detail,
            self.total,
            self.attempted if attempted is None else attempted,
            self.reservation_id,
            tuple(self.receipts),
        )

    def _all_attempted_result(self) -> CoordinatedSendResult:
        detail = (
            "all_platform_calls_returned_with_acceptance_receipts"
            if self.receipts
            else "all_platform_calls_returned_without_delivery_receipts"
        )
        return self._result(CoordinatedSendStatus.ATTEMPTED_UNKNOWN, detail)
