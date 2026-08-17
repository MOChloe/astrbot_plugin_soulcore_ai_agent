"""AstrBot delivery through an exact platform instance."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from ...features.delivery.capabilities import (
    DeliveryCapability,
    PhysicalDeliveryReceipt,
)
from ...features.delivery.transport import (
    DeliveryMessage,
    DeliveryResult,
    DeliveryStatus,
    SelfRetractionResult,
    SelfRetractionStatus,
)
from .capability_detection import (
    detect_delivery_capability,
    personal_wechat_session_ready,
)
from .delivery_actions import EventDeliveryMixin, RetractionTransportMixin
from .delivery_messages import AstrBotDeliveryMessageAdapter
from .qq_delivery import QQDeliveryMixin
from .umo import CapturedUMO, RouteReadinessTracker

if TYPE_CHECKING:
    pass


class DeliveryPlatformSupportMixin:
    def _coerce_message_chain(self, value: Any) -> Any:
        return self._message_adapter.coerce_message_chain(value)

    @staticmethod
    def _bind_native_reply(message_chain: Any, platform_message_id: str) -> Any:
        return AstrBotDeliveryMessageAdapter.bind_native_reply(
            message_chain,
            platform_message_id,
        )

    def _make_message_session(self, captured: CapturedUMO) -> Any:
        assert captured.platform_id is not None
        assert captured.message_type is not None
        assert captured.target_id is not None
        if self.message_session_factory is not None:
            return self.message_session_factory(
                captured.platform_id, captured.message_type, captured.target_id
            )
        try:
            from astrbot.core.platform.astr_message_event import MessageSession
        except ImportError:
            from astrbot.core.platform.message_session import MessageSession
        from astrbot.core.platform.message_type import MessageType

        message_type = (
            MessageType.GROUP_MESSAGE
            if captured.kind.value in {"group", "guild"}
            else MessageType.FRIEND_MESSAGE
        )
        return MessageSession(
            platform_name=captured.platform_id,
            message_type=message_type,
            session_id=captured.target_id,
        )

    def _find_platform(self, platform_id: str | None) -> Any | None:
        if not platform_id:
            return None
        manager = getattr(self.context, "platform_manager", None)
        if manager is None:
            return None
        getter = getattr(manager, "get_insts", None)
        instances = getter() if callable(getter) else None
        if instances is None:
            instances = getattr(manager, "platform_insts", [])
        for platform in instances or []:
            meta = getattr(platform, "meta", None)
            meta = meta() if callable(meta) else meta
            candidate = getattr(meta, "id", None)
            if candidate is None and isinstance(meta, dict):
                candidate = meta.get("id")
            if candidate == platform_id:
                return platform
        return None

    def _is_platform_running(self, platform: Any) -> bool:
        if self.platform_running is not None:
            return self.platform_running(platform)
        try:
            from astrbot.core.platform.platform import PlatformStatus
        except ImportError:
            return False
        return getattr(platform, "status", None) == PlatformStatus.RUNNING

    @staticmethod
    def _acknowledgement_snapshot(platform: Any, target_id: str | None) -> tuple[bool, Any]:
        """Snapshot QQ Official's internal acknowledgement cache when available."""

        if not hasattr(platform, "_session_last_message_id"):
            return False, None
        value = platform._session_last_message_id
        if isinstance(value, dict):
            return True, value.get(target_id)
        return True, repr(value)


__all__ = [
    "DeliveryResult",
    "DeliveryStatus",
    "DeliveryTransport",
    "SelfRetractionResult",
    "SelfRetractionStatus",
]

if TYPE_CHECKING:
    from ...features.delivery.qpm import QPMDispatchFence, QPMDispatchPreparation
    from .send_coordinator import (
        ExpressionOutboxAdmission,
        MainCoreAdmission,
        PreparedQPMReservation,
        UnifiedSendCoordinator,
    )


class DeliveryTransport(
    DeliveryPlatformSupportMixin,
    RetractionTransportMixin,
    EventDeliveryMixin,
    QQDeliveryMixin,
):
    """Dispatch through the platform's authenticated transport without claiming delivery."""

    def __init__(
        self,
        context: Any,
        readiness: RouteReadinessTracker,
        message_chain_factory: Callable[[list[Any]], Any] | None = None,
        message_session_factory: Callable[[str, str, str], Any] | None = None,
        platform_running: Callable[[Any], bool] | None = None,
        qq_route_factory: Callable[..., Any] | None = None,
        policy_store: Any | None = None,
        send_coordinator: UnifiedSendCoordinator | None = None,
        runtime_gate: Any | None = None,
        activation_gate: Any | None = None,
    ) -> None:
        self.context = context
        self.readiness = readiness
        self.message_chain_factory = message_chain_factory
        self._message_adapter = AstrBotDeliveryMessageAdapter(message_chain_factory)
        self.message_session_factory = message_session_factory
        self.platform_running = platform_running
        self.qq_route_factory = qq_route_factory
        self.policy_store = policy_store
        self.send_coordinator = send_coordinator
        self.runtime_gate = runtime_gate
        self.activation_gate = activation_gate

    def capability_for(
        self,
        umo: str | CapturedUMO,
        *,
        qq_environment: str | None = None,
        qq_account_tier: str | None = None,
    ) -> DeliveryCapability | None:
        """Resolve addressing capabilities from the actual AstrBot adapter.

        Platform instance ids are administrator-defined connection handles and
        therefore cannot be used to infer whether a route is QQ Official or
        OneBot/NapCat.
        """

        captured = umo if isinstance(umo, CapturedUMO) else CapturedUMO.parse(umo)
        if not captured.is_valid:
            return None
        platform = self._find_platform(captured.platform_id)
        if platform is None:
            return None
        return detect_delivery_capability(
            platform,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
        )

    def route_ready(self, umo: str | CapturedUMO) -> bool:
        captured = umo if isinstance(umo, CapturedUMO) else CapturedUMO.parse(umo)
        if not captured.is_valid or not self.readiness.check(captured).ready:
            return False
        platform = self._find_platform(captured.platform_id)
        if platform is None or not self._is_platform_running(platform):
            return False
        capability = detect_delivery_capability(platform)
        if not capability.supports_route_kind(captured.kind.value):
            return False
        return not capability.personal_wechat or personal_wechat_session_ready(
            platform, captured.target_id
        )

    def voice_ready(self, umo: str | CapturedUMO) -> bool:
        """Voice presentation currently has one exact transport: OneBot record."""

        from .onebot_transport import supports_onebot_record

        captured = umo if isinstance(umo, CapturedUMO) else CapturedUMO.parse(umo)
        if not captured.is_valid:
            return False
        platform = self._find_platform(captured.platform_id)
        return bool(
            platform is not None
            and self._is_platform_running(platform)
            and supports_onebot_record(platform, captured)
        )

    async def send(
        self,
        umo: str | CapturedUMO,
        message_chain: DeliveryMessage | Any,
        *,
        profile_id: str | None = None,
        instance_id: str | None = None,
        configured_group_limit: int = 20,
        proactive: bool = False,
        qq_environment: str | None = None,
        qq_account_tier: str | None = None,
        qpm_reservation: PreparedQPMReservation | None = None,
        qpm_dispatch_fence: QPMDispatchFence | None = None,
        preauthorized_qpm: bool = False,
        before_platform_call: Callable[[], Awaitable[bool]] | None = None,
        reply_to_platform_message_id: str | None = None,
        reply_to_platform_reference_id: str | None = None,
    ) -> DeliveryResult:
        if self.activation_gate is not None:
            await self.activation_gate.wait()
        runtime_rejection = await self._profile_runtime_rejection(profile_id, instance_id)
        if runtime_rejection is not None:
            return runtime_rejection
        captured = umo if isinstance(umo, CapturedUMO) else CapturedUMO.parse(umo)
        platform, rejection = self._delivery_target(captured)
        if rejection is not None:
            return rejection
        assert platform is not None
        capability = detect_delivery_capability(
            platform,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
        )
        if not capability.supports_route_kind(captured.kind.value):
            return DeliveryResult.failed(
                "personal_wechat_private_chat_only",
                platform_found=True,
            )
        if proactive and not capability.autonomous_contact_allowed:
            return DeliveryResult.failed(
                "autonomous_contact_forbidden_unknown_platform",
                platform_found=True,
            )
        if capability.personal_wechat and not personal_wechat_session_ready(
            platform, captured.target_id
        ):
            return DeliveryResult.deferred(
                DeliveryStatus.ROUTE_NOT_READY,
                "personal_wechat_context_unavailable",
                platform_found=True,
            )

        platform_called = False

        async def begin_platform_call() -> bool:
            nonlocal platform_called
            if platform_called:
                return True
            if before_platform_call is not None and not await before_platform_call():
                return False
            platform_called = True
            return True

        try:
            chain = self._coerce_message_chain(message_chain)
            native_reply_id = (
                reply_to_platform_reference_id
                if capability.qq_official
                else reply_to_platform_message_id
            )
            if capability.quote and native_reply_id:
                chain = self._bind_native_reply(
                    chain,
                    str(native_reply_id).strip(),
                )
            return await self._send_to_platform(
                platform,
                captured,
                chain,
                profile_id=profile_id,
                instance_id=instance_id,
                configured_group_limit=configured_group_limit,
                proactive=proactive,
                qq_environment=qq_environment,
                qq_account_tier=qq_account_tier,
                qpm_reservation=qpm_reservation,
                qpm_dispatch_fence=qpm_dispatch_fence,
                preauthorized_qpm=preauthorized_qpm,
                before_platform_call=begin_platform_call,
            )
        except Exception as exc:  # platform errors are data, not plugin crashes
            if platform_called:
                return DeliveryResult.accepted_unconfirmed(f"send_exception:{type(exc).__name__}")
            return DeliveryResult.failed(
                "send_exception",
                error_message=f"{type(exc).__name__}: {exc}",
                platform_found=True,
            )

    async def _profile_runtime_rejection(
        self,
        profile_id: str | None,
        instance_id: str | None,
    ) -> DeliveryResult | None:
        """Fail closed at the last shared boundary before a platform send."""

        normalized = str(profile_id or "").strip()
        normalized_instance = str(instance_id or "").strip()
        if not normalized or self.runtime_gate is None:
            return None
        try:
            decision = await self.runtime_gate.decision(normalized, normalized_instance)
        except Exception as exc:
            return DeliveryResult.deferred(
                DeliveryStatus.ROUTE_NOT_READY,
                f"profile_runtime_gate_unavailable:{type(exc).__name__}",
                platform_found=True,
            )
        if decision.enabled:
            return None
        reason = (
            "instance_disabled_before_dispatch"
            if decision.reason == "instance_disabled"
            else "profile_disabled_before_dispatch"
        )
        return DeliveryResult.deferred(
            DeliveryStatus.ROUTE_NOT_READY,
            reason,
            platform_found=True,
        )

    def _delivery_target(
        self,
        captured: CapturedUMO,
    ) -> tuple[Any | None, DeliveryResult | None]:
        if not captured.is_valid:
            return None, DeliveryResult.failed("invalid_umo", platform_found=False)
        platform = self._find_platform(captured.platform_id)
        if platform is None:
            detail = f"platform_not_found:{captured.platform_id}"
            return None, DeliveryResult.failed(detail, platform_found=False)
        readiness = self.readiness.check(captured)
        if not readiness.ready:
            rejection = DeliveryResult.deferred(
                DeliveryStatus.ROUTE_NOT_READY,
                readiness.detail,
                platform_found=True,
            )
            return None, rejection
        if not self._is_platform_running(platform):
            detail = f"platform_not_running:{captured.platform_id}"
            return None, DeliveryResult.failed(detail, platform_found=True)
        return platform, None

    async def _send_to_platform(
        self,
        platform: Any,
        captured: CapturedUMO,
        chain: Any,
        *,
        profile_id: str | None,
        instance_id: str | None,
        configured_group_limit: int,
        proactive: bool,
        qq_environment: str | None,
        qq_account_tier: str | None,
        qpm_reservation: PreparedQPMReservation | None,
        qpm_dispatch_fence: QPMDispatchFence | None,
        preauthorized_qpm: bool,
        before_platform_call: Callable[[], Awaitable[bool]],
    ) -> DeliveryResult:
        capability = detect_delivery_capability(
            platform,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
        )
        qq_text = self._qq_text_parts(chain) if capability.qq_official else None
        sender_override = self._group_quote_sender(
            platform, captured, qq_text, capability.qq_official
        )
        if (
            (coordinator := self.send_coordinator) is not None
            and captured.kind.value == "group"
            and not preauthorized_qpm
        ):
            return await self._coordinated_session_send(
                coordinator,
                platform,
                captured,
                chain,
                profile_id,
                instance_id,
                configured_group_limit,
                proactive,
                qpm_reservation,
                qpm_dispatch_fence,
                qq_environment,
                qq_account_tier,
                sender_override,
                before_platform_call,
            )
        if sender_override is not None:
            if not await before_platform_call():
                return DeliveryResult.accepted_unconfirmed("platform_call_fence_already_started")
            receipt = await sender_override(chain)
            return DeliveryResult.accepted_unconfirmed(
                "qq_group_quote_api_accepted_unconfirmed",
                receipts=(receipt,),
            )
        if self._uses_qq_friend_api(platform, captured) and qq_text is not None:
            if capability.sandbox:
                # Keep sandbox text on the same AstrBot adapter path as media.
                # The QQ Official adapter supplies a fresh msg_seq and exposes
                # the real response message id; the raw active-message call
                # does neither and can lose a later text while media succeeds.
                return await self._send_with_session(
                    platform, captured, chain, before_platform_call=before_platform_call
                )
            return await self._qq_friend_delivery_result(
                platform,
                captured,
                chain,
                profile_id,
                instance_id,
                before_platform_call=before_platform_call,
            )
        return await self._send_with_session(
            platform, captured, chain, before_platform_call=before_platform_call
        )

    def _group_quote_sender(
        self, platform: Any, captured: CapturedUMO, qq_text: Any, qq_official: bool
    ) -> Any | None:
        if not qq_official or captured.kind.value != "group" or not qq_text or not qq_text[1]:
            return None

        async def sender(chunk: Any) -> PhysicalDeliveryReceipt:
            return await self._send_qq_group(platform, captured, chunk)

        return sender

    async def _coordinated_session_send(
        self,
        coordinator: UnifiedSendCoordinator,
        platform: Any,
        captured: CapturedUMO,
        chain: Any,
        profile_id: str | None,
        instance_id: str | None,
        configured_group_limit: int,
        proactive: bool,
        reservation: PreparedQPMReservation | None,
        dispatch_fence: QPMDispatchFence | None,
        qq_environment: str | None,
        qq_account_tier: str | None,
        sender: Any | None,
        before_platform_call: Callable[[], Awaitable[bool]],
    ) -> DeliveryResult:
        coordinated = await coordinator.send_by_session(
            platform,
            self._make_message_session(captured),
            captured,
            chain,
            profile_id=str(profile_id or "default"),
            instance_id=str(instance_id or ""),
            configured_group_limit=configured_group_limit,
            proactive=proactive,
            prepared=reservation,
            dispatch_fence=dispatch_fence,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
            sender_override=sender,
            before_platform_call=before_platform_call,
        )
        status = DeliveryStatus.FAILED
        if coordinated.status.value == DeliveryStatus.RATE_LIMITED.value:
            status = DeliveryStatus.RATE_LIMITED
        elif coordinated.status.value == DeliveryStatus.PARTIALLY_ATTEMPTED.value:
            status = DeliveryStatus.PARTIALLY_ATTEMPTED
        elif coordinated.status.value == DeliveryStatus.ATTEMPTED_UNKNOWN.value:
            status = DeliveryStatus.ATTEMPTED_UNKNOWN
        if status is DeliveryStatus.PARTIALLY_ATTEMPTED:
            return DeliveryResult.partially_attempted(
                coordinated.detail, receipts=coordinated.receipts
            )
        if status is DeliveryStatus.ATTEMPTED_UNKNOWN:
            return DeliveryResult.accepted_unconfirmed(
                coordinated.detail, receipts=coordinated.receipts
            )
        if status is DeliveryStatus.RATE_LIMITED:
            return DeliveryResult.deferred(status, coordinated.detail)
        return DeliveryResult.failed(coordinated.detail, platform_found=True)

    async def _qq_friend_delivery_result(
        self,
        platform: Any,
        captured: CapturedUMO,
        chain: Any,
        profile_id: str | None,
        instance_id: str | None,
        *,
        before_platform_call: Callable[[], Awaitable[bool]],
    ) -> DeliveryResult:
        detail, receipts = await self._send_qq_friend(
            platform,
            captured,
            chain,
            profile_id=profile_id,
            instance_id=instance_id,
            before_platform_call=before_platform_call,
        )
        status = (
            DeliveryStatus.FAILED
            if detail.startswith("qq_delivery_unavailable:")
            else DeliveryStatus.ATTEMPTED_UNKNOWN
        )
        if status is DeliveryStatus.ATTEMPTED_UNKNOWN:
            return DeliveryResult.accepted_unconfirmed(detail, receipts=receipts)
        return DeliveryResult.failed(detail, platform_found=True)

    async def _send_with_session(
        self,
        platform: Any,
        captured: CapturedUMO,
        chain: Any,
        *,
        before_platform_call: Callable[[], Awaitable[bool]] | None = None,
    ) -> DeliveryResult:
        from .onebot_transport import (
            extract_platform_message_id,
            prepare_onebot_message,
            send_prepared_onebot_message,
        )
        from .qq_outbound_receipts import capture_qq_outbound_receipt

        capability = detect_delivery_capability(platform)

        async def begin_platform_call() -> bool:
            return True if before_platform_call is None else bool(await before_platform_call())

        acknowledgement_before = self._acknowledgement_snapshot(platform, captured.target_id)
        prepared_onebot = (
            await prepare_onebot_message(platform, captured, chain) if capability.onebot else None
        )
        if capability.qq_official:
            with capture_qq_outbound_receipt(platform) as qq_receipt:
                allowed = await begin_platform_call()
                response = None
                if allowed:
                    try:
                        response = await self._invoke_session_call(
                            platform,
                            captured,
                            chain,
                            prepared_onebot,
                            send_prepared_onebot_message,
                        )
                    except asyncio.CancelledError as exc:
                        receipts = self._qq_captured_receipts(qq_receipt, capability)
                        if receipts:
                            exc.receipts = receipts
                            exc.chunks = self._qq_physical_chunk_count(chain)
                            exc.attempted_chunks = len(receipts)
                        raise
                    except Exception as exc:
                        receipts = self._qq_captured_receipts(qq_receipt, capability)
                        if receipts:
                            chunks = self._qq_physical_chunk_count(chain)
                            if len(receipts) < chunks:
                                return DeliveryResult.partially_attempted(
                                    f"qq_session_partial:{type(exc).__name__}",
                                    receipts=receipts,
                                )
                            return DeliveryResult.accepted_unconfirmed(
                                f"qq_session_exception_after_all_receipts:{type(exc).__name__}",
                                receipts=receipts,
                            )
                        raise
        else:
            qq_receipt = None
            allowed = await begin_platform_call()
            response = (
                await self._invoke_session_call(
                    platform,
                    captured,
                    chain,
                    prepared_onebot,
                    send_prepared_onebot_message,
                )
                if allowed
                else None
            )
        if not allowed:
            return DeliveryResult.accepted_unconfirmed("platform_call_fence_already_started")
        return self._session_delivery_result(
            platform,
            captured,
            capability,
            response,
            qq_receipt,
            acknowledgement_before,
            extract_platform_message_id,
        )

    async def _invoke_session_call(
        self,
        platform: Any,
        captured: CapturedUMO,
        chain: Any,
        prepared_onebot: Any | None,
        send_prepared: Any,
    ) -> Any:
        if prepared_onebot is not None:
            return await send_prepared(prepared_onebot)
        return await platform.send_by_session(self._make_message_session(captured), chain)

    def _session_delivery_result(
        self,
        platform: Any,
        captured: CapturedUMO,
        capability: DeliveryCapability,
        response: Any,
        qq_receipt: Any | None,
        acknowledgement_before: tuple[bool, Any],
        extract_message_id: Any,
    ) -> DeliveryResult:
        acknowledgement_after = self._acknowledgement_snapshot(platform, captured.target_id)
        response_message_id = str(extract_message_id(response) or "")
        receipts = self._qq_captured_receipts(
            qq_receipt,
            capability,
            response_message_id=response_message_id,
        )
        message_id = receipts[-1].platform_message_id if receipts else response_message_id
        if message_id:
            detail = "adapter_api_accepted_with_message_id"
        elif acknowledgement_before != acknowledgement_after:
            detail = "adapter_acknowledged_message_id_change"
            message_id = str(acknowledgement_after[1] or "") or None
        elif acknowledgement_after[0]:
            detail = "adapter_dispatch_returned_without_new_acknowledgement"
        else:
            detail = "adapter_dispatch_returned_without_acknowledgement_support"
        if message_id and not receipts:
            receipts = (
                capability.receipt(
                    message_id,
                    platform_reference_id=(
                        qq_receipt.platform_reference_id if qq_receipt is not None else ""
                    ),
                ),
            )
        return DeliveryResult.accepted_unconfirmed(
            detail,
            receipts=receipts,
        )

    @staticmethod
    def _qq_captured_receipts(
        capture: Any | None,
        capability: DeliveryCapability,
        *,
        response_message_id: str = "",
    ) -> tuple[PhysicalDeliveryReceipt, ...]:
        observations = list(getattr(capture, "observations", ()) or ())
        response_id = str(response_message_id or "").strip()
        if response_id and all(str(item[0]) != response_id for item in observations):
            observations.append(
                (response_id, str(getattr(capture, "platform_reference_id", "") or ""))
            )
        return tuple(
            capability.receipt(
                str(message_id),
                ordinal,
                platform_reference_id=str(reference_id or ""),
            )
            for ordinal, (message_id, reference_id) in enumerate(observations)
            if str(message_id or "").strip()
        )

    @staticmethod
    def _qq_physical_chunk_count(chain: Any) -> int:
        """Mirror AstrBot's one-media-per-QQ-request split without calling it."""

        components = list(getattr(chain, "chain", ()) or ())
        if not components:
            return 1
        chunks = 0
        current_has_media = False
        current_has_component = False
        for component in components:
            is_media = type(component).__name__ in {"Image", "Record", "Video", "File"}
            if is_media and current_has_media:
                chunks += 1
                current_has_media = False
                current_has_component = False
            current_has_component = True
            current_has_media = current_has_media or is_media
        return chunks + int(current_has_component)

    async def reserve_main_core(
        self,
        umo: str | CapturedUMO,
        *,
        profile_id: str,
        instance_id: str = "",
        origin_id: str | None = None,
        configured_group_limit: int = 20,
        proactive: bool = False,
        qq_environment: str | None = None,
        qq_account_tier: str | None = None,
    ) -> MainCoreAdmission | None:
        """Reserve one group-message unit before invoking Main Core.

        ``None`` means no coordinator has been installed.  Non-group routes
        return an admitted result from the coordinator without a reservation.
        """

        if self.send_coordinator is None:
            return None
        captured = umo if isinstance(umo, CapturedUMO) else CapturedUMO.parse(umo)
        platform = self._find_platform(captured.platform_id)
        if platform is None:
            return None
        return await self.send_coordinator.reserve_main_core(
            platform,
            captured,
            profile_id=profile_id,
            instance_id=instance_id,
            origin_id=origin_id,
            configured_group_limit=configured_group_limit,
            proactive=proactive,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
        )

    async def resize_main_core(
        self,
        prepared: PreparedQPMReservation,
        units: int,
    ) -> PreparedQPMReservation | None:
        if self.send_coordinator is None:
            return None
        return await self.send_coordinator.resize_main_core(prepared, units)

    async def reserve_expression_outbox(
        self,
        umo: str | CapturedUMO,
        *,
        profile_id: str,
        instance_id: str,
        origin_id: str,
        configured_group_limit: int = 20,
        proactive: bool = False,
        qq_environment: str | None = None,
        qq_account_tier: str | None = None,
    ) -> ExpressionOutboxAdmission | None:
        if self.send_coordinator is None:
            return None
        captured = umo if isinstance(umo, CapturedUMO) else CapturedUMO.parse(umo)
        platform = self._find_platform(captured.platform_id)
        if platform is None:
            return None
        return await self.send_coordinator.reserve_expression_outbox(
            platform,
            captured,
            profile_id=profile_id,
            instance_id=instance_id,
            origin_id=origin_id,
            configured_group_limit=configured_group_limit,
            proactive=proactive,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
        )

    async def prepare_expression_outbox(
        self,
        prepared: PreparedQPMReservation,
        fence: QPMDispatchFence,
    ) -> QPMDispatchPreparation:
        if self.send_coordinator is None:
            raise RuntimeError("send coordinator is unavailable")
        return await self.send_coordinator.prepare_expression_outbox(prepared, fence)

    async def cancel_expression_outbox(
        self,
        prepared: PreparedQPMReservation,
    ) -> None:
        if self.send_coordinator is not None:
            await self.send_coordinator.cancel_expression_outbox(prepared)

    async def renew_main_core(
        self,
        prepared: PreparedQPMReservation,
    ) -> PreparedQPMReservation | None:
        if self.send_coordinator is None:
            return None
        return await self.send_coordinator.renew_main_core(prepared)

    async def cancel_main_core(
        self,
        prepared: PreparedQPMReservation,
    ) -> None:
        if self.send_coordinator is not None:
            await self.send_coordinator.cancel_main_core(prepared)

    async def qpm_snapshots(
        self,
        umo: str | CapturedUMO,
        *,
        profile_id: str,
        configured_group_limit: int = 20,
        proactive: bool = False,
        qq_environment: str | None = None,
        qq_account_tier: str | None = None,
    ) -> tuple[Any, ...]:
        if self.send_coordinator is None:
            return ()
        captured = umo if isinstance(umo, CapturedUMO) else CapturedUMO.parse(umo)
        platform = self._find_platform(captured.platform_id)
        if platform is None:
            return ()
        return await self.send_coordinator.snapshots(
            platform,
            captured,
            profile_id=profile_id,
            configured_group_limit=configured_group_limit,
            proactive=proactive,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
        )
