"""Platform retraction and foreground-event delivery actions."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from ...features.delivery.capabilities import PhysicalDeliveryReceipt
from ...features.delivery.transport import (
    DeliveryResult,
    DeliveryStatus,
    SelfRetractionResult,
    SelfRetractionStatus,
)
from .capability_detection import detect_delivery_capability
from .umo import CapturedUMO

if TYPE_CHECKING:
    from .send_coordinator import UnifiedSendCoordinator


class RetractionTransportMixin:
    if TYPE_CHECKING:

        def _find_platform(self, platform_id: str | None) -> Any | None: ...
        def _is_platform_running(self, platform: Any) -> bool: ...
        async def _qq_request(self, platform: Any, route: Any, **kwargs: Any) -> Any: ...
        def _qq_route(self, method: str, path: str, **params: Any) -> Any: ...

    async def retract_self(
        self, umo: str | CapturedUMO, platform_message_id: str
    ) -> SelfRetractionResult:
        captured = umo if isinstance(umo, CapturedUMO) else CapturedUMO.parse(umo)
        message_id = str(platform_message_id or "").strip()
        failure = self._retraction_preflight(captured, message_id)
        if failure is not None:
            return failure
        platform = self._find_platform(captured.platform_id)
        assert platform is not None
        attempt_started = False
        try:
            operation = self._retraction_operation(platform, captured, message_id)
            attempt_started = True
            await operation()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            status = (
                SelfRetractionStatus.ATTEMPTED_UNKNOWN
                if attempt_started
                else SelfRetractionStatus.FAILED
            )
            return SelfRetractionResult(
                status, f"retract_exception:{type(exc).__name__}:{exc}", True
            )
        return SelfRetractionResult(
            SelfRetractionStatus.RETRACTED, "platform_retraction_api_returned", True
        )

    def _retraction_preflight(
        self, captured: CapturedUMO, message_id: str
    ) -> SelfRetractionResult | None:
        if not captured.is_valid or captured.kind.value not in {"friend", "group"}:
            return _retraction_failure("invalid_or_unsupported_retraction_route", False)
        if not message_id:
            return _retraction_failure("platform_message_id_required", False)
        platform = self._find_platform(captured.platform_id)
        if platform is None:
            return _retraction_failure(f"platform_not_found:{captured.platform_id}", False)
        if not self._is_platform_running(platform):
            return _retraction_failure(f"platform_not_running:{captured.platform_id}", True)
        if not detect_delivery_capability(platform).retract_self:
            return _retraction_failure("self_retraction_unsupported", True)
        return None

    def _retraction_operation(self, platform: Any, captured: CapturedUMO, message_id: str) -> Any:
        if detect_delivery_capability(platform).qq_official:
            route = self._qq_retraction_route(captured, message_id)

            async def call_qq() -> None:
                await self._qq_request(platform, route)

            return call_qq
        return self._onebot_retraction_operation(platform, message_id)

    def _qq_retraction_route(self, captured: CapturedUMO, message_id: str) -> Any:
        assert captured.target_id is not None
        if captured.kind.value == "friend":
            return self._qq_route(
                "DELETE",
                "/v2/users/{openid}/messages/{message_id}",
                openid=captured.target_id,
                message_id=message_id,
            )
        return self._qq_route(
            "DELETE",
            "/v2/groups/{group_openid}/messages/{message_id}",
            group_openid=captured.target_id,
            message_id=message_id,
        )

    @staticmethod
    def _onebot_retraction_operation(platform: Any, message_id: str) -> Any:
        bot = getattr(platform, "bot", None)
        if bot is None:
            getter = getattr(platform, "get_client", None)
            bot = getter() if callable(getter) else None
        if bot is None:
            raise RuntimeError("onebot_client_unavailable")
        normalized: str | int = int(message_id) if message_id.isdigit() else message_id
        deleter = getattr(bot, "delete_msg", None)
        if callable(deleter):

            def sender() -> Any:
                return deleter(message_id=normalized)
        else:
            sender = RetractionTransportMixin._onebot_action_sender(bot, normalized)

        async def call_onebot() -> None:
            value = sender()
            if hasattr(value, "__await__"):
                await value

        return call_onebot

    @staticmethod
    def _onebot_action_sender(bot: Any, message_id: str | int) -> Any:
        call_action = getattr(bot, "call_action", None)
        if not callable(call_action):
            raise RuntimeError("onebot_delete_msg_unavailable")
        return lambda: call_action("delete_msg", message_id=message_id)


class EventDeliveryMixin:
    if TYPE_CHECKING:
        send_coordinator: UnifiedSendCoordinator | None

        def _coerce_message_chain(self, value: Any) -> Any: ...
        def _find_platform(self, platform_id: str | None) -> Any | None: ...
        def _is_platform_running(self, platform: Any) -> bool: ...
        def _qq_text_parts(self, message_chain: Any) -> Any: ...
        async def _send_qq_group(
            self, platform: Any, captured: CapturedUMO, message_chain: Any
        ) -> PhysicalDeliveryReceipt: ...
        async def _send_qq_friend(
            self,
            platform: Any,
            captured: CapturedUMO,
            message_chain: Any,
            *,
            profile_id: str | None,
            instance_id: str | None,
        ) -> tuple[str, tuple[PhysicalDeliveryReceipt, ...]]: ...
        async def _send_with_session(
            self, platform: Any, captured: CapturedUMO, chain: Any
        ) -> DeliveryResult: ...
        async def _profile_runtime_rejection(
            self,
            profile_id: str | None,
            instance_id: str | None,
        ) -> DeliveryResult | None: ...

    async def send_event(
        self,
        event: Any,
        umo: str | CapturedUMO,
        message_chain: Any,
        *,
        profile_id: str,
        instance_id: str = "",
        configured_group_limit: int = 20,
        proactive: bool = False,
        qq_environment: str | None = None,
        qq_account_tier: str | None = None,
        qpm_reservation: Any | None = None,
        before_platform_call: Callable[[], Awaitable[bool]] | None = None,
    ) -> DeliveryResult:
        runtime_rejection = await self._profile_runtime_rejection(profile_id, instance_id)
        if runtime_rejection is not None:
            return runtime_rejection
        captured = umo if isinstance(umo, CapturedUMO) else CapturedUMO.parse(umo)
        platform, failure = self._event_target(captured)
        if failure is not None:
            return failure
        assert platform is not None
        capability = detect_delivery_capability(
            platform, qq_environment=qq_environment, qq_account_tier=qq_account_tier
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
        chain = self._coerce_message_chain(message_chain)
        coordinator = self.send_coordinator
        if coordinator is not None and captured.kind.value == "group":
            return await self._send_coordinated_event(
                coordinator,
                event,
                platform,
                captured,
                chain,
                profile_id,
                instance_id,
                configured_group_limit,
                proactive,
                qq_environment,
                qq_account_tier,
                qpm_reservation,
                before_platform_call,
            )
        return await self._send_direct_event(
            event,
            platform,
            captured,
            chain,
            profile_id,
            instance_id,
            qq_environment,
            qq_account_tier,
            before_platform_call,
        )

    def _event_target(self, captured: CapturedUMO) -> tuple[Any | None, DeliveryResult | None]:
        if not captured.is_valid:
            return None, DeliveryResult.failed("invalid_umo", platform_found=False)
        platform = self._find_platform(captured.platform_id)
        if platform is None:
            return None, DeliveryResult.failed(
                f"platform_not_found:{captured.platform_id}", platform_found=False
            )
        if not self._is_platform_running(platform):
            return None, DeliveryResult.failed(
                f"platform_not_running:{captured.platform_id}", platform_found=True
            )
        return platform, None

    async def _send_coordinated_event(
        self,
        coordinator: UnifiedSendCoordinator,
        event: Any,
        platform: Any,
        captured: CapturedUMO,
        chain: Any,
        profile_id: str,
        instance_id: str,
        configured_group_limit: int,
        proactive: bool,
        qq_environment: str | None,
        qq_account_tier: str | None,
        qpm_reservation: Any | None,
        before_platform_call: Callable[[], Awaitable[bool]] | None,
    ) -> DeliveryResult:
        capability = detect_delivery_capability(
            platform, qq_environment=qq_environment, qq_account_tier=qq_account_tier
        )
        sender_override = self._quoted_group_sender(platform, captured, chain, capability)
        coordinated = await coordinator.send_event(
            event,
            platform,
            captured,
            chain,
            profile_id=profile_id,
            instance_id=instance_id,
            configured_group_limit=configured_group_limit,
            proactive=proactive,
            prepared=qpm_reservation,
            qq_environment=qq_environment,
            qq_account_tier=qq_account_tier,
            sender_override=sender_override,
            before_platform_call=before_platform_call,
        )
        if coordinated.status.value == DeliveryStatus.RATE_LIMITED.value:
            status = DeliveryStatus.RATE_LIMITED
        elif coordinated.status.value == DeliveryStatus.PARTIALLY_ATTEMPTED.value:
            status = DeliveryStatus.PARTIALLY_ATTEMPTED
        elif coordinated.status.value == DeliveryStatus.ATTEMPTED_UNKNOWN.value:
            status = DeliveryStatus.ATTEMPTED_UNKNOWN
        else:
            status = DeliveryStatus.FAILED
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

    def _quoted_group_sender(
        self, platform: Any, captured: CapturedUMO, chain: Any, capability: Any
    ) -> Any | None:
        qq_text = self._qq_text_parts(chain) if capability.qq_official else None
        if qq_text is None or not qq_text[1]:
            return None

        async def sender(chunk: Any) -> PhysicalDeliveryReceipt:
            return await self._send_qq_group(platform, captured, chunk)

        return sender

    async def _send_direct_event(
        self,
        event: Any,
        platform: Any,
        captured: CapturedUMO,
        chain: Any,
        profile_id: str,
        instance_id: str,
        qq_environment: str | None,
        qq_account_tier: str | None,
        before_platform_call: Callable[[], Awaitable[bool]] | None,
    ) -> DeliveryResult:
        capability = detect_delivery_capability(
            platform, qq_environment=qq_environment, qq_account_tier=qq_account_tier
        )
        if capability.onebot:
            return await self._send_with_session(
                platform,
                captured,
                chain,
                before_platform_call=before_platform_call,
            )
        quoted = self._qq_text_parts(chain) if capability.qq_official else None
        if quoted is not None and quoted[1] and captured.kind.value in {"friend", "group"}:
            return await self._send_quoted_event(
                platform,
                captured,
                chain,
                profile_id,
                instance_id,
                before_platform_call=before_platform_call,
            )
        return await self._call_event_sender(
            event,
            platform,
            chain,
            capability,
            before_platform_call=before_platform_call,
        )

    async def _send_quoted_event(
        self,
        platform: Any,
        captured: CapturedUMO,
        chain: Any,
        profile_id: str,
        instance_id: str,
        *,
        before_platform_call: Callable[[], Awaitable[bool]] | None,
    ) -> DeliveryResult:
        if captured.kind.value == "group":
            if before_platform_call is not None and not await before_platform_call():
                return DeliveryResult.accepted_unconfirmed("platform_call_fence_already_started")
            receipt = await self._send_qq_group(platform, captured, chain)
            return DeliveryResult.accepted_unconfirmed(
                "qq_group_quote_api_accepted_unconfirmed",
                receipts=(receipt,),
            )
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

    @staticmethod
    async def _call_event_sender(
        event: Any,
        platform: Any,
        chain: Any,
        capability: Any,
        *,
        before_platform_call: Callable[[], Awaitable[bool]] | None = None,
    ) -> DeliveryResult:
        from .qq_outbound_receipts import capture_qq_outbound_receipt

        sender = getattr(event, "send", None)
        if not callable(sender):
            return DeliveryResult.failed("event_send_unavailable", platform_found=True)
        if before_platform_call is not None and not await before_platform_call():
            return DeliveryResult.accepted_unconfirmed("platform_call_fence_already_started")
        qq_receipt = None
        try:
            if capability.qq_official:
                with capture_qq_outbound_receipt(platform) as qq_receipt:
                    response = await sender(chain)
            else:
                qq_receipt = None
                response = await sender(chain)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            receipts = EventDeliveryMixin._qq_captured_receipts(qq_receipt, capability)
            if receipts:
                chunks = EventDeliveryMixin._qq_physical_chunk_count(chain)
                if len(receipts) < chunks:
                    return DeliveryResult.partially_attempted(
                        f"platform_call_partial:{type(exc).__name__}", receipts=receipts
                    )
                return DeliveryResult.accepted_unconfirmed(
                    f"platform_call_unknown_after_all_receipts:{type(exc).__name__}",
                    receipts=receipts,
                )
            return DeliveryResult.accepted_unconfirmed(
                f"platform_call_unknown:{type(exc).__name__}:{exc}",
            )
        from .send_coordinator import extract_platform_message_id

        response_message_id = str(extract_platform_message_id(response) or "")
        receipts = EventDeliveryMixin._qq_captured_receipts(
            qq_receipt,
            capability,
            response_message_id=response_message_id,
        )
        return DeliveryResult.accepted_unconfirmed(
            (
                "platform_call_returned_with_acceptance_receipt"
                if receipts
                else "platform_call_returned_without_delivery_receipt"
            ),
            receipts=receipts,
        )

    @staticmethod
    def _qq_captured_receipts(
        capture: Any | None,
        capability: Any,
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


def _retraction_failure(detail: str, platform_found: bool) -> SelfRetractionResult:
    return SelfRetractionResult(SelfRetractionStatus.FAILED, detail, platform_found)
