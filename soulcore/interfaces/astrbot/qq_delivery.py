"""QQ Official delivery helpers shared by AstrBot delivery mixins."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ...features.delivery.capabilities import PhysicalDeliveryReceipt
from .capability_detection import detect_delivery_capability
from .qq_outbound_receipts import extract_qq_message_id, extract_qq_reference_id
from .umo import CapturedUMO

logger = logging.getLogger(__name__)


class QQDeliveryMixin:
    """Send QQ Official friend/group messages through its authenticated API."""

    qq_route_factory: Callable[..., Any] | None
    policy_store: Any | None

    @staticmethod
    def _uses_qq_friend_api(platform: Any, captured: CapturedUMO) -> bool:
        if captured.kind.value != "friend":
            return False
        meta = getattr(platform, "meta", None)
        meta = meta() if callable(meta) else meta
        name = getattr(meta, "name", None)
        if name is None and isinstance(meta, dict):
            name = meta.get("name")
        return name in {"qq_official", "qq_official_webhook"}

    @staticmethod
    def _qq_text_parts(message_chain: Any) -> tuple[str, str | None] | None:
        """Return text and one native quote target for a QQ Official chain."""

        components = getattr(message_chain, "chain", message_chain)
        if not isinstance(components, (list, tuple)):
            return None
        text_parts: list[str] = []
        quote_id: str | None = None
        for component in components:
            name = component.__class__.__name__.lower()
            if name in {"plain", "text"}:
                text_parts.append(str(getattr(component, "text", "")))
                continue
            if name == "reply":
                candidate = str(
                    getattr(component, "id", None) or getattr(component, "message_id", None) or ""
                ).strip()
                if not candidate or quote_id is not None:
                    return None
                quote_id = candidate
                continue
            # QQ Official group member mentions are intentionally unsupported.
            return None
        content = "".join(text_parts)
        return (content, quote_id) if content else None

    async def _send_qq_friend(
        self,
        platform: Any,
        captured: CapturedUMO,
        message_chain: Any,
        *,
        profile_id: str | None,
        instance_id: str | None,
        before_platform_call: Callable[[], Awaitable[bool]] | None = None,
    ) -> tuple[str, tuple[PhysicalDeliveryReceipt, ...]]:
        assert captured.target_id is not None
        parsed = self._qq_text_parts(message_chain)
        if parsed is None:
            raise ValueError("qq_wakeup_requires_plain_text")
        content, quote_id = parsed

        route = self._qq_route("POST", "/v2/users/{openid}/messages", openid=captured.target_id)

        reservation = await self._reserve_qq_delivery(profile_id, instance_id=instance_id)
        mode = str(reservation.get("mode") or "")
        if mode == "unavailable":
            return (
                f"qq_delivery_unavailable:{reservation.get('reason') or 'unknown'}",
                (),
            )
        payload, detail = self._qq_friend_payload(content, quote_id, reservation)
        self._log_qq_friend_addressing(
            platform,
            captured,
            mode=mode,
            quote_id=quote_id,
            reservation=reservation,
            payload=payload,
            detail=detail,
        )
        if not await self._apply_qq_friend_platform_fence(
            before_platform_call,
            profile_id,
            instance_id,
            reservation,
        ):
            return "platform_call_fence_already_started", ()
        return await self._invoke_qq_friend_platform_call(
            platform,
            route,
            payload,
            detail,
            profile_id,
            instance_id,
            reservation,
        )

    @staticmethod
    def _qq_friend_payload(
        content: str,
        quote_id: str | None,
        reservation: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        if str(reservation.get("mode") or "") == "passive_reply":
            payload = {
                "content": content,
                "msg_type": 0,
                "msg_id": reservation["message_id"],
                "msg_seq": int(reservation["msg_seq"]),
            }
            detail = "qq_passive_reference" if quote_id else "qq_passive_reply"
        else:
            payload = {"content": content, "msg_type": 0, "is_wakeup": True}
            detail = "qq_wakeup_reference" if quote_id else "qq_wakeup"
        if quote_id:
            payload["message_reference"] = {
                "message_id": quote_id,
                "ignore_get_message_error": False,
            }
        return payload, f"{detail}_api_accepted_unconfirmed"

    async def _apply_qq_friend_platform_fence(
        self,
        before_platform_call: Callable[[], Awaitable[bool]] | None,
        profile_id: str | None,
        instance_id: str | None,
        reservation: dict[str, Any],
    ) -> bool:
        if before_platform_call is None:
            return True
        try:
            platform_call_allowed = await before_platform_call()
        except asyncio.CancelledError:
            try:
                await asyncio.shield(
                    self._finalize_qq_delivery(
                        profile_id,
                        reservation,
                        instance_id=instance_id,
                        accepted=False,
                        attempted=False,
                        error="platform_call_fence_cancelled",
                    )
                )
            except Exception:
                logger.exception("cancelled QQ send-fence settlement failed")
            raise
        except Exception as exc:
            await self._finalize_qq_delivery(
                profile_id,
                reservation,
                instance_id=instance_id,
                accepted=False,
                attempted=False,
                error=f"platform_call_fence_exception:{type(exc).__name__}",
            )
            raise
        if platform_call_allowed:
            return True
        await self._finalize_qq_delivery(
            profile_id,
            reservation,
            instance_id=instance_id,
            accepted=False,
            attempted=False,
            error="platform_call_fence_already_started",
        )
        return False

    async def _invoke_qq_friend_platform_call(
        self,
        platform: Any,
        route: Any,
        payload: dict[str, Any],
        detail: str,
        profile_id: str | None,
        instance_id: str | None,
        reservation: dict[str, Any],
    ) -> tuple[str, tuple[PhysicalDeliveryReceipt, ...]]:
        try:
            response = await self._qq_request(platform, route, json=payload)
            message_id = self._extract_message_id(response)
            reference_id = self._extract_reference_id(response)
            if not message_id:
                raise RuntimeError("qq_api_missing_message_id")
        except asyncio.CancelledError:
            try:
                await asyncio.shield(
                    self._finalize_qq_delivery(
                        profile_id,
                        reservation,
                        instance_id=instance_id,
                        accepted=True,
                        error="platform_call_cancelled_unknown",
                    )
                )
            except Exception:
                logger.exception("cancelled QQ platform-call settlement failed")
            raise
        except Exception as exc:
            await self._finalize_qq_delivery(
                profile_id,
                reservation,
                instance_id=instance_id,
                accepted=False,
                attempted=True,
                error=f"{type(exc).__name__}:{exc}",
            )
            raise
        await self._finalize_qq_delivery(
            profile_id, reservation, instance_id=instance_id, accepted=True
        )
        capability = detect_delivery_capability(platform)
        receipt = capability.receipt(message_id, platform_reference_id=reference_id or "")
        return detail, (receipt,)

    @staticmethod
    def _log_qq_friend_addressing(
        platform: Any,
        captured: CapturedUMO,
        *,
        mode: str,
        quote_id: str | None,
        reservation: dict[str, Any],
        payload: dict[str, Any],
        detail: str,
    ) -> None:
        """Emit deterministic QQ addressing facts without logging identifiers."""

        try:
            from astrbot.api import logger
        except (ImportError, AttributeError):
            return
        passive_id = str(reservation.get("message_id") or "").strip()
        cached_ids = getattr(platform, "_session_last_message_id", None)
        cached_id = (
            str(cached_ids.get(captured.target_id) or "").strip()
            if isinstance(cached_ids, dict)
            else ""
        )
        logger.info(
            "[SoulCore] QQ official addressing "
            "strategy=%s mode=%s has_quote_target=%s has_msg_id=%s "
            "has_message_reference=%s target_matches_passive=%s "
            "target_matches_adapter_cache=%s",
            detail,
            mode or "unknown",
            bool(quote_id),
            "msg_id" in payload,
            "message_reference" in payload,
            bool(quote_id and passive_id and quote_id == passive_id),
            bool(quote_id and cached_id and quote_id == cached_id),
        )

    async def _send_qq_group(
        self,
        platform: Any,
        captured: CapturedUMO,
        message_chain: Any,
    ) -> PhysicalDeliveryReceipt:
        assert captured.target_id is not None
        parsed = self._qq_text_parts(message_chain)
        if parsed is None or not parsed[1]:
            raise ValueError("qq_group_native_quote_requires_text_and_reply")
        content, quote_id = parsed
        route = self._qq_route(
            "POST",
            "/v2/groups/{group_openid}/messages",
            group_openid=captured.target_id,
        )
        payload = {
            "content": content,
            "msg_type": 0,
            "msg_seq": 1,
            "message_reference": {
                "message_id": quote_id,
                "ignore_get_message_error": False,
            },
        }
        response = await self._qq_request(platform, route, json=payload)
        message_id = self._extract_message_id(response)
        reference_id = self._extract_reference_id(response)
        if not message_id:
            raise RuntimeError("qq_api_missing_message_id")
        capability = detect_delivery_capability(platform)
        return capability.receipt(message_id, platform_reference_id=reference_id or "")

    async def _reserve_qq_delivery(
        self,
        profile_id: str | None,
        *,
        instance_id: str | None = None,
    ) -> dict[str, Any]:
        if not profile_id or not instance_id:
            return {"mode": "unavailable", "reason": "instance_context_required"}
        return dict(await self.policy_store.reserve_instance_qq_delivery(profile_id, instance_id))

    async def _finalize_qq_delivery(
        self,
        profile_id: str | None,
        reservation: dict[str, Any],
        *,
        instance_id: str | None = None,
        accepted: bool,
        attempted: bool = True,
        error: str | None = None,
    ) -> None:
        if not profile_id or not instance_id:
            return
        await self.policy_store.finalize_instance_qq_delivery(
            profile_id,
            instance_id,
            reservation,
            accepted=accepted,
            attempted=attempted,
            error=error,
        )

    @staticmethod
    def _message_chain_text(message_chain: Any) -> str:
        components = getattr(message_chain, "chain", message_chain)
        if not isinstance(components, (list, tuple)):
            return ""
        return "".join(
            str(text)
            for component in components
            if (text := getattr(component, "text", None)) is not None
        )

    @staticmethod
    def _extract_message_id(response: Any) -> str | None:
        return extract_qq_message_id(response) or None

    @staticmethod
    def _extract_reference_id(response: Any) -> str | None:
        return extract_qq_reference_id(response) or None

    def _qq_route(self, method: str, path: str, **params: Any) -> Any:
        if self.qq_route_factory is not None:
            return self.qq_route_factory(method, path, **params)
        from botpy.http import Route

        return Route(method, path, **params)

    @staticmethod
    async def _qq_request(
        platform: Any,
        route: Any,
        *,
        json: dict[str, Any] | None = None,
    ) -> Any:
        client = getattr(platform, "client", None)
        api = getattr(client, "api", None)
        http = getattr(api, "_http", None)
        request = getattr(http, "request", None)
        if not callable(request):
            raise RuntimeError("qq_authenticated_http_unavailable")
        if json is None:
            return await request(route)
        return await request(route, json=json)
