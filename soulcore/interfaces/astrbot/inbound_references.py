"""Inbound reply references and quoted-image restoration."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from astrbot.api.event import AstrMessageEvent

from ...contracts.message_reference import (
    INBOUND_REPLY_REFERENCE_KIND,
    inbound_reply_reference_component,
)
from ...contracts.models import ConversationMessage, PlatformMessageFragment
from ...shared.event_log import record_event
from .delivery import DeliveryTransport
from .event_ids import (
    ReferenceMessageProbe,
    event_reference_message_id,
    event_reference_message_probe,
    opaque_identifier_shape,
)
from .qq_reference_ids import (
    QQReplyCapture,
    event_platform_reference_id,
    event_reply_capture,
    prefer_qq_reference_locator,
    qq_reference_locator_candidates,
)
from .umo import CapturedUMO


def _reference_match_kind(
    quoted_id: str, platform_message_id: str, platform_reference_id: str
) -> str:
    if quoted_id and quoted_id == platform_message_id == platform_reference_id:
        return "both"
    if quoted_id and quoted_id == platform_message_id:
        return "platform_message_id"
    if quoted_id and quoted_id == platform_reference_id:
        return "platform_reference_id"
    return "none"


def _fragment_probe_details(
    fragment: PlatformMessageFragment | None, target: ConversationMessage | None
) -> dict[str, object]:
    platform_message_id = fragment.platform_message_id if fragment is not None else ""
    platform_reference_id = fragment.platform_reference_id if fragment is not None else ""
    ledger_message_id = fragment.ledger_message_id if fragment is not None else None
    return {
        "lookup_status": "matched" if fragment is not None else "not_found_or_ambiguous",
        "fragment_ledger_message_id": ledger_message_id,
        "fragment_direction": fragment.direction.value if fragment is not None else "",
        "fragment_content_kind": fragment.content_kind if fragment is not None else "",
        "fragment_platform_message_id": opaque_identifier_shape(platform_message_id),
        "fragment_platform_reference_id": opaque_identifier_shape(platform_reference_id),
        "ledger_target_loaded": target is not None,
    }


def _raw_capture_probe_details(raw_capture: QQReplyCapture | None) -> dict[str, object]:
    projection = raw_capture.content_projection.strip() if raw_capture is not None else ""
    return {
        "raw_payload_fallback": bool(raw_capture is not None and projection),
        "raw_payload_kind": raw_capture.content_kind if raw_capture is not None else "",
        "raw_payload_attachment_count": raw_capture.attachment_count
        if raw_capture is not None
        else 0,
    }


def _unresolved_reply_component(raw_capture: QQReplyCapture | None) -> dict[str, Any]:
    projection = raw_capture.content_projection.strip() if raw_capture is not None else ""
    kind = raw_capture.content_kind if raw_capture is not None else "OTHER"
    if projection or kind.upper() in {"IMAGE", "FILE"}:
        return inbound_reply_reference_component(
            available=True,
            target_role="unknown",
            target_sender_name="被引用消息",
            content_kind=kind,
            content_projection=projection,
        )
    return inbound_reply_reference_component(available=False)


def _raw_reference_id(raw_capture: QQReplyCapture | None) -> str:
    return raw_capture.reference_id if raw_capture is not None else ""


def _drop_redundant_qq_quote_mention(
    payload: dict[str, Any], probe_source: str, route_kind: str
) -> None:
    """Remove QQ's synthetic C2C ``At bot`` while retaining real group mentions."""

    del probe_source
    private_route = str(route_kind or "").strip().lower() == "friend"
    if not private_route:
        return
    components = list(payload.get("components") or [])
    mentions = _at_components(components)
    if not mentions:
        return
    removed_ids = {id(item) for item in mentions}
    payload["components"] = [item for item in components if id(item) not in removed_ids]
    _drop_mention_prefix(payload)


def _at_components(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in components:
        kind = item.get("type") or ""
        if str(kind).strip().lower() == "at":
            result.append(item)
    return result


def _drop_mention_prefix(payload: dict[str, Any]) -> None:
    plain = str(payload.get("plain_text") or "").strip()
    for prefix in ("[提及本人]", "[提及一位联系人]"):
        if plain.startswith(prefix):
            payload["plain_text"] = plain[len(prefix) :].strip()
            return


def _resolved_target_role(fragment: PlatformMessageFragment) -> str:
    return "assistant" if fragment.direction.value == "OUTBOUND" else "user"


def _resolved_projection(
    fragment: PlatformMessageFragment, target: ConversationMessage | None
) -> str:
    projection = fragment.content_projection.strip()
    if not projection and target is not None:
        return target.plain_text.strip()
    return projection


def _resolved_sender_name(target: ConversationMessage | None, target_role: str) -> str:
    sender_name = target.sender_name.strip() if target is not None else ""
    if sender_name:
        return sender_name
    return "角色" if target_role == "assistant" else "对方"


class InboundReferenceMixin:
    async def _attach_inbound_reply_reference(
        self,
        event: AstrMessageEvent,
        profile_id: str,
        instance_id: str,
        captured: CapturedUMO,
        payload: dict[str, Any],
    ) -> None:
        probe = event_reference_message_probe(event)
        raw_capture = event_reply_capture(event)
        _drop_redundant_qq_quote_mention(
            payload,
            str(probe.source or ""),
            captured.kind.value,
        )
        quoted_id = prefer_qq_reference_locator(
            probe.value,
            _raw_reference_id(raw_capture),
        )
        if not quoted_id and raw_capture is None:
            return
        fragment = None
        if quoted_id:
            try:
                for candidate in qq_reference_locator_candidates(
                    _raw_reference_id(raw_capture),
                    probe.value,
                ):
                    fragment = await self._resolve_inbound_reference_fragment(
                        profile_id, instance_id, captured, candidate
                    )
                    if fragment is not None:
                        quoted_id = candidate
                        break
            except Exception as exc:
                with suppress(Exception):
                    await record_event(
                        self.event_log,
                        profile_id=profile_id,
                        instance_id=instance_id,
                        level="WARN",
                        category="conversation.reply_reference",
                        message="入站引用目标查询失败，正文继续并将引用安全降级",
                        details={"error_code": type(exc).__name__},
                    )
        target = None
        if fragment is not None:
            try:
                target = await self.conversation.get_instance_message(
                    profile_id,
                    instance_id,
                    int(fragment.ledger_message_id),
                )
            except Exception as exc:
                with suppress(Exception):
                    await record_event(
                        self.event_log,
                        profile_id=profile_id,
                        instance_id=instance_id,
                        level="WARN",
                        category="conversation.reply_reference",
                        message="入站引用账本补充查询失败，使用安全片段投影继续",
                        details={"error_code": type(exc).__name__},
                    )
        component = self._resolved_inbound_reply_component(fragment, target, raw_capture)
        existing = [
            item
            for item in list(payload.get("components") or [])
            if str(item.get("type") or "").strip().lower()
            not in {"reply", INBOUND_REPLY_REFERENCE_KIND}
        ]
        payload["components"] = [*existing, component]
        await self._record_inbound_reference_probe(
            profile_id,
            instance_id,
            event=event,
            captured=captured,
            extraction=probe,
            fragment=fragment,
            target=target,
            component=component,
            raw_capture=raw_capture,
        )

    async def _record_inbound_reference_probe(
        self,
        profile_id: str,
        instance_id: str,
        *,
        event: AstrMessageEvent,
        captured: CapturedUMO,
        extraction: ReferenceMessageProbe,
        fragment: PlatformMessageFragment | None,
        target: ConversationMessage | None,
        component: dict[str, Any],
        raw_capture: QQReplyCapture | None = None,
    ) -> None:
        quoted_id = extraction.value.strip()
        platform_message_id = fragment.platform_message_id if fragment is not None else ""
        platform_reference_id = fragment.platform_reference_id if fragment is not None else ""
        details = extraction.safe_details()
        details.update(
            {
                "current_platform_reference_id": opaque_identifier_shape(
                    event_platform_reference_id(event)
                ),
                "platform_instance_id": str(captured.platform_id or ""),
                "platform_instance": opaque_identifier_shape(captured.platform_id),
                "route": opaque_identifier_shape(captured.raw),
                "route_kind": captured.kind.value,
                "message_type": str(captured.message_type or ""),
                "matched_by": _reference_match_kind(
                    quoted_id, platform_message_id, platform_reference_id
                ),
                "component_status": str(component.get("status") or ""),
            }
        )
        details.update(_fragment_probe_details(fragment, target))
        details.update(_raw_capture_probe_details(raw_capture))
        resolved = str(component.get("status") or "").strip().lower() == "resolved"
        await record_event(
            self.event_log,
            profile_id=profile_id,
            instance_id=instance_id,
            level="INFO" if fragment is not None or resolved else "WARN",
            category="conversation.reply_reference_probe",
            message="已记录一次脱敏的入站原生引用诊断",
            details=details,
        )

    @staticmethod
    def _resolved_inbound_reply_component(
        fragment: PlatformMessageFragment | None,
        target: ConversationMessage | None,
        raw_capture: QQReplyCapture | None = None,
    ) -> dict[str, Any]:
        if fragment is None:
            return _unresolved_reply_component(raw_capture)
        target_role = _resolved_target_role(fragment)
        return inbound_reply_reference_component(
            available=True,
            message_ref=fragment.message_ref,
            target_role=target_role,
            target_sender_name=_resolved_sender_name(target, target_role),
            target_sender_id=str(target.sender_id or "") if target is not None else "",
            content_kind=fragment.content_kind,
            content_projection=_resolved_projection(fragment, target),
            retraction_status=(
                fragment.retraction_status.value if fragment.retraction_status is not None else ""
            ),
        )

    async def _record_inbound_platform_fragment(
        self,
        profile_id: str,
        instance_id: str,
        *,
        ledger: Any,
        captured: CapturedUMO,
        platform_message_id: str,
        platform_reference_id: str = "",
        message_text: str,
        components: list[dict[str, Any]],
        sender_id: str,
    ) -> bool:
        if not captured.platform_id or not captured.raw:
            return True
        native_reply, native_mention = _inbound_addressing_capabilities(
            self.delivery,
            captured,
            platform_reference_id=platform_reference_id,
        )
        kind, projection = _inbound_content_projection(message_text, components)
        try:
            await self.delivery_repository.create_message_fragment(
                profile_id,
                instance_id,
                ledger_message_id=int(ledger.message_id),
                fragment_ordinal=0,
                platform_instance_id=str(captured.platform_id),
                route_umo=captured.raw,
                platform_message_id=platform_message_id,
                platform_reference_id=platform_reference_id,
                direction="INBOUND",
                content_kind=kind,
                content_projection=projection,
                sender_id=sender_id,
                native_reply_supported=native_reply,
                member_mention_supported=native_mention,
                self_retraction_supported=False,
                returns_platform_message_id=True,
                accepted_at=ledger.occurred_at,
            )
            return True
        except Exception as exc:
            await record_event(
                self.event_log,
                profile_id=profile_id,
                instance_id=instance_id,
                level="WARN",
                category="delivery.fragment",
                message="入站平台消息片段登记失败，句柄能力安全降级",
                details={"error_code": type(exc).__name__},
            )
            return False

    async def _restore_quoted_images(
        self,
        event: AstrMessageEvent,
        profile_id: str,
        instance_id: str,
        captured: CapturedUMO,
        ledger: Any,
        payload: dict[str, Any],
        asset_ids: list[str],
    ) -> None:
        raw_capture = event_reply_capture(event)
        quoted_id = prefer_qq_reference_locator(
            event_reference_message_id(event),
            _raw_reference_id(raw_capture),
        )
        if not quoted_id:
            return
        resolved_ids = await self._quoted_image_asset_ids(
            profile_id, instance_id, captured, quoted_id
        )
        for ordinal, asset_id in enumerate(resolved_ids):
            if len(asset_ids) >= 5:
                break
            if asset_id in asset_ids:
                continue
            try:
                await self.media.link_media_to_message(
                    profile_id,
                    instance_id,
                    asset_id,
                    ledger.message_id,
                    relation="REFERENCE",
                    ordinal=ordinal,
                )
            except (KeyError, ValueError):
                continue
            asset_ids.append(asset_id)
        payload["quoted_platform_message_id"] = quoted_id

    async def _quoted_image_asset_ids(
        self,
        profile_id: str,
        instance_id: str,
        captured: CapturedUMO,
        quoted_id: str,
    ) -> list[str]:
        resolved_ids: list[str] = []
        try:
            fragment = await self._resolve_inbound_reference_fragment(
                profile_id, instance_id, captured, quoted_id
            )
            if fragment is not None:
                linked = await self.media.list_available_image_asset_ids_for_messages(
                    profile_id,
                    instance_id,
                    [int(fragment.ledger_message_id)],
                    limit=5,
                )
                resolved_ids.extend(str(item) for item in linked)
        except Exception as exc:
            with suppress(Exception):
                await self._media_error(
                    profile_id,
                    instance_id,
                    "引用图片的消息账本映射查询失败，继续检查媒体索引",
                    exc,
                )
        try:
            referenced_assets = await self.media.resolve_platform_media_reference(
                profile_id, instance_id, quoted_id
            )
            resolved_ids.extend(asset.asset_id for asset in referenced_assets)
        except Exception as exc:
            with suppress(Exception):
                await self._media_error(
                    profile_id,
                    instance_id,
                    "引用图片的媒体索引查询失败，正文继续处理",
                    exc,
                )
        return list(dict.fromkeys(resolved_ids))[:5]

    async def _resolve_inbound_reference_fragment(
        self,
        profile_id: str,
        instance_id: str,
        captured: CapturedUMO,
        platform_locator: str,
    ) -> PlatformMessageFragment | None:
        """Resolve the adapter's opaque quoted-target locator in code, not in AI."""

        return await self.delivery_repository.get_message_fragment_by_platform_locator(
            profile_id,
            instance_id,
            platform_instance_id=str(captured.platform_id or ""),
            route_umo=captured.raw,
            platform_locator=platform_locator,
        )


def _inbound_addressing_capabilities(
    delivery: DeliveryTransport,
    captured: CapturedUMO,
    *,
    platform_reference_id: str = "",
) -> tuple[bool, bool]:
    capability = delivery.capability_for(captured)
    if capability is None:
        return False, False
    native_reply = bool(
        capability.quote
        and (not capability.qq_official or str(platform_reference_id or "").strip())
    )
    return native_reply, capability.mention


def _inbound_content_projection(
    message_text: str, components: list[dict[str, Any]]
) -> tuple[str, str]:
    projection = str(message_text or "").strip()[:120]
    if projection:
        return "TEXT", projection
    kinds = {str(item.get("type") or "").strip().lower() for item in components}
    if kinds & {"image", "image_asset", "sticker", "face"}:
        return "IMAGE", "[图片]"
    if kinds & {"file", "file_artifact"}:
        return "FILE", "[文件]"
    if kinds:
        return "OTHER", "[媒体消息]"
    return "TEXT", ""


__all__ = ["InboundReferenceMixin"]
