"""Receipt-returning OneBot transport helpers."""

from __future__ import annotations

import asyncio
import base64
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from ...features.delivery.capabilities import (
    DeliveryCapability,
    PhysicalDeliveryReceipt,
)
from .capability_detection import detect_delivery_capability
from .umo import CapturedUMO, RouteKind


@dataclass(frozen=True, slots=True)
class PreparedOneBotMessage:
    sender: Any
    kwargs: dict[str, Any] = field(repr=False)


@dataclass(frozen=True, slots=True)
class OneBotRecordAudio:
    """Private component carrying only a controlled artifact path."""

    path: str = field(repr=False)
    mime_type: str = "application/octet-stream"
    filename: str = "voice"


def extract_platform_message_id(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (str, int)):
        return str(value).strip() or None
    if isinstance(value, Mapping):
        direct = _direct_message_id(value)
        nested = value.get("data")
        return direct or (extract_platform_message_id(nested) if nested is not value else None)
    return _object_message_id(value)


def _direct_message_id(value: Mapping[Any, Any]) -> str | None:
    for key in ("message_id", "id"):
        candidate = value.get(key)
        if candidate is not None and not isinstance(candidate, bool):
            text = str(candidate).strip()
            if text:
                return text
    return None


def _object_message_id(value: Any) -> str | None:
    for key in ("message_id", "id"):
        candidate = getattr(value, key, None)
        if candidate is not None and not isinstance(candidate, bool):
            text = str(candidate).strip()
            if text:
                return text
    return None


async def try_send_onebot_message(
    platform: Any, captured: CapturedUMO, message_chain: Any
) -> tuple[bool, Any]:
    prepared = await prepare_onebot_message(platform, captured, message_chain)
    if prepared is None:
        return False, None
    return True, await send_prepared_onebot_message(prepared)


async def prepare_onebot_message(
    platform: Any, captured: CapturedUMO, message_chain: Any
) -> PreparedOneBotMessage | None:
    """Finish local OneBot payload construction without crossing the send boundary."""

    capability = detect_delivery_capability(platform)
    if not _supports_direct_onebot(capability, captured):
        return None
    bot = _onebot_client(platform)
    if bot is None:
        return None
    payload = await _onebot_payload(message_chain)
    if not payload:
        return None
    sender, kwargs = _onebot_sender(bot, captured, payload)
    if not callable(sender):
        return None
    return PreparedOneBotMessage(sender, kwargs)


def supports_onebot_record(platform: Any, captured: CapturedUMO) -> bool:
    capability = detect_delivery_capability(platform)
    if not _supports_direct_onebot(capability, captured):
        return False
    bot = _onebot_client(platform)
    if bot is None:
        return False
    sender, _ = _onebot_sender(bot, captured, [])
    return callable(sender)


async def send_prepared_onebot_message(prepared: PreparedOneBotMessage) -> Any:
    value = prepared.sender(**prepared.kwargs)
    return await value if inspect.isawaitable(value) else value


def _supports_direct_onebot(capability: DeliveryCapability, captured: CapturedUMO) -> bool:
    return bool(
        capability.onebot
        and captured.kind in {RouteKind.GROUP, RouteKind.FRIEND}
        and captured.target_id
        and captured.target_id.isdigit()
    )


def _onebot_client(platform: Any) -> Any | None:
    bot = getattr(platform, "bot", None)
    if bot is not None:
        return bot
    getter = getattr(platform, "get_client", None)
    return getter() if callable(getter) else None


def _onebot_sender(
    bot: Any, captured: CapturedUMO, payload: list[dict[str, Any]]
) -> tuple[Any, dict[str, Any]]:
    target = int(str(captured.target_id))
    if captured.kind is RouteKind.GROUP:
        return getattr(bot, "send_group_msg", None), {"group_id": target, "message": payload}
    return getattr(bot, "send_private_msg", None), {"user_id": target, "message": payload}


async def _onebot_payload(message_chain: Any) -> list[dict[str, Any]] | None:
    components = getattr(message_chain, "chain", message_chain)
    if not isinstance(components, (list, tuple)) or _requires_adapter_path(components):
        return None
    native = (
        None
        if any(isinstance(component, OneBotRecordAudio) for component in components)
        else await _astrbot_onebot_payload(message_chain)
    )
    if native:
        return native
    payload: list[dict[str, Any]] = []
    for component in components:
        item = await _component_payload(component)
        if item is None:
            return None
        payload.append(dict(item))
    return payload or None


def _requires_adapter_path(components: list[Any] | tuple[Any, ...]) -> bool:
    return any(item.__class__.__name__.lower() in {"file", "node", "nodes"} for item in components)


async def _astrbot_onebot_payload(message_chain: Any) -> list[dict[str, Any]] | None:
    try:
        from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
            AiocqhttpMessageEvent,
        )

        payload = await AiocqhttpMessageEvent._parse_onebot_json(message_chain)
        return list(payload) if payload else None
    except Exception:
        return None


async def _component_payload(component: Any) -> Mapping[Any, Any] | None:
    if isinstance(component, Mapping):
        return component
    converter = getattr(component, "to_dict", None)
    if callable(converter):
        value = converter()
        value = await value if inspect.isawaitable(value) else value
    else:
        converter = getattr(component, "toDict", None)
        value = converter() if callable(converter) else None
    if isinstance(value, Mapping):
        return value
    return await _simple_component_payload(component)


async def _simple_component_payload(component: Any) -> Mapping[str, Any] | None:
    name = component.__class__.__name__.lower()
    raw = cast(Any, component)
    if name in {"plain", "text"} and hasattr(component, "text"):
        return {"type": "text", "data": {"text": str(raw.text)}}
    if name == "reply" and getattr(component, "id", None):
        return {"type": "reply", "data": {"id": str(raw.id)}}
    if name == "at" and getattr(component, "qq", None):
        return {"type": "at", "data": {"qq": str(raw.qq)}}
    if isinstance(component, OneBotRecordAudio):
        path = Path(component.path)
        if not path.is_file():
            raise FileNotFoundError(path.name)
        encoded = base64.b64encode(await asyncio.to_thread(path.read_bytes)).decode("ascii")
        return {"type": "record", "data": {"file": f"base64://{encoded}"}}
    return None


def receipts_from_sender_result(
    value: Any, fragment_ordinal: int, capability: DeliveryCapability
) -> tuple[PhysicalDeliveryReceipt, ...]:
    if isinstance(value, PhysicalDeliveryReceipt):
        return (_normalized_receipt(value, fragment_ordinal, capability),)
    if _is_receipt_sequence(value):
        return tuple(
            _normalized_receipt(item, fragment_ordinal + offset, capability)
            for offset, item in enumerate(value)
        )
    message_id = extract_platform_message_id(value)
    if not message_id:
        return ()
    return (capability.receipt(message_id, fragment_ordinal),)


def _normalized_receipt(
    receipt: PhysicalDeliveryReceipt, ordinal: int, capability: DeliveryCapability
) -> PhysicalDeliveryReceipt:
    return capability.receipt(
        receipt.platform_message_id,
        ordinal,
        accepted_unconfirmed=receipt.accepted_unconfirmed,
        platform_reference_id=receipt.platform_reference_id,
    )


def _is_receipt_sequence(value: Any) -> bool:
    return bool(
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(item, PhysicalDeliveryReceipt) for item in value)
    )


def message_id_snapshot(platform: Any, target_id: str | None) -> tuple[bool, Any]:
    if not hasattr(platform, "_session_last_message_id"):
        return False, None
    value = platform._session_last_message_id
    if isinstance(value, Mapping):
        return True, value.get(target_id)
    return True, repr(value)
