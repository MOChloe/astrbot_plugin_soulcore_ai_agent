"""Translate delivery-domain messages into AstrBot message chains."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...features.delivery.transport import DeliveryMessage


class AstrBotDeliveryMessageAdapter:
    """Own the AstrBot-specific representation of one delivery message."""

    def __init__(self, message_chain_factory: Callable[[list[Any]], Any] | None = None) -> None:
        self.message_chain_factory = message_chain_factory

    def coerce_message_chain(self, value: Any) -> Any:
        if isinstance(value, DeliveryMessage):
            return self._message_chain_from_delivery(value)
        if not isinstance(value, (list, tuple)):
            return value
        components = list(value)
        if self.message_chain_factory is not None:
            return self.message_chain_factory(components)
        try:
            from astrbot.api.event import MessageChain
        except ImportError as exc:
            raise RuntimeError("AstrBot MessageChain is unavailable") from exc
        return MessageChain(components)

    def _message_chain_from_delivery(self, value: DeliveryMessage) -> Any:
        from astrbot.core.message.components import At, File, Plain, Reply

        from .onebot_transport import OneBotRecordAudio

        constructors = {
            "reply": Reply,
            "mention": At,
            "file": File,
            "audio_record": OneBotRecordAudio,
        }
        components = self._delivery_components(value, constructors)
        if value.content:
            components.append(Plain(text=value.content))
        chain = self.coerce_message_chain(components)
        for item in value.components:
            if str(item.get("type") or "") != "image_file":
                continue
            file_image = getattr(chain, "file_image", None)
            if not callable(file_image):
                raise RuntimeError("AstrBot MessageChain.file_image is unavailable")
            file_image(str(item.get("path") or ""))
        return chain

    @staticmethod
    def _delivery_components(
        value: DeliveryMessage,
        constructors: dict[str, Any],
    ) -> list[Any]:
        components: list[Any] = []
        for item in value.components:
            kind = str(item.get("type") or "")
            if kind == "reply":
                components.append(constructors[kind](id=str(item.get("id") or "")))
            elif kind == "mention":
                components.append(constructors[kind](qq=str(item.get("member_id") or "")))
            elif kind == "file":
                components.append(
                    constructors[kind](
                        file=str(item.get("path") or ""),
                        name=str(item.get("name") or ""),
                    )
                )
            elif kind == "audio_record":
                components.append(
                    constructors[kind](
                        path=str(item.get("path") or ""),
                        mime_type=str(item.get("mime_type") or "application/octet-stream"),
                        filename=str(item.get("filename") or "voice"),
                    )
                )
        return components

    @staticmethod
    def bind_native_reply(message_chain: Any, platform_message_id: str) -> Any:
        """Ensure the transport receives exactly the persisted native quote target."""

        if not platform_message_id:
            return message_chain
        components = getattr(message_chain, "chain", None)
        if not isinstance(components, list):
            raise ValueError("native_reply_requires_message_chain")
        existing = [
            component for component in components if component.__class__.__name__.lower() == "reply"
        ]
        if existing:
            candidate = str(
                getattr(existing[0], "id", None) or getattr(existing[0], "message_id", None) or ""
            ).strip()
            if len(existing) != 1 or candidate != platform_message_id:
                raise ValueError("native_reply_target_mismatch")
            return message_chain
        from astrbot.core.message.components import Reply

        components.insert(0, Reply(id=platform_message_id))
        return message_chain


__all__ = ["AstrBotDeliveryMessageAdapter"]
