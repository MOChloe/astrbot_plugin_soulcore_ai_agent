"""Shared contracts and constants for guided AI model setup."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

MAIN_CAPABILITIES = (
    "chat.completion",
    "conversation.summary",
    "memory.reasoning",
    "text.completion",
    "sticker.collect",
    "sticker.check",
)
FAST_CAPABILITIES = (
    "conversation.turn_buffer",
    "conversation.group_interjection",
    "conversation.group_reply_relocation",
    "conversation.timer_lifecycle_review",
)
SLOT_CAPABILITIES = {
    "main": MAIN_CAPABILITIES,
    "fast": FAST_CAPABILITIES,
    "vision": ("vision.describe",),
    "polish": ("conversation.response_polish",),
    "image": ("image.generate",),
}
SLOT_PROBE_CAPABILITY = {
    "main": "chat.completion",
    "fast": "conversation.turn_buffer",
    "vision": "vision.describe",
    "polish": "conversation.response_polish",
    "image": "image.generate",
}
TEXT_PROTOCOLS = frozenset({"OPENAI", "OPENAI_COMPATIBLE", "ANTHROPIC"})
IMAGE_PROTOCOLS = frozenset({"OPENAI", "OPENAI_COMPATIBLE", "GEMINI", "CUSTOM_HTTP_IMAGE"})


class ConfigurationPort(Protocol):
    async def snapshot(self, profile_id: str) -> dict[str, Any]: ...

    async def save_package(self, payload: Mapping[str, Any], profile_id: str) -> dict[str, Any]: ...

    async def save_credential(
        self, payload: Mapping[str, Any], profile_id: str
    ) -> dict[str, Any]: ...

    async def save_model(self, payload: Mapping[str, Any], profile_id: str) -> dict[str, Any]: ...


class ProbePort(Protocol):
    async def probe_model(
        self, backend_id: str, profile_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]: ...


class RepositoryPort(Protocol):
    async def get_ai_api_model(self, backend_id: str) -> Mapping[str, Any] | None: ...

    async def get_ai_api_package(
        self, package_id: str, **values: object
    ) -> Mapping[str, Any] | None: ...


PoolWriter = Callable[[Mapping[str, Any], str], Awaitable[dict[str, Any]]]


__all__ = [
    "ConfigurationPort",
    "FAST_CAPABILITIES",
    "IMAGE_PROTOCOLS",
    "MAIN_CAPABILITIES",
    "PoolWriter",
    "ProbePort",
    "RepositoryPort",
    "SLOT_CAPABILITIES",
    "SLOT_PROBE_CAPABILITY",
    "TEXT_PROTOCOLS",
]
