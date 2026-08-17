"""Platform-neutral inbound media payloads owned by SoulCore."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InboundMediaSource:
    data: bytes = b""
    locator: str = ""
    mime_type: str | None = None
    sticker_evidence: tuple[str, ...] = ()


__all__ = ["InboundMediaSource"]
