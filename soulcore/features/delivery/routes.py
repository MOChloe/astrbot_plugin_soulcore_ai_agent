"""SoulCore-owned route parsing used by platform delivery adapters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ...contracts.routes import KNOWN_MESSAGE_TYPES


class RouteKind(StrEnum):
    FRIEND = "friend"
    GROUP = "group"
    GUILD = "guild"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CapturedUMO:
    """Opaque platform route plus conservative parsed routing facts."""

    raw: str
    platform_id: str | None
    message_type: str | None
    target_id: str | None
    kind: RouteKind

    @property
    def is_valid(self) -> bool:
        return bool(self.platform_id and self.message_type and self.target_id)

    @classmethod
    def parse(cls, value: object) -> CapturedUMO:
        if not isinstance(value, str) or not value.strip():
            return cls(str(value or ""), None, None, None, RouteKind.UNKNOWN)
        raw = value.strip()
        for message_type in KNOWN_MESSAGE_TYPES:
            anchor = f":{message_type}:"
            index = raw.find(anchor)
            if index <= 0:
                continue
            platform_id = raw[:index]
            target_id = raw[index + len(anchor) :]
            if platform_id and target_id:
                return cls(raw, platform_id, message_type, target_id, _kind_for(message_type))
        return cls(raw, None, None, None, RouteKind.UNKNOWN)


def _kind_for(message_type: str) -> RouteKind:
    if "Friend" in message_type or "Private" in message_type:
        return RouteKind.FRIEND
    if "Guild" in message_type:
        return RouteKind.GUILD
    if "Group" in message_type:
        return RouteKind.GROUP
    return RouteKind.UNKNOWN


__all__ = [
    "CapturedUMO",
    "KNOWN_MESSAGE_TYPES",
    "RouteKind",
]
