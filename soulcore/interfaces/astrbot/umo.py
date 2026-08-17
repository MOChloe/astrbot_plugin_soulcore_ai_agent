"""Opaque UMO parsing and per-boot route readiness tracking."""

from __future__ import annotations

from dataclasses import dataclass

from ...contracts.routes import COLD_BOOT_PERSISTABLE_MESSAGE_TYPES
from ...features.delivery.routes import CapturedUMO, RouteKind


@dataclass(frozen=True, slots=True)
class RouteReadiness:
    ready: bool
    detail: str


class RouteReadinessTracker:
    """Tracks routes observed during this plugin boot.

    QQ C2C routes can be attempted from their persisted UMO. QQ group/guild
    adapters rebuild scene information from an inbound event after a cold boot,
    so those routes are held until observed in this process.
    """

    def __init__(self) -> None:
        self._observed_this_boot: set[str] = set()

    def note_inbound(self, umo: str | CapturedUMO) -> CapturedUMO:
        captured = umo if isinstance(umo, CapturedUMO) else CapturedUMO.parse(umo)
        if captured.is_valid:
            self._observed_this_boot.add(captured.raw)
        return captured

    def check(self, umo: str | CapturedUMO) -> RouteReadiness:
        captured = umo if isinstance(umo, CapturedUMO) else CapturedUMO.parse(umo)
        if not captured.is_valid:
            return RouteReadiness(False, "invalid_umo")
        if captured.message_type in COLD_BOOT_PERSISTABLE_MESSAGE_TYPES:
            return RouteReadiness(True, "friend_route_persistable")
        if captured.kind in (RouteKind.GROUP, RouteKind.GUILD):
            if captured.raw in self._observed_this_boot:
                return RouteReadiness(True, "route_observed_this_boot")
            return RouteReadiness(False, "route_not_observed_this_boot")
        return RouteReadiness(False, "unsupported_message_type")

    def clear(self) -> None:
        self._observed_this_boot.clear()


def physical_event_route(event: object) -> CapturedUMO:
    """Build the delivery route from immutable platform message facts.

    AstrBot's ``unique_session`` option deliberately rewrites ``session_id``
    before plugin handlers run.  That value is an LLM conversation preference,
    not a physical delivery target, so it must not define SoulCore storage or
    outbound ownership.
    """

    platform_id = str(event.get_platform_id()).strip()  # type: ignore[attr-defined]
    message_type_value = event.get_message_type()  # type: ignore[attr-defined]
    message_type = str(getattr(message_type_value, "value", message_type_value)).strip()
    if message_type == "GroupMessage":
        target_id = str(event.get_group_id()).strip()  # type: ignore[attr-defined]
    elif message_type in {"FriendMessage", "PrivateMessage"}:
        target_id = str(event.get_sender_id()).strip()  # type: ignore[attr-defined]
    else:
        return CapturedUMO.parse(str(event.unified_msg_origin))  # type: ignore[attr-defined]
    return CapturedUMO.parse(f"{platform_id}:{message_type}:{target_id}")


__all__ = [
    "CapturedUMO",
    "RouteKind",
    "RouteReadiness",
    "RouteReadinessTracker",
    "physical_event_route",
]
