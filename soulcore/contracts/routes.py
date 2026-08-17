"""Stable route-type contracts shared by persistence and platform adapters."""

KNOWN_MESSAGE_TYPES = (
    "FriendMessage",
    "PrivateMessage",
    "GroupMessage",
    "GuildMessage",
)
COLD_BOOT_PERSISTABLE_MESSAGE_TYPES = (
    "FriendMessage",
    "PrivateMessage",
)

__all__ = ["COLD_BOOT_PERSISTABLE_MESSAGE_TYPES", "KNOWN_MESSAGE_TYPES"]
