"""Stable presentation contract for deterministic player-facing notices."""

from __future__ import annotations

SOULCORE_SYSTEM_NOTICE_PREFIX = "【SoulCore】"


def soulcore_system_notice(message: str) -> str:
    """Mark a fixed system message without exposing internal diagnostics."""

    body = str(message or "").strip()
    if not body:
        raise ValueError("system notice message is required")
    return f"{SOULCORE_SYSTEM_NOTICE_PREFIX}{body}"


__all__ = ["SOULCORE_SYSTEM_NOTICE_PREFIX", "soulcore_system_notice"]
