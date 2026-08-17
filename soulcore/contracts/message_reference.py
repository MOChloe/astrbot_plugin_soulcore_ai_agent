"""Safe cross-layer contracts for model-visible identities and native replies."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

INBOUND_REPLY_REFERENCE_KIND = "reply_reference"
_UNAVAILABLE_REASON = "target_not_available_in_current_instance"
_OPAQUE_SENDER_PATTERNS = (
    re.compile(r"^[A-Fa-f0-9]{16,}$"),
    re.compile(r"^\d{5,20}$"),
    re.compile(r"^[A-Za-z0-9_-]{24,}$"),
)


def safe_model_identity(value: str) -> str:
    normalized = " ".join(str(value or "").strip().split())[:80]
    if any(pattern.fullmatch(normalized) for pattern in _OPAQUE_SENDER_PATTERNS):
        return ""
    return normalized


def normalize_private_fallback_player_name(value: object) -> str:
    """Normalize one private chat's fallback name without accepting opaque ids."""

    normalized = " ".join(str(value or "").strip().split())
    if len(normalized) > 80:
        raise ValueError("private_fallback_player_name must be at most 80 characters")
    safe_name = safe_model_identity(normalized)
    if normalized and not safe_name:
        raise ValueError("private_fallback_player_name must be a readable display name")
    return safe_name


def inbound_reply_reference_component(
    *,
    message_ref: str = "",
    target_role: str = "",
    target_sender_name: str = "",
    target_sender_id: str = "",
    content_kind: str = "OTHER",
    content_projection: str = "",
    retraction_status: str = "",
    available: bool,
) -> dict[str, Any]:
    """Build the persisted, platform-id-free representation of one native reply."""

    if not available:
        return {
            "type": INBOUND_REPLY_REFERENCE_KIND,
            "status": "unavailable",
            "reason": _UNAVAILABLE_REASON,
        }
    role = str(target_role or "").strip().lower()
    if role not in {"assistant", "user"}:
        role = "unknown"
    kind = str(content_kind or "OTHER").strip().upper()
    if kind not in {"TEXT", "IMAGE", "STICKER", "FILE", "OTHER"}:
        kind = "OTHER"
    return {
        "type": INBOUND_REPLY_REFERENCE_KIND,
        "status": "resolved",
        "target_message_ref": str(message_ref or "").strip(),
        "target_role": role,
        "target_sender_name": safe_model_identity(target_sender_name),
        "target_sender_id": str(target_sender_id or "").strip(),
        "content_kind": kind,
        "content_projection": str(content_projection or "").strip()[:120],
        "retraction_status": str(retraction_status or "").strip().upper(),
    }


def inbound_reply_reference(components: Sequence[object]) -> Mapping[str, Any] | None:
    for component in components:
        if not isinstance(component, Mapping):
            continue
        kind = str(component.get("type") or "").strip().lower()
        if kind == INBOUND_REPLY_REFERENCE_KIND:
            return component
    return None


def inbound_reply_projection(components: Sequence[object]) -> str:
    """Render one compact, model-visible quote without transport metadata."""

    component = inbound_reply_reference(components)
    if component is None:
        return ""
    status = str(component.get("status") or "unavailable").strip().lower()
    if status != "resolved":
        return "[引用了一条当前无法读取内容的消息]"

    kind = str(component.get("content_kind") or "OTHER").strip().upper()
    content = " ".join(str(component.get("content_projection") or "").strip()[:120].split())
    if not content:
        content = {
            "IMAGE": "[图片]",
            "STICKER": "[表情包]",
            "FILE": "[文件]",
        }.get(kind, "[消息内容不可用]")
    role = str(component.get("target_role") or "unknown").strip().lower()
    sender = " ".join(str(component.get("target_sender_name") or "").strip()[:80].split())
    if role == "assistant":
        target = "角色"
    elif sender:
        target = sender
    elif role == "user":
        target = "对方"
    else:
        target = "一条"
    return f"[引用{target}消息：{content}]"


def with_inbound_reply_projection(text: str, components: Sequence[object]) -> str:
    projection = inbound_reply_projection(components)
    plain = str(text or "").strip()
    if not projection:
        return plain
    return f"{projection} {plain}".strip()


__all__ = [
    "INBOUND_REPLY_REFERENCE_KIND",
    "inbound_reply_projection",
    "inbound_reply_reference",
    "inbound_reply_reference_component",
    "normalize_private_fallback_player_name",
    "safe_model_identity",
    "with_inbound_reply_projection",
]
