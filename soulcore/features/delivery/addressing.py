"""Build native or textual addressing prefixes for one physical message."""

from __future__ import annotations

from typing import Any


def addressing_prefixes(
    payload: dict[str, Any],
    *,
    native_reply_supported: bool,
    native_mention_supported: bool,
) -> tuple[list[dict[str, object]], list[str]]:
    components: list[dict[str, object]] = []
    fallback: list[str] = []
    target = payload.get("reply_target")
    if isinstance(target, dict):
        _append_reply_prefix(
            components,
            fallback,
            target,
            native_supported=native_reply_supported,
        )
    for member in list(payload.get("mention_targets") or [])[:10]:
        if isinstance(member, dict):
            _append_mention_prefix(
                components,
                fallback,
                member,
                native_supported=native_mention_supported,
            )
    return components, fallback


def _append_reply_prefix(
    components: list[dict[str, object]],
    fallback: list[str],
    target: dict[str, Any],
    *,
    native_supported: bool,
) -> None:
    if native_supported:
        native_id = str(target.get("native_reply_target_id") or "").strip()
        if native_id:
            components.append({"type": "reply", "id": native_id})
            return
    sender = str(target.get("sender_display_name") or "对方")[:80]
    projection = str(target.get("content_projection") or "").strip()[:120]
    if not projection:
        placeholders = {"IMAGE": "[图片]", "STICKER": "[表情包]", "FILE": "[文件]"}
        projection = placeholders.get(str(target.get("content_kind") or "").upper(), "[消息]")
    fallback.append(f"引用: [{sender}：{projection}]")


def _append_mention_prefix(
    components: list[dict[str, object]],
    fallback: list[str],
    member: dict[str, Any],
    *,
    native_supported: bool,
) -> None:
    if native_supported:
        components.append(
            {"type": "mention", "member_id": str(member.get("platform_member_id") or "")}
        )
        return
    display_name = str(member.get("display_name") or "").strip()[:80]
    if display_name:
        fallback.append(f"@{display_name}")
