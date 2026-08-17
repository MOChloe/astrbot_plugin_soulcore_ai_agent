"""Model-visible projection when an Agent transcript moves to text transport."""

from __future__ import annotations

from typing import Any

_CHANNEL_BY_TOOL = {
    "soulcore_continue": "继续行动",
    "soulcore_final": "最终表达",
}

ANTHROPIC_MESSAGES_PROTOCOL = "anthropic_messages"
OPENAI_CHAT_COMPLETIONS_PROTOCOL = "openai_chat_completions"
OPENAI_RESPONSES_PROTOCOL = "openai_responses"


def matching_provider_item(item: Any, provider_protocol: str) -> dict[str, Any] | None:
    """Return a raw item only when it already belongs to the target wire protocol."""

    if item.provider_item and item.provider_protocol == provider_protocol:
        return dict(item.provider_item)
    return None


def text_transport_assistant_output(turn: Any) -> str:
    """Keep visible text and string tool payloads in their original item order."""

    pieces: list[str] = []
    for item in turn.output_items:
        if item.kind == "text":
            pieces.append(item.text)
            continue
        if item.kind != "tool_call":
            continue
        channel = _CHANNEL_BY_TOOL.get(item.name)
        if channel:
            pieces.append(f"<{channel}>{item.text}</{channel}>")
    return "".join(pieces)


__all__ = [
    "ANTHROPIC_MESSAGES_PROTOCOL",
    "OPENAI_CHAT_COMPLETIONS_PROTOCOL",
    "OPENAI_RESPONSES_PROTOCOL",
    "matching_provider_item",
    "text_transport_assistant_output",
]
