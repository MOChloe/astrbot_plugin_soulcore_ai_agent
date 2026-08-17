"""Single model-visible grammar for dialogue timeline rows."""

from __future__ import annotations

from typing import Any

from ...shared.time_display import model_datetime
from ..identity import escape_untrusted_identity_syntax


def render_dialogue_line(
    body: Any,
    *,
    occurred_at: Any = None,
    message_ref: str = "",
    participant_ref: str = "",
    display_name: str = "",
) -> str:
    """Keep time and each reference in independent, consistently ordered fields."""

    time = model_datetime(occurred_at, localize=True)
    message = str(message_ref or "").strip()
    participant = str(participant_ref or "").strip()
    fields = [
        f"[{value}]"
        for value in (
            time,
            message,
            participant,
        )
        if value
    ]
    text = str(body or "").strip()
    name = escape_untrusted_identity_syntax(str(display_name or "").strip())
    content = f"{name}：{text}" if name else text
    return " ".join((*fields, content)).strip()


__all__ = ["render_dialogue_line"]
