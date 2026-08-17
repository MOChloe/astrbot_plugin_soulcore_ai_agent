"""Controlled vocabulary and deterministic projection for objective vision."""

from __future__ import annotations

import re
from enum import StrEnum


class VisionSequenceKind(StrEnum):
    SINGLE_IMAGE = "SINGLE_IMAGE"
    ANIMATION_CONTACT_SHEET = "ANIMATION_CONTACT_SHEET"
    GIF_REPRESENTATIVE_FRAMES = "GIF_REPRESENTATIVE_FRAMES"


class VisionInspectionMode(StrEnum):
    OBJECTIVE = "OBJECTIVE"
    STICKER_QUALITY = "STICKER_QUALITY"
    OCR_DIAGNOSTIC = "OCR_DIAGNOSTIC"


class VisionTextState(StrEnum):
    NO_TEXT = "NO_TEXT"
    TRANSCRIBED = "TRANSCRIBED"
    UNCLEAR_TEXT = "UNCLEAR_TEXT"


def build_history_projection(
    description: object,
    content_text: object = "",
    text_state: VisionTextState | str = VisionTextState.NO_TEXT,
    *,
    social_impression: object = "",
) -> str:
    """Build history text from objective evidence and optional social effect."""

    picture = _compact(description)
    text = _compact(content_text)
    impression = _compact(social_impression)[:80]
    state = VisionTextState(str(text_state))
    parts = [picture] if picture else []
    if impression:
        parts.append(f"交流观感：{impression}")
    if state is VisionTextState.TRANSCRIBED and text:
        parts.append(f"画面正文：{text}")
    elif state is VisionTextState.UNCLEAR_TEXT:
        if text:
            parts.append(f"画面正文仅能辨认：{text}")
        else:
            parts.append("画面中有未能完整辨认的正文文字")
    return "；".join(parts).strip("；")


def _compact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


__all__ = [
    "VisionInspectionMode",
    "VisionSequenceKind",
    "VisionTextState",
    "build_history_projection",
]
