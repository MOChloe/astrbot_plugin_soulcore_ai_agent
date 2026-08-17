"""Sanitized audio attempt presentation for the AI work-record workspace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_AUDIO_PURPOSES = {
    "AUDIO_TRANSCRIPTION": "transcription",
    "AUDIO_SPEECH_GENERATION": "speech",
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _audio_attempt_kind(
    request: Mapping[str, Any], response: Mapping[str, Any], purpose: str = ""
) -> str:
    normalized_purpose = str(purpose or "").strip().upper()
    if normalized_purpose in _AUDIO_PURPOSES:
        return _AUDIO_PURPOSES[normalized_purpose]
    interaction = _mapping(request.get("interaction"))
    normalized_purpose = str(interaction.get("purpose") or "").strip().upper()
    if normalized_purpose in _AUDIO_PURPOSES:
        return _AUDIO_PURPOSES[normalized_purpose]
    capability_input = _mapping(request.get("capability_input"))
    capability_output = _mapping(response.get("capability_output"))
    if "audio" in capability_input and (
        "text" in capability_output or "text_length" in capability_output
    ):
        return "transcription"
    if (
        "text" in capability_input or "text_length" in capability_input
    ) and "audio" in capability_output:
        return "speech"
    return ""


def _audio_byte_length(value: Any, *, depth: int = 0) -> int | None:
    if depth > 4 or not isinstance(value, Mapping):
        return None
    for key in ("byte_length", "bytes"):
        raw = value.get(key)
        if isinstance(raw, (int, float)) and int(raw) >= 0:
            return int(raw)
    for key in ("data", "audio", "metadata"):
        nested = _audio_byte_length(value.get(key), depth=depth + 1)
        if nested is not None:
            return nested
    return None


def _audio_duration_ms(*values: Any) -> int | None:
    for value in values:
        if not isinstance(value, Mapping):
            continue
        for key, multiplier in (
            ("duration_ms", 1.0),
            ("duration_seconds", 1000.0),
            ("duration", 1000.0),
        ):
            raw = value.get(key)
            if isinstance(raw, (int, float)) and float(raw) >= 0:
                return int(round(float(raw) * multiplier))
        nested = _mapping(value.get("audio"))
        result = _audio_duration_ms(nested) if nested else None
        if result is not None:
            return result
    return None


def _audio_format(*values: Any) -> str:
    for value in values:
        if not isinstance(value, Mapping):
            continue
        for key in ("audio_format", "media_type", "mime_type", "format"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
        nested = _mapping(value.get("audio"))
        result = _audio_format(nested) if nested else ""
        if result:
            return result
    return ""


def _audio_text_length(projected: Mapping[str, Any], raw_text: Any) -> int:
    value = projected.get("text_length")
    if isinstance(value, (int, float)) and int(value) >= 0:
        return int(value)
    return len(str(raw_text or ""))


def audio_attempt_summary(
    attempt: Mapping[str, Any], *, purpose: str = "", fallback: bool = False
) -> dict[str, Any] | None:
    """Return the bounded audio view without exposing audio bytes or source locators."""

    request = _mapping(attempt.get("request"))
    response = _mapping(attempt.get("response"))
    kind = _audio_attempt_kind(request, response, purpose)
    if not kind:
        return None
    capability_input = _mapping(request.get("capability_input"))
    capability_output = _mapping(response.get("capability_output"))
    input_audio = _mapping(capability_input.get("audio"))
    output_audio = _mapping(capability_output.get("audio"))
    transcription = kind == "transcription"
    text_length = (
        _audio_text_length(capability_output, capability_output.get("text"))
        if transcription
        else _audio_text_length(capability_input, capability_input.get("text"))
    )
    language = str(
        capability_output.get("language")
        or capability_input.get("language")
        or ("auto" if transcription else "")
    ).strip()
    return {
        "kind": kind,
        "label": "语音转文字" if transcription else "文字转语音",
        "model": str(capability_output.get("model") or attempt.get("model_id") or ""),
        "format": (
            _audio_format(input_audio, capability_output)
            if transcription
            else _audio_format(output_audio, capability_output, capability_input)
        ),
        "duration_ms": _audio_duration_ms(
            capability_output,
            input_audio if transcription else output_audio,
        ),
        "byte_length": _audio_byte_length(input_audio if transcription else output_audio),
        "text_length": text_length,
        "language": language,
        "voice": str(capability_output.get("voice") or capability_input.get("voice") or "").strip(),
        "status": str(attempt.get("status") or "PREPARING"),
        "fallback": bool(fallback),
    }


__all__ = ["audio_attempt_summary"]
