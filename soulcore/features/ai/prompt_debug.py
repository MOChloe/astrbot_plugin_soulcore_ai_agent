"""Readable, credential-safe snapshots of model requests and responses."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from ...contracts.ai_models import (
    AIAudioContent,
    AICapabilityName,
    AICapabilityRequest,
    AICompletion,
    AIModelRequest,
    AISpeechResult,
    AITranscriptionResult,
)

_REDACTED = "[REDACTED]"
_UNHANDLED = object()
_AUDIO_CAPABILITIES = {
    AICapabilityName.AUDIO_TRANSCRIBE.value,
    AICapabilityName.AUDIO_SPEECH.value,
}
_AUDIO_DIAGNOSTIC_FIELDS = {
    "audio",
    "audioformat",
    "bitrate",
    "channel",
    "emotion",
    "instruction",
    "language",
    "pitch",
    "prompt",
    "responseformat",
    "samplerate",
    "speed",
    "text",
    "voice",
    "volume",
}
_AUDIO_TEXT_FIELDS = {"instruction", "prompt", "text"}
_SENSITIVE_KEYS = {
    "apikey",
    "authorization",
    "accesstoken",
    "refreshtoken",
    "token",
    "sessiontoken",
    "auth",
    "bearertoken",
    "password",
    "passwd",
    "secret",
    "clientsecret",
    "credential",
    "credentials",
    "cookie",
    "setcookie",
    "signature",
    "sig",
    "privatekey",
    "clientkey",
}
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?im)(\b(?:authorization|cookie|set-cookie)\s*:\s*)[^\r\n]+"),
    re.compile(
        r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|signature|sig|private[_-]?key|client[_-]?key)\s*[=:]\s*)[^\s&;,]+"
    ),
    re.compile(
        r"(?i)([?&](?:api[_-]?key|access[_-]?token|refresh[_-]?token|signature|sig)=)[^&#\s]+"
    ),
)


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _sensitive_key(value: object) -> bool:
    normalized = _normalized_key(value)
    return normalized in _SENSITIVE_KEYS or normalized.endswith(
        (
            "apikey",
            "accesstoken",
            "refreshtoken",
            "clientsecret",
            "signature",
            "privatekey",
        )
    )


def redact_prompt_text(value: object) -> str:
    result = str(value or "")
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            result = pattern.sub(lambda match: f"{match.group(1)}{_REDACTED}", result)
        else:
            result = pattern.sub(_REDACTED, result)
    return result


def _prompt_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str) and value.startswith("data:") and ";base64," in value[:160]:
        header, _, encoded = value.partition(",")
        return {
            "binary_omitted": True,
            "media_type": header[5:].split(";", 1)[0] or "application/octet-stream",
            "encoded_length": len(encoded),
            "encoded_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        }
    if isinstance(value, str):
        return redact_prompt_text(value)
    if isinstance(value, bytes):
        return {
            "binary_omitted": True,
            "byte_length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, (datetime, Enum)):
        return value.isoformat() if isinstance(value, datetime) else value.value
    return _UNHANDLED


def _audio_value(value: Any) -> Any:
    if isinstance(value, AIAudioContent):
        projected: dict[str, Any] = {
            "binary_omitted": True,
            "mime_type": str(value.mime_type or "application/octet-stream"),
            "byte_length": len(value.data),
        }
        if value.duration_seconds is not None:
            projected["duration_seconds"] = value.duration_seconds
        return projected
    if isinstance(value, AITranscriptionResult):
        return {
            "text_length": len(value.text),
            "language": value.language,
            "duration_seconds": value.duration_seconds,
            "model": value.model,
        }
    if isinstance(value, AISpeechResult):
        return {
            "audio": _audio_value(value.audio),
            "model": value.model,
            "voice": value.voice,
        }
    return _UNHANDLED


def _prompt_mapping(value: Mapping[Any, Any], seen: set[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        normalized = _normalized_key(key)
        if _sensitive_key(key):
            result[str(key)] = _REDACTED
        elif normalized in {"b64json", "base64"} and isinstance(item, str):
            result[str(key)] = {
                "binary_omitted": True,
                "encoded_length": len(item),
                "encoded_sha256": hashlib.sha256(item.encode()).hexdigest(),
            }
        else:
            result[str(key)] = prompt_jsonable(item, _seen=seen)
    return result


def _prompt_object(value: Any, seen: set[int]) -> Any:
    if isinstance(value, Mapping):
        return _prompt_mapping(value, seen)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [prompt_jsonable(item, _seen=seen) for item in value]
    if is_dataclass(value):
        return {
            item.name: (
                _REDACTED
                if _sensitive_key(item.name)
                else prompt_jsonable(getattr(value, item.name), _seen=seen)
            )
            for item in fields(value)
            if item.name
            not in {
                "capability_request",
                "capability_output",
                "transient_source_marker_present",
            }
        }
    dumper = getattr(value, "model_dump", None)
    if callable(dumper):
        return prompt_jsonable(dumper(), _seen=seen)
    attributes = _prompt_attributes(value)
    if attributes:
        return _prompt_mapping(attributes, seen)
    return {"type": type(value).__name__, "value": "[unsupported object omitted]"}


def _prompt_attributes(value: Any) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    with_dict = getattr(value, "__dict__", None)
    if isinstance(with_dict, Mapping):
        attributes.update(with_dict)
    for owner in type(value).__mro__:
        for name in getattr(owner, "__slots__", ()):
            if isinstance(name, str) and hasattr(value, name):
                attributes.setdefault(name, getattr(value, name))
    if not attributes:
        for name, item in vars(type(value)).items():
            if not name.startswith("_") and not callable(item):
                attributes[name] = getattr(value, name, item)
    return {
        str(name): item
        for name, item in attributes.items()
        if not str(name).startswith("_") and not callable(item)
    }


def prompt_jsonable(value: Any, *, _seen: set[int] | None = None) -> Any:
    """Convert diagnostic values without copying credentials or binary payloads."""

    audio_value = _audio_value(value)
    if audio_value is not _UNHANDLED:
        return audio_value
    scalar = _prompt_scalar(value)
    if scalar is not _UNHANDLED:
        return scalar

    seen = _seen if _seen is not None else set()
    identity = id(value)
    if identity in seen:
        return {"recursive_reference": type(value).__name__}
    seen.add(identity)
    try:
        return _prompt_object(value, seen)
    finally:
        seen.discard(identity)


def capability_payload_view(capability: object, payload: Mapping[str, Any]) -> Any:
    """Project an invocation payload without durable audio source identifiers.

    Audio requests are intentionally allowlisted.  This prevents a caller's
    compatibility fields (raw/base64/path/url/hash) from reaching work records
    even when they are ignored by the actual provider adapter.
    """

    if str(capability or "").strip().lower() not in _AUDIO_CAPABILITIES:
        return prompt_jsonable(payload)
    projected: dict[str, Any] = {}
    for key, value in payload.items():
        normalized = _normalized_key(key)
        if normalized not in _AUDIO_DIAGNOSTIC_FIELDS:
            continue
        if normalized in _AUDIO_TEXT_FIELDS:
            projected[f"{key}_length"] = len(str(value or ""))
        else:
            projected[str(key)] = prompt_jsonable(value)
    return projected


def prompt_request_view(request: AIModelRequest) -> dict[str, Any]:
    value: dict[str, Any] = {
        "invocation_id": request.invocation_id,
        "owner_kind": request.owner_kind,
        "owner_id": request.owner_id,
        "execution_mode": request.execution_mode,
        "backend_ids": request.backend_ids,
        "model": request.model,
        "logical_prompt": request.logical_document,
        "context_text": request.context_text,
        "turn_text": request.turn_text,
        "input_images": request.input_images,
        "agent_tools": request.agent_tools,
        "agent_history": request.agent_history,
        "parameters": request.parameters,
        "metadata": request.metadata,
    }
    capability_request = request.capability_request
    if isinstance(capability_request, AICapabilityRequest):
        value["capability_request"] = {
            "capability": capability_request.capability,
            "effect": capability_request.effect,
            "payload": capability_payload_view(
                capability_request.capability,
                capability_request.payload,
            ),
            "metadata": capability_request.metadata,
        }
    return prompt_jsonable(value)


def prompt_response_view(completion: AICompletion) -> dict[str, Any]:
    value = {
        "text": completion.text,
        "finish_reason": completion.finish_reason,
        "usage": completion.usage,
        "model": completion.model,
        "capability_output": completion.capability_output,
        "agent_output_items": completion.agent_output_items,
        "agent_transport_mode": completion.agent_transport_mode,
    }
    return prompt_jsonable(value)


__all__ = [
    "capability_payload_view",
    "prompt_jsonable",
    "prompt_request_view",
    "prompt_response_view",
    "redact_prompt_text",
]
