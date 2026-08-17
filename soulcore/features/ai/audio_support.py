"""Shared contracts, validation, and response handling for audio providers."""

from __future__ import annotations

import base64
import binascii
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast
from urllib.parse import urljoin, urlsplit

from ...contracts.ai_models import (
    AIAudioContent,
    AIBackendDescriptor,
    AICapabilityName,
    AICapabilityRequest,
    AIErrorCode,
    AIErrorInfo,
    AIInvocationError,
    AISpeechResult,
)
from ...shared.http_security import require_secure_http_url, same_origin
from .audio_transport import HTTPAudioResponse
from .openai_compatible import OpenAIHTTPStatusError, OpenAITransportError
from .transport_tracking import mark_transport_send

CredentialResolver = Callable[[str], str]

_AUDIO_MIME_BY_FORMAT = {
    "aac": "audio/aac",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    "mp3": "audio/mpeg",
    "mpeg": "audio/mpeg",
    "ogg": "audio/ogg",
    "opus": "audio/ogg",
    "pcm": "audio/pcm",
    "pcm16": "audio/pcm",
    "wav": "audio/wav",
}
_FORMAT_BY_AUDIO_MIME = {
    "audio/aac": "aac",
    "audio/flac": "flac",
    "audio/m4a": "m4a",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
    "audio/opus": "opus",
    "audio/pcm": "pcm",
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
}
_OPENAI_SPEECH_FORMATS = frozenset({"aac", "flac", "mp3", "opus", "pcm", "wav"})
_MIMO_FORMATS = frozenset({"mp3", "pcm", "pcm16", "wav"})
_MIMO_INPUT_FORMATS = frozenset({"mp3", "wav"})
_MINIMAX_FORMATS = frozenset({"flac", "mp3", "pcm", "wav"})
_LOCAL_SPEECH_FORMATS = frozenset({"aac", "mp3", "ogg", "wav"})


@dataclass(frozen=True, slots=True)
class OpenAIAudioConfig:
    base_url: str
    credential_id: str
    default_model: str = ""
    capabilities: tuple[str, ...] = (
        AICapabilityName.AUDIO_TRANSCRIBE.value,
        AICapabilityName.AUDIO_SPEECH.value,
    )
    transcription_endpoint: str = ""
    speech_endpoint: str = ""
    voice: str = "alloy"
    audio_format: str = "mp3"
    language: str = ""
    max_input_bytes: int = 50 * 1024 * 1024
    max_output_bytes: int = 128 * 1024 * 1024
    extra_headers: Mapping[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class MiniMaxSpeechConfig:
    base_url: str
    credential_id: str
    default_model: str = ""
    endpoint: str = ""
    voice_id: str = ""
    audio_format: str = "mp3"
    sample_rate: int = 32000
    bitrate: int = 128000
    channel: int = 1
    language_boost: str = "auto"
    speed: float = 1.0
    volume: float = 1.0
    pitch: int = 0
    emotion: str = ""
    max_output_bytes: int = 128 * 1024 * 1024
    extra_headers: Mapping[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class MiMoAudioConfig:
    base_url: str
    credential_id: str
    default_model: str = ""
    capabilities: tuple[str, ...] = (AICapabilityName.AUDIO_SPEECH.value,)
    endpoint: str = ""
    voice: str = "mimo_default"
    audio_format: str = "wav"
    language: str = "auto"
    instruction: str = ""
    auth_mode: str = "api_key"
    max_input_bytes: int = 10 * 1024 * 1024
    max_output_bytes: int = 128 * 1024 * 1024
    extra_headers: Mapping[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class GPTSoVITSSpeechConfig:
    base_url: str
    credential_id: str = ""
    default_model: str = ""
    endpoint: str = ""
    ref_audio_path: str = field(default="", repr=False)
    prompt_text: str = field(default="", repr=False)
    prompt_lang: str = ""
    text_lang: str = ""
    media_type: str = "wav"
    text_split_method: str = "cut5"
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    batch_size: int = 1
    speed_factor: float = 1.0
    repetition_penalty: float = 1.35
    max_output_bytes: int = 128 * 1024 * 1024
    extra_headers: Mapping[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class GSVISpeechConfig:
    base_url: str
    credential_id: str = ""
    default_model: str = ""
    endpoint: str = ""
    version: str = "v4"
    gpt_model_name: str = field(default="", repr=False)
    sovits_model_name: str = field(default="", repr=False)
    ref_audio_path: str = field(default="", repr=False)
    prompt_text: str = field(default="", repr=False)
    prompt_text_lang: str = ""
    text_lang: str = ""
    media_type: str = "wav"
    text_split_method: str = "按标点符号切"
    top_k: int = 10
    top_p: float = 1.0
    temperature: float = 1.0
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    seed: int = -1
    sample_steps: int = 16
    super_sampling: bool = False
    max_output_bytes: int = 128 * 1024 * 1024
    extra_headers: Mapping[str, str] = field(default_factory=dict, repr=False)


class _AudioHTTPMixin:
    image_features = None
    credential_resolver: CredentialResolver

    def _credential(self, credential_id: str, backend_id: str, *, required: bool) -> str:
        identifier = str(credential_id or "").strip()
        if not identifier:
            if required:
                raise _invocation_error(
                    AIErrorCode.AUTHENTICATION,
                    "The audio backend credential is not configured",
                    backend_id,
                    phase="prepare",
                    switch_backend=True,
                    open_circuit=True,
                )
            return ""
        try:
            value = str(self.credential_resolver(identifier) or "")
        except Exception as exc:
            raise AIInvocationError(
                AIErrorInfo(
                    AIErrorCode.AUTHENTICATION,
                    "The audio backend credential is unavailable",
                    switch_backend=True,
                    open_circuit=True,
                    backend_id=backend_id,
                    phase="prepare",
                ),
                cause=exc,
            ) from None
        if value:
            return value
        raise _invocation_error(
            AIErrorCode.AUTHENTICATION,
            "The audio backend credential is empty",
            backend_id,
            phase="prepare",
            switch_backend=True,
            open_circuit=True,
        )

    def classify_error(self, exc: BaseException, backend: AIBackendDescriptor) -> AIErrorInfo:
        if isinstance(exc, AIInvocationError):
            return exc.info
        if isinstance(exc, OpenAIHTTPStatusError):
            status = exc.status_code
            api_code = exc.api_code.lower()
            if status == 401:
                code = AIErrorCode.AUTHENTICATION
            elif status == 403:
                code = AIErrorCode.PERMISSION
            elif status == 402 or "quota" in api_code or "billing" in api_code:
                code = AIErrorCode.QUOTA_EXHAUSTED
            elif status == 429:
                code = AIErrorCode.RATE_LIMIT
            elif status in {408, 504}:
                code = AIErrorCode.TIMEOUT
            elif status >= 500:
                code = AIErrorCode.REMOTE_5XX
            else:
                code = AIErrorCode.INVALID_REQUEST
            switch = code in {
                AIErrorCode.AUTHENTICATION,
                AIErrorCode.PERMISSION,
                AIErrorCode.QUOTA_EXHAUSTED,
                AIErrorCode.RATE_LIMIT,
                AIErrorCode.TIMEOUT,
                AIErrorCode.REMOTE_5XX,
            }
            return AIErrorInfo(
                code,
                f"Audio backend returned HTTP {status}",
                retryable=code
                in {AIErrorCode.RATE_LIMIT, AIErrorCode.TIMEOUT, AIErrorCode.REMOTE_5XX},
                switch_backend=switch,
                open_circuit=code
                in {
                    AIErrorCode.AUTHENTICATION,
                    AIErrorCode.PERMISSION,
                    AIErrorCode.QUOTA_EXHAUSTED,
                    AIErrorCode.RATE_LIMIT,
                },
                retry_after_seconds=exc.retry_after_seconds,
                backend_id=backend.backend_id,
                phase="transport",
                status_code=status,
                details={"api_code": exc.api_code} if exc.api_code else {},
            )
        if isinstance(exc, (OpenAITransportError, OSError)):
            return AIErrorInfo(
                AIErrorCode.NETWORK,
                "Could not reach the audio backend",
                retryable=True,
                switch_backend=True,
                backend_id=backend.backend_id,
                phase="transport",
            )
        return AIErrorInfo(
            AIErrorCode.INTERNAL,
            f"Audio capability adapter failed: {type(exc).__name__}",
            backend_id=backend.backend_id,
            phase="adapter",
        )


def _selected_capabilities(values: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(
        capability
        for capability in (
            AICapabilityName.AUDIO_TRANSCRIBE.value,
            AICapabilityName.AUDIO_SPEECH.value,
        )
        if capability in {str(value).strip().lower() for value in values}
    )
    if not selected:
        raise ValueError("audio adapter requires at least one audio capability")
    return selected


def _request_audio(
    request: AICapabilityRequest,
    backend: AIBackendDescriptor,
    maximum: int,
) -> AIAudioContent:
    value = request.payload.get("audio")
    if not isinstance(value, AIAudioContent):
        raise _invalid("audio.transcribe requires AIAudioContent", backend)
    if not isinstance(value.data, bytes) or not value.data:
        raise _invalid("audio.transcribe requires non-empty audio bytes", backend)
    if len(value.data) > max(1, int(maximum)):
        raise _invalid("Input audio exceeds the configured size limit", backend)
    mime_type = str(request.payload.get("mime_type") or value.mime_type).strip().lower()
    if not mime_type.startswith("audio/"):
        raise _invalid("audio.transcribe requires an audio MIME type", backend)
    filename = str(request.payload.get("filename") or value.filename).strip()
    return AIAudioContent(
        value.data,
        mime_type,
        _safe_filename(filename, mime_type),
        value.duration_seconds,
    )


def _speech_text(
    request: AICapabilityRequest,
    backend: AIBackendDescriptor,
    *,
    maximum: int = 10000,
) -> str:
    text = str(request.payload.get("text") or "").strip()
    if not text:
        raise _invalid("audio.speech requires non-empty text", backend)
    if len(text) > max(1, int(maximum)):
        raise _invalid("Speech text exceeds the configured length limit", backend)
    return text


def _model(backend: AIBackendDescriptor, default_model: str) -> str:
    model = str(backend.model or default_model).strip()
    if not model:
        raise _invalid("No audio model is configured", backend)
    return model


def _format(value: object, allowed: frozenset[str], backend: AIBackendDescriptor) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "pcm16" and "pcm" in allowed:
        normalized = "pcm"
    if normalized not in allowed:
        raise _invalid("Unsupported audio format", backend)
    return normalized


def _audio_format(audio: AIAudioContent) -> str:
    return _FORMAT_BY_AUDIO_MIME.get(str(audio.mime_type).split(";", 1)[0].lower(), "")


def _mime_for_format(audio_format: str) -> str:
    return _AUDIO_MIME_BY_FORMAT.get(str(audio_format).lower(), "application/octet-stream")


def _pcm_audio_metadata(
    audio_format: str,
    *,
    sample_rate_hz: int,
    channels: int,
) -> dict[str, object]:
    """Describe raw PCM precisely enough for the delivery boundary to wrap it."""

    if str(audio_format).strip().lower() != "pcm":
        return {}
    return {
        "encoding": "pcm_s16le",
        "sample_rate_hz": int(sample_rate_hz),
        "channels": int(channels),
        "sample_width_bytes": 2,
    }


def _safe_filename(filename: str, mime_type: str) -> str:
    basename = re.split(r"[/\\]", str(filename or ""))[-1]
    basename = re.sub(r"[^A-Za-z0-9._-]", "_", basename).strip("._")
    if basename:
        return basename[:128]
    extension = _FORMAT_BY_AUDIO_MIME.get(str(mime_type).split(";", 1)[0].lower(), "bin")
    return f"audio.{extension}"


def _timeout(request: AICapabilityRequest) -> float:
    return request.retry_policy.normalized().backend_timeout_seconds


def _endpoint(base_url: str, configured: str, suffix: str) -> str:
    if str(configured or "").strip():
        return _require_http_url(str(configured).strip(), "endpoint")
    base = _require_http_url(base_url, "base_url").rstrip("/")
    normalized_suffix = "/" + str(suffix).lstrip("/")
    if base.endswith(normalized_suffix):
        return base
    if normalized_suffix.startswith("/v1/") and base.endswith("/v1"):
        normalized_suffix = normalized_suffix[3:]
    return base + normalized_suffix


def _require_http_url(value: str, label: str) -> str:
    return require_secure_http_url(value, label)


def _same_origin(base_url: str, candidate_url: str) -> bool:
    return same_origin(base_url, candidate_url)


def _resolve_same_origin_audio_url(
    base_url: str,
    candidate: str,
    backend: AIBackendDescriptor,
) -> str:
    resolved = urljoin(str(base_url).rstrip("/") + "/", str(candidate).strip())
    if not _same_origin(base_url, resolved):
        raise _output_error("Audio backend returned an untrusted audio location", backend)
    return resolved


def _looks_like_audio_location(value: str) -> bool:
    text = str(value or "").strip()
    path = urlsplit(text).path.casefold()
    return text.startswith(("http://", "https://", "/", "./", "../")) or path.endswith(
        (".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".pcm", ".wav")
    )


def _decode_audio_text(
    value: str,
    backend: AIBackendDescriptor,
    *,
    provider: str,
) -> bytes:
    encoded = str(value or "").strip()
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header.lower():
            raise _output_error(f"{provider} returned invalid audio encoding", backend)
    if encoded and len(encoded) % 2 == 0 and re.fullmatch(r"[0-9a-fA-F]+", encoded):
        try:
            return bytes.fromhex(encoded)
        except ValueError:
            pass
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise _output_error(f"{provider} returned invalid audio encoding", backend) from None


def _bearer_headers(credential: str, extra: Mapping[str, str]) -> dict[str, str]:
    return {
        "Accept": "application/json, audio/*",
        "Authorization": f"Bearer {credential}",
        **{str(key): str(value) for key, value in extra.items()},
    }


async def _mark_audio_send(
    *,
    provider: str,
    operation: str,
    content_type: str,
    values: Mapping[str, Any],
    audio: AIAudioContent | None = None,
) -> None:
    request: dict[str, Any] = {
        "method": "POST",
        "provider": provider,
        "operation": operation,
        "content_type": content_type,
        "payload": dict(values),
    }
    if audio is not None:
        request["parts"] = [
            {
                "part_name": "file",
                "mime_type": audio.mime_type,
                "size_bytes": len(audio.data),
                **(
                    {"duration_seconds": audio.duration_seconds}
                    if audio.duration_seconds is not None
                    else {}
                ),
            }
        ]
    await mark_transport_send(request)


def _ensure_success(response: HTTPAudioResponse, backend: AIBackendDescriptor) -> None:
    if 200 <= int(response.status_code) < 300:
        return
    raise OpenAIHTTPStatusError(
        int(response.status_code),
        api_code=_error_code(response.body),
    )


def _json_response(response: HTTPAudioResponse, backend: AIBackendDescriptor) -> Mapping[str, Any]:
    _ensure_success(response, backend)
    try:
        value = json.loads(response.body.decode("utf-8")) if response.body else {}
    except (UnicodeDecodeError, ValueError):
        raise _output_error("Audio backend returned invalid JSON", backend) from None
    if not isinstance(value, Mapping):
        raise _output_error("Audio backend returned an invalid response object", backend)
    return _checked_json_mapping(response, value, backend)


def _checked_json_mapping(
    response: HTTPAudioResponse,
    value: Mapping[str, Any],
    backend: AIBackendDescriptor,
) -> Mapping[str, Any]:
    _ensure_success(response, backend)
    error = value.get("error")
    if isinstance(error, Mapping):
        code = str(error.get("code") or error.get("type") or "")
        raise OpenAIHTTPStatusError(400, api_code=code)
    return value


def _maybe_json(response: HTTPAudioResponse) -> Mapping[str, Any] | None:
    content_type = _header(response.headers, "content-type").lower()
    if "json" not in content_type and not response.body.lstrip().startswith((b"{", b"[")):
        return None
    try:
        value = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, Mapping) else None


def _binary_speech_result(
    response: HTTPAudioResponse,
    backend: AIBackendDescriptor,
    *,
    model: str,
    voice: str,
    audio_format: str,
    maximum: int,
    pcm_sample_rate_hz: int = 24_000,
    pcm_channels: int = 1,
) -> AISpeechResult:
    _ensure_success(response, backend)
    if _maybe_json(response) is not None:
        raise _output_error("Speech backend returned JSON instead of audio", backend)
    if not response.body:
        raise _output_error("Speech backend returned empty audio", backend)
    _check_output_size(response.body, maximum, backend)
    mime_type = _header(response.headers, "content-type").split(";", 1)[0].strip().lower()
    if not mime_type.startswith("audio/"):
        mime_type = _mime_for_format(audio_format)
    return AISpeechResult(
        AIAudioContent(
            response.body,
            mime_type,
            f"speech.{audio_format}",
            metadata=_pcm_audio_metadata(
                audio_format,
                sample_rate_hz=pcm_sample_rate_hz,
                channels=pcm_channels,
            ),
        ),
        model,
        voice,
    )


def _check_output_size(data: bytes, maximum: int, backend: AIBackendDescriptor) -> None:
    if len(data) > max(1, int(maximum)):
        raise _output_error("Generated audio exceeds the configured size limit", backend)


def _first_message(data: Mapping[str, Any], backend: AIBackendDescriptor) -> Mapping[str, Any]:
    choices = data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, Mapping) else None
    if not isinstance(message, Mapping):
        raise _output_error("Audio backend returned no completion message", backend)
    return message


def _selected_metadata(value: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: value[key] for key in keys if key in value and value[key] is not None}


def _header(headers: Mapping[str, str], name: str) -> str:
    target = name.lower()
    return next((str(value) for key, value in headers.items() if str(key).lower() == target), "")


def _error_code(raw: bytes) -> str:
    try:
        value = json.loads(raw.decode("utf-8")) if raw else {}
    except (UnicodeDecodeError, ValueError):
        return ""
    if not isinstance(value, Mapping):
        return ""
    error = value.get("error")
    if isinstance(error, Mapping):
        return str(error.get("code") or error.get("type") or "")
    base = value.get("base_resp")
    if isinstance(base, Mapping):
        return str(base.get("status_code") or "")
    return ""


def _retry_after(headers: Mapping[str, str]) -> float | None:
    raw = _header(headers, "retry-after").strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _optional_float(value: object) -> float | None:
    try:
        result = float(cast(Any, value)) if value is not None else None
    except (TypeError, ValueError):
        return None
    return result if result is None or result >= 0 else None


def _bounded_float(value: object, minimum: float, maximum: float, default: float) -> float:
    try:
        result = float(cast(Any, default if value is None else value))
    except (TypeError, ValueError):
        result = float(default)
    return min(maximum, max(minimum, result))


def _bounded_int(value: object, minimum: int, maximum: int, default: int) -> int:
    try:
        result = int(cast(Any, default if value is None else value))
    except (TypeError, ValueError):
        result = int(default)
    return min(maximum, max(minimum, result))


def _invalid(message: str, backend: AIBackendDescriptor) -> AIInvocationError:
    return _invocation_error(
        AIErrorCode.INVALID_REQUEST, message, backend.backend_id, phase="prepare"
    )


def _unsupported(capability: str, backend: AIBackendDescriptor) -> AIInvocationError:
    return _invocation_error(
        AIErrorCode.UNSUPPORTED_CAPABILITY,
        f"Audio backend does not support {capability}",
        backend.backend_id,
        phase="prepare",
    )


def _output_error(message: str, backend: AIBackendDescriptor) -> AIInvocationError:
    return _invocation_error(
        AIErrorCode.OUTPUT_CONTRACT,
        message,
        backend.backend_id,
        phase="response",
        retryable=True,
        switch_backend=True,
    )


def _invocation_error(
    code: AIErrorCode,
    message: str,
    backend_id: str,
    *,
    phase: str,
    retryable: bool = False,
    switch_backend: bool = False,
    open_circuit: bool = False,
) -> AIInvocationError:
    return AIInvocationError(
        AIErrorInfo(
            code,
            message,
            retryable=retryable,
            switch_backend=switch_backend,
            open_circuit=open_circuit,
            backend_id=backend_id,
            phase=phase,
        )
    )
