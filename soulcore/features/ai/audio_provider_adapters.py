"""Provider-specific audio capability adapters."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...contracts.ai_models import (
    AIAudioContent,
    AIBackendDescriptor,
    AICapabilityName,
    AICapabilityRequest,
    AISpeechResult,
    AITranscriptionResult,
)
from .audio_support import (
    _LOCAL_SPEECH_FORMATS,
    _MIMO_FORMATS,
    _MIMO_INPUT_FORMATS,
    _MINIMAX_FORMATS,
    _OPENAI_SPEECH_FORMATS,
    CredentialResolver,
    GPTSoVITSSpeechConfig,
    MiMoAudioConfig,
    MiniMaxSpeechConfig,
    OpenAIAudioConfig,
    _audio_format,
    _AudioHTTPMixin,
    _bearer_headers,
    _binary_speech_result,
    _bounded_float,
    _bounded_int,
    _check_output_size,
    _checked_json_mapping,
    _decode_audio_text,
    _endpoint,
    _ensure_success,
    _first_message,
    _format,
    _invalid,
    _json_response,
    _looks_like_audio_location,
    _mark_audio_send,
    _maybe_json,
    _mime_for_format,
    _model,
    _optional_float,
    _output_error,
    _pcm_audio_metadata,
    _request_audio,
    _require_http_url,
    _resolve_same_origin_audio_url,
    _safe_filename,
    _selected_capabilities,
    _selected_metadata,
    _speech_text,
    _timeout,
    _unsupported,
)
from .audio_transport import AudioHTTPTransport, HTTPAudioResponse, UrllibAudioTransport


class OpenAIAudioCapabilityAdapter(_AudioHTTPMixin):
    adapter_id = "openai_audio"

    def __init__(
        self,
        config: OpenAIAudioConfig,
        credential_resolver: CredentialResolver,
        transport: AudioHTTPTransport | None = None,
    ) -> None:
        _require_http_url(config.base_url, "base_url")
        self.config = config
        self.credential_resolver = credential_resolver
        self.transport = transport or UrllibAudioTransport(
            max_response_bytes=config.max_output_bytes
        )
        self.capabilities = _selected_capabilities(config.capabilities)

    async def invoke(
        self, request: AICapabilityRequest, backend: AIBackendDescriptor
    ) -> AITranscriptionResult | AISpeechResult:
        if request.capability == AICapabilityName.AUDIO_TRANSCRIBE.value:
            return await self._transcribe(request, backend)
        if request.capability == AICapabilityName.AUDIO_SPEECH.value:
            return await self._speech(request, backend)
        raise _unsupported(request.capability, backend)

    async def _transcribe(
        self, request: AICapabilityRequest, backend: AIBackendDescriptor
    ) -> AITranscriptionResult:
        audio = _request_audio(request, backend, self.config.max_input_bytes)
        model = _model(backend, self.config.default_model)
        response_format = str(request.payload.get("response_format") or "verbose_json").lower()
        if response_format not in {"json", "text", "verbose_json"}:
            raise _invalid("Unsupported transcription response format", backend)
        fields = {"model": model, "response_format": response_format}
        language = str(request.payload.get("language") or self.config.language).strip()
        prompt = str(request.payload.get("prompt") or "").strip()
        if language:
            fields["language"] = language
        if prompt:
            fields["prompt"] = prompt
        if "temperature" in request.payload:
            fields["temperature"] = str(
                _bounded_float(request.payload.get("temperature"), 0.0, 1.0, 0.0)
            )
        filename = _safe_filename(audio.filename, audio.mime_type)
        endpoint = _endpoint(
            self.config.base_url,
            self.config.transcription_endpoint,
            "/audio/transcriptions",
        )
        headers = _bearer_headers(
            self._credential(self.config.credential_id, backend.backend_id, required=True),
            self.config.extra_headers,
        )
        await _mark_audio_send(
            provider="openai_compatible",
            operation="audio.transcribe",
            content_type="multipart/form-data",
            audio=audio,
            values={key: value for key, value in fields.items() if key != "prompt"},
        )
        response = await self.transport.post_multipart(
            endpoint,
            headers=headers,
            fields=fields,
            files=(("file", filename, audio.mime_type, audio.data),),
            timeout_seconds=_timeout(request),
        )
        _ensure_success(response, backend)
        parsed = _maybe_json(response)
        if isinstance(parsed, Mapping):
            text = str(parsed.get("text") or "").strip()
            result_language = str(parsed.get("language") or language)
            duration = _optional_float(parsed.get("duration"))
            metadata = _selected_metadata(parsed, ("usage",))
        else:
            text = response.body.decode("utf-8", errors="replace").strip()
            result_language, duration, metadata = language, audio.duration_seconds, {}
        if not text:
            raise _output_error("Audio backend returned an empty transcription", backend)
        return AITranscriptionResult(text, result_language, duration, model, metadata)

    async def _speech(
        self, request: AICapabilityRequest, backend: AIBackendDescriptor
    ) -> AISpeechResult:
        text = _speech_text(request, backend)
        model = _model(backend, self.config.default_model)
        voice = str(request.payload.get("voice") or self.config.voice).strip()
        if not voice:
            raise _invalid("audio.speech requires a configured voice", backend)
        audio_format = _format(
            request.payload.get("audio_format")
            or request.payload.get("response_format")
            or self.config.audio_format,
            _OPENAI_SPEECH_FORMATS,
            backend,
        )
        payload: dict[str, object] = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": audio_format,
        }
        if "speed" in request.payload:
            payload["speed"] = _bounded_float(request.payload.get("speed"), 0.25, 4.0, 1.0)
        instructions = str(request.payload.get("instruction") or "").strip()
        if instructions:
            payload["instructions"] = instructions
        endpoint = _endpoint(self.config.base_url, self.config.speech_endpoint, "/audio/speech")
        headers = _bearer_headers(
            self._credential(self.config.credential_id, backend.backend_id, required=True),
            self.config.extra_headers,
        )
        await _mark_audio_send(
            provider="openai_compatible",
            operation="audio.speech",
            content_type="application/json",
            values={
                "model": model,
                "text_length": len(text),
                "voice": voice,
                "audio_format": audio_format,
            },
        )
        response = await self.transport.post_json(
            endpoint,
            headers=headers,
            payload=payload,
            timeout_seconds=_timeout(request),
        )
        return _binary_speech_result(
            response,
            backend,
            model=model,
            voice=voice,
            audio_format=audio_format,
            maximum=self.config.max_output_bytes,
        )


@dataclass(frozen=True, slots=True)
class _MiniMaxPreparedRequest:
    text: str
    model: str
    voice: str
    audio_format: str
    sample_rate: int
    channels: int
    payload: Mapping[str, object]


class MiniMaxSpeechCapabilityAdapter(_AudioHTTPMixin):
    adapter_id = "minimax_tts"
    capabilities = (AICapabilityName.AUDIO_SPEECH.value,)

    def __init__(
        self,
        config: MiniMaxSpeechConfig,
        credential_resolver: CredentialResolver,
        transport: AudioHTTPTransport | None = None,
    ) -> None:
        _require_http_url(config.base_url, "base_url")
        self.config = config
        self.credential_resolver = credential_resolver
        self.transport = transport or UrllibAudioTransport(
            max_response_bytes=config.max_output_bytes
        )

    async def invoke(
        self, request: AICapabilityRequest, backend: AIBackendDescriptor
    ) -> AISpeechResult:
        if request.capability != AICapabilityName.AUDIO_SPEECH.value:
            raise _unsupported(request.capability, backend)
        prepared = self._prepare_request(request, backend)
        await self._track_request(prepared)
        response = await self.transport.post_json(
            _endpoint(self.config.base_url, self.config.endpoint, "/v1/t2a_v2"),
            headers=self._headers(backend),
            payload=prepared.payload,
            timeout_seconds=_timeout(request),
        )
        return await self._result(response, request, backend, prepared)

    def _prepare_request(
        self, request: AICapabilityRequest, backend: AIBackendDescriptor
    ) -> _MiniMaxPreparedRequest:
        text = _speech_text(request, backend, maximum=10000)
        model = _model(backend, self.config.default_model)
        voice = str(request.payload.get("voice") or self.config.voice_id).strip()
        if not voice:
            raise _invalid("MiniMax speech requires voice_id", backend)
        audio_format = _format(
            request.payload.get("audio_format") or self.config.audio_format,
            _MINIMAX_FORMATS,
            backend,
        )
        sample_rate = max(8000, int(self.config.sample_rate))
        channels = 2 if int(self.config.channel) == 2 else 1
        payload: dict[str, object] = {
            "model": model,
            "text": text,
            "stream": False,
            "output_format": "hex",
            "language_boost": str(self.config.language_boost or "auto"),
            "voice_setting": self._voice_setting(request, voice),
            "audio_setting": {
                "sample_rate": sample_rate,
                "bitrate": max(32000, int(self.config.bitrate)),
                "format": audio_format,
                "channel": channels,
            },
        }
        return _MiniMaxPreparedRequest(
            text,
            model,
            voice,
            audio_format,
            sample_rate,
            channels,
            payload,
        )

    def _voice_setting(self, request: AICapabilityRequest, voice: str) -> dict[str, object]:
        setting: dict[str, object] = {
            "voice_id": voice,
            "speed": _bounded_float(request.payload.get("speed"), 0.5, 2.0, self.config.speed),
            "vol": _bounded_float(request.payload.get("volume"), 0.1, 10.0, self.config.volume),
            "pitch": _bounded_int(request.payload.get("pitch"), -12, 12, self.config.pitch),
        }
        emotion = str(request.payload.get("emotion") or self.config.emotion).strip()
        if emotion:
            setting["emotion"] = emotion
        return setting

    def _headers(self, backend: AIBackendDescriptor) -> dict[str, str]:
        credential = self._credential(
            self.config.credential_id,
            backend.backend_id,
            required=True,
        )
        return _bearer_headers(credential, self.config.extra_headers)

    @staticmethod
    async def _track_request(prepared: _MiniMaxPreparedRequest) -> None:
        await _mark_audio_send(
            provider="minimax",
            operation="audio.speech",
            content_type="application/json",
            values={
                "model": prepared.model,
                "text_length": len(prepared.text),
                "voice": prepared.voice,
                "audio_format": prepared.audio_format,
                "sample_rate": prepared.sample_rate,
            },
        )

    async def _result(
        self,
        response: HTTPAudioResponse,
        request: AICapabilityRequest,
        backend: AIBackendDescriptor,
        prepared: _MiniMaxPreparedRequest,
    ) -> AISpeechResult:
        parsed = _maybe_json(response)
        if parsed is None:
            return _binary_speech_result(
                response,
                backend,
                model=prepared.model,
                voice=prepared.voice,
                audio_format=prepared.audio_format,
                maximum=self.config.max_output_bytes,
                pcm_sample_rate_hz=prepared.sample_rate,
                pcm_channels=prepared.channels,
            )
        data = _checked_json_mapping(response, parsed, backend)
        self._ensure_provider_success(data, backend)
        audio_bytes = await self._audio_bytes(data, request, backend)
        extra = self._extra_info(data)
        duration_ms = _optional_float(extra.get("audio_length"))
        duration = duration_ms / 1000.0 if duration_ms is not None else None
        return AISpeechResult(
            AIAudioContent(
                audio_bytes,
                _mime_for_format(prepared.audio_format),
                f"speech.{prepared.audio_format}",
                duration,
                _pcm_audio_metadata(
                    prepared.audio_format,
                    sample_rate_hz=prepared.sample_rate,
                    channels=prepared.channels,
                ),
            ),
            prepared.model,
            prepared.voice,
            _selected_metadata(
                extra,
                ("audio_length", "audio_sample_rate", "audio_size", "bitrate", "word_count"),
            ),
        )

    @staticmethod
    def _ensure_provider_success(data: Mapping[str, Any], backend: AIBackendDescriptor) -> None:
        base_response = data.get("base_resp")
        if isinstance(base_response, Mapping) and int(base_response.get("status_code") or 0) != 0:
            raise _output_error("MiniMax speech generation failed", backend)

    async def _audio_bytes(
        self,
        data: Mapping[str, Any],
        request: AICapabilityRequest,
        backend: AIBackendDescriptor,
    ) -> bytes:
        encoded = self._encoded_audio(data)
        if _looks_like_audio_location(encoded):
            audio_bytes = await self._download_audio(encoded, request, backend)
        else:
            audio_bytes = _decode_audio_text(encoded, backend, provider="MiniMax")
        if not audio_bytes:
            raise _output_error("MiniMax returned empty audio", backend)
        _check_output_size(audio_bytes, self.config.max_output_bytes, backend)
        return audio_bytes

    @staticmethod
    def _encoded_audio(data: Mapping[str, Any]) -> str:
        result = data.get("data")
        values: Mapping[str, Any] = result if isinstance(result, Mapping) else {}
        return str(
            values.get("audio")
            or values.get("audio_url")
            or data.get("audio")
            or data.get("audio_url")
            or ""
        ).strip()

    async def _download_audio(
        self,
        location: str,
        request: AICapabilityRequest,
        backend: AIBackendDescriptor,
    ) -> bytes:
        audio_url = _resolve_same_origin_audio_url(self.config.base_url, location, backend)
        response = await self.transport.get_bytes(
            audio_url,
            headers={"Accept": "audio/*"},
            timeout_seconds=_timeout(request),
        )
        _ensure_success(response, backend)
        if _maybe_json(response) is not None:
            raise _output_error("MiniMax audio download returned JSON", backend)
        return response.body

    @staticmethod
    def _extra_info(data: Mapping[str, Any]) -> Mapping[str, Any]:
        value = data.get("extra_info")
        return value if isinstance(value, Mapping) else {}


class MiMoAudioCapabilityAdapter(_AudioHTTPMixin):
    adapter_id = "mimo_audio"

    def __init__(
        self,
        config: MiMoAudioConfig,
        credential_resolver: CredentialResolver,
        transport: AudioHTTPTransport | None = None,
    ) -> None:
        _require_http_url(config.base_url, "base_url")
        self.config = config
        self.credential_resolver = credential_resolver
        self.transport = transport or UrllibAudioTransport(
            max_response_bytes=config.max_output_bytes
        )
        self.capabilities = _selected_capabilities(config.capabilities)

    async def invoke(
        self, request: AICapabilityRequest, backend: AIBackendDescriptor
    ) -> AITranscriptionResult | AISpeechResult:
        if request.capability == AICapabilityName.AUDIO_TRANSCRIBE.value:
            return await self._transcribe(request, backend)
        if request.capability == AICapabilityName.AUDIO_SPEECH.value:
            return await self._speech(request, backend)
        raise _unsupported(request.capability, backend)

    def _headers(self, backend: AIBackendDescriptor) -> dict[str, str]:
        credential = self._credential(
            self.config.credential_id,
            backend.backend_id,
            required=True,
        )
        if str(self.config.auth_mode).strip().lower() == "bearer":
            return _bearer_headers(credential, self.config.extra_headers)
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "api-key": credential,
            **{str(key): str(value) for key, value in self.config.extra_headers.items()},
        }

    async def _transcribe(
        self, request: AICapabilityRequest, backend: AIBackendDescriptor
    ) -> AITranscriptionResult:
        audio = _request_audio(request, backend, self.config.max_input_bytes)
        audio_format = _format(_audio_format(audio), _MIMO_INPUT_FORMATS, backend)
        model = _model(backend, self.config.default_model)
        language = str(request.payload.get("language") or self.config.language or "auto")
        encoded = base64.b64encode(audio.data).decode("ascii")
        payload: dict[str, object] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:{audio.mime_type};base64,{encoded}",
                                "format": audio_format,
                            },
                        }
                    ],
                }
            ],
            "asr_options": {"language": language},
            "stream": False,
        }
        endpoint = _endpoint(self.config.base_url, self.config.endpoint, "/chat/completions")
        await _mark_audio_send(
            provider="mimo",
            operation="audio.transcribe",
            content_type="application/json",
            audio=audio,
            values={"model": model, "language": language, "audio_format": audio_format},
        )
        response = await self.transport.post_json(
            endpoint,
            headers=self._headers(backend),
            payload=payload,
            timeout_seconds=_timeout(request),
        )
        data = _json_response(response, backend)
        message = _first_message(data, backend)
        text = str(message.get("content") or "").strip()
        if not text:
            raise _output_error("MiMo returned an empty transcription", backend)
        usage = data.get("usage")
        usage_values: Mapping[str, Any] = usage if isinstance(usage, Mapping) else {}
        duration = _optional_float(usage_values.get("seconds")) or audio.duration_seconds
        return AITranscriptionResult(
            text,
            "" if language == "auto" else language,
            duration,
            str(data.get("model") or model),
            _selected_metadata(usage_values, ("seconds", "prompt_tokens", "completion_tokens")),
        )

    async def _speech(
        self, request: AICapabilityRequest, backend: AIBackendDescriptor
    ) -> AISpeechResult:
        text = _speech_text(request, backend)
        model = _model(backend, self.config.default_model)
        voice = str(request.payload.get("voice") or self.config.voice).strip()
        if not voice:
            raise _invalid("MiMo speech requires a voice", backend)
        audio_format = _format(
            request.payload.get("audio_format") or self.config.audio_format,
            _MIMO_FORMATS,
            backend,
        )
        messages: list[dict[str, object]] = []
        instruction = str(request.payload.get("instruction") or self.config.instruction).strip()
        if instruction:
            messages.append({"role": "user", "content": instruction})
        messages.append({"role": "assistant", "content": text})
        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "audio": {"format": audio_format, "voice": voice},
            "stream": False,
        }
        endpoint = _endpoint(self.config.base_url, self.config.endpoint, "/chat/completions")
        await _mark_audio_send(
            provider="mimo",
            operation="audio.speech",
            content_type="application/json",
            values={
                "model": model,
                "text_length": len(text),
                "voice": voice,
                "audio_format": audio_format,
            },
        )
        response = await self.transport.post_json(
            endpoint,
            headers=self._headers(backend),
            payload=payload,
            timeout_seconds=_timeout(request),
        )
        data = _json_response(response, backend)
        message = _first_message(data, backend)
        audio_value = message.get("audio")
        if not isinstance(audio_value, Mapping):
            raise _output_error("MiMo response has no audio output", backend)
        encoded = str(audio_value.get("data") or "")
        try:
            audio_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise _output_error("MiMo returned invalid audio encoding", backend) from None
        if not audio_bytes:
            raise _output_error("MiMo returned empty audio", backend)
        _check_output_size(audio_bytes, self.config.max_output_bytes, backend)
        transcript = str(audio_value.get("transcript") or "")
        return AISpeechResult(
            AIAudioContent(
                audio_bytes,
                _mime_for_format(audio_format),
                f"speech.{audio_format}",
                metadata=_pcm_audio_metadata(
                    audio_format,
                    sample_rate_hz=24_000,
                    channels=1,
                ),
            ),
            str(data.get("model") or model),
            voice,
            {"transcript_length": len(transcript)} if transcript else {},
        )


class GPTSoVITSSpeechCapabilityAdapter(_AudioHTTPMixin):
    adapter_id = "gpt_sovits_v2_tts"
    capabilities = (AICapabilityName.AUDIO_SPEECH.value,)

    def __init__(
        self,
        config: GPTSoVITSSpeechConfig,
        credential_resolver: CredentialResolver,
        transport: AudioHTTPTransport | None = None,
    ) -> None:
        _require_http_url(config.base_url, "base_url")
        self.config = config
        self.credential_resolver = credential_resolver
        self.transport = transport or UrllibAudioTransport(
            max_response_bytes=config.max_output_bytes
        )

    async def invoke(
        self, request: AICapabilityRequest, backend: AIBackendDescriptor
    ) -> AISpeechResult:
        if request.capability != AICapabilityName.AUDIO_SPEECH.value:
            raise _unsupported(request.capability, backend)
        text = _speech_text(request, backend)
        if (
            not self.config.ref_audio_path
            or not self.config.prompt_lang
            or not self.config.text_lang
        ):
            raise _invalid("GPT-SoVITS reference audio and languages are not configured", backend)
        audio_format = _format(
            request.payload.get("audio_format") or self.config.media_type,
            _LOCAL_SPEECH_FORMATS,
            backend,
        )
        payload: dict[str, object] = {
            "text": text,
            "text_lang": self.config.text_lang,
            "ref_audio_path": self.config.ref_audio_path,
            "prompt_text": self.config.prompt_text,
            "prompt_lang": self.config.prompt_lang,
            "top_k": max(1, int(self.config.top_k)),
            "top_p": _bounded_float(self.config.top_p, 0.0, 1.0, 1.0),
            "temperature": _bounded_float(self.config.temperature, 0.0, 2.0, 1.0),
            "text_split_method": self.config.text_split_method,
            "batch_size": max(1, int(self.config.batch_size)),
            "speed_factor": _bounded_float(
                request.payload.get("speed"), 0.25, 4.0, self.config.speed_factor
            ),
            "media_type": audio_format,
            "streaming_mode": False,
            "repetition_penalty": max(0.1, float(self.config.repetition_penalty)),
        }
        endpoint = _endpoint(self.config.base_url, self.config.endpoint, "/tts")
        credential = self._credential(
            self.config.credential_id,
            backend.backend_id,
            required=False,
        )
        headers = (
            _bearer_headers(credential, self.config.extra_headers)
            if credential
            else {
                "Accept": "audio/*",
                **{str(key): str(value) for key, value in self.config.extra_headers.items()},
            }
        )
        await _mark_audio_send(
            provider="gpt_sovits_v2",
            operation="audio.speech",
            content_type="application/json",
            values={
                "text_length": len(text),
                "text_lang": self.config.text_lang,
                "prompt_lang": self.config.prompt_lang,
                "audio_format": audio_format,
                "model": backend.model,
            },
        )
        response = await self.transport.post_json(
            endpoint,
            headers=headers,
            payload=payload,
            timeout_seconds=_timeout(request),
        )
        return _binary_speech_result(
            response,
            backend,
            model=backend.model or self.config.default_model,
            voice=backend.model,
            audio_format=audio_format,
            maximum=self.config.max_output_bytes,
        )
