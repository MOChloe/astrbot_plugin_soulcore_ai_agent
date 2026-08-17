"""Provider-specific audio capability adapters."""

from __future__ import annotations

from ...contracts.ai_models import (
    AIBackendDescriptor,
    AICapabilityName,
    AICapabilityRequest,
    AISpeechResult,
)
from .audio_support import (
    _LOCAL_SPEECH_FORMATS,
    CredentialResolver,
    GSVISpeechConfig,
    _AudioHTTPMixin,
    _binary_speech_result,
    _bounded_float,
    _endpoint,
    _format,
    _invalid,
    _json_response,
    _mark_audio_send,
    _output_error,
    _require_http_url,
    _resolve_same_origin_audio_url,
    _speech_text,
    _timeout,
    _unsupported,
)
from .audio_transport import AudioHTTPTransport, UrllibAudioTransport


class GSVISpeechCapabilityAdapter(_AudioHTTPMixin):
    adapter_id = "gsvi_tts"
    capabilities = (AICapabilityName.AUDIO_SPEECH.value,)

    def __init__(
        self,
        config: GSVISpeechConfig,
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
        required = (
            self.config.gpt_model_name,
            self.config.sovits_model_name,
            self.config.ref_audio_path,
            self.config.prompt_text_lang,
            self.config.text_lang,
        )
        if not all(str(value).strip() for value in required):
            raise _invalid("GSVI classic inference is not fully configured", backend)
        audio_format = _format(
            request.payload.get("audio_format") or self.config.media_type,
            _LOCAL_SPEECH_FORMATS,
            backend,
        )
        credential = self._credential(
            self.config.credential_id,
            backend.backend_id,
            required=False,
        )
        payload: dict[str, object] = {
            "app_key": credential,
            "version": self.config.version,
            "gpt_model_name": self.config.gpt_model_name,
            "sovits_model_name": self.config.sovits_model_name,
            "ref_audio_path": self.config.ref_audio_path,
            "prompt_text": self.config.prompt_text,
            "prompt_text_lang": self.config.prompt_text_lang,
            "text": text,
            "text_lang": self.config.text_lang,
            "top_k": max(1, int(self.config.top_k)),
            "top_p": _bounded_float(self.config.top_p, 0.0, 1.0, 1.0),
            "temperature": _bounded_float(self.config.temperature, 0.0, 2.0, 1.0),
            "text_split_method": self.config.text_split_method,
            "batch_size": max(1, int(self.config.batch_size)),
            "batch_threshold": _bounded_float(self.config.batch_threshold, 0.0, 1.0, 0.75),
            "split_bucket": bool(self.config.split_bucket),
            "speed_facter": _bounded_float(
                request.payload.get("speed"), 0.25, 4.0, self.config.speed_factor
            ),
            "fragment_interval": max(0.0, float(self.config.fragment_interval)),
            "media_type": audio_format,
            "parallel_infer": bool(self.config.parallel_infer),
            "repetition_penalty": max(0.1, float(self.config.repetition_penalty)),
            "seed": int(self.config.seed),
            "sample_steps": max(1, int(self.config.sample_steps)),
            "if_sr": bool(self.config.super_sampling),
        }
        endpoint = _endpoint(self.config.base_url, self.config.endpoint, "/infer_classic")
        headers = {
            "Accept": "application/json",
            **{str(key): str(value) for key, value in self.config.extra_headers.items()},
        }
        await _mark_audio_send(
            provider="gsvi",
            operation="audio.speech",
            content_type="application/json",
            values={
                "status": "prepared",
                "text_length": len(text),
                "text_lang": self.config.text_lang,
                "prompt_text_lang": self.config.prompt_text_lang,
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
        data = _json_response(response, backend)
        audio_location = str(data.get("audio_url") or "").strip()
        if not audio_location:
            raise _output_error("GSVI returned no generated audio", backend)
        audio_url = _resolve_same_origin_audio_url(
            self.config.base_url,
            audio_location,
            backend,
        )
        # The returned URL stays memory-only and is not logged or copied into the result.
        downloaded = await self.transport.get_bytes(
            audio_url,
            headers={"Accept": "audio/*"},
            timeout_seconds=_timeout(request),
        )
        return _binary_speech_result(
            downloaded,
            backend,
            model=backend.model or self.config.default_model,
            voice=backend.model,
            audio_format=audio_format,
            maximum=self.config.max_output_bytes,
        )


__all__ = ["GSVISpeechCapabilityAdapter"]
