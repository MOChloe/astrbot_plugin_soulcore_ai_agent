"""Build audio capability adapters from persisted runtime configuration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..contracts.ai_models import AICapabilityName
from ..features.ai.audio_capabilities import (
    GPTSoVITSSpeechCapabilityAdapter,
    GPTSoVITSSpeechConfig,
    GSVISpeechCapabilityAdapter,
    GSVISpeechConfig,
    MiMoAudioCapabilityAdapter,
    MiMoAudioConfig,
    MiniMaxSpeechCapabilityAdapter,
    MiniMaxSpeechConfig,
    OpenAIAudioCapabilityAdapter,
    OpenAIAudioConfig,
)

AUDIO_CAPABILITY_NAMES = frozenset(
    {
        AICapabilityName.AUDIO_TRANSCRIBE.value,
        AICapabilityName.AUDIO_SPEECH.value,
    }
)


@dataclass(frozen=True, slots=True)
class AudioRuntimeModel:
    """Resolved package/model inputs needed to build one audio adapter."""

    package_kind: str
    base_url: str
    credential_id: str
    model_key: str
    capabilities: frozenset[str]
    config: Mapping[str, Any]

    @classmethod
    def from_records(
        cls,
        *,
        package_kind: str,
        model: Mapping[str, Any],
        package: Mapping[str, Any],
        base_url: str,
        credential_id: str,
        model_key: str,
        capabilities: set[str],
    ) -> AudioRuntimeModel:
        return cls(
            package_kind=package_kind,
            base_url=base_url,
            credential_id=credential_id,
            model_key=model_key,
            capabilities=frozenset(capabilities),
            config={**dict(package.get("config") or {}), **dict(model.get("config") or {})},
        )


class AudioRuntimeAdapterFactory:
    """Dispatch protocol-specific audio configuration to its adapter builder."""

    def __init__(self, credential_resolver: Callable[[str], Any]) -> None:
        self._credential_resolver = credential_resolver

    def build(self, runtime_model: AudioRuntimeModel) -> Any | None:
        builders = {
            "openai": self._openai,
            "openai_compatible": self._openai,
            "minimax_tts": self._minimax,
            "mimo_tts": self._mimo,
            "gpt_sovits_v2": self._gpt_sovits,
            "gsvi_tts": self._gsvi,
        }
        builder = builders.get(runtime_model.package_kind)
        return builder(runtime_model) if builder is not None else None

    def _openai(self, item: AudioRuntimeModel) -> OpenAIAudioCapabilityAdapter:
        config = item.config
        return OpenAIAudioCapabilityAdapter(
            OpenAIAudioConfig(
                base_url=item.base_url,
                credential_id=item.credential_id,
                default_model=item.model_key,
                capabilities=tuple(sorted(item.capabilities)),
                transcription_endpoint=str(config.get("transcription_endpoint") or ""),
                speech_endpoint=str(config.get("speech_endpoint") or ""),
                voice=str(config.get("voice") or "alloy"),
                audio_format=str(config.get("audio_format") or "mp3"),
                language=str(config.get("language") or ""),
                max_input_bytes=_config_int(
                    config.get("max_input_bytes"), 50 * 1024 * 1024, minimum=1
                ),
                max_output_bytes=_config_int(
                    config.get("max_output_bytes"),
                    128 * 1024 * 1024,
                    minimum=1,
                    maximum=128 * 1024 * 1024,
                ),
                extra_headers=_config_mapping(config.get("extra_headers")),
            ),
            self._credential_resolver,
        )

    def _minimax(self, item: AudioRuntimeModel) -> MiniMaxSpeechCapabilityAdapter | None:
        if AICapabilityName.AUDIO_SPEECH.value not in item.capabilities:
            return None
        config = item.config
        return MiniMaxSpeechCapabilityAdapter(
            MiniMaxSpeechConfig(
                base_url=item.base_url,
                credential_id=item.credential_id,
                default_model=item.model_key,
                endpoint=str(config.get("endpoint") or ""),
                voice_id=str(config.get("voice_id") or ""),
                audio_format=str(config.get("audio_format") or "mp3"),
                sample_rate=_config_int(config.get("sample_rate"), 32000, minimum=8000),
                bitrate=_config_int(config.get("bitrate"), 128000, minimum=32000),
                channel=_config_int(config.get("channel"), 1, minimum=1, maximum=2),
                language_boost=str(config.get("language_boost") or "auto"),
                speed=_config_float(config.get("speed"), 1.0),
                volume=_config_float(config.get("volume"), 1.0),
                pitch=_config_int(config.get("pitch"), 0, minimum=-12, maximum=12),
                emotion=str(config.get("emotion") or ""),
                max_output_bytes=_config_int(
                    config.get("max_output_bytes"),
                    128 * 1024 * 1024,
                    minimum=1,
                    maximum=128 * 1024 * 1024,
                ),
                extra_headers=_config_mapping(config.get("extra_headers")),
            ),
            self._credential_resolver,
        )

    def _mimo(self, item: AudioRuntimeModel) -> MiMoAudioCapabilityAdapter:
        config = item.config
        return MiMoAudioCapabilityAdapter(
            MiMoAudioConfig(
                base_url=item.base_url,
                credential_id=item.credential_id,
                default_model=item.model_key,
                capabilities=tuple(sorted(item.capabilities)),
                endpoint=str(config.get("endpoint") or ""),
                voice=str(config.get("voice") or "mimo_default"),
                audio_format=str(config.get("audio_format") or "wav"),
                language=str(config.get("language") or "auto"),
                instruction=str(config.get("instruction") or ""),
                auth_mode=str(config.get("auth_mode") or "api_key"),
                max_input_bytes=_config_int(
                    config.get("max_input_bytes"), 10 * 1024 * 1024, minimum=1
                ),
                max_output_bytes=_config_int(
                    config.get("max_output_bytes"),
                    128 * 1024 * 1024,
                    minimum=1,
                    maximum=128 * 1024 * 1024,
                ),
                extra_headers=_config_mapping(config.get("extra_headers")),
            ),
            self._credential_resolver,
        )

    def _gpt_sovits(self, item: AudioRuntimeModel) -> GPTSoVITSSpeechCapabilityAdapter | None:
        if AICapabilityName.AUDIO_SPEECH.value not in item.capabilities:
            return None
        config = item.config
        return GPTSoVITSSpeechCapabilityAdapter(
            GPTSoVITSSpeechConfig(
                base_url=item.base_url,
                credential_id=item.credential_id,
                default_model=item.model_key,
                endpoint=str(config.get("endpoint") or ""),
                ref_audio_path=str(config.get("ref_audio_path") or ""),
                prompt_text=str(config.get("prompt_text") or ""),
                prompt_lang=str(config.get("prompt_lang") or ""),
                text_lang=str(config.get("text_lang") or ""),
                media_type=str(config.get("media_type") or "wav"),
                text_split_method=str(config.get("text_split_method") or "cut5"),
                top_k=_config_int(config.get("top_k"), 15, minimum=1),
                top_p=_config_float(config.get("top_p"), 1.0),
                temperature=_config_float(config.get("temperature"), 1.0),
                batch_size=_config_int(config.get("batch_size"), 1, minimum=1),
                speed_factor=_config_float(config.get("speed_factor"), 1.0),
                repetition_penalty=_config_float(config.get("repetition_penalty"), 1.35),
                max_output_bytes=_config_int(
                    config.get("max_output_bytes"),
                    128 * 1024 * 1024,
                    minimum=1,
                    maximum=128 * 1024 * 1024,
                ),
                extra_headers=_config_mapping(config.get("extra_headers")),
            ),
            self._credential_resolver,
        )

    def _gsvi(self, item: AudioRuntimeModel) -> GSVISpeechCapabilityAdapter | None:
        if AICapabilityName.AUDIO_SPEECH.value not in item.capabilities:
            return None
        config = item.config
        return GSVISpeechCapabilityAdapter(
            GSVISpeechConfig(
                base_url=item.base_url,
                credential_id=item.credential_id,
                default_model=item.model_key,
                endpoint=str(config.get("endpoint") or ""),
                version=str(config.get("version") or "v4"),
                gpt_model_name=str(config.get("gpt_model_name") or ""),
                sovits_model_name=str(config.get("sovits_model_name") or ""),
                ref_audio_path=str(config.get("ref_audio_path") or ""),
                prompt_text=str(config.get("prompt_text") or ""),
                prompt_text_lang=str(config.get("prompt_text_lang") or ""),
                text_lang=str(config.get("text_lang") or ""),
                media_type=str(config.get("media_type") or "wav"),
                text_split_method=str(config.get("text_split_method") or "按标点符号切"),
                top_k=_config_int(config.get("top_k"), 10, minimum=1),
                top_p=_config_float(config.get("top_p"), 1.0),
                temperature=_config_float(config.get("temperature"), 1.0),
                batch_size=_config_int(config.get("batch_size"), 1, minimum=1),
                batch_threshold=_config_float(config.get("batch_threshold"), 0.75),
                split_bucket=_config_bool(config.get("split_bucket"), True),
                speed_factor=_config_float(config.get("speed_factor"), 1.0),
                fragment_interval=_config_float(config.get("fragment_interval"), 0.3),
                parallel_infer=_config_bool(config.get("parallel_infer"), True),
                repetition_penalty=_config_float(config.get("repetition_penalty"), 1.35),
                seed=_config_int(config.get("seed"), -1),
                sample_steps=_config_int(config.get("sample_steps"), 16, minimum=1),
                super_sampling=_config_bool(config.get("super_sampling"), False),
                max_output_bytes=_config_int(
                    config.get("max_output_bytes"),
                    128 * 1024 * 1024,
                    minimum=1,
                    maximum=128 * 1024 * 1024,
                ),
                extra_headers=_config_mapping(config.get("extra_headers")),
            ),
            self._credential_resolver,
        )


def _config_mapping(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _config_int(
    value: object,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    try:
        result = int(default if value is None else value)
    except (TypeError, ValueError):
        result = int(default)
    if minimum is not None:
        result = max(int(minimum), result)
    if maximum is not None:
        result = min(int(maximum), result)
    return result


def _config_float(value: object, default: float) -> float:
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return float(default)


def _config_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(default if value is None else value)


__all__ = ["AUDIO_CAPABILITY_NAMES", "AudioRuntimeAdapterFactory", "AudioRuntimeModel"]
