"""Audio model settings exposed through the AI administration API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..presentation import jsonable

AUDIO_CAPABILITIES = frozenset({"audio.transcribe", "audio.speech"})
AUDIO_SPEECH_PROTOCOLS = frozenset({"MINIMAX_TTS", "MIMO_TTS", "GPT_SOVITS_V2", "GSVI_TTS"})
LOCAL_NO_CREDENTIAL_PROTOCOLS = frozenset({"GPT_SOVITS_V2", "GSVI_TTS"})

_AUDIO_CONFIG_KEYS = frozenset(
    {
        "endpoint",
        "transcription_endpoint",
        "speech_endpoint",
        "voice",
        "voice_id",
        "audio_format",
        "language",
        "sample_rate",
        "bitrate",
        "channel",
        "language_boost",
        "speed",
        "volume",
        "pitch",
        "emotion",
        "instruction",
        "auth_mode",
        "ref_audio_path",
        "prompt_text",
        "prompt_lang",
        "prompt_text_lang",
        "text_lang",
        "media_type",
        "text_split_method",
        "top_k",
        "top_p",
        "temperature",
        "batch_size",
        "speed_factor",
        "repetition_penalty",
        "version",
        "gpt_model_name",
        "sovits_model_name",
        "batch_threshold",
        "split_bucket",
        "fragment_interval",
        "parallel_infer",
        "seed",
        "sample_steps",
        "super_sampling",
    }
)


def audio_only(capabilities: list[str]) -> bool:
    selected = set(capabilities)
    return bool(selected) and selected <= AUDIO_CAPABILITIES


def audio_settings_view(
    config: Mapping[str, Any], capabilities: list[str]
) -> dict[str, Any] | None:
    if not set(capabilities).intersection(AUDIO_CAPABILITIES):
        return None
    return {key: jsonable(config[key]) for key in sorted(_AUDIO_CONFIG_KEYS) if key in config}


def apply_audio_settings(
    payload: Mapping[str, Any],
    config: dict[str, Any],
    capabilities: list[str],
    protocol: str,
) -> None:
    selected = set(capabilities)
    source = dict(config)
    for key in _AUDIO_CONFIG_KEYS:
        if key in payload:
            source[key] = payload[key]
        config.pop(key, None)
    if not selected.intersection(AUDIO_CAPABILITIES):
        return

    if protocol in {"OPENAI", "OPENAI_COMPATIBLE"}:
        _apply_openai_audio_settings(config, source, selected)
        return
    handlers = {
        "MINIMAX_TTS": _apply_minimax_audio_settings,
        "MIMO_TTS": _apply_mimo_audio_settings,
        "GPT_SOVITS_V2": _apply_gpt_sovits_audio_settings,
        "GSVI_TTS": _apply_gsvi_audio_settings,
    }
    handler = handlers.get(protocol)
    if handler is None:
        raise ValueError("当前 API 服务类型不支持音频模型")
    handler(config, source)


def _apply_openai_audio_settings(
    config: dict[str, Any], source: Mapping[str, Any], selected: set[str]
) -> None:
    if "audio.transcribe" in selected:
        language = _audio_text(source, "language", "")
        if language and language.lower() != "auto":
            config["language"] = language
        _copy_optional_audio_text(config, source, "transcription_endpoint")
    if "audio.speech" in selected:
        config["voice"] = _audio_text(source, "voice", "alloy")
        config["audio_format"] = _audio_format(source, "audio_format", "mp3")
        config["speed"] = _audio_float(source, "speed", 1.0, 0.25, 4.0)
        _copy_optional_audio_text(config, source, "speech_endpoint")


def _apply_minimax_audio_settings(config: dict[str, Any], source: Mapping[str, Any]) -> None:
    _copy_optional_audio_text(config, source, "endpoint")
    config.update(
        {
            "voice_id": _audio_text(source, "voice_id", required=True),
            "audio_format": _audio_format(source, "audio_format", "mp3"),
            "sample_rate": _audio_int(source, "sample_rate", 32000, 8000, 192000),
            "bitrate": _audio_int(source, "bitrate", 128000, 32000, 1_000_000),
            "channel": _audio_int(source, "channel", 1, 1, 2),
            "language_boost": _audio_text(source, "language_boost", "auto"),
            "speed": _audio_float(source, "speed", 1.0, 0.25, 4.0),
            "volume": _audio_float(source, "volume", 1.0, 0.0, 10.0),
            "pitch": _audio_int(source, "pitch", 0, -12, 12),
        }
    )
    _copy_optional_audio_text(config, source, "emotion")


def _apply_mimo_audio_settings(config: dict[str, Any], source: Mapping[str, Any]) -> None:
    _copy_optional_audio_text(config, source, "endpoint")
    auth_mode = _audio_text(source, "auth_mode", "api_key")
    if auth_mode not in {"api_key", "bearer"}:
        raise ValueError("MiMo TTS 鉴权方式必须是 api_key 或 bearer")
    config.update(
        {
            "voice": _audio_text(source, "voice", "mimo_default"),
            "audio_format": _audio_format(source, "audio_format", "wav"),
            "language": _audio_text(source, "language", "auto"),
            "auth_mode": auth_mode,
        }
    )
    _copy_optional_audio_text(config, source, "instruction")


def _apply_gpt_sovits_audio_settings(config: dict[str, Any], source: Mapping[str, Any]) -> None:
    _copy_optional_audio_text(config, source, "endpoint")
    config.update(
        {
            "ref_audio_path": _audio_text(source, "ref_audio_path", required=True),
            "prompt_text": _audio_text(source, "prompt_text", ""),
            "prompt_lang": _audio_text(source, "prompt_lang", required=True),
            "text_lang": _audio_text(source, "text_lang", required=True),
            "media_type": _audio_format(source, "media_type", "wav"),
            "text_split_method": _audio_text(source, "text_split_method", "cut5"),
            "top_k": _audio_int(source, "top_k", 15, 1, 1000),
            "top_p": _audio_float(source, "top_p", 1.0, 0.0, 1.0),
            "temperature": _audio_float(source, "temperature", 1.0, 0.0, 2.0),
            "batch_size": _audio_int(source, "batch_size", 1, 1, 128),
            "speed_factor": _audio_float(source, "speed_factor", 1.0, 0.25, 4.0),
            "repetition_penalty": _audio_float(source, "repetition_penalty", 1.35, 0.0, 10.0),
        }
    )


def _apply_gsvi_audio_settings(config: dict[str, Any], source: Mapping[str, Any]) -> None:
    _copy_optional_audio_text(config, source, "endpoint")
    config.update(
        {
            "version": _audio_text(source, "version", "v4"),
            "gpt_model_name": _audio_text(source, "gpt_model_name", required=True),
            "sovits_model_name": _audio_text(source, "sovits_model_name", required=True),
            "ref_audio_path": _audio_text(source, "ref_audio_path", required=True),
            "prompt_text": _audio_text(source, "prompt_text", ""),
            "prompt_text_lang": _audio_text(source, "prompt_text_lang", required=True),
            "text_lang": _audio_text(source, "text_lang", required=True),
            "media_type": _audio_format(source, "media_type", "wav"),
            "text_split_method": _audio_text(source, "text_split_method", "按标点符号切"),
            "top_k": _audio_int(source, "top_k", 10, 1, 1000),
            "top_p": _audio_float(source, "top_p", 1.0, 0.0, 1.0),
            "temperature": _audio_float(source, "temperature", 1.0, 0.0, 2.0),
            "batch_size": _audio_int(source, "batch_size", 1, 1, 128),
            "batch_threshold": _audio_float(source, "batch_threshold", 0.75, 0.0, 1.0),
            "split_bucket": _audio_bool(source, "split_bucket", True),
            "speed_factor": _audio_float(source, "speed_factor", 1.0, 0.25, 4.0),
            "fragment_interval": _audio_float(source, "fragment_interval", 0.3, 0.0, 10.0),
            "parallel_infer": _audio_bool(source, "parallel_infer", True),
            "repetition_penalty": _audio_float(source, "repetition_penalty", 1.35, 0.0, 10.0),
            "seed": _audio_int(source, "seed", -1, -1, 2_147_483_647),
            "sample_steps": _audio_int(source, "sample_steps", 16, 1, 1000),
            "super_sampling": _audio_bool(source, "super_sampling", False),
        }
    )


def _copy_optional_audio_text(target: dict[str, Any], source: Mapping[str, Any], key: str) -> None:
    value = str(source.get(key) or "").strip()
    if value:
        target[key] = value


def _audio_text(
    source: Mapping[str, Any], key: str, default: str = "", *, required: bool = False
) -> str:
    value = str(source.get(key, default) or default).strip()
    if any(char in value for char in "\0\r\n"):
        raise ValueError(f"语音配置 {key} 无效")
    if required and not value:
        raise ValueError(f"语音配置必须填写 {key}")
    return value


def _audio_format(source: Mapping[str, Any], key: str, default: str) -> str:
    value = _audio_text(source, key, default).lower()
    if not value or any(char in value for char in "/\\"):
        raise ValueError(f"语音配置 {key} 无效")
    return value


def _audio_float(
    source: Mapping[str, Any], key: str, default: float, minimum: float, maximum: float
) -> float:
    try:
        value = float(source.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"语音配置 {key} 必须是数字") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"语音配置 {key} 必须在 {minimum} 到 {maximum} 之间")
    return value


def _audio_int(
    source: Mapping[str, Any], key: str, default: int, minimum: int, maximum: int
) -> int:
    try:
        value = int(source.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"语音配置 {key} 必须是整数") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"语音配置 {key} 必须在 {minimum} 到 {maximum} 之间")
    return value


def _audio_bool(source: Mapping[str, Any], key: str, default: bool) -> bool:
    value = source.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"语音配置 {key} 必须是开关值")
    return value
