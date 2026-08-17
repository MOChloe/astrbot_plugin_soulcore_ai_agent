"""Validated per-model generation controls for text backends.

The presence of a key in ``generation_parameters`` is also the model's
explicit declaration that the remote API accepts that key.  This matters for
OpenAI-compatible endpoints: protocol compatibility alone does not imply that
every model supports sampling or native reasoning controls.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

MODEL_GENERATION_PARAMETER_KEYS = (
    "max_tokens",
    "max_completion_tokens",
    "reasoning_effort",
    "temperature",
    "top_p",
    "top_k",
)
MODEL_OUTPUT_TOKEN_PARAMETER_KEYS = frozenset({"max_tokens", "max_completion_tokens"})
DEFAULT_MODEL_MAX_CONTEXT_TOKENS = 128_000
MINIMUM_MODEL_MAX_CONTEXT_TOKENS = 128_000
TEXT_GENERATION_CAPABILITIES = frozenset(
    {
        "chat.completion",
        "conversation.turn_buffer",
        "conversation.group_interjection",
        "conversation.group_reply_relocation",
        "conversation.timer_lifecycle_review",
        "conversation.response_polish",
        "conversation.summary",
        "memory.reasoning",
        "text.completion",
        "sticker.collect",
        "sticker.check",
        "vision.describe",
    }
)
_REASONING_EFFORT_VALUES = frozenset({"minimal", "low", "medium", "high", "xhigh", "max"})
_CUSTOM_REQUEST_RESERVED_KEYS = frozenset(
    {
        "model",
        "messages",
        "system",
        "max_tokens",
        "max_completion_tokens",
        "prompt_cache_key",
        "prompt_cache_options",
    }
)
_CUSTOM_REQUEST_MAX_BYTES = 16 * 1024
_CUSTOM_REQUEST_MAX_DEPTH = 8
_CUSTOM_REQUEST_MAX_ITEMS = 256


def _reasoning_effort(raw: Any) -> str:
    effort = str(raw or "").strip().lower()
    if effort not in _REASONING_EFFORT_VALUES:
        raise ValueError("模型原生思考强度必须是 minimal、low、medium、high、xhigh 或 max")
    return effort


def _finite_number(raw: Any, key: str) -> float:
    if isinstance(raw, bool):
        raise ValueError(f"模型生成参数 {key} 必须是数字")
    try:
        number = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"模型生成参数 {key} 必须是数字") from None
    if not math.isfinite(number):
        raise ValueError(f"模型生成参数 {key} 必须是有限数字")
    return number


def _temperature(raw: Any) -> float:
    number = _finite_number(raw, "temperature")
    if not 0.0 <= number <= 2.0:
        raise ValueError("temperature 必须在 0 到 2 之间")
    return number


def _top_p(raw: Any) -> float:
    number = _finite_number(raw, "top_p")
    if not 0.0 <= number <= 1.0:
        raise ValueError("top_p 必须在 0 到 1 之间")
    return number


def _top_k(raw: Any) -> int:
    number = _finite_number(raw, "top_k")
    if not number.is_integer() or not 1 <= number <= 1000:
        raise ValueError("top_k 必须是 1 到 1000 之间的整数")
    return int(number)


def _positive_token_limit(raw: Any, key: str) -> int:
    if isinstance(raw, bool):
        raise ValueError(f"{key} 必须是正整数")
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, float):
        if not math.isfinite(raw) or not raw.is_integer():
            raise ValueError(f"{key} 必须是正整数")
        value = int(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if not text.isascii() or not text.isdigit():
            raise ValueError(f"{key} 必须是正整数")
        value = int(text)
    else:
        raise ValueError(f"{key} 必须是正整数")
    if value < 1:
        raise ValueError(f"{key} 必须是正整数")
    return value


_NORMALIZERS = {
    "max_tokens": lambda raw: _positive_token_limit(raw, "max_tokens"),
    "max_completion_tokens": lambda raw: _positive_token_limit(raw, "max_completion_tokens"),
    "reasoning_effort": _reasoning_effort,
    "temperature": _temperature,
    "top_p": _top_p,
    "top_k": _top_k,
}


def normalize_model_generation_parameters(value: Any) -> dict[str, Any]:
    """Validate and normalize the model-owned default request parameters."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("模型生成参数必须是对象")
    unknown = {str(key) for key in value} - set(MODEL_GENERATION_PARAMETER_KEYS)
    if unknown:
        raise ValueError("不支持的模型生成参数：" + "、".join(sorted(unknown)))

    normalized = {
        key: _NORMALIZERS[key](value[key])
        for key in MODEL_GENERATION_PARAMETER_KEYS
        if key in value
    }
    if MODEL_OUTPUT_TOKEN_PARAMETER_KEYS.issubset(normalized):
        raise ValueError("max_tokens 和 max_completion_tokens 只能配置一个")
    return normalized


def _custom_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth > _CUSTOM_REQUEST_MAX_DEPTH:
        raise ValueError("模型高级请求参数嵌套过深")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("模型高级请求参数不能包含 NaN 或 Infinity")
        return value
    if isinstance(value, Mapping):
        if len(value) > _CUSTOM_REQUEST_MAX_ITEMS:
            raise ValueError("模型高级请求参数对象项目过多")
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = _custom_json_key(raw_key)
            normalized[key] = _custom_json_value(raw_value, depth=depth + 1)
        return normalized
    if isinstance(value, (list, tuple)):
        if len(value) > _CUSTOM_REQUEST_MAX_ITEMS:
            raise ValueError("模型高级请求参数数组项目过多")
        return [_custom_json_value(item, depth=depth + 1) for item in value]
    raise ValueError("模型高级请求参数只能包含 JSON 支持的值")


def _custom_json_key(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("模型高级请求参数的字段名必须是字符串")
    key = value.strip()
    invalid = (
        not key or key != value or len(key) > 128 or any(ord(character) < 32 for character in key)
    )
    if invalid:
        raise ValueError("模型高级请求参数包含无效字段名")
    if key in {"__proto__", "constructor", "prototype"}:
        raise ValueError(f"模型高级请求参数不允许字段 {key}")
    return key


def normalize_model_custom_request_parameters(value: Any) -> dict[str, Any]:
    """Validate model-owned provider fields merged into one JSON request body.

    Structural prompt fields remain owned by SoulCore. Output-token fields have
    dedicated per-model settings and cannot be duplicated here. The transport
    is deliberately non-streaming, so an explicit ``stream: false`` is
    supported for compatibility endpoints while ``true`` is rejected.
    """

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("模型高级请求参数必须是 JSON 对象")
    parameters = _custom_json_value(value)
    forbidden = set(parameters).intersection(_CUSTOM_REQUEST_RESERVED_KEYS)
    if forbidden:
        raise ValueError(
            "以下请求字段不能放在高级参数中，请使用对应的模型设置：" + "、".join(sorted(forbidden))
        )
    if "stream" in parameters and parameters["stream"] is not False:
        raise ValueError("SoulCore 当前只支持非流式响应，stream 只能设置为 false")
    try:
        encoded = json.dumps(
            parameters,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise ValueError("模型高级请求参数必须是有效 JSON") from None
    if len(encoded) > _CUSTOM_REQUEST_MAX_BYTES:
        raise ValueError("模型高级请求参数不能超过 16 KiB")
    return parameters


def resolve_model_generation_parameters(
    request_parameters: Mapping[str, Any],
    backend_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve one backend's model-owned generation controls.

    Output-token fields are accepted only from the selected model's persisted
    configuration. Task code cannot add, lower, raise, or switch either field.
    Other optional controls keep their existing per-request override behavior,
    but only when that backend explicitly declares support for them.
    """

    configured = normalize_model_generation_parameters(
        backend_metadata.get("generation_parameters")
    )
    managed = set(MODEL_GENERATION_PARAMETER_KEYS)
    resolved: dict[str, Any] = {}
    for raw_key, raw in request_parameters.items():
        key = str(raw_key)
        if key in MODEL_OUTPUT_TOKEN_PARAMETER_KEYS:
            continue
        if key not in managed or key in configured:
            resolved[key] = raw
    for key, raw in configured.items():
        if key in MODEL_OUTPUT_TOKEN_PARAMETER_KEYS:
            resolved[key] = raw
        else:
            resolved.setdefault(key, raw)
    return resolved


__all__ = [
    "MODEL_GENERATION_PARAMETER_KEYS",
    "MODEL_OUTPUT_TOKEN_PARAMETER_KEYS",
    "TEXT_GENERATION_CAPABILITIES",
    "normalize_model_custom_request_parameters",
    "normalize_model_generation_parameters",
    "resolve_model_generation_parameters",
]
