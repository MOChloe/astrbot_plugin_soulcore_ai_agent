"Stable, credential-safe projections for one recorded Provider attempt."

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from ....contracts.ai_models import AIErrorCode
from ....features.ai.prompt_debug import prompt_jsonable, redact_prompt_text


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def looks_like_internal_identifier(value: str) -> bool:
    text = str(value or "").strip()
    return bool(text) and bool(re.fullmatch(r"[A-Z][A-Z0-9_.:-]*", text))


def record_duration_ms(started_at: Any, finished_at: Any) -> int | None:
    if not started_at or not finished_at:
        return None
    try:
        start = datetime.fromisoformat(str(started_at))
        finish = datetime.fromisoformat(str(finished_at))
        return max(0, int((finish - start).total_seconds() * 1000))
    except (TypeError, ValueError):
        return None


def debug_attempt_view(attempt: Mapping[str, Any]) -> dict[str, Any]:
    request = _mapping(attempt.get("request"))
    response = _mapping(attempt.get("response"))
    evaluation = attempt.get("evaluation")
    return {
        "input": {
            "logical_prompt": redact_prompt_text(request.get("logical_prompt")),
            "context_text": redact_prompt_text(request.get("context_text")),
            "turn_text": redact_prompt_text(request.get("turn_text")),
            "ordered_messages": _ordered_model_messages(request),
            "capability_input": prompt_jsonable(request.get("capability_input")),
            "prompt_cache": prompt_jsonable(request.get("prompt_cache")),
        },
        "output": {
            "text": redact_prompt_text(response.get("text")),
            "finish_reason": redact_prompt_text(response.get("finish_reason")),
            "capability_output": prompt_jsonable(response.get("capability_output")),
            "model_visible_items": _model_visible_items(response.get("agent_output_items")),
            "agent_transport_mode": redact_prompt_text(response.get("agent_transport_mode")),
        },
        "processing": prompt_jsonable(evaluation) if evaluation is not None else None,
    }


def _ordered_model_messages(request: Mapping[str, Any]) -> Any:
    envelope = _mapping(request.get("provider_envelope"))
    payload = _mapping(envelope.get("payload"))
    if isinstance(payload.get("messages"), list):
        ordered: list[Any] = []
        system = payload.get("system")
        if system:
            ordered.append({"role": "system", "content": system})
        ordered.extend(payload["messages"])
        return _hide_transport_ids(prompt_jsonable(ordered))
    if isinstance(payload.get("input"), list):
        ordered = []
        instructions = payload.get("instructions")
        if instructions:
            ordered.append({"role": "system", "content": instructions})
        ordered.extend(payload["input"])
        return _hide_transport_ids(prompt_jsonable(ordered))
    fallback: list[Any] = []
    if request.get("context_text"):
        fallback.append({"role": "system", "content": request.get("context_text")})
    if request.get("turn_text"):
        fallback.append({"role": "user", "content": request.get("turn_text")})
    fallback.extend(request.get("agent_history") or ())
    return _hide_transport_ids(prompt_jsonable(fallback))


def _hide_transport_ids(value: Any) -> Any:
    if isinstance(value, list):
        return [_hide_transport_ids(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    return {
        str(key): (
            "[隐藏；在 Provider 原始报文中查看]"
            if str(key) in {"id", "call_id", "tool_call_id", "tool_use_id", "item_id"}
            else _hide_transport_ids(item)
        )
        for key, item in value.items()
    }


def _model_visible_items(value: Any) -> Any:
    items = value if isinstance(value, list) else ()
    return [
        {
            key: item.get(key)
            for key in ("kind", "text", "name", "raw_arguments", "argument_error")
            if item.get(key) not in (None, "", {}, [])
        }
        for item in items
        if isinstance(item, Mapping)
    ]


def raw_attempt_view(attempt: Mapping[str, Any]) -> dict[str, Any]:
    request = _mapping(attempt.get("request"))
    response = _mapping(attempt.get("response"))
    return {
        "request": prompt_jsonable(request.get("provider_envelope")),
        "response": prompt_jsonable(response.get("provider_envelope")),
    }


def debug_available(
    request: Mapping[str, Any], response: Mapping[str, Any], evaluation: Any
) -> bool:
    return any(
        (
            request.get("logical_prompt"),
            request.get("context_text"),
            request.get("turn_text"),
            request.get("agent_history"),
            request.get("capability_input") is not None,
            request.get("prompt_cache") is not None,
            response.get("text"),
            response.get("finish_reason"),
            response.get("agent_output_items"),
            response.get("capability_output") is not None,
            evaluation is not None,
        )
    )


_AUDIO_PURPOSES = {
    "AUDIO_TRANSCRIPTION": "transcription",
    "AUDIO_SPEECH_GENERATION": "speech",
}


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


ERROR_GUIDANCE: dict[AIErrorCode, tuple[str, str, str]] = {
    AIErrorCode.INVALID_REQUEST: (
        "请求数据无效",
        "本阶段没有开始执行。",
        "检查阶段输入和必填参数。",
    ),
    AIErrorCode.BACKEND_NOT_FOUND: (
        "没有可用模型",
        "本阶段无法调用所需能力。",
        "检查用途对应的模型和能力池配置。",
    ),
    AIErrorCode.UNSUPPORTED_CAPABILITY: (
        "模型不支持此能力",
        "当前后端无法完成该阶段。",
        "换用支持该能力的模型。",
    ),
    AIErrorCode.AUTHENTICATION: (
        "模型接口鉴权失败",
        "模型请求没有执行成功。",
        "检查 API Key 和接口地址。",
    ),
    AIErrorCode.PERMISSION: (
        "模型接口权限不足",
        "模型请求被服务端拒绝。",
        "检查账号、模型权限和组织设置。",
    ),
    AIErrorCode.QUOTA_EXHAUSTED: (
        "模型额度不足",
        "本阶段无法继续调用模型。",
        "充值或切换可用模型。",
    ),
    AIErrorCode.RATE_LIMIT: (
        "模型接口限流",
        "当前尝试失败，系统可能已重试。",
        "降低并发或等待额度恢复。",
    ),
    AIErrorCode.NETWORK: (
        "模型网络连接失败",
        "请求没有得到有效响应。",
        "检查网络、代理和接口地址。",
    ),
    AIErrorCode.REMOTE_5XX: ("模型服务暂时异常", "服务端没有完成请求。", "稍后重试或切换后端。"),
    AIErrorCode.TIMEOUT: (
        "模型调用超时",
        "请求结果未知或未及时返回。",
        "检查服务延迟和阶段超时配置。",
    ),
    AIErrorCode.EMPTY_OUTPUT: (
        "模型返回空结果",
        "该轮输出无法进入业务流程。",
        "检查模型、Prompt 和内容安全策略。",
    ),
    AIErrorCode.CONTEXT_BUDGET: (
        "上下文超过模型容量",
        "请求在发送前被拒绝。",
        "减少上下文或选择更大窗口模型。",
    ),
    AIErrorCode.OUTPUT_CONTRACT: (
        "模型输出格式不合格",
        "该轮结果没有被业务接受。",
        "查看校验记录并调整 Prompt。",
    ),
    AIErrorCode.SAFETY_REFUSAL: (
        "模型拒绝生成",
        "该阶段没有得到可用结果。",
        "检查输入内容和模型安全策略。",
    ),
    AIErrorCode.COMMAND_TIMEOUT: (
        "内部动作超时",
        "相关工具动作未按时完成。",
        "检查工具服务和超时配置。",
    ),
    AIErrorCode.COMMAND_FAILED: (
        "内部动作失败",
        "相关工具结果没有成功产生。",
        "展开内部动作查看参数和结果。",
    ),
    AIErrorCode.COMMAND_PROTOCOL: (
        "内部动作格式错误",
        "模型动作没有被执行。",
        "查看解析与校验记录。",
    ),
    AIErrorCode.CIRCUIT_OPEN: (
        "模型接口已临时熔断",
        "系统暂时跳过该后端。",
        "检查连续失败原因或等待自动恢复。",
    ),
    AIErrorCode.CAPACITY_BUSY: (
        "模型并发已满",
        "本次调用未获得执行容量。",
        "稍后重试或增加可用后端。",
    ),
    AIErrorCode.ADAPTER_INCOMPATIBLE: (
        "模型适配器不兼容",
        "请求无法由当前适配器处理。",
        "检查适配器版本和模型类型。",
    ),
    AIErrorCode.PROMPT_CACHE_MARKER_UNSUPPORTED: (
        "缓存标记不受支持",
        "系统已在同一轮自动移除标记重试，不影响正常回复。",
        "无需操作；系统会在冷却期后重新协商。",
    ),
    AIErrorCode.INTERNAL: (
        "AI 子系统内部错误",
        "该阶段没有正常完成。",
        "查看高级诊断并提交完整错误信息。",
    ),
}

if set(ERROR_GUIDANCE) != set(AIErrorCode):  # pragma: no cover - import-time contract
    raise RuntimeError("AI error guidance must cover every AIErrorCode")


def known_error_guidance(code: str, message: str) -> dict[str, Any] | None:
    try:
        AIErrorCode(str(code or "").upper())
    except ValueError:
        return None
    return error_view(code, message)


def error_view(code: str, message: str) -> dict[str, Any] | None:
    normalized = str(code or "").upper()
    if not normalized and not message:
        return None
    try:
        title, impact, suggestion = ERROR_GUIDANCE[AIErrorCode(normalized)]
    except ValueError:
        title, impact, suggestion = (
            "处理阶段出现问题",
            "该问题可能只影响当前阶段，具体以运行状态为准。",
            "展开高级详情查看诊断码、输入和结果。",
        )
    return {
        "code": normalized,
        "title": title,
        "message": str(message or title),
        "impact": impact,
        "suggestion": suggestion,
    }


__all__ = ["audio_attempt_summary", "error_view", "known_error_guidance"]
