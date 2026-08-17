"""Stable, credential-safe projections for one recorded Provider attempt."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

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
