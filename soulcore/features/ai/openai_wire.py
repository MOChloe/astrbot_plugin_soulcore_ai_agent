"""OpenAI-compatible Agent payload and response projections."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ...contracts.ai_models import (
    AIAgentOutputItem,
    AIErrorCode,
    AIErrorInfo,
    AIInvocationError,
    AIModelRequest,
)
from .agent_transcript import (
    OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    OPENAI_RESPONSES_PROTOCOL,
    matching_provider_item,
    text_transport_assistant_output,
)
from .openai_http_transport import OpenAIHTTPStatusError


def _openai_responses_tool(tool: Any, *, freeform: bool) -> dict[str, Any]:
    if freeform:
        return {
            "type": "custom",
            "name": tool.name,
            "description": tool.description,
            "format": {"type": "text"},
        }
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _openai_responses_input(
    request: AIModelRequest,
    *,
    agent_transport: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if request.turn_text or request.input_images:
        content: list[dict[str, Any]] = []
        if request.turn_text:
            content.append({"type": "input_text", "text": request.turn_text})
        content.extend(
            {"type": "input_image", "image_url": image_url} for image_url in request.input_images
        )
        items.append({"role": "user", "content": content})
    for turn in request.agent_history:
        items.extend(_openai_responses_history_items(turn, agent_transport=agent_transport))
    return items


def _openai_responses_history_items(
    turn: Any,
    *,
    agent_transport: str,
) -> list[dict[str, Any]]:
    if turn.transport_mode == "state_snapshot":
        return [_responses_snapshot_item(turn.result_text)]
    if agent_transport == "text_envelope" or not turn.tool_results:
        return _responses_text_history(turn)
    items = [
        item
        for output_item in turn.output_items
        if (item := _responses_history_output_item(output_item, agent_transport)) is not None
    ]
    calls_by_id = {item.call_id: item for item in turn.output_items if item.kind == "tool_call"}
    items.extend(
        _responses_tool_result_item(result, calls_by_id, agent_transport)
        for result in turn.tool_results
    )
    return items


def _responses_snapshot_item(result_text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": f"<较早行动压缩状态>\n{result_text}\n</较早行动压缩状态>",
            }
        ],
    }


def _responses_text_history(turn: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    assistant_text = text_transport_assistant_output(turn)
    if assistant_text:
        items.append(
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": assistant_text}],
            }
        )
    if turn.result_text:
        items.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"<行动结果>\n{turn.result_text}\n</行动结果>",
                    }
                ],
            }
        )
    return items


def _responses_history_output_item(
    item: Any,
    agent_transport: str,
) -> dict[str, Any] | None:
    provider_item = matching_provider_item(item, OPENAI_RESPONSES_PROTOCOL)
    if provider_item is not None:
        return provider_item
    if item.kind == "text" and item.text:
        return {
            "role": "assistant",
            "content": [{"type": "output_text", "text": item.text}],
        }
    if item.kind == "tool_call":
        return _reconstructed_responses_tool_call(item, agent_transport)
    return None


def _responses_tool_result_item(
    result: Any,
    calls_by_id: Mapping[str, Any],
    agent_transport: str,
) -> dict[str, Any]:
    call = calls_by_id.get(result.call_id)
    call_type = str((call.provider_item if call else {}).get("type") or "")
    output_type = (
        "custom_tool_call_output"
        if call_type == "custom_tool_call" or agent_transport == "native_freeform"
        else "function_call_output"
    )
    return {"type": output_type, "call_id": result.call_id, "output": result.text}


def _reconstructed_responses_tool_call(item: Any, agent_transport: str) -> dict[str, Any]:
    if agent_transport == "native_freeform":
        return {
            "type": "custom_tool_call",
            "call_id": item.call_id,
            "name": item.name,
            "input": item.text,
        }
    return {
        "type": "function_call",
        "call_id": item.call_id,
        "name": item.name,
        "arguments": item.raw_arguments or json.dumps({"text": item.text}, ensure_ascii=False),
    }


def _openai_responses_output_items(output: list[Any]) -> tuple[AIAgentOutputItem, ...]:
    return tuple(
        _responses_output_item(raw_item) for raw_item in output if isinstance(raw_item, Mapping)
    )


def _responses_output_item(raw_item: Mapping[str, Any]) -> AIAgentOutputItem:
    provider_item = dict(raw_item)
    item_type = str(raw_item.get("type") or "")
    if item_type == "message":
        return AIAgentOutputItem(
            "text",
            text=_responses_message_text(raw_item.get("content")),
            provider_item=provider_item,
            provider_protocol=OPENAI_RESPONSES_PROTOCOL,
        )
    if item_type == "custom_tool_call":
        return _custom_tool_call_item(raw_item, provider_item)
    if item_type == "function_call":
        return _function_tool_call_item(raw_item, provider_item)
    return AIAgentOutputItem(
        "provider_item",
        provider_item=provider_item,
        provider_protocol=OPENAI_RESPONSES_PROTOCOL,
    )


def _responses_message_text(content: Any) -> str:
    return "".join(
        str(part.get("text") or "")
        for part in (content if isinstance(content, list) else ())
        if isinstance(part, Mapping) and part.get("type") == "output_text"
    )


def _custom_tool_call_item(
    raw_item: Mapping[str, Any],
    provider_item: Mapping[str, Any],
) -> AIAgentOutputItem:
    call_id = str(raw_item.get("call_id") or "")
    text = str(raw_item.get("input") or "")
    return AIAgentOutputItem(
        "tool_call",
        text=text,
        name=str(raw_item.get("name") or ""),
        call_id=call_id,
        raw_arguments=text,
        argument_error="" if call_id else "工具调用缺少 call_id",
        provider_item=provider_item,
        provider_protocol=OPENAI_RESPONSES_PROTOCOL,
    )


def _function_tool_call_item(
    raw_item: Mapping[str, Any],
    provider_item: Mapping[str, Any],
) -> AIAgentOutputItem:
    raw_arguments = str(raw_item.get("arguments") or "")
    text_value, argument_error = _single_text_arguments(raw_arguments)
    call_id = str(raw_item.get("call_id") or "")
    if not call_id:
        argument_error = argument_error or "工具调用缺少 call_id"
    return AIAgentOutputItem(
        "tool_call",
        text=text_value,
        name=str(raw_item.get("name") or ""),
        call_id=call_id,
        raw_arguments=raw_arguments,
        argument_error=argument_error,
        provider_item=provider_item,
        provider_protocol=OPENAI_RESPONSES_PROTOCOL,
    )


def _single_text_arguments(raw_arguments: str) -> tuple[str, str]:
    try:
        arguments = json.loads(raw_arguments)
        if not isinstance(arguments, Mapping):
            raise ValueError("工具参数必须是只含 text 的对象")
        if set(arguments) != {"text"} or not isinstance(arguments.get("text"), str):
            raise ValueError("工具参数必须且只能包含字符串 text")
        return str(arguments["text"]), ""
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return "", str(exc)


def _openai_agent_history_messages(turn: Any, *, agent_transport: str) -> list[dict[str, Any]]:
    if turn.transport_mode == "state_snapshot":
        return [
            {
                "role": "user",
                "content": f"<较早行动压缩状态>\n{turn.result_text}\n</较早行动压缩状态>",
            }
        ]
    if agent_transport == "text_envelope" or not turn.tool_results:
        return _chat_text_history(turn)
    text = "".join(item.text for item in turn.output_items if item.kind == "text")
    calls = [item for item in turn.output_items if item.kind == "tool_call"]
    messages = [_chat_assistant_tool_message(text, calls)]
    messages.extend(
        {"role": "tool", "tool_call_id": result.call_id, "content": result.text}
        for result in turn.tool_results
    )
    return messages


def _chat_text_history(turn: Any) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    assistant_text = text_transport_assistant_output(turn)
    if assistant_text:
        messages.append({"role": "assistant", "content": assistant_text})
    if turn.result_text:
        messages.append({"role": "user", "content": f"<行动结果>\n{turn.result_text}\n</行动结果>"})
    return messages


def _chat_assistant_tool_message(text: str, calls: list[Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": text or None,
        "tool_calls": [_chat_history_tool_call(item) for item in calls],
    }


def _chat_history_tool_call(item: Any) -> dict[str, Any]:
    provider_item = matching_provider_item(item, OPENAI_CHAT_COMPLETIONS_PROTOCOL)
    if provider_item is not None:
        return provider_item
    return {
        "id": item.call_id,
        "type": "function",
        "function": {
            "name": item.name,
            "arguments": item.raw_arguments or json.dumps({"text": item.text}, ensure_ascii=False),
        },
    }


def _openai_agent_output_items(message: Mapping[str, Any]) -> tuple[AIAgentOutputItem, ...]:
    items: list[AIAgentOutputItem] = []
    content = _completion_text(message.get("content"))
    if content:
        items.append(
            AIAgentOutputItem(
                "text",
                text=content,
                provider_protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
            )
        )
    raw_calls = message.get("tool_calls")
    items.extend(
        item
        for raw_call in (raw_calls if isinstance(raw_calls, list) else ())
        if isinstance(raw_call, Mapping)
        if (item := _chat_tool_call_item(raw_call)) is not None
    )
    return tuple(items)


def _chat_tool_call_item(raw_call: Mapping[str, Any]) -> AIAgentOutputItem | None:
    function = raw_call.get("function")
    if not isinstance(function, Mapping):
        function = {}
    raw_arguments_value = function.get("arguments")
    raw_arguments = (
        raw_arguments_value
        if isinstance(raw_arguments_value, str)
        else json.dumps(raw_arguments_value, ensure_ascii=False, default=str)
    )
    text_value, argument_error = _single_text_arguments(raw_arguments)
    call_id = str(raw_call.get("id") or "")
    if not call_id:
        argument_error = argument_error or "工具调用缺少 call_id"
    return AIAgentOutputItem(
        "tool_call",
        text=text_value,
        name=str(function.get("name") or ""),
        call_id=call_id,
        raw_arguments=raw_arguments,
        argument_error=argument_error,
        provider_item=dict(raw_call),
        provider_protocol=OPENAI_CHAT_COMPLETIONS_PROTOCOL,
    )


def _is_tool_transport_unsupported(exc: OpenAIHTTPStatusError) -> bool:
    if exc.status_code not in {400, 404, 405, 422}:
        return False
    normalized = _provider_error_text(exc).casefold()
    tool_marker = any(
        marker in normalized
        for marker in (
            "tool_choice",
            "tool_calls",
            "tools",
            "function calling",
            "function_call",
            "custom tool",
            "custom_tool",
        )
    )
    unsupported_marker = any(
        marker in normalized
        for marker in (
            "unsupported",
            "not supported",
            "unknown field",
            "unrecognized field",
            "extra inputs are not permitted",
            "does not support",
            "invalid value",
            "invalid type",
            "expected one of",
            "不支持",
        )
    )
    return tool_marker and unsupported_marker


def _is_responses_endpoint_unsupported(exc: OpenAIHTTPStatusError) -> bool:
    if exc.status_code not in {400, 404, 405, 422}:
        return False
    normalized = _provider_error_text(exc).casefold()
    endpoint_marker = any(
        marker in normalized
        for marker in ("/responses", "responses endpoint", "responses api", "response api")
    )
    unsupported_marker = any(
        marker in normalized
        for marker in (
            "unsupported",
            "not supported",
            "not found",
            "unknown endpoint",
            "unknown route",
            "does not support",
            "不支持",
        )
    )
    return endpoint_marker and unsupported_marker


def _provider_error_text(exc: OpenAIHTTPStatusError) -> str:
    try:
        return json.dumps(exc.provider_response, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(exc.provider_response or "")


def _next_agent_transport(current: str, exc: OpenAIHTTPStatusError) -> str:
    unsupported = _is_tool_transport_unsupported(exc) or (
        current == "native_freeform" and _is_responses_endpoint_unsupported(exc)
    )
    if not unsupported:
        return ""
    if current == "native_freeform":
        return "native_text_field"
    if current == "native_text_field":
        return "text_envelope"
    return ""


def _first_choice(data: Mapping[str, Any], backend_id: str) -> Mapping[str, Any]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _empty_output("Backend returned no completion choices", backend_id)
    return choices[0] if isinstance(choices[0], Mapping) else {}


def _completion_text(content: Any) -> str:
    if not isinstance(content, list):
        return str(content or "")
    return "".join(
        str(part.get("text") or "")
        for part in content
        if isinstance(part, Mapping) and part.get("type") == "text"
    )


def _is_context_window_error(api_code: str, provider_response: Any) -> bool:
    code = str(api_code or "").lower()
    if code in {"context_length_exceeded", "context_window_exceeded"}:
        return True
    lowered = f"{code} {_context_error_text(provider_response)[:4000]}".lower()
    markers = (
        "context_length",
        "context window",
        "maximum context",
        "max context",
        "too many tokens",
    )
    return any(marker in lowered for marker in markers) or (
        "context" in lowered and ("token" in lowered or "length" in lowered)
    )


def _context_error_text(provider_response: Any) -> str:
    if not isinstance(provider_response, Mapping):
        return str(provider_response or "")
    error = provider_response.get("error")
    if not isinstance(error, Mapping):
        return str(provider_response)
    return " ".join(str(error.get(key) or "") for key in ("code", "type", "message"))


def _empty_output(message: str, backend_id: str) -> AIInvocationError:
    return AIInvocationError(
        AIErrorInfo(
            AIErrorCode.EMPTY_OUTPUT,
            message,
            retryable=True,
            switch_backend=True,
            backend_id=backend_id,
            phase="response",
        )
    )


__all__ = [
    "_completion_text",
    "_empty_output",
    "_first_choice",
    "_is_context_window_error",
    "_is_responses_endpoint_unsupported",
    "_is_tool_transport_unsupported",
    "_next_agent_transport",
    "_openai_agent_history_messages",
    "_openai_agent_output_items",
    "_openai_responses_input",
    "_openai_responses_output_items",
    "_openai_responses_tool",
    "_reconstructed_responses_tool_call",
    "_single_text_arguments",
]
