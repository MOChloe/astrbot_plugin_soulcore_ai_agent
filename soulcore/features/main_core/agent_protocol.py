"""MainCore's two all-string Agent channels and exact transcript helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...contracts.ai_models import (
    AIAgentOutputItem,
    AIAgentToolDefinition,
    AIAgentToolResult,
    AIAgentTranscriptTurn,
    AICompletion,
)

CONTINUE_TOOL = "soulcore_continue"
FINAL_TOOL = "soulcore_final"
CONTINUE_CHANNEL = "继续行动"
FINAL_CHANNEL = "最终表达"
MAX_AGENT_RESULT_CHARACTERS = 131_072


@dataclass(frozen=True, slots=True)
class ParsedAgentResponse:
    channel: str
    text: str
    raw_model_text: str
    outside_text: str
    output_items: tuple[AIAgentOutputItem, ...]
    transport_mode: str
    error: str = ""

    @property
    def valid(self) -> bool:
        return not self.error


def main_core_agent_tools() -> tuple[AIAgentToolDefinition, ...]:
    return (
        AIAgentToolDefinition(
            CONTINUE_TOOL,
            "完成当前一步并继续。text：仅含非终止中文 XML 指令的完整字符串。",
        ),
        AIAgentToolDefinition(
            FINAL_TOOL,
            "提交最终表达并结束。text：仅含终止表达中文 XML 指令的完整字符串。",
        ),
    )


def parse_agent_response(completion: AICompletion) -> ParsedAgentResponse:
    raw_text = str(completion.text or "")
    items = tuple(completion.agent_output_items) or (
        (AIAgentOutputItem("text", text=raw_text),) if raw_text else ()
    )
    calls = tuple(item for item in items if item.kind == "tool_call")
    if calls:
        return _parse_native_response(raw_text, items, calls, completion.agent_transport_mode)
    return _parse_text_envelope(raw_text, items, completion.agent_transport_mode or "text_envelope")


def _parse_native_response(
    raw_text: str,
    items: tuple[AIAgentOutputItem, ...],
    calls: tuple[AIAgentOutputItem, ...],
    transport_mode: str,
) -> ParsedAgentResponse:
    if _contains_channel_tag(raw_text):
        return _invalid(
            raw_text,
            items,
            transport_mode,
            "同一响应同时包含原生工具调用和纯文本通道外层。",
        )
    if len(calls) != 1:
        return _invalid(
            raw_text,
            items,
            transport_mode,
            f"同一响应调用了 {len(calls)} 个 Agent 工具；每次只能选择一个通道。",
        )
    call = calls[0]
    if call.argument_error:
        return _invalid(
            raw_text,
            items,
            transport_mode,
            f"{call.name or 'Agent 工具'} 的字符串参数无效：{call.argument_error}",
        )
    channel = {CONTINUE_TOOL: CONTINUE_CHANNEL, FINAL_TOOL: FINAL_CHANNEL}.get(call.name, "")
    if not channel:
        return _invalid(
            raw_text,
            items,
            transport_mode,
            f"未知 Agent 工具：{call.name or '（空名称）'}。",
        )
    return ParsedAgentResponse(
        channel=channel,
        text=call.text,
        raw_model_text=raw_text,
        outside_text=raw_text,
        output_items=items,
        transport_mode=transport_mode or "native_text_field",
    )


def _parse_text_envelope(
    raw_text: str,
    items: tuple[AIAgentOutputItem, ...],
    transport_mode: str,
) -> ParsedAgentResponse:
    occurrences = {
        CONTINUE_CHANNEL: _channel_occurrences(raw_text, CONTINUE_CHANNEL),
        FINAL_CHANNEL: _channel_occurrences(raw_text, FINAL_CHANNEL),
    }
    present = [name for name, value in occurrences.items() if value != (0, 0)]
    if not present:
        return _invalid(
            raw_text,
            items,
            transport_mode,
            "没有选择 Agent 通道；裸 XML 和裸文字都不会执行。",
        )
    if len(present) != 1:
        return _invalid(
            raw_text,
            items,
            transport_mode,
            "同一响应包含多个外层通道；每次只能选择“继续行动”或“最终表达”其中一个。",
        )
    channel = present[0]
    opening_count, closing_count = occurrences[channel]
    if opening_count != 1 or closing_count != 1:
        return _invalid(
            raw_text,
            items,
            transport_mode,
            f"“{channel}”外层必须且只能各有一个开始标签和结束标签。",
        )
    opening = f"<{channel}>"
    closing = f"</{channel}>"
    start = raw_text.find(opening)
    end = raw_text.find(closing)
    if end < start:
        return _invalid(
            raw_text,
            items,
            transport_mode,
            f"“{channel}”结束标签出现在开始标签之前。",
        )
    payload_start = start + len(opening)
    payload = raw_text[payload_start:end]
    if _contains_channel_tag(payload):
        return _invalid(raw_text, items, transport_mode, "Agent 外层通道不能嵌套。")
    return ParsedAgentResponse(
        channel=channel,
        text=payload,
        raw_model_text=raw_text,
        outside_text=raw_text[:start] + raw_text[end + len(closing) :],
        output_items=items,
        transport_mode="text_envelope",
    )


def _channel_occurrences(text: str, channel: str) -> tuple[int, int]:
    return text.count(f"<{channel}>"), text.count(f"</{channel}>")


def _contains_channel_tag(text: str) -> bool:
    return any(
        token in text
        for channel in (CONTINUE_CHANNEL, FINAL_CHANNEL)
        for token in (f"<{channel}>", f"</{channel}>")
    )


def _invalid(
    raw_text: str,
    items: tuple[AIAgentOutputItem, ...],
    transport_mode: str,
    reason: str,
) -> ParsedAgentResponse:
    return ParsedAgentResponse(
        channel="",
        text="",
        raw_model_text=raw_text,
        outside_text=raw_text,
        output_items=items,
        transport_mode=transport_mode or "text_envelope",
        error=protocol_error(reason),
    )


def protocol_error(reason: str) -> str:
    return (
        "协议错误\n\n"
        f"{str(reason or '').strip()}\n"
        "本轮没有执行任何指令，也没有发送任何内容。\n"
        "请根据刚才的完整原始输出重新选择一个通道。"
    )


def command_results_text(results: Any) -> str:
    sections: list[str] = []
    for result in results:
        status = "成功" if bool(result.ok) else "失败"
        heading = f"{result.command_name}：{status}"
        content = str(result.content or "")
        sections.append(f"{heading}\n{content}".rstrip())
    text = "\n\n".join(sections).strip()
    if len(text) <= MAX_AGENT_RESULT_CHARACTERS:
        return text
    return f"行动结果：失败\n完整结果超过 {MAX_AGENT_RESULT_CHARACTERS} 个字符，未返回材料。"


def transcript_turn(
    response: ParsedAgentResponse,
    result_text: str,
    *,
    source_round_number: int = 0,
    contains_plan: bool = False,
    unresolved_failure: bool = False,
    public_references: tuple[str, ...] = (),
    contains_image_material: bool = False,
) -> AIAgentTranscriptTurn:
    calls = tuple(item for item in response.output_items if item.kind == "tool_call")
    tool_results = tuple(AIAgentToolResult(item.name, item.call_id, result_text) for item in calls)
    return AIAgentTranscriptTurn(
        output_items=response.output_items,
        result_text=result_text,
        tool_results=tool_results,
        transport_mode=response.transport_mode,
        source_round_number=max(0, int(source_round_number)),
        contains_plan=bool(contains_plan),
        unresolved_failure=bool(unresolved_failure),
        public_references=tuple(public_references),
        contains_image_material=bool(contains_image_material),
    )


__all__ = [
    "CONTINUE_CHANNEL",
    "CONTINUE_TOOL",
    "FINAL_CHANNEL",
    "FINAL_TOOL",
    "ParsedAgentResponse",
    "command_results_text",
    "main_core_agent_tools",
    "parse_agent_response",
    "protocol_error",
    "transcript_turn",
]
