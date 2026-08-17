"""Human-readable parsing and execution diagnostics for one Main Core round."""

from __future__ import annotations

from typing import Any

from ..ai.service import MainCoreCommandRegistry, parse_model_turn
from .roleplay_prompt_contracts import ExecutionRound


def _normalizations(round_item: ExecutionRound, parsed: Any) -> list[dict[str, Any]]:
    del round_item, parsed
    return []


def _cleaned_fields(parsed: Any, registry: MainCoreCommandRegistry) -> list[dict[str, str]]:
    return [
        {"command": item.name, "parameter": label}
        for item in parsed.commands
        if (spec := registry.get(item.name)) is not None and not spec.terminal
        for label in item.parameters
        if label not in {parameter.label for parameter in spec.parameters}
    ]


def _parsed_command_views(
    parsed: Any,
    registry: MainCoreCommandRegistry,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in parsed.commands:
        parameters = dict(item.parameters)
        spec = registry.get(item.name)
        if spec is not None and spec.body_parameter:
            parameters[spec.body_parameter] = str(item.unlabeled_content or "").strip()
        result.append(
            {
                "ordinal": item.ordinal,
                "name": item.name,
                "parameters": parameters,
            }
        )
    return result


def _result_views(round_item: ExecutionRound) -> list[dict[str, Any]]:
    return [
        {
            "ordinal": item.ordinal,
            "command": item.command_name,
            "ok": item.ok,
            "summary": item.content,
        }
        for item in round_item.results
    ]


def processing_view(
    round_item: ExecutionRound | None,
    *,
    registry: MainCoreCommandRegistry,
    completed: bool,
) -> dict[str, Any]:
    if round_item is None:
        return {
            "captured": False,
            "summary": "本次记录未采集解析与执行结果。",
            "terminal": bool(completed),
            "accepted": False,
            "validation_status": "MISSING",
        }
    parsed = parse_model_turn(round_item.payload_text or round_item.raw_text)
    normalizations = _normalizations(round_item, parsed)
    cleaned_fields = _cleaned_fields(parsed, registry)
    return {
        "captured": True,
        "model_visible_text": round_item.raw_text,
        "model_visible_items": [
            {
                "kind": item.kind,
                "text": item.text,
                "name": item.name,
                "call_id": item.call_id,
                "raw_arguments": item.raw_arguments,
                "argument_error": item.argument_error,
            }
            for item in round_item.output_items
        ],
        "channel": round_item.channel,
        "payload_text": round_item.payload_text,
        "commands": list(round_item.calls),
        "parsed_commands": _parsed_command_views(parsed, registry),
        "results": _result_views(round_item),
        "rejection": round_item.rejection,
        "execution_result": round_item.result_text,
        "normalizations": normalizations,
        "cleaned_fields": cleaned_fields,
        "dropped_command_count": max(0, len(parsed.commands) - len(round_item.calls)),
        "terminal": bool(completed),
        "accepted": not bool(round_item.rejection),
        "validation_status": "ACCEPTED" if not round_item.rejection else "REJECTED",
        "terminal_rejection": False,
    }


__all__ = ["processing_view"]
