from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ...shared.model_image_preview import (
    MAX_MODEL_IMAGE_PREVIEW_BYTES,
    MAX_MODEL_IMAGE_PREVIEW_TOTAL_BYTES,
    bounded_model_image_preview,
    encode_model_image_data_uri,
)
from ...shared.time_display import model_datetime
from .command_protocol_catalog import _TERMINAL_SPECS
from .command_protocol_types import (
    CommandExecutionResult,
    CommandParameter,
    CommandProtocolError,
    CommandSetLike,
    CommandSpec,
    ModelVisibleCommandResult,
    ParsedCommand,
    ParsedModelTurn,
    ValidatedCommand,
)

MAX_MODEL_TURN_CHARACTERS = 65_536
MAX_MODEL_TURN_LINES = 2_048
MAX_COMMANDS_PER_TURN = 64
MAX_PARAMETERS_PER_COMMAND = 64

_OPEN_TAG = re.compile(r"^<(?P<name>[^<>/\s]+)>$")
_CLOSE_TAG = re.compile(r"^</(?P<name>[^<>/\s]+)>$")
_PARAMETER = re.compile(r"^\[\[(?P<label>[^\[\]\r\n]+)\]\][：:](?P<value>.*)$")


@dataclass(slots=True)
class _CommandState:
    name: str
    start_line: int
    raw_lines: list[str] = field(default_factory=list)
    parameters: dict[str, list[str]] = field(default_factory=dict)
    unlabeled: list[str] = field(default_factory=list)
    active_parameter: str = ""


@dataclass(slots=True)
class _ParserState:
    commands: list[ParsedCommand] = field(default_factory=list)
    working: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    current: _CommandState | None = None


def parse_model_turn(text: str) -> ParsedModelTurn:
    """Parse the literal MainCore XML command string without repairing it."""

    raw = str(text or "")
    if len(raw) > MAX_MODEL_TURN_CHARACTERS:
        return _limit_failure(raw, f"模型输出超过 {MAX_MODEL_TURN_CHARACTERS} 个字符")
    lines = raw.splitlines()
    if len(lines) > MAX_MODEL_TURN_LINES:
        return _limit_failure(raw, f"模型输出超过 {MAX_MODEL_TURN_LINES} 行")

    state = _ParserState()
    for line_no, line in enumerate(lines, start=1):
        _consume_line(state, line_no, line)

    if state.current is not None:
        state.errors.append(
            f"第 {state.current.start_line} 行开始的 <{state.current.name}> 没有结束标签"
        )

    return ParsedModelTurn(
        working_text="\n".join(state.working).strip(),
        commands=tuple(state.commands),
        errors=tuple(state.errors),
        raw_text=raw,
    )


def _consume_line(state: _ParserState, line_no: int, line: str) -> None:
    if state.current is None:
        _consume_outside_command(state, line_no, line)
    else:
        _consume_inside_command(state, line_no, line)


def _consume_outside_command(state: _ParserState, line_no: int, line: str) -> None:
    stripped = line.strip()
    opening = _OPEN_TAG.fullmatch(stripped)
    if opening is not None:
        state.current = _CommandState(
            name=opening.group("name"),
            start_line=line_no,
            raw_lines=[line],
        )
        return
    if _CLOSE_TAG.fullmatch(stripped) is not None:
        state.errors.append(f"第 {line_no} 行出现了没有对应开始标签的结束标签")
    elif stripped and ("<" in stripped or ">" in stripped):
        state.errors.append(f"第 {line_no} 行不是独占一行的有效开始标签")
    state.working.append(line)


def _consume_inside_command(state: _ParserState, line_no: int, line: str) -> None:
    current = state.current
    if current is None:
        raise RuntimeError("parser command state disappeared")
    stripped = line.strip()
    current.raw_lines.append(line)
    closing = _CLOSE_TAG.fullmatch(stripped)
    if closing is not None:
        _finish_or_reject_closing(state, line_no, closing.group("name"))
        return
    if _OPEN_TAG.fullmatch(stripped) is not None:
        state.errors.append(f"第 {line_no} 行在 <{current.name}> 尚未结束时开始了另一条指令")
        return
    if stripped.startswith("</") or (stripped.endswith(">") and "<" in stripped):
        state.errors.append(f"第 {line_no} 行含有无效标签")
        return
    parameter = _PARAMETER.fullmatch(line.lstrip())
    if parameter is not None:
        _start_parameter(state, line_no, parameter.group("label"), parameter.group("value"))
    elif current.active_parameter:
        current.parameters[current.active_parameter].append(line)
    elif stripped:
        current.unlabeled.append(line)


def _finish_or_reject_closing(state: _ParserState, line_no: int, closing_name: str) -> None:
    current = state.current
    if current is None:
        raise RuntimeError("parser command state disappeared")
    if closing_name != current.name:
        state.errors.append(
            f"第 {line_no} 行使用 </{closing_name}> 结束 <{current.name}>；标签不匹配"
        )
        return
    if len(state.commands) >= MAX_COMMANDS_PER_TURN:
        state.errors.append(f"每轮最多允许 {MAX_COMMANDS_PER_TURN} 条指令")
    else:
        state.commands.append(
            ParsedCommand(
                name=current.name,
                parameters={
                    label: "\n".join(value_lines).strip()
                    for label, value_lines in current.parameters.items()
                },
                ordinal=len(state.commands) + 1,
                raw_text="\n".join(current.raw_lines),
                unlabeled_content="\n".join(current.unlabeled).strip(),
            )
        )
    state.current = None


def _start_parameter(state: _ParserState, line_no: int, raw_label: str, raw_value: str) -> None:
    current = state.current
    if current is None:
        raise RuntimeError("parser command state disappeared")
    label = raw_label.strip()
    if not label:
        state.errors.append(f"第 {line_no} 行的参数名为空")
        current.active_parameter = ""
        return
    if label in current.parameters:
        state.errors.append(f"第 {line_no} 行重复参数 [[{label}]]")
        current.active_parameter = ""
        return
    if len(current.parameters) >= MAX_PARAMETERS_PER_COMMAND:
        state.errors.append(
            f"第 {line_no} 行超过每条指令最多 {MAX_PARAMETERS_PER_COMMAND} 个参数的限制"
        )
        current.active_parameter = ""
        return
    current.parameters[label] = [raw_value.lstrip()]
    current.active_parameter = label


def _limit_failure(raw: str, reason: str) -> ParsedModelTurn:
    return ParsedModelTurn(
        working_text="",
        commands=(),
        errors=(f"{reason}，未解析任何指令",),
        raw_text=raw,
    )


_RESULT_TIME_FIELDS = frozenset(
    {
        "due_at",
        "start_at",
        "end_at",
        "published_at",
        "period_start",
        "period_end",
    }
)
_MODEL_RESULT_IMAGE_MIME_TYPES = frozenset({"image/gif", "image/jpeg", "image/png", "image/webp"})
_MAX_MODEL_RESULT_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_MODEL_RESULT_IMAGES = 5


def normalize_command_result(
    value: Any,
) -> tuple[str, tuple[str, ...], tuple[tuple[str, str], ...], bool, tuple[str, ...]]:
    if isinstance(value, ModelVisibleCommandResult):
        images = _model_input_images({"content_parts": value.content_parts})
        return (
            value.content.strip() or "行动已完成。",
            tuple(value.media_asset_ids),
            tuple(value.reference_hints),
            False,
            images,
        )
    if isinstance(value, Mapping):
        is_error = bool(
            value.get("is_error", False)
            or value.get("ok") is False
            or value.get("error") not in (None, "", False)
        )
        if not is_error:
            raise ValueError("successful command result requires ModelVisibleCommandResult")
        raw = (
            value.get("content")
            or value.get("message")
            or "行动没有完成；请依据现有结果调整下一步。"
        )
        assets = tuple(str(item) for item in value.get("media_asset_ids", ()) if str(item))
        content = _readable_result(raw)
        references = _reference_hints(value, excluded=assets)
        model_input_images = _model_input_images(value)
        return (
            content or ("行动失败。" if is_error else "行动已完成。"),
            assets,
            references,
            is_error,
            model_input_images,
        )
    if isinstance(value, str) and value.strip().lower().startswith(("error:", "error：")):
        return value.strip(), (), (), True, ()
    raise ValueError("successful command result requires ModelVisibleCommandResult")


def _model_input_images(value: Mapping[Any, Any]) -> tuple[str, ...]:
    parts = value.get("content_parts", ())
    if parts in (None, (), []):
        return ()
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes, bytearray)):
        raise ValueError("multimodal command content_parts must be a sequence")
    candidates: list[tuple[str, bytes]] = []
    for part in parts:
        candidate = _model_image_candidate(part)
        if candidate is None:
            continue
        candidates.append(candidate)
        if len(candidates) >= _MAX_MODEL_RESULT_IMAGES:
            break
    remaining = MAX_MODEL_IMAGE_PREVIEW_TOTAL_BYTES
    images: list[str] = []
    for index, (mime_type, raw) in enumerate(candidates):
        remaining_count = len(candidates) - index
        allocation = min(
            MAX_MODEL_IMAGE_PREVIEW_BYTES,
            max(1, remaining // max(1, remaining_count)),
        )
        preview = bounded_model_image_preview(
            raw,
            mime_type,
            maximum_bytes=allocation,
        )
        images.append(encode_model_image_data_uri(preview.data, preview.mime_type))
        remaining -= len(preview.data)
    return tuple(images)


def _model_image_candidate(part: Any) -> tuple[str, bytes] | None:
    if not isinstance(part, Mapping):
        raise ValueError("multimodal command content part must be an object")
    part_type = str(part.get("type") or "").strip().lower()
    if part_type == "text":
        return None
    if part_type != "image":
        raise ValueError("unsupported multimodal command content part")
    mime_type = str(part.get("mime_type") or "").strip().lower()
    if mime_type not in _MODEL_RESULT_IMAGE_MIME_TYPES:
        raise ValueError("unsupported multimodal command image type")
    data = part.get("data")
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ValueError("multimodal command image bytes are missing")
    raw = bytes(data)
    if len(raw) > _MAX_MODEL_RESULT_IMAGE_BYTES:
        raise ValueError("multimodal command image exceeds 20 MiB")
    return mime_type, raw


def _reference_hints(value: Any, *, excluded: Sequence[str] = ()) -> tuple[tuple[str, str], ...]:
    if isinstance(value, str):
        return ()
    if isinstance(value, Mapping):
        hints: list[tuple[str, str]] = []
        for key, item in value.items():
            prefix = _reference_prefix(str(key))
            if prefix:
                values = item if isinstance(item, (list, tuple, set)) else (item,)
                hints.extend(
                    (prefix, str(raw)) for raw in values if _usable_reference(raw, excluded)
                )
            hints.extend(_reference_hints(item, excluded=excluded))
        return tuple(dict.fromkeys(hints))
    if isinstance(value, (list, tuple, set)):
        return tuple(
            dict.fromkeys(
                hint for item in value for hint in _reference_hints(item, excluded=excluded)
            )
        )
    return ()


def register_result_references(
    results: Sequence[CommandExecutionResult], reference_map: dict[str, Any]
) -> tuple[CommandExecutionResult, ...]:
    """Replace executor-owned identifiers with run-scoped public references."""
    internal_to_public = {
        str(internal): str(public)
        for public, internal in reference_map.items()
        if str(internal).strip()
    }
    normalized: list[CommandExecutionResult] = []
    for result in results:
        content = result.content
        public_refs: list[str] = []
        candidates = [("I", asset_id) for asset_id in result.media_asset_ids]
        candidates.extend(result.references)
        for prefix, internal_value in candidates:
            internal = str(internal_value or "").strip()
            if not internal:
                continue
            public = internal_to_public.get(internal)
            if public is None:
                public = _next_public_reference(reference_map, str(prefix or "R"))
                reference_map[public] = internal_value
                internal_to_public[internal] = public
            content = content.replace(internal, public)
            public_refs.append(public)
        visible_refs = tuple(dict.fromkeys(public_refs))
        if visible_refs and not all(item in content for item in visible_refs):
            content = f"{content}\n可用短引用：{'、'.join(visible_refs)}"
        normalized.append(
            CommandExecutionResult(
                result.ordinal,
                result.command_name,
                result.ok,
                content,
                result.media_asset_ids,
                result.references,
                result.diagnostic,
                visible_refs,
                result.model_input_images,
            )
        )
    return tuple(normalized)


def _next_public_reference(reference_map: Mapping[str, Any], prefix: str) -> str:
    index = 1
    while f"{prefix}{index}" in reference_map:
        index += 1
    return f"{prefix}{index}"


def _reference_prefix(key: str) -> str:
    return {
        "sticker_ref": "S",
        "sticker_refs": "S",
        "sticker_import_source_ref": "I",
        "sticker_import_source_refs": "I",
        "file_ref": "F",
        "file_refs": "F",
        "knowledge_fact_id": "KF",
        "knowledge_fact_ids": "KF",
        "memory_id": "M",
        "memory_ids": "M",
        "image_resource_id": "I",
        "image_resource_ids": "I",
        "image_asset_id": "I",
        "image_asset_ids": "I",
        "asset_ref": "I",
        "asset_ref_id": "I",
        "source_ref": "I",
        "resource_id": "R",
        "resource_ids": "R",
        "schedule_ref": "SC",
        "timer_ref": "TM",
        "timer_refs": "TM",
        "message_ref": "U",
        "message_refs": "U",
        "member_ref": "P",
        "member_refs": "P",
        "profile_entry_ref": "E",
        "profile_entry_refs": "E",
        "history_cursor_ref": "HC",
        "history_cursor_refs": "HC",
        "lore_id": "W",
        "lore_ids": "W",
    }.get(key.casefold(), "")


def _usable_reference(value: Any, excluded: Sequence[str]) -> bool:
    text = str(value or "").strip()
    return bool(text and text not in excluded and len(text) <= 512)


def _readable_result(value: Any, *, field_name: str = "") -> str:
    """Project only registered semantic fields into the next model turn."""
    if isinstance(value, datetime) or field_name in _RESULT_TIME_FIELDS:
        return model_datetime(value)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return ""
    if isinstance(value, (list, tuple, set)):
        return _readable_sequence(value)
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value or "").strip()


def _readable_sequence(value: Sequence[Any] | set[Any]) -> str:
    rendered_items = [_readable_result(item) for item in value]
    return "\n".join(f"- {item}" for item in rendered_items if item)


def command_template(
    spec: CommandSpec,
    *,
    fixed_parameters: Mapping[str, str] | None = None,
    include_execution: bool = True,
    omitted_parameters: Sequence[str] = (),
) -> list[str]:
    lines = [
        "",
        f"{spec.name}｜{_compact_description(spec.description)}",
    ]
    guidance = str(spec.usage_guidance or "").strip()
    if guidance:
        lines.append(guidance)
    if include_execution:
        lines.append(f"执行：{_command_execution_label(spec)}")
    tag_line = f"标签：<{spec.name}> 与 </{spec.name}>"
    fixed = dict(fixed_parameters or {})
    omitted = {str(label or "").strip() for label in omitted_parameters}
    parameters = tuple(parameter for parameter in spec.parameters if parameter.label not in omitted)
    if parameters:
        if include_execution:
            lines.append(tag_line)
        lines.extend(
            _parameter_template(
                parameter,
                fixed_value=fixed.get(parameter.label),
                body_parameter=parameter.label == spec.body_parameter,
            )
            for parameter in parameters
        )
    else:
        lines.append(f"{tag_line}，中间不写内容。" if include_execution else "无字段。")
    return lines


def command_execution_group_lines(specs: Sequence[CommandSpec]) -> list[str]:
    groups = (
        "继续行动｜可并行",
        "继续行动｜顺序",
        "最终表达｜按序提交",
        "最终决定",
        "最终决定｜单独提交",
    )
    names_by_group = {
        group: [spec.name for spec in specs if _command_execution_label(spec) == group]
        for group in groups
    }
    return [
        f"{group}：{'、'.join(names_by_group[group])}" for group in groups if names_by_group[group]
    ]


def _compact_description(value: str) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "执行这一行动"
    sentence = text.split(".", 1)[0].strip()
    return sentence[:160]


def _command_execution_label(spec: CommandSpec) -> str:
    if spec.send_kind in {"SILENT", "ABSENCE"}:
        return "最终决定"
    if spec.terminal and not spec.send_kind:
        return "最终决定｜单独提交"
    if spec.terminal:
        return "最终表达｜按序提交"
    if spec.serial:
        return "继续行动｜顺序"
    return "继续行动｜可并行"


def _parameter_template(
    parameter: CommandParameter,
    *,
    fixed_value: str | None = None,
    body_parameter: bool = False,
) -> str:
    requirement = "必填" if parameter.required else "可选"
    hints: list[str] = []
    if fixed_value is not None:
        hints.append(f"固定填写“{fixed_value}”")
    choices = parameter.choices
    if choices and fixed_value is None:
        hints.append(f"只可填写：{' / '.join(choices)}")
    value_hint = str(parameter.prompt_hint or "").strip()
    if value_hint:
        hints.append(value_hint)
    description = "；".join(hints) or "填写该字段的实际内容"
    label = "标签正文" if body_parameter else f"[[{parameter.label}]]"
    return f"{label}（{requirement}）：{description}"


class MainCoreCommandRegistry:
    def __init__(self, specs: Sequence[CommandSpec]) -> None:
        self.specs = tuple(specs)
        self._by_name = {spec.name: spec for spec in self.specs}

    @classmethod
    def from_command_set(
        cls, command_set: CommandSetLike, *, include_terminal: bool = True
    ) -> MainCoreCommandRegistry:
        specs = list(command_set.commands)
        disabled = {
            str(item or "").strip().upper()
            for item in getattr(command_set, "disabled_terminal_send_kinds", ())
        }
        terminal_specs = (
            tuple(spec for spec in _TERMINAL_SPECS if spec.send_kind not in disabled)
            if include_terminal
            else ()
        )
        return cls([*specs, *terminal_specs])

    @classmethod
    def terminal_only(cls) -> MainCoreCommandRegistry:
        return cls(_TERMINAL_SPECS)

    def get(self, name: str) -> CommandSpec | None:
        return self._by_name.get(str(name or "").strip())

    def identity_modes(self, name: str) -> dict[str, str]:
        spec = self.get(name)
        if spec is None:
            return {}
        return {item.label: item.identity_mode for item in spec.parameters}

    def details_text(self) -> str:
        lines = [
            (
                "每条指令用独占一行的 <指令名> 与 </指令名> 包裹；字段写成 "
                "[[字段名]]：内容。标为标签正文的内容直接写在标签内；"
                "内容可跨行，必填内容不能省略。"
            ),
            (
                "[[留话]]仅保存已确定且无法从已发送内容还原的连续性信息，例如未公开谜底或后续目标；"
                "不写草稿、推理、回复理由或猜测。"
            ),
            "短引用从当前内容或行动结果中复制；多个引用用逗号分隔。",
        ]
        for spec in self.specs:
            if not spec.prompt_visible:
                continue
            lines.extend(
                command_template(
                    spec,
                    include_execution=False,
                )
            )
        return "\n".join(lines)

    def list_text(self, *, include_hidden: Sequence[str] = ()) -> str:
        included = {str(item or "").strip() for item in include_hidden}
        visible = tuple(spec for spec in self.specs if spec.prompt_visible or spec.name in included)
        return "\n".join(command_execution_group_lines(visible))

    def command_template_text(
        self,
        name: str,
        *,
        fixed_parameters: Mapping[str, str] | None = None,
    ) -> str:
        spec = self.get(name)
        if spec is None:
            return ""
        return "\n".join(command_template(spec, fixed_parameters=fixed_parameters)).strip()

    def validate(
        self, command: ParsedCommand, reference_map: Mapping[str, Any]
    ) -> ValidatedCommand:
        spec = self.get(command.name)
        if spec is None:
            raise CommandProtocolError(f"未知指令：{command.name}")
        parameters = {item.label: item for item in spec.parameters}
        arguments = _body_arguments(command, spec, parameters)
        for label, raw in command.parameters.items():
            parameter = parameters.get(label)
            if parameter is None:
                raise CommandProtocolError(f"{command.name} 不支持参数 [[{label}]]")
            arguments[parameter.internal_name] = _validated_parameter_value(
                parameter,
                raw,
                reference_map,
            )
        _require_parameters(command.name, spec.parameters, arguments)
        return ValidatedCommand(command, spec, arguments)


def _body_arguments(
    command: ParsedCommand,
    spec: CommandSpec,
    parameters: Mapping[str, CommandParameter],
) -> dict[str, str]:
    body_parameter = str(spec.body_parameter or "").strip()
    body_content = str(command.unlabeled_content or "").strip()
    if body_content and not body_parameter:
        raise CommandProtocolError(
            f"{command.name} 含有未标注字段的内容；每个参数都必须使用已声明的 [[参数名]]"
        )
    if not body_parameter:
        return {}
    parameter = parameters.get(body_parameter)
    if parameter is None:
        raise RuntimeError(f"{spec.name} 的标签正文参数未在命令字段中声明：{body_parameter}")
    if body_parameter in command.parameters:
        raise CommandProtocolError(f"{command.name} 的 {body_parameter} 应直接写在标签正文中")
    return {parameter.internal_name: body_content}


def _validated_parameter_value(
    parameter: CommandParameter,
    raw: str,
    reference_map: Mapping[str, Any],
) -> str:
    value = str(raw or "").strip()
    if parameter.choices and value not in parameter.choices:
        raise CommandProtocolError(f"[[{parameter.label}]] 只能是：{'、'.join(parameter.choices)}")
    if parameter.validator is None:
        return value
    validation_error = str(parameter.validator(value, reference_map) or "").strip()
    if validation_error:
        raise CommandProtocolError(validation_error)
    return value


def _require_parameters(
    command_name: str,
    parameters: Sequence[CommandParameter],
    arguments: Mapping[str, str],
) -> None:
    missing = [item for item in parameters if item.required and item.internal_name not in arguments]
    if missing:
        raise CommandProtocolError(
            f"{command_name} 缺少必填参数：{'、'.join(item.label for item in missing)}"
        )


async def execute_nonterminal_batch(
    commands: Sequence[ValidatedCommand],
    *,
    event: Any,
    max_parallel: int,
    scope_factory: (
        Callable[[ValidatedCommand], AbstractAsyncContextManager[dict[str, Any]]] | None
    ) = None,
) -> tuple[CommandExecutionResult, ...]:
    semaphore = asyncio.Semaphore(max(1, int(max_parallel)))

    async def invoke(item: ValidatedCommand) -> CommandExecutionResult:
        try:
            value = item.spec.handler(event, **dict(item.arguments))
            if inspect.isawaitable(value):
                value = await value
            content, assets, references, is_error, model_input_images = normalize_command_result(
                value
            )
            return CommandExecutionResult(
                item.parsed.ordinal,
                item.spec.name,
                not is_error,
                content,
                assets,
                references,
                model_input_images=model_input_images,
            )
        except CommandProtocolError as exc:
            return CommandExecutionResult(
                item.parsed.ordinal,
                item.spec.name,
                False,
                str(exc),
            )
        except Exception as exc:
            return CommandExecutionResult(
                item.parsed.ordinal,
                item.spec.name,
                False,
                "行动没有完成；请依据现有结果调整下一步。",
                diagnostic={
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else "",
                },
            )

    async def run(item: ValidatedCommand) -> CommandExecutionResult:
        async with semaphore:
            if scope_factory is None:
                return await invoke(item)
            async with scope_factory(item) as state:
                result = await invoke(item)
                state["result"] = result
                return result

    results: list[CommandExecutionResult] = []
    parallel: list[ValidatedCommand] = []
    for command in commands:
        if command.spec.serial:
            if parallel:
                results.extend(await asyncio.gather(*(run(item) for item in parallel)))
                parallel = []
            results.append(await run(command))
        else:
            parallel.append(command)
    if parallel:
        results.extend(await asyncio.gather(*(run(item) for item in parallel)))
    return tuple(sorted(results, key=lambda item: item.ordinal))


def terminal_decision(
    commands: Sequence[ValidatedCommand],
    reference_map: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    commands = tuple(commands)
    if not commands:
        raise CommandProtocolError("表达批次为空")
    silent = [item for item in commands if item.spec.send_kind == "SILENT"]
    absence = [item for item in commands if item.spec.send_kind == "ABSENCE"]
    if silent and len(commands) != 1:
        raise CommandProtocolError("不说了必须单独作为最终表达，不能与其他指令混合")
    if absence and len(commands) != 1:
        raise CommandProtocolError("暂离必须单独作为最终表达，不能与其他指令混合")
    if len(silent) > 1 or len(absence) > 1:
        raise CommandProtocolError("同一个终止决定不能重复提交")
    silent_decision = _silent_terminal_decision(silent)
    if silent_decision is not None:
        return silent_decision
    steps = _terminal_expression_steps(commands, reference_map or {})
    decision: dict[str, Any] = {"expression_steps": steps, "no_op": False}
    if absence:
        decision["temporary_absence"] = dict(absence[0].arguments)
    return decision


_VISIBLE_SEND_KINDS = frozenset({"TEXT", "IMAGE", "STICKER", "FILE"})
_MAX_SCENE_NARRATION_COUNT = 12
_MAX_SCENE_NARRATION_CHARACTERS = 4_000


def _terminal_expression_steps(
    commands: Sequence[ValidatedCommand],
    reference_map: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Fold hidden narration into the adjacent visible ledger message.

    Narration is a terminal expression item, not a result-producing action.  It
    therefore commits in the same model turn as the send while remaining absent
    from the platform outbox.  Position metadata keeps the original order when
    the normal message row is later projected to MainCore or background authors.
    """

    steps: list[dict[str, Any]] = []
    pending_before: list[str] = []
    last_visible_index: int | None = None
    narration_count = 0
    for item in commands:
        kind = str(item.spec.send_kind or "").strip().upper()
        if kind == "ABSENCE":
            continue
        if kind == "NARRATION":
            content = str(item.arguments.get("content") or "").strip()
            if not content:
                # A blank narration contributes no timeline event.  Ignoring it
                # lets a useful send in the same batch commit without turning a
                # harmless whitespace slip into another model round.
                continue
            narration_count += 1
            if narration_count > _MAX_SCENE_NARRATION_COUNT:
                raise CommandProtocolError(f"每次行动最多使用 {_MAX_SCENE_NARRATION_COUNT} 条旁白")
            if len(content) > _MAX_SCENE_NARRATION_CHARACTERS:
                raise CommandProtocolError(
                    f"旁白内容不能超过 {_MAX_SCENE_NARRATION_CHARACTERS} 个字符"
                )
            if last_visible_index is None:
                pending_before.append(content)
            else:
                steps[last_visible_index].setdefault("scene_narration_after", []).append(content)
            continue
        step = _terminal_step(item, reference_map)
        steps.append(step)
        if kind not in _VISIBLE_SEND_KINDS:
            continue
        last_visible_index = len(steps) - 1
        if pending_before:
            step["scene_narration_before"] = list(pending_before)
            pending_before.clear()
    if narration_count and last_visible_index is None:
        raise CommandProtocolError(
            "旁白不能单独出现，必须与至少一条发文字、发图片、发表情或发文件同批提交"
        )
    return steps


def _silent_terminal_decision(
    silent: Sequence[ValidatedCommand],
) -> dict[str, Any] | None:
    if not silent:
        return None
    return {"expression_steps": [], "no_op": True}


def _terminal_step(
    item: ValidatedCommand,
    reference_map: Mapping[str, Any],
) -> dict[str, Any]:
    kind = item.spec.send_kind
    args = dict(item.arguments)
    delay = _strict_integer(args.get("delay_seconds", ""), "延迟", 0, 120, default=0)
    if kind == "RETRACT":
        if str(args.get("memo") or "").strip():
            raise CommandProtocolError("撤回不能携带留话")
        message_ref = str(args.get("message_ref") or "").strip()
        target_ordinal_text = str(args.get("target_output_ordinal") or "").strip()
        if bool(message_ref) == bool(target_ordinal_text):
            raise CommandProtocolError("撤回必须且只能填写 [[消息]] 或 [[本轮消息]] 其中一个")
        step: dict[str, Any] = {
            "kind": "RETRACT",
            "delay_after_previous_seconds": delay,
        }
        if message_ref:
            step["target_message_ref"] = _strict_reference(message_ref, reference_map, "消息")
        else:
            step["target_output_ordinal"] = _strict_integer(
                target_ordinal_text,
                "本轮消息",
                1,
                10_000,
            )
        return step
    step = _visible_terminal_step(kind, args, delay, reference_map)
    _attach_addressing(step, args, reference_map)
    memo = str(args.get("memo") or "").strip()
    if memo:
        step["memo"] = memo
    return step


def _visible_terminal_step(
    kind: str,
    args: Mapping[str, str],
    delay: int,
    reference_map: Mapping[str, Any],
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "kind": kind,
        "delay_after_previous_seconds": delay,
        "can_be_interrupted": _strict_boolean(
            args.get("interruptible", ""),
            "可被打断",
            default=True,
        ),
    }
    if kind != "TEXT":
        step["asset_ref_id"] = _strict_reference(
            str(args.get("asset_ref") or "").strip(),
            reference_map,
            "图片" if kind == "IMAGE" else "表情" if kind == "STICKER" else "文件",
        )
        return step
    text = str(args.get("content") or "").strip()
    if not text:
        raise CommandProtocolError("发文字的内容不能为空")
    step["text"] = text
    if _strict_boolean(args.get("as_voice", ""), "语音", default=False):
        step["presentation"] = "VOICE"
        step["as_voice"] = True
    return step


def _attach_addressing(
    step: dict[str, Any],
    args: Mapping[str, str],
    reference_map: Mapping[str, Any],
) -> None:
    reply_ref = str(args.get("reply_ref") or "").strip()
    if reply_ref:
        step["reply_to_message_ref"] = _strict_reference(reply_ref, reference_map, "回复")
    mentions = _string_list(args.get("mention_refs") or "")
    if mentions:
        step["mention_member_refs"] = [
            _strict_reference(item, reference_map, "提及") for item in mentions
        ]


def _string_list(value: str) -> list[str]:
    normalized = str(value or "")
    for separator in ("，", "、", "；", ";", "\n"):
        normalized = normalized.replace(separator, ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def _strict_boolean(value: str, label: str, *, default: bool | None = None) -> bool:
    text = str(value or "").strip()
    if not text and default is not None:
        return default
    if text == "是":
        return True
    if text == "否":
        return False
    raise CommandProtocolError(f"[[{label}]] 只能填写“是”或“否”")


def _strict_integer(
    value: str,
    label: str,
    minimum: int,
    maximum: int,
    *,
    default: int | None = None,
) -> int:
    text = str(value or "").strip()
    if not text and default is not None:
        return default
    if not text or any(character not in "+-0123456789" for character in text):
        raise CommandProtocolError(f"[[{label}]] 必须是 {minimum} 到 {maximum} 的整数")
    try:
        parsed = int(text)
    except ValueError as exc:
        raise CommandProtocolError(f"[[{label}]] 必须是 {minimum} 到 {maximum} 的整数") from exc
    if parsed < minimum or parsed > maximum:
        raise CommandProtocolError(f"[[{label}]] 必须是 {minimum} 到 {maximum} 的整数")
    return parsed


def _strict_reference(value: str, reference_map: Mapping[str, Any], label: str) -> str:
    reference = str(value or "").strip()
    if not reference or reference not in reference_map:
        raise CommandProtocolError(f"[[{label}]] 使用了当前不可用的短引用：{reference or '（空）'}")
    return str(reference_map[reference])


__all__ = [
    "CommandParameter",
    "CommandExecutionResult",
    "CommandProtocolError",
    "CommandSpec",
    "MainCoreCommandRegistry",
    "ModelVisibleCommandResult",
    "ParsedCommand",
    "ParsedModelTurn",
    "ValidatedCommand",
    "execute_nonterminal_batch",
    "parse_model_turn",
    "register_result_references",
    "terminal_decision",
]
