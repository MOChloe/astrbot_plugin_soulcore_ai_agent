from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class CommandProtocolError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    name: str
    parameters: Mapping[str, str]
    ordinal: int
    raw_text: str
    unlabeled_content: str = ""


@dataclass(frozen=True, slots=True)
class ParsedModelTurn:
    working_text: str
    commands: tuple[ParsedCommand, ...]
    errors: tuple[str, ...] = ()
    raw_text: str = ""

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class CommandParameter:
    label: str
    internal_name: str
    required: bool = False
    choices: tuple[str, ...] = ()
    prompt_hint: str = ""
    identity_mode: str = "render"
    validator: Callable[[str, Mapping[str, Any]], str] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.identity_mode not in {"literal", "render", "template"}:
            raise ValueError(f"unknown command identity mode: {self.identity_mode}")


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    internal_name: str
    description: str
    parameters: tuple[CommandParameter, ...] = ()
    terminal: bool = False
    send_kind: str = ""
    serial: bool = False
    handler: Any = field(default=None, repr=False, compare=False)
    usage_guidance: str = ""
    prompt_visible: bool = True
    body_parameter: str = ""


@dataclass(frozen=True, slots=True)
class ValidatedCommand:
    parsed: ParsedCommand
    spec: CommandSpec
    arguments: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class CommandExecutionResult:
    ordinal: int
    command_name: str
    ok: bool
    content: str
    media_asset_ids: tuple[str, ...] = ()
    references: tuple[tuple[str, str], ...] = ()
    diagnostic: Mapping[str, Any] = field(default_factory=dict)
    public_references: tuple[str, ...] = ()
    model_input_images: tuple[str, ...] = field(default=(), repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ModelVisibleCommandResult:
    """Producer-owned natural-language projection for the next model turn."""

    content: str
    media_asset_ids: tuple[str, ...] = ()
    reference_hints: tuple[tuple[str, str], ...] = ()
    content_parts: tuple[Mapping[str, Any], ...] = field(default=(), repr=False, compare=False)


class CommandSetLike(Protocol):
    commands: Sequence[CommandSpec]
    terminal_handler: Callable[..., object] | None
    disabled_terminal_send_kinds: frozenset[str]


def parse_boolean(value: str, *, label: str, default: bool | None = None) -> bool:
    text = str(value or "").strip()
    if not text and default is not None:
        return default
    if text == "是":
        return True
    if text == "否":
        return False
    raise CommandProtocolError(f"[[{label}]] 只能填写“是”或“否”")


def parse_integer(
    value: str,
    *,
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
    default: int | None = None,
) -> int:
    text = str(value or "").strip()
    if not text and default is not None:
        return default
    if not re.fullmatch(r"[+-]?\d+", text):
        raise CommandProtocolError(f"[[{label}]] 必须填写整数")
    parsed = int(text)
    if minimum is not None and parsed < minimum or maximum is not None and parsed > maximum:
        if minimum is not None and maximum is not None:
            raise CommandProtocolError(f"[[{label}]] 必须是 {minimum} 到 {maximum} 的整数")
        if minimum is not None:
            raise CommandProtocolError(f"[[{label}]] 不能小于 {minimum}")
        raise CommandProtocolError(f"[[{label}]] 不能大于 {maximum}")
    return parsed


def parse_string_list(value: str) -> tuple[str, ...]:
    normalized = str(value or "")
    for separator in ("，", "、", "；", ";", "\n"):
        normalized = normalized.replace(separator, ",")
    return tuple(item.strip() for item in normalized.split(",") if item.strip())


def resolve_reference(value: str, reference_map: Mapping[str, Any], *, label: str) -> Any:
    reference = str(value or "").strip()
    if not reference or reference not in reference_map:
        raise CommandProtocolError(f"[[{label}]] 使用了当前不可用的短引用：{reference or '（空）'}")
    return reference_map[reference]


def resolve_reference_list(
    value: str,
    reference_map: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Any, ...]:
    references = parse_string_list(value)
    if not references:
        raise CommandProtocolError(f"[[{label}]] 至少需要一个短引用")
    return tuple(resolve_reference(item, reference_map, label=label) for item in references)


def boolean_validator(label: str) -> Callable[[str, Mapping[str, Any]], str]:
    def validate(value: str, _references: Mapping[str, Any]) -> str:
        try:
            parse_boolean(value, label=label)
        except CommandProtocolError as exc:
            return str(exc)
        return ""

    return validate


def integer_validator(
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> Callable[[str, Mapping[str, Any]], str]:
    def validate(value: str, _references: Mapping[str, Any]) -> str:
        try:
            parse_integer(
                value,
                label=label,
                minimum=minimum,
                maximum=maximum,
            )
        except CommandProtocolError as exc:
            return str(exc)
        return ""

    return validate


def reference_validator(
    label: str,
    *,
    multiple: bool = False,
) -> Callable[[str, Mapping[str, Any]], str]:
    def validate(value: str, references: Mapping[str, Any]) -> str:
        try:
            if multiple:
                resolve_reference_list(value, references, label=label)
            else:
                resolve_reference(value, references, label=label)
        except CommandProtocolError as exc:
            return str(exc)
        return ""

    return validate


__all__ = [
    "CommandExecutionResult",
    "CommandParameter",
    "CommandProtocolError",
    "CommandSetLike",
    "CommandSpec",
    "ModelVisibleCommandResult",
    "ParsedCommand",
    "ParsedModelTurn",
    "ValidatedCommand",
    "boolean_validator",
    "integer_validator",
    "parse_boolean",
    "parse_integer",
    "parse_string_list",
    "reference_validator",
    "resolve_reference",
    "resolve_reference_list",
]
