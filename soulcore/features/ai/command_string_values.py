"""Domain-owned parsers for MainCore's model-visible string arguments."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from .command_protocol_types import CommandProtocolError


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
    "boolean_validator",
    "integer_validator",
    "parse_boolean",
    "parse_integer",
    "parse_string_list",
    "reference_validator",
    "resolve_reference",
    "resolve_reference_list",
]
