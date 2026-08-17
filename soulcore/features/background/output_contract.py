"""Small parsing primitives for the background authors' prose protocol."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


class BackgroundOutputError(ValueError):
    pass


class BackgroundOutputFormatError(BackgroundOutputError):
    """The model did not return a parseable background author document."""


@dataclass(frozen=True, slots=True)
class TextBlock:
    """One exact outer tag and its otherwise untouched prose body."""

    name: str
    body: str


_OPENING_TAG = re.compile(r"<([^/<>\s]+)>")
_FIELD_LINE = re.compile(r"^\s*([^：:\r\n]+?)\s*[：:](.*)$")


def tagged_document(
    text: str,
    *,
    label: str,
    allowed: frozenset[str],
    reserved: frozenset[str] = frozenset(),
) -> tuple[TextBlock, ...]:
    """Parse a document made only of exact, non-nested outer blocks.

    Angle brackets inside a block remain ordinary prose unless they spell one
    of the protocol's current or retired structural tags.  This keeps
    comparisons and quoted notation intact while still making damaged,
    mismatched, or nested protocol markup fail loudly.
    """

    raw = str(text or "")
    if not raw.strip():
        raise BackgroundOutputFormatError(f"{label} returned empty output")

    blocks: list[TextBlock] = []
    position = 0
    length = len(raw)
    structural = frozenset((*allowed, *reserved))

    while True:
        while position < length and raw[position].isspace():
            position += 1
        if position >= length:
            break

        opening = _OPENING_TAG.match(raw, position)
        if opening is None:
            raise BackgroundOutputFormatError(
                f"{label} cannot contain text or damaged tags outside its blocks"
            )
        name = opening.group(1)
        if name not in allowed:
            raise BackgroundOutputFormatError(f"{label} returned unknown outer tag: {name}")

        close_token = f"</{name}>"
        close_at = raw.find(close_token, opening.end())
        if close_at < 0:
            raise BackgroundOutputFormatError(
                f"{label} outer tag {name} is missing its matching closing tag"
            )

        body = raw[opening.end() : close_at]
        _reject_structural_tag_in_body(
            body,
            label=label,
            structural=structural,
        )
        normalized = body.strip()
        if not normalized:
            raise BackgroundOutputError(f"{name}正文不能为空")
        blocks.append(TextBlock(name=name, body=normalized))
        position = close_at + len(close_token)

    if not blocks:
        raise BackgroundOutputFormatError(f"{label} must return at least one outer block")
    return tuple(blocks)


def field_lines(
    text: str,
    *,
    label: str,
    allowed: Iterable[str],
    required: Iterable[str],
) -> dict[str, str]:
    """Parse one ``字段：值`` line per current-role snapshot field."""

    allowed_fields = frozenset(str(item) for item in allowed)
    required_fields = frozenset(str(item) for item in required)
    result: dict[str, str] = {}

    for raw_line in str(text or "").splitlines():
        if not raw_line.strip():
            continue
        match = _FIELD_LINE.fullmatch(raw_line)
        if match is None:
            raise BackgroundOutputFormatError(f"{label}每个非空行都必须是“字段：值”")
        name = match.group(1).strip()
        if name not in allowed_fields:
            raise BackgroundOutputError(f"{label}包含未知字段：{name}")
        if name in result:
            raise BackgroundOutputError(f"{label}字段不能重复：{name}")
        value = match.group(2).strip()
        if not value:
            raise BackgroundOutputError(f"{label}字段不能为空：{name}")
        if _is_placeholder(value):
            raise BackgroundOutputError(f"{label}字段不能保留占位文字：{name}")
        result[name] = value

    missing = sorted(required_fields - result.keys())
    if missing:
        raise BackgroundOutputError(f"{label}缺少必填字段：{'、'.join(missing)}")
    return result


def _reject_structural_tag_in_body(
    body: str,
    *,
    label: str,
    structural: frozenset[str],
) -> None:
    for name in structural:
        if f"<{name}>" in body or f"</{name}>" in body:
            raise BackgroundOutputFormatError(f"{label}正文包含嵌套或错配的协议标签：{name}")


def _is_placeholder(value: str) -> bool:
    raw = str(value or "").strip()
    normalized = raw.strip("【】[]（）()<>《》").strip()
    if normalized in {"可选", "必填", "无", "没有", "清空", "-", "—", "/"}:
        return True
    wrapped = len(raw) >= 2 and raw[0] in "【[（(<《" and raw[-1] in "】]）)>》"
    return wrapped and normalized.startswith(("可选", "必填"))


__all__ = [
    "BackgroundOutputError",
    "BackgroundOutputFormatError",
    "TextBlock",
    "field_lines",
    "tagged_document",
]
