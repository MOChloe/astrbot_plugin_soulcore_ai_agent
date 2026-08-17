"""Markdown parsing and safe ReportLab inline projections for file artifacts."""

from __future__ import annotations

import html
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from ..._vendor.markdown_it import MarkdownIt

_MARKDOWN = MarkdownIt(
    "js-default",
    {"html": False, "linkify": False, "typographer": False},
)


@dataclass(slots=True)
class MarkdownNode:
    kind: str
    text: str = ""
    target: str = ""
    children: list[MarkdownNode] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    start: int = 1


def pdf_inline(text: str) -> str:
    parsed = _MARKDOWN.parseInline(str(text or ""))
    children = parsed[0].children if parsed and parsed[0].children else []
    result: list[str] = []
    link_stack: list[bool] = []
    for token in children:
        result.append(_inline_token(token, link_stack))
    return "".join(result)


def _inline_token(token: Any, link_stack: list[bool]) -> str:
    kind = token.type
    if kind == "link_open":
        return _open_link(token, link_stack)
    if kind == "link_close":
        return "</link>" if link_stack and link_stack.pop() else ""
    if kind == "image":
        return f"[图片：{html.escape(token.content or '图片')}]"
    literal = {
        "strong_open": "<b>",
        "strong_close": "</b>",
        "em_open": "<i>",
        "em_close": "</i>",
        "hardbreak": "<br/>",
        "softbreak": " ",
        "s_open": "",
        "s_close": "",
    }.get(kind)
    if literal is not None:
        return literal
    if kind == "text":
        return html.escape(token.content).replace("\n", "<br/>")
    if kind == "code_inline":
        return f'<font color="#475467">{html.escape(token.content)}</font>'
    return html.escape(token.content) if token.content else ""


def _open_link(token: Any, link_stack: list[bool]) -> str:
    target = str(token.attrGet("href") or "")
    allowed = target.lower().startswith(("https://", "http://"))
    link_stack.append(allowed)
    if not allowed:
        return ""
    return f'<link href="{html.escape(target, quote=True)}" color="#2563EB">'


class _MarkdownParser:
    def __init__(self, content: str) -> None:
        self.tokens = _MARKDOWN.parse(str(content or ""))
        self.index = 0

    def parse(self) -> list[MarkdownNode]:
        return self._sequence()

    def _sequence(self, end_type: str | None = None) -> list[MarkdownNode]:
        nodes: list[MarkdownNode] = []
        while self.index < len(self.tokens):
            token = self.tokens[self.index]
            if end_type and token.type == end_type:
                break
            node = self._consume(token.type)
            if node is not None:
                nodes.append(node)
        return nodes

    def _consume(self, token_type: str) -> MarkdownNode | None:
        handler = {
            "heading_open": self._heading,
            "paragraph_open": self._paragraph,
            "ordered_list_open": self._list,
            "bullet_list_open": self._list,
            "blockquote_open": self._quote,
            "table_open": self._table,
            "fence": self._code,
            "code_block": self._code,
            "hr": self._rule,
            "inline": self._inline,
        }.get(token_type, self._skip)
        return handler()

    def _heading(self) -> MarkdownNode:
        opening = self.tokens[self.index]
        level = int(opening.tag[1:]) if opening.tag.startswith("h") else 3
        self.index += 1
        text = self._current_inline_text()
        self._skip_type("inline")
        self._skip_type("heading_close")
        kind = "h1" if level == 1 else "h2" if level == 2 else "h3"
        return MarkdownNode(kind, text=text)

    def _paragraph(self) -> MarkdownNode:
        self.index += 1
        inline = self._current("inline")
        text = inline.content if inline is not None else ""
        image = _sole_image(inline)
        self._skip_type("inline")
        self._skip_type("paragraph_close")
        if image is None:
            return MarkdownNode("paragraph", text=text)
        return MarkdownNode(
            "image",
            text=str(image.content or "").strip(),
            target=str(image.attrGet("src") or "").strip(),
        )

    def _list(self) -> MarkdownNode:
        opening = self.tokens[self.index]
        ordered = opening.type == "ordered_list_open"
        kind = "ol" if ordered else "ul"
        start_value = opening.attrGet("start") if ordered else None
        close_type = "ordered_list_close" if ordered else "bullet_list_close"
        self.index += 1
        items: list[MarkdownNode] = []
        while self.index < len(self.tokens) and self.tokens[self.index].type != close_type:
            if self.tokens[self.index].type != "list_item_open":
                self.index += 1
                continue
            self.index += 1
            children = self._sequence("list_item_close")
            self._skip_type("list_item_close")
            items.append(MarkdownNode("item", children=children))
        self._skip_type(close_type)
        return MarkdownNode(kind, children=items, start=int(start_value or 1))

    def _quote(self) -> MarkdownNode:
        self.index += 1
        children = self._sequence("blockquote_close")
        self._skip_type("blockquote_close")
        return MarkdownNode("quote", children=children)

    def _table(self) -> MarkdownNode:
        self.index += 1
        rows: list[list[str]] = []
        while self.index < len(self.tokens) and self.tokens[self.index].type != "table_close":
            if self.tokens[self.index].type == "tr_open":
                rows.append(self._table_row())
            else:
                self.index += 1
        self._skip_type("table_close")
        return MarkdownNode("table", rows=rows)

    def _table_row(self) -> list[str]:
        self.index += 1
        row: list[str] = []
        while self.index < len(self.tokens) and self.tokens[self.index].type != "tr_close":
            if self.tokens[self.index].type in {"th_open", "td_open"}:
                row.append(self._table_cell())
            else:
                self.index += 1
        self._skip_type("tr_close")
        return row

    def _table_cell(self) -> str:
        opening = self.tokens[self.index].type
        close_type = "th_close" if opening == "th_open" else "td_close"
        self.index += 1
        values: list[str] = []
        while self.index < len(self.tokens) and self.tokens[self.index].type != close_type:
            token = self.tokens[self.index]
            if token.content:
                values.append(token.content)
            self.index += 1
        self._skip_type(close_type)
        return "\n".join(values).strip()

    def _code(self) -> MarkdownNode:
        token = self.tokens[self.index]
        self.index += 1
        return MarkdownNode("code", text=token.content.rstrip("\n"))

    def _rule(self) -> MarkdownNode:
        self.index += 1
        return MarkdownNode("rule")

    def _inline(self) -> MarkdownNode:
        token = self.tokens[self.index]
        self.index += 1
        return MarkdownNode("paragraph", text=token.content)

    def _skip(self) -> None:
        self.index += 1
        return None

    def _current(self, token_type: str) -> Any | None:
        if self.index < len(self.tokens) and self.tokens[self.index].type == token_type:
            return self.tokens[self.index]
        return None

    def _current_inline_text(self) -> str:
        token = self._current("inline")
        return str(token.content) if token is not None else ""

    def _skip_type(self, token_type: str) -> None:
        if self._current(token_type) is not None:
            self.index += 1


def _sole_image(inline: Any | None) -> Any | None:
    if inline is None:
        return None
    meaningful = [
        child
        for child in (inline.children or [])
        if child.type != "text" or str(child.content or "").strip()
    ]
    if len(meaningful) == 1 and meaningful[0].type == "image":
        return meaningful[0]
    return None


def parse_markdown_nodes(content: str) -> list[MarkdownNode]:
    return _MarkdownParser(content).parse()


def parse_markdown_blocks(content: str) -> list[tuple[str, Any]]:
    """Return the compact block projection used by artifact diagnostics."""

    return [_project_block(node) for node in parse_markdown_nodes(content)]


def _project_block(node: MarkdownNode) -> tuple[str, Any]:
    if node.kind == "table":
        return "table", node.rows
    if node.kind in {"ol", "ul"}:
        return node.kind, [_item_text(item) for item in node.children]
    if node.kind == "quote":
        return "quote", "\n".join(child.text for child in node.children if child.text)
    if node.kind == "rule":
        return "rule", None
    return node.kind, node.text


def _item_text(item: MarkdownNode) -> str:
    return next((child.text for child in item.children if child.kind == "paragraph"), "")


def pdf_table_widths(rows: list[list[str]], available: float) -> list[float]:
    column_count = max(len(row) for row in rows)
    weights = [_column_weight(rows, column) for column in range(column_count)]
    minimum = min(34.0, available / column_count)
    distributable = max(0.0, available - minimum * column_count)
    total = sum(weights) or 1.0
    return [minimum + distributable * weight / total for weight in weights]


def _column_weight(rows: list[list[str]], column: int) -> float:
    visual = max(_visual_width(row[column] if column < len(row) else "") for row in rows)
    return float(max(5, min(visual, 36)))


def _visual_width(value: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(character) in {"W", "F", "A"} else 1
        for character in value
    )


def effective_document_style(requested: str, content: str, image_count: int) -> str:
    import re

    if requested != "AUTO":
        return requested
    if re.search(r"\|[^\n]+\|\s*\n\s*\|\s*:?-", content):
        return "DATA_BRIEF"
    if image_count >= 1:
        return "EDITORIAL"
    return "REPORT"


def document_title(content: str) -> str:
    import re

    lines = content.splitlines()
    heading = next(
        (line.strip()[2:].strip() for line in lines if line.strip().startswith("# ")), ""
    )
    if heading:
        return heading[:80]
    fallback = next((line for line in lines if line.strip()), "")
    return re.sub(r"^[#>*+\-\d.)\s]+", "", fallback).strip()[:80] or "SoulCore 文档"


__all__ = [
    "MarkdownNode",
    "document_title",
    "effective_document_style",
    "parse_markdown_blocks",
    "parse_markdown_nodes",
    "pdf_inline",
    "pdf_table_widths",
]
