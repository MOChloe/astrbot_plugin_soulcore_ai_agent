"""Shared role-free prompt documents for every SoulCore model request.

Domain objects, execution records and output contracts live inside one
continuous document.  Provider protocol wrapping belongs exclusively to the
outer transport adapter and must never leak back into this compiler.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from html import escape
from typing import Any

from ..contracts.ai_models import (
    AIPromptCacheBreakpoint,
    AIPromptCacheHint,
    AIPromptCacheSection,
    AIPromptCacheSemanticKind,
)
from .token_meter import ConservativeTokenMeter

_BLOCK_NAME = re.compile(
    r"^[A-Za-z0-9_.\-\u4e00-\u9fff]+"
    r"(?:（[A-Za-z0-9_.\-\u4e00-\u9fff ·]+）)?$"
)
_FIELD_NAME = re.compile(r"^[A-Za-z0-9_.\-\u4e00-\u9fff]+$")


def xml_text(value: Any) -> str:
    """Escape untrusted text while retaining readable line breaks."""

    return escape(str(value or ""), quote=False).strip()


class TrustedPromptMarkup(str):
    """System-owned markup whose dynamic values were escaped before composition."""


@dataclass(frozen=True, slots=True)
class PromptCacheBoundary:
    """Compiler-owned cache boundary attached to a block or its content.

    ``content_end`` is measured against the normalized block content. ``None``
    means the boundary follows the complete rendered block, including its
    closing tag. Boundaries are transport metadata and never render into the
    model-visible document.
    """

    boundary_id: str
    semantic_kind: AIPromptCacheSemanticKind
    selection_slot: int
    content_end: int | None = None
    selection_reason: str = ""


@dataclass(frozen=True, slots=True)
class PromptBlock:
    name: str
    content: str
    cache_boundaries: tuple[PromptCacheBoundary, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def normalized_name(self) -> str:
        name = str(self.name or "").strip()
        if not _BLOCK_NAME.fullmatch(name):
            raise ValueError(f"invalid prompt block name: {name!r}")
        return name

    def normalized_content(self) -> str:
        return str(self.content or "").strip()

    def render(self) -> str:
        name = self.normalized_name()
        content = self.normalized_content()
        return f"<{name}>\n{content}\n</{name}>" if content else ""


@dataclass(frozen=True, slots=True)
class CompiledPrompt:
    """One logical document with context/turn sections and no Provider roles."""

    context_text: str
    turn_text: str
    document: str
    blocks: tuple[PromptBlock, ...]
    token_counts: Mapping[str, int]
    total_tokens: int
    prompt_cache_hint: AIPromptCacheHint = field(default_factory=AIPromptCacheHint)
    reference_map: Mapping[str, Any] = field(default_factory=dict)
    trim_reasons: tuple[str, ...] = ()
    image_urls: tuple[str, ...] = ()
    source_message_ids: tuple[int, ...] = ()
    source_summary_ids: tuple[int, ...] = ()
    source_summary_coverage: tuple[tuple[int, int, int], ...] = ()
    background_item_refs: tuple[str, ...] = ()

    def debug_payload(self) -> dict[str, Any]:
        return {
            "document": self.document,
            "context_text": self.context_text,
            "turn_text": self.turn_text,
            "blocks": [
                {
                    "name": block.name,
                    "tokens": int(self.token_counts.get(block.name, 0)),
                    "cache_boundaries": [
                        {
                            "boundary_id": boundary.boundary_id,
                            "semantic_kind": boundary.semantic_kind.value,
                            "selection_slot": boundary.selection_slot,
                            "content_end": boundary.content_end,
                        }
                        for boundary in block.cache_boundaries
                    ],
                    "metadata": dict(block.metadata),
                }
                for block in self.blocks
            ],
            "total_tokens": self.total_tokens,
            "prompt_cache_hint": {
                "prompt_protocol_version": self.prompt_cache_hint.prompt_protocol_version,
                "candidates": [
                    _breakpoint_debug(item) for item in self.prompt_cache_hint.candidates
                ],
                "selected": [_breakpoint_debug(item) for item in self.prompt_cache_hint.selected],
                "rebase_reasons": list(self.prompt_cache_hint.rebase_reasons),
                "eligible": self.prompt_cache_hint.eligible,
            },
            "trim_reasons": list(self.trim_reasons),
            "source_message_ids": list(self.source_message_ids),
            "source_summary_ids": list(self.source_summary_ids),
            "source_summary_coverage": [
                {
                    "summary_id": int(summary_id),
                    "covered_from_message_id": int(covered_from),
                    "covered_through_message_id": int(covered_through),
                }
                for summary_id, covered_from, covered_through in self.source_summary_coverage
            ],
            "background_item_refs": list(self.background_item_refs),
        }


def compile_prompt_document(
    context_blocks: Sequence[PromptBlock],
    turn_blocks: Sequence[PromptBlock],
    *,
    model_id: str = "",
    reference_map: Mapping[str, Any] | None = None,
    trim_reasons: Iterable[str] = (),
    image_urls: Sequence[str] = (),
    prompt_protocol_version: str = "soulcore-prompt-v2",
    cache_rebase_reasons: Iterable[str] = (),
) -> CompiledPrompt:
    meter = ConservativeTokenMeter(model_id)
    clean_context = _nonempty_blocks(context_blocks)
    clean_turn = _nonempty_blocks(turn_blocks)
    context_text = _render_blocks(clean_context)
    turn_text = _render_blocks(clean_turn)
    document = _join_nonempty((context_text, turn_text))
    blocks = (*clean_context, *clean_turn)
    token_counts = _block_token_counts(blocks, meter)
    candidates = _cache_breakpoints(
        clean_context,
        clean_turn,
        context_text=context_text,
        document=document,
        meter=meter,
    )
    prompt_cache_hint = AIPromptCacheHint(
        prompt_protocol_version=str(prompt_protocol_version or "soulcore-prompt-v2"),
        candidates=candidates,
        selected=_select_cache_breakpoints(candidates),
        rebase_reasons=_unique_strings(cache_rebase_reasons),
    )
    return CompiledPrompt(
        context_text=context_text,
        turn_text=turn_text,
        document=document,
        blocks=blocks,
        token_counts=token_counts,
        total_tokens=meter.count_text(document),
        prompt_cache_hint=prompt_cache_hint,
        reference_map=dict(reference_map or {}),
        trim_reasons=_unique_strings(trim_reasons),
        image_urls=_unique_strings(image_urls)[:5],
    )


def _nonempty_blocks(blocks: Sequence[PromptBlock]) -> tuple[PromptBlock, ...]:
    return tuple(block for block in blocks if block.render())


def _render_blocks(blocks: Sequence[PromptBlock]) -> str:
    return "\n\n".join(block.render() for block in blocks)


def _cache_breakpoints(
    context_blocks: Sequence[PromptBlock],
    turn_blocks: Sequence[PromptBlock],
    *,
    context_text: str,
    document: str,
    meter: ConservativeTokenMeter,
) -> tuple[AIPromptCacheBreakpoint, ...]:
    candidates: list[AIPromptCacheBreakpoint] = []
    candidates.extend(
        _section_cache_breakpoints(
            context_blocks,
            section=AIPromptCacheSection.CONTEXT,
            document_base=0,
        )
    )
    turn_base = len(context_text) + 2 if context_text and turn_blocks else 0
    candidates.extend(
        _section_cache_breakpoints(
            turn_blocks,
            section=AIPromptCacheSection.TURN,
            document_base=turn_base,
        )
    )
    measured = _measure_cache_breakpoints(candidates, document=document, meter=meter)
    return tuple(sorted(measured, key=lambda item: (item.document_end, item.selection_slot)))


def _section_cache_breakpoints(
    blocks: Sequence[PromptBlock],
    *,
    section: AIPromptCacheSection,
    document_base: int,
) -> list[AIPromptCacheBreakpoint]:
    result: list[AIPromptCacheBreakpoint] = []
    section_cursor = 0
    for block in blocks:
        rendered = block.render()
        if not rendered:
            continue
        if section_cursor:
            section_cursor += 2
        block_start = section_cursor
        content = block.normalized_content()
        content_start = block_start + len(f"<{block.normalized_name()}>\n")
        block_end = block_start + len(rendered)
        for boundary in block.cache_boundaries:
            if boundary.selection_slot not in {1, 2, 3, 4}:
                raise ValueError("prompt cache selection_slot must be between 1 and 4")
            if boundary.content_end is None:
                section_end = block_end
            else:
                content_end = int(boundary.content_end)
                if content_end < 0 or content_end > len(content):
                    raise ValueError(
                        f"cache boundary {boundary.boundary_id!r} is outside block content"
                    )
                section_end = content_start + content_end
            document_end = document_base + section_end
            result.append(
                AIPromptCacheBreakpoint(
                    boundary_id=str(boundary.boundary_id),
                    section=section,
                    semantic_kind=boundary.semantic_kind,
                    section_end=section_end,
                    document_end=document_end,
                    prefix_tokens=0,
                    prefix_hash="",
                    selection_slot=boundary.selection_slot,
                    selection_reason=str(boundary.selection_reason or ""),
                    block_name=block.normalized_name(),
                )
            )
        section_cursor = block_end
    return result


def _measure_cache_breakpoints(
    candidates: Sequence[AIPromptCacheBreakpoint],
    *,
    document: str,
    meter: ConservativeTokenMeter,
) -> tuple[AIPromptCacheBreakpoint, ...]:
    """Fill cumulative metrics for every candidate in one document scan."""

    if not candidates:
        return ()
    ends = sorted({item.document_end for item in candidates})
    token_counts = meter.count_text_prefixes(document, ends)
    prefix_hashes: dict[int, str] = {}
    digest = hashlib.sha256()
    cursor = 0
    for end in ends:
        digest.update(document[cursor:end].encode("utf-8"))
        prefix_hashes[end] = digest.hexdigest()
        cursor = end
    return tuple(
        replace(
            item,
            prefix_tokens=token_counts[item.document_end],
            prefix_hash=prefix_hashes[item.document_end],
        )
        for item in candidates
    )


def _select_cache_breakpoints(
    candidates: Sequence[AIPromptCacheBreakpoint],
) -> tuple[AIPromptCacheBreakpoint, ...]:
    by_slot: dict[int, list[AIPromptCacheBreakpoint]] = {}
    for item in candidates:
        by_slot.setdefault(item.selection_slot, []).append(item)
    chosen_by_position: dict[tuple[AIPromptCacheSection, int], AIPromptCacheBreakpoint] = {}
    for slot in (1, 2, 3, 4):
        values = by_slot.get(slot, ())
        if not values:
            continue
        item = max(values, key=lambda value: value.document_end)
        position = (item.section, item.section_end)
        previous = chosen_by_position.get(position)
        if previous is None or item.selection_slot > previous.selection_slot:
            # The deeper semantic wins when two logical slots describe the
            # same byte position.  In particular, a Plan rebase should be
            # diagnosed as CURRENT_DIALOGUE rather than PREVIOUS_DIALOGUE.
            chosen_by_position[position] = item
    return tuple(sorted(chosen_by_position.values(), key=lambda item: item.document_end))[:4]


def _breakpoint_debug(item: AIPromptCacheBreakpoint) -> dict[str, Any]:
    return {
        "boundary_id": item.boundary_id,
        "section": item.section.value,
        "semantic_kind": item.semantic_kind.value,
        "section_end": item.section_end,
        "document_end": item.document_end,
        "prefix_tokens": item.prefix_tokens,
        "prefix_hash": item.prefix_hash,
        "selection_slot": item.selection_slot,
        "selection_reason": item.selection_reason,
        "block_name": item.block_name,
    }


def _join_nonempty(values: Sequence[str]) -> str:
    return "\n\n".join(value for value in values if value)


def _block_token_counts(
    blocks: Sequence[PromptBlock], meter: ConservativeTokenMeter
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for block in blocks:
        counts[block.name] = counts.get(block.name, 0) + meter.count_text(block.render())
    return counts


def _unique_strings(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))


def prompt_markup_block(name: str, content: Any) -> TrustedPromptMarkup:
    """Build one trusted structural block around escaped or explicitly trusted content."""

    rendered_content = (
        str(content).strip() if isinstance(content, TrustedPromptMarkup) else xml_text(content)
    )
    return TrustedPromptMarkup(PromptBlock(name, rendered_content).render())


def prompt_field_lines(
    fields: Mapping[str, Any] | Sequence[tuple[str, Any]],
    *,
    omit_empty: bool = True,
) -> TrustedPromptMarkup:
    """Render dynamic values with the same ``[[字段]]: 内容`` syntax as MainCore.

    The labels are trusted schema names while every value is escaped independently.
    This keeps nested task inputs readable without falling back to JSON DTO dumps.
    """

    items = fields.items() if isinstance(fields, Mapping) else fields
    rendered: list[str] = []
    for raw_label, value in items:
        label = str(raw_label or "").strip()
        if not _FIELD_NAME.fullmatch(label):
            raise ValueError(f"invalid prompt field name: {label!r}")
        if omit_empty and value in (None, "", (), [], {}):
            continue
        rendered_value = xml_text(value)
        if omit_empty and not rendered_value:
            continue
        rendered.append(f"[[{label}]]: {rendered_value}")
    return TrustedPromptMarkup("\n".join(rendered))


def prompt_markup_record(
    name: str,
    fields: Mapping[str, Any] | Sequence[tuple[str, Any]],
    *,
    omit_empty: bool = True,
) -> TrustedPromptMarkup:
    """Build one trusted nested record from independently escaped field values."""

    return prompt_markup_block(
        name,
        prompt_field_lines(fields, omit_empty=omit_empty),
    )


def join_prompt_markup(blocks: Iterable[TrustedPromptMarkup]) -> TrustedPromptMarkup:
    """Join only blocks that have already crossed the trusted-markup boundary."""

    rendered: list[str] = []
    for block in blocks:
        if not isinstance(block, TrustedPromptMarkup):
            raise TypeError("prompt markup can only join trusted blocks")
        text = str(block).strip()
        if text:
            rendered.append(text)
    return TrustedPromptMarkup("\n\n".join(rendered))


def prompt_markup_text(value: Any) -> TrustedPromptMarkup:
    """Treat existing trusted markup as structure and escape every other value."""

    if isinstance(value, TrustedPromptMarkup):
        return value
    return TrustedPromptMarkup(xml_text(value))


def project_prompt_text(
    value: str,
    projector: Callable[[str], str],
) -> str:
    """Apply a trusted text projection without losing the prompt trust boundary."""

    projected = projector(str(value))
    return TrustedPromptMarkup(projected) if isinstance(value, TrustedPromptMarkup) else projected


def compile_task_prompt(
    *,
    task_definition: str,
    task_input: str,
    output_contract: str,
    execution_record: str = "",
    model_id: str = "",
) -> CompiledPrompt:
    """Compile a compact non-RolePlay request used by every peripheral AI task."""

    context = [
        # Task definitions and output contracts are SoulCore-owned markup.
        # Escaping either whole value would turn documented structure and
        # command tags into &lt;...&gt; noise. Only dynamic inputs and execution
        # records are untrusted and therefore escaped below.
        PromptBlock("任务定义", str(task_definition or "").strip()),
        PromptBlock(
            "输出格式",
            str(output_contract or "").strip(),
            cache_boundaries=(
                PromptCacheBoundary(
                    "task-protocol",
                    AIPromptCacheSemanticKind.PROTOCOL,
                    1,
                    selection_reason="任务定义与输出协议末端",
                ),
            ),
        ),
    ]
    task_input_content = (
        str(task_input).strip()
        if isinstance(task_input, TrustedPromptMarkup)
        else xml_text(task_input)
    )
    turn = [PromptBlock("任务输入", task_input_content)]
    if str(execution_record or "").strip():
        turn.append(PromptBlock("上次输出的问题", xml_text(execution_record)))
    return compile_prompt_document(
        context,
        turn,
        model_id=model_id,
        prompt_protocol_version="soulcore-task-prompt-v2",
    )


__all__ = [
    "CompiledPrompt",
    "PromptCacheBoundary",
    "PromptBlock",
    "TrustedPromptMarkup",
    "compile_prompt_document",
    "compile_task_prompt",
    "join_prompt_markup",
    "project_prompt_text",
    "prompt_field_lines",
    "prompt_markup_block",
    "prompt_markup_record",
    "prompt_markup_text",
    "xml_text",
]
