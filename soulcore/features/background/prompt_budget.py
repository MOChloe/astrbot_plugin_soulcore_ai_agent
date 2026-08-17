"""Deterministic, bounded input selection for background free-writing."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ...shared.prompt_document import (
    TrustedPromptMarkup,
    compile_task_prompt,
    join_prompt_markup,
    prompt_markup_block,
)
from ...shared.token_meter import ConservativeTokenMeter
from .domain import (
    BackgroundAuthorInput,
    BackgroundAuthorKind,
    BackgroundAuthorState,
    BackgroundVisibleReferences,
)
from .prompt_rendering import (
    RECENT_BACKGROUND_LIFE_BLOCK_NAME,
    RECENT_FOREGROUND_BLOCK_NAME,
    ROLE_CURRENT_STATE_BLOCK_NAME,
    ROLE_LOCATION_BLOCK_NAME,
    ROLE_PROFILE_BLOCK_NAME,
)

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BackgroundPromptBudget:
    max_context_tokens: int = 128_000
    safety_margin_tokens: int = 512

    def __post_init__(self) -> None:
        if self.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be positive")
        if self.safety_margin_tokens < 0:
            raise ValueError("safety_margin_tokens must not be negative")

    def stage_input_limit(
        self,
        *,
        task_definition: str,
        output_contract: str,
        output_reserve_tokens: int,
    ) -> int:
        # A CJK sentinel contributes exactly one token without sharing the
        # fallback meter's ceil(non-CJK/3) bucket with surrounding XML.
        sentinel = TrustedPromptMarkup("测")
        meter = ConservativeTokenMeter()
        compiled = compile_task_prompt(
            task_definition=task_definition,
            task_input=sentinel,
            output_contract=output_contract,
        )
        fixed = compiled.total_tokens - meter.count_text(sentinel)
        return max(
            0,
            self.max_context_tokens
            - max(1, int(output_reserve_tokens))
            - self.safety_margin_tokens
            - fixed,
        )


@dataclass(frozen=True, slots=True)
class FrozenBackgroundProjection:
    source: BackgroundAuthorInput
    snapshot: dict[str, Any]
    source_tokens: int


class BackgroundPromptBudgetExceeded(RuntimeError):
    reason_code = "background_prompt_required_context_exceeds_window"

    def __init__(self, *, stage: str, block_name: str, block_tokens: int, limit: int) -> None:
        self.stage = stage
        self.block_name = block_name
        self.block_tokens = block_tokens
        self.limit = limit
        super().__init__(
            f"{self.reason_code}: stage={stage} block={block_name} "
            f"block_tokens={block_tokens} input_limit={limit}"
        )


def freeze_background_projection(
    author_kind: BackgroundAuthorKind,
    source: BackgroundAuthorInput,
) -> FrozenBackgroundProjection:
    """Select a bounded, immutable source before the model call.

    Selection is by author responsibility.  No model request is spent deciding
    what to look at, and a future repository field cannot silently leak into a
    prompt.
    """

    if source.author_kind is not author_kind:
        raise ValueError("background author input kind does not match requested author")
    snapshot: dict[str, Any] = {
        "seed": dict(source.seed),
        "lore": tuple(source.lore[:24]),
        "boundaries": tuple(source.boundaries[:32]),
        "prompt_now": source.prompt_now,
        "timezone_name": source.timezone_name,
        "world_state": _author_state(source, BackgroundAuthorKind.WORLD),
    }
    if author_kind in {
        BackgroundAuthorKind.LIFE_DIRECTION,
        BackgroundAuthorKind.STORY_SOURCE,
        BackgroundAuthorKind.KEYFRAME,
        BackgroundAuthorKind.ORDINARY,
    }:
        snapshot["character_view"] = source.current_view
    if author_kind in {
        BackgroundAuthorKind.LIFE_DIRECTION,
        BackgroundAuthorKind.STORY_SOURCE,
        BackgroundAuthorKind.KEYFRAME,
        BackgroundAuthorKind.ORDINARY,
    }:
        snapshot["life_state"] = _author_state(
            source,
            BackgroundAuthorKind.LIFE_DIRECTION,
        )
    if (
        author_kind is BackgroundAuthorKind.LIFE_DIRECTION
        and str(source.initialization_state).upper() == "INITIALIZING"
        and source.initialization_step.value == "LIFE_DIRECTION"
    ):
        snapshot["initial_life_direction"] = str(
            source.config.get("initial_life_direction") or ""
        ).strip()
    if source.recent_timeline:
        snapshot["recent_timeline"] = tuple(source.recent_timeline[-16:])
    if author_kind in {
        BackgroundAuthorKind.STORY_SOURCE,
        BackgroundAuthorKind.KEYFRAME,
        BackgroundAuthorKind.ORDINARY,
    }:
        snapshot["story_sources"] = tuple(source.story_sources[:12])
    if author_kind is not BackgroundAuthorKind.WORLD:
        snapshot["foreground_messages"] = tuple(source.foreground_messages)
        snapshot["foreground_runs"] = tuple(source.foreground_runs)
    if author_kind is BackgroundAuthorKind.ORDINARY:
        snapshot["ordinary_frame_interval"] = source.ordinary_frame_interval
    if author_kind is BackgroundAuthorKind.KEYFRAME:
        snapshot["keyframe_frame_interval"] = source.keyframe_frame_interval
    meter = ConservativeTokenMeter()
    source_tokens = meter.count_value(snapshot)
    _LOG.info(
        "background_prompt_projection author=%s source_tokens=%d",
        author_kind.value,
        source_tokens,
    )
    return FrozenBackgroundProjection(source=source, snapshot=snapshot, source_tokens=source_tokens)


def visible_background_references(
    frozen: FrozenBackgroundProjection,
    task_input: TrustedPromptMarkup,
) -> BackgroundVisibleReferences:
    """Resolve only numbered records that survived into the final stage input."""

    snapshot = frozen.snapshot
    story_by_ordinal = {
        index: item
        for index, item in enumerate(snapshot.get("story_sources") or (), start=1)
        if item.module_text
    }
    timeline_by_ordinal = {
        index: item
        for index, item in enumerate(
            reversed(snapshot.get("recent_timeline") or ()),
            start=1,
        )
        if item.content or item.leftover_text
    }
    return BackgroundVisibleReferences(
        story_sources=MappingProxyType(
            {
                f"M{ordinal}": story_by_ordinal[ordinal]
                for ordinal in _visible_record_ordinals(
                    task_input,
                    container_name="可选故事模组",
                    record_name="模组",
                    prefix="M",
                )
                if ordinal in story_by_ordinal
            }
        ),
        timeline_events=MappingProxyType(
            {
                f"V{ordinal}": timeline_by_ordinal[ordinal]
                for ordinal in _visible_record_ordinals(
                    task_input,
                    container_name=RECENT_BACKGROUND_LIFE_BLOCK_NAME,
                    record_name="经历",
                    prefix="V",
                )
                if ordinal in timeline_by_ordinal
            }
        ),
    )


def _visible_record_ordinals(
    markup: TrustedPromptMarkup,
    *,
    container_name: str,
    record_name: str,
    prefix: str,
) -> tuple[int, ...]:
    container = next(
        (block for name, block in outer_prompt_blocks(markup) if name == container_name),
        None,
    )
    if container is None:
        return ()
    parsed = _record_container(container)
    if parsed is None:
        return ()
    _name, records = parsed
    ordinals: list[int] = []
    for record in records:
        lines = str(record).splitlines()
        if len(lines) < 3 or lines[0].strip() != f"<{record_name}>":
            continue
        number_line = lines[1].strip()
        marker = "[[编号]]:"
        if not number_line.startswith(marker):
            continue
        value = number_line[len(marker) :].strip()
        if value.startswith(prefix) and value[len(prefix) :].isdigit():
            ordinals.append(int(value[len(prefix) :]))
    return tuple(ordinals)


def _author_state(
    source: BackgroundAuthorInput,
    kind: BackgroundAuthorKind,
) -> BackgroundAuthorState | None:
    if source.author_state.author_kind is kind:
        return source.author_state
    return next(
        (item for item in source.reference_states if item.author_kind is kind),
        None,
    )


def fit_stage_markup(
    *,
    author_kind: BackgroundAuthorKind,
    stage: str,
    candidates: Sequence[tuple[str, TrustedPromptMarkup]],
    limit: int,
    required_name_fragments: Sequence[str] = (),
) -> TrustedPromptMarkup:
    """Keep required blocks and fill remaining space with complete records."""

    meter = ConservativeTokenMeter()
    expanded = _expanded_blocks(candidates)
    required_positions = {
        index
        for index, (name, _markup) in enumerate(expanded)
        if _is_required(name, required_name_fragments)
    }
    selected: dict[int, TrustedPromptMarkup] = {}
    for position in sorted(required_positions):
        name, markup = expanded[position]
        candidate = _join_selected(expanded, {**selected, position: markup})
        tokens = meter.count_text(candidate)
        if tokens > limit:
            raise BackgroundPromptBudgetExceeded(
                stage=stage,
                block_name=name,
                block_tokens=tokens,
                limit=limit,
            )
        selected[position] = markup
    optional = sorted(
        (
            (_stage_priority(stage, name), index, name, markup)
            for index, (name, markup) in enumerate(expanded)
            if index not in required_positions
        ),
        key=lambda item: (item[0], item[1]),
    )
    dropped: list[str] = []
    for _priority, position, name, markup in optional:
        candidate = _join_selected(expanded, {**selected, position: markup})
        if meter.count_text(candidate) <= limit:
            selected[position] = markup
            continue
        fitted = _fit_record_prefix(
            meter=meter,
            expanded=expanded,
            selected=selected,
            position=position,
            markup=markup,
            limit=limit,
        )
        if fitted is None:
            dropped.append(name)
            continue
        selected[position] = fitted
        dropped.append(f"{name}[部分记录]")
    result = _join_selected(expanded, selected)
    _LOG.info(
        "background_prompt_stage author=%s stage=%s allowed_tokens=%d used_tokens=%d dropped=%s",
        author_kind.value,
        stage,
        limit,
        meter.count_text(result),
        ",".join(dropped),
    )
    return result


def required_stage_context_tokens(
    *,
    task_definition: str,
    task_input: TrustedPromptMarkup,
    output_contract: str,
    output_reserve_tokens: int,
    safety_margin_tokens: int,
    required_name_fragments: Sequence[str],
) -> int:
    expanded = _expanded_blocks((("source", task_input),))
    required = _join_selected(
        expanded,
        {
            index: markup
            for index, (name, markup) in enumerate(expanded)
            if _is_required(name, required_name_fragments)
        },
    )
    return stage_context_tokens(
        task_definition=task_definition,
        task_input=required,
        output_contract=output_contract,
        output_reserve_tokens=output_reserve_tokens,
        safety_margin_tokens=safety_margin_tokens,
    )


def stage_context_tokens(
    *,
    task_definition: str,
    task_input: TrustedPromptMarkup,
    output_contract: str,
    output_reserve_tokens: int,
    safety_margin_tokens: int,
) -> int:
    compiled = compile_task_prompt(
        task_definition=task_definition,
        task_input=task_input,
        output_contract=output_contract,
    )
    return (
        compiled.total_tokens
        + max(1, int(output_reserve_tokens))
        + max(0, int(safety_margin_tokens))
    )


def _fit_record_prefix(
    *,
    meter: ConservativeTokenMeter,
    expanded: Sequence[tuple[str, TrustedPromptMarkup]],
    selected: dict[int, TrustedPromptMarkup],
    position: int,
    markup: TrustedPromptMarkup,
    limit: int,
) -> TrustedPromptMarkup | None:
    container = _record_container(markup)
    if container is None:
        return None
    name, records = container
    best: TrustedPromptMarkup | None = None
    for count in range(1, len(records) + 1):
        prefix = prompt_markup_block(name, join_prompt_markup(records[:count]))
        candidate = _join_selected(expanded, {**selected, position: prefix})
        if meter.count_text(candidate) > limit:
            break
        best = prefix
    return best


def _join_selected(
    expanded: Sequence[tuple[str, TrustedPromptMarkup]],
    selected: dict[int, TrustedPromptMarkup],
) -> TrustedPromptMarkup:
    return join_prompt_markup(
        selected[index] for index in range(len(expanded)) if index in selected
    )


def _expanded_blocks(
    candidates: Sequence[tuple[str, TrustedPromptMarkup]],
) -> list[tuple[str, TrustedPromptMarkup]]:
    expanded: list[tuple[str, TrustedPromptMarkup]] = []
    for source_name, markup in candidates:
        blocks = outer_prompt_blocks(markup)
        expanded.extend(blocks or [(source_name, markup)])
    return expanded


def _is_required(name: str, fragments: Sequence[str]) -> bool:
    return any(fragment in name for fragment in fragments)


_OPEN_BLOCK = re.compile(r"^<([^/][^>]*)>$")


def outer_prompt_blocks(
    markup: TrustedPromptMarkup,
) -> list[tuple[str, TrustedPromptMarkup]]:
    lines = str(markup).strip().splitlines()
    result: list[tuple[str, TrustedPromptMarkup]] = []
    start = 0
    name = ""
    depth = 0
    outside: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        opening = _OPEN_BLOCK.fullmatch(stripped)
        if opening:
            if depth == 0:
                start, name = index, opening.group(1)
            depth += 1
            continue
        if depth and stripped == f"</{name}>" and depth == 1:
            result.append((name, TrustedPromptMarkup("\n".join(lines[start : index + 1]))))
            depth, name = 0, ""
            continue
        if depth and stripped.startswith("</"):
            depth -= 1
            continue
        if not depth and stripped:
            outside.append(stripped)
    if outside:
        result.append(("background_notice", TrustedPromptMarkup("\n".join(outside))))
    return result


def _record_container(
    markup: TrustedPromptMarkup,
) -> tuple[str, tuple[TrustedPromptMarkup, ...]] | None:
    lines = str(markup).strip().splitlines()
    if len(lines) < 3:
        return None
    opening = _OPEN_BLOCK.fullmatch(lines[0].strip())
    if opening is None:
        return None
    name = opening.group(1)
    if lines[-1].strip() != f"</{name}>":
        return None
    records = _direct_blocks(lines[1:-1])
    return (name, records) if records else None


def _direct_blocks(lines: Sequence[str]) -> tuple[TrustedPromptMarkup, ...]:
    result: list[TrustedPromptMarkup] = []
    start = -1
    depth = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if _OPEN_BLOCK.fullmatch(stripped):
            if depth == 0:
                start = index
            depth += 1
            continue
        if stripped.startswith("</"):
            if depth < 1:
                return ()
            depth -= 1
            if depth == 0:
                result.append(TrustedPromptMarkup("\n".join(lines[start : index + 1])))
            continue
        if depth == 0 and stripped:
            return ()
    return tuple(result) if depth == 0 else ()


def _stage_priority(stage: str, name: str) -> int:
    value = str(name)
    important = (
        ROLE_PROFILE_BLOCK_NAME,
        ROLE_CURRENT_STATE_BLOCK_NAME,
        ROLE_LOCATION_BLOCK_NAME,
        RECENT_BACKGROUND_LIFE_BLOCK_NAME,
        RECENT_FOREGROUND_BLOCK_NAME,
        "已过去",
        "待审",
    )
    if any(part in value for part in important):
        return 0
    if any(part in value for part in ("世界", "边界", "作者状态", "故事模组")):
        return 1
    return 2


__all__ = [
    "BackgroundPromptBudget",
    "BackgroundPromptBudgetExceeded",
    "FrozenBackgroundProjection",
    "fit_stage_markup",
    "freeze_background_projection",
    "outer_prompt_blocks",
    "required_stage_context_tokens",
    "stage_context_tokens",
    "visible_background_references",
]
