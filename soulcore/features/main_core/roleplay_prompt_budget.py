"""Bounded MainCore Prompt reduction without changing rendered prose."""

from __future__ import annotations

from collections.abc import Callable

from ...shared.prompt_document import CompiledPrompt
from ...shared.token_meter import ConservativeTokenMeter
from ..character_model.service import budget_rendered_character_projection
from ..conversation import ContextSource
from .roleplay_prompt_contracts import BoundedPromptState

PromptRenderer = Callable[[BoundedPromptState], CompiledPrompt]


def budget_persona_projection(
    state: BoundedPromptState,
    compiled: CompiledPrompt,
    maximum: int,
    render: PromptRenderer,
) -> CompiledPrompt:
    meter = ConservativeTokenMeter(state.model_id)
    persona_tokens = meter.count_text(state.persona)
    overflow = compiled.total_tokens - maximum
    target = max(1, persona_tokens - overflow - 8)
    projected = budget_rendered_character_projection(
        state.persona,
        max_tokens=target,
        token_meter=meter,
    )
    if not projected or projected == state.persona:
        return compiled
    state.persona = projected
    state.trim_reasons.append(
        f"final_prompt_structure_budget_persona:{persona_tokens}->{meter.count_text(projected)}"
    )
    return render(state)


def trim_bounded_sources(
    state: BoundedPromptState,
    compiled: CompiledPrompt,
    maximum: int,
    render: PromptRenderer,
) -> CompiledPrompt:
    trim_order = (
        ContextSource.CURRENT_WEB_RESOURCE,
        ContextSource.STICKER,
        ContextSource.CHARACTER_INTENT,
        ContextSource.PLAYER_PROFILE,
    )
    for source in trim_order:
        compiled, removed = _drop_source_until_bounded(
            state,
            compiled,
            maximum,
            source,
            render,
        )
        if removed:
            state.trim_reasons.append(f"final_prompt_drop_{source.value}:{removed}")
            compiled = render(state)
        if compiled.total_tokens <= maximum:
            return compiled

    compiled, removed_background = _drop_background_until_bounded(
        state,
        compiled,
        maximum,
        render,
    )
    if removed_background:
        state.trim_reasons.append(f"final_prompt_drop_background_material:{removed_background}")
        compiled = render(state)
    if compiled.total_tokens <= maximum:
        return compiled

    for source in (
        ContextSource.CURRENT_DIALOGUE,
        ContextSource.HISTORY_FRAGMENT,
        ContextSource.HISTORY_SUMMARY,
    ):
        compiled, removed = _drop_source_until_bounded(
            state,
            compiled,
            maximum,
            source,
            render,
        )
        if removed:
            state.trim_reasons.append(f"final_prompt_drop_{source.value}:{removed}")
            compiled = render(state)
        if compiled.total_tokens <= maximum:
            return compiled
    return compiled


def _drop_source_until_bounded(
    state: BoundedPromptState,
    compiled: CompiledPrompt,
    maximum: int,
    source: ContextSource,
    render: PromptRenderer,
) -> tuple[CompiledPrompt, int]:
    removed = 0
    while state.working[source] and compiled.total_tokens > maximum:
        if not drop_bounded_item(state, source):
            break
        removed += 1
        compiled = render(state)
    return compiled, removed


def _drop_background_until_bounded(
    state: BoundedPromptState,
    compiled: CompiledPrompt,
    maximum: int,
    render: PromptRenderer,
) -> tuple[CompiledPrompt, int]:
    removed = 0
    while compiled.total_tokens > maximum and drop_last_loaded_background_item(state):
        removed += 1
        compiled = render(state)
    return compiled, removed


def drop_last_loaded_background_item(state: BoundedPromptState) -> bool:
    """Undo the last optional background work admitted by category rotation."""

    optional_sources = (
        ContextSource.BACKGROUND_WORLD,
        ContextSource.BACKGROUND_STORY,
        ContextSource.BACKGROUND_LEFTOVER,
        ContextSource.BACKGROUND_EXPERIENCE,
        ContextSource.BACKGROUND_KEYFRAME,
    )
    loaded = [
        (load_order, source, index)
        for source in optional_sources
        for index, load_order in enumerate(state.working_background_load_orders[source])
        if load_order >= 0
    ]
    if not loaded:
        return False

    _load_order, source, index = max(loaded, key=lambda item: item[0])
    _pop_working_item(state, source, index)
    return True


def drop_bounded_item(state: BoundedPromptState, source: ContextSource) -> bool:
    if source is ContextSource.CURRENT_DIALOGUE:
        removable = next(
            (index for index, anchored in enumerate(state.dialogue_flags) if not anchored),
            None,
        )
        if removable is None:
            return False
        state.has_searchable_earlier_history = True
        _pop_working_item(state, source, removable)
        state.dialogue_flags.pop(removable)
        return True
    if source is ContextSource.HISTORY_SUMMARY and len(state.working[source]) <= 1:
        # The raw messages covered by the cumulative summary are not present in
        # the dialogue window.  Never turn final Prompt fitting into a history
        # hole by deleting the last surviving summary.
        return False
    if source in {ContextSource.HISTORY_FRAGMENT, ContextSource.HISTORY_SUMMARY}:
        state.has_searchable_earlier_history = True
    index = 0 if source in {ContextSource.HISTORY_FRAGMENT, ContextSource.HISTORY_SUMMARY} else -1
    _pop_working_item(state, source, index)
    return True


def _pop_working_item(
    state: BoundedPromptState,
    source: ContextSource,
    index: int,
) -> None:
    state.working[source].pop(index)
    state.working_message_ids[source].pop(index)
    state.working_summary_ids[source].pop(index)
    state.working_sequences[source].pop(index)
    state.working_item_refs[source].pop(index)
    state.working_background_load_orders[source].pop(index)


__all__ = [
    "budget_persona_projection",
    "drop_bounded_item",
    "drop_last_loaded_background_item",
    "trim_bounded_sources",
]
