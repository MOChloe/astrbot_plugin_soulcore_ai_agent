"""Compile the model-facing RolePlay document from bounded SoulCore context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from functools import partial
from typing import Any, TypeVar

from ...contracts.models import CoreState, ScopeConfig
from ...contracts.thinking import DEFAULT_THINKING_POLICY, MainCoreThinkingPolicy
from ...shared.prompt_document import (
    CompiledPrompt,
    PromptBlock,
    PromptCacheBoundary,
    TrustedPromptMarkup,
    project_prompt_text,
    xml_text,
)
from ..ai.service import (
    MainCoreCommandRegistry,
)
from ..character_model import (
    DEFAULT_RELATIONSHIP_CONTEXT_PROMPT,
    MainCoreModePrompts,
    MainCoreStylePrompts,
    StoryStylePrompts,
)
from ..conversation import ContextBudgetExceeded, ContextSource
from ..conversation.service import ConversationContextService, PreparedMainCoreContext
from ..identity import project_identity_text_for_model
from . import turn_prompt
from .roleplay_prompt_budget import (
    budget_persona_projection,
    drop_bounded_item,
    drop_last_loaded_background_item,
    trim_bounded_sources,
)
from .roleplay_prompt_contracts import (
    BoundedPromptState,
    DialoguePromptEntry,
    ExecutionRound,
    SemanticConversationProjection,
    full_datetime_label,
    project_main_core_styles,
    project_story_styles,
)
from .roleplay_prompt_rendering import (
    MAIN_CORE_PROMPT_PROTOCOL_VERSION as MAIN_CORE_PROMPT_PROTOCOL_VERSION,
)
from .roleplay_prompt_rendering import (
    RolePlayPromptRenderingMixin,
)
from .roleplay_references import RolePlayReferenceMixin
from .thinking_prompt import thinking_requirement
from .turn_responsibility import (
    DEFAULT_MESSAGE_RESPONSIBILITY,
    MainCoreTurnResponsibility,
)


def project_prepared_identity(value: Any, prepared_context: Any | None) -> str:
    if (
        prepared_context is None
        or prepared_context.identity_context is None
        or prepared_context.identity_catalog is None
    ):
        return value if isinstance(value, TrustedPromptMarkup) else str(value or "")
    return project_prompt_text(
        value,
        lambda text: project_identity_text_for_model(
            text,
            prepared_context.identity_catalog,
            scope=str(prepared_context.identity_context.scope),
        ),
    )


def project_catalog_identity(
    value: Any,
    identity_catalog: Any | None,
    identity_scope: str,
) -> str:
    if identity_catalog is None:
        return value if isinstance(value, TrustedPromptMarkup) else str(value or "")
    return project_prompt_text(
        value,
        lambda text: project_identity_text_for_model(
            text,
            identity_catalog,
            scope=identity_scope,
        ),
    )


_T = TypeVar("_T")


def _copy_source_lists(
    values: Mapping[ContextSource, Sequence[_T]],
) -> dict[ContextSource, list[_T]]:
    return {source: list(items) for source, items in values.items()}


def build_bounded_prompt_state(
    *,
    persona: str,
    main_core_style_prompts: MainCoreStylePrompts,
    story_style_prompts: StoryStylePrompts,
    background_enabled: bool,
    world: str,
    current_time: str,
    groups: Mapping[ContextSource, list[str]],
    runtime_note: str,
    situation_note: str,
    mode_guidance: str,
    thinking_requirement: str,
    current_entries: Sequence[DialoguePromptEntry],
    registry: MainCoreCommandRegistry,
    model_id: str,
    reference_map: Mapping[str, Any],
    message_reference_by_ledger_id: Mapping[int, str],
    current_message_ids: frozenset[int],
    working_message_ids: Mapping[ContextSource, Sequence[int]],
    working_summary_ids: Mapping[ContextSource, Sequence[int]],
    working_sequences: Mapping[ContextSource, Sequence[int]],
    working_item_refs: Mapping[ContextSource, Sequence[str]],
    working_background_load_orders: Mapping[ContextSource, Sequence[int]],
    summary_coverage: tuple[tuple[int, int, int], ...],
    has_searchable_earlier_history: bool,
    image_urls: Sequence[str],
    trim_reasons: list[str],
    dialogue_anchors: Sequence[bool],
    identity_catalog: Any | None,
    identity_scope: str,
    identity_catalog_text: str,
    trigger_reminders: tuple[str, ...],
    previous_context_message_ids: tuple[int, ...],
) -> BoundedPromptState:
    """Build one independent working copy without changing projection semantics."""

    return BoundedPromptState(
        persona=persona,
        main_core_style_prompts=main_core_style_prompts,
        story_style_prompts=story_style_prompts,
        background_enabled=background_enabled,
        world=world,
        current_time=current_time,
        runtime_note=runtime_note,
        situation_note=situation_note,
        mode_guidance=mode_guidance,
        thinking_requirement=thinking_requirement,
        current_lines=tuple(entry.text for entry in current_entries),
        current_line_message_ids=tuple(int(entry.ledger_message_id) for entry in current_entries),
        registry=registry,
        model_id=model_id,
        reference_map=reference_map,
        message_reference_by_ledger_id=message_reference_by_ledger_id,
        current_message_ids=current_message_ids if current_entries else frozenset(),
        previous_context_message_ids=previous_context_message_ids,
        cache_rebase_reasons=(),
        working_message_ids=_copy_source_lists(working_message_ids),
        working_summary_ids=_copy_source_lists(working_summary_ids),
        working_sequences=_copy_source_lists(working_sequences),
        working_item_refs=_copy_source_lists(working_item_refs),
        working_background_load_orders=_copy_source_lists(working_background_load_orders),
        summary_coverage=summary_coverage,
        has_searchable_earlier_history=bool(has_searchable_earlier_history),
        image_urls=image_urls,
        trim_reasons=trim_reasons,
        working=_copy_source_lists(groups),
        dialogue_flags=list(dialogue_anchors),
        identity_catalog=identity_catalog,
        identity_scope=identity_scope,
        identity_catalog_text=identity_catalog_text,
        trigger_reminders=trigger_reminders,
    )


DEFAULT_MAIN_CORE_STYLE_PROMPTS = MainCoreStylePrompts(
    relationship_context=DEFAULT_RELATIONSHIP_CONTEXT_PROMPT,
)
DEFAULT_STORY_STYLE_PROMPTS = StoryStylePrompts()


class RolePlayPromptCompiler(RolePlayPromptRenderingMixin, RolePlayReferenceMixin):
    def compile(
        self,
        *,
        persona: str,
        main_core_mode_prompts: MainCoreModePrompts = turn_prompt.DEFAULT_MAIN_CORE_MODE_PROMPTS,
        main_core_style_prompts: MainCoreStylePrompts = DEFAULT_MAIN_CORE_STYLE_PROMPTS,
        story_style_prompts: StoryStylePrompts = DEFAULT_STORY_STYLE_PROMPTS,
        role: ScopeConfig,
        state: CoreState,
        prepared_context: PreparedMainCoreContext | None,
        current_input: str,
        occurred_at: datetime,
        registry: MainCoreCommandRegistry,
        timezone_name: str = "",
        current_plan: str = "",
        thinking_policy: MainCoreThinkingPolicy = DEFAULT_THINKING_POLICY,
        model_id: str = "",
        runtime_notes: str = "",
        current_asset_ids: Sequence[str] = (),
        file_references: Mapping[str, Any] | None = None,
        image_urls: Sequence[str] = (),
        max_prompt_tokens: int | None = None,
        responsibility: MainCoreTurnResponsibility = DEFAULT_MESSAGE_RESPONSIBILITY,
        trigger_reminders: Sequence[str] = (),
        previous_context_message_ids: Sequence[int] = (),
    ) -> CompiledPrompt:
        items = list(prepared_context.compiled.items) if prepared_context is not None else []
        identity_context, identity_catalog, project_identity = self._prepared_identity_projection(
            prepared_context
        )
        refs = self._short_references(items, prepared_context, file_references or {})
        fallback_input = self._current_input_with_refs(current_input, current_asset_ids, refs)
        (
            groups,
            working_message_ids,
            working_summary_ids,
            working_sequences,
            working_item_refs,
            working_background_load_orders,
        ) = self._group_items_with_sources(items, refs)
        persona = project_identity(persona)
        main_core_style_prompts = project_main_core_styles(
            main_core_style_prompts, project_identity, registry
        )
        story_style_prompts = project_story_styles(story_style_prompts, project_identity)
        main_core_mode_prompts = self._project_mode_prompts(
            main_core_mode_prompts, project_identity
        )
        current_time, situation_note, mode_guidance = self._turn_guidance(
            occurred_at=occurred_at,
            timezone_name=timezone_name,
            responsibility=responsibility,
            items=items,
            main_core_mode_prompts=main_core_mode_prompts,
        )
        del state
        runtime_note = self._runtime_note(runtime_notes, project_identity, refs)
        plan = self._clean_model_text(project_identity(current_plan), refs)
        work_requirement = thinking_requirement(registry, plan, thinking_policy)
        reminders = turn_prompt.normalize_trigger_reminders(trigger_reminders, project_identity)
        current_entries = self._current_entries(
            responsibility,
            prepared_context,
            refs,
            fallback_input,
            occurred_at,
        )
        base_trim_reasons = (
            list(prepared_context.compiled.report.trim_steps) if prepared_context else []
        )
        identity_scope, identity_catalog_text = self._identity_catalog_fields(
            identity_context, identity_catalog, role
        )
        summary_coverage = self._visible_summary_coverage(prepared_context)
        previous_message_ids = self._positive_message_ids(previous_context_message_ids)
        return self._compile_bounded_document(
            persona=persona,
            main_core_style_prompts=main_core_style_prompts,
            story_style_prompts=story_style_prompts,
            background_enabled=bool(
                prepared_context is not None and prepared_context.background_enabled
            ),
            world="",
            current_time=current_time,
            groups=groups,
            runtime_note=runtime_note,
            situation_note=situation_note,
            mode_guidance=mode_guidance,
            thinking_requirement=work_requirement,
            current_entries=current_entries,
            registry=registry,
            model_id=model_id,
            reference_map=refs.public_to_internal,
            message_reference_by_ledger_id=refs.message_by_ledger_id,
            current_message_ids=self._current_message_ids(items, prepared_context),
            working_message_ids=working_message_ids,
            working_summary_ids=working_summary_ids,
            working_sequences=working_sequences,
            working_item_refs=working_item_refs,
            working_background_load_orders=working_background_load_orders,
            summary_coverage=summary_coverage,
            has_searchable_earlier_history=bool(
                prepared_context is not None and prepared_context.has_searchable_earlier_history
            ),
            image_urls=image_urls,
            max_prompt_tokens=max_prompt_tokens,
            trim_reasons=base_trim_reasons,
            dialogue_anchors=self._dialogue_anchor_flags(items, refs),
            identity_catalog=identity_catalog,
            identity_scope=identity_scope,
            identity_catalog_text=identity_catalog_text,
            trigger_reminders=reminders,
            previous_context_message_ids=previous_message_ids,
        )

    @staticmethod
    def _turn_guidance(
        *,
        occurred_at: datetime,
        timezone_name: str,
        responsibility: MainCoreTurnResponsibility,
        items: Sequence[Any],
        main_core_mode_prompts: MainCoreModePrompts,
    ) -> tuple[str, str, str]:
        current_time = turn_prompt.compose_current_time(
            label=full_datetime_label(occurred_at, timezone_name=timezone_name),
            responsibility=responsibility,
            items=items,
            occurred_at=occurred_at,
        )
        situation_note = turn_prompt.compose_situation(
            responsibility=responsibility,
            items=items,
            occurred_at=occurred_at,
        )
        mode_guidance = turn_prompt.compose_mode_guidance(
            responsibility=responsibility,
            prompts=main_core_mode_prompts,
        )
        return current_time, situation_note, mode_guidance

    @staticmethod
    def _prepared_identity_projection(
        prepared_context: PreparedMainCoreContext | None,
    ) -> tuple[Any | None, Any | None, Any]:
        identity_context = prepared_context.identity_context if prepared_context else None
        identity_catalog = prepared_context.identity_catalog if prepared_context else None
        return (
            identity_context,
            identity_catalog,
            partial(project_prepared_identity, prepared_context=prepared_context),
        )

    @staticmethod
    def _project_mode_prompts(
        prompts: MainCoreModePrompts, project_identity: Any
    ) -> MainCoreModePrompts:
        return MainCoreModePrompts(
            self_initiated=str(project_identity(prompts.self_initiated) or ""),
        )

    def _runtime_note(self, runtime_notes: str, project_identity: Any, refs: Any) -> str:
        cleaned = self._clean_model_text(project_identity(runtime_notes), refs)
        return cleaned if isinstance(cleaned, TrustedPromptMarkup) else xml_text(cleaned)

    def _current_entries(
        self,
        responsibility: MainCoreTurnResponsibility,
        prepared_context: PreparedMainCoreContext | None,
        refs: Any,
        fallback_input: str,
        occurred_at: datetime,
    ) -> tuple[DialoguePromptEntry, ...]:
        if not responsibility.has_current_message:
            return ()
        return self._current_turn_entries(
            prepared_context,
            refs,
            fallback=fallback_input,
            occurred_at=occurred_at,
        )

    @staticmethod
    def _identity_catalog_fields(
        identity_context: Any | None, identity_catalog: Any | None, role: ScopeConfig
    ) -> tuple[str, str]:
        scope = (
            str(identity_context.scope)
            if identity_context is not None
            else str(role.scope or "profile")
        )
        return scope, identity_catalog.prompt_text() if identity_catalog else ""

    @staticmethod
    def _visible_summary_coverage(
        prepared_context: PreparedMainCoreContext | None,
    ) -> tuple[tuple[int, int, int], ...]:
        if prepared_context is None:
            return ()
        return tuple(prepared_context.visible_summary_coverage)

    @staticmethod
    def _positive_message_ids(values: Sequence[int]) -> tuple[int, ...]:
        return tuple(dict.fromkeys(int(value) for value in values if int(value) > 0))

    def _compile_bounded_document(
        self,
        *,
        persona: str,
        main_core_style_prompts: MainCoreStylePrompts,
        story_style_prompts: StoryStylePrompts,
        background_enabled: bool,
        world: str,
        current_time: str,
        groups: Mapping[ContextSource, list[str]],
        runtime_note: str,
        situation_note: str,
        mode_guidance: str,
        thinking_requirement: str,
        current_entries: Sequence[DialoguePromptEntry],
        registry: MainCoreCommandRegistry,
        model_id: str,
        reference_map: Mapping[str, Any],
        message_reference_by_ledger_id: Mapping[int, str],
        current_message_ids: frozenset[int],
        working_message_ids: dict[ContextSource, list[int]],
        working_summary_ids: dict[ContextSource, list[int]],
        working_sequences: dict[ContextSource, list[int]],
        working_item_refs: dict[ContextSource, list[str]],
        working_background_load_orders: dict[ContextSource, list[int]],
        summary_coverage: tuple[tuple[int, int, int], ...],
        has_searchable_earlier_history: bool,
        image_urls: Sequence[str],
        max_prompt_tokens: int | None,
        trim_reasons: list[str],
        dialogue_anchors: Sequence[bool],
        identity_catalog: Any | None,
        identity_scope: str,
        identity_catalog_text: str,
        trigger_reminders: tuple[str, ...],
        previous_context_message_ids: tuple[int, ...],
    ) -> CompiledPrompt:
        state = build_bounded_prompt_state(
            persona=persona,
            main_core_style_prompts=main_core_style_prompts,
            story_style_prompts=story_style_prompts,
            background_enabled=background_enabled,
            world=world,
            current_time=current_time,
            runtime_note=runtime_note,
            situation_note=situation_note,
            mode_guidance=mode_guidance,
            thinking_requirement=thinking_requirement,
            current_entries=current_entries,
            registry=registry,
            model_id=model_id,
            reference_map=reference_map,
            message_reference_by_ledger_id=message_reference_by_ledger_id,
            current_message_ids=current_message_ids,
            previous_context_message_ids=previous_context_message_ids,
            working_message_ids=working_message_ids,
            working_summary_ids=working_summary_ids,
            working_sequences=working_sequences,
            working_item_refs=working_item_refs,
            working_background_load_orders=working_background_load_orders,
            summary_coverage=summary_coverage,
            has_searchable_earlier_history=has_searchable_earlier_history,
            image_urls=image_urls,
            trim_reasons=trim_reasons,
            groups=groups,
            dialogue_anchors=dialogue_anchors,
            identity_catalog=identity_catalog,
            identity_scope=identity_scope,
            identity_catalog_text=identity_catalog_text,
            trigger_reminders=trigger_reminders,
        )
        compiled = self._render_bounded_prompt(state)
        if max_prompt_tokens is None:
            return compiled
        maximum = int(max_prompt_tokens)
        if maximum < 1:
            raise ContextBudgetExceeded(
                "模型窗口没有为主交流文档留下可用空间",
                reason_code="main_prompt_budget_unavailable",
            )
        compiled = self._trim_bounded_sources(state, compiled, maximum)
        if compiled.total_tokens <= maximum:
            return compiled
        compiled = self._budget_persona_projection(state, compiled, maximum)
        if compiled.total_tokens <= maximum:
            return compiled
        if compiled.total_tokens > maximum:
            raise ContextBudgetExceeded(
                "核心角色、当前输入和指令协议超过模型窗口",
                reason_code="protected_main_prompt_exceeds_model_window",
            )
        return compiled

    def _budget_persona_projection(
        self,
        state: BoundedPromptState,
        compiled: CompiledPrompt,
        maximum: int,
    ) -> CompiledPrompt:
        return budget_persona_projection(
            state,
            compiled,
            maximum,
            self._render_bounded_prompt,
        )

    def _trim_bounded_sources(
        self,
        state: BoundedPromptState,
        compiled: CompiledPrompt,
        maximum: int,
    ) -> CompiledPrompt:
        return trim_bounded_sources(
            state,
            compiled,
            maximum,
            self._render_bounded_prompt,
        )

    @staticmethod
    def _drop_last_loaded_background_item(state: BoundedPromptState) -> bool:
        return drop_last_loaded_background_item(state)

    @staticmethod
    def _drop_bounded_item(state: BoundedPromptState, source: ContextSource) -> bool:
        return drop_bounded_item(state, source)

    def _render_bounded_prompt(self, state: BoundedPromptState) -> CompiledPrompt:
        return super()._render_bounded_prompt(state)

    def _bounded_context_blocks(self, state: BoundedPromptState) -> list[PromptBlock]:
        return super()._bounded_context_blocks(state)

    def _bounded_turn_blocks(self, state: BoundedPromptState) -> list[PromptBlock]:
        return super()._bounded_turn_blocks(state)

    def _dialogue_blocks(
        self, state: BoundedPromptState
    ) -> tuple[PromptBlock, PromptBlock, list[str]]:
        return super()._dialogue_blocks(state)

    @staticmethod
    def _attach_cache_boundaries_to_last_nonempty(
        blocks: Sequence[PromptBlock],
        boundaries: tuple[PromptCacheBoundary, ...],
    ) -> list[PromptBlock]:
        return RolePlayPromptRenderingMixin._attach_cache_boundaries_to_last_nonempty(
            blocks, boundaries
        )

    @staticmethod
    def _joined_item_end_offsets(values: Sequence[str]) -> tuple[int, ...]:
        return RolePlayPromptRenderingMixin._joined_item_end_offsets(values)

    @staticmethod
    def _current_message_ids(
        items: Sequence[Any],
        prepared_context: PreparedMainCoreContext | None,
    ) -> frozenset[int]:
        values = {
            int(item.metadata.get("ledger_message_id") or 0)
            for item in items
            if item.source is ContextSource.CURRENT_PLAYER_MESSAGE
        }
        values.update(
            int(row.get("ledger_message_id") or 0)
            for row in tuple(prepared_context.current_turn if prepared_context else ())
            if isinstance(row, Mapping)
        )
        return frozenset(value for value in values if value > 0)

    def project_conversation(
        self,
        prepared_context: PreparedMainCoreContext | None,
        *,
        fallback_input: str = "",
        occurred_at: datetime | None = None,
    ) -> SemanticConversationProjection:
        """Reuse the exact short-reference and cleaning layer outside Main Core."""

        items = list(prepared_context.compiled.items) if prepared_context is not None else []
        refs = self._short_references(items, prepared_context)
        groups = self._group_items(items, refs)
        fallback = ConversationContextService._current_player_text(fallback_input)
        current = self._current_turn_text(
            prepared_context,
            refs,
            fallback=fallback,
            occurred_at=occurred_at or datetime.now().astimezone(),
        )
        return SemanticConversationProjection(
            recent_lines=tuple(groups[ContextSource.CURRENT_DIALOGUE]),
            current_lines=tuple(line for line in current.splitlines() if line.strip()),
            public_to_internal=dict(refs.public_to_internal),
            internal_to_public=dict(refs.internal_to_public),
        )

    @staticmethod
    def _joined(values: Sequence[str]) -> str:
        return RolePlayPromptRenderingMixin._joined(values)

    def _background_block(
        self,
        state: BoundedPromptState,
        sources: ContextSource | Sequence[ContextSource],
    ) -> str:
        return super()._background_block(state, sources)

    def _recent_experience_block(self, state: BoundedPromptState) -> str:
        return super()._recent_experience_block(state)


__all__ = [
    "ExecutionRound",
    "RolePlayPromptCompiler",
]
