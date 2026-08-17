"""SoulCore-owned prompt budgeting and hard-limit enforcement.

This module deliberately has no AstrBot imports.  It is the stable application
boundary used by the AstrBot adapter, the message ledger and the diagnostics
page.  Provider-specific tokenizers can replace :class:`ConservativeTokenMeter`
without changing the compiler or guard.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol

from ...contracts.message_reference import (
    INBOUND_REPLY_REFERENCE_KIND,
    inbound_reply_projection,
)
from ...shared.prompt_document import (
    TrustedPromptMarkup,
    join_prompt_markup,
    prompt_markup_block,
    prompt_markup_record,
)
from ...shared.token_meter import (
    ConservativeTokenMeter,
    TokenCountMode,
    TokenMeasurement,
    TokenMeter,
)
from ._context_selection import (
    MIN_DIALOGUE_MESSAGES,
    _fit_chronological_history,
    _fit_dialogue_with_floor,
    _fit_newest,
    _fit_scored,
    _ordered,
)
from ._context_selection import (
    _truncate_string as _truncate_string,
)
from .dialogue_display import render_dialogue_line


class RequestBudgetGuard:
    """Enforce the hard limit before every provider request."""

    def __init__(self, meter: TokenMeter | None = None) -> None:
        self.meter = meter or ConservativeTokenMeter()

    def enforce(
        self,
        items: Sequence[ContextItem],
        *,
        effective_max_tokens: int,
        report: ContextBuildReport | None = None,
        custom_prompt_tokens: int = 0,
    ) -> CompiledContext:
        maximum = int(effective_max_tokens)
        custom_tokens = max(0, int(custom_prompt_tokens))
        if maximum < 1:
            raise ValueError("effective_max_tokens must be positive")
        working = _ordered(items)
        if report is None:
            initial = self.meter.measure(working)
            report = ContextBuildReport(
                max_context_tokens=maximum,
                target_context_tokens=maximum,
                fill_budget=maximum,
                effective_max_tokens=maximum,
                provider_limit_known=False,
                count_mode=initial.mode,
                model_id=initial.model_id,
                total_tokens=initial.tokens,
            )
        self._trim_to_maximum(working, max(0, maximum - custom_tokens), report)
        measurement = self._finish_report(
            working,
            report,
            custom_prompt_tokens=custom_tokens,
        )
        self._validate_measurement(
            working,
            maximum,
            report,
            measurement.tokens + custom_tokens,
            custom_prompt_tokens=custom_tokens,
        )
        return CompiledContext(tuple(working), report)

    def _trim_to_maximum(
        self,
        working: list[ContextItem],
        maximum: int,
        report: ContextBuildReport,
    ) -> None:
        def total() -> int:
            return self.meter.measure(working).tokens

        def drop(predicate: Any, step: str, *, limit: int | None = None) -> int:
            dropped = 0
            for item in list(working):
                if total() <= maximum:
                    break
                if predicate(item) and (limit is None or dropped < limit):
                    working.remove(item)
                    report.dropped_item_ids.append(item.item_id)
                    dropped += 1
            if dropped:
                report.trim_steps.append(f"{step}:{dropped}")
            return dropped

        # Tool/query results are elastic request data.  They must give way
        # before any material already admitted through FillBudget.
        if total() > maximum:
            self._drop_elastic_search_material(working, maximum, report)

        if total() > maximum:
            self._drop_background_materials(working, maximum, report)

        if total() > maximum:
            self._drop_superseded_history_summaries(working, maximum, report)

        source_steps = ((ContextSource.CHARACTER_INTENT, "drop_character_intents"),)
        for source, step in source_steps:
            if total() > maximum:
                drop(
                    lambda item, current=source: not item.protected and item.source is current,
                    step,
                )
        dialogue = [
            item
            for item in working
            if not item.protected
            and item.source is ContextSource.CURRENT_DIALOGUE
            and not bool(item.metadata.get("dialogue_anchor"))
        ]
        max_drop = min(
            math.floor(len(dialogue) * 2 / 3),
            max(0, len(dialogue) - MIN_DIALOGUE_MESSAGES),
        )
        if total() > maximum and max_drop:
            drop(
                lambda item: (
                    not item.protected
                    and item.source is ContextSource.CURRENT_DIALOGUE
                    and not bool(item.metadata.get("dialogue_anchor"))
                ),
                "drop_old_dialogue",
                limit=max_drop,
            )
        if total() > maximum:
            self._shorten_dialogue(working, maximum, report)
        if total() > maximum:
            self._downgrade_and_drop_history_fragments(
                working,
                maximum,
                report,
            )

    def _drop_superseded_history_summaries(
        self,
        working: list[ContextItem],
        maximum: int,
        report: ContextBuildReport,
    ) -> None:
        """Drop only redundant older summaries; the latest one closes a raw-history gap."""

        summaries = sorted(
            (
                item
                for item in working
                if not item.protected and item.source is ContextSource.HISTORY_SUMMARY
            ),
            key=lambda item: (item.sequence, item.item_id),
        )
        dropped = 0
        for item in summaries[:-1]:
            if self.meter.measure(working).tokens <= maximum:
                break
            working.remove(item)
            report.dropped_item_ids.append(item.item_id)
            dropped += 1
        if dropped:
            report.trim_steps.append(f"drop_superseded_history_summary:{dropped}")

    def _drop_background_materials(
        self,
        working: list[ContextItem],
        maximum: int,
        report: ContextBuildReport,
    ) -> None:
        background_items = [
            item
            for item in working
            if not item.protected and item.source in BACKGROUND_OPTIONAL_SOURCES
        ]
        if not background_items:
            return
        marked = all(
            isinstance(item.metadata.get("background_load_order"), int) for item in background_items
        )
        load_order = (
            sorted(
                background_items,
                key=lambda item: int(item.metadata["background_load_order"]),
            )
            if marked
            else _background_material_attempt_order(background_items)
        )
        dropped_by_source: dict[ContextSource, int] = {}
        for item in reversed(load_order):
            if self.meter.measure(working).tokens <= maximum:
                break
            if item not in working:
                continue
            working.remove(item)
            report.dropped_item_ids.append(item.item_id)
            dropped_by_source[item.source] = dropped_by_source.get(item.source, 0) + 1
        step_names = {
            ContextSource.BACKGROUND_LEFTOVER: "drop_background_leftovers",
            ContextSource.BACKGROUND_EXPERIENCE: "drop_background_experiences",
            ContextSource.BACKGROUND_KEYFRAME: "drop_background_keyframes",
            ContextSource.BACKGROUND_STORY: "drop_background_story_material",
            ContextSource.BACKGROUND_WORLD: "drop_background_world_material",
        }
        for source, count in dropped_by_source.items():
            report.trim_steps.append(f"{step_names[source]}:{count}")

    def _finish_report(
        self,
        working: list[ContextItem],
        report: ContextBuildReport,
        *,
        custom_prompt_tokens: int = 0,
    ) -> Any:
        measurement = self.meter.measure(working)
        report.total_tokens = measurement.tokens + custom_prompt_tokens
        report.count_mode = measurement.mode
        report.model_id = measurement.model_id
        report.selected_item_ids = [item.item_id for item in working]
        report.source_tokens.clear()
        for source in ContextSource:
            source_items = [item for item in working if item.source is source]
            if source_items:
                report.source_tokens[source.value] = self.meter.measure(source_items).tokens
        protected = [item for item in working if item.protected]
        report.protected_tokens = self.meter.measure(protected).tokens + custom_prompt_tokens
        report.custom_prompt_tokens = custom_prompt_tokens
        return measurement

    def _validate_measurement(
        self,
        working: list[ContextItem],
        maximum: int,
        report: ContextBuildReport,
        total: int,
        *,
        custom_prompt_tokens: int = 0,
    ) -> None:
        if total <= maximum:
            return
        reason_code, message = self._failure_reason(
            working,
            maximum,
            custom_prompt_tokens=custom_prompt_tokens,
        )
        report.trim_steps.append(reason_code)
        raise ContextBudgetExceeded(message, reason_code=reason_code, report=report)

    def _downgrade_and_drop_history_fragments(
        self,
        working: list[ContextItem],
        maximum: int,
        report: ContextBuildReport,
    ) -> None:
        """Compact and remove historical prose strictly from oldest to newest.

        Normal selection has already happened inside FillBudget.  This is only
        a hard-window fallback and must never reinterpret history by relevance,
        importance, keywords, or any other search signal.
        """

        ranked = sorted(
            (
                (index, item)
                for index, item in enumerate(working)
                if not item.protected and item.source is ContextSource.HISTORY_FRAGMENT
            ),
            key=lambda pair: (
                pair[1].sequence,
                pair[1].item_id,
            ),
        )
        downgraded = self._downgrade_ranked(working, ranked, maximum, report)
        if downgraded:
            report.trim_steps.append(f"downgrade_old_history_fragments:{downgraded}")
        removed = self._drop_ranked(working, ranked, maximum, report)
        if removed:
            report.trim_steps.append(f"drop_old_history_fragments:{removed}")

    def _downgrade_ranked(
        self,
        working: list[ContextItem],
        ranked: list[tuple[int, ContextItem]],
        maximum: int,
        report: ContextBuildReport,
    ) -> int:
        downgraded = 0
        for index, item in ranked:
            if self.meter.measure(working).tokens <= maximum:
                break
            compact = item.metadata.get("compact_content")
            if not isinstance(compact, str) or not compact.strip():
                continue
            if str(item.body) == compact:
                continue
            working[index] = replace(
                item,
                body=compact,
                metadata={**dict(item.metadata), "render_level": "compact"},
            )
            report.shortened_item_ids.append(item.item_id)
            downgraded += 1
        return downgraded

    def _drop_ranked(
        self,
        working: list[ContextItem],
        ranked: list[tuple[int, ContextItem]],
        maximum: int,
        report: ContextBuildReport,
    ) -> int:
        removed = 0
        for _, original in ranked:
            if self.meter.measure(working).tokens <= maximum:
                break
            index = next(
                (
                    idx
                    for idx, candidate in enumerate(working)
                    if candidate.item_id == original.item_id
                ),
                None,
            )
            if index is None:
                continue
            working.pop(index)
            report.dropped_item_ids.append(original.item_id)
            removed += 1
        return removed

    def _shorten_dialogue(
        self,
        working: list[ContextItem],
        maximum: int,
        report: ContextBuildReport,
    ) -> None:
        # The active player prompt is a separate protected SYSTEM item, so all
        # retained historical dialogue items may be shortened when necessary.
        candidates = [
            index
            for index, item in enumerate(working)
            if not item.protected
            and item.source is ContextSource.CURRENT_DIALOGUE
            and not bool(item.metadata.get("dialogue_anchor"))
            and isinstance(item.body, str)
        ]
        shortened_now = 0
        for index in candidates:
            current_total = self.meter.measure(working).tokens
            if current_total <= maximum:
                return
            item = working[index]
            current_cost = self.meter.measure_item(item).tokens
            needed = current_total - maximum
            # Leave an extra role/protocol margin.  The fallback meter's item
            # cost includes more than just content, so using the exact deficit
            # as a content target can otherwise miss the hard limit by a few
            # tokens.
            target = max(
                0,
                current_cost - needed - ConservativeTokenMeter.MESSAGE_OVERHEAD - 8,
            )
            fallback = (
                self.meter
                if isinstance(self.meter, ConservativeTokenMeter)
                else ConservativeTokenMeter()
            )
            shortened = _truncate_string(str(item.body), target, fallback)
            if shortened != item.body:
                working[index] = replace(item, body=shortened)
                report.shortened_item_ids.append(item.item_id)
                shortened_now += 1
        if shortened_now:
            report.trim_steps.append(f"shorten_retained_dialogue:{shortened_now}")

    def _drop_elastic_search_material(
        self,
        working: list[ContextItem],
        maximum: int,
        report: ContextBuildReport,
    ) -> None:
        units = self._elastic_units(working)
        shortened = self._shorten_elastic_units(working, units, maximum, report)
        if shortened:
            report.trim_steps.append(f"shorten_search_material:{shortened}")
        removed = self._drop_old_elastic_units(working, units, maximum, report)
        if removed:
            report.trim_steps.append(f"drop_old_search_material:{removed}")

    @staticmethod
    def _elastic_units(working: list[ContextItem]) -> list[list[ContextItem]]:
        elastic = {ContextSource.AI_SEARCH}
        units: list[list[ContextItem]] = []
        seen_pairs: set[str] = set()
        for item in working:
            if item.protected or item.source not in elastic:
                continue
            if item.cohesion_key:
                if item.cohesion_key in seen_pairs:
                    continue
                seen_pairs.add(item.cohesion_key)
                group = [
                    candidate
                    for candidate in working
                    if candidate.cohesion_key == item.cohesion_key
                ]
                if any(candidate.protected for candidate in group):
                    continue
                units.append(group)
            else:
                units.append([item])

        return units

    def _shorten_elastic_units(
        self,
        working: list[ContextItem],
        units: list[list[ContextItem]],
        maximum: int,
        report: ContextBuildReport,
    ) -> int:
        shortened = 0
        for unit in units:
            if self.meter.measure(working).tokens <= maximum:
                break
            for original in unit:
                if original.source is not ContextSource.AI_SEARCH or not isinstance(
                    original.body, str
                ):
                    continue
                current_total = self.meter.measure(working).tokens
                if current_total <= maximum:
                    break
                index = next(
                    (
                        idx
                        for idx, candidate in enumerate(working)
                        if candidate.item_id == original.item_id
                    ),
                    None,
                )
                if index is None:
                    continue
                item = working[index]
                fallback = (
                    self.meter
                    if isinstance(self.meter, ConservativeTokenMeter)
                    else ConservativeTokenMeter()
                )
                content_tokens = fallback.count_text(item.body)
                target = max(0, content_tokens - (current_total - maximum) - 1)
                body = _truncate_string(item.body, target, fallback)
                if body != item.body:
                    working[index] = replace(item, body=body)
                    report.shortened_item_ids.append(item.item_id)
                    shortened += 1
        return shortened

    def _drop_old_elastic_units(
        self,
        working: list[ContextItem],
        units: list[list[ContextItem]],
        maximum: int,
        report: ContextBuildReport,
    ) -> int:
        cohesive_units = [unit for unit in units if self._unit_cohesion_key(unit) is not None]
        newest_cohesion_key = (
            self._unit_cohesion_key(cohesive_units[-1]) if cohesive_units else None
        )
        removed = 0
        for unit in units:
            if self.meter.measure(working).tokens <= maximum:
                break
            cohesion_key = self._unit_cohesion_key(unit)
            if cohesion_key is not None and cohesion_key == newest_cohesion_key:
                continue
            ids = {item.item_id for item in unit}
            before = len(working)
            working[:] = [item for item in working if item.item_id not in ids]
            removed += before - len(working)
            report.dropped_item_ids.extend(sorted(ids))
        return removed

    @staticmethod
    def _unit_cohesion_key(unit: list[ContextItem]) -> str | None:
        for item in unit:
            if item.cohesion_key:
                return item.cohesion_key
        return None

    def _failure_reason(
        self,
        working: Sequence[ContextItem],
        maximum: int,
        *,
        custom_prompt_tokens: int = 0,
    ) -> tuple[str, str]:
        protected = [item for item in working if item.protected]
        protected_tokens = self.meter.measure(protected).tokens + custom_prompt_tokens
        if protected_tokens > maximum:
            return (
                "protected_content_exceeds_hard_limit",
                "SoulCore protected context exceeds the effective MaxToken; "
                "increase max_context_tokens or use a model with a larger context window",
            )

        dialogue_failure = self._dialogue_floor_failure(
            working,
            protected,
            maximum,
            custom_prompt_tokens=custom_prompt_tokens,
        )
        if dialogue_failure is not None:
            return dialogue_failure
        remaining_data = sorted({item.source.value for item in working if not item.protected})
        return (
            "unhandled_data_source_exceeds_hard_limit",
            "Context data still exceeds MaxToken after all permitted trimming; "
            f"remaining data sources: {', '.join(remaining_data) or 'unknown'}",
        )

    def _dialogue_floor_failure(
        self,
        working: Sequence[ContextItem],
        protected: list[ContextItem],
        maximum: int,
        *,
        custom_prompt_tokens: int = 0,
    ) -> tuple[str, str] | None:
        dialogue = [
            item
            for item in working
            if item.source is ContextSource.CURRENT_DIALOGUE and not item.protected
        ]
        if dialogue:
            minimum = [
                replace(item, body="") if isinstance(item.body, str) else item
                for item in dialogue[-min(MIN_DIALOGUE_MESSAGES, len(dialogue)) :]
            ]
            minimum_tokens = (
                self.meter.measure([*protected, *minimum]).tokens + custom_prompt_tokens
            )
            other_data = [
                item
                for item in working
                if not item.protected and item.source is not ContextSource.CURRENT_DIALOGUE
            ]
            if minimum_tokens > maximum or (
                len(dialogue) <= MIN_DIALOGUE_MESSAGES and not other_data
            ):
                return (
                    "minimum_dialogue_floor_exceeds_hard_limit",
                    "The protected context plus SoulCore's minimum recent-dialogue "
                    "structures cannot fit MaxToken; increase the configured limit",
                )
        return None


class ContextProjector(Protocol):
    def project(
        self,
        components: Sequence[object],
        *,
        supported_modalities: Iterable[str] = (),
    ) -> ProjectionResult: ...


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    content: Any
    components: tuple[Mapping[str, Any], ...]
    degraded: bool
    degradation_notes: tuple[str, ...] = ()


class DefaultContextProjector:
    """Project already normalized SoulCore component mappings."""

    _TEXT_KINDS = {"plain", "text"}
    _ALIASES = {"record": "audio", "voice": "audio"}

    def project(
        self,
        components: Sequence[Mapping[str, Any]],
        *,
        supported_modalities: Iterable[str] = (),
    ) -> ProjectionResult:
        supported = {str(value).lower() for value in supported_modalities}
        normalized: list[Mapping[str, Any]] = []
        content: list[Any] = []
        notes: list[str] = []
        degraded = False
        for raw in components:
            component = self._normalize(raw)
            normalized.append(component)
            kind = str(component.get("type", "unknown")).lower()
            canonical = self._ALIASES.get(kind, kind)
            if canonical in self._TEXT_KINDS:
                content.append(str(component.get("text", "")))
            elif canonical in supported:
                content.append(dict(component))
            else:
                degraded = True
                placeholder = self._placeholder(canonical, component)
                content.append(placeholder)
                notes.append(f"unsupported_{canonical}")
        if all(isinstance(item, str) for item in content):
            projected: Any = "\n".join(item for item in content if item)
        else:
            projected = content
        return ProjectionResult(projected, tuple(normalized), degraded, tuple(notes))

    @staticmethod
    def _normalize(raw: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(raw, Mapping):
            raise TypeError("context components must be normalized mappings")
        kind = str(raw.get("type") or "unknown").lower()
        result = {str(key): value for key, value in raw.items()}
        result["type"] = kind
        return result

    def _placeholder(self, kind: str, component: Mapping[str, Any]) -> str:
        if kind == INBOUND_REPLY_REFERENCE_KIND:
            return inbound_reply_projection((component,))
        labels = {
            "image": "对方发送了一张图片",
            "audio": "对方发送了一条语音",
            "file": "对方发送了一个文件",
            "video": "对方发送了一段视频",
        }
        label = labels.get(kind)
        if label is None:
            return ""
        # Only file attachments have a user-visible filename. Image, audio and
        # video components can contain adapter-generated transport labels;
        # projecting those labels would leak implementation identifiers into
        # the dialogue timeline.
        name = (component.get("name") or component.get("file")) if kind == "file" else ""
        return f"[{label}{f'：{name}' if name else ''}]"


class SummaryStrategy(Protocol):
    strategy_id: str
    version: int

    def build_prompt(self, messages: Sequence[ContextItem], token_limit: int) -> str: ...


class DialogueSummaryStrategy:
    """Build the dynamic source material for one cumulative dialogue compaction."""

    strategy_id = "dialogue_summary"
    version = 5

    def build_prompt(
        self,
        messages: Sequence[ContextItem],
        token_limit: int,
        *,
        previous_summary_text: str = "",
    ) -> TrustedPromptMarkup:
        del token_limit
        prior = prompt_markup_block(
            "上一版累计摘要",
            previous_summary_text.strip() or "（尚无累计摘要）",
        )
        dialogue = prompt_markup_block(
            "本次新增对话",
            join_prompt_markup(
                prompt_markup_record(
                    "消息",
                    (("内容", line),),
                )
                for line in _render_dialogue_line_values(messages)
                if line.strip()
            ),
        )
        return join_prompt_markup((prior, dialogue))


def _render_dialogue_line_values(messages: Sequence[ContextItem]) -> list[str]:
    counters = {"A": 0, "U": 0}
    lines: list[str] = []
    for item in _ordered(messages):
        if item.metadata.get("timeline_event_kind"):
            lines.append(
                render_dialogue_line(
                    item.body,
                    occurred_at=item.metadata.get("occurred_at"),
                )
            )
            continue
        prefix = "A" if str(item.speaker).lower() == "assistant" else "U"
        counters[prefix] += 1
        message_ref = f"{prefix}{counters[prefix]}"
        participant = str(item.metadata.get("participant_ref") or "").strip()
        if not participant and prefix == "A":
            participant = "C"
        name = str(item.metadata.get("sender_name") or "").strip()
        lines.append(
            render_dialogue_line(
                item.body,
                occurred_at=item.metadata.get("occurred_at"),
                message_ref=message_ref,
                participant_ref=participant,
                display_name=name,
            )
        )
    return lines


MIN_MAX_CONTEXT_TOKENS = 128_000
MIN_TARGET_CONTEXT_TOKENS = 20_000
DEFAULT_MAX_CONTEXT_TOKENS = 128_000
DEFAULT_TARGET_CONTEXT_TOKENS = 64_000
FILL_RATIO = 0.70
SUMMARY_OUTPUT_FILL_RATIO = 0.075
CURRENT_DIALOGUE_FILL_WEIGHT = 0.25


class BudgetClass(StrEnum):
    """Budget semantics, independent from the eventual LLM message role."""

    SYSTEM = "SYSTEM"
    DATA = "DATA"


class ContextSource(StrEnum):
    CURRENT_PLAYER_MESSAGE = "current_player_message"
    ROLE_LIFE_DIRECTION = "role_life_direction"
    ROLE_STATE = "role_state"
    ROLE_LATEST_EXPERIENCE = "role_latest_experience"
    CURRENT_DIALOGUE = "current_dialogue"
    PLAYER_PROFILE = "player_profile"
    BACKGROUND_WORLD = "background_world"
    BACKGROUND_EXPERIENCE = "background_experience"
    BACKGROUND_KEYFRAME = "background_keyframe"
    BACKGROUND_LEFTOVER = "background_leftover"
    BACKGROUND_STORY = "background_story"
    HISTORY_SUMMARY = "history_summary"
    HISTORY_FRAGMENT = "history_fragment"
    CURRENT_WEB_RESOURCE = "current_web_resource"
    STICKER = "sticker"
    CHARACTER_INTENT = "character_intent"
    AI_SEARCH = "ai_search"
    OTHER = "other"


class ContextBudgetError(RuntimeError):
    """Base class for a context request that cannot be safely constructed."""


class ContextBudgetExceeded(ContextBudgetError):
    """Protected request content alone cannot fit the effective hard limit."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "context_budget_exceeded",
        report: ContextBuildReport | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.report = report


@dataclass(frozen=True, slots=True)
class ContextBudgetConfig:
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    target_context_tokens: int = DEFAULT_TARGET_CONTEXT_TOKENS
    fill_ratio: float = FILL_RATIO

    def normalized(self) -> ContextBudgetConfig:
        maximum = int(self.max_context_tokens)
        target = int(self.target_context_tokens)
        if isinstance(self.fill_ratio, bool):
            raise ValueError("fill_ratio must be a numeric fraction, not a boolean")
        ratio = float(self.fill_ratio)
        if maximum < MIN_MAX_CONTEXT_TOKENS:
            raise ValueError(f"max_context_tokens must be at least {MIN_MAX_CONTEXT_TOKENS}")
        if target < MIN_TARGET_CONTEXT_TOKENS:
            raise ValueError(f"target_context_tokens must be at least {MIN_TARGET_CONTEXT_TOKENS}")
        if not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("fill_ratio must be a finite value greater than 0 and at most 1")
        return ContextBudgetConfig(maximum, min(target, maximum), ratio)

    @property
    def fill_budget(self) -> int:
        normalized = self.normalized()
        return math.floor(normalized.target_context_tokens * normalized.fill_ratio)

    @property
    def summary_output_limit(self) -> int:
        return math.floor(self.fill_budget * SUMMARY_OUTPUT_FILL_RATIO)

    @property
    def summary_trigger_tokens(self) -> int:
        return math.floor(self.fill_budget * CURRENT_DIALOGUE_FILL_WEIGHT)

    def effective_max(self, provider_context_limit: int | None = None) -> int:
        normalized = self.normalized()
        if provider_context_limit is None:
            return normalized.max_context_tokens
        provider_limit = int(provider_context_limit)
        if provider_limit < MIN_MAX_CONTEXT_TOKENS:
            raise ValueError(
                "provider context window is below SoulCore's minimum safe limit "
                f"({provider_limit} < {MIN_MAX_CONTEXT_TOKENS})"
            )
        return min(normalized.max_context_tokens, provider_limit)


@dataclass(frozen=True, slots=True)
class ContextItem:
    """One independently budgeted SoulCore prompt fragment.

    ``budget_class`` controls trimming. ``speaker`` records the dialogue
    participant without pretending this fragment is a Provider message.
    ``cohesion_key`` keeps related fragments indivisible during trimming.
    ``sequence`` is monotonically increasing within one conversation; larger
    values are newer.
    """

    item_id: str
    budget_class: BudgetClass
    source: ContextSource
    speaker: str
    body: Any
    sequence: int = 0
    cohesion_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def protected(self) -> bool:
        return self.budget_class is BudgetClass.SYSTEM


@dataclass(frozen=True, slots=True)
class SourcePolicy:
    source: ContextSource
    fill_weight: float
    priority: int

    def __post_init__(self) -> None:
        weight = float(self.fill_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("fill_weight must be a finite positive number")
        object.__setattr__(self, "fill_weight", weight)

    def normalized_limit(self, available_fill: int, total_weight: float) -> int:
        if available_fill <= 0:
            return 0
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("active source fill weights must have a positive sum")
        return math.floor(max(0, int(available_fill)) * self.fill_weight / total_weight)


DEFAULT_SOURCE_POLICIES = (
    # The latest cumulative summary replaces the raw messages through its
    # coverage boundary, so it must be admitted before any still-raw dialogue.
    SourcePolicy(ContextSource.HISTORY_SUMMARY, 0.075, 5),
    SourcePolicy(ContextSource.CURRENT_DIALOGUE, CURRENT_DIALOGUE_FILL_WEIGHT, 10),
    SourcePolicy(ContextSource.PLAYER_PROFILE, 0.05, 15),
    # Compressed Memory prose is ordinary chronological history.  It is not a
    # semantic-search result and therefore owns the former 0.175 fill share.
    SourcePolicy(ContextSource.HISTORY_FRAGMENT, 0.175, 30),
    SourcePolicy(ContextSource.CURRENT_WEB_RESOURCE, 0.10, 35),
    SourcePolicy(ContextSource.STICKER, 0.05, 37),
    SourcePolicy(ContextSource.CHARACTER_INTENT, 0.05, 41),
)

BACKGROUND_FILL_WEIGHT = 0.25
BACKGROUND_FILL_PRIORITY = 18
BACKGROUND_OPTIONAL_SOURCES = frozenset(
    {
        ContextSource.BACKGROUND_WORLD,
        ContextSource.BACKGROUND_EXPERIENCE,
        ContextSource.BACKGROUND_KEYFRAME,
        ContextSource.BACKGROUND_LEFTOVER,
        ContextSource.BACKGROUND_STORY,
    }
)
BACKGROUND_EXPERIENCE_SOURCES = frozenset(
    {
        ContextSource.BACKGROUND_EXPERIENCE,
        ContextSource.BACKGROUND_KEYFRAME,
    }
)
MAX_OPTIONAL_BACKGROUND_EXPERIENCES = 5
MAX_VISIBLE_BACKGROUND_STORIES = 2


@dataclass(slots=True)
class _BackgroundMaterialFitter:
    meter: TokenMeter
    token_limit: int
    queues: dict[ContextSource, list[ContextItem]]
    kept: list[ContextItem] = field(default_factory=list)
    closed: set[ContextSource] = field(default_factory=set)
    used: int = 0
    experience_count: int = 0
    story_count: int = 0

    def try_item(self, item: ContextItem) -> bool:
        if item.source in self.closed:
            return False
        if (
            item.source in BACKGROUND_EXPERIENCE_SOURCES
            and self.experience_count >= MAX_OPTIONAL_BACKGROUND_EXPERIENCES
        ):
            return False
        cost = self.meter.measure_item(item).tokens
        if self.used + cost > self.token_limit:
            if item.source is not ContextSource.BACKGROUND_STORY:
                self.closed.add(item.source)
            return False
        metadata = dict(item.metadata)
        metadata["background_load_order"] = len(self.kept)
        self.kept.append(replace(item, metadata=metadata))
        self.used += cost
        if item.source in BACKGROUND_EXPERIENCE_SOURCES:
            self.experience_count += 1
        elif item.source is ContextSource.BACKGROUND_STORY:
            self.story_count += 1
        return True

    def fill_story_slot(self) -> None:
        if self.story_count >= MAX_VISIBLE_BACKGROUND_STORIES:
            return
        story_queue = self.queues[ContextSource.BACKGROUND_STORY]
        while story_queue:
            if self.try_item(story_queue.pop(0)):
                return


def _background_material_attempt_order(
    candidates: Sequence[ContextItem],
) -> list[ContextItem]:
    """Return the deterministic order in which complete optional works are tried."""

    queues = {
        source: sorted(
            (item for item in candidates if item.source is source),
            key=lambda item: (item.sequence, item.item_id),
        )
        for source in BACKGROUND_OPTIONAL_SOURCES
    }
    ordered: list[ContextItem] = []
    for source in (ContextSource.BACKGROUND_WORLD, ContextSource.BACKGROUND_STORY):
        if queues[source]:
            ordered.append(queues[source].pop(0))

    rotating = (
        ContextSource.BACKGROUND_LEFTOVER,
        ContextSource.BACKGROUND_EXPERIENCE,
        ContextSource.BACKGROUND_KEYFRAME,
    )
    while any(queues[source] for source in rotating):
        for source in rotating:
            if queues[source]:
                ordered.append(queues[source].pop(0))

    for source in (ContextSource.BACKGROUND_WORLD, ContextSource.BACKGROUND_STORY):
        ordered.extend(queues[source])
    return ordered


@dataclass(slots=True)
class ContextBuildReport:
    max_context_tokens: int
    target_context_tokens: int
    fill_budget: int
    effective_max_tokens: int
    provider_limit_known: bool
    count_mode: TokenCountMode
    fill_ratio: float = FILL_RATIO
    provider_selection_deferred: bool = False
    model_id: str = ""
    total_tokens: int = 0
    protected_tokens: int = 0
    custom_prompt_tokens: int = 0
    source_weights: dict[str, float] = field(default_factory=dict)
    source_limits: dict[str, int] = field(default_factory=dict)
    source_tokens: dict[str, int] = field(default_factory=dict)
    selected_item_ids: list[str] = field(default_factory=list)
    dropped_item_ids: list[str] = field(default_factory=list)
    shortened_item_ids: list[str] = field(default_factory=list)
    trim_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CompiledContext:
    items: tuple[ContextItem, ...]
    report: ContextBuildReport


def _policy_candidates(
    items: Sequence[ContextItem], policies: Sequence[SourcePolicy]
) -> dict[ContextSource, list[ContextItem]]:
    candidates = [item for item in items if not item.protected]
    return {
        policy.source: [item for item in candidates if item.source is policy.source]
        for policy in policies
    }


def _background_candidates(items: Sequence[ContextItem]) -> list[ContextItem]:
    return [
        item for item in items if not item.protected and item.source in BACKGROUND_OPTIONAL_SOURCES
    ]


def _active_source_weight(
    policies: Sequence[SourcePolicy],
    candidates: Mapping[ContextSource, Sequence[ContextItem]],
    *,
    include_background: bool,
) -> float:
    weight = sum(policy.fill_weight for policy in policies if candidates[policy.source])
    return weight + (BACKGROUND_FILL_WEIGHT if include_background else 0.0)


def _elastic_items(
    items: Sequence[ContextItem],
    policy_sources: set[ContextSource],
) -> list[ContextItem]:
    return [item for item in items if not item.protected and item.source not in policy_sources]


def _source_selection_entries(
    policies: Sequence[SourcePolicy],
    *,
    include_background: bool,
) -> list[tuple[int, SourcePolicy | None]]:
    entries: list[tuple[int, SourcePolicy | None]] = [
        (policy.priority, policy) for policy in policies
    ]
    if include_background:
        entries.append((BACKGROUND_FILL_PRIORITY, None))
    return sorted(
        entries,
        key=lambda entry: (
            entry[0],
            (entry[1].source.value if entry[1] is not None else "background_material"),
        ),
    )


@dataclass(frozen=True, slots=True)
class _SourceFit:
    key: str
    fitted: tuple[ContextItem, ...]
    dropped: tuple[ContextItem, ...]
    shortened: tuple[str, ...] = ()
    dialogue_floor_exceeds_limit: bool = False


class ContextCompiler:
    """Fill normal context before the first model request.

    Active source weights reserve the first allocation round.  Any room that an
    active source cannot use is then lent to still-truncated sources in priority
    order, so weights cannot strand an otherwise usable FillBudget.  Elastic
    command/search data is not assigned a normal-fill weight and is checked
    later by :class:`RequestBudgetGuard` against the hard limit.
    """

    def __init__(
        self,
        config: ContextBudgetConfig | None = None,
        meter: TokenMeter | None = None,
        policies: Sequence[SourcePolicy] = DEFAULT_SOURCE_POLICIES,
    ) -> None:
        self.config = (config or ContextBudgetConfig()).normalized()
        self.meter = meter or ConservativeTokenMeter()
        self.policies = tuple(policies)

    def compile(
        self,
        items: Sequence[ContextItem],
        *,
        provider_context_limit: int | None = None,
        defer_provider_selection: bool = False,
        custom_prompt_text: str = "",
    ) -> CompiledContext:
        effective_provider_limit = None if defer_provider_selection else provider_context_limit
        effective_max = self.config.effective_max(effective_provider_limit)
        report = ContextBuildReport(
            max_context_tokens=self.config.max_context_tokens,
            target_context_tokens=self.config.target_context_tokens,
            fill_budget=self.config.fill_budget,
            effective_max_tokens=effective_max,
            provider_limit_known=(
                provider_context_limit is not None and not defer_provider_selection
            ),
            count_mode=TokenCountMode.ESTIMATED,
            fill_ratio=self.config.fill_ratio,
            provider_selection_deferred=bool(defer_provider_selection),
        )
        if provider_context_limit is None and not defer_provider_selection:
            report.warnings.append("provider_context_window_unknown")

        custom_prompt_tokens = self.meter.count_text(str(custom_prompt_text or ""))
        report.custom_prompt_tokens = custom_prompt_tokens
        selected, protected_measurement = self._select_sources(
            items,
            report,
            custom_prompt_tokens=custom_prompt_tokens,
        )
        selected = _ordered(selected)
        measurement = self.meter.measure(selected)
        self._finalize_report(
            report,
            selected,
            measurement,
            protected_measurement,
            custom_prompt_tokens=custom_prompt_tokens,
        )
        guard = RequestBudgetGuard(self.meter)
        return guard.enforce(
            selected,
            effective_max_tokens=effective_max,
            report=report,
            custom_prompt_tokens=custom_prompt_tokens,
        )

    def _select_sources(
        self,
        items: Sequence[ContextItem],
        report: ContextBuildReport,
        *,
        custom_prompt_tokens: int = 0,
    ) -> tuple[list[ContextItem], TokenMeasurement]:
        protected = [item for item in items if item.protected]
        selected: list[ContextItem] = list(protected)
        protected_measurement = self.meter.measure(protected)
        available_fill = max(
            0,
            self.config.fill_budget - protected_measurement.tokens - custom_prompt_tokens,
        )
        remaining_fill = available_fill
        policy_sources = {policy.source for policy in self.policies}
        ordered_policies = sorted(self.policies, key=lambda item: item.priority)
        candidates_by_source = _policy_candidates(items, ordered_policies)
        background_candidates = _background_candidates(items)
        total_weight = _active_source_weight(
            ordered_policies,
            candidates_by_source,
            include_background=bool(background_candidates),
        )

        entries = _source_selection_entries(
            ordered_policies,
            include_background=bool(background_candidates),
        )
        source_fits, remaining_fill = self._initial_source_fits(
            entries=entries,
            candidates_by_source=candidates_by_source,
            background_candidates=background_candidates,
            available_fill=available_fill,
            remaining_fill=remaining_fill,
            total_weight=total_weight,
            report=report,
        )
        self._borrow_unused_fill(
            source_fits=source_fits,
            candidates_by_source=candidates_by_source,
            background_candidates=background_candidates,
            available_fill=available_fill,
            remaining_fill=remaining_fill,
            total_weight=total_weight,
            report=report,
        )
        self._record_source_fits(selected, source_fits, report)

        # Runtime-elastic sources (for example fresh search material) are not
        # allowed to borrow normal-fill shares, but are retained for the hard
        # guard to assess against the effective maximum.
        policy_sources.update(BACKGROUND_OPTIONAL_SOURCES)
        selected.extend(_elastic_items(items, policy_sources))
        return selected, protected_measurement

    def _initial_source_fits(
        self,
        *,
        entries: Sequence[tuple[int, SourcePolicy | None]],
        candidates_by_source: Mapping[ContextSource, list[ContextItem]],
        background_candidates: Sequence[ContextItem],
        available_fill: int,
        remaining_fill: int,
        total_weight: float,
        report: ContextBuildReport,
    ) -> tuple[list[tuple[SourcePolicy | None, _SourceFit]], int]:
        source_fits: list[tuple[SourcePolicy | None, _SourceFit]] = []
        for _priority, policy in entries:
            fit = self._fit_source_entry(
                policy=policy,
                candidates_by_source=candidates_by_source,
                background_candidates=background_candidates,
                available_fill=available_fill,
                remaining_fill=remaining_fill,
                total_weight=total_weight,
                report=report,
            )
            source_fits.append((policy, fit))
            remaining_fill = max(0, remaining_fill - self._fit_tokens(fit))
        return source_fits, remaining_fill

    def _borrow_unused_fill(
        self,
        *,
        source_fits: list[tuple[SourcePolicy | None, _SourceFit]],
        candidates_by_source: Mapping[ContextSource, list[ContextItem]],
        background_candidates: Sequence[ContextItem],
        available_fill: int,
        remaining_fill: int,
        total_weight: float,
        report: ContextBuildReport,
    ) -> None:
        # Weights reserve a fair first round; they are not independent hard
        # ceilings. Re-fit shortened sources when another source leaves room.
        for index, (policy, current) in enumerate(source_fits):
            if remaining_fill <= 0:
                break
            if not current.dropped and not current.shortened:
                continue
            current_tokens = self._fit_tokens(current)
            allowed = current_tokens + remaining_fill
            expanded = self._fit_source_entry(
                policy=policy,
                candidates_by_source=candidates_by_source,
                background_candidates=background_candidates,
                available_fill=available_fill,
                remaining_fill=remaining_fill,
                total_weight=total_weight,
                report=report,
                allowed_override=allowed,
            )
            expanded_tokens = self._fit_tokens(expanded)
            if expanded_tokens <= current_tokens or expanded_tokens > allowed:
                continue
            borrowed = expanded_tokens - current_tokens
            source_fits[index] = (policy, expanded)
            remaining_fill -= borrowed
            report.source_limits[expanded.key] = max(
                report.source_limits.get(expanded.key, 0), expanded_tokens
            )
            report.trim_steps.append(f"borrow_unused_fill:{expanded.key}:{borrowed}")

    @staticmethod
    def _record_source_fits(
        selected: list[ContextItem],
        source_fits: Sequence[tuple[SourcePolicy | None, _SourceFit]],
        report: ContextBuildReport,
    ) -> None:
        for _policy, fit in source_fits:
            selected.extend(fit.fitted)
            report.dropped_item_ids.extend(item.item_id for item in fit.dropped)
            report.shortened_item_ids.extend(fit.shortened)
            if fit.shortened:
                step = (
                    "compact_history_to_source_share"
                    if fit.key == ContextSource.HISTORY_FRAGMENT.value
                    else "shorten_dialogue_to_source_share"
                )
                report.trim_steps.append(f"{step}:{len(fit.shortened)}")
            if fit.dialogue_floor_exceeds_limit:
                report.warnings.append("dialogue_floor_exceeds_source_share")

    def _fit_source_entry(
        self,
        *,
        policy: SourcePolicy | None,
        candidates_by_source: Mapping[ContextSource, list[ContextItem]],
        background_candidates: Sequence[ContextItem],
        available_fill: int,
        remaining_fill: int,
        total_weight: float,
        report: ContextBuildReport,
        allowed_override: int | None = None,
    ) -> _SourceFit:
        if policy is None:
            key = "background_material"
            weight = BACKGROUND_FILL_WEIGHT
            candidates = background_candidates
        else:
            key = policy.source.value
            weight = policy.fill_weight
            candidates = candidates_by_source[policy.source]
        limit = self._weighted_limit(
            available_fill,
            weight if candidates else 0.0,
            total_weight,
        )
        report.source_weights[key] = weight
        report.source_limits[key] = limit
        allowed = (
            min(limit, remaining_fill)
            if allowed_override is None
            else max(0, int(allowed_override))
        )
        if policy is None:
            fitted, dropped = self._fit_background_materials(candidates, allowed)
            return _SourceFit(key, tuple(fitted), tuple(dropped))
        fitted, dropped, shortened = self._fit_policy(policy, candidates, allowed)
        fitted_tokens = sum(self.meter.measure_item(item).tokens for item in fitted)
        if policy.source is ContextSource.HISTORY_SUMMARY and fitted_tokens > allowed:
            report.source_limits[key] = max(report.source_limits.get(key, 0), fitted_tokens)
            report.trim_steps.append(f"preserve_latest_history_summary:{fitted_tokens - allowed}")
        return _SourceFit(
            key,
            tuple(fitted),
            tuple(dropped),
            tuple(shortened),
            policy.source is ContextSource.CURRENT_DIALOGUE and fitted_tokens > allowed,
        )

    def _fit_tokens(self, fit: _SourceFit) -> int:
        return sum(self.meter.measure_item(item).tokens for item in fit.fitted)

    @staticmethod
    def _weighted_limit(
        available_fill: int,
        weight: float,
        total_weight: float,
    ) -> int:
        if available_fill <= 0 or weight <= 0 or total_weight <= 0:
            return 0
        return math.floor(available_fill * weight / total_weight)

    def _fit_background_materials(
        self,
        candidates: Sequence[ContextItem],
        token_limit: int,
    ) -> tuple[list[ContextItem], list[ContextItem]]:
        """Fit complete works in deterministic category rounds.

        The first world change and first story module are attempted before any
        optional lived-history material.  One recent scene narration, one
        leftover, one ordinary experience, and one recent keyframe are then
        attempted per round.  A second world change and story module are
        offered only after the whole four-category rotation, so they use
        genuine spare room.
        """

        queues = {
            source: sorted(
                (item for item in candidates if item.source is source),
                key=lambda item: (item.sequence, item.item_id),
            )
            for source in BACKGROUND_OPTIONAL_SOURCES
        }
        fitter = _BackgroundMaterialFitter(self.meter, token_limit, queues)

        world_queue = queues[ContextSource.BACKGROUND_WORLD]
        if world_queue:
            fitter.try_item(world_queue.pop(0))
        fitter.fill_story_slot()

        rotating = (
            ContextSource.BACKGROUND_LEFTOVER,
            ContextSource.BACKGROUND_EXPERIENCE,
            ContextSource.BACKGROUND_KEYFRAME,
        )
        while any(queues[source] for source in rotating):
            for source in rotating:
                if queues[source]:
                    fitter.try_item(queues[source].pop(0))

        while world_queue:
            fitter.try_item(world_queue.pop(0))
        fitter.fill_story_slot()
        kept_ids = {item.item_id for item in fitter.kept}
        return fitter.kept, [item for item in candidates if item.item_id not in kept_ids]

    def _fit_policy(
        self,
        policy: SourcePolicy,
        candidates: Sequence[ContextItem],
        allowed: int,
    ) -> tuple[list[ContextItem], list[ContextItem], list[str]]:
        if policy.source is ContextSource.HISTORY_SUMMARY and candidates:
            # At runtime this source contains the latest cumulative summary.
            # Its covered raw messages were deliberately excluded while loading
            # the dialogue window, so dropping the last summary would create a
            # silent hole in the model's history.  Let that one atomic item
            # exceed its weighted first-round share when necessary; admission
            # still counts against remaining FillBudget and the hard guard.
            newest = max(candidates, key=lambda item: (item.sequence, item.item_id))
            required = self.meter.measure_item(newest).tokens
            fitted, dropped = _fit_newest(candidates, max(allowed, required), self.meter)
            return fitted, dropped, []
        if policy.source is ContextSource.CURRENT_DIALOGUE:
            fitted, dropped, shortened = _fit_dialogue_with_floor(candidates, allowed, self.meter)
            return fitted, dropped, shortened
        if policy.source is ContextSource.HISTORY_FRAGMENT:
            return _fit_chronological_history(candidates, allowed, self.meter)
        scored_sources = {
            ContextSource.CURRENT_WEB_RESOURCE,
            ContextSource.STICKER,
            ContextSource.CHARACTER_INTENT,
        }
        if policy.source in scored_sources:
            fitted, dropped = _fit_scored(candidates, allowed, self.meter)
            return fitted, dropped, []
        fitted, dropped = _fit_newest(candidates, allowed, self.meter)
        return fitted, dropped, []

    def _finalize_report(
        self,
        report: ContextBuildReport,
        selected: list[ContextItem],
        measurement: TokenMeasurement,
        protected_measurement: TokenMeasurement,
        *,
        custom_prompt_tokens: int = 0,
    ) -> None:
        report.count_mode = measurement.mode
        report.model_id = measurement.model_id
        report.total_tokens = measurement.tokens + custom_prompt_tokens
        report.protected_tokens = protected_measurement.tokens + custom_prompt_tokens
        report.custom_prompt_tokens = custom_prompt_tokens
        for source in ContextSource:
            source_items = [item for item in selected if item.source is source]
            if source_items:
                report.source_tokens[source.value] = self.meter.measure(source_items).tokens
        report.selected_item_ids = [item.item_id for item in selected]


# Imported after the model definitions because guard imports those definitions.
# noqa: E402

__all__ = [
    "BACKGROUND_FILL_WEIGHT",
    "BACKGROUND_OPTIONAL_SOURCES",
    "BudgetClass",
    "CompiledContext",
    "ConservativeTokenMeter",
    "ContextBudgetConfig",
    "ContextBudgetError",
    "ContextBudgetExceeded",
    "ContextBuildReport",
    "ContextCompiler",
    "ContextItem",
    "ContextProjector",
    "ContextSource",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_SOURCE_POLICIES",
    "DEFAULT_TARGET_CONTEXT_TOKENS",
    "DefaultContextProjector",
    "DialogueSummaryStrategy",
    "MIN_DIALOGUE_MESSAGES",
    "MIN_MAX_CONTEXT_TOKENS",
    "MIN_TARGET_CONTEXT_TOKENS",
    "ProjectionResult",
    "RequestBudgetGuard",
    "SourcePolicy",
    "SummaryStrategy",
    "TokenCountMode",
    "TokenMeasurement",
    "TokenMeter",
]
