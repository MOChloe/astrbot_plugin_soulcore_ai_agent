"""Compile context items and project only workset references that survived trimming."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ...contracts.text_fingerprint import content_fingerprint
from ..stickers.service import StickerWorkset
from .context import CompiledContext, ContextItem, ContextSource, RequestBudgetGuard
from .player_profile_context import PlayerProfileRunTarget
from .preparation_inputs import ContextPreparationInputs


@dataclass(slots=True)
class PreparedMainCoreContext:
    compiled: CompiledContext
    guard: RequestBudgetGuard
    effective_max_tokens: int
    model_id: str
    stickers: StickerWorkset = StickerWorkset()
    message_ref_allowlist: dict[str, dict[str, Any]] | None = None
    member_ref_allowlist: dict[str, dict[str, Any]] | None = None
    player_profile_targets: dict[str, PlayerProfileRunTarget] | None = None
    current_turn: tuple[dict[str, Any], ...] = ()
    identity_context: Any | None = None
    identity_catalog: Any | None = None
    history_before_message_id: int | None = None
    visible_history_summary_ids: frozenset[int] = frozenset()
    visible_message_ids: frozenset[int] = frozenset()
    visible_history_fingerprints: frozenset[str] = frozenset()
    visible_recall_document_keys: frozenset[str] = frozenset()
    visible_summary_coverage: tuple[tuple[int, int, int], ...] = ()
    background_enabled: bool = False
    has_searchable_earlier_history: bool = False


@dataclass(slots=True)
class ContextCompilationResult:
    compiled: CompiledContext
    stickers: StickerWorkset
    message_ref_allowlist: dict[str, dict[str, Any]]
    member_ref_allowlist: dict[str, dict[str, Any]]
    identity_context: Any
    identity_catalog: Any
    history_before_message_id: int | None
    visible_history_summary_ids: frozenset[int]
    visible_message_ids: frozenset[int]
    visible_history_fingerprints: frozenset[str]
    visible_summary_coverage: tuple[tuple[int, int, int], ...]


class ContextCompilationMixin:
    async def _compile_context_result(
        self,
        *,
        profile_id: str,
        instance_id: str,
        model_id: str,
        core_run_id: int,
        provider_context_limit: int | None,
        defer_provider_selection: bool,
        items: list[ContextItem],
        stickers: StickerWorkset,
        inputs: ContextPreparationInputs,
        custom_prompt_text: str = "",
    ) -> ContextCompilationResult:
        compiled = inputs.compiler.compile(
            items,
            provider_context_limit=provider_context_limit,
            defer_provider_selection=defer_provider_selection,
            custom_prompt_text=custom_prompt_text,
        )
        stickers = self._compiled_sticker_workset(compiled, stickers, inputs)
        await self._save_context_build_report(
            profile_id,
            instance_id,
            model_id=model_id,
            compiled=compiled,
        )
        visible_message_ids = self._visible_message_ids(compiled, inputs)
        message_ref_allowlist = self._visible_message_ref_allowlist(
            inputs.message_ref_allowlist, visible_message_ids
        )
        member_ref_allowlist = self._visible_member_ref_allowlist(
            inputs.member_ref_allowlist, visible_message_ids
        )
        visible_summary_ids = self._visible_summary_ids(compiled)
        visible_history_fingerprints = self._visible_history_fingerprints(compiled)
        visible_summary_coverage = self._visible_summary_coverage(inputs, visible_summary_ids)
        return ContextCompilationResult(
            compiled,
            stickers,
            message_ref_allowlist,
            member_ref_allowlist,
            inputs.identity_context,
            inputs.identity_catalog,
            self._history_before_message_id(visible_message_ids, inputs),
            visible_summary_ids,
            frozenset(visible_message_ids),
            visible_history_fingerprints,
            visible_summary_coverage,
        )

    @staticmethod
    def _visible_message_ids(
        compiled: CompiledContext, inputs: ContextPreparationInputs
    ) -> set[int]:
        values = {
            int(item.metadata.get("ledger_message_id") or 0)
            for item in compiled.items
            if int(item.metadata.get("ledger_message_id") or 0) > 0
        }
        values.update(int(message.message_id) for message in inputs.current_messages)
        return values

    @staticmethod
    def _visible_message_ref_allowlist(
        allowlist: dict[str, dict[str, Any]], visible_message_ids: set[int]
    ) -> dict[str, dict[str, Any]]:
        return {
            ref: value
            for ref, value in allowlist.items()
            if int(value.get("ledger_message_id") or 0) in visible_message_ids
        }

    @staticmethod
    def _visible_member_ref_allowlist(
        allowlist: dict[str, dict[str, Any]], visible_message_ids: set[int]
    ) -> dict[str, dict[str, Any]]:
        return {
            ref: value
            for ref, value in allowlist.items()
            if any(
                int(message_id) in visible_message_ids
                for message_id in value.get("ledger_message_ids", ())
            )
        }

    @staticmethod
    def _visible_summary_ids(compiled: CompiledContext) -> frozenset[int]:
        return frozenset(
            int(item.item_id.partition(":")[2])
            for item in compiled.items
            if item.source is ContextSource.HISTORY_SUMMARY
            and item.item_id.startswith("summary:")
            and item.item_id.partition(":")[2].isdigit()
        )

    @staticmethod
    def _visible_history_fingerprints(compiled: CompiledContext) -> frozenset[str]:
        return frozenset(
            fingerprint
            for item in compiled.items
            if item.source in {ContextSource.HISTORY_SUMMARY, ContextSource.CURRENT_DIALOGUE}
            if (fingerprint := content_fingerprint(item.body))
        )

    @staticmethod
    def _visible_summary_coverage(
        inputs: ContextPreparationInputs, visible_summary_ids: frozenset[int]
    ) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (
                int(summary.summary_id),
                int(summary.covered_from_message_id),
                int(summary.covered_through_message_id),
            )
            for summary in inputs.summaries
            if int(summary.summary_id) in visible_summary_ids
        )

    @staticmethod
    def _history_before_message_id(
        visible_message_ids: set[int], inputs: ContextPreparationInputs
    ) -> int | None:
        visible_suffix_ids = sorted(
            visible_message_ids.intersection(inputs.recent_dialogue_suffix_ids)
        )
        if visible_suffix_ids:
            return visible_suffix_ids[0]
        current_ids = sorted(int(message.message_id) for message in inputs.current_messages)
        return current_ids[0] if current_ids else None

    @staticmethod
    def _compiled_sticker_workset(
        compiled: CompiledContext,
        stickers: StickerWorkset,
        inputs: ContextPreparationInputs,
    ) -> StickerWorkset:
        visible = frozenset(
            str(item.metadata.get("sticker_ref") or "")
            for item in compiled.items
            if item.source is ContextSource.STICKER and str(item.metadata.get("sticker_ref") or "")
        )
        selected = tuple(item for item in stickers.items if item.sticker_ref in visible)
        return StickerWorkset(
            items=selected,
            visible_refs=visible,
            token_limit=compiled.report.source_limits.get(ContextSource.STICKER.value, 0),
            used_tokens=sum(
                inputs.meter.count_text(item.content) + inputs.meter.MESSAGE_OVERHEAD
                for item in selected
            ),
        )

    async def _save_context_build_report(
        self,
        profile_id: str,
        instance_id: str,
        *,
        model_id: str,
        compiled: CompiledContext,
    ) -> None:
        report = compiled.report
        await self.repository.save_context_build_report(
            profile_id,
            instance_id,
            model_id=model_id,
            token_count_mode=report.count_mode.value,
            hard_token_limit=report.effective_max_tokens,
            target_token_budget=report.target_context_tokens,
            fill_budget=report.fill_budget,
            total_tokens=report.total_tokens,
            report=asdict(report),
        )


__all__ = [
    "ContextCompilationMixin",
    "ContextCompilationResult",
    "PreparedMainCoreContext",
]
