"""Assemble typed context items from already bounded feature worksets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from ...shared.prompt_document import (
    TrustedPromptMarkup,
    prompt_field_lines,
    prompt_markup_block,
    xml_text,
)
from ...shared.role_current_view import (
    MainCoreBackgroundViewProjection,
    main_core_background_fragments,
)
from ...shared.time_display import model_datetime
from ..stickers.service import StickerWorkset
from .context import BudgetClass, ContextItem, ContextSource
from .preparation_inputs import ContextPreparationInputs

_CONTEXT_FIELD_LABELS = {
    "summary": "事项",
    "goal": "目标",
    "motivation": "原因",
    "constraints": "限制",
    "not_before": "不早于",
    "target_at": "预计时间",
    "expires_at": "失效时间",
}
_CONTEXT_TIME_FIELDS = frozenset({"not_before", "target_at", "expires_at"})


_BACKGROUND_SOURCE_BY_KIND = {
    "life-direction": ContextSource.ROLE_LIFE_DIRECTION,
    "current": ContextSource.ROLE_STATE,
    "latest-experience": ContextSource.ROLE_LATEST_EXPERIENCE,
    "world": ContextSource.BACKGROUND_WORLD,
    "ordinary-experience": ContextSource.BACKGROUND_EXPERIENCE,
    "keyframe-experience": ContextSource.BACKGROUND_KEYFRAME,
    "leftover": ContextSource.BACKGROUND_LEFTOVER,
    "story": ContextSource.BACKGROUND_STORY,
}


def background_material_context_items(value: Any | None) -> tuple[ContextItem, ...]:
    if not isinstance(value, MainCoreBackgroundViewProjection):
        return ()
    background_as_of = _background_as_of(value)
    return tuple(
        ContextItem(
            item_id=f"background-material:{fragment.fragment_id}",
            budget_class=BudgetClass.SYSTEM if fragment.protected else BudgetClass.DATA,
            source=_BACKGROUND_SOURCE_BY_KIND[fragment.kind],
            speaker="system",
            body=fragment.body,
            sequence=fragment.sequence,
            metadata={
                "rank": -fragment.sequence,
                "background_kind": fragment.kind,
                "background_ref": fragment.reference_key,
                **({"background_as_of": background_as_of} if background_as_of is not None else {}),
            },
        )
        for fragment in main_core_background_fragments(value)
    )


def _background_as_of(value: MainCoreBackgroundViewProjection) -> datetime | None:
    current = getattr(value, "current_view", None)
    current_as_of = getattr(current, "as_of", None)
    if isinstance(current_as_of, datetime):
        return current_as_of
    timeline_times = tuple(
        frame_end
        for event in getattr(value, "timeline", ())
        if isinstance((frame_end := getattr(event, "frame_end_at", None)), datetime)
    )
    return max(timeline_times) if timeline_times else None


class ContextItemAssemblerMixin:
    async def _assemble_context_items(
        self,
        *,
        profile_id: str,
        instance_id: str,
        run_prompt: str,
        current_message_id: int | None,
        core_run_id: int,
        active_intents: Sequence[Mapping[str, Any]],
        inputs: ContextPreparationInputs,
        background_view: Any | None = None,
    ) -> tuple[
        list[ContextItem],
        StickerWorkset,
    ]:
        items = self._summary_items(inputs.summaries)
        await self._append_history_fragments(
            items,
            profile_id=profile_id,
            instance_id=instance_id,
            inputs=inputs,
        )
        self._append_character_intents(items, active_intents)
        stickers = await self._sticker_workset(profile_id, instance_id, core_run_id, inputs)
        self._append_feature_worksets(items, stickers)
        self._append_background_material(items, background_view)
        self._append_current_web_resources(items, inputs)
        self._append_dialogue(items, inputs)
        self._append_current_player(items, run_prompt, inputs)
        return items, stickers

    async def _append_history_fragments(
        self,
        items: list[ContextItem],
        *,
        profile_id: str,
        instance_id: str,
        inputs: ContextPreparationInputs,
    ) -> None:
        """Append only compressed Memory prose in chronological order.

        This is normal history filling, not Recall.  The dedicated repository
        projection deliberately omits keywords, scores, states, evidence text,
        and other administration fields that do not belong in the role's story.
        """

        visible_message_ids = {
            int(message.message_id)
            for message in (*inputs.messages, *inputs.current_messages)
            if int(message.message_id) > 0
        }
        records = await self.knowledge.list_context_memories(
            profile_id,
            instance_id,
            limit=5000,
        )
        chronological = [
            record
            for record in reversed(records)
            if not (set(self._memory_source_message_ids(record)) & visible_message_ids)
            and str(record.get("brief") or "").strip()
        ]
        for sequence, record in enumerate(chronological, start=1):
            memory_id = int(record.get("memory_id") or 0)
            revision = max(1, int(record.get("revision") or 1))
            brief = str(record.get("brief") or "").strip()
            compact = str(record.get("ultra_brief") or "").strip() or brief
            occurred_at = record.get("occurred_at")
            items.append(
                ContextItem(
                    item_id=f"history-fragment:{memory_id}",
                    budget_class=BudgetClass.DATA,
                    source=ContextSource.HISTORY_FRAGMENT,
                    speaker="system",
                    body=self._history_fragment_text(occurred_at, brief),
                    sequence=sequence,
                    metadata={
                        "compact_content": self._history_fragment_text(occurred_at, compact),
                        "document_keys": (f"memory:{memory_id}:r{revision}",),
                    },
                )
            )

    @staticmethod
    def _memory_source_message_ids(record: Mapping[str, Any]) -> tuple[int, ...]:
        return tuple(
            int(value) for value in (record.get("source_message_ids") or ()) if int(value) > 0
        )

    @staticmethod
    def _history_fragment_text(occurred_at: Any, content: str) -> str:
        timestamp = model_datetime(occurred_at)
        return f"{timestamp}：{content}" if timestamp else content

    @staticmethod
    def _append_background_material(items: list[ContextItem], value: Any | None) -> None:
        items.extend(background_material_context_items(value))

    @staticmethod
    def _summary_items(summaries: Sequence[Any]) -> list[ContextItem]:
        items: list[ContextItem] = []
        for summary in summaries:
            rendered = str(summary.rendered_text or "").strip()
            if not rendered:
                continue
            items.append(
                ContextItem(
                    item_id=f"summary:{summary.summary_id}",
                    budget_class=BudgetClass.DATA,
                    source=ContextSource.HISTORY_SUMMARY,
                    speaker="system",
                    body=rendered,
                    sequence=int(summary.version),
                    metadata={"rank": int(summary.version)},
                )
            )
        return items

    def _append_character_intents(
        self,
        items: list[ContextItem],
        active_intents: Sequence[Mapping[str, Any]],
    ) -> None:
        self._append_structured_rows(
            items,
            rows=list(active_intents)[:32],
            fields=(
                "intent_ref",
                "intent_id",
                "summary",
                "goal",
                "motivation",
                "constraints",
                "not_before",
                "target_at",
                "expires_at",
                "priority",
            ),
            item_prefix="character-intent",
            source=ContextSource.CHARACTER_INTENT,
            sequence_base=-18,
            identity_fields=("intent_id", "intent_ref"),
            score_field="priority",
        )

    def _append_structured_rows(
        self,
        items: list[ContextItem],
        *,
        rows: Sequence[Mapping[str, Any]],
        fields: Sequence[str],
        item_prefix: str,
        source: ContextSource,
        sequence_base: int,
        identity_fields: tuple[str, str],
        score_field: str,
    ) -> None:
        for rank, row in enumerate(rows, start=1):
            value = self._compact_structured_row(row, fields=fields)
            identity = value.get(identity_fields[0]) or value.get(identity_fields[1]) or rank
            content = "；".join(
                f"{_CONTEXT_FIELD_LABELS[key]}：{self._readable_value(item, field_name=key)}"
                for key, item in value.items()
                if key in _CONTEXT_FIELD_LABELS and item not in (None, "", [], {})
            )
            if not content:
                continue
            items.append(
                ContextItem(
                    item_id=f"{item_prefix}:{identity}",
                    budget_class=BudgetClass.DATA,
                    source=source,
                    speaker="system",
                    body=content,
                    sequence=sequence_base + rank,
                    metadata={"score": float(value.get(score_field) or 0), "rank": -rank},
                )
            )

    @staticmethod
    def _readable_value(value: Any, *, field_name: str = "") -> str:
        if isinstance(value, datetime) or field_name in _CONTEXT_TIME_FIELDS:
            return model_datetime(value)
        if isinstance(value, Mapping):
            return "，".join(
                f"{key}={ContextItemAssemblerMixin._readable_value(item)}"
                for key, item in value.items()
                if item not in (None, "", [], {})
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return "、".join(str(item) for item in value)
        return str(value)

    async def _sticker_workset(
        self,
        profile_id: str,
        instance_id: str,
        core_run_id: int,
        inputs: ContextPreparationInputs,
    ) -> StickerWorkset:
        enabled = await self._stickers_enabled(profile_id, inputs.instance)
        if not enabled:
            return StickerWorkset()
        return await self.stickers.build_workset(
            profile_id=profile_id,
            instance_id=instance_id,
            run_id=core_run_id,
            current_text=inputs.sticker_current_text,
            recent_texts=inputs.sticker_recent_texts,
            candidate_token_limit=inputs.config.fill_budget,
            meter=inputs.meter,
        )

    async def _stickers_enabled(self, profile_id: str, instance: Any) -> bool:
        if instance is None:
            return True
        return await self.stickers.is_enabled(profile_id, str(instance.scope))

    @staticmethod
    def _append_feature_worksets(
        items: list[ContextItem],
        stickers: StickerWorkset,
    ) -> None:
        for rank, projection in enumerate(stickers.items, start=1):
            items.append(
                ContextItem(
                    item_id=f"sticker:{projection.sticker_ref}",
                    budget_class=BudgetClass.DATA,
                    source=ContextSource.STICKER,
                    speaker="system",
                    body=projection.content,
                    sequence=-4 + rank,
                    metadata={
                        "score": projection.score,
                        "rank": -rank,
                        "sticker_ref": projection.sticker_ref,
                    },
                )
            )

    @staticmethod
    def _append_current_web_resources(
        items: list[ContextItem], inputs: ContextPreparationInputs
    ) -> None:
        for rank, resource in enumerate(inputs.current_web_resources, start=1):
            internal = str(resource.resource_id).strip()
            url = str(resource.canonical_url).strip()
            if not internal or not url:
                continue
            items.append(
                ContextItem(
                    item_id=f"current-web:{rank}",
                    budget_class=BudgetClass.DATA,
                    source=ContextSource.CURRENT_WEB_RESOURCE,
                    speaker="system",
                    body=f"对方在当前消息中给出的网页，可按需读取：{url}",
                    sequence=-3 + rank,
                    metadata={"item_id": internal, "rank": 2000 - rank},
                )
            )

    def _append_dialogue(self, items: list[ContextItem], inputs: ContextPreparationInputs) -> None:
        sequence = max((item.sequence for item in items if item.sequence >= 0), default=0)
        for message in inputs.messages:
            sequence = self._append_interrupted_dialogue_entries(
                items,
                message,
                inputs,
                sequence=sequence,
            )
            sequence += 1
            items.append(
                ContextItem(
                    item_id=f"message:{message.message_id}",
                    budget_class=BudgetClass.DATA,
                    source=ContextSource.CURRENT_DIALOGUE,
                    speaker=str(message.role or "user"),
                    body=self.project_message(
                        message,
                        include_sender=False,
                        media_projections=inputs.media_projections.get(int(message.message_id), []),
                        member_ref=inputs.member_ref_by_sender_id.get(
                            str(message.sender_id or ""), ""
                        ),
                        identity_context=inputs.identity_context,
                        identity_catalog=inputs.identity_catalog,
                    ),
                    sequence=sequence,
                    metadata={
                        "ledger_message_id": int(message.message_id),
                        "timeline_event_kind": str(
                            message.metadata.get("timeline_event_kind") or ""
                        ),
                        "dialogue_anchor": int(message.message_id) in inputs.dialogue_anchor_ids,
                        "occurred_at": message.occurred_at,
                        "sender_name": self._current_sender_name(
                            message,
                            inputs.identity_context,
                        )[:80],
                        "sender_id": str(message.sender_id or ""),
                        "member_ref": inputs.member_ref_by_sender_id.get(
                            str(message.sender_id or ""), ""
                        ),
                    },
                )
            )

    def _append_interrupted_dialogue_entries(
        self,
        items: list[ContextItem],
        message: Any,
        inputs: ContextPreparationInputs,
        *,
        sequence: int,
        current_turn_shadow: bool = False,
    ) -> int:
        for index, interrupted in enumerate(self.interrupted_expression_messages(message), start=1):
            sequence += 1
            items.append(
                ContextItem(
                    item_id=f"interrupted:{int(message.message_id)}:{index}",
                    budget_class=BudgetClass.DATA,
                    source=ContextSource.CURRENT_DIALOGUE,
                    speaker="assistant",
                    body=self.project_message(
                        interrupted,
                        include_sender=False,
                        identity_context=inputs.identity_context,
                        identity_catalog=inputs.identity_catalog,
                    ),
                    sequence=sequence,
                    metadata={
                        "interrupted_unsent": True,
                        "current_turn_shadow": current_turn_shadow,
                        "interrupting_message_id": int(message.message_id),
                        "occurred_at": message.occurred_at,
                        "sender_name": str(inputs.identity_context.character_name or "")[:80],
                    },
                )
            )
        return sequence

    def _append_current_player(
        self, items: list[ContextItem], run_prompt: str, inputs: ContextPreparationInputs
    ) -> None:
        sequence = max([item.sequence for item in items if item.sequence >= 0], default=0)
        for message in inputs.current_messages:
            sequence = self._append_interrupted_dialogue_entries(
                items,
                message,
                inputs,
                sequence=sequence,
                current_turn_shadow=True,
            )
        sequence += 1
        current = inputs.current_message
        content = run_prompt
        if current is None and not str(content or "").strip():
            return
        if current is not None:
            projected = (
                self._with_sender_identity(
                    "",
                    current,
                    member_ref=inputs.member_ref_by_sender_id.get(str(current.sender_id or ""), ""),
                )
                if inputs.include_sender
                else ""
            )
            if projected:
                identity_block = prompt_markup_block(
                    "当前消息的称呼与引用关系",
                    prompt_field_lines({"关系说明": projected}),
                )
                visible = (
                    str(content).strip()
                    if isinstance(content, TrustedPromptMarkup)
                    else xml_text(content)
                )
                content = TrustedPromptMarkup(f"{visible}\n\n{identity_block}".strip())
        items.append(
            ContextItem(
                item_id="protected:current",
                budget_class=BudgetClass.SYSTEM,
                source=ContextSource.CURRENT_PLAYER_MESSAGE,
                speaker="user",
                body=content,
                sequence=sequence,
                metadata={
                    "ledger_message_id": int(current.message_id) if current is not None else 0
                },
            )
        )


__all__ = ["ContextItemAssemblerMixin", "background_material_context_items"]
