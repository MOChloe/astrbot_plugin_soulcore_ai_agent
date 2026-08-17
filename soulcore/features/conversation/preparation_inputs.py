"""Load the bounded ledger inputs used by one Main Core context build."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ...contracts.message_reference import inbound_reply_reference
from ...contracts.models import CharacterInstance, ConversationMessage, DialogueSummary, ScopeConfig
from ...contracts.web import WebSearchResult
from .context import (
    ConservativeTokenMeter,
    ContextBudgetConfig,
    ContextCompiler,
)
from .expression_handles import load_expression_handles

MAX_DIALOGUE_ANCHORS = 8
_DIALOGUE_LEDGER_PAGE_SIZE = 512


@dataclass(slots=True)
class ContextPreparationInputs:
    config: ContextBudgetConfig
    meter: ConservativeTokenMeter
    compiler: ContextCompiler
    instance: CharacterInstance | None
    include_sender: bool
    summaries: tuple[DialogueSummary, ...]
    messages: list[ConversationMessage]
    dialogue_anchor_ids: frozenset[int]
    recent_dialogue_suffix_ids: frozenset[int]
    has_searchable_earlier_history: bool
    current_message: ConversationMessage | None
    current_messages: list[ConversationMessage]
    media_projections: dict[int, list[dict[str, str]]]
    message_ref_allowlist: dict[str, dict[str, Any]]
    member_ref_allowlist: dict[str, dict[str, Any]]
    member_ref_by_sender_id: dict[str, str]
    identity_context: Any
    identity_catalog: Any
    sticker_current_text: str
    sticker_recent_texts: tuple[str, ...]
    current_web_resources: tuple[WebSearchResult, ...]


class ConversationInputLoaderMixin:
    async def _load_preparation_inputs(
        self,
        *,
        profile_id: str,
        instance_id: str,
        role: ScopeConfig,
        model_id: str,
        run_prompt: str,
        current_message_id: int | None,
        current_message_ids: Sequence[int] = (),
        core_run_id: int,
        current_web_resources: Sequence[WebSearchResult] = (),
        thinking_policy: Any | None = None,
    ) -> ContextPreparationInputs:
        config = self.budget_config(role, thinking_policy)
        meter = ConservativeTokenMeter(model_id)
        instance = await self.profiles.get_character_instance(profile_id, instance_id)
        normalized_current_ids = self._normalized_current_message_ids(
            current_message_id,
            current_message_ids,
        )
        current_messages = await self._load_current_messages(
            profile_id, instance_id, current_message_ids=normalized_current_ids
        )
        (
            summaries,
            messages,
            dialogue_anchor_ids,
            recent_dialogue_suffix_ids,
            has_searchable_earlier_history,
        ) = await self._load_dialogue_window(
            profile_id,
            instance_id,
            current_message_ids=normalized_current_ids,
            current_messages=current_messages,
            fill_token_budget=config.fill_budget,
            meter=meter,
        )
        current_message = current_messages[-1] if current_messages else None
        (
            message_ref_allowlist,
            member_ref_allowlist,
            member_ref_by_sender_id,
        ) = await self._load_expression_handles(
            profile_id,
            instance_id,
            instance=instance,
            messages=[*messages, *current_messages],
            core_run_id=core_run_id,
        )
        participant_ids = tuple(
            sender_id
            for _member_ref, value in sorted(
                member_ref_allowlist.items(),
                key=lambda item: max(item[1].get("ledger_message_ids", ()) or (0,)),
                reverse=True,
            )
            if (sender_id := str(value.get("sender_id") or "").strip())
        )
        identity_context, identity_catalog = await self.identity.catalog(
            profile_id,
            instance_id,
            participant_ids=participant_ids or None,
        )
        media = await self._load_media_projections(profile_id, instance_id, messages)
        current_text = await self._sticker_current_message_text(
            profile_id,
            instance_id,
            current_message_id=current_message_id,
            run_prompt=run_prompt,
        )
        recent_dialogue = [
            entry
            for message in messages
            for entry in (*self.interrupted_expression_messages(message), message)
        ]
        recent_texts = tuple(
            self.project_message(
                item,
                include_sender=False,
                media_projections=media.get(int(item.message_id), []),
                identity_context=identity_context,
                identity_catalog=identity_catalog,
            )
            for item in recent_dialogue[-8:]
            if str(item.plain_text or "").strip()
        )
        return ContextPreparationInputs(
            config=config,
            meter=meter,
            compiler=ContextCompiler(config, meter),
            instance=instance,
            include_sender=bool(instance and str(instance.scope) == "group"),
            summaries=summaries,
            messages=messages,
            dialogue_anchor_ids=dialogue_anchor_ids,
            recent_dialogue_suffix_ids=recent_dialogue_suffix_ids,
            has_searchable_earlier_history=has_searchable_earlier_history,
            current_message=current_message,
            current_messages=current_messages,
            media_projections=media,
            message_ref_allowlist=message_ref_allowlist,
            member_ref_allowlist=member_ref_allowlist,
            member_ref_by_sender_id=member_ref_by_sender_id,
            identity_context=identity_context,
            identity_catalog=identity_catalog,
            sticker_current_text=current_text,
            sticker_recent_texts=recent_texts,
            current_web_resources=tuple(current_web_resources),
        )

    @staticmethod
    def _normalized_current_message_ids(
        current_message_id: int | None,
        current_message_ids: Sequence[int],
    ) -> tuple[int, ...]:
        return tuple(
            dict.fromkeys(
                [
                    *(int(value) for value in current_message_ids if int(value) > 0),
                    *([int(current_message_id)] if current_message_id is not None else []),
                ]
            )
        )

    async def _load_dialogue_window(
        self,
        profile_id: str,
        instance_id: str,
        *,
        current_message_ids: Sequence[int],
        current_messages: Sequence[ConversationMessage],
        fill_token_budget: int,
        meter: ConservativeTokenMeter,
    ) -> tuple[
        tuple[DialogueSummary, ...],
        list[ConversationMessage],
        frozenset[int],
        frozenset[int],
        bool,
    ]:
        latest = await self.repository.get_latest_dialogue_summary(profile_id, instance_id)
        summaries = (latest,) if latest is not None else ()
        messages, has_older_raw_dialogue = await self._load_dialogue_candidates_for_fill(
            profile_id,
            instance_id,
            after_message_id=(int(latest.covered_through_message_id) if latest else None),
            excluded_message_ids=current_message_ids,
            fill_token_budget=fill_token_budget,
            meter=meter,
        )
        messages, anchors, recent_suffix_ids = await self._select_recent_dialogue(
            profile_id, instance_id, messages, current_messages=current_messages
        )
        return (
            summaries,
            messages,
            anchors,
            recent_suffix_ids,
            latest is not None or has_older_raw_dialogue,
        )

    async def _load_dialogue_candidates_for_fill(
        self,
        profile_id: str,
        instance_id: str,
        *,
        after_message_id: int | None,
        excluded_message_ids: Sequence[int],
        fill_token_budget: int,
        meter: ConservativeTokenMeter,
    ) -> tuple[list[ConversationMessage], bool]:
        """Page newest-first until the fill budget, never a message-count quota.

        A page size is only a storage batching detail.  The context compiler is
        the sole authority that decides which candidates survive into MainCore.
        """

        excluded = {int(value) for value in excluded_message_ids}
        candidates: list[ConversationMessage] = []
        estimated_tokens = 0
        through_message_id: int | None = None
        next_older_message_id: int | None = None
        while estimated_tokens < max(1, int(fill_token_budget)):
            page = await self.repository.list_instance_messages(
                profile_id,
                instance_id,
                after_message_id=after_message_id,
                through_message_id=through_message_id,
                limit=_DIALOGUE_LEDGER_PAGE_SIZE,
                ascending=False,
                context_eligible_only=True,
            )
            if not page:
                return sorted(candidates, key=lambda message: int(message.message_id)), False
            for message in page:
                if int(message.message_id) in excluded:
                    continue
                candidates.append(message)
                for entry in (*self.interrupted_expression_messages(message), message):
                    projected = self.project_message(entry, include_sender=False)
                    estimated_tokens += (
                        meter.count_text(str(projected or "")) + meter.MESSAGE_OVERHEAD
                    )
            oldest_message_id = min(int(message.message_id) for message in page)
            if len(page) < _DIALOGUE_LEDGER_PAGE_SIZE:
                return sorted(candidates, key=lambda message: int(message.message_id)), False
            next_older_message_id = oldest_message_id - 1
            if after_message_id is not None and next_older_message_id <= int(after_message_id):
                return sorted(candidates, key=lambda message: int(message.message_id)), False
            through_message_id = next_older_message_id

        has_older = False
        if next_older_message_id is not None:
            has_older = bool(
                await self.repository.list_instance_messages(
                    profile_id,
                    instance_id,
                    after_message_id=after_message_id,
                    through_message_id=next_older_message_id,
                    limit=1,
                    ascending=False,
                    context_eligible_only=True,
                )
            )
        return sorted(candidates, key=lambda message: int(message.message_id)), has_older

    async def _select_recent_dialogue(
        self,
        profile_id: str,
        instance_id: str,
        messages: list[ConversationMessage],
        *,
        current_messages: Sequence[ConversationMessage],
    ) -> tuple[list[ConversationMessage], frozenset[int], frozenset[int]]:
        anchor_ids = await self._dialogue_anchor_ids(
            profile_id, instance_id, messages, current_messages
        )
        selected = await self._append_missing_dialogue_anchors(
            profile_id,
            instance_id,
            messages,
            anchor_ids,
        )
        selected_ids = {int(item.message_id) for item in selected}
        return (
            selected,
            frozenset(value for value in anchor_ids if value in selected_ids),
            frozenset(int(message.message_id) for message in messages),
        )

    async def _append_missing_dialogue_anchors(
        self,
        profile_id: str,
        instance_id: str,
        messages: Sequence[ConversationMessage],
        anchor_ids: Sequence[int],
    ) -> list[ConversationMessage]:
        by_id = {int(message.message_id): message for message in messages}
        for message_id in anchor_ids:
            if int(message_id) in by_id:
                continue
            message = await self.repository.get_instance_message(
                profile_id,
                instance_id,
                int(message_id),
            )
            if message is not None:
                by_id[int(message.message_id)] = message
        return sorted(by_id.values(), key=lambda message: int(message.message_id))

    async def _dialogue_anchor_ids(
        self,
        profile_id: str,
        instance_id: str,
        messages: Sequence[ConversationMessage],
        current_messages: Sequence[ConversationMessage],
    ) -> tuple[int, ...]:
        anchors: list[int] = []
        for current in current_messages:
            reply_id = await self._reply_anchor_id(profile_id, instance_id, current)
            if reply_id is not None:
                anchors.append(reply_id)
            anchors.extend(self._mention_anchor_ids(current, messages))
        return tuple(dict.fromkeys(anchors))[:MAX_DIALOGUE_ANCHORS]

    async def _reply_anchor_id(
        self, profile_id: str, instance_id: str, current: ConversationMessage
    ) -> int | None:
        reference = inbound_reply_reference(current.components)
        if not reference or str(reference.get("status") or "") != "resolved":
            return None
        message_ref = str(reference.get("target_message_ref") or "").strip()
        if not message_ref:
            return None
        fragment = await self.platform_messages.get_message_fragment(
            profile_id, instance_id, message_ref
        )
        return int(fragment.ledger_message_id) if fragment is not None else None

    @staticmethod
    def _mention_anchor_ids(
        current: ConversationMessage, messages: Sequence[ConversationMessage]
    ) -> tuple[int, ...]:
        targets = tuple(
            dict.fromkeys(
                str(component.get("qq") or component.get("sender_id") or "").strip()
                for component in current.components
                if isinstance(component, dict)
                and str(component.get("type") or "").lower() == "at"
                and str(component.get("qq") or component.get("sender_id") or "").strip()
            )
        )
        matches: list[int] = []
        for target in targets:
            match = next(
                (item for item in reversed(messages) if str(item.sender_id or "") == target),
                None,
            )
            if match is not None:
                matches.append(int(match.message_id))
        return tuple(matches)

    async def _load_current_messages(
        self,
        profile_id: str,
        instance_id: str,
        *,
        current_message_ids: Sequence[int],
    ) -> list[ConversationMessage]:
        if not current_message_ids:
            return []
        result = []
        for message_id in current_message_ids:
            message = await self.repository.get_instance_message(
                profile_id, instance_id, int(message_id)
            )
            if message is not None:
                result.append(message)
        return result

    async def _load_expression_handles(
        self,
        profile_id: str,
        instance_id: str,
        *,
        instance: CharacterInstance | None,
        messages: list[ConversationMessage],
        core_run_id: int,
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, str],
    ]:
        return await load_expression_handles(
            self.platform_messages,
            profile_id,
            instance_id,
            instance=instance,
            messages=messages,
            core_run_id=core_run_id,
        )

    async def _load_media_projections(
        self, profile_id: str, instance_id: str, messages: list[ConversationMessage]
    ) -> dict[int, list[dict[str, str]]]:
        return await self.media_repository.media_history_projections_for_messages(
            profile_id,
            instance_id,
            [item.message_id for item in messages],
        )

    async def _sticker_current_message_text(
        self,
        profile_id: str,
        instance_id: str,
        *,
        current_message_id: int | None,
        run_prompt: str,
    ) -> str:
        if current_message_id is None:
            return ""
        message = await self.repository.get_instance_message(
            profile_id, instance_id, int(current_message_id)
        )
        if message is not None:
            context, catalog = await self.identity.catalog(profile_id, instance_id)
            return self.project_message(
                message,
                include_sender=False,
                identity_context=context,
                identity_catalog=catalog,
            )
        return self._current_player_text(run_prompt)


__all__ = ["ContextPreparationInputs", "ConversationInputLoaderMixin"]
