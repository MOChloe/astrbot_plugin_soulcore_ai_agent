"""Bounded, identity-safe projection for earlier conversation pages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ...contracts.message_reference import safe_model_identity
from ...contracts.models import ConversationMessage
from ...shared.time_display import model_datetime
from .context import ConservativeTokenMeter


@dataclass(frozen=True, slots=True)
class ConversationHistoryPage:
    content: str
    message_count: int
    next_before_message_id: int | None
    has_more: bool
    participant_references: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class _HistoryRowProjection:
    source: ConversationMessage
    lines: tuple[str, ...]


class ConversationHistoryMixin:
    async def browse_earlier_dialogue(
        self,
        profile_id: str,
        instance_id: str,
        *,
        before_message_id: int | None,
        cutoff_at: datetime | None = None,
        limit: int = 20,
        token_limit: int = 2000,
        participant_references: Mapping[str, str] | None = None,
    ) -> ConversationHistoryPage:
        """Read one bounded, continuous page from the instance-owned ledger."""

        page_limit = max(1, min(int(limit or 20), 20))
        maximum_tokens = max(64, min(int(token_limit or 2000), 2000))
        boundary = int(before_message_id) if before_message_id is not None else None
        if boundary is not None and boundary <= 1:
            return ConversationHistoryPage("", 0, None, False)
        rows = await self.repository.list_instance_messages(
            profile_id,
            instance_id,
            through_message_id=(boundary - 1 if boundary is not None else None),
            through_occurred_at=cutoff_at,
            limit=page_limit + 1,
            ascending=False,
            context_eligible_only=True,
        )
        if not rows:
            return ConversationHistoryPage("", 0, None, False)
        projection = await self._history_projection_inputs(
            profile_id,
            instance_id,
            rows[:page_limit],
            participant_references=participant_references,
        )
        selected = self._bounded_history_lines(
            rows[:page_limit],
            media=projection[0],
            identity_context=projection[1],
            identity_catalog=projection[2],
            include_group_name=projection[3],
            maximum_tokens=maximum_tokens,
        )
        return self._history_page(rows, selected, participant_references=projection[4])

    async def _history_projection_inputs(
        self,
        profile_id: str,
        instance_id: str,
        rows: Sequence[ConversationMessage],
        *,
        participant_references: Mapping[str, str] | None,
    ) -> tuple[dict[int, list[dict[str, str]]], Any, Any, bool, dict[str, str]]:
        media = await self.media_repository.media_history_projections_for_messages(
            profile_id,
            instance_id,
            [int(item.message_id) for item in rows],
        )
        instance = await self.profiles.get_character_instance(profile_id, instance_id)
        include_group_name = bool(instance and str(instance.scope) == "group")
        references = _history_reference_seed(participant_references, include_group_name)
        participant_ids = _history_participant_ids(rows, include_group_name, references)
        identity_context, identity_catalog = await self.identity.catalog(
            profile_id,
            instance_id,
            participant_ids=participant_ids or None,
            participant_references=references or None,
        )
        return media, identity_context, identity_catalog, include_group_name, references

    def _bounded_history_lines(
        self,
        rows: Sequence[ConversationMessage],
        *,
        media: Mapping[int, Sequence[dict[str, str]]],
        identity_context: Any,
        identity_catalog: Any,
        include_group_name: bool,
        maximum_tokens: int,
    ) -> list[_HistoryRowProjection]:
        meter = ConservativeTokenMeter()
        selected: list[_HistoryRowProjection] = []
        used_tokens = 0
        for message in rows:
            entries = (*self.interrupted_expression_messages(message), message)
            newest_first_lines: list[str] = []
            for entry in reversed(entries):
                prefix = self._history_line_prefix(
                    entry,
                    identity_context=identity_context,
                    identity_catalog=identity_catalog,
                    include_group_name=include_group_name,
                )
                body = self.project_message(
                    entry,
                    include_sender=False,
                    media_projections=(
                        media.get(int(message.message_id), []) if entry is message else ()
                    ),
                    identity_context=identity_context,
                    identity_catalog=identity_catalog,
                )
                separator_tokens = meter.count_text("\n") if selected or newest_first_lines else 0
                remaining = maximum_tokens - used_tokens - separator_tokens
                if remaining <= 0:
                    break
                line = self._fit_history_line(prefix, body, remaining, meter)
                if not line:
                    break
                cost = meter.count_text(line)
                newest_first_lines.append(line)
                used_tokens += separator_tokens + cost
                if cost >= remaining:
                    break
            if not newest_first_lines:
                break
            selected.append(_HistoryRowProjection(message, tuple(reversed(newest_first_lines))))
            if used_tokens >= maximum_tokens:
                break
        return selected

    @staticmethod
    def _history_page(
        rows: Sequence[ConversationMessage],
        selected: Sequence[_HistoryRowProjection],
        *,
        participant_references: Mapping[str, str],
    ) -> ConversationHistoryPage:
        if not selected:
            return ConversationHistoryPage("", 0, None, False)
        has_more = len(rows) > len(selected)
        next_before = int(selected[-1].source.message_id) if has_more else None
        return ConversationHistoryPage(
            "\n".join(line for projection in reversed(selected) for line in projection.lines),
            len(selected),
            next_before,
            has_more,
            tuple(participant_references.items()),
        )

    @staticmethod
    def _history_line_prefix(
        message: ConversationMessage,
        *,
        identity_context: Any,
        identity_catalog: Any,
        include_group_name: bool,
    ) -> str:
        time_label = _history_time_label(message.occurred_at)
        participant_ref, display_name = _history_speaker_identity(
            message,
            identity_context=identity_context,
            identity_catalog=identity_catalog,
            include_group_name=include_group_name,
        )
        speaker = display_name or ("本人" if _is_assistant(message) else "对方")
        identity_field = f" [{participant_ref}]" if participant_ref else ""
        return f"[{time_label}]{identity_field} {speaker}："

    @staticmethod
    def _fit_history_line(
        prefix: str,
        body: str,
        token_limit: int,
        meter: ConservativeTokenMeter,
    ) -> str:
        normalized = str(body or "").strip()
        line = f"{prefix}{normalized}"
        if meter.count_text(line) <= token_limit:
            return line
        suffix = "…"
        low, high = 0, len(normalized)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = f"{prefix}{normalized[:middle].rstrip()}{suffix}"
            if meter.count_text(candidate) <= token_limit:
                low = middle
            else:
                high = middle - 1
        candidate = f"{prefix}{normalized[:low].rstrip()}{suffix}"
        return candidate if meter.count_text(candidate) <= token_limit else ""


def _history_reference_seed(
    participant_references: Mapping[str, str] | None,
    include_group_name: bool,
) -> dict[str, str]:
    if not include_group_name:
        return {}
    return dict(participant_references or {})


def _history_participant_ids(
    rows: Sequence[ConversationMessage],
    include_group_name: bool,
    references: dict[str, str],
) -> tuple[str, ...]:
    participant_ids: list[str] = []
    next_history_index = _next_history_reference_index(references)
    for item in reversed(rows):
        if _is_assistant(item):
            continue
        sender_id = str(item.sender_id or "").strip()
        if not sender_id:
            continue
        participant_ids.append(sender_id)
        if include_group_name and sender_id not in references:
            references[sender_id] = f"H{next_history_index}"
            next_history_index += 1
    return tuple(dict.fromkeys([*references.keys(), *participant_ids]))


def _next_history_reference_index(references: Mapping[str, str]) -> int:
    indexes = (
        int(reference[1:])
        for reference in references.values()
        if reference.startswith("H") and reference[1:].isdigit()
    )
    return max(indexes, default=0) + 1


def _history_time_label(value: Any) -> str:
    return model_datetime(value, localize=True) if isinstance(value, datetime) else "时间未知"


def _history_speaker_identity(
    message: ConversationMessage,
    *,
    identity_context: Any,
    identity_catalog: Any,
    include_group_name: bool,
) -> tuple[str, str]:
    if _is_assistant(message):
        display_name = str(getattr(identity_context, "character_name", "") or "")
        return "C", safe_model_identity(display_name)
    if include_group_name:
        return _group_history_speaker(message, identity_context, identity_catalog)
    display_name = str(getattr(identity_context, "private_display_name", "") or "")
    if not display_name:
        display_name = str(message.sender_name or "")
    return "P1", safe_model_identity(display_name)


def _group_history_speaker(
    message: ConversationMessage,
    identity_context: Any,
    identity_catalog: Any,
) -> tuple[str, str]:
    sender_id = str(message.sender_id or "")
    participant_ref = str(identity_catalog.group_participant_reference(sender_id) or "")
    participant = identity_context.participant_by_id.get(sender_id)
    display_name = participant.display_name if participant is not None else message.sender_name
    return participant_ref, safe_model_identity(str(display_name or ""))


def _is_assistant(message: ConversationMessage) -> bool:
    return str(message.role or "").lower() == "assistant"


__all__ = ["ConversationHistoryMixin", "ConversationHistoryPage"]
