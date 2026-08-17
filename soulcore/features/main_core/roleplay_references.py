"""Run-local references and dialogue projection for RolePlay prompts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from ...contracts.message_reference import safe_model_identity
from ...shared.prompt_document import TrustedPromptMarkup, xml_text
from ..conversation import ContextItem, ContextSource, render_dialogue_line
from ..conversation.service import PreparedMainCoreContext
from ..identity import escape_untrusted_identity_syntax, resolve_private_display_name
from .roleplay_prompt_contracts import (
    LONG_MESSAGE_REF,
    MEMBER_REF,
    OPAQUE_SENDER_PREFIX,
    DialoguePromptEntry,
    ShortReferenceMap,
)


class RolePlayReferenceMixin:
    def _short_references(
        self,
        items: Sequence[ContextItem],
        prepared_context: PreparedMainCoreContext | None,
        file_references: Mapping[str, Any] | None = None,
    ) -> ShortReferenceMap:
        public_to_internal: dict[str, Any] = {}
        internal_to_public: dict[str, str] = {}
        identity_catalog = getattr(prepared_context, "identity_catalog", None)
        message_by_ledger_id = self._dialogue_references(
            items,
            prepared_context,
            public_to_internal,
            internal_to_public,
        )
        participant_by_internal = self._participant_references(
            prepared_context,
            identity_catalog,
            public_to_internal,
            internal_to_public,
        )
        self._profile_entry_references(
            items,
            participant_by_internal,
            public_to_internal,
            internal_to_public,
        )
        item_public = self._item_references(
            items,
            public_to_internal,
            internal_to_public,
        )
        self._file_references(file_references or {}, public_to_internal, internal_to_public)
        identity_context = getattr(prepared_context, "identity_context", None)
        return ShortReferenceMap(
            public_to_internal=public_to_internal,
            internal_to_public=internal_to_public,
            message_by_ledger_id=message_by_ledger_id,
            participant_by_internal=participant_by_internal,
            item_public=item_public,
            character_name=str(getattr(identity_context, "character_name", "") or ""),
            private_display_name=str(getattr(identity_context, "private_display_name", "") or ""),
            private_name_override_enabled=bool(
                getattr(identity_context, "private_name_override_enabled", False)
            ),
            identity_scope=str(getattr(identity_context, "scope", "profile") or "profile"),
        )

    @staticmethod
    def _file_references(
        values: Mapping[str, Any],
        public_to_internal: dict[str, Any],
        internal_to_public: dict[str, str],
    ) -> None:
        for index, internal in enumerate(values, start=1):
            key = str(internal or "").strip()
            if not key:
                continue
            public = f"F{index}"
            public_to_internal[public] = key
            internal_to_public[key] = public

    def _dialogue_anchor_flags(
        self, items: Sequence[ContextItem], refs: ShortReferenceMap
    ) -> list[bool]:
        flags: list[bool] = []
        for item in items:
            if item.source is not ContextSource.CURRENT_DIALOGUE:
                continue
            if self._clean_item(item, refs):
                flags.append(bool(item.metadata.get("dialogue_anchor")))
        return flags

    def _dialogue_references(
        self,
        items: Sequence[ContextItem],
        prepared_context: PreparedMainCoreContext | None,
        public_to_internal: dict[str, Any],
        internal_to_public: dict[str, str],
    ) -> dict[int, str]:
        message_by_ledger_id: dict[int, str] = {}
        counters = {"A": 0, "U": 0}
        allowlist = dict(prepared_context.message_ref_allowlist or {}) if prepared_context else {}
        seen_ledger_ids: set[int] = set()
        for role_name, ledger_id in self._dialogue_entries(items, prepared_context):
            if ledger_id <= 0:
                continue
            if ledger_id and ledger_id in seen_ledger_ids:
                continue
            if ledger_id:
                seen_ledger_ids.add(ledger_id)
            public = self._allocate_dialogue_reference(role_name, counters)
            if ledger_id:
                message_by_ledger_id[ledger_id] = public
            self._bind_message_reference(
                ledger_id,
                public,
                allowlist,
                public_to_internal,
                internal_to_public,
            )
        return message_by_ledger_id

    @staticmethod
    def _dialogue_entries(
        items: Sequence[ContextItem], prepared_context: PreparedMainCoreContext | None
    ) -> list[tuple[str, int]]:
        dialogue = sorted(
            (
                item
                for item in items
                if item.source is ContextSource.CURRENT_DIALOGUE
                and not item.metadata.get("current_turn_shadow")
            ),
            key=lambda item: int(item.sequence),
        )
        entries = [
            (
                str(item.speaker or "user"),
                int(item.metadata.get("ledger_message_id") or 0),
            )
            for item in dialogue
            if not item.metadata.get("timeline_event_kind")
        ]
        current = tuple(prepared_context.current_turn if prepared_context else ())
        entries.extend(
            (str(row.get("speaker") or "user"), int(row.get("ledger_message_id") or 0))
            for row in current
            if isinstance(row, Mapping)
        )
        return entries

    @staticmethod
    def _allocate_dialogue_reference(role_name: str, counters: dict[str, int]) -> str:
        prefix = "A" if role_name == "assistant" else "U"
        counters[prefix] += 1
        return f"{prefix}{counters[prefix]}"

    @staticmethod
    def _bind_message_reference(
        ledger_id: int,
        public: str,
        allowlist: Mapping[str, Any],
        public_to_internal: dict[str, Any],
        internal_to_public: dict[str, str],
    ) -> None:
        for internal, value in allowlist.items():
            if int(value.get("ledger_message_id") or 0) != ledger_id:
                continue
            public_to_internal.setdefault(public, internal)
            internal_to_public[internal] = public

    @staticmethod
    def _participant_references(
        prepared_context: PreparedMainCoreContext | None,
        identity_catalog: Any | None,
        public_to_internal: dict[str, Any],
        internal_to_public: dict[str, str],
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        participants = sorted(
            dict(prepared_context.member_ref_allowlist or {}).items() if prepared_context else (),
            key=lambda item: max(item[1].get("ledger_message_ids", ()) or (0,)),
            reverse=True,
        )
        used: set[str] = set()
        fallback_index = 1
        for internal, value in participants:
            sender_id = str(value.get("sender_id") or "").strip()
            public = (
                str(identity_catalog.group_participant_reference(sender_id) or "")
                if identity_catalog is not None and sender_id
                else ""
            )
            while not public or public in used:
                candidate = f"P{fallback_index}"
                fallback_index += 1
                if candidate not in used:
                    public = candidate
            used.add(public)
            result[internal] = public
            public_to_internal[public] = internal
            internal_to_public[internal] = public
        return result

    @staticmethod
    def _profile_entry_references(
        items: Sequence[ContextItem],
        participant_by_internal: Mapping[str, str],
        public_to_internal: dict[str, Any],
        internal_to_public: dict[str, str],
    ) -> None:
        for item in items:
            if item.source is not ContextSource.PLAYER_PROFILE:
                continue
            for row in item.metadata.get("profile_entries", ()):
                internal = str(row.get("entry_id") or "").strip()
                if not internal:
                    continue
                member_ref = str(row.get("member_ref") or "")
                entry_index = max(1, int(row.get("entry_index") or 1))
                person = participant_by_internal.get(member_ref)
                public = f"{person}.{entry_index}" if person else f"E{entry_index}"
                public_to_internal[public] = internal
                internal_to_public[internal] = public

    @staticmethod
    def _item_references(
        items: Sequence[ContextItem],
        public_to_internal: dict[str, Any],
        internal_to_public: dict[str, str],
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        source_refs = {
            ContextSource.STICKER: ("S", "sticker_ref"),
            ContextSource.CURRENT_WEB_RESOURCE: ("R", "item_id"),
        }
        counters: dict[str, int] = {}
        ordered_items = (
            *(item for item in items if item.source is ContextSource.ROLE_LATEST_EXPERIENCE),
            *(item for item in items if item.source is not ContextSource.ROLE_LATEST_EXPERIENCE),
        )
        for item in ordered_items:
            descriptor = source_refs.get(item.source)
            if descriptor is None:
                continue
            prefix, metadata_key = descriptor
            internal = str(item.metadata.get(metadata_key) or "").strip()
            if not internal:
                continue
            counters[prefix] = counters.get(prefix, 0) + 1
            public = f"{prefix}{counters[prefix]}"
            result[str(item.item_id)] = public
            public_to_internal[public] = internal
            internal_to_public[internal] = public
        return result

    @staticmethod
    def _current_input_with_refs(
        value: str, asset_ids: Sequence[str], refs: ShortReferenceMap
    ) -> str:
        text = str(value or "")
        next_index = 1 + sum(1 for key in refs.public_to_internal if key.startswith("I"))
        public_to_internal = refs.public_to_internal
        internal_to_public = refs.internal_to_public
        if isinstance(public_to_internal, dict) and isinstance(internal_to_public, dict):
            for asset_id in asset_ids:
                internal = str(asset_id or "").strip()
                if not internal:
                    continue
                public = f"I{next_index}"
                next_index += 1
                public_to_internal[public] = internal
                internal_to_public[internal] = public
                text = text.replace(internal, public)
        text = OPAQUE_SENDER_PREFIX.sub("", text)
        return text.strip()

    def _current_turn_text(
        self,
        prepared_context: PreparedMainCoreContext | None,
        refs: ShortReferenceMap,
        *,
        fallback: str,
        occurred_at: datetime,
    ) -> str:
        entries = self._current_turn_entries(
            prepared_context,
            refs,
            fallback=fallback,
            occurred_at=occurred_at,
        )
        return "\n".join(entry.text for entry in entries)

    def _current_turn_entries(
        self,
        prepared_context: PreparedMainCoreContext | None,
        refs: ShortReferenceMap,
        *,
        fallback: str,
        occurred_at: datetime,
    ) -> tuple[DialoguePromptEntry, ...]:
        entries = self._structured_current_entries(
            prepared_context,
            refs,
            occurred_at=occurred_at,
        )
        if entries or not str(fallback or "").strip():
            return tuple(entries)
        cleaned = self._clean_model_text(escape_untrusted_identity_syntax(fallback), refs)
        if not cleaned:
            return ()
        public = self._next_user_reference(refs)
        return (
            DialoguePromptEntry(
                self._message_line(
                    public,
                    cleaned,
                    occurred_at=occurred_at,
                    member_ref="",
                    sender_name="",
                    refs=refs,
                ),
            ),
        )

    def _structured_current_entries(
        self,
        prepared_context: PreparedMainCoreContext | None,
        refs: ShortReferenceMap,
        *,
        occurred_at: datetime,
    ) -> list[DialoguePromptEntry]:
        entries: list[DialoguePromptEntry] = []
        for row in tuple(prepared_context.current_turn if prepared_context else ()):
            if not isinstance(row, Mapping):
                continue
            text = self._clean_model_text(row.get("text") or "", refs)
            if not text:
                continue
            ledger_id = int(row.get("ledger_message_id") or 0)
            speaker = str(row.get("speaker") or "user").lower()
            public = refs.message_by_ledger_id.get(ledger_id, "")
            if not public and speaker != "assistant":
                public = self._next_user_reference(refs)
            entries.append(
                DialoguePromptEntry(
                    self._message_line(
                        public,
                        text,
                        occurred_at=row.get("occurred_at") or occurred_at,
                        member_ref=str(row.get("member_ref") or ""),
                        sender_name=str(row.get("sender_name") or ""),
                        refs=refs,
                        speaker=speaker,
                    ),
                    ledger_message_id=ledger_id,
                )
            )
        return entries

    @staticmethod
    def _next_user_reference(refs: ShortReferenceMap) -> str:
        indexes = [
            int(value[1:])
            for value in refs.message_by_ledger_id.values()
            if value.startswith("U") and value[1:].isdigit()
        ]
        return f"U{max(indexes, default=0) + 1}"

    def _message_line(
        self,
        public: str,
        text: str,
        *,
        occurred_at: Any,
        member_ref: str,
        sender_name: str,
        refs: ShortReferenceMap,
        speaker: str = "",
    ) -> str:
        is_assistant = str(speaker or "").lower() == "assistant" or str(
            public or ""
        ).upper().startswith("A")
        if is_assistant:
            participant = "C"
        elif refs.identity_scope == "private":
            participant = "P1"
        else:
            participant = refs.participant_by_internal.get(member_ref, "")
        name = self._dialogue_display_name(
            public,
            sender_name=sender_name,
            refs=refs,
            speaker="assistant" if is_assistant else speaker,
        )
        marker = "__SOULCORE_TRUSTED_DIALOGUE_BODY__"
        rendered = xml_text(
            render_dialogue_line(
                marker if isinstance(text, TrustedPromptMarkup) else text,
                occurred_at=occurred_at,
                message_ref=public,
                participant_ref=participant,
                display_name=name,
            )
        )
        if isinstance(text, TrustedPromptMarkup):
            return TrustedPromptMarkup(rendered.replace(marker, str(text)))
        return rendered

    @staticmethod
    def _dialogue_display_name(
        public: str,
        *,
        sender_name: str,
        refs: ShortReferenceMap,
        speaker: str = "",
    ) -> str:
        if str(speaker or "").lower() == "assistant" or str(public or "").upper().startswith("A"):
            value = refs.character_name or sender_name
        elif refs.identity_scope == "private":
            value = resolve_private_display_name(
                sender_name,
                refs.private_display_name,
                override_enabled=refs.private_name_override_enabled,
            )
        else:
            value = sender_name
        return safe_model_identity(value)

    @staticmethod
    def _clean_model_text(value: str, refs: ShortReferenceMap) -> str:
        trusted = isinstance(value, TrustedPromptMarkup)
        content = str(value or "").strip()
        for internal, public in refs.internal_to_public.items():
            content = content.replace(internal, public)
        content = LONG_MESSAGE_REF.sub("", content)
        content = MEMBER_REF.sub("", content)
        content = OPAQUE_SENDER_PREFIX.sub("", content)
        cleaned = content.strip()
        return TrustedPromptMarkup(cleaned) if trusted else cleaned

    def _group_items(
        self,
        items: Sequence[ContextItem],
        refs: ShortReferenceMap,
    ) -> dict[ContextSource, list[str]]:
        groups: dict[ContextSource, list[str]] = {source: [] for source in ContextSource}
        for item in items:
            if item.source is ContextSource.CURRENT_PLAYER_MESSAGE:
                continue
            content = self._clean_item(item, refs)
            if content:
                groups.setdefault(item.source, []).append(content)
        return groups

    def _group_items_with_sources(
        self,
        items: Sequence[ContextItem],
        refs: ShortReferenceMap,
    ) -> tuple[
        dict[ContextSource, list[str]],
        dict[ContextSource, list[int]],
        dict[ContextSource, list[int]],
        dict[ContextSource, list[int]],
        dict[ContextSource, list[str]],
        dict[ContextSource, list[int]],
    ]:
        groups: dict[ContextSource, list[str]] = {source: [] for source in ContextSource}
        message_ids: dict[ContextSource, list[int]] = {source: [] for source in ContextSource}
        summary_ids: dict[ContextSource, list[int]] = {source: [] for source in ContextSource}
        sequences: dict[ContextSource, list[int]] = {source: [] for source in ContextSource}
        item_refs: dict[ContextSource, list[str]] = {source: [] for source in ContextSource}
        background_load_orders: dict[ContextSource, list[int]] = {
            source: [] for source in ContextSource
        }
        for item in items:
            if item.source is ContextSource.CURRENT_PLAYER_MESSAGE:
                continue
            content = self._clean_item(item, refs)
            if not content:
                continue
            groups[item.source].append(content)
            message_ids[item.source].append(int(item.metadata.get("ledger_message_id") or 0))
            summary_ids[item.source].append(self._summary_id(item))
            sequences[item.source].append(int(item.sequence))
            item_refs[item.source].append(str(item.metadata.get("background_ref") or ""))
            background_load_orders[item.source].append(
                int(item.metadata.get("background_load_order", -1))
            )
        return (
            groups,
            message_ids,
            summary_ids,
            sequences,
            item_refs,
            background_load_orders,
        )

    @staticmethod
    def _summary_id(item: ContextItem) -> int:
        value = item.item_id.partition(":")[2]
        return (
            int(value)
            if item.source is ContextSource.HISTORY_SUMMARY
            and item.item_id.startswith("summary:")
            and value.isdigit()
            else 0
        )

    def _clean_item(
        self,
        item: ContextItem,
        refs: ShortReferenceMap,
    ) -> str:
        if item.metadata.get("current_turn_shadow"):
            return ""
        content = self._clean_model_text(item.body or "", refs)
        if item.source is ContextSource.CURRENT_DIALOGUE:
            metadata = dict(item.metadata)
            if metadata.get("timeline_event_kind"):
                return xml_text(
                    render_dialogue_line(
                        content,
                        occurred_at=metadata.get("occurred_at"),
                    )
                )
            ledger_id = int(metadata.get("ledger_message_id") or 0)
            public = refs.message_by_ledger_id.get(ledger_id, "")
            if public:
                return self._message_line(
                    public,
                    content,
                    occurred_at=metadata.get("occurred_at"),
                    member_ref=str(metadata.get("member_ref") or ""),
                    sender_name=str(metadata.get("sender_name") or ""),
                    refs=refs,
                )
            if metadata.get("interrupted_unsent"):
                return self._message_line(
                    "",
                    content,
                    occurred_at=metadata.get("occurred_at"),
                    member_ref="",
                    sender_name=str(metadata.get("sender_name") or ""),
                    refs=refs,
                    speaker="assistant",
                )
        item_public = refs.item_public.get(str(item.item_id), "")
        if item_public:
            content = f"[{item_public}] {content}"
        return xml_text(content)


__all__ = ["RolePlayReferenceMixin"]
