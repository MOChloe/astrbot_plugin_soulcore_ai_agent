"""Application service for stable chat identities and current display names."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any

from ...contracts.message_reference import safe_model_identity
from .domain import (
    CHARACTER_PLACEHOLDER,
    IDENTITY_REFERENCE_GUIDANCE,
    PRIVATE_USER_PLACEHOLDER,
    IdentityAnnotationCandidate,
    IdentityCatalog,
    IdentityParticipant,
    IdentityRenderContext,
    build_identity_catalog,
    decode_model_identity_text,
    encode_identity_template_for_model,
    group_user_placeholder,
    identity_annotation_candidates,
    project_identity_text_for_model,
    render_identity_text,
    resolve_private_display_name,
)


def identity_reference_guidance() -> str:
    """Expose shared model guidance through the feature's public service boundary."""

    return IDENTITY_REFERENCE_GUIDANCE


class IdentityService:
    def __init__(self, profiles: Any, character_models: Any) -> None:
        self.profiles = profiles
        self.character_models = character_models
        self._group_refresh_after: dict[tuple[str, str], float] = {}
        self._group_refresh_locks: dict[tuple[str, str], asyncio.Lock] = {}

    async def observe_participant(
        self,
        profile_id: str,
        instance_id: str,
        *,
        participant_id: str,
        display_name: str,
        source: str = "OBSERVED",
        message_id: int | None = None,
    ) -> None:
        if not str(participant_id or "").strip():
            return
        normalized_name = safe_model_identity(display_name)
        if normalized_name in {"对方", "一位群成员"}:
            normalized_name = ""
        await self.profiles.upsert_participant_identity(
            profile_id,
            instance_id,
            participant_id=str(participant_id).strip(),
            display_name=normalized_name,
            name_source=str(source or "OBSERVED").strip().upper(),
            last_message_id=message_id,
        )

    async def refresh_group_directory(
        self,
        profile_id: str,
        instance_id: str,
        event: Any,
        *,
        ttl_seconds: float = 3600.0,
    ) -> None:
        """Best-effort OneBot refresh; unsupported platforms keep last-seen names."""

        key = (profile_id, instance_id)
        now = time.monotonic()
        if self._group_refresh_after.get(key, 0.0) > now:
            return
        lock = self._group_refresh_locks.setdefault(key, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            if self._group_refresh_after.get(key, 0.0) > now:
                return
            rows = await self._load_group_member_rows(event)
            if rows is None:
                return
            if rows:
                await self.profiles.upsert_participant_identities(
                    profile_id,
                    instance_id,
                    rows=tuple(rows),
                    name_source="PLATFORM_REFRESH",
                )
            self._group_refresh_after[key] = time.monotonic() + max(
                60.0,
                float(ttl_seconds),
            )

    @classmethod
    async def _load_group_member_rows(cls, event: Any) -> list[tuple[str, str]] | None:
        getter = getattr(event, "get_group", None)
        if not callable(getter):
            return None
        try:
            group = await asyncio.wait_for(getter(), timeout=8.0)
        except Exception:
            return None
        if group is None:
            return None
        members = (
            group.get("members", ()) if isinstance(group, dict) else getattr(group, "members", ())
        )
        return [row for member in members or () if (row := cls._member_row(member))]

    @staticmethod
    def _member_row(member: Any) -> tuple[str, str] | None:
        if isinstance(member, dict):
            participant_id = member.get("user_id") or member.get("member_openid")
            display_name = member.get("card") or member.get("nickname")
        else:
            participant_id = getattr(member, "user_id", "") or getattr(member, "member_openid", "")
            display_name = getattr(member, "card", "") or getattr(member, "nickname", "")
        normalized_id = str(participant_id or "").strip()
        if not normalized_id:
            return None
        return normalized_id, safe_model_identity(str(display_name or ""))

    @staticmethod
    def _participants(
        rows: list[Mapping[str, Any]],
        *,
        scope: str,
        private_fallback_player_name: str,
        private_name_override_enabled: bool,
        participant_fallback: str,
    ) -> tuple[IdentityParticipant, ...]:
        participants: list[IdentityParticipant] = []
        for row in rows:
            participant_id = str(row.get("participant_id") or "")
            if not participant_id.strip():
                continue
            observed_name = str(row.get("display_name") or "")
            display_name = (
                resolve_private_display_name(
                    observed_name,
                    private_fallback_player_name,
                    override_enabled=private_name_override_enabled,
                )
                if scope == "private"
                else safe_model_identity(observed_name) or participant_fallback
            )
            last_message_id = row.get("last_message_id")
            participants.append(
                IdentityParticipant(
                    participant_id=participant_id,
                    display_name=display_name,
                    name_source=str(row.get("name_source") or "OBSERVED"),
                    last_message_id=(int(last_message_id) if last_message_id is not None else None),
                )
            )
        return tuple(participants)

    async def context(self, profile_id: str, instance_id: str) -> IdentityRenderContext:
        instance = await self.profiles.get_character_instance(profile_id, instance_id)
        if instance is None:
            raise KeyError((profile_id, instance_id))
        snapshot = await self.character_models.get_current(profile_id)
        name = str(snapshot.model.identity.name if snapshot is not None else "").strip()
        rows = await self.profiles.list_participant_identities(profile_id, instance_id)
        scope = str(instance.scope)
        chat_policy = await self.profiles.get_instance_chat_policy(profile_id, instance_id)
        private_fallback_player_name = (
            safe_model_identity(str(chat_policy.private_fallback_player_name or ""))
            if scope == "private"
            else ""
        )
        private_name_override_enabled = bool(
            scope == "private"
            and private_fallback_player_name
            and chat_policy.private_name_override_enabled
        )
        participant_fallback = "对方" if scope == "private" else "一位群成员"
        participants = self._participants(
            rows,
            scope=scope,
            private_fallback_player_name=private_fallback_player_name,
            private_name_override_enabled=private_name_override_enabled,
            participant_fallback=participant_fallback,
        )
        return IdentityRenderContext(
            profile_id=profile_id,
            instance_id=instance_id,
            scope=scope,
            character_name=name or "角色本人",
            participants=participants,
            private_fallback_player_name=private_fallback_player_name,
            private_name_override_enabled=private_name_override_enabled,
        )

    async def profile_context(self, profile_id: str) -> IdentityRenderContext:
        snapshot = await self.character_models.get_current(profile_id)
        name = str(snapshot.model.identity.name if snapshot is not None else "").strip()
        return IdentityRenderContext(
            profile_id=profile_id,
            instance_id="",
            scope="profile",
            character_name=name or "角色本人",
        )

    async def catalog(
        self,
        profile_id: str,
        instance_id: str,
        *,
        participant_ids: tuple[str, ...] | None = None,
        participant_references: Mapping[str, str] | None = None,
    ) -> tuple[IdentityRenderContext, IdentityCatalog]:
        context = await self.context(profile_id, instance_id)
        return context, build_identity_catalog(
            context,
            participant_ids=participant_ids,
            participant_references=participant_references,
        )

    @staticmethod
    def render(value: str, context: IdentityRenderContext) -> str:
        return render_identity_text(value, context)

    @classmethod
    def render_data(cls, value: Any, context: IdentityRenderContext) -> Any:
        if isinstance(value, str):
            return cls.render(value, context)
        if isinstance(value, dict):
            return {key: cls.render_data(item, context) for key, item in value.items()}
        if isinstance(value, list):
            return [cls.render_data(item, context) for item in value]
        if isinstance(value, tuple):
            return tuple(cls.render_data(item, context) for item in value)
        return value

    @staticmethod
    def encode_for_model(value: str, catalog: IdentityCatalog) -> str:
        return encode_identity_template_for_model(value, catalog)

    @staticmethod
    def project_for_model(
        value: str,
        catalog: IdentityCatalog,
        *,
        scope: str,
    ) -> str:
        return project_identity_text_for_model(value, catalog, scope=scope)

    @classmethod
    def encode_data_for_model(cls, value: Any, catalog: IdentityCatalog) -> Any:
        if isinstance(value, str):
            return cls.encode_for_model(value, catalog)
        if isinstance(value, dict):
            return {key: cls.encode_data_for_model(item, catalog) for key, item in value.items()}
        if isinstance(value, list):
            return [cls.encode_data_for_model(item, catalog) for item in value]
        if isinstance(value, tuple):
            return tuple(cls.encode_data_for_model(item, catalog) for item in value)
        return value

    @classmethod
    def project_data_for_model(
        cls,
        value: Any,
        catalog: IdentityCatalog,
        *,
        scope: str,
    ) -> Any:
        if isinstance(value, str):
            return cls.project_for_model(value, catalog, scope=scope)
        if isinstance(value, dict):
            return {
                key: cls.project_data_for_model(item, catalog, scope=scope)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls.project_data_for_model(item, catalog, scope=scope) for item in value]
        if isinstance(value, tuple):
            return tuple(cls.project_data_for_model(item, catalog, scope=scope) for item in value)
        return value

    @staticmethod
    def decode_model(
        value: str,
        catalog: IdentityCatalog,
        *,
        scope: str,
    ) -> str:
        return str(decode_model_identity_text(value, catalog, scope=scope))

    @staticmethod
    def annotation_candidates(
        value: str,
        *,
        field_path: str,
        context: IdentityRenderContext,
    ) -> tuple[IdentityAnnotationCandidate, ...]:
        identities: list[tuple[str, str]] = [(context.character_name, CHARACTER_PLACEHOLDER)]
        if context.scope == "private" and context.participants:
            identities.append((context.private_display_name, PRIVATE_USER_PLACEHOLDER))
        elif context.scope == "group":
            identities.extend(
                (item.display_name, group_user_placeholder(item.participant_id))
                for item in context.participants
                if item.display_name
            )
        return identity_annotation_candidates(
            value,
            field_path=field_path,
            identities=tuple(identities),
        )

    @classmethod
    def annotation_candidates_data(
        cls,
        value: Any,
        *,
        context: IdentityRenderContext,
        included_field_paths: frozenset[str] | None = None,
    ) -> tuple[IdentityAnnotationCandidate, ...]:
        result: list[IdentityAnnotationCandidate] = []

        def path_is_included(path: str) -> bool:
            if included_field_paths is None:
                return True
            return any(
                path == selected
                or path.startswith(f"{selected}[")
                or path.startswith(f"{selected}.")
                for selected in included_field_paths
            )

        def visit(item: Any, path: str) -> None:
            # identity.name is the authoritative source used to render
            # {[character]}. It must never be rewritten into its own placeholder.
            if path.endswith(".identity.name"):
                return
            if isinstance(item, str):
                if not path_is_included(path):
                    return
                result.extend(cls.annotation_candidates(item, field_path=path, context=context))
                return
            if isinstance(item, dict):
                for key, child in item.items():
                    if str(key).startswith("_"):
                        continue
                    visit(child, f"{path}.{key}")
                return
            if isinstance(item, (list, tuple)):
                for index, child in enumerate(item):
                    visit(child, f"{path}[{index}]")

        visit(value, "$")
        return tuple(result)

    @staticmethod
    def apply_annotations(
        value: Any,
        candidates: tuple[IdentityAnnotationCandidate, ...],
        selected_ids: set[str],
    ) -> Any:
        by_path: dict[str, list[IdentityAnnotationCandidate]] = {}
        for candidate in candidates:
            if candidate.candidate_id in selected_ids:
                by_path.setdefault(candidate.field_path, []).append(candidate)

        def apply(item: Any, path: str) -> Any:
            if isinstance(item, str):
                text = item
                for candidate in sorted(
                    by_path.get(path, ()),
                    key=lambda row: (row.start, row.end),
                    reverse=True,
                ):
                    if text[candidate.start : candidate.end] != candidate.matched_text:
                        raise ValueError("姓名确认内容已经变化，请重新检查")
                    text = text[: candidate.start] + candidate.replacement + text[candidate.end :]
                return text
            if isinstance(item, dict):
                return {key: apply(child, f"{path}.{key}") for key, child in item.items()}
            if isinstance(item, list):
                return [apply(child, f"{path}[{index}]") for index, child in enumerate(item)]
            if isinstance(item, tuple):
                return tuple(apply(child, f"{path}[{index}]") for index, child in enumerate(item))
            return item

        return apply(value, "$")


__all__ = ["IdentityService"]
