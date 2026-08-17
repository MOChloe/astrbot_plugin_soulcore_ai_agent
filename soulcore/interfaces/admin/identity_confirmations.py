from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from secrets import token_urlsafe
from typing import Any

from ...features.identity import (
    CHARACTER_PLACEHOLDER,
    PRIVATE_USER_PLACEHOLDER,
    internal_identity_placeholders,
    validate_identity_template,
)
from .console_errors import ConsoleValidationError, require_successful_settings_result

SettingsSaver = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
WORLD_DEFINITION_FIELDS = (
    "world_brief",
    "world_rules",
    "life_direction",
    "world_texture",
    "expansion_policy",
)
WORLD_LORE_FIELDS = ("title", "content", "aliases", "tags", "importance")
WORLD_BOUNDARY_FIELDS = ("severity", "category", "rule_text", "positive_space", "enabled")


def identity_target_label(replacement: str) -> str:
    if replacement == CHARACTER_PLACEHOLDER:
        return "当前角色"
    if replacement == PRIVATE_USER_PLACEHOLDER:
        return "正在聊天的对方"
    if replacement.startswith("{[User:"):
        return "这位群成员"
    return "聊天中的人物"


def candidates_for_payload(
    identity: Any,
    value: Any,
    context: Any,
    payload: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Scan only fields changed by the current editor operation.

    ``None`` preserves full-document scanning for callers that do not yet send
    field-level change information. An explicit empty list means that the
    changed field is authoritative/non-semantic and needs no name annotation.
    """

    raw = payload.get("field_paths")
    if raw is None:
        paths = None
    elif not isinstance(raw, list) or any(
        not isinstance(item, str) or not item.startswith("$.") for item in raw
    ):
        raise ValueError("field_paths must be a list of document paths")
    else:
        paths = frozenset(raw)
    return identity.annotation_candidates_data(
        value,
        context=context,
        included_field_paths=paths,
    )


def editable_settings_value(section: str, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Project a settings snapshot into the exact value shape submitted by its editor."""

    if section == "character":
        model = snapshot.get("character_model")
        character = model if isinstance(model, Mapping) else snapshot
        return {
            "expected_revision": int(character.get("revision") or 0),
            "idempotency_key": "",
            "model": dict(character.get("model") or {}),
        }
    if section == "conversation":
        value = snapshot.get("scope_config")
        return dict(value) if isinstance(value, Mapping) else dict(snapshot)
    if section == "world":
        return _editable_world_settings(snapshot)
    if section.startswith("automation."):
        value = snapshot.get("config")
        return dict(value) if isinstance(value, Mapping) else dict(snapshot)
    return dict(snapshot)


def _editable_world_settings(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    value = snapshot.get("definition")
    definition = value if isinstance(value, Mapping) else snapshot
    return {
        "expected_revision": int(definition.get("revision") or 0),
        "world_brief": str(definition.get("world_brief") or ""),
        "world_rules": str(definition.get("world_rules") or ""),
        "life_direction": str(definition.get("life_direction") or ""),
        "world_texture": str(definition.get("world_texture") or ""),
        "expansion_policy": str(definition.get("expansion_policy") or "OPEN"),
    }


def settings_identity_scope(section: str, scope: object) -> str:
    """Return the identity-template scope owned by one settings editor."""

    if section == "conversation" or section.startswith("automation."):
        normalized = str(scope or "private").strip().lower()
        if normalized not in {"private", "group"}:
            raise ConsoleValidationError("聊天范围必须是私聊或群聊")
        return normalized
    return "profile"


async def existing_settings_action(
    action: str,
    profile_id: str,
    section: str,
    payload: Mapping[str, Any],
    *,
    identity: Any,
    character_models: Any,
    worlds: Any,
    confirmations: IdentityConfirmationGrants,
    save_settings: SettingsSaver,
) -> dict[str, Any]:
    """Review and save the authoritative stored value, never a browser-side form copy."""

    if section == "character":
        return await _existing_character_settings_action(
            action,
            profile_id,
            section,
            payload,
            identity=identity,
            character_models=character_models,
            confirmations=confirmations,
            save_settings=save_settings,
        )
    if section == "world":
        return await _existing_world_settings_action(
            action,
            profile_id,
            payload,
            identity=identity,
            worlds=worlds,
            confirmations=confirmations,
        )
    raise ConsoleValidationError("当前设置区域暂不支持检查已有姓名")


async def _existing_character_settings_action(
    action: str,
    profile_id: str,
    section: str,
    payload: Mapping[str, Any],
    *,
    identity: Any,
    character_models: Any,
    confirmations: IdentityConfirmationGrants,
    save_settings: SettingsSaver,
) -> dict[str, Any]:
    snapshot = await character_models.snapshot(profile_id)
    value = editable_settings_value(section, snapshot)
    context = await identity.profile_context(profile_id)
    candidates = identity.annotation_candidates_data(value, context=context)
    if action == "preview_existing":
        return {"ok": True, "candidates": candidate_views(candidates)}

    selected_ids = _validated_candidate_selection(payload, candidates, "角色资料")
    if not selected_ids:
        return {
            "ok": True,
            "saved": False,
            "result": snapshot,
            "identity_confirmation_token": confirmations.grant(
                profile_id, "profile", section, value
            ),
        }

    transformed = identity.apply_annotations(value, candidates, selected_ids)
    if not isinstance(transformed, Mapping):
        raise ConsoleValidationError("角色资料格式无效，请重新载入页面")
    merged = {**dict(transformed), "profile_id": profile_id, "scope": "private"}
    merged["idempotency_key"] = token_urlsafe(18)
    result = await save_settings(section, merged)
    require_successful_settings_result(result)
    return {
        "ok": True,
        "saved": True,
        "saved_at": datetime.now().astimezone().isoformat(),
        "result": result,
        "identity_confirmation_token": confirmations.grant(
            profile_id, "profile", section, transformed
        ),
    }


async def _existing_world_settings_action(
    action: str,
    profile_id: str,
    payload: Mapping[str, Any],
    *,
    identity: Any,
    worlds: Any,
    confirmations: IdentityConfirmationGrants,
) -> dict[str, Any]:
    snapshot = await worlds.world_snapshot(profile_id)
    value = _world_review_value(snapshot)
    context = await identity.profile_context(profile_id)
    candidates = identity.annotation_candidates_data(value, context=context)
    if action == "preview_existing":
        return {"ok": True, "candidates": candidate_views(candidates)}

    selected_ids = _validated_candidate_selection(payload, candidates, "世界设定")
    saved = False
    if selected_ids:
        transformed = identity.apply_annotations(value, candidates, selected_ids)
        if not isinstance(transformed, Mapping):
            raise ConsoleValidationError("世界设定格式无效，请重新载入页面")
        saved = await _save_world_review_value(worlds, profile_id, value, transformed)
    result = await worlds.world_snapshot(profile_id)
    editable_definition = editable_settings_value("world", result)
    response = {
        "ok": True,
        "saved": saved,
        "result": result,
        "identity_confirmation_token": confirmations.grant(
            profile_id, "profile", "world", editable_definition
        ),
    }
    if saved:
        response["saved_at"] = datetime.now().astimezone().isoformat()
    return response


def _world_review_value(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    definition = dict(snapshot.get("definition") or {})
    return {
        "definition": {
            "_revision": int(definition.get("revision") or 0),
            **{name: definition.get(name) for name in WORLD_DEFINITION_FIELDS},
        },
        "lore": {
            str(item.get("lore_id")): {
                "_lore_id": int(item.get("lore_id") or 0),
                "_revision": int(item.get("revision") or 0),
                **{name: item.get(name) for name in WORLD_LORE_FIELDS},
            }
            for item in snapshot.get("lore", ())
            if isinstance(item, Mapping)
        },
        "boundaries": {
            str(item.get("boundary_id")): {
                "_boundary_id": int(item.get("boundary_id") or 0),
                "_revision": int(item.get("revision") or 0),
                **{name: item.get(name) for name in WORLD_BOUNDARY_FIELDS},
            }
            for item in snapshot.get("boundaries", ())
            if isinstance(item, Mapping)
        },
    }


async def _save_world_review_value(
    worlds: Any,
    profile_id: str,
    original: Mapping[str, Any],
    transformed: Mapping[str, Any],
) -> bool:
    saved = False
    original_definition = _review_record(original, "definition")
    transformed_definition = _review_record(transformed, "definition")
    if _record_changed(original_definition, transformed_definition, WORLD_DEFINITION_FIELDS):
        await worlds.world_action(
            profile_id,
            {
                "action": "save_definition",
                "expected_revision": original_definition["_revision"],
                "value": {
                    name: transformed_definition.get(name) for name in WORLD_DEFINITION_FIELDS
                },
            },
        )
        saved = True
    saved = (
        await _save_world_review_records(
            worlds,
            profile_id,
            original,
            transformed,
            collection="lore",
            fields=WORLD_LORE_FIELDS,
            id_field="_lore_id",
            action="lore_update",
        )
        or saved
    )
    return (
        await _save_world_review_records(
            worlds,
            profile_id,
            original,
            transformed,
            collection="boundaries",
            fields=WORLD_BOUNDARY_FIELDS,
            id_field="_boundary_id",
            action="boundary_update",
        )
        or saved
    )


async def _save_world_review_records(
    worlds: Any,
    profile_id: str,
    original: Mapping[str, Any],
    transformed: Mapping[str, Any],
    *,
    collection: str,
    fields: tuple[str, ...],
    id_field: str,
    action: str,
) -> bool:
    saved = False
    original_rows = _review_record(original, collection)
    transformed_rows = _review_record(transformed, collection)
    for key, original_row in original_rows.items():
        if not isinstance(original_row, Mapping):
            continue
        transformed_row = transformed_rows.get(key)
        if not isinstance(transformed_row, Mapping):
            raise ConsoleValidationError("世界设定已经发生变化，请重新检查姓名")
        if not _record_changed(original_row, transformed_row, fields):
            continue
        await worlds.world_action(
            profile_id,
            {
                "action": action,
                id_field.removeprefix("_"): int(original_row[id_field]),
                "expected_revision": int(original_row["_revision"]),
                "value": {name: transformed_row.get(name) for name in fields},
            },
        )
        saved = True
    return saved


def _review_record(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    item = value.get(name)
    if not isinstance(item, Mapping):
        raise ConsoleValidationError("世界设定已经发生变化，请重新检查姓名")
    return dict(item)


def _record_changed(
    original: Mapping[str, Any], transformed: Mapping[str, Any], fields: tuple[str, ...]
) -> bool:
    return any(original.get(name) != transformed.get(name) for name in fields)


def _validated_candidate_selection(
    payload: Mapping[str, Any], candidates: tuple[Any, ...], label: str
) -> set[str]:
    offered_ids = _candidate_ids(payload.get("offered_ids"), "offered_ids")
    selected_ids = _candidate_ids(payload.get("selected_ids"), "selected_ids")
    current_ids = {item.candidate_id for item in candidates}
    if offered_ids != current_ids:
        raise ConsoleValidationError(f"{label}已经发生变化，请重新检查姓名")
    if not selected_ids.issubset(offered_ids):
        raise ConsoleValidationError("本次选择已经失效，请重新检查姓名")
    return selected_ids


def candidate_views(candidates: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": item.candidate_id,
            "field_path": item.field_path,
            "matched_text": item.matched_text,
            "target_label": identity_target_label(item.replacement),
            "context": item.context,
        }
        for item in candidates
    ]


def _candidate_ids(value: Any, field: str) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConsoleValidationError(f"{field} 必须是姓名候选列表")
    return {item for item in value if item}


class IdentityConfirmationGrants:
    """Bind confirmed placeholder positions to one exact settings value."""

    def __init__(self) -> None:
        self._grants: dict[
            str,
            tuple[str, str, str, str, tuple[tuple[str, tuple[str, ...]], ...]],
        ] = {}

    def grant(self, profile_id: str, scope: str, section: str, value: Any) -> str:
        token = token_urlsafe(24)
        self._grants[token] = (
            profile_id,
            scope,
            section,
            self._value_digest(value),
            self._signature(value),
        )
        if len(self._grants) > 2048:
            self._grants.pop(next(iter(self._grants)), None)
        return token

    def require(
        self,
        profile_id: str,
        scope: str,
        section: str,
        value: Any,
        token: str,
    ) -> None:
        signature = self._signature(value)
        if not signature:
            return
        self._require_grant(profile_id, scope, section, value, token, signature)

    def require_exact(
        self,
        profile_id: str,
        scope: str,
        section: str,
        value: Any,
        token: str,
    ) -> None:
        """Require a preview grant even when the confirmed value has no placeholders."""

        signature = self._signature(value)
        self._require_grant(profile_id, scope, section, value, token, signature)

    def _require_grant(
        self,
        profile_id: str,
        scope: str,
        section: str,
        value: Any,
        token: str,
        signature: tuple[tuple[str, tuple[str, ...]], ...],
    ) -> None:
        if self._grants.get(token) != (
            profile_id,
            scope,
            section,
            self._value_digest(value),
            signature,
        ):
            raise ConsoleValidationError("这些内容中的人物名称还没有确认，请重新确认后保存")
        for _path, placeholders in signature:
            validate_identity_template("".join(placeholders), scope=scope)

    @classmethod
    def _signature(cls, value: Any, path: str = "$") -> tuple[tuple[str, tuple[str, ...]], ...]:
        rows: list[tuple[str, tuple[str, ...]]] = []
        cls._collect(value, path, rows)
        return tuple(rows)

    @staticmethod
    def _value_digest(value: Any) -> str:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @classmethod
    def _collect(
        cls,
        value: Any,
        path: str,
        rows: list[tuple[str, tuple[str, ...]]],
    ) -> None:
        if isinstance(value, str):
            placeholders = internal_identity_placeholders(value)
            if placeholders:
                rows.append((path, placeholders))
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                cls._collect(child, f"{path}.{key}", rows)
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                cls._collect(child, f"{path}[{index}]", rows)


def initial_settings_grants(
    grants: IdentityConfirmationGrants,
    profile_id: str,
    scope: str,
    snapshots: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Issue grants against each editor's actual submission shape."""

    return {
        section: grants.grant(
            profile_id,
            settings_identity_scope(section, scope),
            section,
            editable_settings_value(section, snapshot),
        )
        for section, snapshot in snapshots.items()
    }
