"""Sparse character projection and deterministic import merge semantics."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..character_model import (
    BackgroundCreationPrompts,
    BackgroundCreationPromptSelections,
    CapabilityProfile,
    CharacterCustomPrompts,
    CharacterIdentity,
    CharacterModel,
    CharacterPromptSelections,
    CharacterTriggerRule,
    LanguageProfile,
    MainCoreModePrompts,
    MainCoreModePromptSelections,
    MainCoreStylePrompts,
    MainCoreStylePromptSelections,
    PersonalityProfile,
    PreferenceProfile,
    ResponsePolishPrompts,
    ResponsePolishPromptSelections,
    SocialProfile,
    StoryStylePrompts,
    StoryStylePromptSelections,
    VisualProfile,
    model_to_payload,
    normalize_character_model,
)
from .domain import ImportState, ParsedRolePackage, RoleDatabaseSnapshot, RolePackageError

CHARACTER_SECTION_FIELDS: dict[str, frozenset[str]] = {
    "identity": frozenset({"name", "aliases", "overview", "facts"}),
    "personality": frozenset({"traits_and_values", "thinking_and_behavior", "habits_and_emotions"}),
    "social": frozenset({"interaction_style", "boundaries"}),
    "preferences": frozenset({"likes_and_interests", "dislikes"}),
    "language": frozenset({"speaking_style", "messaging_habits", "address_habits"}),
    "visual": frozenset({"appearance", "clothing", "visual_boundaries"}),
    "capabilities": frozenset({"abilities", "knowledge_scope", "limitations"}),
}

PROMPT_GROUP_FIELDS: dict[str, frozenset[str]] = {
    "main_core_modes": frozenset({"self_initiated"}),
    "main_core_styles": frozenset(
        {
            "relationship_context",
            "speaking_style",
            "sticker_style",
            "thinking_style",
            "content_style",
            "conversation_content",
        }
    ),
    "response_polish": frozenset({"writing_correction"}),
    "story_styles": frozenset({"involvement", "stance"}),
    "background_creation": frozenset(
        {"world_change", "story_boundary", "imagination", "temperature"}
    ),
}

CHARACTER_ROOT_FIELDS = frozenset(
    set(CHARACTER_SECTION_FIELDS) | {"custom_prompts", "dialogue_reference", "trigger_rules"}
)
WORLD_DEFINITION_FIELDS = frozenset(
    {"world_brief", "world_rules", "life_direction", "world_texture", "expansion_policy"}
)
WORLD_ROOT_FIELDS = frozenset({"definition", "lore", "boundaries"})
PORTRAIT_SCOPES = ("private", "group")

_REQUIRED_PROMPTS = frozenset(
    {
        ("main_core_styles", "relationship_context"),
        ("background_creation", "story_boundary"),
    }
)
_SCOPED_IDENTITY = re.compile(r"\{\[(?:User|character):[^\]\r\n]*\]\}")
_MISSING = object()


def sparse_character_payload(model: CharacterModel) -> dict[str, Any]:
    """Return only portable character content differing from current built-ins."""

    current = model_to_payload(model)
    default = model_to_payload(CharacterModel())
    current.pop("schema_version", None)
    current.pop("prompt_selections", None)
    default.pop("schema_version", None)
    default.pop("prompt_selections", None)
    difference = _sparse_difference(current, default)
    return difference if isinstance(difference, dict) else {}


def sparse_role_document(snapshot: RoleDatabaseSnapshot) -> dict[str, Any]:
    """Project one database snapshot into the version-one sparse role document."""

    document: dict[str, Any] = {}
    character = sparse_character_payload(snapshot.character)
    if character:
        document["character"] = character

    definition_default = {
        "world_brief": "",
        "world_rules": "",
        "life_direction": "",
        "world_texture": "",
        "expansion_policy": "OPEN",
    }
    definition = _sparse_difference(snapshot.world_definition, definition_default)
    world: dict[str, Any] = {}
    if isinstance(definition, dict) and definition:
        world["definition"] = definition
    if snapshot.lore:
        world["lore"] = [dict(item) for item in snapshot.lore]
    if snapshot.boundaries:
        world["boundaries"] = [dict(item) for item in snapshot.boundaries]
    if world:
        document["world"] = world
    return document


def merge_import_state(
    package: ParsedRolePackage,
    snapshot: RoleDatabaseSnapshot,
) -> ImportState:
    document = package.role
    character = _merged_import_character(document, snapshot)
    definition, lore, boundaries, world_presence = _merged_import_world(document, snapshot)
    actions, portrait_changed = _portrait_import_actions(package, snapshot)
    character_changed = model_to_payload(character) != model_to_payload(snapshot.character)
    world_changed = _world_import_changed(definition, lore, boundaries, snapshot)
    return ImportState(
        character=character,
        world_definition=definition,
        lore=lore,
        boundaries=boundaries,
        world_definition_present=world_presence[0],
        lore_present=world_presence[1],
        boundaries_present=world_presence[2],
        portrait_actions=actions,
        changed=character_changed or world_changed or any(portrait_changed.values()),
        character_changed=character_changed,
        world_changed=world_changed,
        portrait_changed=portrait_changed,
    )


def _merged_import_character(
    document: Mapping[str, Any], snapshot: RoleDatabaseSnapshot
) -> CharacterModel:
    patch = document.get("character")
    return (
        _merge_character(snapshot.character, patch)
        if isinstance(patch, Mapping)
        else snapshot.character
    )


def _merged_import_world(
    document: Mapping[str, Any], snapshot: RoleDatabaseSnapshot
) -> tuple[
    dict[str, Any],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[bool, bool, bool],
]:
    world = document.get("world")
    patch = dict(world) if isinstance(world, Mapping) else {}
    definition_present = "definition" in patch
    lore_present = "lore" in patch
    boundaries_present = "boundaries" in patch
    definition = dict(snapshot.world_definition)
    if definition_present:
        definition.update(dict(patch["definition"]))
    lore = tuple(dict(item) for item in patch["lore"]) if lore_present else snapshot.lore
    boundaries = (
        tuple(dict(item) for item in patch["boundaries"])
        if boundaries_present
        else snapshot.boundaries
    )
    return (
        definition,
        lore,
        boundaries,
        (
            definition_present,
            lore_present,
            boundaries_present,
        ),
    )


def _portrait_import_actions(
    package: ParsedRolePackage, snapshot: RoleDatabaseSnapshot
) -> tuple[dict[str, dict[str, Any]], dict[str, bool]]:
    root = package.role.get("portraits")
    patch = dict(root) if isinstance(root, Mapping) else {}
    actions: dict[str, dict[str, Any]] = {}
    portrait_changed: dict[str, bool] = {}
    for scope in PORTRAIT_SCOPES:
        actions[scope], portrait_changed[scope] = _portrait_import_action(
            package, snapshot, scope, patch.get(scope)
        )
    return actions, portrait_changed


def _portrait_import_action(
    package: ParsedRolePackage,
    snapshot: RoleDatabaseSnapshot,
    scope: str,
    value: Any,
) -> tuple[dict[str, Any], bool]:
    current = snapshot.portraits.get(scope)
    if not isinstance(value, Mapping):
        return {"action": "keep"}, False
    if value.get("clear") is True:
        return {"action": "clear"}, current is not None
    asset = package.assets[scope]
    label = str(value.get("label") or "")
    action = {
        "action": "replace",
        "asset_path": asset.path,
        "sha256": asset.sha256,
        "label": label,
    }
    unchanged = current is not None and current.sha256 == asset.sha256 and current.label == label
    return action, not unchanged


def _world_import_changed(
    definition: Mapping[str, Any],
    lore: Sequence[Mapping[str, Any]],
    boundaries: Sequence[Mapping[str, Any]],
    snapshot: RoleDatabaseSnapshot,
) -> bool:
    return (
        definition != snapshot.world_definition
        or _canonical_records(lore) != _canonical_records(snapshot.lore)
        or _canonical_records(boundaries) != _canonical_records(snapshot.boundaries)
    )


def scoped_identity_fields(value: Any) -> tuple[str, ...]:
    """Return field paths only; never echo the participant token itself."""

    result: list[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, str):
            if _SCOPED_IDENTITY.search(item):
                result.append(path or "role")
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{path}.{key}" if path else str(key))
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "")
    return tuple(result)


def require_portable_identities(value: Any) -> None:
    fields = scoped_identity_fields(value)
    if not fields:
        return
    visible = "、".join(fields[:4])
    suffix = "等字段" if len(fields) > 4 else ""
    raise RolePackageError(
        f"角色包不能包含绑定具体群成员的身份标记；请先处理 {visible}{suffix}。",
        field=fields[0],
    )


def _sparse_difference(current: Any, default: Any) -> Any:
    if isinstance(current, dict) and isinstance(default, dict):
        result: dict[str, Any] = {}
        for key in sorted(current):
            difference = _sparse_difference(current[key], default.get(key, _MISSING))
            if difference is not _MISSING:
                result[key] = difference
        return result if result else _MISSING
    if current == default:
        return _MISSING
    if isinstance(current, list) and not current:
        return _MISSING
    return copy.deepcopy(current)


def _merge_character(current: CharacterModel, patch: Mapping[str, Any]) -> CharacterModel:
    payload = copy.deepcopy(model_to_payload(current))
    default = model_to_payload(CharacterModel())
    _recursive_merge(payload, patch)
    prompt_patch = patch.get("custom_prompts")
    if isinstance(prompt_patch, Mapping):
        for group, raw_fields in prompt_patch.items():
            if not isinstance(raw_fields, Mapping):
                continue
            for field, value in raw_fields.items():
                if value == "" and (str(group), str(field)) in _REQUIRED_PROMPTS:
                    payload["custom_prompts"][group][field] = default["custom_prompts"][group][
                        field
                    ]
                    payload["prompt_selections"][group][field] = default["prompt_selections"][
                        group
                    ][field]
                else:
                    payload["prompt_selections"][group][field] = "custom"
    try:
        return normalize_character_model(_model_from_payload(payload))
    except (KeyError, TypeError, ValueError) as exc:
        raise RolePackageError(f"角色资料不符合当前 SoulCore 规则：{exc}") from exc


def _recursive_merge(target: dict[str, Any], patch: Mapping[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _recursive_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _model_from_payload(value: Mapping[str, Any]) -> CharacterModel:
    prompts = value["custom_prompts"]
    selections = value["prompt_selections"]
    return CharacterModel(
        identity=CharacterIdentity(**_tuple_fields(value["identity"], {"aliases", "facts"})),
        personality=PersonalityProfile(**_all_tuple_fields(value["personality"])),
        social=SocialProfile(**_all_tuple_fields(value["social"])),
        preferences=PreferenceProfile(**_all_tuple_fields(value["preferences"])),
        language=LanguageProfile(**_all_tuple_fields(value["language"])),
        custom_prompts=CharacterCustomPrompts(
            main_core_modes=MainCoreModePrompts(**prompts["main_core_modes"]),
            main_core_styles=MainCoreStylePrompts(**prompts["main_core_styles"]),
            response_polish=ResponsePolishPrompts(**prompts["response_polish"]),
            story_styles=StoryStylePrompts(**prompts["story_styles"]),
            background_creation=BackgroundCreationPrompts(**prompts["background_creation"]),
        ),
        prompt_selections=CharacterPromptSelections(
            main_core_modes=MainCoreModePromptSelections(**selections["main_core_modes"]),
            main_core_styles=MainCoreStylePromptSelections(**selections["main_core_styles"]),
            response_polish=ResponsePolishPromptSelections(**selections["response_polish"]),
            story_styles=StoryStylePromptSelections(**selections["story_styles"]),
            background_creation=BackgroundCreationPromptSelections(
                **selections["background_creation"]
            ),
        ),
        dialogue_reference=str(value["dialogue_reference"]),
        visual=VisualProfile(**_all_tuple_fields(value["visual"])),
        capabilities=CapabilityProfile(**_all_tuple_fields(value["capabilities"])),
        trigger_rules=tuple(
            CharacterTriggerRule(
                keys=tuple(item["keys"]),
                lookback_turns=item["lookback_turns"],
                content=item["content"],
            )
            for item in value["trigger_rules"]
        ),
    )


def _tuple_fields(value: Mapping[str, Any], fields: set[str]) -> dict[str, Any]:
    return {key: tuple(item) if key in fields else item for key, item in value.items()}


def _all_tuple_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: tuple(item) for key, item in value.items()}


def _canonical_records(values: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    import json

    return tuple(
        sorted(
            json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for value in values
        )
    )


__all__ = [
    "CHARACTER_ROOT_FIELDS",
    "CHARACTER_SECTION_FIELDS",
    "PORTRAIT_SCOPES",
    "PROMPT_GROUP_FIELDS",
    "WORLD_DEFINITION_FIELDS",
    "WORLD_ROOT_FIELDS",
    "merge_import_state",
    "require_portable_identities",
    "scoped_identity_fields",
    "sparse_character_payload",
    "sparse_role_document",
]
