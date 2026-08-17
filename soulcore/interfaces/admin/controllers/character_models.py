"""Administrator JSON boundary for profile-owned character models."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from ....features.character_model.domain import (
    MAX_IDEMPOTENCY_KEY_CHARS,
    MAX_LIST_ITEM_CHARS,
    MAX_LIST_ITEMS,
    MAX_MODEL_BYTES,
    MAX_NAME_CHARS,
    MAX_TEXT_CHARS,
    MAX_TRIGGER_CONTENT_CHARS,
    MAX_TRIGGER_KEY_CHARS,
    MAX_TRIGGER_KEYS,
    MAX_TRIGGER_LOOKBACK_TURNS,
    MAX_TRIGGER_RULES,
    MAX_TRIGGER_TOTAL_CONTENT_CHARS,
    MIN_TRIGGER_LOOKBACK_TURNS,
    MODEL_SCHEMA_VERSION,
    BackgroundCreationPrompts,
    CapabilityProfile,
    CharacterCustomPrompts,
    CharacterIdentity,
    CharacterModel,
    CharacterModelError,
    CharacterModelIdempotencyConflict,
    CharacterModelRevisionConflict,
    CharacterModelSnapshot,
    CharacterTriggerRule,
    LanguageProfile,
    MainCoreModePrompts,
    MainCoreStylePrompts,
    PersonalityProfile,
    PreferenceProfile,
    ResponsePolishPrompts,
    SocialProfile,
    StoryStylePrompts,
    VisualProfile,
    model_to_payload,
)
from ....features.character_model.ports import CharacterModelAdminPort
from ....features.character_model.prompt_selections import (
    MAX_PROMPT_PRESET_ID_CHARS,
    BackgroundCreationPromptSelections,
    CharacterPromptSelections,
    MainCoreModePromptSelections,
    MainCoreStylePromptSelections,
    ResponsePolishPromptSelections,
    StoryStylePromptSelections,
)

_FIELD_PREFIX = re.compile(r"^([a-z_]+(?:\.[a-z_]+|\[\d+\])*(?:\.[a-z_]+)?)(?=\s|$)")


class CharacterModelPayloadError(CharacterModelError):
    def __init__(self, field: str, message: str) -> None:
        self.field_errors = {field: message}
        super().__init__(message)


class CharacterModelsAdminController:
    """Parse complete snapshots and expose deterministic local error metadata."""

    def __init__(self, service: CharacterModelAdminPort) -> None:
        self.service = service

    async def snapshot(self, profile_id: str) -> dict[str, Any]:
        snapshot = await self.service.get_current(profile_id)
        return {"ok": True, "character_model": _snapshot_view(profile_id, snapshot)}

    async def save(self, profile_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            expected_revision = _non_negative_int(
                payload.get("expected_revision"), "expected_revision"
            )
            idempotency_key = payload.get("idempotency_key")
            if not isinstance(idempotency_key, str):
                raise CharacterModelPayloadError("idempotency_key", "幂等键必须是字符串。")
            model = _parse_model(payload.get("model"))
            snapshot = await self.service.save_model(
                profile_id,
                model,
                expected_revision=expected_revision,
                idempotency_key=idempotency_key,
            )
        except CharacterModelRevisionConflict:
            current = await self.service.get_current(profile_id)
            return _error_view(
                "revision_conflict",
                "角色模型已被其他管理员更新；当前修改尚未覆盖服务端内容。",
                current=_snapshot_view(profile_id, current),
            )
        except CharacterModelIdempotencyConflict as exc:
            return _error_view(
                "idempotency_conflict",
                "本次保存标识已用于其他内容，请重新编辑后再保存。",
                field_errors={"idempotency_key": str(exc)},
            )
        except CharacterModelPayloadError as exc:
            return _error_view("validation_error", str(exc), field_errors=exc.field_errors)
        except CharacterModelError as exc:
            field = _error_field(str(exc))
            errors = {field: str(exc)} if field else {}
            return _error_view("validation_error", str(exc), field_errors=errors)
        return {"ok": True, "character_model": _snapshot_view(profile_id, snapshot)}


def _snapshot_view(profile_id: str, snapshot: CharacterModelSnapshot | None) -> dict[str, Any]:
    model = snapshot.model if snapshot is not None else CharacterModel()
    return {
        "profile_id": profile_id,
        "revision": snapshot.revision if snapshot is not None else 0,
        "content_fingerprint": snapshot.content_fingerprint if snapshot is not None else "",
        "saved_at": snapshot.saved_at.isoformat() if snapshot is not None else None,
        "model": model_to_payload(model),
        "limits": {
            "schema_version": MODEL_SCHEMA_VERSION,
            "name_chars": MAX_NAME_CHARS,
            "text_chars": MAX_TEXT_CHARS,
            "list_item_chars": MAX_LIST_ITEM_CHARS,
            "list_items": MAX_LIST_ITEMS,
            "dialogue_reference_chars": MAX_TEXT_CHARS,
            "custom_prompt_chars": MAX_TEXT_CHARS,
            "prompt_preset_id_chars": MAX_PROMPT_PRESET_ID_CHARS,
            "main_core_mode_prompt_chars": MAX_TEXT_CHARS,
            "model_bytes": MAX_MODEL_BYTES,
            "idempotency_key_chars": MAX_IDEMPOTENCY_KEY_CHARS,
            "trigger_rules": MAX_TRIGGER_RULES,
            "trigger_keys": MAX_TRIGGER_KEYS,
            "trigger_key_chars": MAX_TRIGGER_KEY_CHARS,
            "trigger_content_chars": MAX_TRIGGER_CONTENT_CHARS,
            "trigger_total_content_chars": MAX_TRIGGER_TOTAL_CONTENT_CHARS,
            "trigger_lookback_turns_min": MIN_TRIGGER_LOOKBACK_TURNS,
            "trigger_lookback_turns_max": MAX_TRIGGER_LOOKBACK_TURNS,
        },
    }


def _error_view(
    code: str,
    message: str,
    *,
    field_errors: Mapping[str, str] | None = None,
    current: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "field_errors": dict(field_errors or {}),
    }
    if current is not None:
        error["current"] = dict(current)
    return {"ok": False, "error": error}


def _parse_model(value: Any) -> CharacterModel:
    root = _mapping(value, "model")
    _validate_model_root(root)
    identity = _section(root, "identity")
    personality = _section(root, "personality")
    social = _section(root, "social")
    preferences = _section(root, "preferences")
    language = _section(root, "language")
    (
        main_core_modes,
        main_core_styles,
        response_polish,
        story_styles,
        background_creation,
    ) = _custom_prompt_sections(_section(root, "custom_prompts"))
    prompt_selections = _parse_prompt_selections(_section(root, "prompt_selections"))
    visual = _section(root, "visual")
    capabilities = _section(root, "capabilities")
    return CharacterModel(
        identity=_parse_identity(identity),
        personality=_parse_personality(personality),
        social=_parse_social(social),
        preferences=_parse_preferences(preferences),
        language=_parse_language(language),
        custom_prompts=_parse_custom_prompts(
            main_core_modes,
            main_core_styles,
            response_polish,
            story_styles,
            background_creation,
        ),
        prompt_selections=prompt_selections,
        dialogue_reference=_root_string(root, "dialogue_reference"),
        visual=_parse_visual(visual),
        capabilities=_parse_capabilities(capabilities),
        trigger_rules=_trigger_rules(root.get("trigger_rules", ())),
    )


def _validate_model_root(root: Mapping[str, Any]) -> None:
    schema_version = root.get("schema_version")
    if type(schema_version) is not int or schema_version != MODEL_SCHEMA_VERSION:
        raise CharacterModelPayloadError(
            "model.schema_version", f"schema_version 必须为 {MODEL_SCHEMA_VERSION}。"
        )
    _reject_unknown(
        root,
        {
            "schema_version",
            "identity",
            "personality",
            "social",
            "preferences",
            "language",
            "custom_prompts",
            "prompt_selections",
            "dialogue_reference",
            "visual",
            "capabilities",
            "trigger_rules",
        },
        "model",
    )


def _parse_identity(value: Mapping[str, Any]) -> CharacterIdentity:
    _reject_unknown(value, {"name", "aliases", "overview", "facts"}, "identity")
    return CharacterIdentity(
        name=_string(value, "name", "identity.name"),
        aliases=_strings(value, "aliases", "identity.aliases"),
        overview=_string(value, "overview", "identity.overview"),
        facts=_strings(value, "facts", "identity.facts"),
    )


def _parse_personality(value: Mapping[str, Any]) -> PersonalityProfile:
    _reject_unknown(
        value,
        {"traits_and_values", "thinking_and_behavior", "habits_and_emotions"},
        "personality",
    )
    return PersonalityProfile(
        traits_and_values=_strings(value, "traits_and_values", "personality.traits_and_values"),
        thinking_and_behavior=_strings(
            value, "thinking_and_behavior", "personality.thinking_and_behavior"
        ),
        habits_and_emotions=_strings(
            value, "habits_and_emotions", "personality.habits_and_emotions"
        ),
    )


def _parse_social(value: Mapping[str, Any]) -> SocialProfile:
    _reject_unknown(value, {"interaction_style", "boundaries"}, "social")
    return SocialProfile(
        interaction_style=_strings(value, "interaction_style", "social.interaction_style"),
        boundaries=_strings(value, "boundaries", "social.boundaries"),
    )


def _parse_preferences(value: Mapping[str, Any]) -> PreferenceProfile:
    _reject_unknown(value, {"likes_and_interests", "dislikes"}, "preferences")
    return PreferenceProfile(
        likes_and_interests=_strings(
            value, "likes_and_interests", "preferences.likes_and_interests"
        ),
        dislikes=_strings(value, "dislikes", "preferences.dislikes"),
    )


def _parse_language(value: Mapping[str, Any]) -> LanguageProfile:
    _reject_unknown(
        value,
        {
            "speaking_style",
            "messaging_habits",
            "address_habits",
        },
        "language",
    )
    return LanguageProfile(
        speaking_style=_strings(value, "speaking_style", "language.speaking_style"),
        messaging_habits=_strings(value, "messaging_habits", "language.messaging_habits"),
        address_habits=_strings(value, "address_habits", "language.address_habits"),
    )


def _custom_prompt_sections(
    value: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    _reject_unknown(
        value,
        {
            "main_core_modes",
            "main_core_styles",
            "response_polish",
            "story_styles",
            "background_creation",
        },
        "custom_prompts",
    )
    modes = _mapping(value["main_core_modes"], "custom_prompts.main_core_modes")
    _reject_unknown(
        modes,
        {"self_initiated"},
        "custom_prompts.main_core_modes",
    )
    styles = _mapping(value["main_core_styles"], "custom_prompts.main_core_styles")
    _reject_unknown(
        styles,
        {
            "relationship_context",
            "speaking_style",
            "sticker_style",
            "thinking_style",
            "content_style",
            "conversation_content",
        },
        "custom_prompts.main_core_styles",
    )
    response_polish = _mapping(
        value["response_polish"],
        "custom_prompts.response_polish",
    )
    _reject_unknown(
        response_polish,
        {"writing_correction"},
        "custom_prompts.response_polish",
    )
    story_styles = _mapping(value["story_styles"], "custom_prompts.story_styles")
    _reject_unknown(
        story_styles,
        {"involvement", "stance"},
        "custom_prompts.story_styles",
    )
    background_creation = _mapping(
        value["background_creation"],
        "custom_prompts.background_creation",
    )
    _reject_unknown(
        background_creation,
        {"world_change", "story_boundary", "imagination", "temperature"},
        "custom_prompts.background_creation",
    )
    return modes, styles, response_polish, story_styles, background_creation


def _parse_custom_prompts(
    modes: Mapping[str, Any],
    styles: Mapping[str, Any],
    response_polish: Mapping[str, Any],
    story_styles: Mapping[str, Any],
    background_creation: Mapping[str, Any],
) -> CharacterCustomPrompts:
    return CharacterCustomPrompts(
        main_core_modes=MainCoreModePrompts(
            self_initiated=_string(
                modes, "self_initiated", "custom_prompts.main_core_modes.self_initiated"
            ),
        ),
        main_core_styles=MainCoreStylePrompts(
            relationship_context=_string(
                styles,
                "relationship_context",
                "custom_prompts.main_core_styles.relationship_context",
            ),
            speaking_style=_string(
                styles,
                "speaking_style",
                "custom_prompts.main_core_styles.speaking_style",
            ),
            sticker_style=_string(
                styles,
                "sticker_style",
                "custom_prompts.main_core_styles.sticker_style",
            ),
            thinking_style=_string(
                styles,
                "thinking_style",
                "custom_prompts.main_core_styles.thinking_style",
            ),
            content_style=_string(
                styles,
                "content_style",
                "custom_prompts.main_core_styles.content_style",
            ),
            conversation_content=_string(
                styles,
                "conversation_content",
                "custom_prompts.main_core_styles.conversation_content",
            ),
        ),
        response_polish=ResponsePolishPrompts(
            writing_correction=_string(
                response_polish,
                "writing_correction",
                "custom_prompts.response_polish.writing_correction",
            ),
        ),
        story_styles=StoryStylePrompts(
            involvement=_string(
                story_styles,
                "involvement",
                "custom_prompts.story_styles.involvement",
            ),
            stance=_string(
                story_styles,
                "stance",
                "custom_prompts.story_styles.stance",
            ),
        ),
        background_creation=BackgroundCreationPrompts(
            world_change=_string(
                background_creation,
                "world_change",
                "custom_prompts.background_creation.world_change",
            ),
            story_boundary=_string(
                background_creation,
                "story_boundary",
                "custom_prompts.background_creation.story_boundary",
            ),
            imagination=_string(
                background_creation,
                "imagination",
                "custom_prompts.background_creation.imagination",
            ),
            temperature=_string(
                background_creation,
                "temperature",
                "custom_prompts.background_creation.temperature",
            ),
        ),
    )


def _parse_prompt_selections(value: Mapping[str, Any]) -> CharacterPromptSelections:
    _reject_unknown(
        value,
        {
            "main_core_modes",
            "main_core_styles",
            "response_polish",
            "story_styles",
            "background_creation",
        },
        "prompt_selections",
    )
    modes = _prompt_selection_group(value, "main_core_modes", {"self_initiated"})
    styles = _prompt_selection_group(
        value,
        "main_core_styles",
        {
            "relationship_context",
            "speaking_style",
            "sticker_style",
            "thinking_style",
            "content_style",
            "conversation_content",
        },
    )
    response_polish = _prompt_selection_group(
        value,
        "response_polish",
        {"writing_correction"},
    )
    stories = _prompt_selection_group(value, "story_styles", {"involvement", "stance"})
    background = _prompt_selection_group(
        value,
        "background_creation",
        {"world_change", "story_boundary", "imagination", "temperature"},
    )
    return CharacterPromptSelections(
        main_core_modes=MainCoreModePromptSelections(
            self_initiated=_string(
                modes,
                "self_initiated",
                "prompt_selections.main_core_modes.self_initiated",
            )
        ),
        main_core_styles=MainCoreStylePromptSelections(
            relationship_context=_string(
                styles,
                "relationship_context",
                "prompt_selections.main_core_styles.relationship_context",
            ),
            speaking_style=_string(
                styles,
                "speaking_style",
                "prompt_selections.main_core_styles.speaking_style",
            ),
            sticker_style=_string(
                styles,
                "sticker_style",
                "prompt_selections.main_core_styles.sticker_style",
            ),
            thinking_style=_string(
                styles,
                "thinking_style",
                "prompt_selections.main_core_styles.thinking_style",
            ),
            content_style=_string(
                styles,
                "content_style",
                "prompt_selections.main_core_styles.content_style",
            ),
            conversation_content=_string(
                styles,
                "conversation_content",
                "prompt_selections.main_core_styles.conversation_content",
            ),
        ),
        response_polish=ResponsePolishPromptSelections(
            writing_correction=_string(
                response_polish,
                "writing_correction",
                "prompt_selections.response_polish.writing_correction",
            )
        ),
        story_styles=StoryStylePromptSelections(
            involvement=_string(
                stories,
                "involvement",
                "prompt_selections.story_styles.involvement",
            ),
            stance=_string(
                stories,
                "stance",
                "prompt_selections.story_styles.stance",
            ),
        ),
        background_creation=BackgroundCreationPromptSelections(
            world_change=_string(
                background,
                "world_change",
                "prompt_selections.background_creation.world_change",
            ),
            story_boundary=_string(
                background,
                "story_boundary",
                "prompt_selections.background_creation.story_boundary",
            ),
            imagination=_string(
                background,
                "imagination",
                "prompt_selections.background_creation.imagination",
            ),
            temperature=_string(
                background,
                "temperature",
                "prompt_selections.background_creation.temperature",
            ),
        ),
    )


def _prompt_selection_group(
    root: Mapping[str, Any],
    name: str,
    fields: set[str],
) -> Mapping[str, Any]:
    path = f"prompt_selections.{name}"
    value = _mapping(root[name], path)
    _reject_unknown(value, fields, path)
    return value


def _parse_visual(value: Mapping[str, Any]) -> VisualProfile:
    _reject_unknown(value, {"appearance", "clothing", "visual_boundaries"}, "visual")
    return VisualProfile(
        appearance=_strings(value, "appearance", "visual.appearance"),
        clothing=_strings(value, "clothing", "visual.clothing"),
        visual_boundaries=_strings(value, "visual_boundaries", "visual.visual_boundaries"),
    )


def _parse_capabilities(value: Mapping[str, Any]) -> CapabilityProfile:
    _reject_unknown(value, {"abilities", "knowledge_scope", "limitations"}, "capabilities")
    return CapabilityProfile(
        abilities=_strings(value, "abilities", "capabilities.abilities"),
        knowledge_scope=_strings(value, "knowledge_scope", "capabilities.knowledge_scope"),
        limitations=_strings(value, "limitations", "capabilities.limitations"),
    )


def _trigger_rules(value: Any) -> tuple[CharacterTriggerRule, ...]:
    field = "trigger_rules"
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CharacterModelPayloadError(field, "trigger_rules 必须是数组。")
    if len(value) > MAX_TRIGGER_RULES:
        raise CharacterModelPayloadError(
            field, f"trigger_rules 最多包含 {MAX_TRIGGER_RULES} 组规则。"
        )
    result: list[CharacterTriggerRule] = []
    total_content_chars = 0
    for index, raw in enumerate(value):
        rule = _trigger_rule(raw, index)
        total_content_chars += len(rule.content.strip())
        if total_content_chars > MAX_TRIGGER_TOTAL_CONTENT_CHARS:
            content_field = f"trigger_rules[{index}].content"
            raise CharacterModelPayloadError(
                content_field,
                f"全部触发内容合计不能超过 {MAX_TRIGGER_TOTAL_CONTENT_CHARS} 个字符。",
            )
        result.append(rule)
    return tuple(result)


def _trigger_rule(value: Any, index: int) -> CharacterTriggerRule:
    item_field = f"trigger_rules[{index}]"
    item = _mapping(value, item_field)
    unknown = sorted(set(item) - {"keys", "lookback_turns", "content"})
    if unknown:
        unknown_field = f"{item_field}.{unknown[0]}"
        raise CharacterModelPayloadError(unknown_field, f"{unknown_field} 是未知字段。")
    return CharacterTriggerRule(
        _trigger_rule_keys(item, item_field),
        _trigger_rule_lookback(item, item_field),
        _trigger_rule_content(item, item_field),
    )


def _trigger_rule_keys(item: Mapping[str, Any], item_field: str) -> tuple[str, ...]:
    field = f"{item_field}.keys"
    keys = _strings(item, "keys", field)
    if not keys or any(not key.strip() for key in keys):
        raise CharacterModelPayloadError(field, f"{field} 至少需要一个非空 Key。")
    if len(keys) > MAX_TRIGGER_KEYS:
        raise CharacterModelPayloadError(field, f"{field} 最多包含 {MAX_TRIGGER_KEYS} 个 Key。")
    for index, key in enumerate(keys):
        if len(key.strip()) > MAX_TRIGGER_KEY_CHARS:
            key_field = f"{field}[{index}]"
            raise CharacterModelPayloadError(
                key_field, f"{key_field} 不能超过 {MAX_TRIGGER_KEY_CHARS} 个字符。"
            )
    return keys


def _trigger_rule_lookback(item: Mapping[str, Any], item_field: str) -> int:
    field = f"{item_field}.lookback_turns"
    value = item.get("lookback_turns")
    if type(value) is not int or not (
        MIN_TRIGGER_LOOKBACK_TURNS <= value <= MAX_TRIGGER_LOOKBACK_TURNS
    ):
        raise CharacterModelPayloadError(
            field,
            f"{field} 必须是 {MIN_TRIGGER_LOOKBACK_TURNS}–{MAX_TRIGGER_LOOKBACK_TURNS} 之间的整数。",
        )
    return value


def _trigger_rule_content(item: Mapping[str, Any], item_field: str) -> str:
    field = f"{item_field}.content"
    content = _string(item, "content", field)
    if not content.strip():
        raise CharacterModelPayloadError(field, f"{field} 不能为空。")
    if len(content.strip()) > MAX_TRIGGER_CONTENT_CHARS:
        raise CharacterModelPayloadError(
            field, f"{field} 不能超过 {MAX_TRIGGER_CONTENT_CHARS} 个字符。"
        )
    return content


def _section(root: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _mapping(root[name], name)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CharacterModelPayloadError(field, f"{field} 必须是对象。")
    return value


def _string(section: Mapping[str, Any], name: str, field: str) -> str:
    value = section[name]
    if not isinstance(value, str):
        raise CharacterModelPayloadError(field, f"{field} 必须是字符串。")
    return value


def _strings(section: Mapping[str, Any], name: str, field: str) -> tuple[str, ...]:
    value = section[name]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CharacterModelPayloadError(field, f"{field} 必须是字符串数组。")
    if any(not isinstance(item, str) for item in value):
        raise CharacterModelPayloadError(field, f"{field} 只能包含字符串。")
    return tuple(value)


def _root_string(root: Mapping[str, Any], name: str) -> str:
    value = root[name]
    if not isinstance(value, str):
        raise CharacterModelPayloadError(name, f"{name} 必须是字符串。")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise CharacterModelPayloadError(field, f"{field} 包含未知字段：{', '.join(unknown)}。")
    missing = sorted(allowed - set(value))
    if missing:
        raise CharacterModelPayloadError(field, f"{field} 缺少字段：{', '.join(missing)}。")


def _non_negative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise CharacterModelPayloadError(field, f"{field} 必须是非负整数。")
    return value


def _error_field(message: str) -> str | None:
    match = _FIELD_PREFIX.match(message)
    return match.group(1) if match else None


__all__ = ["CharacterModelsAdminController"]
