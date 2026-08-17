"""Strict JSON mapping kept inside the character-model SQLite adapter."""

from __future__ import annotations

import json

from ..domain import (
    BackgroundCreationPrompts,
    CapabilityProfile,
    CharacterCustomPrompts,
    CharacterIdentity,
    CharacterModel,
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
from ..prompt_selections import (
    BackgroundCreationPromptSelections,
    CharacterPromptSelections,
    MainCoreModePromptSelections,
    MainCoreStylePromptSelections,
    ResponsePolishPromptSelections,
    StoryStylePromptSelections,
)


def encode_model_columns(model: CharacterModel) -> dict[str, str]:
    payload = model_to_payload(model)
    columns = {
        f"{name}_json": _dump(payload[name])
        for name in (
            "identity",
            "personality",
            "social",
            "preferences",
            "language",
            "visual",
            "capabilities",
        )
    }
    personality = dict(payload["personality"])
    personality["custom_prompts"] = payload["custom_prompts"]
    personality["prompt_selections"] = payload["prompt_selections"]
    columns["personality_json"] = _dump(personality)
    columns["dialogue_reference"] = model.dialogue_reference
    columns["trigger_rules_json"] = _dump(payload["trigger_rules"])
    return columns


def decode_model(columns: dict[str, object]) -> CharacterModel:
    identity = _object(columns["identity_json"], "identity_json")
    personality = _object(columns["personality_json"], "personality_json")
    social = _object(columns["social_json"], "social_json")
    preferences = _object(columns["preferences_json"], "preferences_json")
    language = _object(columns["language_json"], "language_json")
    visual = _object(columns["visual_json"], "visual_json")
    capabilities = _object(columns["capabilities_json"], "capabilities_json")
    trigger_rules = _array(columns["trigger_rules_json"], "trigger_rules_json")
    _require_shape(identity, {"name", "aliases", "overview", "facts"}, "identity_json")
    _require_shape(
        personality,
        {
            "traits_and_values",
            "thinking_and_behavior",
            "habits_and_emotions",
            "custom_prompts",
            "prompt_selections",
        },
        "personality_json",
    )
    _require_shape(social, {"interaction_style", "boundaries"}, "social_json")
    _require_shape(preferences, {"likes_and_interests", "dislikes"}, "preferences_json")
    _require_shape(
        language,
        {"speaking_style", "messaging_habits", "address_habits"},
        "language_json",
    )
    _require_shape(
        visual,
        {"appearance", "clothing", "visual_boundaries"},
        "visual_json",
    )
    _require_shape(
        capabilities,
        {"abilities", "knowledge_scope", "limitations"},
        "capabilities_json",
    )
    custom_prompts = _decode_custom_prompts(personality)
    return CharacterModel(
        identity=CharacterIdentity(
            name=str(identity.get("name") or ""),
            aliases=_tuple(identity.get("aliases")),
            overview=str(identity.get("overview") or ""),
            facts=_tuple(identity.get("facts")),
        ),
        personality=PersonalityProfile(
            traits_and_values=_tuple(personality.get("traits_and_values")),
            thinking_and_behavior=_tuple(personality.get("thinking_and_behavior")),
            habits_and_emotions=_tuple(personality.get("habits_and_emotions")),
        ),
        custom_prompts=custom_prompts,
        prompt_selections=_decode_prompt_selections(personality["prompt_selections"]),
        social=SocialProfile(
            interaction_style=_tuple(social.get("interaction_style")),
            boundaries=_tuple(social.get("boundaries")),
        ),
        preferences=PreferenceProfile(
            likes_and_interests=_tuple(preferences.get("likes_and_interests")),
            dislikes=_tuple(preferences.get("dislikes")),
        ),
        language=LanguageProfile(
            speaking_style=_tuple(language.get("speaking_style")),
            messaging_habits=_tuple(language.get("messaging_habits")),
            address_habits=_tuple(language.get("address_habits")),
        ),
        dialogue_reference=str(columns["dialogue_reference"] or ""),
        visual=VisualProfile(
            appearance=_tuple(visual.get("appearance")),
            clothing=_tuple(visual.get("clothing")),
            visual_boundaries=_tuple(visual.get("visual_boundaries")),
        ),
        capabilities=CapabilityProfile(
            abilities=_tuple(capabilities.get("abilities")),
            knowledge_scope=_tuple(capabilities.get("knowledge_scope")),
            limitations=_tuple(capabilities.get("limitations")),
        ),
        trigger_rules=tuple(
            _decode_trigger_rule(rule, index) for index, rule in enumerate(trigger_rules)
        ),
    )


def _decode_custom_prompts(personality: dict[str, object]) -> CharacterCustomPrompts:
    custom = _nested_object(personality.get("custom_prompts"), "custom_prompts")
    _require_shape(
        custom,
        {
            "main_core_modes",
            "main_core_styles",
            "response_polish",
            "story_styles",
            "background_creation",
        },
        "custom_prompts",
    )
    modes = _nested_object(custom.get("main_core_modes"), "custom_prompts.main_core_modes")
    styles = _nested_object(custom.get("main_core_styles"), "custom_prompts.main_core_styles")
    stories = _nested_object(custom.get("story_styles"), "custom_prompts.story_styles")
    background = _nested_object(
        custom.get("background_creation"), "custom_prompts.background_creation"
    )
    response_polish = _nested_object(
        custom.get("response_polish"), "custom_prompts.response_polish"
    )
    _require_shape(modes, {"self_initiated"}, "custom_prompts.main_core_modes")
    _require_shape(
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
    _require_shape(
        stories,
        {"involvement", "stance"},
        "custom_prompts.story_styles",
    )
    _require_shape(
        background,
        {"world_change", "story_boundary", "imagination", "temperature"},
        "custom_prompts.background_creation",
    )
    _require_shape(
        response_polish,
        {"writing_correction"},
        "custom_prompts.response_polish",
    )
    return CharacterCustomPrompts(
        main_core_modes=MainCoreModePrompts(
            self_initiated=str(modes["self_initiated"] or ""),
        ),
        main_core_styles=_decode_main_core_styles(styles),
        response_polish=ResponsePolishPrompts(
            writing_correction=str(response_polish["writing_correction"] or ""),
        ),
        story_styles=StoryStylePrompts(
            involvement=str(stories.get("involvement") or ""),
            stance=str(stories.get("stance") or ""),
        ),
        background_creation=BackgroundCreationPrompts(
            world_change=str(background.get("world_change") or ""),
            story_boundary=str(background["story_boundary"] or ""),
            imagination=str(background.get("imagination") or ""),
            temperature=str(background.get("temperature") or ""),
        ),
    )


def _decode_main_core_styles(styles: dict[str, object]) -> MainCoreStylePrompts:
    return MainCoreStylePrompts(
        relationship_context=str(styles["relationship_context"] or ""),
        speaking_style=str(styles["speaking_style"] or ""),
        sticker_style=str(styles["sticker_style"] or ""),
        thinking_style=str(styles["thinking_style"] or ""),
        content_style=str(styles["content_style"] or ""),
        conversation_content=str(styles.get("conversation_content") or ""),
    )


def _decode_prompt_selections(
    value: object,
) -> CharacterPromptSelections:
    root = _nested_object(value, "prompt_selections")
    _require_shape(
        root,
        {
            "main_core_modes",
            "main_core_styles",
            "response_polish",
            "story_styles",
            "background_creation",
        },
        "prompt_selections",
    )
    modes = _selection_group(root, "main_core_modes", {"self_initiated"})
    styles = _selection_group(
        root,
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
    response_polish = _selection_group(root, "response_polish", {"writing_correction"})
    stories = _selection_group(root, "story_styles", {"involvement", "stance"})
    background = _selection_group(
        root,
        "background_creation",
        {"world_change", "story_boundary", "imagination", "temperature"},
    )
    return CharacterPromptSelections(
        main_core_modes=MainCoreModePromptSelections(
            self_initiated=str(modes["self_initiated"] or ""),
        ),
        main_core_styles=MainCoreStylePromptSelections(
            relationship_context=str(styles["relationship_context"] or ""),
            speaking_style=str(styles["speaking_style"] or ""),
            sticker_style=str(styles["sticker_style"] or ""),
            thinking_style=str(styles["thinking_style"] or ""),
            content_style=str(styles["content_style"] or ""),
            conversation_content=str(styles["conversation_content"] or ""),
        ),
        response_polish=ResponsePolishPromptSelections(
            writing_correction=str(response_polish["writing_correction"] or ""),
        ),
        story_styles=StoryStylePromptSelections(
            involvement=str(stories["involvement"] or ""),
            stance=str(stories["stance"] or ""),
        ),
        background_creation=BackgroundCreationPromptSelections(
            world_change=str(background["world_change"] or ""),
            story_boundary=str(background["story_boundary"] or ""),
            imagination=str(background["imagination"] or ""),
            temperature=str(background["temperature"] or ""),
        ),
    )


def _selection_group(
    root: dict[str, object],
    name: str,
    keys: set[str],
) -> dict[str, object]:
    field = f"prompt_selections.{name}"
    value = _nested_object(root.get(name), field)
    _require_shape(value, keys, field)
    return value


def _object(value: object, name: str) -> dict[str, object]:
    parsed = _load(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"invalid persisted {name}")
    return {str(key): item for key, item in parsed.items()}


def _nested_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"invalid persisted {name}")
    return {str(key): item for key, item in value.items()}


def _require_shape(value: dict[str, object], keys: set[str], name: str) -> None:
    if set(value) != keys:
        raise ValueError(f"invalid persisted {name} shape")


def _decode_trigger_rule(value: object, index: int) -> CharacterTriggerRule:
    if not isinstance(value, dict):
        raise ValueError(f"invalid persisted trigger_rules_json[{index}]")
    rule = {str(key): item for key, item in value.items()}
    _require_shape(rule, {"keys", "lookback_turns", "content"}, f"trigger_rules_json[{index}]")
    return CharacterTriggerRule(
        keys=_tuple(rule["keys"]),
        lookback_turns=int(rule["lookback_turns"]),
        content=str(rule["content"] or ""),
    )


def _tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("invalid persisted character-model list")
    return tuple(str(item) for item in value)


def _array(value: object, name: str) -> list[dict[str, object]]:
    parsed = _load(value)
    if not isinstance(parsed, list) or any(not isinstance(item, dict) for item in parsed):
        raise ValueError(f"invalid persisted {name}")
    return [{str(key): item for key, item in value.items()} for value in parsed]


def _load(value: object) -> object:
    return json.loads(str(value or "null"))


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = ["decode_model", "encode_model_columns"]
