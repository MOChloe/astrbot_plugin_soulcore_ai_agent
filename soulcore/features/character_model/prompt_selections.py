"""Administrator-owned preset identities kept separate from model-visible Prompt text."""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_PROMPT_PRESET_ID_CHARS = 64


class PromptSelectionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MainCoreModePromptSelections:
    self_initiated: str = "cater_to_interests"


@dataclass(frozen=True, slots=True)
class MainCoreStylePromptSelections:
    relationship_context: str = "cross_world_communication"
    speaking_style: str = "natural_chat"
    sticker_style: str = "casual_fun"
    thinking_style: str = "balanced"
    content_style: str = "friend_style_sharing"
    conversation_content: str = "soulcore_free"


@dataclass(frozen=True, slots=True)
class ResponsePolishPromptSelections:
    writing_correction: str = "remove_ai_formula"


@dataclass(frozen=True, slots=True)
class StoryStylePromptSelections:
    involvement: str = "soulcore_free"
    stance: str = "soulcore_free"


@dataclass(frozen=True, slots=True)
class BackgroundCreationPromptSelections:
    world_change: str = "soulcore_free"
    story_boundary: str = "follow_original"
    imagination: str = "soulcore_free"
    temperature: str = "soulcore_free"


@dataclass(frozen=True, slots=True)
class CharacterPromptSelections:
    main_core_modes: MainCoreModePromptSelections = field(
        default_factory=MainCoreModePromptSelections
    )
    main_core_styles: MainCoreStylePromptSelections = field(
        default_factory=MainCoreStylePromptSelections
    )
    response_polish: ResponsePolishPromptSelections = field(
        default_factory=ResponsePolishPromptSelections
    )
    story_styles: StoryStylePromptSelections = field(default_factory=StoryStylePromptSelections)
    background_creation: BackgroundCreationPromptSelections = field(
        default_factory=BackgroundCreationPromptSelections
    )


def normalize_prompt_selections(value: CharacterPromptSelections) -> CharacterPromptSelections:
    modes = value.main_core_modes
    styles = value.main_core_styles
    response_polish = value.response_polish
    stories = value.story_styles
    background = value.background_creation
    return CharacterPromptSelections(
        main_core_modes=MainCoreModePromptSelections(
            self_initiated=_preset(
                modes.self_initiated,
                {"cater_to_interests", "soulcore_free", "online_friend"},
                "prompt_selections.main_core_modes.self_initiated",
            )
        ),
        main_core_styles=MainCoreStylePromptSelections(
            relationship_context=_preset(
                styles.relationship_context,
                {"cross_world_communication", "same_world_separate_lives"},
                "prompt_selections.main_core_styles.relationship_context",
            ),
            speaking_style=_preset(
                styles.speaking_style,
                {"natural_chat", "soulcore_free"},
                "prompt_selections.main_core_styles.speaking_style",
            ),
            sticker_style=_preset(
                styles.sticker_style,
                {"casual_fun", "only_when_fitting", "soulcore_free"},
                "prompt_selections.main_core_styles.sticker_style",
            ),
            thinking_style=_preset(
                styles.thinking_style,
                {"balanced", "soulcore_free", "lovers", "devoted_lover", "center_on_the_other"},
                "prompt_selections.main_core_styles.thinking_style",
            ),
            content_style=_preset(
                styles.content_style,
                {"often_share_background", "friend_style_sharing", "background_as_subtext"},
                "prompt_selections.main_core_styles.content_style",
            ),
            conversation_content=_preset(
                styles.conversation_content,
                {"soulcore_free"},
                "prompt_selections.main_core_styles.conversation_content",
            ),
        ),
        response_polish=ResponsePolishPromptSelections(
            writing_correction=_preset(
                response_polish.writing_correction,
                {"remove_ai_formula"},
                "prompt_selections.response_polish.writing_correction",
            )
        ),
        story_styles=StoryStylePromptSelections(
            involvement=_preset(
                stories.involvement,
                {"soulcore_free"},
                "prompt_selections.story_styles.involvement",
            ),
            stance=_preset(
                stories.stance,
                {"soulcore_free"},
                "prompt_selections.story_styles.stance",
            ),
        ),
        background_creation=BackgroundCreationPromptSelections(
            world_change=_preset(
                background.world_change,
                {"soulcore_free"},
                "prompt_selections.background_creation.world_change",
            ),
            story_boundary=_preset(
                background.story_boundary,
                {"follow_original", "bold_expansion"},
                "prompt_selections.background_creation.story_boundary",
            ),
            imagination=_preset(
                background.imagination,
                {"soulcore_free"},
                "prompt_selections.background_creation.imagination",
            ),
            temperature=_preset(
                background.temperature,
                {"soulcore_free"},
                "prompt_selections.background_creation.temperature",
            ),
        ),
    )


def _preset(value: str, allowed: set[str], field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return "custom"
    if len(normalized) > MAX_PROMPT_PRESET_ID_CHARS:
        raise PromptSelectionError(f"{field_name} exceeds {MAX_PROMPT_PRESET_ID_CHARS} characters")
    if normalized != "custom" and normalized not in allowed:
        raise PromptSelectionError(f"{field_name} contains an unsupported preset")
    return normalized


__all__ = [
    "BackgroundCreationPromptSelections",
    "CharacterPromptSelections",
    "MAX_PROMPT_PRESET_ID_CHARS",
    "MainCoreModePromptSelections",
    "MainCoreStylePromptSelections",
    "PromptSelectionError",
    "ResponsePolishPromptSelections",
    "StoryStylePromptSelections",
    "normalize_prompt_selections",
]
