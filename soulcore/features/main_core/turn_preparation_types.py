"""Shared value objects for Main Core turn preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...contracts.ai_models import AIBackendDescriptor
from ...contracts.models import CharacterInstance
from ...contracts.thinking import MainCoreThinkingPolicy
from ..character_model import MainCoreModePrompts, MainCoreStylePrompts, StoryStylePrompts
from .turn_responsibility import DEFAULT_MESSAGE_RESPONSIBILITY, MainCoreTurnResponsibility


@dataclass(slots=True)
class TurnRoute:
    routes: list[CharacterInstance]
    route_umo: str
    required_proactive_umo: str | None
    preferred_backend_id: str
    backend_hint: AIBackendDescriptor | None
    persona: str
    character_projection: Any | None = None
    polish_backend_hint: AIBackendDescriptor | None = None
    thinking_policy: MainCoreThinkingPolicy | None = None
    background_view: Any | None = None
    main_core_mode_prompts: MainCoreModePrompts = MainCoreModePrompts()
    main_core_style_prompts: MainCoreStylePrompts = MainCoreStylePrompts()
    story_style_prompts: StoryStylePrompts = StoryStylePrompts()


@dataclass(slots=True)
class TurnFeatures:
    foreground_only: bool
    restricted_decline: bool
    image_generation_enabled: bool
    file_generation_enabled: bool
    important_todo_refs: dict[str, dict[str, Any]]
    web_search_enabled: bool
    sticker_enabled: bool
    commands: Any
    temporary_absence_enabled: bool = False
    temporary_absence_max_duration_seconds: int = 0
    responsibility: MainCoreTurnResponsibility = DEFAULT_MESSAGE_RESPONSIBILITY
    character_identity_reference: dict[str, Any] | None = None
    run_prompt: str = ""
    model_runtime_note: str = ""
    sticker_command_context: Any | None = None
    current_web_resources: tuple[Any, ...] = ()


@dataclass(slots=True)
class TurnContexts:
    request_context_manager: Any | None
    prepared_context: Any | None
    active_intents: list[Any]
    web_command_context: Any | None = None
    trigger_evaluation: Any | None = None
    previous_context_message_ids: tuple[int, ...] = ()
