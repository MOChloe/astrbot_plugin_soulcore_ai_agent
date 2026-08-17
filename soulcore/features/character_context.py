"""Run-scoped access to immutable character-model projections.

Consumers freeze the opaque handle once, then reuse this context for every
projection and nested AI operation in the same run.  No raw character model is
exposed here.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field

from .character_model import (
    CharacterFieldCategory,
    CharacterProjection,
    CharacterTriggerEvaluation,
    FrozenCharacterModel,
    MainCoreModePrompts,
    MainCoreStylePrompts,
    ProjectionPurpose,
    StoryStylePrompts,
)
from .character_model.ports import CharacterModelReadPort


@dataclass(slots=True)
class CharacterRunContext:
    port: CharacterModelReadPort
    frozen: FrozenCharacterModel
    _projections: dict[tuple[ProjectionPurpose, str], CharacterProjection] = field(
        default_factory=dict
    )

    @classmethod
    async def start(
        cls,
        port: CharacterModelReadPort | None,
        profile_id: str,
    ) -> CharacterRunContext:
        if port is None:
            raise RuntimeError("character model read port is unavailable")
        frozen = await port.freeze(str(profile_id or ""))
        return cls(port=port, frozen=frozen)

    async def project(
        self,
        purpose: ProjectionPurpose,
        *,
        relevance_text: str = "",
    ) -> CharacterProjection:
        relevance = str(relevance_text or "")[:4_000]
        key = (purpose, relevance)
        projection = self._projections.get(key)
        if projection is None:
            projection = await self.port.project(
                self.frozen,
                purpose,
                relevance_text=relevance,
            )
            self._projections[key] = projection
        return projection

    async def evaluate_triggers(
        self,
        inbound_turns: tuple[str, ...],
    ) -> CharacterTriggerEvaluation:
        return await self.port.evaluate_triggers(self.frozen, inbound_turns)


_CURRENT: ContextVar[CharacterRunContext | None] = ContextVar(
    "soulcore_character_run_context",
    default=None,
)


class CharacterRunScope:
    def __init__(self, context: CharacterRunContext) -> None:
        self.context = context
        self._token: Token[CharacterRunContext | None] | None = None

    def __enter__(self) -> CharacterRunContext:
        self._token = _CURRENT.set(self.context)
        return self.context

    def __exit__(self, *_exc: object) -> None:
        if self._token is not None:
            _CURRENT.reset(self._token)


def current_character_run(profile_id: str | None = None) -> CharacterRunContext | None:
    context = _CURRENT.get()
    if context is None:
        return None
    expected = str(profile_id or "").strip()
    if expected and context.frozen.profile_id != expected:
        return None
    return context


def require_character_run(profile_id: str | None = None) -> CharacterRunContext:
    context = current_character_run(profile_id)
    if context is None:
        raise RuntimeError("character model was not frozen for this AI run")
    return context


def projection_diagnostic(projection: CharacterProjection) -> dict[str, object]:
    """Return safe diagnostics without projected or hidden character text."""

    return {
        "revision": projection.revision,
        "content_fingerprint": projection.content_fingerprint,
        "projection_fingerprint": projection.projection_fingerprint,
        "purpose": projection.purpose.value,
        "selected_categories": [item.value for item in projection.selected_categories],
        "token_count": projection.token_count,
        "character_count": projection.character_count,
    }


def projected_dialogue_reference(projection: CharacterProjection) -> str:
    """Extract the voice-reference text from an already-authorized polish projection."""

    for section in projection.sections:
        if section.category is not CharacterFieldCategory.DIALOGUE_REFERENCE:
            continue
        body = "\n".join(section.text.splitlines()[1:]).strip()
        return body.removeprefix("- ").removeprefix("原声台词：").strip()
    return ""


def projected_main_core_mode_prompts(
    projection: CharacterProjection,
) -> MainCoreModePrompts:
    """Return per-mode MainCore overrides authorized for this projection."""

    return projection.custom_prompts.main_core_modes


def projected_main_core_style_prompts(
    projection: CharacterProjection,
) -> MainCoreStylePrompts:
    """Return the four role-author style prompts authorized for this projection."""

    return projection.custom_prompts.main_core_styles


def projected_story_style_prompts(projection: CharacterProjection) -> StoryStylePrompts:
    """Return the role's optional story-facing preferences."""

    return projection.custom_prompts.story_styles


__all__ = [
    "CharacterRunContext",
    "CharacterRunScope",
    "current_character_run",
    "projected_dialogue_reference",
    "projected_main_core_mode_prompts",
    "projected_main_core_style_prompts",
    "projected_story_style_prompts",
    "projection_diagnostic",
    "require_character_run",
]
