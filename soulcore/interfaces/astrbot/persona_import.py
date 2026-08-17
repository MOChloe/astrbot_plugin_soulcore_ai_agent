"""Read the selected AstrBot Persona only for explicit administrator import."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...features.identity import CHARACTER_PLACEHOLDER

_ASTRBOT_DEFAULT_PROMPT = "You are a helpful and friendly assistant."
_MAX_PROMPT_CHARS = 120_000
_MAX_DIALOGUE_CHARS = 40_000


@dataclass(frozen=True, slots=True)
class AstrBotPersonaSource:
    name: str
    prompt: str
    begin_dialogs: tuple[str, ...]

    @property
    def task_name(self) -> str:
        return _safe_source_text(self.name)

    @property
    def task_prompt(self) -> str:
        return _safe_source_text(self.prompt)

    @property
    def task_dialogues(self) -> tuple[str, ...]:
        return tuple(_safe_source_text(item) for item in self.begin_dialogs)


class AstrBotPersonaImportAdapter:
    """Bound AstrBot's profile configuration to its selected Persona."""

    def __init__(self, context: Any) -> None:
        self.context = context

    async def selected_source(self, profile_id: str) -> AstrBotPersonaSource | None:
        config_manager = self.context.astrbot_config_mgr
        configurations = config_manager.confs
        config = configurations.get(str(profile_id))
        if not isinstance(config, Mapping):
            return None
        provider_settings = config.get("provider_settings")
        if not isinstance(provider_settings, Mapping):
            return None
        persona_id = str(provider_settings.get("default_personality") or "default").strip()
        if not persona_id or persona_id == "default":
            return None

        persona = self.context.persona_manager.get_persona_v3_by_id(persona_id)
        if not isinstance(persona, Mapping):
            return None
        prompt = str(persona.get("prompt") or "").strip()
        if _normalized_prompt(prompt) == _normalized_prompt(_ASTRBOT_DEFAULT_PROMPT):
            prompt = ""
        dialogues = _dialogues(persona.get("begin_dialogs"))
        if not prompt and not dialogues:
            return None
        return AstrBotPersonaSource(
            name=str(persona.get("name") or persona_id).strip(),
            prompt=prompt[:_MAX_PROMPT_CHARS],
            begin_dialogs=_bounded_dialogues(dialogues),
        )


def _dialogues(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return ()
    return tuple(text for item in value if (text := str(item or "").strip()))


def _bounded_dialogues(values: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    remaining = _MAX_DIALOGUE_CHARS
    for value in values:
        if remaining <= 0:
            break
        result.append(value[:remaining])
        remaining -= len(result[-1])
    return tuple(result)


def _normalized_prompt(value: str) -> str:
    return " ".join(str(value or "").split()).casefold()


def _safe_source_text(value: str) -> str:
    return str(value or "").replace(CHARACTER_PLACEHOLDER, "角色本人")


__all__ = ["AstrBotPersonaImportAdapter", "AstrBotPersonaSource"]
