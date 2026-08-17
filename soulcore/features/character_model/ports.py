"""Narrow public and persistence ports for character-model consumers."""

from __future__ import annotations

from typing import Protocol

from .domain import (
    CharacterModel,
    CharacterModelSave,
    CharacterModelSnapshot,
    CharacterProjection,
    CharacterTriggerEvaluation,
    FrozenCharacterModel,
    ProjectionPurpose,
)


class CharacterModelRepositoryPort(Protocol):
    async def load(
        self, profile_id: str, revision: int | None = None
    ) -> CharacterModelSnapshot | None: ...

    async def save(self, command: CharacterModelSave) -> CharacterModelSnapshot: ...


class CharacterModelReadPort(Protocol):
    async def freeze(self, profile_id: str) -> FrozenCharacterModel: ...

    async def project(
        self,
        frozen: FrozenCharacterModel,
        purpose: ProjectionPurpose,
        *,
        relevance_text: str = "",
    ) -> CharacterProjection: ...

    async def evaluate_triggers(
        self,
        frozen: FrozenCharacterModel,
        inbound_turns: tuple[str, ...],
    ) -> CharacterTriggerEvaluation: ...


class CharacterModelAdminPort(Protocol):
    async def get_current(self, profile_id: str) -> CharacterModelSnapshot | None: ...

    async def save_model(
        self,
        profile_id: str,
        model: CharacterModel,
        *,
        expected_revision: int,
        idempotency_key: str,
    ) -> CharacterModelSnapshot: ...


__all__ = [
    "CharacterModelAdminPort",
    "CharacterModelReadPort",
    "CharacterModelRepositoryPort",
]
