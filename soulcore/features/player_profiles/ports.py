"""Narrow persistence ports for the player-profile feature."""

from __future__ import annotations

from typing import Protocol

from .domain import PlayerProfileScope, PlayerProfileSnapshot


class PlayerProfileReader(Protocol):
    async def load_player_profile(
        self,
        scope: PlayerProfileScope,
    ) -> PlayerProfileSnapshot: ...


__all__ = ["PlayerProfileReader"]
