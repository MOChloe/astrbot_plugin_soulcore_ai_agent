"""Shared SQLite composition surface for atomic player-profile mutations."""

from __future__ import annotations

from ...features.player_profiles.sqlite.mutations import commit_profile_command

__all__ = ["commit_profile_command"]
