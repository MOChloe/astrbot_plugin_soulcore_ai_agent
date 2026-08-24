"""Role-scoped SoulCore thinking policy administration."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from ....contracts.thinking import (
    require_thinking_policy,
    thinking_policy_from_value,
    thinking_policy_options,
)
from ....features.profiles.ports import ProfilesRepositoryPort


class ThinkingSettingsController:
    def __init__(
        self,
        profiles_repository: ProfilesRepositoryPort,
    ) -> None:
        self.profiles_repository = profiles_repository
        self._save_lock = asyncio.Lock()

    async def snapshot(
        self,
        profile_id: str,
        scope_config: Mapping[str, Any],
        ai_packages: Mapping[str, Any],
    ) -> dict[str, Any]:
        profile = await self.profiles_repository.get_profile(profile_id)
        if profile is None:
            raise KeyError(profile_id)
        del scope_config, ai_packages
        policy = thinking_policy_from_value(profile.thinking_complexity)
        return {
            "complexity": policy.complexity.value,
            "policies": thinking_policy_options(),
            "hard_max_steps": policy.hard_max_steps,
            "preload_tokens": policy.preload_tokens,
            "current_fill_budget": policy.preload_tokens,
        }

    async def quick_setup_snapshot(self, profile_id: str) -> dict[str, Any]:
        profile = await self.profiles_repository.get_profile(profile_id)
        if profile is None:
            raise KeyError(profile_id)
        return {
            "complexity": thinking_policy_from_value(profile.thinking_complexity).complexity.value,
            "policies": thinking_policy_options(),
        }

    async def save(self, value: Mapping[str, Any]) -> dict[str, Any]:
        async with self._save_lock:
            return await self._save(value)

    async def _save(self, value: Mapping[str, Any]) -> dict[str, Any]:
        profile_id = str(value.get("profile_id") or "").strip()
        if not profile_id:
            raise ValueError("profile_id is required")
        profile = await self.profiles_repository.get_profile(profile_id)
        if profile is None:
            raise KeyError(profile_id)
        expected = str(value.get("expected_complexity") or "").strip()
        current = thinking_policy_from_value(profile.thinking_complexity)
        if expected and expected != current.complexity.value:
            raise ValueError("思考档位已被其他页面修改，请刷新后重试")
        policy = require_thinking_policy(value.get("complexity"))
        await self.profiles_repository.set_profile_thinking_complexity(
            profile_id,
            policy.complexity.value,
        )
        return {
            "complexity": policy.complexity.value,
            "hard_max_steps": policy.hard_max_steps,
            "preload_tokens": policy.preload_tokens,
            "message": (
                f"已切换至{policy.complexity.value}；私聊和群聊平时最多预装 "
                f"{policy.preload_tokens} Token，最多行动 {policy.hard_max_steps} 步。"
            ),
        }


__all__ = ["ThinkingSettingsController"]
