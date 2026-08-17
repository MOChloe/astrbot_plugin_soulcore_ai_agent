"""Role-scoped SoulCore thinking policy administration."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
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
        policy = thinking_policy_from_value(profile.thinking_complexity)
        stored_maximum = int(scope_config.get("max_context_tokens") or policy.max_context_tokens)
        stored_target = int(
            scope_config.get("target_context_tokens") or policy.target_context_tokens
        )
        compatible, incompatible = self._model_compatibility(ai_packages, stored_maximum)
        return {
            "complexity": policy.complexity.value,
            "policies": thinking_policy_options(),
            "hard_max_steps": policy.hard_max_steps,
            "max_context_tokens": policy.max_context_tokens,
            "target_context_tokens": policy.target_context_tokens,
            "fill_ratio": policy.fill_ratio,
            "preload_tokens": policy.preload_tokens,
            "current_fill_budget": int(stored_target * policy.fill_ratio),
            "compatible_models": compatible,
            "incompatible_models": incompatible,
            "has_compatible_model": bool(compatible),
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
            "max_context_tokens": policy.max_context_tokens,
            "target_context_tokens": policy.target_context_tokens,
            "fill_ratio": policy.fill_ratio,
            "preload_tokens": policy.preload_tokens,
            "message": (
                f"已切换至{policy.complexity.value}；私聊和群聊上下文均已设置为"
                f"最大 {policy.max_context_tokens}、目标 {policy.target_context_tokens} Token。"
            ),
        }

    @staticmethod
    def _model_compatibility(
        ai_packages: Mapping[str, Any], required_context_tokens: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        compatible: list[dict[str, Any]] = []
        incompatible: list[dict[str, Any]] = []
        for model in _configured_chat_models(ai_packages):
            row = _model_compatibility_view(model)
            target = (
                compatible
                if row["max_context_tokens"] >= int(required_context_tokens)
                else incompatible
            )
            target.append(row)
        return compatible, incompatible


def _configured_chat_models(ai_packages: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for package in _object_sequence(ai_packages.get("api_packages")):
        if not _configured_package(package):
            continue
        for model in _object_sequence(package.get("models")):
            if _configured_chat_model(model):
                result.append(model)
    return result


def _configured_package(value: Any) -> bool:
    if not isinstance(value, Mapping) or not bool(value.get("enabled", True)):
        return False
    credential = value.get("credential")
    return isinstance(credential, Mapping) and bool(credential.get("configured"))


def _configured_chat_model(value: Any) -> bool:
    if not isinstance(value, Mapping) or not bool(value.get("enabled", True)):
        return False
    return "chat.completion" in {str(item) for item in value.get("capabilities") or ()}


def _object_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _model_compatibility_view(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "backend_id": str(model.get("backend_id") or ""),
        "display_name": str(model.get("display_name") or model.get("model_key") or "主对话模型"),
        "max_context_tokens": int(model.get("max_context_tokens") or 0),
    }


__all__ = ["ThinkingSettingsController"]
