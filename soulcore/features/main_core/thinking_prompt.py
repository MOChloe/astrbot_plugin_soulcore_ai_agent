"""Model-facing multi-round Agent and Plan guidance for MainCore."""

from __future__ import annotations

from ...contracts.thinking import MainCoreThinkingPolicy
from ..ai.service import MainCoreCommandRegistry

_INFORMATION_COMMANDS = frozenset(
    {
        "想想对某人的印象",
        "回想",
        "翻聊天记录",
        "看看我的安排",
        "查资料",
        "看链接",
        "找图片",
        "看清这张图",
        "找表情",
    }
)

_EXISTING_PLAN_REQUIREMENT = "已有 Plan，按其推进；只有新结果改变目标或关键取舍时才整体替换。"


def thinking_requirement(
    registry: MainCoreCommandRegistry,
    current_plan: str,
    policy: MainCoreThinkingPolicy,
) -> str:
    requirements: list[str] = []
    if any(_visible_command(registry, name) for name in _INFORMATION_COMMANDS):
        requirements.append(policy.research_requirement)
    if _visible_command(registry, "制定Plan"):
        if str(current_plan or "").strip():
            requirements.append(_EXISTING_PLAN_REQUIREMENT)
        else:
            requirements.append(policy.plan_requirement)
    return "\n\n".join(requirements)


def _visible_command(registry: MainCoreCommandRegistry, name: str) -> bool:
    command = registry.get(name)
    return command is not None and command.prompt_visible and not command.terminal


__all__ = ["thinking_requirement"]
