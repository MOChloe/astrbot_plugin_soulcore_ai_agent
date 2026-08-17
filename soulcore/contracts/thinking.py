"""Shared work-depth, action, and context policy selected by runtime setting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final


class ThinkingComplexity(StrEnum):
    MINIMAL = "极简"
    LIGHT = "轻量"
    BALANCED = "均衡"
    STANDARD = "标准"
    DEEP = "深入"
    EXTREME = "极致"


@dataclass(frozen=True, slots=True)
class MainCoreThinkingPolicy:
    complexity: ThinkingComplexity
    hard_max_steps: int
    research_requirement: str
    plan_requirement: str
    max_context_tokens: int
    target_context_tokens: int
    fill_ratio: float

    @property
    def preload_tokens(self) -> int:
        return int(self.target_context_tokens * self.fill_ratio)


_MINIMAL_RESEARCH = "材料缺少会直接影响结果的关键事实时才查询；足以判断和表达时停止。"
_BALANCED_RESEARCH = (
    "检查材料是否足够；补足会改变结论、立场或表达的重要事实与不确定点，足够就停止。"
)
_DEEP_RESEARCH = (
    "从相关角度补足并交叉核对会影响结果的关键事实；材料足以支持判断与表达时停止，不展开无关方向。"
)
_EXTREME_RESEARCH = (
    "在相关且能够取得的范围内广泛、深入地收集信息，追查底层问题并交叉核对关键事实；证据充分即停止。"
)
_MINIMAL_PLAN = (
    "通常不制定 Plan；只有复杂结果没有整体方案很可能失去核心时才用“制定Plan”。可以先取得必要信息。"
)
_COMPLEX_PLAN = (
    "简单结果不需要 Plan；复杂表达或作品需要取舍、组织时用“制定Plan”。可以先取得必要信息。"
)
_BENEFICIAL_PLAN = (
    "整体方案会明显改善最终结果时用“制定Plan”；可以先取得必要信息，工具外草稿不算 Plan。"
)
_ALWAYS_PLAN = "第一次有效调用只用“继续行动”提交一条“制定Plan”；收到结果后再继续。"


_POLICIES: Final[Mapping[ThinkingComplexity, MainCoreThinkingPolicy]] = MappingProxyType(
    {
        ThinkingComplexity.MINIMAL: MainCoreThinkingPolicy(
            ThinkingComplexity.MINIMAL,
            hard_max_steps=8,
            research_requirement=_MINIMAL_RESEARCH,
            plan_requirement=_MINIMAL_PLAN,
            max_context_tokens=128_000,
            target_context_tokens=20_000,
            fill_ratio=0.70,
        ),
        ThinkingComplexity.LIGHT: MainCoreThinkingPolicy(
            ThinkingComplexity.LIGHT,
            hard_max_steps=12,
            research_requirement=_MINIMAL_RESEARCH,
            plan_requirement=_COMPLEX_PLAN,
            max_context_tokens=128_000,
            target_context_tokens=40_000,
            fill_ratio=0.65,
        ),
        ThinkingComplexity.BALANCED: MainCoreThinkingPolicy(
            ThinkingComplexity.BALANCED,
            hard_max_steps=18,
            research_requirement=_BALANCED_RESEARCH,
            plan_requirement=_COMPLEX_PLAN,
            max_context_tokens=128_000,
            target_context_tokens=60_000,
            fill_ratio=0.60,
        ),
        ThinkingComplexity.STANDARD: MainCoreThinkingPolicy(
            ThinkingComplexity.STANDARD,
            hard_max_steps=36,
            research_requirement=_BALANCED_RESEARCH,
            plan_requirement=_BENEFICIAL_PLAN,
            max_context_tokens=128_000,
            target_context_tokens=80_000,
            fill_ratio=0.55,
        ),
        ThinkingComplexity.DEEP: MainCoreThinkingPolicy(
            ThinkingComplexity.DEEP,
            hard_max_steps=60,
            research_requirement=_DEEP_RESEARCH,
            plan_requirement=_BENEFICIAL_PLAN,
            max_context_tokens=160_000,
            target_context_tokens=100_000,
            fill_ratio=0.45,
        ),
        ThinkingComplexity.EXTREME: MainCoreThinkingPolicy(
            ThinkingComplexity.EXTREME,
            hard_max_steps=96,
            research_requirement=_EXTREME_RESEARCH,
            plan_requirement=_ALWAYS_PLAN,
            max_context_tokens=200_000,
            target_context_tokens=150_000,
            fill_ratio=0.30,
        ),
    }
)

DEFAULT_THINKING_POLICY: Final = _POLICIES[ThinkingComplexity.STANDARD]


def thinking_policy_from_value(value: Any) -> MainCoreThinkingPolicy:
    try:
        complexity = ThinkingComplexity(str(value or "").strip())
    except ValueError:
        return DEFAULT_THINKING_POLICY
    return _POLICIES[complexity]


def require_thinking_policy(value: Any) -> MainCoreThinkingPolicy:
    """Resolve an administrator value without silently accepting a typo."""

    return _POLICIES[ThinkingComplexity(str(value or "").strip())]


def thinking_policy_options() -> list[dict[str, Any]]:
    return [
        {
            "complexity": policy.complexity.value,
            "hard_max_steps": policy.hard_max_steps,
            "max_context_tokens": policy.max_context_tokens,
            "target_context_tokens": policy.target_context_tokens,
            "fill_ratio": policy.fill_ratio,
            "preload_tokens": policy.preload_tokens,
        }
        for policy in _POLICIES.values()
    ]


__all__ = [
    "DEFAULT_THINKING_POLICY",
    "MainCoreThinkingPolicy",
    "ThinkingComplexity",
    "thinking_policy_from_value",
    "require_thinking_policy",
    "thinking_policy_options",
]
