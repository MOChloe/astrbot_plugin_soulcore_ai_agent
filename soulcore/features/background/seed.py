"""Role-shared world seeds and creative boundaries.

These user-authored records survive generated-background rebuilds. They are
inputs to the five authors and are not generated WorldInfo.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class BoundarySeverity(StrEnum):
    HARD = "HARD"
    PREFERENCE = "PREFERENCE"


class ExpansionPolicy(StrEnum):
    OPEN = "OPEN"
    CANON_GUARDED = "CANON_GUARDED"


def normalize_world_lore_input(
    title: Any,
    content: Any,
    aliases: Sequence[Any],
    tags: Sequence[Any],
    importance: Any,
) -> dict[str, Any]:
    """Normalize one user-authored lore record independently of persistence."""

    return {
        "title": _required_text(title, "title", maximum=200),
        "aliases": _normalized_strings(aliases, maximum=30),
        "tags": _normalized_strings(tags, maximum=30),
        "content": _required_text(content, "content", maximum=100_000),
        "importance": _normalized_score(importance),
    }


def normalize_creative_boundary_input(
    severity: Any,
    category: Any,
    rule_text: Any,
    positive_space: Any,
    enabled: Any,
) -> dict[str, Any]:
    """Normalize one creative boundary independently of persistence."""

    return {
        "severity": BoundarySeverity(str(severity).upper()).value,
        "category": _required_text(category, "category", maximum=80).upper(),
        "rule_text": _required_text(rule_text, "rule_text", maximum=4000),
        "positive_space": str(positive_space or "").strip()[:4000],
        "enabled": bool(enabled),
    }


def _required_text(value: Any, field: str, *, maximum: int) -> str:
    text = str(value or "").strip()
    if not text or len(text) > maximum:
        raise ValueError(f"{field} must contain between 1 and {maximum} characters")
    return text


def _normalized_strings(values: Sequence[Any], *, maximum: int) -> list[str]:
    result = list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    if len(result) > maximum or any(len(item) > 200 for item in result):
        raise ValueError("string list is too large")
    return result


def _normalized_score(value: Any) -> float:
    score = float(value)
    if not 0 <= score <= 1:
        raise ValueError("importance must be between 0 and 1")
    return score


def expansion_boundary(policy: ExpansionPolicy | str) -> str:
    try:
        normalized = ExpansionPolicy(str(policy).strip().upper())
    except ValueError:
        return ""
    if normalized is ExpansionPolicy.OPEN:
        return (
            "可以主动创造尚未规定的地区、人物、组织、关系和局部历史，"
            "让世界在不改写已成立事实与规则的前提下自然扩展。"
        )
    return (
        "可以在既有世界格局与作品主线内补充局部人物、地点、关系和事件；"
        "不得改写已知世界格局、作品主线或重大既定事实。"
    )


@dataclass(frozen=True, slots=True)
class WorldLoreEntry:
    lore_id: int
    title: str
    content: str
    aliases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    importance: float = 0.5
    revision: int = 1


@dataclass(frozen=True, slots=True)
class CreativeBoundary:
    boundary_id: int
    severity: BoundarySeverity
    category: str
    rule_text: str
    positive_space: str = ""
    revision: int = 1
    enabled: bool = True

    def render(self) -> str:
        label = "不可突破" if self.severity is BoundarySeverity.HARD else "优先遵循"
        alternative = f"；仍可发展的空间：{self.positive_space}" if self.positive_space else ""
        return f"- [{label}｜{self.category}] {self.rule_text}{alternative}"


@dataclass(frozen=True, slots=True)
class WorldDefinition:
    profile_id: str
    revision: int = 1
    world_brief: str = ""
    world_rules: str = ""
    life_direction: str = ""
    world_texture: str = ""
    expansion_policy: ExpansionPolicy = ExpansionPolicy.OPEN
    lore_index: tuple[WorldLoreEntry, ...] = ()
    boundaries: tuple[CreativeBoundary, ...] = ()

    def render_shared(self) -> str:
        parts = [self.world_brief.strip()]
        if self.world_rules.strip():
            parts.append("世界中始终成立的规则：\n" + self.world_rules.strip())
        boundaries = [item.render() for item in self.boundaries]
        if boundaries:
            parts.append("当前人物的发展边界与偏好：\n" + "\n".join(boundaries))
        if self.world_texture.strip():
            parts.append("世界氛围与叙事基调：\n" + self.world_texture.strip())
        return "\n\n".join(part for part in parts if part)

    def render_for_creation(self) -> str:
        parts = [self.render_shared()]
        if self.life_direction.strip():
            parts.append("默认初始人生方向：\n" + self.life_direction.strip())
        parts.append(expansion_boundary(self.expansion_policy))
        return "\n\n".join(part for part in parts if part)


__all__ = [
    "BoundarySeverity",
    "CreativeBoundary",
    "ExpansionPolicy",
    "WorldDefinition",
    "WorldLoreEntry",
    "expansion_boundary",
    "normalize_creative_boundary_input",
    "normalize_world_lore_input",
]
