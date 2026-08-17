"""Small projections shared by background writers and MainCore.

The author-facing character projection stays deliberately narrow.  The
MainCore projection is a frozen read model made only from author prose and
stable neutral keys; it carries no derived title, summary, task, entry point,
or canon judgement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .domain import (
    BackgroundAuthorInput,
    BackgroundAuthorKind,
    BackgroundAuthorState,
    BackgroundStorySource,
    BackgroundTimelineEvent,
    RoleCurrentView,
)


@dataclass(frozen=True, slots=True)
class MainCoreBackgroundText:
    """One complete author-written work with a non-creative stable key."""

    stable_key: str
    content: str


@dataclass(frozen=True, slots=True)
class MainCoreStorySituation:
    """One story module together with the character's recent lived progress."""

    story: BackgroundStorySource
    recent_progress: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MainCoreBackgroundView:
    """All background material frozen for one MainCore run."""

    enabled: bool = False
    world_changes: tuple[MainCoreBackgroundText, ...] = ()
    life_direction: MainCoreBackgroundText | None = None
    current_view: RoleCurrentView | None = None
    story_sources: tuple[MainCoreStorySituation, ...] = ()
    timeline: tuple[BackgroundTimelineEvent, ...] = ()


def main_core_background_view(
    *,
    enabled: bool = False,
    world_state: BackgroundAuthorState | None,
    life_state: BackgroundAuthorState | None,
    current_view: RoleCurrentView | None,
    story_sources: tuple[MainCoreStorySituation, ...],
    timeline: tuple[BackgroundTimelineEvent, ...],
) -> MainCoreBackgroundView:
    """Build the immutable projection without interpreting the prose."""

    return MainCoreBackgroundView(
        enabled=bool(enabled),
        world_changes=_world_materials(world_state),
        life_direction=_life_material(life_state),
        current_view=current_view,
        story_sources=tuple(story_sources),
        timeline=tuple(timeline),
    )


def background_relevance(source: BackgroundAuthorInput) -> str:
    parts = (
        source.current_view.doing,
        source.current_view.current_concern,
        _author_content(source.author_state),
    )
    return "\n".join(item for item in parts if item)[:4_000]


def filter_background_character_projection(
    kind: BackgroundAuthorKind,
    rendered: str,
) -> str:
    if kind is BackgroundAuthorKind.WORLD:
        return ""
    groups = _projection_groups(rendered)
    allowed = {
        "核心身份",
        "身份、经历与生活现状",
        "性格与看重的事",
        "思考与行动方式",
        "日常习惯与情绪表现",
        "与人相处的方式",
        "关系边界与禁区",
        "喜欢和感兴趣的事",
        "不喜欢的事",
        "会做什么",
        "知道与不知道的事",
        "做不到或受限制的事",
    }
    selected = tuple((label, lines) for label, lines in groups if label in allowed)
    return "\n\n".join(f"[{label}]\n" + "\n".join(lines) for label, lines in selected if lines)


def _projection_groups(rendered: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    groups: list[tuple[str, tuple[str, ...]]] = []
    label = ""
    lines: list[str] = []
    for raw in str(rendered or "").splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]") and len(line) > 2:
            if label and lines:
                groups.append((label, tuple(lines)))
            label = line[1:-1].strip()
            lines = []
        elif label and line:
            lines.append(line)
    if label and lines:
        groups.append((label, tuple(lines)))
    return tuple(groups)


def _world_materials(
    state: BackgroundAuthorState | None,
) -> tuple[MainCoreBackgroundText, ...]:
    if state is None:
        return ()
    raw_items = state.content.get("items")
    bodies: list[str] = []
    if isinstance(raw_items, (list, tuple)):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            body = str(item.get("body") or "").strip()
            if body:
                bodies.append(body)
    if not bodies:
        fallback = str(state.content.get("text") or "").strip()
        if fallback:
            bodies.append(fallback)
    return tuple(
        MainCoreBackgroundText(
            stable_key=f"background:world:{state.state_version}:{ordinal}",
            content=body,
        )
        for ordinal, body in enumerate(bodies, start=1)
    )


def _life_material(
    state: BackgroundAuthorState | None,
) -> MainCoreBackgroundText | None:
    if state is None:
        return None
    content = _author_content(state)
    if not content:
        return None
    return MainCoreBackgroundText(
        stable_key=f"background:life-direction:{state.state_version}",
        content=content,
    )


def _author_content(state: BackgroundAuthorState) -> str:
    text = str(state.content.get("text") or "").strip()
    if text:
        return text
    raw_items: Any = state.content.get("items")
    if isinstance(raw_items, (list, tuple)):
        values = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            value = str(item.get("body") or item.get("life") or "").strip()
            if value:
                values.append(value)
        return "\n\n".join(values)
    return ""


__all__ = [
    "MainCoreBackgroundText",
    "MainCoreBackgroundView",
    "MainCoreStorySituation",
    "background_relevance",
    "filter_background_character_projection",
    "main_core_background_view",
]
