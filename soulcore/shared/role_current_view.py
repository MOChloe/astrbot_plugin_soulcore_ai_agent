"""Deterministic, prose-preserving background projections for MainCore."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

MAX_MAIN_CORE_EXPERIENCES = 6
MAX_MAIN_CORE_LEFTOVERS = 8
MAX_MAIN_CORE_WORLD_CHANGES = 2
MAX_MAIN_CORE_STORY_CANDIDATES = 12


@runtime_checkable
class RoleCurrentViewProjection(Protocol):
    narrative_time: str
    location: str
    doing: str
    body_state: str
    mood: str
    intention: str
    current_concern: str


@runtime_checkable
class MainCoreStorySituationProjection(Protocol):
    story: Any
    recent_progress: tuple[str, ...]


@runtime_checkable
class MainCoreBackgroundViewProjection(Protocol):
    world_changes: tuple[Any, ...]
    life_direction: Any | None
    current_view: RoleCurrentViewProjection | None
    story_sources: tuple[MainCoreStorySituationProjection, ...]
    timeline: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class RoleCurrentViewFragment:
    """One complete work offered to the context allocator as an atomic unit."""

    fragment_id: str
    kind: str
    body: str
    sequence: int
    reference_key: str = ""
    protected: bool = False


def role_current_view_fields(value: RoleCurrentViewProjection) -> tuple[tuple[str, str], ...]:
    return (
        ("时间", value.narrative_time),
        ("地点", value.location),
        ("正在做", value.doing),
        ("身体状态", value.body_state),
        ("心情", value.mood),
        ("打算", value.intention),
        ("当前牵挂", value.current_concern),
    )


def main_core_background_fragments(
    value: MainCoreBackgroundViewProjection | None,
) -> tuple[RoleCurrentViewFragment, ...]:
    """Project complete works without summaries, scoring, or inferred facts."""

    if value is None:
        return ()
    fragments: list[RoleCurrentViewFragment] = []
    _append_core_fragments(fragments, value)
    _append_timeline_fragments(fragments, value.timeline)
    _append_world_fragments(fragments, value.world_changes)
    _append_story_fragments(fragments, value.story_sources)
    return tuple(fragments)


def _append_core_fragments(
    fragments: list[RoleCurrentViewFragment],
    value: MainCoreBackgroundViewProjection,
) -> None:
    life = value.life_direction
    life_text = _attribute(life, "content")
    if life_text:
        fragments.append(
            RoleCurrentViewFragment(
                fragment_id="life-direction",
                kind="life-direction",
                body=life_text,
                sequence=0,
                reference_key=_attribute(life, "stable_key"),
                protected=True,
            )
        )

    current = value.current_view
    if current is not None:
        rendered = _render_current(current)
        if rendered:
            fragments.append(
                RoleCurrentViewFragment(
                    fragment_id="current",
                    kind="current",
                    body=rendered,
                    sequence=0,
                    protected=True,
                )
            )


def _append_timeline_fragments(
    fragments: list[RoleCurrentViewFragment],
    timeline: tuple[Any, ...],
) -> None:
    events = _newest_events(timeline)
    if events:
        latest = events[0]
        fragments.append(
            RoleCurrentViewFragment(
                fragment_id=f"latest-experience:{_event_id(latest)}",
                kind="latest-experience",
                body=_attribute(latest, "content"),
                sequence=0,
                reference_key=_attribute(latest, "public_ref"),
                protected=True,
            )
        )
        remaining = events[1:]
        optional_limit = max(0, MAX_MAIN_CORE_EXPERIENCES - 1)
        ranked_remaining = tuple(enumerate(remaining, start=1))
        ordinary_events = tuple(
            (rank, event) for rank, event in ranked_remaining if not _is_keyframe(event)
        )[:optional_limit]
        keyframe_events = tuple(
            (rank, event) for rank, event in ranked_remaining if _is_keyframe(event)
        )[:optional_limit]
        for kind, candidates in (
            ("ordinary-experience", ordinary_events),
            ("keyframe-experience", keyframe_events),
        ):
            for rank, event in candidates:
                fragments.append(
                    RoleCurrentViewFragment(
                        fragment_id=f"experience:{_event_id(event)}",
                        kind=kind,
                        body=_attribute(event, "content"),
                        sequence=rank,
                        reference_key=_attribute(event, "public_ref"),
                    )
                )

    leftover_events = tuple(
        event
        for event in _newest_events(timeline)
        if _attribute(event, "leftover_text") and not _attribute(event, "leftover_retired_at")
    )[:MAX_MAIN_CORE_LEFTOVERS]
    for rank, event in enumerate(leftover_events):
        public_ref = _attribute(event, "public_ref")
        fragments.append(
            RoleCurrentViewFragment(
                fragment_id=f"leftover:{_event_id(event)}",
                kind="leftover",
                body=_attribute(event, "leftover_text"),
                sequence=rank,
                reference_key=f"{public_ref}:leftover" if public_ref else "",
            )
        )


def _append_world_fragments(
    fragments: list[RoleCurrentViewFragment],
    world_changes: tuple[Any, ...],
) -> None:
    for rank, change in enumerate(world_changes[:MAX_MAIN_CORE_WORLD_CHANGES]):
        content = _attribute(change, "content")
        if not content:
            continue
        fragments.append(
            RoleCurrentViewFragment(
                fragment_id=f"world:{rank + 1}",
                kind="world",
                body=content,
                sequence=rank,
                reference_key=_attribute(change, "stable_key"),
            )
        )


def _append_story_fragments(
    fragments: list[RoleCurrentViewFragment],
    story_sources: tuple[MainCoreStorySituationProjection, ...],
) -> None:
    seen = 0
    for situation in story_sources[:MAX_MAIN_CORE_STORY_CANDIDATES]:
        story = situation.story
        recent_progress = situation.recent_progress
        content = _attribute(story, "module_text")
        if not content:
            continue
        story_id = getattr(story, "story_source_id", seen + 1)
        ref = _attribute(story, "public_ref")
        body_parts = [content]
        if recent_progress:
            progress_text = "\n\n".join(str(p) for p in recent_progress if p)
            body_parts.append(f"你在场时——\n{progress_text}")
        fragments.append(
            RoleCurrentViewFragment(
                fragment_id=f"story:{story_id}",
                kind="story",
                body="\n\n".join(b for b in body_parts if b),
                sequence=seen,
                reference_key=ref,
            )
        )
        seen += 1


def role_current_view_fragments(
    value: RoleCurrentViewProjection | None,
) -> tuple[RoleCurrentViewFragment, ...]:
    """Project a current-view utility without manufacturing titles or summaries."""

    if value is None:
        return ()
    fragments: list[RoleCurrentViewFragment] = []
    current = _render_current(value)
    if current:
        fragments.append(
            RoleCurrentViewFragment(
                fragment_id="current",
                kind="current",
                body=current,
                sequence=0,
                protected=True,
            )
        )
    events = _newest_events(getattr(value, "recent_experiences", ()))
    for rank, event in enumerate(events[:MAX_MAIN_CORE_EXPERIENCES]):
        body = _attribute(event, "content")
        if body:
            fragments.append(
                RoleCurrentViewFragment(
                    fragment_id=f"experience:{_event_id(event)}",
                    kind="latest-experience" if rank == 0 else "experience",
                    body=body,
                    sequence=rank,
                    reference_key=_attribute(event, "public_ref"),
                    protected=rank == 0,
                )
            )
    return tuple(fragments)


def render_role_current_view(value: RoleCurrentViewProjection | None) -> str:
    """Readable current snapshot plus complete recent prose, for diagnostics."""

    if value is None:
        return ""
    current = _render_current(value)
    experiences = [
        _attribute(event, "content")
        for event in _newest_events(getattr(value, "recent_experiences", ()))
        if _attribute(event, "content")
    ][:MAX_MAIN_CORE_EXPERIENCES]
    parts = [current]
    if experiences:
        parts.append("近期经历：\n" + "\n\n".join(experiences))
    return "\n\n".join(part for part in parts if part)


def _render_current(value: RoleCurrentViewProjection) -> str:
    return "\n".join(
        f"{label}：{text}"
        for label, raw in role_current_view_fields(value)
        if (text := str(raw or "").strip())
    )


def _newest_events(values: Any) -> tuple[Any, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    nonempty = tuple(item for item in values if _attribute(item, "content"))
    return tuple(sorted(nonempty, key=_event_order_key, reverse=True))


def _event_order_key(value: Any) -> tuple[str, int]:
    end_at = getattr(value, "frame_end_at", None)
    end_key = end_at.isoformat() if isinstance(end_at, datetime) else str(end_at or "")
    return end_key, _event_id(value)


def _event_id(value: Any) -> int:
    return int(getattr(value, "timeline_event_id", 0) or 0)


def _is_keyframe(value: Any) -> bool:
    source = getattr(value, "source", "")
    source_value = getattr(source, "value", source)
    return str(source_value or "").upper() == "KEYFRAME"


def _attribute(value: Any, name: str) -> str:
    return str(getattr(value, name, "") or "").strip()


__all__ = [
    "MAX_MAIN_CORE_EXPERIENCES",
    "MAX_MAIN_CORE_LEFTOVERS",
    "MAX_MAIN_CORE_STORY_CANDIDATES",
    "MAX_MAIN_CORE_WORLD_CHANGES",
    "MainCoreBackgroundViewProjection",
    "MainCoreStorySituationProjection",
    "RoleCurrentViewFragment",
    "RoleCurrentViewProjection",
    "main_core_background_fragments",
    "render_role_current_view",
    "role_current_view_fields",
    "role_current_view_fragments",
]
