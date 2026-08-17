"""Normalize one parsed background creation into the persistence DTO."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from .domain import (
    BackgroundAuthorInput,
    BackgroundAuthorKind,
    BackgroundDraft,
    BackgroundStorySource,
    BackgroundStorySourceDraft,
    BackgroundTimelineEvent,
    BackgroundTimelineEventDraft,
    BackgroundTimelineSource,
    BackgroundVisibleReferences,
)
from .output_contract import BackgroundOutputError

_VISIBLE_REFERENCE = re.compile(r"(?i)(?P<prefix>[MV])\s*(?P<ordinal>[0-9]+)")


def normalize_optional_references(
    author_kind: BackgroundAuthorKind,
    creator: Mapping[str, Any],
    *,
    visible_references: BackgroundVisibleReferences | None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Keep valid optional M/V actions without rejecting the narrative around them."""

    result = dict(creator)
    if author_kind not in {BackgroundAuthorKind.ORDINARY, BackgroundAuthorKind.KEYFRAME}:
        return result, ()
    visible = visible_references or BackgroundVisibleReferences()
    normalizations: list[str] = []
    for field, prefix, limit, available in (
        ("engage_module_ordinals", "M", 3, visible.story_sources),
        ("conclude_module_ordinal", "M", 3, visible.story_sources),
        ("retire_leftover_ordinal", "V", 1, visible.timeline_events),
    ):
        raw = str(result.get(field) or "").strip()
        canonical, changes = _normalize_optional_reference_field(
            raw,
            prefix=prefix,
            limit=limit,
            available=available,
        )
        result[field] = canonical
        normalizations.extend(f"{field}_{change}" for change in changes)
    return result, tuple(dict.fromkeys(normalizations))


def _normalize_optional_reference_field(
    raw: str,
    *,
    prefix: str,
    limit: int,
    available: Mapping[str, object],
) -> tuple[str, tuple[str, ...]]:
    if not raw:
        return "", ()
    normalized = unicodedata.normalize("NFKC", raw).upper()
    candidates = [
        f"{match.group('prefix').upper()}{int(match.group('ordinal'))}"
        for match in _VISIBLE_REFERENCE.finditer(normalized)
        if match.group("prefix").upper() == prefix and int(match.group("ordinal")) >= 1
    ]
    deduplicated = list(dict.fromkeys(candidates))
    available_values = [item for item in deduplicated if item in available]
    canonical = ", ".join(available_values[:limit])
    changes: list[str] = []
    if normalized != canonical.upper():
        changes.append("syntax_normalized")
    if len(available_values) < len(deduplicated) or not candidates:
        changes.append("invalid_references_dropped")
    if len(available_values) > limit:
        changes.append("excess_references_dropped")
    return canonical, tuple(changes)


def draft_from_creator(
    author_kind: BackgroundAuthorKind,
    creator: Mapping[str, Any],
    *,
    source: BackgroundAuthorInput,
    visible_references: BackgroundVisibleReferences | None = None,
) -> BackgroundDraft:
    """Turn one structurally parsed creation into the only persistence DTO."""

    if source.author_kind is not author_kind:
        raise BackgroundOutputError("creator source belongs to another background author")
    if author_kind in {
        BackgroundAuthorKind.ORDINARY,
        BackgroundAuthorKind.KEYFRAME,
    }:
        if visible_references is None:
            raise BackgroundOutputError("life frame is missing its final visible reference map")
        normalized = _role_frame_draft(
            author_kind,
            creator,
            source,
            visible_references,
        )
    else:
        normalized = _upper_draft(author_kind, creator)
    return BackgroundDraft(
        **normalized,
        creator_output=dict(creator),
    )


def _upper_draft(
    kind: BackgroundAuthorKind,
    creator: Mapping[str, Any],
) -> dict[str, Any]:
    if kind is BackgroundAuthorKind.WORLD:
        return _world_draft(creator)
    if kind is BackgroundAuthorKind.LIFE_DIRECTION:
        return _life_direction_draft(creator)
    return _story_source_draft(creator)


def _world_draft(
    creator: Mapping[str, Any],
) -> dict[str, Any]:
    items = _records(creator.get("items"))
    changes = tuple(_required(item.get("body"), "world change") for item in items)
    if not changes:
        raise BackgroundOutputError("WORLD creator has no world change")
    return {
        "content": {
            "items": [{"body": body} for body in changes],
        },
    }


def _life_direction_draft(creator: Mapping[str, Any]) -> dict[str, Any]:
    items = _records(creator.get("items"))
    if len(items) != 1:
        raise BackgroundOutputError("LIFE_DIRECTION creator must return one direction")
    body = _required(items[0].get("life"), "life direction")
    return {
        "content": {"text": body},
    }


def _story_source_draft(creator: Mapping[str, Any]) -> dict[str, Any]:
    items = _records(creator.get("story_sources"))
    if not items:
        raise BackgroundOutputError("STORY_SOURCE creator has no story module")
    story_sources: list[BackgroundStorySourceDraft] = []
    for item in items:
        body = _required(item.get("module_text"), "story module")
        story_sources.append(
            BackgroundStorySourceDraft(
                module_text=body,
            )
        )
    return {
        "content": {},
        "story_sources": tuple(story_sources),
    }


def _role_frame_draft(
    author_kind: BackgroundAuthorKind,
    creator: Mapping[str, Any],
    source: BackgroundAuthorInput,
    visible_references: BackgroundVisibleReferences,
) -> dict[str, Any]:
    if author_kind is BackgroundAuthorKind.ORDINARY:
        interval = source.ordinary_frame_interval
        expected_source = BackgroundTimelineSource.ORDINARY
    else:
        interval = source.keyframe_frame_interval
        expected_source = BackgroundTimelineSource.KEYFRAME
    if interval is None:
        raise BackgroundOutputError(
            f"{author_kind.value.lower()} frame has no settled time interval"
        )
    raw_events = _records(creator.get("timeline_events"))
    if len(raw_events) != 1:
        raise BackgroundOutputError("life frame must contain one event")
    leftover_text = _text(creator.get("leftover_text"))
    event = _offline_event(
        raw_events[0],
        expected_source=expected_source,
        frame_start_at=interval.start_at,
        frame_end_at=interval.end_at,
        leftover_text=leftover_text,
    )
    normalized: dict[str, Any] = {
        "content": {},
        "timeline_events": (event,),
        "current_view": _current_view(creator.get("current_view")),
        "retired_timeline_event_ids": _resolve_retire_ordinal(
            creator.get("retire_leftover_ordinal"),
            visible_references.timeline_events,
        ),
        "engaged_story_ids": _resolve_module_ordinals(
            creator.get("engage_module_ordinals"),
            visible_references.story_sources,
        ),
        "concluded_story_ids": _resolve_module_ordinals(
            creator.get("conclude_module_ordinal"),
            visible_references.story_sources,
        ),
    }
    if author_kind is BackgroundAuthorKind.ORDINARY:
        normalized.update(
            consumed_foreground_through_message_id=_message_cursor(source),
            consumed_foreground_through_run_id=_run_cursor(source),
        )
    return normalized


def _offline_event(
    raw: Mapping[str, Any],
    *,
    expected_source: BackgroundTimelineSource,
    frame_start_at: Any,
    frame_end_at: Any,
    leftover_text: str,
) -> BackgroundTimelineEventDraft:
    event_source = str(raw.get("source") or "").strip().upper()
    if event_source != expected_source.value:
        raise BackgroundOutputError("life frame returned an invalid timeline source")
    content = _required(raw.get("content"), "life event content")
    return BackgroundTimelineEventDraft(
        source=expected_source,
        content=content,
        frame_start_at=frame_start_at,
        frame_end_at=frame_end_at,
        leftover_text=leftover_text,
    )


def _current_view(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BackgroundOutputError("life frame must publish one current role view")
    updates = dict(value)
    narrative_time = _required_frame_text(
        updates.get("narrative_time"),
        "current role narrative time",
    )
    location = _required_frame_text(
        updates.get("location"),
        "current role location",
    )
    result: dict[str, Any] = {
        "narrative_time": narrative_time,
        "location": location,
    }
    for name, field in (
        ("doing", "current role activity"),
        ("body_state", "current role body state"),
        ("mood", "current role mood"),
        ("intention", "current role intention"),
        ("current_concern", "current role concern"),
    ):
        if name in updates:
            result[name] = _present_frame_text(updates[name], field)
    return result


def _resolve_retire_ordinal(
    ordinal_raw: Any,
    recent_timeline: Mapping[str, BackgroundTimelineEvent],
) -> tuple[int, ...]:
    """Resolve a V{n} ordinal from the author prompt to a timeline_event_id.

    The mapping contains only exact V labels visible to the model.
    """
    ordinal_str = str(ordinal_raw or "").strip()
    if not ordinal_str:
        return ()
    if not ordinal_str.startswith("V") or not ordinal_str[1:].isdigit():
        raise BackgroundOutputError(
            f"留下变化已解决 ordinal must be V1, V2, … (got {ordinal_str!r})"
        )
    if int(ordinal_str[1:]) < 1:
        raise BackgroundOutputError(f"留下变化已解决 ordinal must be >= V1 (got {ordinal_str!r})")
    target = recent_timeline.get(ordinal_str)
    if target is None:
        raise BackgroundOutputError(
            f"留下变化已解决 ordinal {ordinal_str!r} is out of range "
            "or was not visible in the final author prompt"
        )
    if not target.leftover_text or target.leftover_retired_at:
        raise BackgroundOutputError(
            f"留下变化已解决 ordinal {ordinal_str!r} points to an experience with no active leftover"
        )
    return (target.timeline_event_id,)


def _resolve_module_ordinals(
    ordinals_raw: Any,
    story_sources: Mapping[str, BackgroundStorySource],
) -> tuple[int, ...]:
    """Resolve M{n} ordinals from the author prompt to story_source_ids.

    Each label resolves only to the exact module shown under that label.
    Upper limit: 3 modules per experience.
    """
    ordinals_str = str(ordinals_raw or "").strip()
    if not ordinals_str:
        return ()
    raw_parts = ordinals_str.split(",")
    if any(not part.strip() for part in raw_parts):
        raise BackgroundOutputError("介入模组/模组已了结 ordinals contain an empty item")
    parts = [part.strip() for part in raw_parts]
    if len(parts) > 3:
        raise BackgroundOutputError(
            f"介入模组/模组已了结 ordinals must reference at most 3 modules (got {len(parts)})"
        )
    result: list[int] = []
    for part in parts:
        if not part.startswith("M") or not part[1:].isdigit():
            raise BackgroundOutputError(
                f"介入模组/模组已了结 ordinal must be M1, M2, … (got {part!r})"
            )
        if int(part[1:]) < 1:
            raise BackgroundOutputError(f"介入模组/模组已了结 ordinal must be >= M1 (got {part!r})")
        target = story_sources.get(part)
        if target is None:
            raise BackgroundOutputError(
                f"介入模组/模组已了结 ordinal {part!r} is out of range "
                "or was not visible in the final author prompt"
            )
        result.append(target.story_source_id)
    return tuple(dict.fromkeys(result))  # deduplicate while preserving order


def _message_cursor(source: BackgroundAuthorInput) -> int:
    return max(
        int(source.foreground_message_through),
        max((item.message_id for item in source.foreground_messages), default=0),
    )


def _run_cursor(source: BackgroundAuthorInput) -> int:
    return max(
        int(source.foreground_run_through),
        max((item.run_id for item in source.foreground_runs), default=0),
    )


def _required(value: Any, field: str) -> str:
    normalized = _text(value)
    if not normalized:
        raise BackgroundOutputError(f"{field} is required")
    return normalized


def _required_frame_text(value: Any, field: str) -> str:
    return _present_frame_text(value, field)


def _present_frame_text(value: Any, field: str) -> str:
    return _required(value, field)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _records(value: Any) -> list[dict[str, Any]]:
    if value in (None, (), []):
        return []
    if not isinstance(value, (list, tuple)):
        raise BackgroundOutputError("creator records must be a list")
    if any(not isinstance(item, Mapping) for item in value):
        raise BackgroundOutputError("creator records must contain mappings")
    return [dict(item) for item in value]


__all__ = ["draft_from_creator", "normalize_optional_references"]
