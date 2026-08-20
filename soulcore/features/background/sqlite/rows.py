"""Stable SQLite row-to-domain mappings for background read models."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import Any

from ....contracts.delivery_visibility import delivery_visibility
from ....storage.sqlite.codec import _load, _parse
from ..domain import (
    BackgroundAuthorKind,
    BackgroundAuthorState,
    BackgroundStorySource,
    BackgroundTimelineEvent,
    BackgroundTimelineSource,
    ForegroundContinuityMessage,
    ForegroundContinuityResult,
    ForegroundContinuityRun,
    RoleCurrentView,
)

RowData = sqlite3.Row | Mapping[str, Any]


def json_mapping(value: object) -> dict[str, Any]:
    loaded = _load(value) if isinstance(value, str) else value
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def json_list(value: object) -> list[Any]:
    loaded = _load(value) if isinstance(value, str) else value
    return list(loaded) if isinstance(loaded, list) else []


def author_state_from_row(row: RowData) -> BackgroundAuthorState:
    return BackgroundAuthorState(
        author_kind=BackgroundAuthorKind(str(row["author_kind"])),
        state_version=int(row["state_version"]),
        content=json_mapping(row["state_json"]),
        backend_id=str(row["backend_id"] or ""),
        last_success_at=_parse(row["last_success_at"]),
    )


def story_source_from_row(row: RowData) -> BackgroundStorySource:
    return BackgroundStorySource(
        story_source_id=int(row["story_source_id"]),
        public_ref=str(row["public_ref"]),
        module_text=str(row["module_text"] or "").strip(),
        shown_count=int(row["shown_count"] or 0),
        engagement_state=str(row["engagement_state"] or "PENDING"),
    )


def timeline_event_from_row(row: RowData) -> BackgroundTimelineEvent:
    return BackgroundTimelineEvent(
        timeline_event_id=int(row["event_id"]),
        public_ref=str(row["public_ref"]),
        source=BackgroundTimelineSource(str(row["source"])),
        content=str(row["content"] or "").strip(),
        frame_start_at=_parse(row["frame_start_at"]),
        frame_end_at=_parse(row["frame_end_at"]),
        leftover_text=str(row["leftover_text"] or "").strip(),
        leftover_retired_at=str(row["leftover_retired_at"] or "").strip(),
    )


def foreground_message_from_row(row: RowData) -> ForegroundContinuityMessage:
    components = tuple(item for item in json_list(row["components_json"]) if isinstance(item, dict))
    metadata = json_mapping(row["metadata_json"])
    direction = str(row["direction"] or "")
    return ForegroundContinuityMessage(
        message_id=int(row["message_id"]),
        direction=direction,
        role=str(row["role"] or ""),
        participant_id=str(row["sender_id"] or ""),
        speaker_name=str(row["sender_name"] or ""),
        plain_text=str(row["plain_text"] or ""),
        internal_memo=str(row["internal_memo"] or ""),
        components=components,
        delivery_visibility=delivery_visibility(
            direction,
            str(row["delivery_status"] or ""),
        ),
        occurred_at=_parse(row["occurred_at"]),
        scene_narration_before=_scene_narration_values(metadata.get("scene_narration_before")),
        scene_narration_after=_scene_narration_values(metadata.get("scene_narration_after")),
    )


def _scene_narration_values(value: object) -> tuple[str, ...]:
    raw_values = (value,) if isinstance(value, str) else value if isinstance(value, list) else ()
    return tuple(str(item or "").strip() for item in raw_values if str(item or "").strip())


def foreground_run_from_row(row: RowData) -> ForegroundContinuityRun:
    decision = json_mapping(row["decision_json"])
    raw_results = decision.get("foreground_continuity")
    results = (
        tuple(
            ForegroundContinuityResult(
                command=str(item.get("command") or ""),
                ok=bool(item.get("ok")),
                result=str(item.get("result") or ""),
            )
            for item in raw_results
            if isinstance(item, Mapping)
        )
        if isinstance(raw_results, list)
        else ()
    )
    return ForegroundContinuityRun(
        run_id=int(row["run_id"]),
        source=str(row["source"] or ""),
        reason=str(row["reason"] or ""),
        finished_at=_parse(row["finished_at"]),
        results=results,
    )


def current_view_from_row(row: RowData | None) -> RoleCurrentView:
    if row is None:
        return RoleCurrentView()
    return RoleCurrentView(
        revision=int(row["revision"]),
        as_of=_parse(row["as_of"]),
        source=str(row["source"] or ""),
        source_event_id=(
            int(row["source_event_id"]) if row["source_event_id"] is not None else None
        ),
        source_publication_id=(
            int(row["source_publication_id"]) if row["source_publication_id"] is not None else None
        ),
        narrative_time=str(row["narrative_time"] or "").strip(),
        location=str(row["location"] or "").strip(),
        doing=str(row["doing"] or "").strip(),
        body_state=str(row["body_state"] or "").strip(),
        mood=str(row["mood"] or "").strip(),
        intention=str(row["intention"] or "").strip(),
        current_concern=str(row["current_concern"] or "").strip(),
    )


def instance_config_from_row(row: RowData) -> dict[str, Any]:
    fields = (
        "proactive_frame_prewarm_enabled",
        "initial_life_direction",
        "default_backend_id",
        "ordinary_min_minutes",
        "ordinary_max_minutes",
        "keyframe_every_ordinary",
        "keyframe_max_minutes",
        "story_source_min_minutes",
        "story_source_max_minutes",
        "life_direction_min_minutes",
        "life_direction_max_minutes",
        "world_min_minutes",
        "world_max_minutes",
        "ordinary_since_keyframe",
        "simulated_through_at",
        "foreground_message_cursor",
        "foreground_run_cursor",
        "last_foreground_at",
    )
    result = {name: row[name] for name in fields}
    for name in ("simulated_through_at", "last_foreground_at"):
        result[name] = _parse(result[name])
    result["proactive_frame_prewarm_enabled"] = bool(result["proactive_frame_prewarm_enabled"])
    return result


__all__ = [
    "author_state_from_row",
    "current_view_from_row",
    "foreground_message_from_row",
    "foreground_run_from_row",
    "instance_config_from_row",
    "json_list",
    "story_source_from_row",
    "timeline_event_from_row",
]
