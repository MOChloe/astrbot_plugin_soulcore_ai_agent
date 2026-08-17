"""Frozen MainCore background projection reads."""

from __future__ import annotations

import sqlite3

from ....shared.role_current_view import MAX_MAIN_CORE_STORY_CANDIDATES
from ..domain import (
    BackgroundAuthorKind,
    BackgroundAuthorState,
    BackgroundTimelineEvent,
    RoleCurrentView,
)
from ..prompt_projection import (
    MainCoreBackgroundView,
    MainCoreStorySituation,
    main_core_background_view,
)
from .rows import (
    author_state_from_row,
    current_view_from_row,
    timeline_event_from_row,
)
from .story_supply import ranked_story_supply

_TIMELINE_SCAN_LIMIT = 100
_ACTIVE_LEFTOVER_LIMIT = 8


def read_main_core_background_view_sql(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
) -> MainCoreBackgroundView | None:
    enabled = _read_enabled(conn, profile_id, instance_id)
    states = _read_states(conn, profile_id, instance_id)
    projection = main_core_background_view(
        enabled=enabled,
        world_state=states.get(BackgroundAuthorKind.WORLD),
        life_state=states.get(BackgroundAuthorKind.LIFE_DIRECTION),
        current_view=_read_current_view(conn, profile_id, instance_id),
        story_sources=_read_stories(conn, profile_id, instance_id),
        timeline=_read_timeline(conn, profile_id, instance_id),
    )
    return projection if enabled or _has_content(projection) else None


def _read_enabled(conn: sqlite3.Connection, profile_id: str, instance_id: str) -> bool:
    row = conn.execute(
        """SELECT enabled FROM background_instances
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    return bool(row is not None and int(row["enabled"] or 0))


def _read_current_view(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
) -> RoleCurrentView | None:
    row = conn.execute(
        """SELECT * FROM background_role_current_views
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    current_view = current_view_from_row(row)
    return None if current_view.empty else current_view


def _read_states(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
) -> dict[BackgroundAuthorKind, BackgroundAuthorState]:
    rows = conn.execute(
        """SELECT * FROM background_author_states
        WHERE profile_id = ? AND instance_id = ?
          AND author_kind IN (?, ?)""",
        (
            profile_id,
            instance_id,
            BackgroundAuthorKind.WORLD.value,
            BackgroundAuthorKind.LIFE_DIRECTION.value,
        ),
    ).fetchall()
    return {
        BackgroundAuthorKind(str(row["author_kind"])): author_state_from_row(row) for row in rows
    }


def _read_stories(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
) -> tuple[MainCoreStorySituation, ...]:
    stories = tuple(
        candidate.story
        for candidate in ranked_story_supply(conn, profile_id, instance_id)[
            :MAX_MAIN_CORE_STORY_CANDIDATES
        ]
    )
    progress_by_story = _read_story_progress_batch(
        conn,
        profile_id,
        instance_id,
        tuple(story.story_source_id for story in stories),
    )
    return tuple(
        MainCoreStorySituation(
            story=story,
            recent_progress=progress_by_story.get(story.story_source_id, ()),
        )
        for story in stories
    )


def _read_story_progress_batch(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    story_source_ids: tuple[int, ...],
) -> dict[int, tuple[str, ...]]:
    if not story_source_ids:
        return {}
    placeholders = ",".join("?" for _ in story_source_ids)
    rows = conn.execute(
        f"""SELECT story_source_id, content FROM (
            SELECT link.story_source_id, rte.content,
                   ROW_NUMBER() OVER (
                       PARTITION BY link.story_source_id
                       ORDER BY rte.frame_end_at DESC, rte.event_id DESC
                   ) AS progress_rank
            FROM background_role_timeline_events AS rte
            JOIN background_timeline_event_story_sources AS link
              ON link.event_id = rte.event_id
            WHERE link.story_source_id IN ({placeholders})
              AND rte.profile_id = ? AND rte.instance_id = ?
        )
        WHERE progress_rank <= 3
        ORDER BY story_source_id ASC, progress_rank DESC""",
        (*story_source_ids, profile_id, instance_id),
    ).fetchall()
    grouped: dict[int, list[str]] = {}
    for row in rows:
        content = str(row["content"] or "").strip()
        if content:
            grouped.setdefault(int(row["story_source_id"]), []).append(content)
    return {story_id: tuple(items) for story_id, items in grouped.items()}


def _read_timeline(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
) -> tuple[BackgroundTimelineEvent, ...]:
    recent_rows = conn.execute(
        """SELECT * FROM background_role_timeline_events
        WHERE profile_id = ? AND instance_id = ?
        ORDER BY frame_end_at DESC, event_id DESC
        LIMIT ?""",
        (profile_id, instance_id, _TIMELINE_SCAN_LIMIT),
    ).fetchall()
    active_leftover_rows = conn.execute(
        """SELECT * FROM background_role_timeline_events
        WHERE profile_id = ? AND instance_id = ?
          AND NULLIF(TRIM(leftover_text), '') IS NOT NULL
          AND NULLIF(TRIM(COALESCE(leftover_retired_at, '')), '') IS NULL
        ORDER BY frame_end_at DESC, event_id DESC
        LIMIT ?""",
        (profile_id, instance_id, _ACTIVE_LEFTOVER_LIMIT),
    ).fetchall()
    rows_by_id: dict[int, sqlite3.Row] = {}
    for row in (*recent_rows, *active_leftover_rows):
        rows_by_id.setdefault(int(row["event_id"]), row)
    return tuple(timeline_event_from_row(row) for row in rows_by_id.values())


def _has_content(projection: MainCoreBackgroundView) -> bool:
    return any(
        (
            projection.world_changes,
            projection.life_direction,
            projection.current_view,
            projection.story_sources,
            projection.timeline,
        )
    )


__all__ = ["read_main_core_background_view_sql"]
