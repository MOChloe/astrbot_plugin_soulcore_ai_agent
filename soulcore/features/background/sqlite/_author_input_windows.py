"""Private SQLite queries for bounded foreground and role-frame input."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from ....storage.sqlite.codec import _dt, _parse
from ....storage.sqlite.foreground_continuity import (
    FOREGROUND_RUN_HAS_BACKGROUND_PROJECTION_SQL,
    foreground_message_is_background_cursor_target_sql,
    foreground_message_is_background_evidence_sql,
)
from ..domain import (
    BackgroundAuthorKind,
    BackgroundDraftStale,
    BackgroundFrameInterval,
    ForegroundContinuityMessage,
    ForegroundContinuityRun,
)
from ..opening_time import opening_handoff_at
from .rows import foreground_message_from_row, foreground_run_from_row

_RECENT_FOREGROUND_LIMIT = 48
_RECENT_FOREGROUND_RUN_LIMIT = 24
_FOREGROUND_INPUT_AUTHORS = frozenset(
    {
        BackgroundAuthorKind.LIFE_DIRECTION,
        BackgroundAuthorKind.STORY_SOURCE,
        BackgroundAuthorKind.KEYFRAME,
        BackgroundAuthorKind.ORDINARY,
    }
)


@dataclass(frozen=True, slots=True)
class OrdinaryAuthorInput:
    foreground_messages: tuple[ForegroundContinuityMessage, ...]
    foreground_runs: tuple[ForegroundContinuityRun, ...]
    message_through: int
    run_through: int
    frame_interval: BackgroundFrameInterval | None


@dataclass(frozen=True, slots=True)
class AuthorInputWindows:
    opening_anchor_at: datetime | None
    opening_keyframe_completed: bool
    ordinary: OrdinaryAuthorInput
    keyframe_frame_interval: BackgroundFrameInterval | None


def _latest_foreground_targets(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    *,
    prompt_now: datetime,
) -> tuple[int, int]:
    # The projection marker proves foreground continuity. Knowledge eligibility
    # belongs to the independent long-term-memory pipeline and must not hide a
    # visible role send from the next background frame.
    through_at = _dt(prompt_now)
    message = conn.execute(
        f"""SELECT COALESCE(MAX(message_id), 0) AS value
        FROM instance_messages AS message
        WHERE message.profile_id = ? AND message.instance_id = ?
          AND {foreground_message_is_background_cursor_target_sql("message")}
          AND json_extract(
                message.metadata_json,
                '$.background_foreground_projection.projected_at'
              ) <= ?""",
        (profile_id, instance_id, through_at),
    ).fetchone()
    run = conn.execute(
        f"""SELECT COALESCE(MAX(run_id), 0) AS value
        FROM instance_core_runs AS core_run
        WHERE profile_id = ? AND instance_id = ?
          AND status = 'COMPLETED'
          AND source IN ('FOREGROUND_MESSAGE', 'DEFERRED_MESSAGE')
          AND decision_json IS NOT NULL
          AND finished_at <= ?
          AND {FOREGROUND_RUN_HAS_BACKGROUND_PROJECTION_SQL}""",
        (profile_id, instance_id, through_at),
    ).fetchone()
    return int(message["value"]), int(run["value"])


def _foreground_targets(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    instance: sqlite3.Row,
    kind: BackgroundAuthorKind,
    *,
    prompt_now: datetime,
) -> tuple[int, int]:
    if kind is BackgroundAuthorKind.KEYFRAME:
        return (
            int(instance["foreground_message_cursor"]),
            int(instance["foreground_run_cursor"]),
        )
    if kind in _FOREGROUND_INPUT_AUTHORS:
        return _latest_foreground_targets(
            conn,
            profile_id,
            instance_id,
            prompt_now=prompt_now,
        )
    return 0, 0


def foreground_for_author(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    kind: BackgroundAuthorKind,
    *,
    after: int,
    through: int,
) -> tuple[ForegroundContinuityMessage, ...]:
    if kind not in _FOREGROUND_INPUT_AUTHORS or through <= after:
        return ()
    rows = conn.execute(
        f"""WITH recent AS (
            SELECT message_id, direction, role, sender_id, sender_name,
                plain_text, internal_memo, components_json, delivery_status,
                metadata_json,
                json_extract(
                  metadata_json,
                  '$.background_foreground_projection.projected_at'
                ) AS occurred_at
            FROM instance_messages AS message
            WHERE message.profile_id = ? AND message.instance_id = ?
              AND message.message_id > ? AND message.message_id <= ?
              AND {foreground_message_is_background_evidence_sql("message")}
            ORDER BY message.message_id DESC
            LIMIT ?
        )
        SELECT * FROM recent ORDER BY message_id""",
        (
            profile_id,
            instance_id,
            int(after),
            int(through),
            _RECENT_FOREGROUND_LIMIT,
        ),
    ).fetchall()
    return tuple(foreground_message_from_row(row) for row in rows)


def foreground_runs_for_author(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    kind: BackgroundAuthorKind,
    *,
    after: int,
    through: int,
) -> tuple[ForegroundContinuityRun, ...]:
    if kind not in _FOREGROUND_INPUT_AUTHORS or through <= after:
        return ()
    rows = conn.execute(
        f"""WITH recent AS (
            SELECT run_id, source, reason, decision_json, finished_at
            FROM instance_core_runs AS core_run
            WHERE profile_id = ? AND instance_id = ?
              AND run_id > ? AND run_id <= ?
              AND status = 'COMPLETED'
              AND source IN ('FOREGROUND_MESSAGE', 'DEFERRED_MESSAGE')
              AND decision_json IS NOT NULL
              AND {FOREGROUND_RUN_HAS_BACKGROUND_PROJECTION_SQL}
            ORDER BY run_id DESC
            LIMIT ?
        )
        SELECT * FROM recent ORDER BY run_id""",
        (
            profile_id,
            instance_id,
            int(after),
            int(through),
            _RECENT_FOREGROUND_RUN_LIMIT,
        ),
    ).fetchall()
    return tuple(foreground_run_from_row(row) for row in rows)


def _continuous_frame_interval(
    instance: sqlite3.Row,
    kind: BackgroundAuthorKind,
    prompt_now: datetime,
) -> BackgroundFrameInterval | None:
    if kind not in {BackgroundAuthorKind.ORDINARY, BackgroundAuthorKind.KEYFRAME}:
        return None
    start_at = _parse(instance["simulated_through_at"]) or _parse(instance["created_at"])
    if start_at is None:
        raise BackgroundDraftStale("background frame has no valid continuity start")
    if prompt_now < start_at:
        raise BackgroundDraftStale("background frame clock moved backwards")
    return BackgroundFrameInterval(start_at=start_at, end_at=prompt_now)


def ordinary_author_input(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    instance: sqlite3.Row,
    kind: BackgroundAuthorKind,
    prompt_now: datetime,
) -> OrdinaryAuthorInput:
    message_target, run_target = _foreground_targets(
        conn,
        profile_id,
        instance_id,
        instance,
        kind,
        prompt_now=prompt_now,
    )
    message_after = (
        int(instance["foreground_message_cursor"]) if kind is BackgroundAuthorKind.ORDINARY else 0
    )
    run_after = (
        int(instance["foreground_run_cursor"]) if kind is BackgroundAuthorKind.ORDINARY else 0
    )
    return OrdinaryAuthorInput(
        foreground_messages=foreground_for_author(
            conn,
            profile_id,
            instance_id,
            kind,
            after=message_after,
            through=message_target,
        ),
        foreground_runs=foreground_runs_for_author(
            conn,
            profile_id,
            instance_id,
            kind,
            after=run_after,
            through=run_target,
        ),
        message_through=message_target if kind is BackgroundAuthorKind.ORDINARY else 0,
        run_through=run_target if kind is BackgroundAuthorKind.ORDINARY else 0,
        frame_interval=(
            _continuous_frame_interval(instance, kind, prompt_now)
            if kind is BackgroundAuthorKind.ORDINARY
            else None
        ),
    )


def keyframe_frame_interval(
    conn: sqlite3.Connection,
    instance: sqlite3.Row,
    kind: BackgroundAuthorKind,
    prompt_now: datetime,
) -> BackgroundFrameInterval | None:
    if kind is not BackgroundAuthorKind.KEYFRAME:
        return None
    latest = _latest_foreground_targets(
        conn,
        str(instance["profile_id"]),
        str(instance["instance_id"]),
        prompt_now=prompt_now,
    )
    current = (
        int(instance["foreground_message_cursor"]),
        int(instance["foreground_run_cursor"]),
    )
    if latest != current:
        raise BackgroundDraftStale(
            "Keyframe cannot pass foreground continuity not yet archived by Ordinary"
        )
    return _continuous_frame_interval(instance, kind, prompt_now)


def _opening_ordinary_input(
    *,
    kind: BackgroundAuthorKind,
    opening_life_stage: bool,
    opening_keyframe_completed: bool,
    opening_anchor: datetime | None,
    published_handoff: datetime | None,
    timezone_name: str,
) -> OrdinaryAuthorInput | None:
    if kind is not BackgroundAuthorKind.ORDINARY:
        return None
    if not opening_life_stage or not opening_keyframe_completed or opening_anchor is None:
        return None
    # Once the Keyframe has published, its actual end is the durable seam.  This
    # also keeps an initialization interrupted after the Keyframe publication
    # continuous instead of recomputing a different Ordinary start.
    cutoff = published_handoff or opening_handoff_at(
        opening_anchor,
        timezone_name=timezone_name,
    )
    return OrdinaryAuthorInput(
        foreground_messages=(),
        foreground_runs=(),
        message_through=0,
        run_through=0,
        frame_interval=BackgroundFrameInterval(start_at=cutoff, end_at=opening_anchor),
    )


def _opening_keyframe_interval(
    *,
    kind: BackgroundAuthorKind,
    opening_life_stage: bool,
    opening_keyframe_completed: bool,
    opening_anchor: datetime | None,
    timezone_name: str,
) -> BackgroundFrameInterval | None:
    if kind is not BackgroundAuthorKind.KEYFRAME:
        return None
    if not opening_life_stage or opening_keyframe_completed or opening_anchor is None:
        return None
    cutoff = opening_handoff_at(opening_anchor, timezone_name=timezone_name)
    # Only the handoff point is machine-owned. The prose before this seam has
    # no canonical machine-audited beginning.
    return BackgroundFrameInterval(
        start_at=cutoff - timedelta(microseconds=1),
        end_at=cutoff,
    )


def author_input_windows(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    instance: sqlite3.Row,
    kind: BackgroundAuthorKind,
    prompt_now: datetime,
    *,
    timezone_name: str,
) -> AuthorInputWindows:
    """Resolve continuous role-frame windows and the durable opening seam."""

    opening = conn.execute(
        """SELECT anchor_at, keyframe_completed
        FROM background_initialization_openings
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    opening_anchor = _parse(opening["anchor_at"]) if opening is not None else None
    opening_keyframe_completed = bool(opening is not None and int(opening["keyframe_completed"]))
    opening_life_stage = (
        str(instance["initialization_state"]) == "INITIALIZING"
        and str(instance["initialization_step"]) == "ORDINARY_CURRENT"
        and opening_anchor is not None
    )
    ordinary = _opening_ordinary_input(
        kind=kind,
        opening_life_stage=opening_life_stage,
        opening_keyframe_completed=opening_keyframe_completed,
        opening_anchor=opening_anchor,
        published_handoff=_parse(instance["simulated_through_at"]),
        timezone_name=timezone_name,
    )
    if ordinary is None:
        ordinary = ordinary_author_input(
            conn,
            profile_id,
            instance_id,
            instance,
            kind,
            prompt_now,
        )
    keyframe = _opening_keyframe_interval(
        kind=kind,
        opening_life_stage=opening_life_stage,
        opening_keyframe_completed=opening_keyframe_completed,
        opening_anchor=opening_anchor,
        timezone_name=timezone_name,
    )
    if keyframe is None:
        keyframe = keyframe_frame_interval(conn, instance, kind, prompt_now)
    return AuthorInputWindows(
        opening_anchor_at=opening_anchor,
        opening_keyframe_completed=opening_keyframe_completed,
        ordinary=ordinary,
        keyframe_frame_interval=keyframe,
    )


__all__ = [
    "AuthorInputWindows",
    "OrdinaryAuthorInput",
    "author_input_windows",
    "foreground_for_author",
    "foreground_runs_for_author",
    "keyframe_frame_interval",
    "ordinary_author_input",
]
