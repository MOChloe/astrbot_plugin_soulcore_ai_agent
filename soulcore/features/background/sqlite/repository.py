"""SQLite boundary for the five independent background authors.

The repository deliberately reads foreground continuity from ``instance_messages``.
There is no second evidence ledger and no world-fact adjudication layer.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from ....storage.sqlite.background_projection import ensure_background_instance_sql
from ....storage.sqlite.codec import _dt, _now, _parse
from ....storage.sqlite.repository import SqliteRepository
from ..domain import (
    REFERENCE_AUTHORS,
    BackgroundAuthorInput,
    BackgroundAuthorKind,
    BackgroundDisabled,
    BackgroundDraftStale,
    BackgroundInitializationStep,
    BackgroundInputVersions,
    BackgroundStorySource,
    BackgroundTimelineEvent,
    RoleCurrentView,
)
from ..proactive_sources import (
    PredictableProactiveSource,
    scan_predictable_proactive_sources,
)
from ..prompt_projection import MainCoreBackgroundView
from ._author_input_windows import author_input_windows
from ._main_core_view import read_main_core_background_view_sql
from .admin_actions import BackgroundAdminActions
from .publication import BackgroundPublicationMixin
from .rows import (
    author_state_from_row,
    current_view_from_row,
    instance_config_from_row,
    json_list,
    timeline_event_from_row,
)
from .story_supply import ranked_story_supply

_RECENT_TIMELINE_LIMIT = 80
_STORY_SOURCE_LIMIT = 32
_ROLE_AUTHORS = frozenset(
    {
        BackgroundAuthorKind.KEYFRAME,
        BackgroundAuthorKind.ORDINARY,
    }
)
_FOREGROUND_INPUT_AUTHORS = _ROLE_AUTHORS & {
    BackgroundAuthorKind.ORDINARY,
}
_VIEW_AUTHORS = frozenset(
    {
        BackgroundAuthorKind.LIFE_DIRECTION,
        BackgroundAuthorKind.STORY_SOURCE,
        BackgroundAuthorKind.KEYFRAME,
        BackgroundAuthorKind.ORDINARY,
    }
)
_TIMELINE_AUTHORS = frozenset(
    {
        BackgroundAuthorKind.WORLD,
        BackgroundAuthorKind.LIFE_DIRECTION,
        BackgroundAuthorKind.STORY_SOURCE,
        BackgroundAuthorKind.KEYFRAME,
        BackgroundAuthorKind.ORDINARY,
    }
)


def _aware(value: datetime) -> datetime:
    return (value if value.tzinfo is not None else value.replace(tzinfo=UTC)).astimezone(UTC)


class SqliteBackgroundRepository(
    BackgroundAdminActions,
    BackgroundPublicationMixin,
    SqliteRepository,
):
    async def ensure_instance(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            ensure_background_instance_sql(conn, profile_id, instance_id, now)
            row = conn.execute(
                """SELECT * FROM background_instances
                WHERE profile_id = ? AND instance_id = ?""",
                (profile_id, instance_id),
            ).fetchone()
            if row is None:
                raise KeyError((profile_id, instance_id))
            return row

        return dict(await self.uow.run(operation))

    async def ensure_all_instances(self) -> int:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            rows = conn.execute(
                "SELECT profile_id, instance_id FROM character_instances"
            ).fetchall()
            for row in rows:
                ensure_background_instance_sql(
                    conn,
                    str(row["profile_id"]),
                    str(row["instance_id"]),
                    now,
                )
            return len(rows)

        return int(await self.uow.run(operation))

    async def list_predictable_proactive_sources(
        self,
        *,
        now: datetime,
        limit: int = 100,
    ) -> tuple[PredictableProactiveSource, ...]:
        return await scan_predictable_proactive_sources(self.db, now=now, limit=limit)

    async def start_author_task(
        self,
        profile_id: str,
        instance_id: str,
        author_kind: BackgroundAuthorKind | str,
        *,
        generation: int,
        task_id: int,
    ) -> bool:
        """Bind a claimed durable task to its still-current author generation."""

        kind = BackgroundAuthorKind(author_kind).value
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            instance = conn.execute(
                """SELECT profile.background_life_enabled, instance.foreground_lease_until
                FROM background_instances instance
                JOIN role_profiles profile ON profile.profile_id = instance.profile_id
                WHERE instance.profile_id = ? AND instance.instance_id = ?""",
                (profile_id, instance_id),
            ).fetchone()
            if instance is None or not bool(instance["background_life_enabled"]):
                return 0
            lease_until = _parse(instance["foreground_lease_until"])
            if lease_until is not None and lease_until > _now():
                return 0
            return int(
                conn.execute(
                    """UPDATE background_author_states
                    SET status = 'RUNNING', last_started_at = ?, updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND author_kind = ?
                      AND generation = ? AND active_task_id = ?
                      AND status IN ('ENQUEUED', 'FAILED', 'RUNNING')""",
                    (
                        now,
                        now,
                        profile_id,
                        instance_id,
                        kind,
                        int(generation),
                        int(task_id),
                    ),
                ).rowcount
            )

        return bool(await self.uow.run(operation))

    async def mark_author_failure(
        self,
        profile_id: str,
        instance_id: str,
        author_kind: BackgroundAuthorKind | str,
        *,
        generation: int,
        task_id: int,
        error: str,
    ) -> bool:
        """Record one failure; task settlement later releases the durable slot."""

        kind = BackgroundAuthorKind(author_kind).value
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            return int(
                conn.execute(
                    """UPDATE background_author_states
                    SET status = 'FAILED', failure_count = failure_count + 1,
                        last_error = ?, updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND author_kind = ?
                      AND generation = ? AND active_task_id = ?
                      AND status <> 'FAILED'""",
                    (
                        str(error)[:1000],
                        now,
                        profile_id,
                        instance_id,
                        kind,
                        int(generation),
                        int(task_id),
                    ),
                ).rowcount
            )

        return bool(await self.uow.run(operation))

    async def load_author_input(
        self,
        profile_id: str,
        instance_id: str,
        author_kind: BackgroundAuthorKind | str,
        *,
        frame_end_at: datetime | None = None,
    ) -> BackgroundAuthorInput:
        kind = BackgroundAuthorKind(author_kind)
        await self.ensure_instance(profile_id, instance_id)
        observed_now = _aware(_now())
        prompt_now = _aware(frame_end_at) if frame_end_at is not None else observed_now
        if prompt_now > observed_now:
            raise ValueError("background frame end cannot be in the future")
        return await self.db.call(
            lambda conn: self._load_author_input_sql(
                conn,
                profile_id,
                instance_id,
                kind,
                prompt_now,
            )
        )

    @classmethod
    def _load_author_input_sql(
        cls,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        kind: BackgroundAuthorKind,
        prompt_now: datetime,
    ) -> BackgroundAuthorInput:
        instance = conn.execute(
            """SELECT instance.*, profile.background_life_enabled AS role_background_enabled
            FROM background_instances instance
            JOIN role_profiles profile ON profile.profile_id = instance.profile_id
            WHERE instance.profile_id = ? AND instance.instance_id = ?""",
            (profile_id, instance_id),
        ).fetchone()
        state = conn.execute(
            """SELECT * FROM background_author_states
            WHERE profile_id = ? AND instance_id = ? AND author_kind = ?""",
            (profile_id, instance_id, kind.value),
        ).fetchone()
        core_state = conn.execute(
            """SELECT activity_epoch FROM instance_core_state
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        ).fetchone()
        runtime_settings = conn.execute(
            """SELECT timezone FROM profile_runtime_settings
            WHERE profile_id = ?""",
            (profile_id,),
        ).fetchone()
        if instance is None or state is None or core_state is None:
            raise KeyError((profile_id, instance_id, kind.value))
        if not bool(instance["role_background_enabled"]):
            raise BackgroundDisabled("background simulation is disabled")
        lease_until = _parse(instance["foreground_lease_until"])
        if lease_until is not None and lease_until > prompt_now:
            raise BackgroundDraftStale("foreground turn owns the character timeline")
        reference_rows = cls._reference_state_rows(
            conn,
            profile_id,
            instance_id,
            kind,
        )
        reference_states = tuple(author_state_from_row(row) for row in reference_rows)
        story_sources = cls._story_sources_for_author(
            conn,
            profile_id,
            instance_id,
            kind,
        )
        timeline = cls._timeline_for_author(
            conn,
            profile_id,
            instance_id,
            kind,
        )
        timezone_name = (
            str(runtime_settings["timezone"] or "") if runtime_settings is not None else ""
        )
        windows = author_input_windows(
            conn,
            profile_id,
            instance_id,
            instance,
            kind,
            prompt_now,
            timezone_name=timezone_name,
        )
        ordinary_input = windows.ordinary
        current_view = (
            cls._load_current_view_sql(conn, profile_id, instance_id)
            if kind in _VIEW_AUTHORS
            else RoleCurrentView()
        )
        keyframe_interval = windows.keyframe_frame_interval
        frame_interval = (
            ordinary_input.frame_interval
            if kind is BackgroundAuthorKind.ORDINARY
            else keyframe_interval
        )
        seed, lore, boundaries = cls._creative_foundation(conn, profile_id)
        versions = BackgroundInputVersions(
            config_version=int(instance["config_version"]),
            continuity_version=int(instance["continuity_version"]),
            activity_epoch=int(core_state["activity_epoch"]),
            timeline_version=int(instance["timeline_version"]),
            view_version=int(instance["view_version"]),
            publication_version=int(instance["publication_version"]),
            author_state_version=int(state["state_version"]),
            frame_start_at=(frame_interval.start_at if frame_interval is not None else None),
            frame_end_at=(frame_interval.end_at if frame_interval is not None else None),
        )
        return BackgroundAuthorInput(
            profile_id=profile_id,
            instance_id=instance_id,
            author_kind=kind,
            generation=int(state["generation"]),
            initialization_state=str(instance["initialization_state"]),
            initialization_step=BackgroundInitializationStep(str(instance["initialization_step"])),
            config=instance_config_from_row(instance),
            author_state=author_state_from_row(state),
            reference_states=reference_states,
            story_sources=story_sources,
            recent_timeline=timeline,
            foreground_messages=ordinary_input.foreground_messages,
            foreground_runs=ordinary_input.foreground_runs,
            current_view=current_view,
            ordinary_frame_interval=ordinary_input.frame_interval,
            keyframe_frame_interval=keyframe_interval,
            seed=seed,
            lore=lore,
            boundaries=boundaries,
            versions=versions,
            prompt_now=prompt_now,
            timezone_name=timezone_name,
            initialization_anchor_at=windows.opening_anchor_at,
            opening_keyframe_completed=windows.opening_keyframe_completed,
            foreground_message_through=ordinary_input.message_through,
            foreground_run_through=ordinary_input.run_through,
        )

    @staticmethod
    def _reference_state_rows(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        kind: BackgroundAuthorKind,
    ) -> tuple[sqlite3.Row, ...]:
        references = REFERENCE_AUTHORS[kind]
        if not references:
            return ()
        placeholders = ",".join("?" for _ in references)
        rows = conn.execute(
            f"""SELECT * FROM background_author_states
            WHERE profile_id = ? AND instance_id = ?
              AND author_kind IN ({placeholders})""",
            (profile_id, instance_id, *(item.value for item in references)),
        ).fetchall()
        by_kind = {str(row["author_kind"]): row for row in rows}
        return tuple(by_kind[item.value] for item in references if item.value in by_kind)

    @classmethod
    def _story_sources_for_author(
        cls,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        kind: BackgroundAuthorKind,
    ) -> tuple[BackgroundStorySource, ...]:
        if kind not in {
            BackgroundAuthorKind.STORY_SOURCE,
            BackgroundAuthorKind.KEYFRAME,
            BackgroundAuthorKind.ORDINARY,
        }:
            return ()
        return tuple(
            candidate.story
            for candidate in ranked_story_supply(conn, profile_id, instance_id)[
                :_STORY_SOURCE_LIMIT
            ]
        )

    @classmethod
    def _timeline_for_author(
        cls,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        kind: BackgroundAuthorKind,
    ) -> tuple[BackgroundTimelineEvent, ...]:
        if kind not in _TIMELINE_AUTHORS:
            return ()
        rows = conn.execute(
            """SELECT * FROM background_role_timeline_events
            WHERE profile_id = ? AND instance_id = ?
            ORDER BY frame_end_at DESC, event_id DESC
            LIMIT ?""",
            (profile_id, instance_id, _RECENT_TIMELINE_LIMIT),
        ).fetchall()
        return tuple(timeline_event_from_row(row) for row in reversed(rows))

    @staticmethod
    def _creative_foundation(
        conn: sqlite3.Connection,
        profile_id: str,
    ) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
        seed_row = conn.execute(
            "SELECT * FROM world_definitions WHERE profile_id = ?",
            (profile_id,),
        ).fetchone()
        lore_rows = conn.execute(
            """SELECT lore_id, revision, title, aliases_json, tags_json,
                content, importance, updated_at
            FROM world_lore_entries WHERE profile_id = ?
            ORDER BY importance DESC, updated_at DESC LIMIT 80""",
            (profile_id,),
        ).fetchall()
        boundary_rows = conn.execute(
            """SELECT boundary_id, revision, severity, category, rule_text,
                positive_space
            FROM creative_boundaries
            WHERE profile_id = ? AND enabled = 1
            ORDER BY CASE severity WHEN 'HARD' THEN 0 ELSE 1 END, boundary_id""",
            (profile_id,),
        ).fetchall()
        lore: list[dict[str, Any]] = []
        for row in lore_rows:
            item = dict(row)
            item["aliases"] = json_list(item.pop("aliases_json"))
            item["tags"] = json_list(item.pop("tags_json"))
            lore.append(item)
        return (
            dict(seed_row) if seed_row is not None else {},
            tuple(lore),
            tuple(dict(row) for row in boundary_rows),
        )

    @classmethod
    def _load_current_view_sql(
        cls,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
    ) -> RoleCurrentView:
        row = conn.execute(
            """SELECT * FROM background_role_current_views
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        ).fetchone()
        return current_view_from_row(row)

    async def read_role_current_view(
        self,
        profile_id: str,
        instance_id: str,
    ) -> RoleCurrentView | None:
        """Return the role's lived state even while background authoring is disabled."""

        return await self.uow.run(
            lambda conn: self._read_role_current_view_sql(
                conn,
                profile_id,
                instance_id,
            )
        )

    async def read_main_core_background_view(
        self,
        profile_id: str,
        instance_id: str,
    ) -> MainCoreBackgroundView | None:
        """Freeze all MainCore-visible background prose in one read transaction."""

        return await self.uow.run(
            lambda conn: read_main_core_background_view_sql(
                conn,
                profile_id,
                instance_id,
            )
        )

    async def settle_successful_story_exposure(
        self,
        profile_id: str,
        instance_id: str,
        run_id: int,
        *,
        invocation_id: str,
        round_no: int,
        story_refs: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Count each story at most once after a successful model call in one run."""

        normalized_refs = tuple(
            dict.fromkeys(str(ref or "").strip() for ref in story_refs if str(ref or "").strip())
        )
        if not normalized_refs:
            return ()
        normalized_invocation_id = str(invocation_id or "").strip()
        normalized_round_no = int(round_no)
        if not normalized_invocation_id:
            raise ValueError("story exposure invocation_id must not be empty")
        if normalized_round_no < 1:
            raise ValueError("story exposure round_no must be positive")
        exposed_at = _dt(_now())

        def operation(conn: sqlite3.Connection) -> tuple[str, ...]:
            run = conn.execute(
                """SELECT profile_id, instance_id FROM instance_core_runs
                WHERE run_id = ?""",
                (int(run_id),),
            ).fetchone()
            if run is None:
                raise KeyError(("instance_core_run", int(run_id)))
            if str(run["profile_id"]) != profile_id or str(run["instance_id"]) != instance_id:
                raise ValueError("MainCore run does not belong to the story exposure scope")

            counted: list[str] = []
            for story_ref in normalized_refs:
                story = conn.execute(
                    """SELECT story_source_id FROM background_story_sources
                    WHERE profile_id = ? AND instance_id = ? AND public_ref = ?""",
                    (profile_id, instance_id, story_ref),
                ).fetchone()
                if story is None:
                    continue
                story_source_id = int(story["story_source_id"])
                inserted = conn.execute(
                    """INSERT INTO background_story_run_exposures(
                        run_id, story_source_id, first_invocation_id,
                        first_round_no, exposed_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, story_source_id) DO NOTHING""",
                    (
                        int(run_id),
                        story_source_id,
                        normalized_invocation_id,
                        normalized_round_no,
                        exposed_at,
                    ),
                )
                if inserted.rowcount != 1:
                    continue
                updated = conn.execute(
                    """UPDATE background_story_sources
                    SET shown_count = shown_count + 1,
                        last_shown_at = ?, updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND story_source_id = ?""",
                    (
                        exposed_at,
                        exposed_at,
                        profile_id,
                        instance_id,
                        story_source_id,
                    ),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("story exposure target disappeared inside its transaction")
                counted.append(story_ref)
            return tuple(counted)

        return tuple(await self.uow.run(operation))

    @classmethod
    def _read_role_current_view_sql(
        cls,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
    ) -> RoleCurrentView | None:
        row = conn.execute(
            """SELECT * FROM background_role_current_views
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        ).fetchone()
        if row is None:
            return None
        event_rows = conn.execute(
            """SELECT * FROM background_role_timeline_events
            WHERE profile_id = ? AND instance_id = ?
            ORDER BY frame_end_at DESC, event_id DESC LIMIT 5""",
            (profile_id, instance_id),
        ).fetchall()
        return replace(
            current_view_from_row(row),
            recent_experiences=tuple(
                timeline_event_from_row(item) for item in reversed(event_rows)
            ),
        )

    async def read_recent_role_timeline(
        self,
        profile_id: str,
        instance_id: str,
        *,
        limit: int = 24,
    ) -> tuple[BackgroundTimelineEvent, ...]:
        bounded = max(1, min(int(limit), 100))
        rows = await self.db.fetch_all(
            """SELECT * FROM background_role_timeline_events
            WHERE profile_id = ? AND instance_id = ?
            ORDER BY frame_end_at DESC, event_id DESC LIMIT ?""",
            (profile_id, instance_id, bounded),
        )
        return tuple(timeline_event_from_row(row) for row in reversed(rows))

    async def read_role_continuity(
        self,
        profile_id: str,
        instance_id: str,
        *,
        timeline_limit: int = 24,
    ) -> dict[str, Any]:
        """Return current state and lived events for non-MainCore continuity readers."""

        bounded = max(1, min(int(timeline_limit), 100))

        def operation(conn: sqlite3.Connection) -> dict[str, Any]:
            view = self._read_role_current_view_sql(
                conn,
                profile_id,
                instance_id,
            )
            rows = conn.execute(
                """SELECT * FROM background_role_timeline_events
                WHERE profile_id = ? AND instance_id = ?
                ORDER BY frame_end_at DESC, event_id DESC LIMIT ?""",
                (profile_id, instance_id, bounded),
            ).fetchall()
            return {
                "current_view": view,
                "timeline": tuple(timeline_event_from_row(row) for row in reversed(rows)),
            }

        return dict(await self.uow.run(operation))

    async def query_role_visible_world_info(
        self,
        profile_id: str,
        instance_id: str,
        query: str,
        *,
        limit: int = 200,
    ) -> list[Mapping[str, str]]:
        """Search current role state, lived events, and their leftover material.

        This is the WorldInfo command boundary.  Story modules plus world and
        life-direction author prose are intentionally unreachable here.
        """

        bounded = max(1, min(int(limit), 200))
        needle = " ".join(str(query or "").split()).casefold()

        def operation(
            conn: sqlite3.Connection,
        ) -> tuple[RoleCurrentView | None, tuple[BackgroundTimelineEvent, ...]]:
            view = self._read_role_current_view_sql(
                conn,
                profile_id,
                instance_id,
            )
            rows = conn.execute(
                """SELECT * FROM background_role_timeline_events
                WHERE profile_id = ? AND instance_id = ?
                ORDER BY frame_end_at DESC, event_id DESC LIMIT ?""",
                (profile_id, instance_id, bounded),
            ).fetchall()
            return view, tuple(timeline_event_from_row(row) for row in reversed(rows))

        view, events = await self.uow.run(operation)
        result: list[Mapping[str, str]] = []
        if view is not None:
            parts = (
                view.narrative_time,
                view.location,
                view.doing,
                view.body_state,
                view.mood,
                view.intention,
                view.current_concern,
            )
            content = "；".join(part.strip() for part in parts if part.strip())
            if content and (not needle or needle in content.casefold()):
                result.append(
                    {
                        "stable_key": f"view:{view.revision}",
                        "kind": "current_state",
                        "content": content,
                        "status": "ACTIVE",
                        "provenance": "角色当前生活",
                    }
                )
        for event in reversed(events):
            content = "\n".join(
                part for part in (event.content.strip(), event.leftover_text.strip()) if part
            )
            if not content:
                continue
            if needle and needle not in content.casefold():
                continue
            result.append(
                {
                    "stable_key": event.public_ref,
                    "kind": "role_event",
                    "content": content,
                    "status": "ACTIVE",
                    "provenance": "角色已发生经历",
                }
            )
            if len(result) >= bounded:
                break
        return result[:bounded]


__all__ = ["SqliteBackgroundRepository"]
