"""Publication ownership, structure and time-window checks."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from ....contracts.ai_task_payload import decode_task_payload
from ....storage.sqlite.codec import _parse
from ..domain import (
    BackgroundAuthorKind,
    BackgroundDisabled,
    BackgroundDraft,
    BackgroundDraftStale,
    BackgroundInputVersions,
    BackgroundTimelineEventDraft,
    BackgroundTimelineSource,
)
from ..opening_time import opening_handoff_at
from ._author_input_windows import _latest_foreground_targets
from .publication_models import ROLE_AUTHORS, PublishContext, aware


class BackgroundPublicationForegroundMixin:
    def _load_foreground_fence(
        self,
        conn: sqlite3.Connection,
        context: PublishContext,
        instance: sqlite3.Row,
        draft: BackgroundDraft,
        versions: BackgroundInputVersions,
    ) -> None:
        consumed_message = int(draft.consumed_foreground_through_message_id)
        consumed_run = int(draft.consumed_foreground_through_run_id)
        if context.kind not in ROLE_AUTHORS and (consumed_message or consumed_run):
            raise ValueError("only role authors may consume foreground continuity")

        if self._is_opening_ordinary(conn, context, instance):
            # Initialization owns the fixed [T-6h, T] seam and must not consume
            # the private trigger that is held for MainCore after READY.
            if consumed_message or consumed_run:
                raise BackgroundDraftStale("opening Ordinary cannot consume foreground continuity")
            return

        if context.kind is BackgroundAuthorKind.ORDINARY:
            frame_end_at = versions.frame_end_at
            if frame_end_at is None:
                raise BackgroundDraftStale("Ordinary has no owned frame end")
            expected = _latest_foreground_targets(
                conn,
                context.profile_id,
                context.instance_id,
                prompt_now=frame_end_at,
            )
            if (consumed_message, consumed_run) != expected:
                raise BackgroundDraftStale(
                    "Ordinary must consume exactly the foreground prefix it was shown"
                )
        elif consumed_message or consumed_run:
            raise ValueError("only Ordinary may advance foreground cursors")

    @staticmethod
    def _is_opening_ordinary(
        conn: sqlite3.Connection,
        context: PublishContext,
        instance: sqlite3.Row,
    ) -> bool:
        if (
            context.kind is not BackgroundAuthorKind.ORDINARY
            or str(instance["initialization_state"]) != "INITIALIZING"
            or str(instance["initialization_step"]) != "ORDINARY_CURRENT"
        ):
            return False
        opening = conn.execute(
            """SELECT keyframe_completed
            FROM background_initialization_openings
            WHERE profile_id = ? AND instance_id = ?""",
            (context.profile_id, context.instance_id),
        ).fetchone()
        return opening is not None and bool(int(opening["keyframe_completed"]))


_CURRENT_VIEW_ALLOWED_FIELDS = frozenset(
    {
        "narrative_time",
        "location",
        "doing",
        "body_state",
        "mood",
        "intention",
        "current_concern",
    }
)
_CURRENT_VIEW_REQUIRED_TEXT_FIELDS = ("narrative_time", "location")
_CURRENT_VIEW_OPTIONAL_TEXT_FIELDS = (
    "doing",
    "body_state",
    "mood",
    "intention",
    "current_concern",
)


@dataclass(frozen=True, slots=True)
class _PublishFenceRows:
    instance: sqlite3.Row
    state: sqlite3.Row
    core_state: sqlite3.Row
    task: sqlite3.Row


def _validate_current_view_required_text_fields(view: Mapping[str, object]) -> None:
    for name in _CURRENT_VIEW_REQUIRED_TEXT_FIELDS:
        value = view.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"current role {name} must be a non-empty string")


def _validate_current_view_optional_text_fields(view: Mapping[str, object]) -> None:
    for name in _CURRENT_VIEW_OPTIONAL_TEXT_FIELDS:
        if name not in view:
            continue
        value = view[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"current role {name} must be a non-empty string when present")


def _validate_opening_keyframe_window(
    conn: sqlite3.Connection,
    context: PublishContext,
    instance: sqlite3.Row,
    event: BackgroundTimelineEventDraft,
    *,
    expected_start: datetime,
    expected_end: datetime,
) -> bool:
    opening = conn.execute(
        """SELECT anchor_at, keyframe_completed
        FROM background_initialization_openings
        WHERE profile_id = ? AND instance_id = ?""",
        (context.profile_id, context.instance_id),
    ).fetchone()
    is_opening = (
        str(instance["initialization_state"]) == "INITIALIZING"
        and str(instance["initialization_step"]) == "ORDINARY_CURRENT"
        and opening is not None
        and not bool(opening["keyframe_completed"])
    )
    if not is_opening:
        return False
    assert opening is not None
    anchor = _parse(opening["anchor_at"])
    if anchor is None:
        raise BackgroundDraftStale("background opening anchor is invalid")
    runtime_settings = conn.execute(
        """SELECT timezone FROM profile_runtime_settings
        WHERE profile_id = ?""",
        (context.profile_id,),
    ).fetchone()
    timezone_name = str(runtime_settings["timezone"] or "") if runtime_settings is not None else ""
    cutoff = opening_handoff_at(anchor, timezone_name=timezone_name)
    start = aware(event.frame_start_at)
    end = aware(event.frame_end_at)
    if (
        start != aware(expected_start)
        or end != aware(expected_end)
        or start != cutoff - timedelta(microseconds=1)
        or end != cutoff
    ):
        raise BackgroundDraftStale(
            "opening Keyframe must land at 16:00 on the prior local calendar day"
        )
    return True


class BackgroundPublicationValidationMixin(BackgroundPublicationForegroundMixin):
    def _load_publish_fence(
        self,
        conn: sqlite3.Connection,
        context: PublishContext,
        versions: BackgroundInputVersions,
    ) -> tuple[sqlite3.Row, sqlite3.Row]:
        rows = self._read_publish_fence_rows(conn, context)
        self._validate_publish_ownership(rows, context)
        self._validate_version_fence(rows, versions)
        self._validate_task_input_snapshot(
            rows.task,
            context,
            rows.instance,
            versions,
        )
        return rows.instance, rows.state

    @staticmethod
    def _read_publish_fence_rows(
        conn: sqlite3.Connection,
        context: PublishContext,
    ) -> _PublishFenceRows:
        instance = conn.execute(
            """SELECT * FROM background_instances
            WHERE profile_id = ? AND instance_id = ?""",
            (context.profile_id, context.instance_id),
        ).fetchone()
        state = conn.execute(
            """SELECT * FROM background_author_states
            WHERE profile_id = ? AND instance_id = ? AND author_kind = ?""",
            (context.profile_id, context.instance_id, context.kind.value),
        ).fetchone()
        core_state = conn.execute(
            """SELECT activity_epoch FROM instance_core_state
            WHERE profile_id = ? AND instance_id = ?""",
            (context.profile_id, context.instance_id),
        ).fetchone()
        task = conn.execute(
            """SELECT task_type, status, generation, input_json
            FROM ai_tasks WHERE task_id = ?""",
            (context.task_id,),
        ).fetchone()
        if instance is None or state is None or core_state is None or task is None:
            raise BackgroundDraftStale("background publication fence no longer exists")
        return _PublishFenceRows(
            instance=instance,
            state=state,
            core_state=core_state,
            task=task,
        )

    @staticmethod
    def _validate_publish_ownership(
        rows: _PublishFenceRows,
        context: PublishContext,
    ) -> None:
        if not bool(rows.instance["enabled"]):
            raise BackgroundDisabled("background simulation was disabled")
        lease_until = _parse(rows.instance["foreground_lease_until"])
        if lease_until is not None and lease_until > context.published_at:
            raise BackgroundDraftStale("foreground turn owns the character timeline")
        if (
            int(rows.state["generation"]) != context.generation
            or int(rows.state["active_task_id"] or 0) != context.task_id
            or str(rows.state["status"]) != "RUNNING"
        ):
            raise BackgroundDraftStale("author task generation changed")
        if (
            str(rows.task["task_type"]) != "BACKGROUND_AUTHOR"
            or str(rows.task["status"]) != "RUNNING"
            or int(rows.task["generation"]) != context.generation
        ):
            raise BackgroundDraftStale("durable background task is no longer running")

    @staticmethod
    def _validate_version_fence(
        rows: _PublishFenceRows,
        versions: BackgroundInputVersions,
    ) -> None:
        current = {
            "config_version": int(rows.instance["config_version"]),
            "continuity_version": int(rows.instance["continuity_version"]),
            "activity_epoch": int(rows.core_state["activity_epoch"]),
            "timeline_version": int(rows.instance["timeline_version"]),
            "view_version": int(rows.instance["view_version"]),
            "publication_version": int(rows.instance["publication_version"]),
            "author_state_version": int(rows.state["state_version"]),
        }
        expected = versions.as_dict()
        for name, value in current.items():
            if int(expected[name]) != value:
                raise BackgroundDraftStale(f"background dependency changed: {name}")

    @staticmethod
    def _validate_task_input_snapshot(
        task: sqlite3.Row,
        context: PublishContext,
        instance: sqlite3.Row,
        versions: BackgroundInputVersions,
    ) -> None:
        try:
            payload = decode_task_payload("input", task["input_json"])
        except ValueError as exc:
            raise BackgroundDraftStale("durable task input is invalid") from exc
        identity = {
            "profile_id": context.profile_id,
            "instance_id": context.instance_id,
            "author_kind": context.kind.value,
            "generation": context.generation,
        }
        for name, expected in identity.items():
            if payload.get(name) != expected:
                raise BackgroundDraftStale(f"durable task identity changed: {name}")
        snapshot_values = {
            "initialization_step": str(instance["initialization_step"]),
            "foreground_message_cursor": int(instance["foreground_message_cursor"]),
            "foreground_run_cursor": int(instance["foreground_run_cursor"]),
            "simulated_through_at": (
                str(instance["simulated_through_at"])
                if instance["simulated_through_at"] is not None
                else None
            ),
        }
        for name, expected in snapshot_values.items():
            if payload.get(name) != expected:
                raise BackgroundDraftStale(f"durable task snapshot changed: {name}")
        for name, expected in versions.as_dict().items():
            if int(payload.get(name, -1)) != int(expected):
                raise BackgroundDraftStale(f"durable task snapshot changed: {name}")

    def _validate_draft(
        self,
        conn: sqlite3.Connection,
        context: PublishContext,
        instance: sqlite3.Row,
        draft: BackgroundDraft,
        versions: BackgroundInputVersions,
    ) -> None:
        if context.kind is not BackgroundAuthorKind.STORY_SOURCE and draft.story_sources:
            raise ValueError("only StorySource may publish hidden story modules")
        if context.kind is BackgroundAuthorKind.STORY_SOURCE and not draft.story_sources:
            raise ValueError("StorySource must publish at least one story module")
        if context.kind not in ROLE_AUTHORS:
            if draft.timeline_events or draft.current_view:
                raise ValueError("upper authors cannot write the role timeline")
            return
        self._validate_current_view_content(draft)
        if context.kind is BackgroundAuthorKind.ORDINARY:
            self._validate_ordinary_draft(
                conn,
                context,
                instance,
                draft,
                versions,
            )
            return
        self._validate_keyframe_draft(
            conn,
            context,
            instance,
            draft,
            versions,
        )

    @staticmethod
    def _validate_current_view_content(draft: BackgroundDraft) -> None:
        view = draft.current_view
        if not isinstance(view, Mapping):
            raise ValueError("current role view must be a mapping")
        unknown = set(view) - _CURRENT_VIEW_ALLOWED_FIELDS
        if unknown:
            raise ValueError(f"current role view contains unknown field: {sorted(unknown)[0]}")
        _validate_current_view_required_text_fields(view)
        _validate_current_view_optional_text_fields(view)

    def _validate_ordinary_draft(
        self,
        conn: sqlite3.Connection,
        context: PublishContext,
        instance: sqlite3.Row,
        draft: BackgroundDraft,
        versions: BackgroundInputVersions,
    ) -> None:
        if not draft.timeline_events:
            raise ValueError("Ordinary must write at least one past life event")
        if not draft.current_view:
            raise ValueError("Ordinary must publish the resulting current role state")
        ordinary_events = tuple(
            event
            for event in draft.timeline_events
            if event.source is BackgroundTimelineSource.ORDINARY
        )
        if len(ordinary_events) != 1:
            raise ValueError("Ordinary must publish exactly one Ordinary life event")
        for event in draft.timeline_events:
            if event.source is not BackgroundTimelineSource.ORDINARY:
                raise ValueError("Ordinary may only publish role-lived events")
            self._validate_event_window(context, event)
            self._validate_role_frame_window(event, versions)

    def _validate_keyframe_draft(
        self,
        conn: sqlite3.Connection,
        context: PublishContext,
        instance: sqlite3.Row,
        draft: BackgroundDraft,
        versions: BackgroundInputVersions,
    ) -> None:
        if draft.consumed_foreground_through_message_id or draft.consumed_foreground_through_run_id:
            raise ValueError("Keyframe cannot consume foreground continuity")
        self._validate_injected_keyframe_draft(
            conn,
            context,
            instance,
            draft,
            versions,
        )

    def _validate_injected_keyframe_draft(
        self,
        conn: sqlite3.Connection,
        context: PublishContext,
        instance: sqlite3.Row,
        draft: BackgroundDraft,
        versions: BackgroundInputVersions,
    ) -> None:
        if len(draft.timeline_events) != 1 or not draft.current_view:
            raise ValueError(
                "injected Keyframe must write exactly one event and resulting role state"
            )
        event = draft.timeline_events[0]
        if event.source is not BackgroundTimelineSource.KEYFRAME:
            raise ValueError("injected Keyframe events must be authored as KEYFRAME")
        self._validate_event_window(context, event)
        self._validate_injected_keyframe_window(
            conn,
            context,
            instance,
            event,
            versions,
        )

    @staticmethod
    def _validate_event_window(
        context: PublishContext,
        event: BackgroundTimelineEventDraft,
    ) -> None:
        start = aware(event.frame_start_at)
        end = aware(event.frame_end_at)
        if end < start or end > context.published_at:
            raise ValueError("timeline event falls outside settled background time")

    @staticmethod
    def _validate_role_frame_window(
        event: BackgroundTimelineEventDraft,
        versions: BackgroundInputVersions,
    ) -> None:
        expected_start = versions.frame_start_at
        expected_end = versions.frame_end_at
        if expected_start is None or expected_end is None:
            raise BackgroundDraftStale("role frame has no owned time interval")
        if aware(event.frame_start_at) != aware(expected_start) or aware(
            event.frame_end_at
        ) != aware(expected_end):
            raise BackgroundDraftStale("role frame must cover its exact owned time interval")

    @staticmethod
    def _validate_injected_keyframe_window(
        conn: sqlite3.Connection,
        context: PublishContext,
        instance: sqlite3.Row,
        event: BackgroundTimelineEventDraft,
        versions: BackgroundInputVersions,
    ) -> None:
        expected_start = versions.frame_start_at
        expected_end = versions.frame_end_at
        if expected_start is None or expected_end is None:
            raise BackgroundDraftStale("injected Keyframe has no owned frame interval")
        if _validate_opening_keyframe_window(
            conn,
            context,
            instance,
            event,
            expected_start=expected_start,
            expected_end=expected_end,
        ):
            return
        start = aware(event.frame_start_at)
        end = aware(event.frame_end_at)
        if start != aware(expected_start) or end != aware(expected_end):
            raise BackgroundDraftStale(
                "injected Keyframe must cover its exact owned frame interval"
            )
        view = conn.execute(
            """SELECT as_of FROM background_role_current_views
            WHERE profile_id = ? AND instance_id = ?""",
            (context.profile_id, context.instance_id),
        ).fetchone()
        view_as_of = _parse(view["as_of"]) if view is not None else None
        if view_as_of is not None and end <= view_as_of:
            raise BackgroundDraftStale(
                "injected Keyframe must advance beyond the current role state"
            )


__all__ = ["BackgroundPublicationValidationMixin"]
