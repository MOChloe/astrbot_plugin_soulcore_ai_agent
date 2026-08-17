"""Atomic publication orchestration for the role-centric background runtime."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from ....shared.time_display import model_datetime
from ....storage.sqlite.codec import _dt, _dump, _load, _now, _parse
from ..domain import (
    BackgroundAuthorKind,
    BackgroundDraft,
    BackgroundDraftStale,
    BackgroundInitializationStep,
    BackgroundInputVersions,
    BackgroundPublicationResult,
)
from .publication_models import (
    ROLE_AUTHORS,
    PublicationOutcome,
    PublicationProgress,
    PublishContext,
    PublishedContent,
    aware,
)
from .publication_validation import BackgroundPublicationValidationMixin
from .publication_writes import BackgroundPublicationWritesMixin

_OUTCOME_FIELDS = {
    "schema_version",
    "task_id",
    "initialization_step",
    "timeline_event_ids",
    "story_source_refs",
    "foreground_message_cursor",
    "foreground_run_cursor",
    "next_due_at",
    "hard_due_at",
}


@dataclass(frozen=True, slots=True)
class _ReplayValues:
    initialization_step: str
    event_ids: tuple[int, ...]
    story_refs: tuple[str, ...]
    message_cursor: int
    run_cursor: int
    next_due: str
    hard_due: str


def _required_int(snapshot: Mapping[str, object], name: str) -> int:
    value = snapshot[name]
    if type(value) is not int:
        raise ValueError(f"outcome {name} must be an integer")
    return value


def _required_str(snapshot: Mapping[str, object], name: str) -> str:
    value = snapshot[name]
    if not isinstance(value, str):
        raise ValueError(f"outcome {name} must be a string")
    return value


def _required_int_list(snapshot: Mapping[str, object], name: str) -> tuple[int, ...]:
    values = snapshot[name]
    if not isinstance(values, list):
        raise ValueError(f"outcome {name} must be a list")
    if any(type(value) is not int for value in values):
        raise ValueError(f"outcome {name} values must be integers")
    return tuple(values)


def _required_str_list(snapshot: Mapping[str, object], name: str) -> tuple[str, ...]:
    values = snapshot[name]
    if not isinstance(values, list):
        raise ValueError(f"outcome {name} must be a list")
    if any(not isinstance(value, str) for value in values):
        raise ValueError(f"outcome {name} values must be strings")
    return tuple(values)


class BackgroundPublicationReplayMixin:
    @classmethod
    def _existing_outcome(
        cls,
        conn: sqlite3.Connection,
        context: PublishContext,
    ) -> PublicationOutcome | None:
        publication = conn.execute(
            """SELECT * FROM background_author_publications
            WHERE profile_id = ? AND instance_id = ? AND author_kind = ?
              AND generation = ?""",
            (
                context.profile_id,
                context.instance_id,
                context.kind.value,
                context.generation,
            ),
        ).fetchone()
        if publication is None:
            return None
        try:
            values = cls._decode_outcome_snapshot(publication, context)
            cls._validate_outcome_relations(
                conn,
                context,
                publication_id=int(publication["publication_id"]),
                event_ids=values.event_ids,
                story_refs=values.story_refs,
            )
        except BackgroundDraftStale:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "background publication outcome snapshot is missing or corrupt"
            ) from exc
        return PublicationOutcome(
            publication=publication,
            initialization_step=values.initialization_step,
            timeline_event_ids=values.event_ids,
            story_source_refs=values.story_refs,
            foreground_message_cursor=values.message_cursor,
            foreground_run_cursor=values.run_cursor,
            next_due_at=values.next_due,
            hard_due_at=values.hard_due,
        )

    @staticmethod
    def _decode_outcome_snapshot(
        publication: sqlite3.Row,
        context: PublishContext,
    ) -> _ReplayValues:
        snapshot = BackgroundPublicationReplayMixin._snapshot_mapping(publication)
        task_id = _required_int(snapshot, "task_id")
        if task_id != context.task_id:
            raise BackgroundDraftStale("publication generation belongs to another durable task")
        values = _ReplayValues(
            initialization_step=BackgroundInitializationStep(
                _required_str(snapshot, "initialization_step")
            ).value,
            event_ids=_required_int_list(snapshot, "timeline_event_ids"),
            story_refs=_required_str_list(snapshot, "story_source_refs"),
            message_cursor=_required_int(snapshot, "foreground_message_cursor"),
            run_cursor=_required_int(snapshot, "foreground_run_cursor"),
            next_due=_required_str(snapshot, "next_due_at"),
            hard_due=_required_str(snapshot, "hard_due_at"),
        )
        BackgroundPublicationReplayMixin._validate_replay_values(values)
        return values

    @staticmethod
    def _snapshot_mapping(publication: sqlite3.Row) -> Mapping[str, object]:
        snapshot = _load(publication["outcome_json"])
        if not isinstance(snapshot, Mapping):
            raise ValueError("unsupported outcome snapshot")
        if type(snapshot.get("schema_version")) is not int:
            raise ValueError("unsupported outcome snapshot")
        if snapshot.get("schema_version") != 1:
            raise ValueError("unsupported outcome snapshot")
        if set(snapshot) != _OUTCOME_FIELDS:
            raise ValueError("outcome snapshot fields are invalid")
        return snapshot

    @staticmethod
    def _validate_replay_values(values: _ReplayValues) -> None:
        if any(value < 1 for value in values.event_ids):
            raise ValueError("outcome values are invalid")
        if len(set(values.event_ids)) != len(values.event_ids):
            raise ValueError("outcome values are invalid")
        if any(not value for value in values.story_refs):
            raise ValueError("outcome values are invalid")
        if values.message_cursor < 0 or values.run_cursor < 0:
            raise ValueError("outcome values are invalid")
        if _parse(values.next_due) is None or _parse(values.hard_due) is None:
            raise ValueError("outcome values are invalid")

    @staticmethod
    def _validate_outcome_relations(
        conn: sqlite3.Connection,
        context: PublishContext,
        *,
        publication_id: int,
        event_ids: tuple[int, ...],
        story_refs: tuple[str, ...],
    ) -> None:
        BackgroundPublicationReplayMixin._validate_direct_event_relations(
            conn,
            context,
            publication_id=publication_id,
            event_ids=event_ids,
        )
        story_rows = conn.execute(
            """SELECT public_ref
            FROM background_story_sources
            WHERE profile_id = ? AND instance_id = ?
              AND source_publication_id = ?
            ORDER BY story_source_id""",
            (
                context.profile_id,
                context.instance_id,
                publication_id,
            ),
        ).fetchall()
        actual_story_refs = tuple(str(row["public_ref"]) for row in story_rows)
        actual_story_ref_set = frozenset(actual_story_refs)
        # Story material has a bounded retention window.  A later publication
        # may deterministically delete old modules, so replay accepts missing
        # historical rows but never accepts an unexpected or reordered row.
        expected_retained_refs = tuple(item for item in story_refs if item in actual_story_ref_set)
        if actual_story_refs != expected_retained_refs:
            raise ValueError("publication story-source outcome is inconsistent")

    @staticmethod
    def _validate_direct_event_relations(
        conn: sqlite3.Connection,
        context: PublishContext,
        *,
        publication_id: int,
        event_ids: tuple[int, ...],
    ) -> None:
        rows = conn.execute(
            """SELECT event_id
            FROM background_role_timeline_events
            WHERE profile_id = ? AND instance_id = ?
              AND source_publication_id = ?
            ORDER BY event_id""",
            (
                context.profile_id,
                context.instance_id,
                publication_id,
            ),
        ).fetchall()
        if tuple(int(row["event_id"]) for row in rows) != event_ids:
            raise ValueError("publication timeline outcome is inconsistent")

    @staticmethod
    def _persist_outcome_snapshot(
        conn: sqlite3.Connection,
        context: PublishContext,
        *,
        publication_id: int,
        initialization_step: str,
        event_ids: tuple[int, ...],
        story_refs: tuple[str, ...],
        message_cursor: int,
        run_cursor: int,
    ) -> PublicationOutcome:
        state_after = conn.execute(
            """SELECT next_due_at, hard_due_at
            FROM background_author_states
            WHERE profile_id = ? AND instance_id = ? AND author_kind = ?""",
            (context.profile_id, context.instance_id, context.kind.value),
        ).fetchone()
        assert state_after is not None
        next_due = str(state_after["next_due_at"] or context.next_due)
        hard_due = str(state_after["hard_due_at"] or context.hard_due)
        outcome_snapshot = {
            "schema_version": 1,
            "task_id": context.task_id,
            "initialization_step": initialization_step,
            "timeline_event_ids": list(event_ids),
            "story_source_refs": list(story_refs),
            "foreground_message_cursor": message_cursor,
            "foreground_run_cursor": run_cursor,
            "next_due_at": next_due,
            "hard_due_at": hard_due,
        }
        updated = conn.execute(
            """UPDATE background_author_publications
            SET outcome_json = ?
            WHERE publication_id = ? AND outcome_json = '{}'""",
            (_dump(outcome_snapshot), publication_id),
        )
        if updated.rowcount != 1:
            raise RuntimeError("background publication outcome snapshot was not persisted")
        publication = conn.execute(
            """SELECT * FROM background_author_publications
            WHERE publication_id = ?""",
            (publication_id,),
        ).fetchone()
        assert publication is not None
        return PublicationOutcome(
            publication=publication,
            initialization_step=initialization_step,
            timeline_event_ids=event_ids,
            story_source_refs=story_refs,
            foreground_message_cursor=message_cursor,
            foreground_run_cursor=run_cursor,
            next_due_at=next_due,
            hard_due_at=hard_due,
        )


def _load_current_view_publication_rows(
    conn: sqlite3.Connection,
    context: PublishContext,
    event_ids: tuple[int, ...],
) -> tuple[sqlite3.Row, sqlite3.Row]:
    previous = conn.execute(
        """SELECT * FROM background_role_current_views
        WHERE profile_id = ? AND instance_id = ?""",
        (context.profile_id, context.instance_id),
    ).fetchone()
    if previous is None:
        raise BackgroundDraftStale("current role view no longer exists")
    placeholders = ",".join("?" for _ in event_ids)
    last_event = conn.execute(
        f"""SELECT event_id, source, frame_end_at
        FROM background_role_timeline_events
        WHERE profile_id = ? AND instance_id = ?
          AND event_id IN ({placeholders})
        ORDER BY frame_end_at DESC, event_id DESC
        LIMIT 1""",
        (
            context.profile_id,
            context.instance_id,
            *event_ids,
        ),
    ).fetchone()
    if last_event is None:
        raise BackgroundDraftStale("current role event no longer exists")
    return previous, last_event


class BackgroundPublicationCurrentViewMixin:
    @staticmethod
    def _validate_current_view_time(
        previous: sqlite3.Row,
        last_event: sqlite3.Row,
    ) -> None:
        # The seeded INITIALIZATION row is only an empty storage placeholder;
        # its ``as_of`` timestamp is not part of the role's narrative timeline.
        # Opening keyframes may deliberately establish the first real view at
        # the prior-local-day handoff before the initialization anchor.
        if str(previous["source"]) == "INITIALIZATION":
            return
        previous_as_of = _parse(previous["as_of"])
        next_as_of = _parse(last_event["frame_end_at"])
        if previous_as_of is not None and next_as_of is not None and next_as_of < previous_as_of:
            raise BackgroundDraftStale("current role view cannot move backwards in time")

    @staticmethod
    def _publish_current_view(
        conn: sqlite3.Connection,
        context: PublishContext,
        publication_id: int,
        draft: BackgroundDraft,
        event_ids: tuple[int, ...],
    ) -> bool:
        if not draft.current_view:
            return False
        if context.kind not in ROLE_AUTHORS or not event_ids:
            raise ValueError("current role state requires a newly published role event")
        previous, last_event = _load_current_view_publication_rows(
            conn,
            context,
            event_ids,
        )
        BackgroundPublicationCurrentViewMixin._validate_current_view_time(previous, last_event)
        view = dict(draft.current_view)
        runtime_settings = conn.execute(
            """SELECT timezone FROM profile_runtime_settings
            WHERE profile_id = ?""",
            (context.profile_id,),
        ).fetchone()
        timezone_name = (
            str(runtime_settings["timezone"] or "") if runtime_settings is not None else ""
        )
        narrative_time = model_datetime(
            str(last_event["frame_end_at"]),
            timezone_name=timezone_name,
        )
        conn.execute(
            """UPDATE background_role_current_views
            SET revision = revision + 1, narrative_time = ?, location = ?, doing = ?,
                body_state = ?, mood = ?, intention = ?, current_concern = ?, as_of = ?,
                source = ?, source_event_id = ?, source_publication_id = ?,
                updated_at = ?
            WHERE profile_id = ? AND instance_id = ?""",
            (
                narrative_time,
                str(view["location"]),
                str(view.get("doing") or ""),
                str(view.get("body_state") or ""),
                str(view.get("mood") or ""),
                str(view.get("intention") or ""),
                str(view.get("current_concern") or ""),
                str(last_event["frame_end_at"]),
                str(last_event["source"]),
                int(last_event["event_id"]),
                publication_id,
                context.now,
                context.profile_id,
                context.instance_id,
            ),
        )
        return True


class BackgroundPublicationMixin(
    BackgroundPublicationReplayMixin,
    BackgroundPublicationValidationMixin,
    BackgroundPublicationCurrentViewMixin,
    BackgroundPublicationWritesMixin,
):
    async def publish(
        self,
        profile_id: str,
        instance_id: str,
        author_kind: BackgroundAuthorKind | str,
        *,
        generation: int,
        task_id: int,
        draft: BackgroundDraft,
        versions: BackgroundInputVersions,
        next_due_at: datetime,
        hard_due_at: datetime,
        preserve_schedule: bool = False,
    ) -> BackgroundPublicationResult:
        published_at = aware(_now())
        context = PublishContext(
            profile_id=profile_id,
            instance_id=instance_id,
            kind=BackgroundAuthorKind(author_kind),
            generation=int(generation),
            task_id=int(task_id),
            published_at=published_at,
            now=_dt(published_at),
            next_due=_dt(aware(next_due_at)),
            hard_due=_dt(aware(hard_due_at)),
            preserve_schedule=bool(preserve_schedule),
        )
        outcome = await self.uow.run(
            lambda conn: self._publish_transaction(conn, context, draft, versions)
        )
        return BackgroundPublicationResult(
            publication_id=int(outcome.publication["publication_id"]),
            public_ref=str(outcome.publication["public_ref"]),
            author_kind=context.kind,
            generation=int(outcome.publication["generation"]),
            next_due_at=_parse(outcome.next_due_at) or published_at,
            hard_due_at=_parse(outcome.hard_due_at) or published_at,
            initialization_step=BackgroundInitializationStep(outcome.initialization_step),
            timeline_event_ids=outcome.timeline_event_ids,
            story_source_refs=outcome.story_source_refs,
            foreground_message_cursor=outcome.foreground_message_cursor,
            foreground_run_cursor=outcome.foreground_run_cursor,
        )

    def _publish_transaction(
        self,
        conn: sqlite3.Connection,
        context: PublishContext,
        draft: BackgroundDraft,
        versions: BackgroundInputVersions,
    ) -> PublicationOutcome:
        existing = self._existing_outcome(conn, context)
        if existing is not None:
            return existing
        instance, state = self._load_publish_fence(conn, context, versions)
        self._load_foreground_fence(
            conn,
            context,
            instance,
            draft,
            versions,
        )
        self._validate_draft(
            conn,
            context,
            instance,
            draft,
            versions,
        )
        content = self._write_publication_content(
            conn,
            context,
            state,
            draft,
            versions,
        )
        progress = self._advance_publication_state(
            conn,
            context,
            instance,
            content,
            draft,
            versions,
        )
        return self._persist_outcome_snapshot(
            conn,
            context,
            publication_id=content.publication_id,
            initialization_step=progress.initialization_step,
            event_ids=content.event_ids,
            story_refs=content.story_refs,
            message_cursor=progress.message_cursor,
            run_cursor=progress.run_cursor,
        )

    def _write_publication_content(
        self,
        conn: sqlite3.Connection,
        context: PublishContext,
        state: sqlite3.Row,
        draft: BackgroundDraft,
        versions: BackgroundInputVersions,
    ) -> PublishedContent:
        frame_start, frame_end = self._publication_frame_range(draft)
        publication_id, state_version = self._insert_publication(
            conn,
            context,
            state,
            draft,
            versions,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        story_refs = self._publish_story_sources(
            conn,
            context,
            publication_id,
            draft,
        )
        event_ids, timeline_changed = self._publish_timeline(
            conn,
            context,
            publication_id,
            draft,
        )
        self._link_and_advance_modules(conn, context, draft, event_ids)
        self._retire_leftover_events(conn, context, draft)
        view_changed = self._publish_current_view(
            conn,
            context,
            publication_id,
            draft,
            event_ids,
        )
        return PublishedContent(
            publication_id=publication_id,
            state_version=state_version,
            story_refs=story_refs,
            event_ids=event_ids,
            timeline_changed=timeline_changed,
            view_changed=view_changed,
        )

    def _advance_publication_state(
        self,
        conn: sqlite3.Connection,
        context: PublishContext,
        instance: sqlite3.Row,
        content: PublishedContent,
        draft: BackgroundDraft,
        versions: BackgroundInputVersions,
    ) -> PublicationProgress:
        ordinary_frame_completed = context.kind is BackgroundAuthorKind.ORDINARY
        initialization_step = self._next_initialization_step(
            context.kind,
            str(instance["initialization_state"]),
            str(instance["initialization_step"]),
        )
        message_cursor = max(
            int(instance["foreground_message_cursor"]),
            int(draft.consumed_foreground_through_message_id),
        )
        run_cursor = max(
            int(instance["foreground_run_cursor"]),
            int(draft.consumed_foreground_through_run_id),
        )
        simulated_through = self._next_simulated_through(
            context,
            instance,
            draft,
        )
        self._update_author_state(
            conn,
            context,
            state_version=content.state_version,
            publication_id=content.publication_id,
            draft=draft,
        )
        self._update_instance(
            conn,
            context,
            instance,
            versions,
            timeline_changed=content.timeline_changed,
            view_changed=content.view_changed,
            initialization_step=initialization_step,
            simulated_through=simulated_through,
            message_cursor=message_cursor,
            run_cursor=run_cursor,
            ordinary_frame_completed=ordinary_frame_completed,
        )
        self._advance_opening_progress(conn, context, instance)
        self._schedule_followups(
            conn,
            context,
            instance,
            draft,
            initialization_step=initialization_step,
            message_cursor=message_cursor,
            run_cursor=run_cursor,
            ordinary_frame_completed=ordinary_frame_completed,
        )
        completed_steps = self._initialization_completed_steps(instance, context.kind)
        if completed_steps is not None:
            self._queue_instance_initialization_progress(
                conn,
                context,
                completed_steps=completed_steps,
            )
        if (
            initialization_step == BackgroundInitializationStep.READY.value
            and str(instance["initialization_step"]) != BackgroundInitializationStep.READY.value
        ):
            self._complete_instance_initialization(conn, context)
        return PublicationProgress(
            initialization_step=initialization_step,
            message_cursor=message_cursor,
            run_cursor=run_cursor,
        )


__all__ = ["BackgroundPublicationMixin"]
