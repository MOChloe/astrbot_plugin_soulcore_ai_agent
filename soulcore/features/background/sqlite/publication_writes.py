"""Database writes and follow-up scheduling for background publication."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import timedelta

from ....contracts.initialization import (
    INSTANCE_INITIALIZATION_PROGRESS_NOTICE_KIND,
    INSTANCE_INITIALIZATION_READY_NOTICE,
    INSTANCE_INITIALIZATION_READY_NOTICE_KIND,
    SYSTEM_NOTICE_KIND_KEY,
    instance_initialization_progress_notice,
)
from ....storage.sqlite.codec import _dt, _dump, _parse
from ..domain import (
    BackgroundAuthorKind,
    BackgroundDraft,
    BackgroundDraftStale,
    BackgroundInitializationStep,
    BackgroundInputVersions,
    BackgroundTimelineSource,
)
from .publication_models import (
    INITIALIZATION_AUTHOR,
    INITIALIZATION_NEXT,
    MAX_STORY_SOURCES,
    PublishContext,
    aware,
)
from .story_supply import read_story_supply_candidates, story_supply_eviction_order


class BackgroundPublicationWritesMixin:
    @staticmethod
    def _publication_frame_range(
        draft: BackgroundDraft,
    ) -> tuple[str | None, str | None]:
        if not draft.timeline_events:
            return None, None
        starts = tuple(aware(item.frame_start_at) for item in draft.timeline_events)
        ends = tuple(aware(item.frame_end_at) for item in draft.timeline_events)
        return _dt(min(starts)), _dt(max(ends))

    @staticmethod
    def _insert_publication(
        conn: sqlite3.Connection,
        context: PublishContext,
        state: sqlite3.Row,
        draft: BackgroundDraft,
        versions: BackgroundInputVersions,
        *,
        frame_start: str | None,
        frame_end: str | None,
    ) -> tuple[int, int]:
        state_version = int(state["state_version"]) + 1
        cursor = conn.execute(
            """INSERT INTO background_author_publications(
                public_ref, profile_id, instance_id, author_kind, generation,
                state_version, content_json,
                creator_output_json, input_versions_json,
                frame_start_at, frame_end_at, task_id, published_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"bgpub:{uuid.uuid4().hex}",
                context.profile_id,
                context.instance_id,
                context.kind.value,
                context.generation,
                state_version,
                _dump(draft.content),
                _dump(draft.creator_output),
                _dump(versions.as_dict()),
                frame_start,
                frame_end,
                context.task_id,
                context.now,
            ),
        )
        return int(cursor.lastrowid), state_version

    @staticmethod
    def _publish_story_sources(
        conn: sqlite3.Connection,
        context: PublishContext,
        publication_id: int,
        draft: BackgroundDraft,
    ) -> tuple[str, ...]:
        if not draft.story_sources:
            return ()
        refs: list[str] = []
        for source in draft.story_sources:
            module_text = str(source.module_text or "").strip()
            if not module_text:
                raise ValueError("story source requires complete module text")
            public_ref = f"bgstory:{uuid.uuid4().hex}"
            conn.execute(
                """INSERT INTO background_story_sources(
                    public_ref, profile_id, instance_id, module_text,
                    source_publication_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    public_ref,
                    context.profile_id,
                    context.instance_id,
                    module_text,
                    publication_id,
                    context.now,
                    context.now,
                ),
            )
            refs.append(public_ref)
        candidates = read_story_supply_candidates(
            conn,
            context.profile_id,
            context.instance_id,
        )
        overflow = max(0, len(candidates) - MAX_STORY_SOURCES)
        stale = story_supply_eviction_order(candidates)[:overflow]
        if stale:
            placeholders = ",".join("?" for _ in stale)
            stale_refs = {candidate.story.public_ref for candidate in stale}
            conn.execute(
                f"""DELETE FROM background_story_sources
                WHERE story_source_id IN ({placeholders})""",
                tuple(candidate.story.story_source_id for candidate in stale),
            )
            refs = [item for item in refs if item not in stale_refs]
        return tuple(refs)

    @staticmethod
    def _publish_timeline(
        conn: sqlite3.Connection,
        context: PublishContext,
        publication_id: int,
        draft: BackgroundDraft,
    ) -> tuple[tuple[int, ...], bool]:
        event_ids: list[int] = []
        for event in draft.timeline_events:
            cursor = conn.execute(
                """INSERT INTO background_role_timeline_events(
                    public_ref, profile_id, instance_id, source, content,
                    frame_start_at, frame_end_at, leftover_text,
                    source_publication_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"bgevent:{uuid.uuid4().hex}",
                    context.profile_id,
                    context.instance_id,
                    event.source.value,
                    str(event.content),
                    _dt(aware(event.frame_start_at)),
                    _dt(aware(event.frame_end_at)),
                    str(event.leftover_text or "").strip(),
                    publication_id,
                    context.now,
                ),
            )
            event_ids.append(int(cursor.lastrowid))
        return tuple(event_ids), bool(event_ids)

    @staticmethod
    def _link_and_advance_modules(
        conn: sqlite3.Connection,
        context: PublishContext,
        draft: BackgroundDraft,
        event_ids: tuple[int, ...],
    ) -> None:
        if not event_ids:
            return
        event_id = event_ids[0]
        now = context.now
        engaged_ids = tuple(dict.fromkeys(draft.engaged_story_ids))
        concluded_ids = tuple(dict.fromkeys(draft.concluded_story_ids))
        linked_ids = tuple(dict.fromkeys((*engaged_ids, *concluded_ids)))
        for story_source_id in linked_ids:
            conn.execute(
                """INSERT OR IGNORE INTO background_timeline_event_story_sources(
                    event_id, story_source_id, created_at
                ) VALUES (?, ?, ?)""",
                (event_id, story_source_id, now),
            )
        concluded_set = set(concluded_ids)
        for story_source_id in engaged_ids:
            if story_source_id in concluded_set:
                continue
            conn.execute(
                """UPDATE background_story_sources
                SET engagement_state = CASE
                        WHEN engagement_state = 'PENDING' THEN 'ACTIVE'
                        ELSE engagement_state
                    END,
                    shown_count = 0,
                    last_shown_at = NULL,
                    updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND story_source_id = ?
                  AND engagement_state != 'CONCLUDED'""",
                (now, context.profile_id, context.instance_id, story_source_id),
            )
        for story_source_id in concluded_ids:
            conn.execute(
                """UPDATE background_story_sources
                SET engagement_state = 'CONCLUDED', updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND story_source_id = ?
                  AND engagement_state != 'CONCLUDED'""",
                (now, context.profile_id, context.instance_id, story_source_id),
            )

    @staticmethod
    def _retire_leftover_events(
        conn: sqlite3.Connection,
        context: PublishContext,
        draft: BackgroundDraft,
    ) -> None:
        if not draft.retired_timeline_event_ids:
            return
        placeholders = ",".join("?" for _ in draft.retired_timeline_event_ids)
        cursor = conn.execute(
            f"""UPDATE background_role_timeline_events
            SET leftover_retired_at = ?
            WHERE profile_id = ? AND instance_id = ?
              AND event_id IN ({placeholders})
              AND leftover_text != '' AND leftover_retired_at IS NULL""",
            (
                context.now,
                context.profile_id,
                context.instance_id,
                *draft.retired_timeline_event_ids,
            ),
        )
        expected = len(set(draft.retired_timeline_event_ids))
        if cursor.rowcount != expected:
            raise BackgroundDraftStale("leftover resolution changed during publication")

    @staticmethod
    def _next_initialization_step(
        kind: BackgroundAuthorKind,
        initialization_state: str,
        initialization_step: str,
    ) -> str:
        if initialization_state == "READY":
            return BackgroundInitializationStep.READY.value
        if initialization_state != "INITIALIZING":
            raise BackgroundDraftStale("background initialization is not active")
        current = BackgroundInitializationStep(initialization_step)
        if current is BackgroundInitializationStep.ORDINARY_CURRENT:
            if kind is BackgroundAuthorKind.KEYFRAME:
                return BackgroundInitializationStep.ORDINARY_CURRENT.value
            if kind is BackgroundAuthorKind.ORDINARY:
                return BackgroundInitializationStep.READY.value
            raise BackgroundDraftStale("background opening life author changed")
        expected = INITIALIZATION_AUTHOR.get(current)
        if expected is not kind:
            raise BackgroundDraftStale("background initialization author changed")
        return INITIALIZATION_NEXT[current].value

    @staticmethod
    def _advance_opening_progress(
        conn: sqlite3.Connection,
        context: PublishContext,
        instance: sqlite3.Row,
    ) -> None:
        if (
            str(instance["initialization_state"]) != "INITIALIZING"
            or str(instance["initialization_step"])
            != BackgroundInitializationStep.ORDINARY_CURRENT.value
        ):
            return
        if context.kind is BackgroundAuthorKind.KEYFRAME:
            changed = conn.execute(
                """UPDATE background_initialization_openings
                SET keyframe_completed = 1, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?
                  AND keyframe_completed = 0""",
                (context.now, context.profile_id, context.instance_id),
            ).rowcount
            if changed != 1:
                raise BackgroundDraftStale("background opening Keyframe changed")
            return
        if context.kind is BackgroundAuthorKind.ORDINARY:
            row = conn.execute(
                """SELECT keyframe_completed
                FROM background_initialization_openings
                WHERE profile_id = ? AND instance_id = ?""",
                (context.profile_id, context.instance_id),
            ).fetchone()
            if row is None or not bool(row["keyframe_completed"]):
                raise BackgroundDraftStale("background opening handoff is incomplete")

    @staticmethod
    def _next_simulated_through(
        context: PublishContext,
        instance: sqlite3.Row,
        draft: BackgroundDraft,
    ) -> str | None:
        current = _parse(instance["simulated_through_at"])
        if context.kind is BackgroundAuthorKind.ORDINARY:
            expected_source = BackgroundTimelineSource.ORDINARY
        elif context.kind is BackgroundAuthorKind.KEYFRAME:
            expected_source = BackgroundTimelineSource.KEYFRAME
        else:
            return _dt(current) if current is not None else None
        role_frame_ends = tuple(
            aware(item.frame_end_at)
            for item in draft.timeline_events
            if item.source is expected_source
        )
        if not role_frame_ends:
            return _dt(current) if current is not None else None
        frame_end = max(role_frame_ends)
        return _dt(max(current, frame_end) if current is not None else frame_end)

    @staticmethod
    def _update_author_state(
        conn: sqlite3.Connection,
        context: PublishContext,
        *,
        state_version: int,
        publication_id: int,
        draft: BackgroundDraft,
    ) -> None:
        cursor = conn.execute(
            """UPDATE background_author_states
            SET state_version = ?, schedule_version = schedule_version + 1,
                state_json = ?, status = 'IDLE',
                next_due_at = ?, hard_due_at = ?, last_success_at = ?,
                last_publication_id = ?, active_task_id = NULL,
                failure_count = 0, last_error = '', updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND author_kind = ?
              AND generation = ? AND active_task_id = ? AND status = 'RUNNING'""",
            (
                state_version,
                _dump(draft.content),
                context.next_due,
                context.hard_due,
                context.now,
                publication_id,
                context.now,
                context.profile_id,
                context.instance_id,
                context.kind.value,
                context.generation,
                context.task_id,
            ),
        )
        if cursor.rowcount != 1:
            raise BackgroundDraftStale("author state changed during publication")

    @staticmethod
    def _update_instance(
        conn: sqlite3.Connection,
        context: PublishContext,
        instance: sqlite3.Row,
        versions: BackgroundInputVersions,
        *,
        timeline_changed: bool,
        view_changed: bool,
        initialization_step: str,
        simulated_through: str | None,
        message_cursor: int,
        run_cursor: int,
        ordinary_frame_completed: bool,
    ) -> None:
        is_initial_ordinary = (
            str(instance["initialization_step"])
            == BackgroundInitializationStep.ORDINARY_CURRENT.value
        )
        ordinary_delta = int(ordinary_frame_completed and not is_initial_ordinary)
        reset_keyframe = context.kind is BackgroundAuthorKind.KEYFRAME
        cursor = conn.execute(
            """UPDATE background_instances
            SET publication_version = publication_version + 1,
                timeline_version = timeline_version + ?,
                view_version = view_version + ?,
                ordinary_since_keyframe = CASE
                    WHEN ? = 1 THEN 0
                    ELSE ordinary_since_keyframe + ? END,
                simulated_through_at = ?,
                foreground_message_cursor = ?,
                foreground_run_cursor = ?,
                initialization_state = CASE
                    WHEN ? = 'READY' THEN 'READY' ELSE 'INITIALIZING' END,
                initialization_step = ?, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?
              AND config_version = ? AND continuity_version = ?
              AND timeline_version = ? AND view_version = ?
              AND publication_version = ?""",
            (
                int(timeline_changed),
                int(view_changed),
                int(reset_keyframe),
                ordinary_delta,
                simulated_through,
                message_cursor,
                run_cursor,
                initialization_step,
                initialization_step,
                context.now,
                context.profile_id,
                context.instance_id,
                versions.config_version,
                versions.continuity_version,
                versions.timeline_version,
                versions.view_version,
                versions.publication_version,
            ),
        )
        if cursor.rowcount != 1:
            raise BackgroundDraftStale("background instance changed during publication")

    @staticmethod
    def _replace_author_schedule(
        conn: sqlite3.Connection,
        context: PublishContext,
        author_kind: BackgroundAuthorKind,
        *,
        next_due: str | None,
        hard_due: str | None,
    ) -> None:
        conn.execute(
            """UPDATE background_author_states
            SET next_due_at = ?, hard_due_at = ?,
                schedule_version = schedule_version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND author_kind = ?""",
            (
                next_due,
                hard_due,
                context.now,
                context.profile_id,
                context.instance_id,
                author_kind.value,
            ),
        )

    @staticmethod
    def _schedule_followups(
        conn: sqlite3.Connection,
        context: PublishContext,
        instance: sqlite3.Row,
        draft: BackgroundDraft,
        *,
        initialization_step: str,
        message_cursor: int,
        run_cursor: int,
        ordinary_frame_completed: bool,
    ) -> None:
        if initialization_step != BackgroundInitializationStep.READY.value:
            step = BackgroundInitializationStep(initialization_step)
            if step is BackgroundInitializationStep.ORDINARY_CURRENT:
                opening = conn.execute(
                    """SELECT keyframe_completed
                    FROM background_initialization_openings
                    WHERE profile_id = ? AND instance_id = ?""",
                    (context.profile_id, context.instance_id),
                ).fetchone()
                if opening is None:
                    raise BackgroundDraftStale("background opening anchor disappeared")
                target = (
                    BackgroundAuthorKind.ORDINARY
                    if bool(opening["keyframe_completed"])
                    else BackgroundAuthorKind.KEYFRAME
                )
            else:
                target = INITIALIZATION_AUTHOR[step]
            BackgroundPublicationWritesMixin._replace_author_schedule(
                conn,
                context,
                target,
                next_due=context.now,
                hard_due=context.now,
            )
            return
        if (
            ordinary_frame_completed
            and str(instance["initialization_step"])
            == BackgroundInitializationStep.ORDINARY_CURRENT.value
        ):
            keyframe_due = _dt(
                context.published_at + timedelta(minutes=int(instance["keyframe_max_minutes"]))
            )
            BackgroundPublicationWritesMixin._replace_author_schedule(
                conn,
                context,
                BackgroundAuthorKind.KEYFRAME,
                next_due=keyframe_due,
                hard_due=keyframe_due,
            )
            return
        if context.kind is BackgroundAuthorKind.KEYFRAME:
            if context.preserve_schedule:
                return
            keyframe_due = _dt(
                context.published_at + timedelta(minutes=int(instance["keyframe_max_minutes"]))
            )
            BackgroundPublicationWritesMixin._replace_author_schedule(
                conn,
                context,
                BackgroundAuthorKind.ORDINARY,
                next_due=context.next_due,
                hard_due=context.hard_due,
            )
            BackgroundPublicationWritesMixin._replace_author_schedule(
                conn,
                context,
                BackgroundAuthorKind.KEYFRAME,
                next_due=keyframe_due,
                hard_due=keyframe_due,
            )
            return
        if (
            ordinary_frame_completed
            and str(instance["initialization_step"])
            != BackgroundInitializationStep.ORDINARY_CURRENT.value
        ):
            next_count = int(instance["ordinary_since_keyframe"]) + 1
            if next_count >= int(instance["keyframe_every_ordinary"]):
                conn.execute(
                    """UPDATE background_author_states
                    SET next_due_at = CASE
                            WHEN next_due_at IS NULL OR next_due_at > ? THEN ? ELSE next_due_at END,
                        hard_due_at = CASE
                            WHEN hard_due_at IS NULL OR hard_due_at > ? THEN ? ELSE hard_due_at END,
                        schedule_version = schedule_version + 1, updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND author_kind = 'KEYFRAME'""",
                    (
                        context.next_due,
                        context.next_due,
                        context.hard_due,
                        context.hard_due,
                        context.now,
                        context.profile_id,
                        context.instance_id,
                    ),
                )
                conn.execute(
                    """UPDATE background_author_states
                    SET next_due_at = NULL, hard_due_at = NULL,
                        schedule_version = schedule_version + 1, updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND author_kind = 'ORDINARY'""",
                    (
                        context.now,
                        context.profile_id,
                        context.instance_id,
                    ),
                )

    @staticmethod
    def _initialization_completed_steps(
        instance: sqlite3.Row,
        author_kind: BackgroundAuthorKind,
    ) -> int | None:
        if str(instance["initialization_state"]) != "INITIALIZING":
            return None
        step = BackgroundInitializationStep(str(instance["initialization_step"]))
        milestones = {
            (BackgroundInitializationStep.WORLD, BackgroundAuthorKind.WORLD): 1,
            (
                BackgroundInitializationStep.LIFE_DIRECTION,
                BackgroundAuthorKind.LIFE_DIRECTION,
            ): 2,
            (
                BackgroundInitializationStep.STORY_SOURCE,
                BackgroundAuthorKind.STORY_SOURCE,
            ): 3,
            (
                BackgroundInitializationStep.ORDINARY_CURRENT,
                BackgroundAuthorKind.KEYFRAME,
            ): 4,
        }
        return milestones.get((step, author_kind))

    @staticmethod
    def _initialization_delivery_target(
        conn: sqlite3.Connection,
        context: PublishContext,
    ) -> tuple[str, int]:
        target = conn.execute(
            """SELECT character.route_umo, core.activity_epoch
            FROM character_instances AS character
            JOIN instance_core_state AS core
              ON core.profile_id = character.profile_id
             AND core.instance_id = character.instance_id
            WHERE character.profile_id = ? AND character.instance_id = ?""",
            (context.profile_id, context.instance_id),
        ).fetchone()
        if target is None:
            raise KeyError((context.profile_id, context.instance_id))
        return str(target["route_umo"] or ""), int(target["activity_epoch"])

    @staticmethod
    def _enqueue_initialization_notice(
        conn: sqlite3.Connection,
        context: PublishContext,
        *,
        route_umo: str,
        activity_epoch: int,
        content: str,
        notice_kind: str,
        idempotency_key: str,
        depends_on_idempotency_key: str | None,
    ) -> None:
        payload = {
            "content": content,
            "origin_kind": "SYSTEM_EVENT",
            "context_record": False,
            SYSTEM_NOTICE_KIND_KEY: notice_kind,
        }
        conn.execute(
            """INSERT INTO instance_outbox(
                workflow_id, profile_id, instance_id, route_umo, payload_json,
                status, idempotency_key, activity_epoch, origin_kind,
                origin_task_id, interrupt_policy, depends_on_idempotency_key,
                created_at, updated_at
            ) VALUES (
                (SELECT workflow_id FROM ai_tasks WHERE task_id = ?),
                ?, ?, ?, ?, 'PENDING', ?, ?, 'SYSTEM_EVENT', ?, 'PRESERVE', ?, ?, ?
            )
            ON CONFLICT(profile_id, instance_id, idempotency_key) DO NOTHING""",
            (
                context.task_id,
                context.profile_id,
                context.instance_id,
                route_umo,
                _dump(payload),
                idempotency_key,
                activity_epoch,
                context.task_id,
                depends_on_idempotency_key,
                context.now,
                context.now,
            ),
        )

    @staticmethod
    def _queue_instance_initialization_progress(
        conn: sqlite3.Connection,
        context: PublishContext,
        *,
        completed_steps: int,
    ) -> None:
        route_umo, activity_epoch = (
            BackgroundPublicationWritesMixin._initialization_delivery_target(
                conn,
                context,
            )
        )
        previous_key = (
            f"background-initialization-progress-{completed_steps - 1}"
            if completed_steps > 1
            else None
        )
        BackgroundPublicationWritesMixin._enqueue_initialization_notice(
            conn,
            context,
            route_umo=route_umo,
            activity_epoch=activity_epoch,
            content=instance_initialization_progress_notice(completed_steps),
            notice_kind=INSTANCE_INITIALIZATION_PROGRESS_NOTICE_KIND,
            idempotency_key=f"background-initialization-progress-{completed_steps}",
            depends_on_idempotency_key=previous_key,
        )

    @staticmethod
    def _complete_instance_initialization(
        conn: sqlite3.Connection,
        context: PublishContext,
    ) -> None:
        character = conn.execute(
            """SELECT route_umo, initialization_state FROM character_instances
            WHERE profile_id = ? AND instance_id = ?""",
            (context.profile_id, context.instance_id),
        ).fetchone()
        core_state = conn.execute(
            """SELECT activity_epoch FROM instance_core_state
            WHERE profile_id = ? AND instance_id = ?""",
            (context.profile_id, context.instance_id),
        ).fetchone()
        if character is None or core_state is None:
            raise KeyError((context.profile_id, context.instance_id))
        if str(character["initialization_state"]) == "READY":
            return
        if str(character["initialization_state"]) != "INITIALIZING":
            raise BackgroundDraftStale("conversation initialization state changed")
        cursor = conn.execute(
            """UPDATE character_instances
            SET initialization_state = 'READY',
                initialization_completed_at = ?, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?
              AND initialization_state = 'INITIALIZING'""",
            (
                context.now,
                context.now,
                context.profile_id,
                context.instance_id,
            ),
        )
        if cursor.rowcount != 1:
            raise BackgroundDraftStale("conversation initialization state changed")
        conn.execute(
            """UPDATE deferred_message_batches
            SET due_at = ?, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?
              AND creation_key LIKE 'instance-initialization:%'
              AND status = 'PENDING'""",
            (
                context.now,
                context.now,
                context.profile_id,
                context.instance_id,
            ),
        )
        BackgroundPublicationWritesMixin._enqueue_initialization_notice(
            conn,
            context,
            route_umo=str(character["route_umo"] or ""),
            activity_epoch=int(core_state["activity_epoch"]),
            content=INSTANCE_INITIALIZATION_READY_NOTICE,
            notice_kind=INSTANCE_INITIALIZATION_READY_NOTICE_KIND,
            idempotency_key="background-initialization-ready",
            depends_on_idempotency_key="background-initialization-progress-4",
        )


__all__ = ["BackgroundPublicationWritesMixin"]
