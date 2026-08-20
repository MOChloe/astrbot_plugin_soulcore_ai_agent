from __future__ import annotations

from datetime import datetime, timedelta

from ....contracts.runtime_limits import DURABLE_AI_MAX_ATTEMPTS
from ....shared.background_story_time import (
    BACKGROUND_STORY_LAG_MINUTES,
    background_story_cutoff_at,
)
from ....storage.sqlite.foreground_lease_registry import (
    recover_expired_foreground_leases_sql,
)
from .background_task_queries import (
    DUE_BACKGROUND_SLOTS_SQL,
    due_background_slot_params,
)
from .proactive_background_admission import proactive_frame_has_foreground_blocker
from .support import Any, _dt, _now, decode_task_payload, encode_task_payload, sqlite3
from .task_transactions import AiTaskCreationContext, AiTaskCreationTransaction

_BACKGROUND_RETRY_DELAYS_HOURS = [1 / 60, 5 / 60, 15 / 60, 1, 1]
_BACKGROUND_PRIORITIES = {
    "KEYFRAME": 55,
    "ORDINARY": 50,
    "STORY_SOURCE": 40,
    "LIFE_DIRECTION": 30,
    "WORLD": 20,
}
_TERMINAL_TASK_STATUSES = ("DEFERRED", "SUCCEEDED", "FAILED", "CANCELLED")
_PROACTIVE_FRAME_PRIORITY = 58
_PROACTIVE_FRAME_FRESH_MINUTES = 60
_PROACTIVE_FRAME_COOLDOWN_MINUTES = 60


class AiBackgroundTaskRecords:
    """Atomic bridge between background author slots and durable AI tasks."""

    @staticmethod
    def _clear_proactive_frame_cooldown_after_success_sql(
        conn: sqlite3.Connection,
        task: sqlite3.Row,
        *,
        now: str,
    ) -> None:
        """Successful publication is throttled by freshness, not failure cooldown."""

        if str(task["task_type"] or "").upper() != "BACKGROUND_AUTHOR":
            return
        input_data = decode_task_payload("input", task["input_json"])
        if not isinstance(input_data.get("proactive_frame"), dict):
            return
        conn.execute(
            """UPDATE background_instances
            SET last_proactive_frame_attempt_at = NULL, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?""",
            (now, task["profile_id"], task["instance_id"]),
        )

    @staticmethod
    def _requeue_background_task_slot_sql(
        conn: sqlite3.Connection,
        task: sqlite3.Row,
        *,
        now: str,
    ) -> bool:
        """Return a released durable task's author slot to its claimable state."""

        if str(task["task_type"] or "").upper() != "BACKGROUND_AUTHOR":
            return False
        input_data = decode_task_payload("input", task["input_json"])
        changed = conn.execute(
            """UPDATE background_author_states
            SET status = 'ENQUEUED',
                schedule_version = schedule_version + 1,
                updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND author_kind = ?
              AND generation = ? AND active_task_id = ?
              AND status IN ('RUNNING', 'ENQUEUED', 'FAILED')""",
            (
                now,
                task["profile_id"],
                task["instance_id"],
                str(input_data["author_kind"]),
                int(task["generation"]),
                int(task["task_id"]),
            ),
        ).rowcount
        return changed == 1

    async def materialize_due_background_tasks(self, *, limit: int = 12) -> list[dict[str, Any]]:
        now = _dt(_now())
        bounded_limit = max(1, min(int(limit), 100))

        def operation(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            recover_expired_foreground_leases_sql(conn, now=now)
            rows = self._due_background_slots(conn, now, bounded_limit)
            created: list[sqlite3.Row] = []
            for slot in rows:
                row = self._materialize_background_slot(conn, slot, now)
                if row is not None:
                    created.append(row)
            return created

        rows = await self.uow.run(operation)
        return [self._ai_task(row) for row in rows]

    @staticmethod
    def _due_background_slots(
        conn: sqlite3.Connection,
        now: str,
        limit: int,
    ) -> list[sqlite3.Row]:
        return list(
            conn.execute(
                DUE_BACKGROUND_SLOTS_SQL,
                due_background_slot_params(now, limit),
            )
        )

    def _materialize_background_slot(
        self,
        conn: sqlite3.Connection,
        slot: sqlite3.Row,
        now: str,
        *,
        proactive_frame: dict[str, Any] | None = None,
    ) -> sqlite3.Row | None:
        frame_end_at = self._background_frame_end_at(
            slot,
            now,
            proactive_frame=proactive_frame,
        )
        if frame_end_at is None:
            return None
        generation = int(slot["generation"]) + 1
        cursor = conn.execute(
            """UPDATE background_author_states
            SET status = 'ENQUEUED', generation = ?,
                schedule_version = schedule_version + 1,
                last_error = '', updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND author_kind = ?
              AND generation = ? AND status IN ('IDLE', 'FAILED')
              AND active_task_id IS NULL""",
            (
                generation,
                now,
                slot["profile_id"],
                slot["instance_id"],
                slot["author_kind"],
                slot["generation"],
            ),
        )
        if cursor.rowcount != 1:
            return None
        context = self._background_task_context(
            conn,
            slot,
            generation,
            now,
            proactive_frame=proactive_frame,
            frame_end_at=frame_end_at,
        )
        task = AiTaskCreationTransaction(self, context)(conn)
        bound = conn.execute(
            """UPDATE background_author_states
            SET active_task_id = ?, updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND author_kind = ?
              AND generation = ? AND status = 'ENQUEUED'
              AND active_task_id IS NULL""",
            (
                task["task_id"],
                now,
                slot["profile_id"],
                slot["instance_id"],
                slot["author_kind"],
                generation,
            ),
        )
        if bound.rowcount != 1:
            raise RuntimeError("background author task binding lost inside transaction")
        return task

    def _background_task_context(
        self,
        conn: sqlite3.Connection,
        slot: sqlite3.Row,
        generation: int,
        now: str,
        *,
        proactive_frame: dict[str, Any] | None = None,
        frame_end_at: str,
    ) -> AiTaskCreationContext:
        profile_id = str(slot["profile_id"])
        instance_id = str(slot["instance_id"])
        author_kind = str(slot["author_kind"])
        input_data = {
            "profile_id": profile_id,
            "instance_id": instance_id,
            "author_kind": author_kind,
            "generation": generation,
            "config_version": int(slot["config_version"]),
            "initialization_step": str(slot["initialization_step"]),
            "activity_epoch": int(slot["activity_epoch"]),
            "continuity_version": int(slot["continuity_version"]),
            "author_state_version": int(slot["state_version"]),
            "publication_version": int(slot["publication_version"]),
            "timeline_version": int(slot["timeline_version"]),
            "view_version": int(slot["view_version"]),
            "simulated_through_at": (
                str(slot["simulated_through_at"])
                if slot["simulated_through_at"] is not None
                else None
            ),
            "foreground_message_cursor": int(slot["foreground_message_cursor"]),
            "foreground_run_cursor": int(slot["foreground_run_cursor"]),
            "frame_end_at": frame_end_at,
        }
        if proactive_frame:
            input_data["proactive_frame"] = dict(proactive_frame)
        return AiTaskCreationContext(
            workflow_id=None,
            caused_by_workflow_id=None,
            origin_work_node_id=None,
            profile_id=profile_id,
            instance_id=instance_id,
            task_type="BACKGROUND_AUTHOR",
            task_class="BACKGROUND",
            capability="text.completion",
            initial_status="READY",
            priority=(
                _PROACTIVE_FRAME_PRIORITY
                if proactive_frame
                else _BACKGROUND_PRIORITIES[author_kind]
            ),
            due=now,
            step_key=None,
            mutex_key=f"background_instance:{instance_id}",
            backend_id=None,
            idempotency_key=f"background:{instance_id}:{author_kind}:{generation}",
            generation=generation,
            input_data=input_data,
            checkpoint={},
            retry_policy=(
                {"delays_hours": []}
                if proactive_frame
                else {"delays_hours": list(_BACKGROUND_RETRY_DELAYS_HOURS)}
            ),
            recovery_policy="RESUME_CHECKPOINT",
            max_attempts=1 if proactive_frame else DURABLE_AI_MAX_ATTEMPTS,
            actor_type="SYSTEM",
            actor_id=("proactive_frame_prewarmer" if proactive_frame else "background_scheduler"),
            now=now,
        )

    @staticmethod
    def _background_frame_end_at(
        slot: sqlite3.Row,
        now: str,
        *,
        proactive_frame: dict[str, Any] | None,
    ) -> str | None:
        if proactive_frame:
            return str(proactive_frame["frame_end_at"])
        opening_anchor = str(slot["initialization_anchor_at"] or "")
        if str(slot["initialization_step"] or "") != "READY" and opening_anchor:
            return opening_anchor
        stable_ref = ":".join(
            (
                "background-frame",
                str(slot["profile_id"]),
                str(slot["instance_id"]),
                str(slot["author_kind"]),
                str(int(slot["generation"]) + 1),
                str(slot["next_due_at"] or ""),
                str(slot["hard_due_at"] or ""),
            )
        )
        observed_now = datetime.fromisoformat(now)
        cutoff = background_story_cutoff_at(
            observed_now,
            stable_ref=stable_ref,
        )
        simulated = str(slot["simulated_through_at"] or "")
        if simulated:
            simulated_at = datetime.fromisoformat(simulated)
            latest_allowed = observed_now - timedelta(minutes=BACKGROUND_STORY_LAG_MINUTES[0])
            if simulated_at >= latest_allowed:
                return None
            if cutoff <= simulated_at:
                cutoff = latest_allowed
        return _dt(cutoff)

    async def materialize_proactive_frame_task(
        self,
        profile_id: str,
        instance_id: str,
        *,
        source_ref: str,
        planned_main_core_at: datetime,
        frame_end_at: datetime,
        deadline_at: datetime,
        now: datetime | None = None,
        requester_task_id: int | None = None,
    ) -> dict[str, Any]:
        """Atomically reuse or reserve one role frame for a proactive MainCore."""

        reference = str(source_ref or "").strip()
        if not reference:
            raise ValueError("proactive frame source_ref cannot be empty")
        for value in (planned_main_core_at, frame_end_at, deadline_at):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("proactive frame times must be timezone-aware")
        current_dt = now or _now()
        if current_dt.tzinfo is None or current_dt.utcoffset() is None:
            raise ValueError("proactive frame admission time must be timezone-aware")
        if frame_end_at > current_dt:
            return {"outcome": "NOT_DUE"}
        now_text = _dt(current_dt)
        planned_text = _dt(planned_main_core_at)
        fresh_cutoff = _dt(planned_main_core_at - timedelta(minutes=_PROACTIVE_FRAME_FRESH_MINUTES))
        cooldown_cutoff = _dt(current_dt - timedelta(minutes=_PROACTIVE_FRAME_COOLDOWN_MINUTES))

        return dict(
            await self.uow.run(
                lambda conn: self._materialize_proactive_frame_task_sql(
                    conn,
                    profile_id=profile_id,
                    instance_id=instance_id,
                    reference=reference,
                    planned_text=planned_text,
                    frame_end_text=_dt(frame_end_at),
                    deadline_text=_dt(deadline_at),
                    now_text=now_text,
                    fresh_cutoff=fresh_cutoff,
                    cooldown_cutoff=cooldown_cutoff,
                    requester_task_id=requester_task_id,
                )
            )
        )

    def _materialize_proactive_frame_task_sql(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        instance_id: str,
        reference: str,
        planned_text: str,
        frame_end_text: str,
        deadline_text: str,
        now_text: str,
        fresh_cutoff: str,
        cooldown_cutoff: str,
        requester_task_id: int | None,
    ) -> dict[str, Any]:
        recover_expired_foreground_leases_sql(conn, now=now_text)
        instance = self._proactive_frame_instance(conn, profile_id, instance_id)
        if instance is None:
            return {"outcome": "INELIGIBLE"}
        simulated = str(instance["simulated_through_at"] or "")
        if simulated and simulated > fresh_cutoff:
            return {"outcome": "FRESH", "simulated_through_at": simulated}
        if proactive_frame_has_foreground_blocker(
            conn,
            instance,
            now_text=now_text,
            requester_task_id=requester_task_id,
        ):
            return {"outcome": "BLOCKED"}

        active = self._active_role_frame_task(conn, profile_id, instance_id)
        if active is not None:
            return self._reuse_or_promote_role_frame(
                conn,
                active,
                reference=reference,
                planned_text=planned_text,
                frame_end_text=frame_end_text,
                deadline_text=deadline_text,
                now_text=now_text,
                cooldown_cutoff=cooldown_cutoff,
            )
        last_attempt = str(instance["last_proactive_frame_attempt_at"] or "")
        if last_attempt and last_attempt > cooldown_cutoff:
            return {"outcome": "COOLDOWN", "last_attempt_at": last_attempt}
        if self._instance_has_active_background_task(conn, profile_id, instance_id):
            return {"outcome": "BLOCKED"}
        slot = self._proactive_frame_slot(conn, profile_id, instance_id, now_text)
        if slot is None:
            return {"outcome": "INELIGIBLE"}
        metadata = self._proactive_frame_metadata(
            slot,
            reference=reference,
            planned_text=planned_text,
            frame_end_text=frame_end_text,
            deadline_text=deadline_text,
            now_text=now_text,
        )
        if metadata is None:
            return {"outcome": "INELIGIBLE"}
        if not self._reserve_proactive_frame_attempt(
            conn,
            profile_id,
            instance_id,
            now_text=now_text,
            cooldown_cutoff=cooldown_cutoff,
        ):
            return {"outcome": "COOLDOWN"}
        task = self._materialize_background_slot(
            conn,
            slot,
            now_text,
            proactive_frame=metadata,
        )
        if task is None:
            raise RuntimeError("proactive frame reservation lost after cooldown claim")
        return self._proactive_frame_created(task, metadata, reference, planned_text)

    @staticmethod
    def _proactive_frame_instance(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
    ) -> sqlite3.Row | None:
        instance = conn.execute(
            """SELECT instance.*,
                profile.background_life_enabled AS role_background_enabled
            FROM background_instances instance
            JOIN role_profiles profile ON profile.profile_id = instance.profile_id
            WHERE instance.profile_id = ? AND instance.instance_id = ?""",
            (profile_id, instance_id),
        ).fetchone()
        if (
            instance is None
            or not bool(instance["role_background_enabled"])
            or not bool(instance["proactive_frame_prewarm_enabled"])
            or str(instance["initialization_state"]) != "READY"
            or str(instance["initialization_step"]) != "READY"
            or int(instance["foreground_lease_count"] or 0) > 0
        ):
            return None
        return instance

    def _reuse_or_promote_role_frame(
        self,
        conn: sqlite3.Connection,
        task: sqlite3.Row,
        *,
        reference: str,
        planned_text: str,
        frame_end_text: str,
        deadline_text: str,
        now_text: str,
        cooldown_cutoff: str,
    ) -> dict[str, Any]:
        data = decode_task_payload("input", task["input_json"])
        if isinstance(data.get("proactive_frame"), dict) or str(task["status"]) not in {
            "READY",
            "SCHEDULED",
            "RETRY_WAIT",
        }:
            return {"outcome": "ACTIVE", "task": self._ai_task(task)}
        profile_id = str(task["profile_id"])
        instance_id = str(task["instance_id"])
        slot = conn.execute(
            """SELECT * FROM background_author_states
            WHERE profile_id = ? AND instance_id = ? AND active_task_id = ?
              AND author_kind IN ('ORDINARY','KEYFRAME')""",
            (profile_id, instance_id, int(task["task_id"])),
        ).fetchone()
        if slot is None:
            return {"outcome": "BLOCKED"}
        metadata = self._proactive_frame_metadata(
            slot,
            reference=reference,
            planned_text=planned_text,
            frame_end_text=frame_end_text,
            deadline_text=deadline_text,
            now_text=now_text,
        )
        if metadata is None:
            return {"outcome": "INELIGIBLE"}
        if not self._reserve_proactive_frame_attempt(
            conn,
            profile_id,
            instance_id,
            now_text=now_text,
            cooldown_cutoff=cooldown_cutoff,
        ):
            return {"outcome": "COOLDOWN"}
        data["proactive_frame"] = metadata
        changed = conn.execute(
            """UPDATE ai_tasks SET input_json = ?, priority = ?, due_at = ?,
                retry_policy_json = ?, max_attempts = 1, updated_at = ?,
                version = version + 1
            WHERE task_id = ? AND version = ?
              AND status IN ('READY','SCHEDULED','RETRY_WAIT')""",
            (
                encode_task_payload("input", data),
                _PROACTIVE_FRAME_PRIORITY,
                now_text,
                encode_task_payload("retry_policy", {"delays_hours": []}),
                now_text,
                int(task["task_id"]),
                int(task["version"]),
            ),
        ).rowcount
        if changed != 1:
            raise RuntimeError("active role frame changed during proactive promotion")
        promoted = conn.execute(
            "SELECT * FROM ai_tasks WHERE task_id = ?",
            (int(task["task_id"]),),
        ).fetchone()
        assert promoted is not None
        self._audit_ai_task(
            conn,
            promoted,
            "PROMOTE_PROACTIVE_FRAME",
            from_status=str(task["status"]),
            to_status=str(promoted["status"]),
            actor_type="SYSTEM",
            actor_id="proactive_frame_prewarmer",
            details={"source_ref": reference},
            created_at=now_text,
        )
        return {"outcome": "ACTIVE", "task": self._ai_task(promoted)}

    @staticmethod
    def _proactive_frame_metadata(
        slot: sqlite3.Row,
        *,
        reference: str,
        planned_text: str,
        frame_end_text: str,
        deadline_text: str,
        now_text: str,
    ) -> dict[str, Any] | None:
        preserve_schedule = not AiBackgroundTaskRecords._slot_is_due(slot, now_text)
        if preserve_schedule and (slot["next_due_at"] is None or slot["hard_due_at"] is None):
            return None
        return {
            "source_ref": reference,
            "planned_main_core_at": planned_text,
            "frame_end_at": frame_end_text,
            "deadline_at": deadline_text,
            "preserve_schedule": preserve_schedule,
            "original_next_due_at": (str(slot["next_due_at"]) if preserve_schedule else None),
            "original_hard_due_at": (str(slot["hard_due_at"]) if preserve_schedule else None),
        }

    @staticmethod
    def _reserve_proactive_frame_attempt(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        *,
        now_text: str,
        cooldown_cutoff: str,
    ) -> bool:
        return (
            conn.execute(
                """UPDATE background_instances
                SET last_proactive_frame_attempt_at = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?
                  AND proactive_frame_prewarm_enabled = 1
                  AND (last_proactive_frame_attempt_at IS NULL
                       OR last_proactive_frame_attempt_at <= ?)""",
                (
                    now_text,
                    now_text,
                    profile_id,
                    instance_id,
                    cooldown_cutoff,
                ),
            ).rowcount
            == 1
        )

    def _proactive_frame_created(
        self,
        task: sqlite3.Row,
        metadata: dict[str, Any],
        reference: str,
        planned_text: str,
    ) -> dict[str, Any]:
        return {
            "outcome": "CREATED",
            "task": self._ai_task(task),
            "source_ref": reference,
            "planned_main_core_at": planned_text,
            "frame_end_at": metadata["frame_end_at"],
            "deadline_at": metadata["deadline_at"],
        }

    @staticmethod
    def _active_role_frame_task(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
    ) -> sqlite3.Row | None:
        return conn.execute(
            """SELECT task.* FROM background_author_states author
            JOIN ai_tasks task ON task.task_id = author.active_task_id
            WHERE author.profile_id = ? AND author.instance_id = ?
              AND author.author_kind IN ('ORDINARY','KEYFRAME')
              AND task.status NOT IN ('DEFERRED','SUCCEEDED','FAILED','CANCELLED')
            ORDER BY CASE author.author_kind WHEN 'KEYFRAME' THEN 0 ELSE 1 END,
                     task.task_id LIMIT 1""",
            (profile_id, instance_id),
        ).fetchone()

    @staticmethod
    def _instance_has_active_background_task(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
    ) -> bool:
        placeholders = ",".join("?" for _ in _TERMINAL_TASK_STATUSES)
        return (
            conn.execute(
                f"""SELECT 1 FROM ai_tasks WHERE profile_id = ? AND instance_id = ?
                AND task_type = 'BACKGROUND_AUTHOR'
                AND status NOT IN ({placeholders}) LIMIT 1""",
                (profile_id, instance_id, *_TERMINAL_TASK_STATUSES),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _slot_is_due(slot: sqlite3.Row, now: str) -> bool:
        return bool(
            (slot["next_due_at"] is not None and str(slot["next_due_at"]) <= now)
            or (slot["hard_due_at"] is not None and str(slot["hard_due_at"]) <= now)
        )

    @staticmethod
    def _proactive_frame_slot(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        now: str,
    ) -> sqlite3.Row | None:
        rows = conn.execute(
            """SELECT state.*, instance.initialization_state,
                instance.initialization_step, instance.config_version,
                instance.continuity_version, instance.publication_version,
                instance.timeline_version, instance.view_version,
                instance.simulated_through_at,
                instance.foreground_message_cursor,
                instance.foreground_run_cursor, instance.ordinary_since_keyframe,
                core.activity_epoch
            FROM background_author_states state
            JOIN background_instances instance
              ON instance.profile_id = state.profile_id
             AND instance.instance_id = state.instance_id
            JOIN instance_core_state core
              ON core.profile_id = state.profile_id
             AND core.instance_id = state.instance_id
            WHERE state.profile_id = ? AND state.instance_id = ?
              AND state.author_kind IN ('KEYFRAME','ORDINARY')
              AND state.status IN ('IDLE','FAILED')
              AND state.active_task_id IS NULL
            ORDER BY CASE
                WHEN state.author_kind = 'KEYFRAME'
                 AND instance.ordinary_since_keyframe > 0
                 AND ((state.next_due_at IS NOT NULL AND state.next_due_at <= ?)
                   OR (state.hard_due_at IS NOT NULL AND state.hard_due_at <= ?)) THEN 0
                WHEN state.author_kind = 'ORDINARY'
                 AND ((state.next_due_at IS NOT NULL AND state.next_due_at <= ?)
                   OR (state.hard_due_at IS NOT NULL AND state.hard_due_at <= ?)) THEN 1
                WHEN state.author_kind = 'ORDINARY' THEN 2
                ELSE 3 END
            LIMIT 1""",
            (profile_id, instance_id, now, now, now, now),
        ).fetchall()
        return rows[0] if rows else None

    @staticmethod
    def _settle_background_task_sql(
        conn: sqlite3.Connection,
        task: sqlite3.Row,
        *,
        outcome: str,
        error: str,
        now: str,
    ) -> None:
        if str(task["task_type"] or "").upper() != "BACKGROUND_AUTHOR":
            return
        if outcome == "RECOVERY_REQUIRED":
            return
        input_data = decode_task_payload("input", task["input_json"])
        proactive_frame = input_data.get("proactive_frame")
        proactive_frame = dict(proactive_frame) if isinstance(proactive_frame, dict) else {}
        author_kind_value = input_data.get("author_kind")
        generation_value = input_data.get("generation")
        if not author_kind_value or generation_value is None:
            raise ValueError(
                "BACKGROUND_AUTHOR task settlement requires author_kind and generation"
            )
        author_kind = str(author_kind_value)
        try:
            generation = int(generation_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "BACKGROUND_AUTHOR task settlement requires an integer generation"
            ) from exc
        slot = conn.execute(
            """SELECT status, failure_count FROM background_author_states
            WHERE profile_id = ? AND instance_id = ? AND author_kind = ?
              AND generation = ? AND active_task_id = ?""",
            (
                task["profile_id"],
                task["instance_id"],
                author_kind,
                generation,
                int(task["task_id"]),
            ),
        ).fetchone()
        if slot is None:
            return
        counted_failure = outcome == "FAILED" and str(slot["status"]) != "FAILED"
        failure_count = int(slot["failure_count"]) + int(counted_failure)
        delay_index = min(max(failure_count, 1) - 1, len(_BACKGROUND_RETRY_DELAYS_HOURS) - 1)
        preserve_schedule = bool(proactive_frame.get("preserve_schedule"))
        retry_at = _dt(
            datetime.fromisoformat(now)
            + timedelta(hours=_BACKGROUND_RETRY_DELAYS_HOURS[delay_index])
        )
        next_due_at = proactive_frame.get("original_next_due_at") if preserve_schedule else retry_at
        hard_due_at = proactive_frame.get("original_hard_due_at") if preserve_schedule else retry_at
        conn.execute(
            """UPDATE background_author_states
            SET status = 'IDLE', active_task_id = NULL,
                failure_count = ?, next_due_at = ?, hard_due_at = ?,
                last_error = ?, schedule_version = schedule_version + 1,
                updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND author_kind = ?
              AND generation = ? AND active_task_id = ?
              AND status IN ('ENQUEUED', 'RUNNING', 'FAILED')""",
            (
                failure_count,
                next_due_at,
                hard_due_at,
                str(error)[:1000],
                now,
                task["profile_id"],
                task["instance_id"],
                author_kind,
                generation,
                int(task["task_id"]),
            ),
        )
        if proactive_frame:
            conn.execute(
                """UPDATE background_instances
                SET last_proactive_frame_attempt_at = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?""",
                (now, now, task["profile_id"], task["instance_id"]),
            )


__all__ = ["AiBackgroundTaskRecords"]
