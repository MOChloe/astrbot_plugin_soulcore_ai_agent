from __future__ import annotations

from datetime import UTC

from ....contracts.runtime_limits import DURABLE_AI_MAX_ATTEMPTS
from ..durable_task_runtime import (
    PrerequisiteTaskClaim,
    PrerequisiteTaskClaimOutcome,
)
from ..workflow_context import current_ai_work_context
from .support import (
    AI_TASK_RETRY_HOURS,
    Any,
    _dt,
    _dump,
    _now,
    datetime,
    decode_task_payload,
    encode_task_payload,
    sqlite3,
    timedelta,
)
from .task_claim_queries import (
    build_active_prerequisite_task_query,
    build_prerequisite_task_claim_query,
)
from .task_transactions import (
    AiTaskClaimContext,
    AiTaskClaimTransaction,
    AiTaskCreationContext,
    AiTaskCreationTransaction,
)

_RECOVERY_POLICIES = {
    "RESTART_SAFE",
    "RESUME_CHECKPOINT",
    "RECONCILE_EXTERNAL",
    "NO_RETRY",
}
_PROACTIVE_MAIN_CORE_SOURCES = {"PLUGIN_WAKE", "TIMER"}


def _proactive_main_core_input(
    input_data: dict[str, Any] | None,
    *,
    task_type: str,
    task_class: str,
    due_at: datetime,
    idempotency_key: str | None,
) -> dict[str, Any]:
    payload = dict(input_data or {})
    source = str(payload.get("source") or "").strip().upper()
    if (
        task_type != "MAIN_CORE"
        or task_class != "BACKGROUND"
        or source not in _PROACTIVE_MAIN_CORE_SOURCES
    ):
        return payload
    raw_schedule = payload.get("_proactive_frame_schedule")
    schedule = dict(raw_schedule) if isinstance(raw_schedule, dict) else {}
    planned = _proactive_schedule_datetime(
        schedule.get("planned_main_core_at"),
        fallback=due_at,
    )
    reference = str(schedule.get("source_ref") or "").strip()
    if not reference:
        wakeup_id = int(payload.get("wakeup_id") or 0)
        metadata = payload.get("metadata")
        metadata_values = metadata if isinstance(metadata, dict) else {}
        contact_ref = str(metadata_values.get("contact_attempt_ref") or "").strip()
        if wakeup_id > 0:
            reference = f"instance-wakeup:{wakeup_id}"
        elif contact_ref:
            reference = f"contact-attempt:{contact_ref}"
        elif idempotency_key:
            reference = f"ai-task-key:MAIN_CORE:{idempotency_key}"
    payload["_proactive_frame_schedule"] = {
        "planned_main_core_at": _dt(planned),
        "source_ref": reference,
    }
    return payload


def _proactive_schedule_datetime(value: object, *, fallback: datetime) -> datetime:
    if value in (None, ""):
        parsed = fallback
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("proactive MainCore planned time must be timezone-aware")
    return parsed.astimezone(UTC)


def _task_identity(task_type: str, instance_id: str, task_class: str) -> tuple[str, str, str]:
    normalized_type = str(task_type or "").strip().upper()
    normalized_instance = str(instance_id or "").strip()
    normalized_class = str(task_class).upper()
    if not normalized_type:
        raise ValueError("task_type cannot be empty")
    if not normalized_instance:
        raise ValueError("instance_id cannot be empty")
    if normalized_class not in {"FOREGROUND", "BACKGROUND"}:
        raise ValueError("task_class must be FOREGROUND or BACKGROUND")
    return normalized_type, normalized_instance, normalized_class


def _task_attempts(task_class: str, maximum: int | None) -> int:
    attempts = (
        (DURABLE_AI_MAX_ATTEMPTS if task_class == "BACKGROUND" else 5)
        if maximum is None
        else int(maximum)
    )
    if attempts < 1:
        raise ValueError("invalid max_attempts")
    return attempts


class AiTaskRecords:
    @staticmethod
    def _ai_retry_due(
        attempts: int,
        now: datetime | None = None,
        retry_policy: dict[str, Any] | None = None,
    ) -> datetime:
        base = now or _now()
        configured = (retry_policy or {}).get("delays_hours")
        delays: list[float] = []
        if isinstance(configured, list):
            for value in configured:
                try:
                    delay = float(value)
                except (TypeError, ValueError):
                    continue
                if delay >= 0:
                    delays.append(delay)
        if not delays:
            delays = [float(value) for value in AI_TASK_RETRY_HOURS]
        index = min(max(1, int(attempts)) - 1, len(delays) - 1)
        return base + timedelta(hours=delays[index])

    @staticmethod
    def _audit_ai_task(
        conn: sqlite3.Connection,
        row: sqlite3.Row | dict[str, Any],
        action: str,
        *,
        from_status: str | None = None,
        to_status: str | None = None,
        actor_type: str = "SYSTEM",
        actor_id: str = "",
        details: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> None:
        conn.execute(
            """INSERT INTO ai_task_audit(
                task_id, profile_id, instance_id, actor_type, actor_id,
                action, from_status, to_status, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["task_id"],
                row["profile_id"],
                row["instance_id"],
                actor_type,
                actor_id,
                action,
                from_status,
                to_status,
                _dump(details or {}),
                created_at or _dt(_now()),
            ),
        )

    async def create_ai_task(
        self,
        profile_id: str,
        task_type: str,
        *,
        instance_id: str,
        task_class: str = "BACKGROUND",
        capability: str | None = None,
        due_at: datetime | None = None,
        priority: int = 0,
        step_key: str | None = None,
        mutex_key: str | None = None,
        backend_id: str | None = None,
        idempotency_key: str | None = None,
        generation: int = 1,
        input_data: dict[str, Any] | None = None,
        checkpoint: dict[str, Any] | None = None,
        retry_policy: dict[str, Any] | None = None,
        recovery_policy: str = "RESUME_CHECKPOINT",
        max_attempts: int | None = None,
        actor_type: str = "SYSTEM",
        actor_id: str = "",
        workflow_id: int | None = None,
        caused_by_workflow_id: int | None = None,
        origin_work_node_id: int | None = None,
    ) -> dict[str, Any]:
        normalized_type, normalized_instance_id, normalized_class = _task_identity(
            task_type, instance_id, task_class
        )
        effective_attempts = _task_attempts(normalized_class, max_attempts)
        if int(generation) < 1:
            raise ValueError("invalid task generation")
        if recovery_policy not in _RECOVERY_POLICIES:
            raise ValueError("unsupported recovery_policy")
        now_dt = _now()
        due_dt = due_at or now_dt
        trace = current_ai_work_context()
        if trace is not None:
            origin_work_node_id = origin_work_node_id or trace.node_id
            if due_dt <= now_dt:
                workflow_id = workflow_id or trace.workflow_id
            else:
                caused_by_workflow_id = caused_by_workflow_id or trace.workflow_id
        normalized_key = str(idempotency_key).strip() if idempotency_key else None
        prepared_input = _proactive_main_core_input(
            input_data,
            task_type=normalized_type,
            task_class=normalized_class,
            due_at=due_dt,
            idempotency_key=normalized_key,
        )
        context = AiTaskCreationContext(
            workflow_id=workflow_id,
            caused_by_workflow_id=caused_by_workflow_id,
            origin_work_node_id=origin_work_node_id,
            profile_id=profile_id,
            instance_id=normalized_instance_id,
            task_type=normalized_type,
            task_class=normalized_class,
            capability=str(capability).strip() if capability else None,
            initial_status="SCHEDULED" if due_dt > now_dt else "READY",
            priority=int(priority),
            due=_dt(due_dt),
            step_key=step_key,
            mutex_key=mutex_key,
            backend_id=backend_id,
            idempotency_key=normalized_key,
            generation=int(generation),
            input_data=prepared_input,
            checkpoint=checkpoint or {},
            retry_policy=retry_policy,
            recovery_policy=recovery_policy,
            max_attempts=effective_attempts,
            actor_type=actor_type,
            actor_id=actor_id,
            now=_dt(now_dt),
        )
        row = await self.uow.run(AiTaskCreationTransaction(self, context))
        return self._ai_task(row)

    async def get_ai_task(self, task_id: int) -> dict[str, Any] | None:
        row = await self.db.fetch_one("SELECT * FROM ai_tasks WHERE task_id = ?", (int(task_id),))
        return self._ai_task(row) if row else None

    async def list_ai_tasks(
        self,
        *,
        profile_id: str | None = None,
        instance_id: str | None = None,
        status: str | None = None,
        task_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses = ["1 = 1"]
        params: list[Any] = []
        if profile_id is not None:
            clauses.append("profile_id = ?")
            params.append(profile_id)
        if instance_id is not None:
            clauses.append("instance_id = ?")
            params.append(instance_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(str(status).upper())
        if task_type is not None:
            clauses.append("task_type = ?")
            params.append(str(task_type).upper())
        params.append(max(1, min(int(limit), 1000)))
        rows = await self.db.fetch_all(
            f"""SELECT * FROM ai_tasks WHERE {" AND ".join(clauses)}
            ORDER BY created_at DESC, task_id DESC LIMIT ?""",
            params,
        )
        return [self._ai_task(row) for row in rows]

    def _claim_ai_tasks_sql(
        self,
        conn: sqlite3.Connection,
        *,
        worker_id: str,
        limit: int,
        lease_until: str,
        now_text: str,
        normalized_types: list[str],
    ) -> list[sqlite3.Row]:
        context = AiTaskClaimContext(
            worker_id=worker_id,
            limit=int(limit),
            lease_until=lease_until,
            now_text=now_text,
            normalized_types=normalized_types,
        )
        return AiTaskClaimTransaction(self, context)(conn)

    async def claim_ai_tasks(
        self,
        worker_id: str,
        *,
        limit: int = 1,
        lease_seconds: int = 300,
        task_types: tuple[str, ...] | list[str] | None = None,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        worker_id = str(worker_id or "").strip()
        if not worker_id:
            raise ValueError("worker_id cannot be empty")
        current = now or _now()
        now_text = _dt(current)
        lease_until = _dt(current + timedelta(seconds=max(1, int(lease_seconds))))
        normalized_types = [str(item).upper() for item in (task_types or [])]

        def operation(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            return self._claim_ai_tasks_sql(
                conn,
                worker_id=worker_id,
                limit=limit,
                lease_until=lease_until,
                now_text=now_text,
                normalized_types=normalized_types,
            )

        rows = await self.uow.run(operation)
        return [self._ai_task(row) for row in rows]

    async def claim_ai_task_prerequisite(
        self,
        worker_id: str,
        task_id: int,
        requester_task_id: int,
        requester_lease_token: int,
        *,
        lease_seconds: int = 300,
        now: datetime | None = None,
    ) -> PrerequisiteTaskClaim:
        """Claim one exact proactive frame without relaxing unrelated fences."""

        worker_id = str(worker_id or "").strip()
        if not worker_id:
            raise ValueError("worker_id cannot be empty")
        task_id = int(task_id)
        requester_task_id = int(requester_task_id)
        requester_lease_token = int(requester_lease_token)
        if task_id < 1 or requester_task_id < 1 or requester_lease_token < 1:
            raise ValueError("invalid prerequisite task identity")
        if task_id == requester_task_id:
            raise ValueError("a durable task cannot be its own prerequisite")
        current = now or _now()
        now_text = _dt(current)
        lease_until = _dt(current + timedelta(seconds=max(1, int(lease_seconds))))
        context = AiTaskClaimContext(
            worker_id=worker_id,
            limit=1,
            lease_until=lease_until,
            now_text=now_text,
            normalized_types=[],
        )

        def operation(conn: sqlite3.Connection) -> PrerequisiteTaskClaim:
            query = build_prerequisite_task_claim_query(
                now_text,
                task_id=task_id,
                requester_task_id=requester_task_id,
                requester_lease_token=requester_lease_token,
                worker_id=worker_id,
            )
            candidate = conn.execute(query.sql, query.params).fetchone()
            if candidate is not None:
                claimed = AiTaskClaimTransaction(self, context)._claim_task(conn, candidate)
                if claimed is not None:
                    return PrerequisiteTaskClaim(
                        PrerequisiteTaskClaimOutcome.CLAIMED,
                        self._ai_task(claimed),
                    )

            active_query = build_active_prerequisite_task_query(
                now_text,
                task_id=task_id,
                requester_task_id=requester_task_id,
                requester_lease_token=requester_lease_token,
                worker_id=worker_id,
            )
            active = conn.execute(active_query.sql, active_query.params).fetchone()
            if active is not None:
                return PrerequisiteTaskClaim(
                    PrerequisiteTaskClaimOutcome.ACTIVE,
                    self._ai_task(active),
                )
            current_task = conn.execute(
                "SELECT * FROM ai_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            return PrerequisiteTaskClaim(
                PrerequisiteTaskClaimOutcome.NOT_CLAIMABLE,
                self._ai_task(current_task) if current_task is not None else None,
            )

        return await self.uow.run(operation)

    async def heartbeat_ai_task(
        self,
        task_id: int,
        lease_token: int,
        worker_id: str,
        *,
        lease_seconds: int = 300,
        checkpoint: dict[str, Any] | None = None,
        progress: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        now_dt = _now()
        now = _dt(now_dt)
        lease_until = _dt(now_dt + timedelta(seconds=max(1, int(lease_seconds))))

        def operation(conn: sqlite3.Connection) -> sqlite3.Row | None:
            row = conn.execute(
                "SELECT * FROM ai_tasks WHERE task_id = ?", (int(task_id),)
            ).fetchone()
            if (
                row is None
                or int(row["lease_token"]) != int(lease_token)
                or row["lease_owner"] != worker_id
                or row["status"] not in {"RUNNING", "PAUSE_REQUESTED", "CANCEL_REQUESTED"}
            ):
                return None
            conn.execute(
                """UPDATE ai_tasks SET lease_until = ?,
                    checkpoint_json = COALESCE(?, checkpoint_json),
                    progress_json = COALESCE(?, progress_json), updated_at = ?,
                    version = version + 1
                WHERE task_id = ? AND lease_token = ? AND lease_owner = ?""",
                (
                    lease_until,
                    encode_task_payload("checkpoint", checkpoint)
                    if checkpoint is not None
                    else None,
                    encode_task_payload("progress", progress) if progress is not None else None,
                    now,
                    int(task_id),
                    int(lease_token),
                    worker_id,
                ),
            )
            conn.execute(
                """UPDATE ai_task_attempts SET heartbeat_at = ?
                WHERE task_id = ? AND lease_token = ? AND status = 'RUNNING'""",
                (now, int(task_id), int(lease_token)),
            )
            return conn.execute(
                "SELECT * FROM ai_tasks WHERE task_id = ?", (int(task_id),)
            ).fetchone()

        row = await self.uow.run(operation)
        return self._ai_task(row) if row else None

    async def complete_ai_task(
        self,
        task_id: int,
        lease_token: int,
        worker_id: str,
        *,
        result: dict[str, Any] | None = None,
    ) -> bool:
        return await self._finish_ai_task_lease(
            task_id, lease_token, worker_id, "SUCCEEDED", result=result
        )

    async def defer_ai_task(
        self,
        task_id: int,
        lease_token: int,
        worker_id: str,
        *,
        result: dict[str, Any] | None,
        reason: str = "",
    ) -> bool:
        """Finish a leased task without reporting success or occupying a worker.

        DEFERRED is a durable terminal outcome.  A recoverable sticker candidate
        owns the subsequent retry schedule; this old task is never reclaimed.
        """

        return await self._finish_ai_task_lease(
            task_id,
            lease_token,
            worker_id,
            "DEFERRED",
            result=result,
            terminal_reason=reason,
        )

    async def fail_ai_task(
        self,
        task_id: int,
        lease_token: int,
        worker_id: str,
        error: str,
        *,
        retryable: bool = True,
        recovery_required: bool = False,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        now_dt = now or _now()
        now_text = _dt(now_dt)

        def operation(conn: sqlite3.Connection) -> sqlite3.Row | None:
            row = conn.execute(
                "SELECT * FROM ai_tasks WHERE task_id = ?", (int(task_id),)
            ).fetchone()
            if (
                row is None
                or row["status"] != "RUNNING"
                or int(row["lease_token"]) != int(lease_token)
                or row["lease_owner"] != worker_id
            ):
                return None
            can_retry = bool(retryable) and int(row["attempts"]) < int(row["max_attempts"])
            if recovery_required or row["recovery_policy"] in {"RECONCILE_EXTERNAL", "NO_RETRY"}:
                status, due, finished = "RECOVERY_REQUIRED", row["due_at"], None
            elif can_retry:
                status = "RETRY_WAIT"
                try:
                    retry_policy = decode_task_payload("retry_policy", row["retry_policy_json"])
                except ValueError:
                    status, due, finished = "RECOVERY_REQUIRED", row["due_at"], None
                else:
                    due = _dt(self._ai_retry_due(int(row["attempts"]), now_dt, retry_policy))
                    finished = None
            else:
                status, due, finished = "FAILED", row["due_at"], now_text
            conn.execute(
                """UPDATE ai_tasks SET status = ?, due_at = ?, lease_owner = NULL,
                lease_until = NULL, last_error = ?, updated_at = ?, finished_at = ?,
                version = version + 1
                WHERE task_id = ? AND lease_token = ? AND lease_owner = ?""",
                (
                    status,
                    due,
                    str(error),
                    now_text,
                    finished,
                    int(task_id),
                    int(lease_token),
                    worker_id,
                ),
            )
            conn.execute(
                """UPDATE ai_task_attempts SET status = 'FAILED', error = ?,
                finished_at = ? WHERE task_id = ? AND lease_token = ?
                AND status = 'RUNNING'""",
                (str(error), now_text, int(task_id), int(lease_token)),
            )
            updated = conn.execute(
                "SELECT * FROM ai_tasks WHERE task_id = ?", (int(task_id),)
            ).fetchone()
            assert updated is not None
            self._audit_ai_task(
                conn,
                updated,
                "FAIL",
                from_status="RUNNING",
                to_status=status,
                actor_type="WORKER",
                actor_id=worker_id,
                details={"lease_token": lease_token, "error": str(error)},
                created_at=now_text,
            )
            if status == "FAILED":
                self._finish_task_workflow_sql(
                    conn,
                    updated,
                    status="FAILED",
                    error_code=status,
                    message=str(error),
                    now=now_text,
                )
            if status == "RETRY_WAIT":
                self._requeue_background_task_slot_sql(conn, updated, now=now_text)
            if status in {"FAILED", "RECOVERY_REQUIRED"}:
                self._settle_background_task_sql(
                    conn,
                    updated,
                    outcome=status,
                    error=str(error),
                    now=now_text,
                )
            return updated

        row = await self.uow.run(operation)
        return self._ai_task(row) if row else None
