from __future__ import annotations

from ...timers.service import has_permanent_domain_cancel, settle_permanent_task_cancel
from .support import (
    AI_TASK_TERMINAL_STATUSES,
    KNOWLEDGE_TASK_TYPE,
    Any,
    _dt,
    _now,
    datetime,
    decode_task_payload,
    encode_task_payload,
    sqlite3,
)


class AiTaskLeaseSettlement:
    def _settle_finished_lease_sql(
        self,
        conn: sqlite3.Connection,
        previous: sqlite3.Row,
        updated: sqlite3.Row,
        *,
        target_status: str,
        lease_token: int,
        worker_id: str,
        terminal_reason: str,
        now: str,
        now_dt: datetime,
    ) -> None:
        self._audit_ai_task(
            conn,
            updated,
            target_status,
            from_status=previous["status"],
            to_status=target_status,
            actor_type="WORKER",
            actor_id=worker_id,
            details={"lease_token": lease_token, "reason": terminal_reason},
            created_at=now,
        )
        if target_status in {"DEFERRED", "SUCCEEDED"}:
            self._finish_task_workflow_sql(
                conn,
                updated,
                status="SUCCEEDED",
                error_code="",
                message="",
                now=now,
            )
            if target_status == "SUCCEEDED":
                self._clear_proactive_frame_cooldown_after_success_sql(
                    conn,
                    updated,
                    now=now,
                )
        elif target_status == "CANCELLED":
            self._finish_task_workflow_sql(
                conn,
                updated,
                status="CANCELLED",
                error_code="CANCELLED",
                message=terminal_reason or "任务已取消",
                now=now,
            )
            self._settle_background_task_sql(
                conn,
                updated,
                outcome="CANCELLED",
                error=terminal_reason or "任务已取消",
                now=now,
            )
        elif target_status == "PAUSED":
            self._requeue_background_task_slot_sql(conn, updated, now=now)
        if target_status == "SUCCEEDED" and updated["task_type"] == KNOWLEDGE_TASK_TYPE:
            self._refresh_knowledge_after_success(conn, updated, now_dt)

    def _refresh_knowledge_after_success(
        self,
        conn: sqlite3.Connection,
        task: sqlite3.Row,
        now_dt: datetime,
    ) -> None:
        knowledge_state = conn.execute(
            """SELECT desired_through_message_id, committed_through_message_id
            FROM knowledge_processing_state
            WHERE profile_id = ? AND instance_id = ?""",
            (task["profile_id"], task["instance_id"]),
        ).fetchone()
        self._refresh_knowledge_task_sql(
            conn,
            str(task["profile_id"]),
            str(task["instance_id"]),
            now_dt=now_dt,
            force=bool(
                knowledge_state
                and int(knowledge_state["desired_through_message_id"])
                > int(knowledge_state["committed_through_message_id"])
            ),
        )


class AiTaskCommands(AiTaskLeaseSettlement):
    async def _finish_ai_task_lease(
        self,
        task_id: int,
        lease_token: int,
        worker_id: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        checkpoint: dict[str, Any] | None = None,
        terminal_reason: str = "",
    ) -> bool:
        if status not in {"DEFERRED", "SUCCEEDED", "PAUSED", "CANCELLED", "RECOVERY_REQUIRED"}:
            raise ValueError("unsupported leased terminal status")
        expected = {
            # If a pause request wins the tiny interval after the executor has
            # returned, its externally visible work is already complete. The
            # terminal result must win; acknowledging PAUSED would replay the
            # side effect after resume.
            "DEFERRED": {"RUNNING", "PAUSE_REQUESTED"},
            "SUCCEEDED": {"RUNNING", "PAUSE_REQUESTED"},
            "PAUSED": {"PAUSE_REQUESTED"},
            "CANCELLED": {"CANCEL_REQUESTED"},
            "RECOVERY_REQUIRED": {"CANCEL_REQUESTED"},
        }[status]
        now_dt = _now()
        now = _dt(now_dt)

        def operation(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                "SELECT * FROM ai_tasks WHERE task_id = ?", (int(task_id),)
            ).fetchone()
            if (
                row is None
                or row["status"] not in expected
                or int(row["lease_token"]) != int(lease_token)
                or row["lease_owner"] != worker_id
            ):
                return False
            target_status = (
                "CANCELLED"
                if status == "RECOVERY_REQUIRED" and has_permanent_domain_cancel(conn, row)
                else status
            )
            finished = now if target_status in {"DEFERRED", "SUCCEEDED", "CANCELLED"} else None
            conn.execute(
                """UPDATE ai_tasks SET status = ?, result_json = COALESCE(?, result_json),
                checkpoint_json = COALESCE(?, checkpoint_json), lease_owner = NULL,
                lease_until = NULL, last_error = CASE WHEN ? = '' THEN last_error ELSE ? END,
                updated_at = ?, finished_at = ?,
                version = version + 1
                WHERE task_id = ? AND lease_token = ? AND lease_owner = ?""",
                (
                    target_status,
                    encode_task_payload("result", result) if result is not None else None,
                    encode_task_payload("checkpoint", checkpoint)
                    if checkpoint is not None
                    else None,
                    terminal_reason,
                    terminal_reason,
                    now,
                    finished,
                    int(task_id),
                    int(lease_token),
                    worker_id,
                ),
            )
            attempt_status = {
                "DEFERRED": "DEFERRED",
                "SUCCEEDED": "SUCCEEDED",
                "PAUSED": "PAUSED",
                "CANCELLED": "CANCELLED",
                "RECOVERY_REQUIRED": "RECOVERY_REQUIRED",
            }[target_status]
            conn.execute(
                """UPDATE ai_task_attempts SET status = ?, finished_at = ?
                WHERE task_id = ? AND lease_token = ? AND status = 'RUNNING'""",
                (attempt_status, now, int(task_id), int(lease_token)),
            )
            updated = conn.execute(
                "SELECT * FROM ai_tasks WHERE task_id = ?", (int(task_id),)
            ).fetchone()
            assert updated is not None
            self._settle_finished_lease_sql(
                conn,
                row,
                updated,
                target_status=target_status,
                lease_token=lease_token,
                worker_id=worker_id,
                terminal_reason=terminal_reason,
                now=now,
                now_dt=now_dt,
            )
            return True

        return await self.uow.run(operation)

    async def request_pause_ai_task(
        self,
        task_id: int,
        *,
        actor_id: str = "admin",
        reason: str = "",
        expected_version: int | None = None,
    ) -> dict[str, Any] | None:
        return await self._request_ai_control(
            task_id,
            "PAUSE",
            actor_id=actor_id,
            reason=reason,
            expected_version=expected_version,
        )

    async def acknowledge_pause_ai_task(
        self,
        task_id: int,
        lease_token: int,
        worker_id: str,
        *,
        checkpoint: dict[str, Any] | None = None,
    ) -> bool:
        return await self._finish_ai_task_lease(
            task_id, lease_token, worker_id, "PAUSED", checkpoint=checkpoint
        )

    async def resume_ai_task(
        self,
        task_id: int,
        *,
        actor_id: str = "admin",
        expected_version: int | None = None,
    ) -> dict[str, Any] | None:
        now = _dt(_now())
        return await self._simple_ai_transition(
            task_id,
            {"PAUSED"},
            "READY",
            actor_id=actor_id,
            action="RESUME",
            due_at=now,
            expected_version=expected_version,
        )

    async def request_cancel_ai_task(
        self,
        task_id: int,
        *,
        actor_id: str = "admin",
        reason: str = "",
        expected_version: int | None = None,
        settle_domain: bool = True,
    ) -> dict[str, Any] | None:
        return await self._request_ai_control(
            task_id,
            "CANCEL",
            actor_id=actor_id,
            reason=reason,
            expected_version=expected_version,
            settle_domain=settle_domain,
        )

    async def acknowledge_cancel_ai_task(
        self,
        task_id: int,
        lease_token: int,
        worker_id: str,
        *,
        recovery_required: bool = False,
    ) -> bool:
        return await self._finish_ai_task_lease(
            task_id,
            lease_token,
            worker_id,
            "RECOVERY_REQUIRED" if recovery_required else "CANCELLED",
        )

    @staticmethod
    def _retry_image_ids(row: sqlite3.Row) -> list[str] | None:
        try:
            input_data = decode_task_payload("input", row["input_json"])
        except ValueError:
            return None
        return list(
            dict.fromkeys(
                str(item or "").strip()
                for item in (input_data.get("image_asset_ids") or [])
                if str(item or "").strip()
            )
        )

    @staticmethod
    def _retry_images_available(
        conn: sqlite3.Connection, job: sqlite3.Row, image_ids: list[str]
    ) -> bool:
        for asset_id in image_ids:
            asset = conn.execute(
                """SELECT 1 FROM media_assets WHERE asset_id = ?
                AND profile_id = ? AND instance_id = ?
                AND file_status = 'AVAILABLE' AND mime_type LIKE 'image/%'""",
                (asset_id, job["profile_id"], job["instance_id"]),
            ).fetchone()
            if asset is None:
                return False
        return True

    @staticmethod
    def _remove_retry_failure_todo(
        conn: sqlite3.Connection, job: sqlite3.Row, todo: sqlite3.Row | None
    ) -> bool:
        if todo is None:
            return True
        valid = (
            str(todo["kind"]) == "FILE_FAILED"
            and str(todo["status"]) in {"PENDING", "CANCELLED"}
            and todo["delivery_outbox_id"] is None
            and todo["selected_run_id"] is None
        )
        if not valid:
            return False
        conn.execute(
            """DELETE FROM instance_wakeups WHERE profile_id = ?
            AND instance_id = ? AND idempotency_key = ?
            AND status IN ('PENDING','CANCELLED','DEAD')""",
            (
                job["profile_id"],
                job["instance_id"],
                f"important-todo:{todo['todo_id']}",
            ),
        )
        conn.execute("DELETE FROM important_todos WHERE todo_id = ?", (todo["todo_id"],))
        return True

    @staticmethod
    def _restore_retry_image_holds(
        conn: sqlite3.Connection, job: sqlite3.Row, image_ids: list[str], now: str
    ) -> None:
        for asset_id in image_ids:
            conn.execute(
                """INSERT INTO media_retention_holds(
                    profile_id, instance_id, asset_id, holder_kind,
                    holder_id, created_at
                ) VALUES (?, ?, ?, 'FILE_GENERATION_JOB', ?, ?)
                ON CONFLICT(asset_id, holder_kind, holder_id) DO UPDATE SET
                    released_at = NULL""",
                (job["profile_id"], job["instance_id"], asset_id, job["job_id"], now),
            )

    def _prepare_file_artifact_retry(
        self, conn: sqlite3.Connection, row: sqlite3.Row, now: str
    ) -> bool:
        job = conn.execute(
            "SELECT * FROM file_generation_jobs WHERE ai_task_id = ?",
            (int(row["task_id"]),),
        ).fetchone()
        if job is None or str(job["status"]) == "SUCCEEDED":
            return False
        bound = conn.execute(
            """SELECT status FROM main_core_work_file_bindings
            WHERE job_id = ?""",
            (job["job_id"],),
        ).fetchone()
        if bound is not None and str(bound["status"]) != "PENDING":
            # A trusted callback is already committed.  Retrying this generation
            # would need a newly authorized work generation, not mutation of the
            # callback that may already be claimed by a recovery run.
            return False
        image_ids = self._retry_image_ids(row)
        if image_ids is None or not self._retry_images_available(conn, job, image_ids):
            return False
        todo = conn.execute(
            "SELECT * FROM important_todos WHERE source_job_id = ?", (job["job_id"],)
        ).fetchone()
        if not self._remove_retry_failure_todo(conn, job, todo):
            return False
        self._restore_retry_image_holds(conn, job, image_ids, now)
        conn.execute(
            """UPDATE file_generation_jobs SET status = 'QUEUED',
            safe_error_code = '', safe_error_message = '', finished_at = NULL,
            updated_at = ?, version = version + 1 WHERE job_id = ?""",
            (now, job["job_id"]),
        )
        return True

    def _manual_retry_ai_task_sql(
        self,
        conn: sqlite3.Connection,
        task_id: int,
        *,
        actor_id: str,
        expected_version: int | None,
        now: str,
    ) -> sqlite3.Row | None:
        row = conn.execute("SELECT * FROM ai_tasks WHERE task_id = ?", (int(task_id),)).fetchone()
        # RECOVERY_REQUIRED means an external side effect may already have
        # happened. It must be explicitly reconciled/cancelled, never replayed.
        if row is None or row["status"] not in {"FAILED", "CANCELLED"}:
            return None
        if expected_version is not None and int(row["version"]) != int(expected_version):
            return None
        if str(
            row["task_type"] or ""
        ).upper() == "FILE_ARTIFACT_GENERATION" and not self._prepare_file_artifact_retry(
            conn, row, now
        ):
            return None
        # A background-author task is one generation of a domain-owned slot.
        # Once terminal, that slot is already released for the materializer to
        # create a fresh generation. Replaying the old task cannot satisfy the
        # slot binding and may use stale activity/configuration snapshots.
        if str(row["task_type"] or "").upper() == "BACKGROUND_AUTHOR":
            return None
        conn.execute(
            """UPDATE ai_tasks SET status = 'READY', due_at = ?,
            generation = generation + 1, attempts = 0,
            lease_owner = NULL, lease_until = NULL, result_json = NULL,
            last_error = NULL, finished_at = NULL, updated_at = ?,
            version = version + 1 WHERE task_id = ? AND version = ?""",
            (now, now, int(task_id), int(row["version"])),
        )
        updated = conn.execute(
            "SELECT * FROM ai_tasks WHERE task_id = ?", (int(task_id),)
        ).fetchone()
        assert updated is not None
        self._audit_ai_task(
            conn,
            updated,
            "MANUAL_RETRY",
            from_status=row["status"],
            to_status="READY",
            actor_type="ADMIN",
            actor_id=actor_id,
            created_at=now,
        )
        return updated

    async def manual_retry_ai_task(
        self,
        task_id: int,
        *,
        actor_id: str = "admin",
        expected_version: int | None = None,
    ) -> dict[str, Any] | None:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row | None:
            return self._manual_retry_ai_task_sql(
                conn,
                task_id,
                actor_id=actor_id,
                expected_version=expected_version,
                now=now,
            )

        row = await self.uow.run(operation)
        return self._ai_task(row) if row else None

    async def expedite_ai_task(
        self, task_id: int, *, actor_id: str = "admin"
    ) -> dict[str, Any] | None:
        """Make a queued/retrying/paused task eligible to run immediately."""

        now = _dt(_now())
        return await self._simple_ai_transition(
            task_id,
            {"SCHEDULED", "READY", "RETRY_WAIT", "PAUSED"},
            "READY",
            actor_id=actor_id,
            action="EXPEDITE",
            due_at=now,
        )

    @staticmethod
    def _requested_control_target(status: str, control: str) -> str | None:
        if control == "PAUSE":
            if status in {"RUNNING", "CANCEL_REQUESTED"}:
                return "PAUSE_REQUESTED" if status == "RUNNING" else status
            return status if status == "PAUSE_REQUESTED" else "PAUSED"
        if status == "CANCEL_REQUESTED":
            return None
        return "CANCEL_REQUESTED" if status in {"RUNNING", "PAUSE_REQUESTED"} else "CANCELLED"

    def _audit_permanent_domain_cancel(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        now_dt: datetime,
        *,
        permanent_cancel: bool,
        actor_id: str,
        reason: str,
        created_at: str,
    ) -> bool:
        if not permanent_cancel or not settle_permanent_task_cancel(conn, row, now_dt):
            return False
        self._audit_ai_task(
            conn,
            row,
            "REQUEST_CANCEL",
            from_status=row["status"],
            to_status=row["status"],
            actor_type="ADMIN",
            actor_id=actor_id,
            details={"reason": reason, "permanent_domain_cancel": True},
            created_at=created_at,
        )
        return True

    async def _request_ai_control(
        self,
        task_id: int,
        control: str,
        *,
        actor_id: str,
        reason: str,
        expected_version: int | None = None,
        settle_domain: bool = False,
    ) -> dict[str, Any] | None:
        now_dt = _now()
        now = _dt(now_dt)
        control = control.upper()

        def operation(conn: sqlite3.Connection) -> sqlite3.Row | None:
            row = conn.execute(
                "SELECT * FROM ai_tasks WHERE task_id = ?", (int(task_id),)
            ).fetchone()
            if row is None or (
                expected_version is not None and int(row["version"]) != int(expected_version)
            ):
                return None
            permanent_cancel = control == "CANCEL" and bool(settle_domain)
            if row["status"] in AI_TASK_TERMINAL_STATUSES:
                self._audit_permanent_domain_cancel(
                    conn,
                    row,
                    now_dt,
                    permanent_cancel=permanent_cancel,
                    actor_id=actor_id,
                    reason=reason,
                    created_at=now,
                )
                return None
            target = self._requested_control_target(str(row["status"]), control)
            if target is None:
                self._audit_permanent_domain_cancel(
                    conn,
                    row,
                    now_dt,
                    permanent_cancel=permanent_cancel,
                    actor_id=actor_id,
                    reason=reason,
                    created_at=now,
                )
                return row
            finished = now if target == "CANCELLED" else None
            conn.execute(
                """UPDATE ai_tasks SET status = ?, last_error = ?,
                lease_owner = CASE WHEN ? IN ('PAUSED','CANCELLED') THEN NULL ELSE lease_owner END,
                lease_until = CASE WHEN ? IN ('PAUSED','CANCELLED') THEN NULL ELSE lease_until END,
                updated_at = ?, finished_at = COALESCE(?, finished_at),
                version = version + 1
                WHERE task_id = ? AND version = ?""",
                (
                    target,
                    str(reason),
                    target,
                    target,
                    now,
                    finished,
                    int(task_id),
                    int(row["version"]),
                ),
            )
            updated = conn.execute(
                "SELECT * FROM ai_tasks WHERE task_id = ?", (int(task_id),)
            ).fetchone()
            assert updated is not None
            if permanent_cancel:
                settle_permanent_task_cancel(conn, updated, now_dt)
            if target == "CANCELLED":
                self._finish_task_workflow_sql(
                    conn,
                    updated,
                    status="CANCELLED",
                    error_code="CANCELLED",
                    message=str(reason or "任务已取消"),
                    now=now,
                )
                self._settle_background_task_sql(
                    conn,
                    updated,
                    outcome="CANCELLED",
                    error=str(reason or "任务已取消"),
                    now=now,
                )
            self._audit_ai_task(
                conn,
                updated,
                f"REQUEST_{control}",
                from_status=row["status"],
                to_status=target,
                actor_type="ADMIN",
                actor_id=actor_id,
                details={
                    "reason": reason,
                    "permanent_domain_cancel": permanent_cancel,
                },
                created_at=now,
            )
            return updated

        row = await self.uow.run(operation)
        return self._ai_task(row) if row else None

    async def _simple_ai_transition(
        self,
        task_id: int,
        allowed: set[str],
        target: str,
        *,
        actor_id: str,
        action: str,
        due_at: str | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any] | None:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row | None:
            row = conn.execute(
                "SELECT * FROM ai_tasks WHERE task_id = ?", (int(task_id),)
            ).fetchone()
            if (
                row is None
                or row["status"] not in allowed
                or (expected_version is not None and int(row["version"]) != int(expected_version))
            ):
                return None
            conn.execute(
                """UPDATE ai_tasks SET status = ?, due_at = COALESCE(?, due_at),
                lease_owner = NULL, lease_until = NULL, updated_at = ?,
                version = version + 1
                WHERE task_id = ? AND version = ?""",
                (target, due_at, now, int(task_id), int(row["version"])),
            )
            updated = conn.execute(
                "SELECT * FROM ai_tasks WHERE task_id = ?", (int(task_id),)
            ).fetchone()
            assert updated is not None
            self._audit_ai_task(
                conn,
                updated,
                action,
                from_status=row["status"],
                to_status=target,
                actor_type="ADMIN",
                actor_id=actor_id,
                created_at=now,
            )
            return updated

        row = await self.uow.run(operation)
        return self._ai_task(row) if row else None

    async def release_ai_task(
        self,
        task_id: int,
        lease_token: int,
        worker_id: str,
        *,
        reason: str = "worker_stopped",
        due_at: datetime | None = None,
    ) -> bool:
        now_dt = _now()
        now = _dt(now_dt)
        release_due = due_at if due_at is not None else now_dt
        target_status = "SCHEDULED" if release_due > now_dt else "READY"
        due_text = _dt(release_due)

        def operation(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                "SELECT * FROM ai_tasks WHERE task_id = ?", (int(task_id),)
            ).fetchone()
            if (
                row is None
                or row["status"] != "RUNNING"
                or int(row["lease_token"]) != int(lease_token)
                or row["lease_owner"] != worker_id
            ):
                return False
            conn.execute(
                """UPDATE ai_tasks SET status = ?, attempts = MAX(0, attempts - 1),
                due_at = ?, lease_owner = NULL, lease_until = NULL,
                last_error = ?, updated_at = ?, version = version + 1
                WHERE task_id = ?
                AND status = 'RUNNING' AND lease_token = ? AND lease_owner = ?""",
                (
                    target_status,
                    due_text,
                    reason,
                    now,
                    int(task_id),
                    int(lease_token),
                    worker_id,
                ),
            )
            conn.execute(
                """UPDATE ai_task_attempts SET status = 'RELEASED', error = ?,
                finished_at = ? WHERE task_id = ? AND lease_token = ?
                AND status = 'RUNNING'""",
                (reason, now, int(task_id), int(lease_token)),
            )
            self._requeue_background_task_slot_sql(conn, row, now=now)
            updated = conn.execute(
                "SELECT * FROM ai_tasks WHERE task_id = ?", (int(task_id),)
            ).fetchone()
            assert updated is not None
            self._audit_ai_task(
                conn,
                updated,
                "RELEASE",
                from_status=str(row["status"]),
                to_status=target_status,
                actor_type="WORKER",
                actor_id=worker_id,
                details={"reason": reason, "due_at": due_text},
                created_at=now,
            )
            return True

        return await self.uow.run(operation)

    def _expired_ai_task_transition(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        current: datetime,
        now: str,
    ) -> tuple[str, Any, str | None]:
        due = row["due_at"]
        if row["status"] == "CANCEL_REQUESTED":
            if has_permanent_domain_cancel(conn, row) or (
                str(row["task_type"] or "").upper() == "FILE_ARTIFACT_GENERATION"
            ):
                return "CANCELLED", due, now
            return "RECOVERY_REQUIRED", due, None
        if row["status"] == "PAUSE_REQUESTED":
            return "PAUSED", due, None
        if row["recovery_policy"] in {"RECONCILE_EXTERNAL", "NO_RETRY"}:
            return "RECOVERY_REQUIRED", due, None
        if int(row["attempts"]) >= int(row["max_attempts"]):
            return "FAILED", due, now
        try:
            retry_policy = decode_task_payload("retry_policy", row["retry_policy_json"])
        except ValueError:
            return "RECOVERY_REQUIRED", due, None
        retry_due = self._ai_retry_due(int(row["attempts"]), current, retry_policy)
        return "RETRY_WAIT", _dt(retry_due), None

    def _recover_expired_ai_task_sql(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        current: datetime,
        now: str,
    ) -> sqlite3.Row:
        status, due, finished = self._expired_ai_task_transition(
            conn,
            row,
            current=current,
            now=now,
        )
        conn.execute(
            """UPDATE ai_tasks SET status = ?, due_at = ?,
            lease_owner = NULL, lease_until = NULL,
            last_error = 'lease_expired_worker_crash', updated_at = ?,
            finished_at = ?, version = version + 1
            WHERE task_id = ? AND lease_token = ?""",
            (
                status,
                due,
                now,
                finished,
                int(row["task_id"]),
                int(row["lease_token"]),
            ),
        )
        conn.execute(
            """UPDATE ai_task_attempts SET status = 'CRASHED',
            error = 'lease_expired_worker_crash', finished_at = ?
            WHERE task_id = ? AND lease_token = ? AND status = 'RUNNING'""",
            (now, int(row["task_id"]), int(row["lease_token"])),
        )
        if status in {"RETRY_WAIT", "PAUSED"}:
            self._requeue_background_task_slot_sql(conn, row, now=now)
        updated = conn.execute(
            "SELECT * FROM ai_tasks WHERE task_id = ?",
            (int(row["task_id"]),),
        ).fetchone()
        assert updated is not None
        self._audit_ai_task(
            conn,
            updated,
            "RECOVER_EXPIRED_LEASE",
            from_status=row["status"],
            to_status=status,
            details={"stale_lease_token": row["lease_token"]},
            created_at=now,
        )
        if status in {"FAILED", "CANCELLED"}:
            workflow_status = "CANCELLED" if status == "CANCELLED" else "FAILED"
            self._finish_task_workflow_sql(
                conn,
                updated,
                status=workflow_status,
                error_code=status,
                message="lease_expired_worker_crash",
                now=now,
            )
        if status in {"FAILED", "CANCELLED", "RECOVERY_REQUIRED"}:
            self._settle_background_task_sql(
                conn,
                updated,
                outcome=status,
                error="lease_expired_worker_crash",
                now=now,
            )
        return updated

    async def recover_expired_ai_tasks(
        self,
        *,
        now: datetime | None = None,
        current_worker_id: str | None = None,
    ) -> list[dict[str, Any]]:
        current = now or _now()
        now_text = _dt(current)

        def operation(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            rows = conn.execute(
                """SELECT * FROM ai_tasks WHERE status IN (
                'RUNNING','PAUSE_REQUESTED','CANCEL_REQUESTED'
            ) AND (
                lease_owner IS NULL OR lease_until IS NULL OR lease_until <= ?
                OR (? IS NOT NULL AND lease_owner <> ?)
            ) ORDER BY task_id""",
                (now_text, current_worker_id, current_worker_id),
            )
            return [
                self._recover_expired_ai_task_sql(
                    conn,
                    row,
                    current=current,
                    now=now_text,
                )
                for row in rows
            ]

        rows = await self.uow.run(operation)
        return [self._ai_task(row) for row in rows]

    async def list_ai_task_attempts(
        self, task_id: int, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT * FROM ai_task_attempts WHERE task_id = ?
            ORDER BY attempt_id DESC LIMIT ?""",
            (int(task_id), max(1, min(int(limit), 1000))),
        )
        return [self._record(row, json_columns=("metrics_json",)) for row in rows]
