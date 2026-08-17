"""MainCore run history and generic durable wakeup storage."""

from __future__ import annotations

from .support import (
    Any,
    RunStatus,
    WakeSource,
    Wakeup,
    WakeupStatus,
    _dt,
    _dump,
    _load,
    _now,
    datetime,
    sqlite3,
)


class CoreRunRecords:
    async def start_instance_run(
        self,
        profile_id: str,
        instance_id: str,
        source: WakeSource | str,
        *,
        reason: str = "",
        request: dict[str, Any] | None = None,
        expected_state_epoch: int | None = None,
    ) -> int:
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """INSERT INTO instance_core_runs(
                    profile_id, instance_id, source, status, reason, request_json,
                    expected_state_epoch, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    profile_id,
                    instance_id,
                    WakeSource(source).value,
                    RunStatus.RUNNING.value,
                    reason,
                    _dump(request or {}),
                    expected_state_epoch,
                    _dt(_now()),
                ),
            ),
            transaction=True,
        )
        return int(cursor.lastrowid)

    async def finish_instance_run(
        self,
        profile_id: str,
        instance_id: str,
        run_id: int,
        status: RunStatus,
        *,
        decision: dict[str, Any] | None = None,
        committed_state_epoch: int | None = None,
        error: str | None = None,
    ) -> bool:
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE instance_core_runs SET status = ?, decision_json = ?,
                    committed_state_epoch = ?, error = ?, finished_at = ?
                WHERE profile_id = ? AND instance_id = ? AND run_id = ?
                    AND status = ?""",
                (
                    status.value,
                    _dump(decision) if decision is not None else None,
                    committed_state_epoch,
                    error,
                    _dt(_now()),
                    profile_id,
                    instance_id,
                    run_id,
                    RunStatus.RUNNING.value,
                ),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1

    async def bind_instance_run_workflow(
        self,
        profile_id: str,
        instance_id: str,
        run_id: int,
        workflow_id: int,
    ) -> bool:
        """Attach the advanced-settings-visible AI workflow to its MainCore run."""

        def operation(conn: sqlite3.Connection) -> bool:
            workflow = conn.execute(
                """SELECT profile_id, instance_id FROM ai_workflows
                WHERE workflow_id = ?""",
                (int(workflow_id),),
            ).fetchone()
            if workflow is None:
                raise KeyError(("ai_workflow", int(workflow_id)))
            if (
                str(workflow["profile_id"]) != profile_id
                or str(workflow["instance_id"] or "") != instance_id
            ):
                raise ValueError("AI workflow does not belong to this character instance")
            cursor = conn.execute(
                """UPDATE instance_core_runs SET workflow_id = ?
                WHERE profile_id = ? AND instance_id = ? AND run_id = ?
                  AND (workflow_id IS NULL OR workflow_id = ?)""",
                (
                    int(workflow_id),
                    profile_id,
                    instance_id,
                    int(run_id),
                    int(workflow_id),
                ),
            )
            return cursor.rowcount == 1

        return bool(await self.uow.run(operation))

    async def get_instance_run_workflow(
        self,
        profile_id: str,
        instance_id: str,
        run_id: int,
    ) -> int | None:
        row = await self.db.fetch_one(
            """SELECT workflow_id FROM instance_core_runs
            WHERE profile_id = ? AND instance_id = ? AND run_id = ?""",
            (profile_id, instance_id, int(run_id)),
        )
        if row is None or row["workflow_id"] is None:
            return None
        return int(row["workflow_id"])

    async def list_instance_runs(
        self, profile_id: str, instance_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT * FROM instance_core_runs
            WHERE profile_id = ? AND instance_id = ?
            ORDER BY run_id DESC LIMIT ?""",
            (profile_id, instance_id, limit),
        )
        return [self._record(row, json_columns=("request_json", "decision_json")) for row in rows]

    async def get_previous_instance_run_context_message_ids(
        self,
        profile_id: str,
        instance_id: str,
        before_run_id: int,
    ) -> tuple[int, ...]:
        """Read the immediately preceding run's explicit inbound batch boundary."""

        row = await self.db.fetch_one(
            """SELECT request_json FROM instance_core_runs
            WHERE profile_id = ? AND instance_id = ? AND run_id < ?
            ORDER BY run_id DESC LIMIT 1""",
            (profile_id, instance_id, int(before_run_id)),
        )
        if row is None:
            return ()
        request = _load(row["request_json"]) or {}
        metadata = request.get("metadata") if isinstance(request, dict) else None
        values = metadata.get("context_message_ids", ()) if isinstance(metadata, dict) else ()
        result: list[int] = []
        for raw in values or ():
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value > 0 and value not in result:
                result.append(value)
        return tuple(result)

    async def schedule_instance_wakeup(
        self,
        profile_id: str,
        instance_id: str,
        source: WakeSource | str,
        due_at: datetime,
        reason: str = "",
        conversation_ref: str | None = None,
        idempotency_key: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> int:
        now = _dt(_now())
        source_value = WakeSource(source).value
        intent_kind = "PLUGIN_WAKE"

        def operation(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                """INSERT INTO instance_wakeups(
                    profile_id, instance_id, source, due_at, reason,
                    conversation_ref, idempotency_key, payload_json, status,
                    intent_kind, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id, instance_id, idempotency_key)
                    WHERE idempotency_key IS NOT NULL
                DO UPDATE SET due_at = excluded.due_at, reason = excluded.reason,
                    conversation_ref = excluded.conversation_ref,
                    payload_json = excluded.payload_json, status = 'PENDING',
                    lease_until = NULL, last_error = NULL,
                    updated_at = excluded.updated_at""",
                (
                    profile_id,
                    instance_id,
                    source_value,
                    _dt(due_at),
                    reason,
                    conversation_ref,
                    idempotency_key,
                    _dump(payload or {}),
                    WakeupStatus.PENDING.value,
                    intent_kind,
                    now,
                    now,
                ),
            )
            if not idempotency_key:
                return int(cursor.lastrowid)
            row = conn.execute(
                """SELECT wakeup_id FROM instance_wakeups
                WHERE profile_id = ? AND instance_id = ? AND idempotency_key = ?""",
                (profile_id, instance_id, idempotency_key),
            ).fetchone()
            return int(row[0])

        return await self.uow.run(operation)

    async def list_instance_wakeups(
        self, profile_id: str, instance_id: str, limit: int = 50
    ) -> list[Wakeup]:
        rows = await self.db.fetch_all(
            """SELECT * FROM instance_wakeups
            WHERE profile_id = ? AND instance_id = ?
            ORDER BY due_at DESC LIMIT ?""",
            (profile_id, instance_id, limit),
        )
        return [self._wakeup(row) for row in rows]


__all__ = ["CoreRunRecords"]
