from __future__ import annotations

from .support import (
    Any,
    _dt,
    _now,
    datetime,
    sqlite3,
    timedelta,
)


def _record_declared_backend_health_sql(
    conn: sqlite3.Connection,
    *,
    backend_id: str,
    status: str,
    state: str,
    failure_count: int,
    next_probe_at: Any,
    error: str,
    now: str,
) -> sqlite3.Row:
    conn.execute(
        """UPDATE ai_backends SET health_status = ?,
        circuit_state = ?, consecutive_failures = ?,
        total_failures = total_failures + 1,
        opened_at = CASE WHEN ? = 'OPEN'
            THEN COALESCE(opened_at, ?) ELSE opened_at END,
        next_probe_at = ?, last_failure_at = ?, last_error = ?,
        version = version + 1, updated_at = ? WHERE backend_id = ?""",
        (
            "UNHEALTHY" if status != "HALF_OPEN" else "UNKNOWN",
            state,
            failure_count,
            state,
            now,
            next_probe_at,
            now,
            error,
            now,
            backend_id,
        ),
    )
    row = conn.execute(
        "SELECT * FROM ai_backends WHERE backend_id = ?",
        (backend_id,),
    ).fetchone()
    assert row is not None
    return row


class AiTelemetryRecords:
    async def list_ai_task_audit(
        self,
        *,
        task_id: int | None = None,
        profile_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses = ["1 = 1"]
        params: list[Any] = []
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(int(task_id))
        if profile_id is not None:
            clauses.append("profile_id = ?")
            params.append(profile_id)
        params.append(max(1, min(int(limit), 1000)))
        rows = await self.db.fetch_all(
            f"""SELECT * FROM ai_task_audit WHERE {" AND ".join(clauses)}
            ORDER BY audit_id DESC LIMIT ?""",
            params,
        )
        return [self._record(row, json_columns=("details_json",)) for row in rows]

    async def cleanup_ai_task_history(self, *, now: datetime | None = None) -> dict[str, Any]:
        current = now or _now()
        success_before = _dt(current - timedelta(days=30))
        failure_before = _dt(current - timedelta(days=90))
        workflow_expiry = _dt(current)

        def operation(conn: sqlite3.Connection) -> dict[str, Any]:
            workflow_cursor = conn.execute(
                "DELETE FROM ai_workflows WHERE expires_at <= ?", (workflow_expiry,)
            )
            task_cursor = conn.execute(
                """DELETE FROM ai_tasks WHERE
                (status = 'SUCCEEDED' AND finished_at < ?)
                OR (status IN ('DEFERRED','FAILED','CANCELLED') AND finished_at < ?)""",
                (success_before, failure_before),
            )
            return {
                "deleted_tasks": int(task_cursor.rowcount),
                "deleted_workflows": int(workflow_cursor.rowcount),
                "workflow_retention_days": 30,
                "success_retention_days": 30,
                "failure_retention_days": 90,
            }

        return await self.uow.run(operation)

    async def recover_orphaned_ai_invocations(self) -> int:
        """Close workflow rows left running by a process crash."""

        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            conn.execute(
                """UPDATE ai_provider_attempts SET status = 'INTERRUPTED',
                error_code = 'PROCESS_RESTART',
                error_message = CASE WHEN sent_at IS NULL
                    THEN '接口请求尚未发送，处理被进程重启中断'
                    ELSE '接口请求已经发送，但响应因进程重启而丢失' END,
                finished_at = ? WHERE status IN ('PREPARING', 'IN_FLIGHT')""",
                (now,),
            )
            conn.execute(
                """UPDATE ai_work_nodes SET status = 'INTERRUPTED',
                error_code = 'PROCESS_RESTART', error_message = '处理阶段被进程重启中断',
                finished_at = ? WHERE status = 'RUNNING'""",
                (now,),
            )
            cursor = conn.execute(
                """UPDATE ai_workflows SET status = 'INTERRUPTED',
                final_error_code = 'PROCESS_RESTART',
                final_message = '处理被进程重启中断',
                finished_at = ?, updated_at = ?, version = version + 1
                WHERE status = 'RUNNING'""",
                (now, now),
            )
            return int(cursor.rowcount)

        return await self.uow.run(operation)

    async def record_ai_backend_health(self, record: dict[str, Any]) -> dict[str, Any]:
        backend_id = str(record.get("backend_id") or "").strip()
        if not backend_id:
            raise ValueError("backend_id is required")
        current = await self._ensure_ai_backend_for_health(backend_id, record)
        status = self._ai_backend_health_status(record)
        if status in {"HEALTHY", "SUCCEEDED", "OK"}:
            return await self.record_ai_backend_success(backend_id)
        if status in {"UNHEALTHY", "FAILED", "ERROR"}:
            return await self.record_ai_backend_failure(
                backend_id, str(record.get("error") or "backend health failure")
            )
        if status in {"OPEN", "HALF_OPEN", "DEGRADED", "DISABLED"}:
            return await self._record_declared_ai_backend_health(backend_id, status, record)
        return current

    async def _ensure_ai_backend_for_health(
        self,
        backend_id: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        current = await self.get_ai_backend(backend_id)
        if current is not None:
            return current
        return await self.upsert_ai_backend(
            backend_id,
            str(record.get("backend_kind") or "MODEL"),
            display_name=str(record.get("display_name") or backend_id),
            metadata=dict(record.get("metadata") or {}),
        )

    @staticmethod
    def _ai_backend_health_status(record: dict[str, Any]) -> str:
        return str(record.get("status") or "").upper()

    async def _record_declared_ai_backend_health(
        self,
        backend_id: str,
        status: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        state_by_status = {"OPEN": "OPEN", "HALF_OPEN": "HALF_OPEN"}
        state = state_by_status.get(status, "CLOSED")
        circuit = dict(record.get("circuit") or {})
        now = _dt(_now())
        row = await self.uow.run(
            lambda conn: _record_declared_backend_health_sql(
                conn,
                backend_id=backend_id,
                status=status,
                state=state,
                failure_count=max(0, int(record.get("failure_count") or 0)),
                next_probe_at=circuit.get("opened_until"),
                error=str(record.get("error") or ""),
                now=now,
            )
        )
        return self._ai_backend(row)

    async def record_ai_circuit_health(self, record: dict[str, Any]) -> dict[str, Any]:
        """Persist one adapter/backend/credential/capability circuit scope."""

        scope = str(record.get("circuit_scope") or "").strip()
        backend_id = str(record.get("backend_id") or "").strip()
        if not scope or not backend_id:
            raise ValueError("circuit_scope and backend_id are required")
        state = str(record.get("state") or "HEALTHY").upper()
        if state not in {"HEALTHY", "DEGRADED", "OPEN", "HALF_OPEN", "DISABLED"}:
            raise ValueError("invalid circuit state")
        now = _dt(_now())
        opened_until = record.get("opened_until")
        if isinstance(opened_until, datetime):
            opened_until = _dt(opened_until)
        last_success_at = record.get("last_success_at")
        if isinstance(last_success_at, datetime):
            last_success_at = _dt(last_success_at)
        last_failure_at = record.get("last_failure_at")
        if isinstance(last_failure_at, datetime):
            last_failure_at = _dt(last_failure_at)

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            conn.execute(
                """INSERT INTO ai_circuit_states(
                    circuit_scope, backend_id, adapter_id, credential_id,
                    capability, state, failure_count, opened_until,
                    last_error_code, last_success_at, last_failure_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(circuit_scope) DO UPDATE SET
                    backend_id = excluded.backend_id,
                    adapter_id = excluded.adapter_id,
                    credential_id = excluded.credential_id,
                    capability = excluded.capability,
                    state = excluded.state,
                    failure_count = excluded.failure_count,
                    opened_until = excluded.opened_until,
                    last_error_code = excluded.last_error_code,
                    last_success_at = excluded.last_success_at,
                    last_failure_at = excluded.last_failure_at,
                    updated_at = excluded.updated_at""",
                (
                    scope,
                    backend_id,
                    str(record.get("adapter_id") or ""),
                    str(record.get("credential_id") or ""),
                    str(record.get("capability") or ""),
                    state,
                    max(0, int(record.get("failure_count") or 0)),
                    opened_until,
                    str(record.get("last_error_code") or ""),
                    last_success_at,
                    last_failure_at,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM ai_circuit_states WHERE circuit_scope = ?",
                (scope,),
            ).fetchone()
            assert row is not None
            return row

        return self._record(await self.uow.run(operation), json_columns=())

    async def list_ai_circuit_states(
        self, *, backend_id: str | None = None
    ) -> list[dict[str, Any]]:
        if backend_id is None:
            rows = await self.db.fetch_all(
                "SELECT * FROM ai_circuit_states ORDER BY updated_at DESC"
            )
        else:
            rows = await self.db.fetch_all(
                """SELECT * FROM ai_circuit_states WHERE backend_id = ?
                ORDER BY updated_at DESC""",
                (backend_id,),
            )
        return [self._record(row, json_columns=()) for row in rows]

    async def clear_web_research_circuits(self) -> int:
        """Remove obsolete cross-request breakers for web capabilities."""

        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            backend_rows = conn.execute(
                "SELECT backend_id FROM ai_backends WHERE backend_kind = 'WEB_RESEARCH'"
            ).fetchall()
            backend_ids = [str(row["backend_id"]) for row in backend_rows]
            if not backend_ids:
                return 0
            placeholders = ",".join("?" for _ in backend_ids)
            cursor = conn.execute(
                f"DELETE FROM ai_circuit_states WHERE backend_id IN ({placeholders})",
                backend_ids,
            )
            conn.execute(
                f"""UPDATE ai_backends SET health_status = 'UNKNOWN',
                circuit_state = 'CLOSED', consecutive_failures = 0,
                opened_at = NULL, next_probe_at = NULL, last_error = '',
                version = version + 1, updated_at = ?
                WHERE backend_id IN ({placeholders}) AND (
                    health_status != 'UNKNOWN' OR circuit_state != 'CLOSED'
                    OR consecutive_failures != 0 OR opened_at IS NOT NULL
                    OR next_probe_at IS NOT NULL OR last_error != ''
                )""",
                (now, *backend_ids),
            )
            return max(0, int(cursor.rowcount))

        return int(await self.uow.run(operation))
