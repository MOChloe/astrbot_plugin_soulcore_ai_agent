from __future__ import annotations

from dataclasses import dataclass

from .support import (
    AI_TASK_RETRY_HOURS,
    Any,
    decode_task_payload,
    encode_task_payload,
    sqlite3,
)
from .task_claim_queries import build_task_claim_query


@dataclass(frozen=True, slots=True)
class AiTaskCreationContext:
    workflow_id: int | None
    caused_by_workflow_id: int | None
    origin_work_node_id: int | None
    profile_id: str
    instance_id: str
    task_type: str
    task_class: str
    capability: str | None
    initial_status: str
    priority: int
    due: str
    step_key: str | None
    mutex_key: str | None
    backend_id: str | None
    idempotency_key: str | None
    generation: int
    input_data: dict[str, Any]
    checkpoint: dict[str, Any]
    retry_policy: dict[str, Any] | None
    recovery_policy: str
    max_attempts: int
    actor_type: str
    actor_id: str
    now: str


class AiTaskCreationTransaction:
    def __init__(self, owner: Any, context: AiTaskCreationContext) -> None:
        self.owner = owner
        self.context = context

    def __call__(self, conn: sqlite3.Connection) -> sqlite3.Row:
        existing = self._existing_task(conn)
        if existing is not None:
            return existing
        task_id = self._insert_task(conn)
        row = conn.execute("SELECT * FROM ai_tasks WHERE task_id = ?", (task_id,)).fetchone()
        assert row is not None
        context = self.context
        self.owner._audit_ai_task(
            conn,
            row,
            "CREATE",
            to_status=context.initial_status,
            actor_type=context.actor_type,
            actor_id=context.actor_id,
            created_at=context.now,
        )
        return row

    def _existing_task(self, conn: sqlite3.Connection) -> sqlite3.Row | None:
        context = self.context
        if context.idempotency_key is None:
            return None
        return conn.execute(
            """SELECT * FROM ai_tasks WHERE profile_id = ?
            AND instance_id = ?
            AND task_type = ? AND idempotency_key = ?
            ORDER BY generation DESC LIMIT 1""",
            (
                context.profile_id,
                context.instance_id,
                context.task_type,
                context.idempotency_key,
            ),
        ).fetchone()

    def _insert_task(self, conn: sqlite3.Connection) -> int:
        context = self.context
        cursor = conn.execute(
            """INSERT INTO ai_tasks(
                workflow_id, caused_by_workflow_id, origin_work_node_id,
                profile_id, instance_id, task_type, task_class, capability,
                status, priority, due_at,
                step_key, mutex_key, backend_id, idempotency_key, generation,
                input_json, checkpoint_json, retry_policy_json, recovery_policy,
                max_attempts, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                context.workflow_id,
                context.caused_by_workflow_id,
                context.origin_work_node_id,
                context.profile_id,
                context.instance_id,
                context.task_type,
                context.task_class,
                context.capability,
                context.initial_status,
                context.priority,
                context.due,
                context.step_key,
                context.mutex_key,
                context.backend_id,
                context.idempotency_key,
                context.generation,
                encode_task_payload("input", context.input_data),
                encode_task_payload("checkpoint", context.checkpoint),
                encode_task_payload(
                    "retry_policy",
                    context.retry_policy or {"delays_hours": list(AI_TASK_RETRY_HOURS)},
                ),
                context.recovery_policy,
                context.max_attempts,
                context.now,
                context.now,
            ),
        )
        return int(cursor.lastrowid)


@dataclass(frozen=True, slots=True)
class AiTaskClaimContext:
    worker_id: str
    limit: int
    lease_until: str
    now_text: str
    normalized_types: list[str]


class AiTaskClaimTransaction:
    def __init__(self, owner: Any, context: AiTaskClaimContext) -> None:
        self.owner = owner
        self.context = context

    def __call__(self, conn: sqlite3.Connection) -> list[sqlite3.Row]:
        claimed: list[sqlite3.Row] = []
        for _ in range(max(0, min(self.context.limit, 100))):
            row = self._select_task(conn)
            if row is None:
                break
            updated = self._claim_task(conn, row)
            if updated is not None:
                claimed.append(updated)
        return claimed

    def _select_task(self, conn: sqlite3.Connection) -> sqlite3.Row | None:
        context = self.context
        query = build_task_claim_query(
            context.now_text,
            context.normalized_types,
        )
        return conn.execute(query.sql, query.params).fetchone()

    def _claim_task(self, conn: sqlite3.Connection, row: sqlite3.Row) -> sqlite3.Row | None:
        context = self.context
        new_token = int(row["lease_token"]) + 1
        cursor = conn.execute(
            """UPDATE ai_tasks SET status = 'RUNNING',
            attempts = attempts + 1, lease_owner = ?, lease_token = ?,
            lease_until = ?, started_at = COALESCE(started_at, ?),
            updated_at = ?, last_error = NULL, version = version + 1
            WHERE task_id = ? AND status IN ('READY','SCHEDULED','RETRY_WAIT')""",
            (
                context.worker_id,
                new_token,
                context.lease_until,
                context.now_text,
                context.now_text,
                int(row["task_id"]),
            ),
        )
        if cursor.rowcount != 1:
            return None
        self._mark_background_author_running(conn, row)
        self._half_open_backend(conn, row)
        updated = conn.execute(
            "SELECT * FROM ai_tasks WHERE task_id = ?", (int(row["task_id"]),)
        ).fetchone()
        assert updated is not None
        self._record_attempt(conn, updated, new_token)
        self.owner._audit_ai_task(
            conn,
            updated,
            "CLAIM",
            from_status=row["status"],
            to_status="RUNNING",
            actor_type="WORKER",
            actor_id=context.worker_id,
            details={"lease_token": new_token},
            created_at=context.now_text,
        )
        return updated

    def _mark_background_author_running(
        self,
        conn: sqlite3.Connection,
        task: sqlite3.Row,
    ) -> None:
        if str(task["task_type"] or "").upper() != "BACKGROUND_AUTHOR":
            return
        payload = decode_task_payload("input", task["input_json"])
        changed = conn.execute(
            """UPDATE background_author_states
            SET status = 'RUNNING', last_started_at = ?,
                schedule_version = schedule_version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND author_kind = ?
              AND active_task_id = ? AND generation = ? AND status = 'ENQUEUED'""",
            (
                self.context.now_text,
                self.context.now_text,
                task["profile_id"],
                task["instance_id"],
                str(payload["author_kind"]),
                int(task["task_id"]),
                int(task["generation"]),
            ),
        ).rowcount
        if changed != 1:
            raise RuntimeError("background author claim lost its publication fence")

    def _half_open_backend(self, conn: sqlite3.Connection, row: sqlite3.Row) -> None:
        if not row["backend_id"]:
            return
        conn.execute(
            """UPDATE ai_backends SET circuit_state = CASE
                WHEN circuit_state = 'OPEN' THEN 'HALF_OPEN'
                ELSE circuit_state END, updated_at = ?
            WHERE backend_id = ?""",
            (self.context.now_text, row["backend_id"]),
        )

    def _record_attempt(
        self, conn: sqlite3.Connection, task: sqlite3.Row, lease_token: int
    ) -> None:
        context = self.context
        conn.execute(
            """INSERT INTO ai_task_attempts(
                task_id, attempt_no, lease_token, worker_id, status,
                started_at, heartbeat_at
            ) VALUES (?, ?, ?, ?, 'RUNNING', ?, ?)""",
            (
                task["task_id"],
                task["attempts"],
                lease_token,
                context.worker_id,
                context.now_text,
                context.now_text,
            ),
        )
