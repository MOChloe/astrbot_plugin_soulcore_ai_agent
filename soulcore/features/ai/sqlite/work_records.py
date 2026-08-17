from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from ....contracts.ai_models import AIWorkPurpose
from ..work_taxonomy import (
    AIWorkNodeKind,
    AIWorkNodeRole,
    durable_task_owns_workflow,
    normalize_work_purpose,
    work_purpose_spec,
)
from .support import _dt, _dump, _load, _now
from .work_record_projection import (
    model_visible_summary_coverage,
    project_model_visible_message_ids,
)
from .work_record_queries import AiWorkRecordQueries


def coherent_workflow_terminal(
    conn: Any,
    workflow_id: int,
    status: str,
    error_code: str,
    message: str,
) -> tuple[str, str, str]:
    """Prevent a failed business stage from being summarized as success."""

    if status != "SUCCEEDED":
        return status, error_code, message
    failed = conn.execute(
        """SELECT error_code, error_message, summary FROM ai_work_nodes
        WHERE workflow_id = ? AND status = 'FAILED'
        ORDER BY sequence DESC LIMIT 1""",
        (int(workflow_id),),
    ).fetchone()
    if failed is None:
        return status, error_code, message
    return (
        "FAILED",
        str(failed["error_code"] or "FAILED_CHILD_STAGE"),
        str(failed["error_message"] or failed["summary"] or "AI 工作阶段失败"),
    )


_WORKFLOW_TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"}
_NODE_TERMINAL = {
    "SUCCEEDED",
    "SKIPPED",
    "FALLBACK",
    "FAILED",
    "CANCELLED",
    "INTERRUPTED",
}

_ATTEMPT_TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"}


def _public_ref(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _workflow_status(value: Any) -> str:
    return str(value or "SUCCEEDED").upper()


def _node_status(value: Any) -> str:
    return str(value or "SUCCEEDED").upper()


def _as_utc_text(value: Any) -> str:
    if isinstance(value, datetime):
        return _dt(value)
    return _dt(_now())


class AiWorkRecords(AiWorkRecordQueries):
    """Causal AI work persistence with immutable terminal transitions."""

    if TYPE_CHECKING:
        db: Any
        uow: Any

        def _record(self, row: Any, *, json_columns: tuple[str, ...]) -> dict[str, Any]: ...

    # ------------------------------------------------------------------
    # Workflow lifecycle

    async def create_ai_workflow(self, record: Mapping[str, Any]) -> dict[str, Any]:
        started = record.get("started_at")
        started_at = started if isinstance(started, datetime) else _now()
        started_text = _dt(started_at)
        profile_id = str(record.get("profile_id") or "default")
        idempotency_key = str(record.get("idempotency_key") or "").strip()
        if not idempotency_key:
            raise ValueError("AI workflow idempotency_key is required")
        if not record.get("primary_purpose"):
            raise ValueError("AI workflow primary_purpose is required")
        purpose = normalize_work_purpose(record.get("primary_purpose"))

        def operation(conn: Any) -> Any:
            conn.execute(
                """INSERT INTO ai_workflows(
                    public_ref, profile_id, instance_id, workflow_kind,
                    primary_purpose, trigger_kind, trigger_ref,
                    caused_by_workflow_id, reason, idempotency_key,
                    started_at, expires_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id, idempotency_key) DO NOTHING""",
                (
                    _public_ref("work"),
                    profile_id,
                    str(record.get("instance_id") or "") or None,
                    str(record.get("workflow_kind") or "BACKGROUND").upper(),
                    purpose.value,
                    str(record.get("trigger_kind") or "SYSTEM").upper(),
                    str(record.get("trigger_ref") or ""),
                    int(record["caused_by_workflow_id"])
                    if record.get("caused_by_workflow_id") is not None
                    else None,
                    str(record.get("reason") or work_purpose_spec(purpose).reason),
                    idempotency_key,
                    started_text,
                    _dt(started_at + timedelta(days=30)),
                    started_text,
                ),
            )
            row = conn.execute(
                "SELECT * FROM ai_workflows WHERE profile_id = ? AND idempotency_key = ?",
                (profile_id, idempotency_key),
            ).fetchone()
            assert row is not None
            return row

        return self._record(await self.uow.run(operation), json_columns=())

    async def ensure_ai_task_workflow(self, task_id: int) -> dict[str, Any] | None:
        now = _now()
        now_text = _dt(now)

        def operation(conn: Any) -> Any:
            task = conn.execute(
                "SELECT * FROM ai_tasks WHERE task_id = ?", (int(task_id),)
            ).fetchone()
            if task is None:
                return None
            if durable_task_owns_workflow(str(task["task_type"] or "")):
                return None
            predecessor_workflow_id: int | None = None
            if task["workflow_id"] is not None:
                predecessor_workflow_id = int(task["workflow_id"])
                existing = conn.execute(
                    "SELECT * FROM ai_workflows WHERE workflow_id = ?",
                    (predecessor_workflow_id,),
                ).fetchone()
                if existing is not None and str(existing["status"]) == "RUNNING":
                    return existing
                conn.execute(
                    """UPDATE ai_tasks SET workflow_id = NULL,
                    updated_at = ?, version = version + 1
                    WHERE task_id = ? AND workflow_id = ?""",
                    (now_text, int(task_id), predecessor_workflow_id),
                )
            purpose = self._background_work_purpose(str(task["task_type"] or ""))
            base_key = (
                f"durable-task:{int(task_id)}:"
                f"generation:{int(task['generation'])}:attempt:{int(task['attempts'])}"
            )
            key = (
                f"{base_key}:after:{predecessor_workflow_id}"
                if predecessor_workflow_id is not None
                else base_key
            )
            while True:
                conn.execute(
                    """INSERT INTO ai_workflows(
                        public_ref, profile_id, instance_id, workflow_kind,
                        primary_purpose, trigger_kind, trigger_ref,
                        caused_by_workflow_id, reason, idempotency_key,
                        started_at, expires_at, updated_at
                    ) VALUES (?, ?, ?, 'BACKGROUND', ?, 'DURABLE_TASK', ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(profile_id, idempotency_key) DO NOTHING""",
                    (
                        _public_ref("work"),
                        task["profile_id"],
                        task["instance_id"],
                        purpose.value,
                        str(task_id),
                        int(task["caused_by_workflow_id"])
                        if task["caused_by_workflow_id"] is not None
                        else None,
                        work_purpose_spec(purpose).reason,
                        key,
                        now_text,
                        _dt(now + timedelta(days=30)),
                        now_text,
                    ),
                )
                workflow = conn.execute(
                    """SELECT * FROM ai_workflows
                    WHERE profile_id = ? AND idempotency_key = ?""",
                    (task["profile_id"], key),
                ).fetchone()
                assert workflow is not None
                if str(workflow["status"]) == "RUNNING":
                    break
                key = f"{base_key}:after:{int(workflow['workflow_id'])}"
            conn.execute(
                "UPDATE ai_tasks SET workflow_id = ? WHERE task_id = ? AND workflow_id IS NULL",
                (int(workflow["workflow_id"]), int(task_id)),
            )
            return workflow

        row = await self.uow.run(operation)
        return self._record(row, json_columns=()) if row is not None else None

    @staticmethod
    def _background_work_purpose(task_type: str) -> AIWorkPurpose:
        return {
            "STICKER_COLLECTION": AIWorkPurpose.STICKER_COLLECTION,
            "STICKER_CHECK": AIWorkPurpose.STICKER_CHECK,
            "STICKER_INTAKE": AIWorkPurpose.STICKER_COLLECTION,
            "DIALOGUE_SUMMARY": AIWorkPurpose.CONVERSATION_SUMMARY,
            "KNOWLEDGE_FORMATION": AIWorkPurpose.KNOWLEDGE_ORGANIZATION,
            "FILE_ARTIFACT_GENERATION": AIWorkPurpose.FILE_GENERATION,
            "VISION_DESCRIPTION": AIWorkPurpose.IMAGE_UNDERSTANDING,
            "TIMER_RUN": AIWorkPurpose.TIMER_RUN,
            "TIMER_LIFECYCLE_REVIEW": AIWorkPurpose.TIMER_LIFECYCLE_REVIEW,
        }.get(str(task_type or "").upper(), AIWorkPurpose.MODEL_REQUEST)

    @staticmethod
    def _finish_task_workflow_sql(
        conn: Any,
        task: Any,
        *,
        status: str,
        error_code: str,
        message: str,
        now: str,
    ) -> None:
        workflow_id = int(task["workflow_id"] or 0)
        if workflow_id <= 0:
            return
        status, error_code, message = coherent_workflow_terminal(
            conn,
            workflow_id,
            status,
            error_code,
            message,
        )
        conn.execute(
            """UPDATE ai_workflows SET status = ?, final_error_code = ?,
            final_message = ?, finished_at = ?, updated_at = ?, version = version + 1
            WHERE workflow_id = ? AND status = 'RUNNING'""",
            (status, error_code, message, now, now, workflow_id),
        )

    async def finish_ai_workflow(self, workflow_id: int, **values: Any) -> dict[str, Any] | None:
        status = _workflow_status(values.get("status"))
        if status not in _WORKFLOW_TERMINAL:
            raise ValueError("invalid AI workflow status")
        now = _as_utc_text(values.get("finished_at"))
        final_error_code = str(values.get("final_error_code") or "")
        final_message = str(values.get("final_message") or "")

        def operation(conn: Any) -> None:
            coherent_status, coherent_error, coherent_message = coherent_workflow_terminal(
                conn,
                int(workflow_id),
                status,
                final_error_code,
                final_message,
            )
            conn.execute(
                """UPDATE ai_workflows SET status = ?, final_error_code = ?,
                final_message = ?, finished_at = ?, updated_at = ?, version = version + 1
                WHERE workflow_id = ? AND status = 'RUNNING'""",
                (
                    coherent_status,
                    coherent_error,
                    coherent_message,
                    now,
                    now,
                    int(workflow_id),
                ),
            )

        await self.uow.run(operation)
        return await self.get_ai_workflow(workflow_id)

    async def get_ai_workflow(self, workflow_id: int) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            "SELECT * FROM ai_workflows WHERE workflow_id = ?", (int(workflow_id),)
        )
        return self._record(row, json_columns=()) if row is not None else None

    async def get_ai_workflow_by_ref(
        self, *, profile_id: str, work_ref: str
    ) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            """SELECT workflow.*, instance.scope AS object_scope,
            instance.platform_id AS object_platform_id,
            instance.target_id AS object_target_id,
            durable_task.status AS durable_task_status,
            durable_task.due_at AS durable_task_due_at,
            durable_task.updated_at AS durable_task_updated_at
            FROM ai_workflows AS workflow
            LEFT JOIN character_instances AS instance
              ON instance.profile_id = workflow.profile_id
             AND instance.instance_id = workflow.instance_id
            LEFT JOIN ai_tasks AS durable_task
              ON workflow.trigger_kind = 'DURABLE_TASK'
             AND durable_task.task_id = CAST(workflow.trigger_ref AS INTEGER)
             AND durable_task.workflow_id = workflow.workflow_id
            WHERE workflow.profile_id = ? AND workflow.public_ref = ?""",
            (str(profile_id), str(work_ref)),
        )
        return self._record(row, json_columns=()) if row is not None else None

    # ------------------------------------------------------------------
    # Work nodes

    async def start_ai_work_node(self, record: Mapping[str, Any]) -> dict[str, Any]:
        workflow_id = int(record["workflow_id"])
        node_key = str(record.get("node_key") or "").strip()
        if not node_key:
            raise ValueError("AI work node key is required")
        parent_node_id = record.get("parent_node_id")
        role = str(record.get("node_role") or AIWorkNodeRole.BUSINESS_STAGE).upper()
        raw_purpose = str(record.get("purpose") or "").strip()
        if role == AIWorkNodeRole.BUSINESS_STAGE.value:
            normalized_purpose = normalize_work_purpose(raw_purpose)
            purpose = normalized_purpose.value
            default_kind = work_purpose_spec(normalized_purpose).kind.value
        else:
            purpose = raw_purpose or role
            default_kind = (
                AIWorkNodeKind.COMMAND.value
                if role == AIWorkNodeRole.INTERNAL_ACTION.value
                else AIWorkNodeKind.SYSTEM.value
            )
        kind = str(record.get("node_kind") or default_kind).upper()
        started_at = _as_utc_text(record.get("started_at"))

        def operation(conn: Any) -> Any:
            workflow = conn.execute(
                "SELECT status FROM ai_workflows WHERE workflow_id = ?", (workflow_id,)
            ).fetchone()
            if workflow is None:
                raise KeyError(f"AI workflow {workflow_id} does not exist")
            existing = conn.execute(
                """SELECT * FROM ai_work_nodes
                WHERE workflow_id = ? AND node_key = ?""",
                (workflow_id, node_key),
            ).fetchone()
            if existing is not None:
                expected_parent = int(parent_node_id) if parent_node_id is not None else None
                if (
                    str(existing["node_role"]) != role
                    or str(existing["node_kind"]) != kind
                    or str(existing["purpose"]) != purpose
                    or existing["parent_node_id"] != expected_parent
                ):
                    raise ValueError("AI work node key was reused with different semantics")
                return existing
            if str(workflow["status"]) != "RUNNING":
                raise ValueError("cannot add a node to a terminal AI workflow")
            if parent_node_id is not None:
                parent = conn.execute(
                    "SELECT workflow_id FROM ai_work_nodes WHERE node_id = ?",
                    (int(parent_node_id),),
                ).fetchone()
                if parent is None or int(parent["workflow_id"]) != workflow_id:
                    raise ValueError("parent AI work node belongs to another workflow")
            sequence = int(
                conn.execute(
                    """SELECT COALESCE(MAX(sequence), 0) + 1 AS value
                    FROM ai_work_nodes WHERE workflow_id = ?""",
                    (workflow_id,),
                ).fetchone()["value"]
            )
            cursor = conn.execute(
                """INSERT INTO ai_work_nodes(
                    public_ref, workflow_id, parent_node_id, sequence,
                    node_role, node_kind, purpose, node_key, input_json,
                    started_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    _public_ref("node"),
                    workflow_id,
                    int(parent_node_id) if parent_node_id is not None else None,
                    sequence,
                    role,
                    kind,
                    purpose,
                    node_key,
                    _dump(record.get("input") or {}),
                    started_at,
                    started_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM ai_work_nodes WHERE node_id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
            assert row is not None
            return row

        row = await self.uow.run(operation)
        return self._record(row, json_columns=("input_json", "result_json"))

    async def project_model_visible_message_ids(
        self,
        run_id: int,
        node_id: int,
        message_ids: Sequence[int],
        *,
        summary_ids: Sequence[int] = (),
        summary_coverage: Sequence[Sequence[int]] = (),
    ) -> dict[str, Any] | None:
        visible_ids = tuple(sorted({int(value) for value in message_ids if int(value) > 0}))
        visible_summary_ids = {int(value) for value in summary_ids if int(value) > 0}
        coverage = model_visible_summary_coverage(summary_coverage)
        if visible_summary_ids != {summary_id for summary_id, _, _ in coverage}:
            return None

        def operation(conn: Any) -> dict[str, Any] | None:
            return project_model_visible_message_ids(
                conn,
                run_id=run_id,
                node_id=node_id,
                visible_ids=visible_ids,
                visible_summary_ids=visible_summary_ids,
                coverage=coverage,
            )

        return await self.uow.run(operation)

    async def finish_ai_work_node(self, node_id: int, **values: Any) -> dict[str, Any] | None:
        status = _node_status(values.get("status"))
        if status not in _NODE_TERMINAL:
            raise ValueError("invalid AI work node status")
        finished_at = _as_utc_text(values.get("finished_at"))
        has_result = "result" in values

        def operation(conn: Any) -> Any:
            current = conn.execute(
                "SELECT * FROM ai_work_nodes WHERE node_id = ?", (int(node_id),)
            ).fetchone()
            if current is None:
                return None
            if str(current["status"]) != "RUNNING":
                return current
            conn.execute(
                """UPDATE ai_work_nodes SET status = ?, error_code = ?,
                error_message = ?, warning_code = ?, warning_message = ?, summary = ?,
                result_json = CASE WHEN ? = 1 THEN ? ELSE result_json END,
                finished_at = ? WHERE node_id = ? AND status = 'RUNNING'""",
                (
                    status,
                    str(values.get("error_code") or ""),
                    str(values.get("error_message") or ""),
                    str(values.get("warning_code") or ""),
                    str(values.get("warning_message") or values.get("warning") or ""),
                    str(values.get("summary") or ""),
                    int(has_result),
                    _dump(values.get("result")) if has_result else None,
                    finished_at,
                    int(node_id),
                ),
            )
            return conn.execute(
                "SELECT * FROM ai_work_nodes WHERE node_id = ?", (int(node_id),)
            ).fetchone()

        row = await self.uow.run(operation)
        return (
            self._record(row, json_columns=("input_json", "result_json"))
            if row is not None
            else None
        )

    async def get_ai_work_node(self, node_id: int) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            "SELECT * FROM ai_work_nodes WHERE node_id = ?", (int(node_id),)
        )
        return (
            self._record(row, json_columns=("input_json", "result_json"))
            if row is not None
            else None
        )

    async def get_ai_work_node_by_model_invocation(
        self, invocation_id: str
    ) -> dict[str, Any] | None:
        rows = await self.db.fetch_all(
            """SELECT node.* FROM ai_work_nodes AS node
            JOIN ai_provider_attempts AS attempt ON attempt.node_id = node.node_id
            WHERE attempt.invocation_id = ?
            GROUP BY node.node_id ORDER BY node.node_id LIMIT 2""",
            (str(invocation_id),),
        )
        row = rows[0] if len(rows) == 1 else None
        return (
            self._record(row, json_columns=("input_json", "result_json"))
            if row is not None
            else None
        )

    # ------------------------------------------------------------------
    # Provider attempts

    async def start_ai_provider_attempt(self, record: Mapping[str, Any]) -> dict[str, Any]:
        node_id = int(record.get("node_id") or 0)
        invocation_id = str(record.get("invocation_id") or "").strip()
        if not invocation_id:
            raise ValueError("provider attempt invocation_id is required")
        round_no = max(1, int(record.get("round_no") or 1))
        attempt_no = max(1, int(record.get("attempt_no") or 1))
        started_at = _as_utc_text(record.get("started_at"))

        def operation(conn: Any) -> Any:
            node = conn.execute(
                """SELECT node.node_role, node.status,
                workflow.status AS workflow_status FROM ai_work_nodes AS node
                JOIN ai_workflows AS workflow ON workflow.workflow_id = node.workflow_id
                WHERE node.node_id = ?""",
                (node_id,),
            ).fetchone()
            if node is None:
                raise KeyError(f"AI work node {node_id} does not exist")
            if str(node["node_role"]) != AIWorkNodeRole.BUSINESS_STAGE.value:
                raise ValueError("provider attempts must belong to a business stage")
            existing = conn.execute(
                """SELECT * FROM ai_provider_attempts WHERE node_id = ?
                AND invocation_id = ? AND round_no = ? AND attempt_no = ?""",
                (node_id, invocation_id, round_no, attempt_no),
            ).fetchone()
            if existing is not None:
                return existing
            if str(node["status"]) != "RUNNING" or str(node["workflow_status"]) != "RUNNING":
                raise ValueError("cannot add a provider attempt to a terminal AI work node")
            cursor = conn.execute(
                """INSERT INTO ai_provider_attempts(
                    public_ref, node_id, invocation_id, round_no, attempt_no,
                    backend_id, model_id, status, request_json,
                    started_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PREPARING', ?, ?, ?)""",
                (
                    _public_ref("attempt"),
                    node_id,
                    invocation_id,
                    round_no,
                    attempt_no,
                    str(record.get("backend_id") or "") or None,
                    str(record.get("model_id") or ""),
                    _dump(record.get("request")) if record.get("request") is not None else None,
                    started_at,
                    started_at,
                ),
            )
            row = conn.execute(
                "SELECT * FROM ai_provider_attempts WHERE attempt_id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
            assert row is not None
            return row

        row = await self.uow.run(operation)
        return self._attempt_record(row)

    async def enrich_ai_provider_attempt(
        self, attempt_id: int, **values: Any
    ) -> dict[str, Any] | None:
        has_request = values.get("request") is not None
        has_transport = values.get("transport") is not None

        def operation(conn: Any) -> Any:
            conn.execute(
                """UPDATE ai_provider_attempts SET
                backend_id = CASE WHEN ? <> '' THEN ? ELSE backend_id END,
                model_id = CASE WHEN ? <> '' THEN ? ELSE model_id END,
                request_json = CASE WHEN ? = 1 THEN ? ELSE request_json END,
                transport_json = CASE WHEN ? = 1 THEN ? ELSE transport_json END
                WHERE attempt_id = ? AND status = 'PREPARING' AND sent_at IS NULL""",
                (
                    str(values.get("backend_id") or ""),
                    str(values.get("backend_id") or ""),
                    str(values.get("model_id") or ""),
                    str(values.get("model_id") or ""),
                    int(has_request),
                    _dump(values.get("request")) if has_request else None,
                    int(has_transport),
                    _dump(values.get("transport")) if has_transport else None,
                    int(attempt_id),
                ),
            )
            return conn.execute(
                "SELECT * FROM ai_provider_attempts WHERE attempt_id = ?", (int(attempt_id),)
            ).fetchone()

        row = await self.uow.run(operation)
        return self._attempt_record(row) if row is not None else None

    async def mark_ai_provider_attempt_sent(
        self, attempt_id: int, **values: Any
    ) -> dict[str, Any] | None:
        sent_at = _as_utc_text(values.get("sent_at"))
        has_transport = values.get("transport") is not None

        def operation(conn: Any) -> Any:
            conn.execute(
                """UPDATE ai_provider_attempts SET status = 'IN_FLIGHT', sent_at = ?,
                transport_json = CASE WHEN ? = 1 THEN ? ELSE transport_json END
                WHERE attempt_id = ? AND status = 'PREPARING' AND sent_at IS NULL""",
                (
                    sent_at,
                    int(has_transport),
                    _dump(values.get("transport")) if has_transport else None,
                    int(attempt_id),
                ),
            )
            return conn.execute(
                "SELECT * FROM ai_provider_attempts WHERE attempt_id = ?", (int(attempt_id),)
            ).fetchone()

        row = await self.uow.run(operation)
        return self._attempt_record(row) if row is not None else None

    async def finish_ai_provider_attempt(
        self, attempt_id: int, **values: Any
    ) -> dict[str, Any] | None:
        status = str(values.get("status") or "FAILED").upper()
        if status not in _ATTEMPT_TERMINAL:
            raise ValueError("invalid provider attempt status")
        usage = dict(values.get("usage") or {})
        input_tokens = max(0, int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0))
        output_tokens = max(
            0, int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        )
        cache_read_tokens = max(0, int(usage.get("cache_read_tokens") or 0))
        cache_write_tokens = max(0, int(usage.get("cache_write_tokens") or 0))
        cache_mode = str(usage.get("cache_mode") or "")
        cache_status = str(usage.get("cache_status") or "")
        finished_at = _as_utc_text(values.get("finished_at"))

        def operation(conn: Any) -> Any:
            current = conn.execute(
                "SELECT * FROM ai_provider_attempts WHERE attempt_id = ?", (int(attempt_id),)
            ).fetchone()
            if current is None:
                return None
            if str(current["status"]) in _ATTEMPT_TERMINAL or current["finished_at"] is not None:
                return current
            sent_at = current["sent_at"]
            if sent_at is None and bool(values.get("sent")):
                sent_at = finished_at
            conn.execute(
                """UPDATE ai_provider_attempts SET status = ?, sent_at = ?,
                response_json = ?, transport_json = ?,
                error_code = ?, error_message = ?, input_tokens = ?, output_tokens = ?,
                cache_read_tokens = ?, cache_write_tokens = ?,
                cache_mode = ?, cache_status = ?,
                finished_at = ? WHERE attempt_id = ?
                AND status IN ('PREPARING', 'IN_FLIGHT') AND finished_at IS NULL""",
                (
                    status,
                    sent_at,
                    _dump(values.get("response")) if values.get("response") is not None else None,
                    _dump(values.get("transport") or {}),
                    str(values.get("error_code") or ""),
                    str(values.get("error_message") or ""),
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_write_tokens,
                    cache_mode,
                    cache_status,
                    finished_at,
                    int(attempt_id),
                ),
            )
            return conn.execute(
                "SELECT * FROM ai_provider_attempts WHERE attempt_id = ?", (int(attempt_id),)
            ).fetchone()

        row = await self.uow.run(operation)
        return self._attempt_record(row) if row is not None else None

    async def annotate_ai_provider_attempt(
        self, record: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        attempt_id = int(record.get("attempt_id") or 0)

        def operation(conn: Any) -> Any:
            if attempt_id > 0:
                current = conn.execute(
                    "SELECT * FROM ai_provider_attempts WHERE attempt_id = ?", (attempt_id,)
                ).fetchone()
            else:
                nodes = list(
                    conn.execute(
                        """SELECT DISTINCT node_id FROM ai_provider_attempts
                        WHERE invocation_id = ? AND round_no = ? LIMIT 2""",
                        (
                            str(record.get("invocation_id") or ""),
                            max(1, int(record.get("round_no") or 1)),
                        ),
                    )
                )
                current = (
                    None
                    if len(nodes) != 1
                    else conn.execute(
                        """SELECT * FROM ai_provider_attempts WHERE node_id = ?
                    AND invocation_id = ? AND round_no = ? ORDER BY
                    CASE status WHEN 'SUCCEEDED' THEN 0 ELSE 1 END,
                    attempt_no DESC, attempt_id DESC LIMIT 1""",
                        (
                            int(nodes[0]["node_id"]),
                            str(record.get("invocation_id") or ""),
                            max(1, int(record.get("round_no") or 1)),
                        ),
                    ).fetchone()
                )
            if current is None:
                return None
            if str(current["error_code"] or "") == "SOURCE_MESSAGE_RECALLED":
                return current
            loaded = _load(current["evaluation_json"]) if current["evaluation_json"] else {}
            previous = dict(loaded or {})
            incoming = dict(record.get("evaluation") or {})
            incoming.pop("annotations", None)
            annotations = list(previous.pop("annotations", ()) or ())
            annotations.append(incoming)
            evaluation = {**previous, **incoming, "annotations": annotations}
            conn.execute(
                "UPDATE ai_provider_attempts SET evaluation_json = ? WHERE attempt_id = ?",
                (_dump(evaluation), int(current["attempt_id"])),
            )
            return conn.execute(
                """SELECT attempt.*, node.workflow_id FROM ai_provider_attempts AS attempt
                JOIN ai_work_nodes AS node ON node.node_id = attempt.node_id
                WHERE attempt.attempt_id = ?""",
                (int(current["attempt_id"]),),
            ).fetchone()

        row = await self.uow.run(operation)
        return self._attempt_record(row) if row is not None else None

    def _attempt_record(self, row: Any) -> dict[str, Any]:
        return self._record(
            row,
            json_columns=(
                "request_json",
                "response_json",
                "transport_json",
                "evaluation_json",
            ),
        )

    # ------------------------------------------------------------------
    # Events and queries

    async def record_ai_work_event(self, record: Mapping[str, Any]) -> dict[str, Any]:
        workflow_id = int(record["workflow_id"])
        node_id = int(record.get("node_id") or 0) or None
        occurred_at = _as_utc_text(record.get("occurred_at"))

        def operation(conn: Any) -> Any:
            workflow = conn.execute(
                """SELECT status, final_error_code FROM ai_workflows
                WHERE workflow_id = ?""",
                (workflow_id,),
            ).fetchone()
            if workflow is None:
                raise KeyError(f"AI workflow {workflow_id} does not exist")
            recalled = str(workflow["final_error_code"] or "") == "SOURCE_MESSAGE_RECALLED"
            if node_id is not None:
                node = conn.execute(
                    "SELECT workflow_id FROM ai_work_nodes WHERE node_id = ?", (node_id,)
                ).fetchone()
                if node is None or int(node["workflow_id"]) != workflow_id:
                    raise ValueError("AI work event node belongs to another workflow")
            sequence = int(
                conn.execute(
                    """SELECT COALESCE(MAX(sequence), 0) + 1 AS value
                    FROM ai_work_events WHERE workflow_id = ?""",
                    (workflow_id,),
                ).fetchone()["value"]
            )
            cursor = conn.execute(
                """INSERT INTO ai_work_events(
                    public_ref, workflow_id, node_id, sequence, event_category,
                    severity, code, summary, details_json, occurred_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    _public_ref("event"),
                    workflow_id,
                    node_id,
                    sequence,
                    str(record.get("event_category") or "NOTE").upper(),
                    str(record.get("severity") or "INFO").upper(),
                    "SOURCE_MESSAGE_RECALLED" if recalled else str(record.get("code") or ""),
                    (
                        "源消息撤回后已脱敏"
                        if recalled
                        else str(record.get("summary") or "处理记录")
                    ),
                    _dump({} if recalled else record.get("details") or {}),
                    occurred_at,
                ),
            )
            return conn.execute(
                "SELECT * FROM ai_work_events WHERE event_id = ?", (int(cursor.lastrowid),)
            ).fetchone()

        row = await self.uow.run(operation)
        return self._record(row, json_columns=("details_json",))


__all__ = ["AiWorkRecords"]
