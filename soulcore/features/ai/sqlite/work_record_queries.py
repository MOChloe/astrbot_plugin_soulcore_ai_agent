from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from ..work_taxonomy import normalize_work_purpose
from .support import _dt


class AiWorkRecordQueries:
    """Read projections for the causal AI work record domain."""

    if TYPE_CHECKING:
        db: Any

        def _record(self, row: Any, *, json_columns: tuple[str, ...]) -> dict[str, Any]: ...

        def _attempt_record(self, row: Any) -> dict[str, Any]: ...

    async def list_ai_workflow_summaries(
        self,
        *,
        profile_id: str,
        instance_id: str | None = None,
        run_status: str = "",
        delivery_status: str = "",
        purpose: str = "",
        model: str = "",
        issue_type: str = "",
        since: datetime | None = None,
        issues_only: bool = False,
        cursor_started_at: str = "",
        cursor_workflow_id: int = 0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses = ["workflow.profile_id = ?"]
        params: list[Any] = [str(profile_id)]
        self._append_summary_filters(
            clauses,
            params,
            instance_id=instance_id,
            run_status=run_status,
            purpose=purpose,
            model=model,
            since=since,
            cursor_started_at=cursor_started_at,
            cursor_workflow_id=cursor_workflow_id,
        )
        issue_clause, issue_params = self._issue_filter_sql(issue_type, issues_only)
        self._append_clause(clauses, params, issue_clause, issue_params)
        delivery_clause, delivery_params = self._delivery_filter_sql(delivery_status)
        self._append_clause(clauses, params, delivery_clause, delivery_params)
        page_size = max(1, min(int(limit), 100))
        params.append(page_size + 1)
        rows = await self.db.fetch_all(
            self._summary_sql(clauses),
            tuple(params),
        )
        records = [self._record(row, json_columns=()) for row in rows]
        await self._attach_composition(records)
        return records

    @staticmethod
    def _append_clause(
        clauses: list[str], params: list[Any], clause: str, values: list[Any]
    ) -> None:
        if clause:
            clauses.append(clause)
            params.extend(values)

    @staticmethod
    def _append_summary_filters(
        clauses: list[str],
        params: list[Any],
        *,
        instance_id: str | None,
        run_status: str,
        purpose: str,
        model: str,
        since: datetime | None,
        cursor_started_at: str,
        cursor_workflow_id: int,
    ) -> None:
        filters: tuple[tuple[bool, str, Any], ...] = (
            (bool(instance_id), "workflow.instance_id = ?", str(instance_id or "")),
            (bool(run_status), "workflow.status = ?", str(run_status).upper()),
            (
                bool(purpose),
                "workflow.primary_purpose = ?",
                normalize_work_purpose(purpose).value if purpose else "",
            ),
            (
                bool(model),
                """EXISTS (SELECT 1 FROM ai_work_nodes filter_node
                JOIN ai_provider_attempts filter_attempt
                  ON filter_attempt.node_id = filter_node.node_id
                WHERE filter_node.workflow_id = workflow.workflow_id
                  AND LOWER(filter_attempt.model_id) LIKE ?)""",
                f"%{str(model).lower()}%",
            ),
            (since is not None, "workflow.started_at >= ?", _dt(since) if since else ""),
        )
        for enabled, clause, value in filters:
            if enabled:
                clauses.append(clause)
                params.append(value)
        if cursor_started_at and cursor_workflow_id > 0:
            clauses.append(
                "(workflow.started_at < ? OR (workflow.started_at = ? AND workflow.workflow_id < ?))"
            )
            params.extend((cursor_started_at, cursor_started_at, int(cursor_workflow_id)))

    @staticmethod
    def _summary_sql(clauses: list[str]) -> str:
        return f"""SELECT workflow.*,
        instance.scope AS object_scope,
        instance.platform_id AS object_platform_id,
        instance.target_id AS object_target_id,
        durable_task.status AS durable_task_status,
        durable_task.due_at AS durable_task_due_at,
        durable_task.updated_at AS durable_task_updated_at,
        (SELECT node.started_at FROM ai_work_nodes node
          WHERE node.workflow_id = workflow.workflow_id
            AND node.node_role = 'BUSINESS_STAGE'
          ORDER BY node.sequence DESC, node.node_id DESC LIMIT 1) AS latest_stage_started_at,
        (SELECT node.finished_at FROM ai_work_nodes node
          WHERE node.workflow_id = workflow.workflow_id
            AND node.node_role = 'BUSINESS_STAGE'
          ORDER BY node.sequence DESC, node.node_id DESC LIMIT 1) AS latest_stage_finished_at,
        (SELECT COUNT(*) FROM ai_work_nodes node
          WHERE node.workflow_id = workflow.workflow_id
            AND node.node_role = 'BUSINESS_STAGE') AS business_stage_count,
        (SELECT COUNT(*) FROM ai_work_nodes node
          WHERE node.workflow_id = workflow.workflow_id
            AND node.node_role = 'INTERNAL_ACTION') AS internal_action_count,
        (SELECT COUNT(*) FROM ai_provider_attempts attempt
          JOIN ai_work_nodes node ON node.node_id = attempt.node_id
          WHERE node.workflow_id = workflow.workflow_id
            AND attempt.sent_at IS NOT NULL) AS provider_send_count,
        (SELECT COUNT(*) FROM ai_provider_attempts attempt
          JOIN ai_work_nodes node ON node.node_id = attempt.node_id
          WHERE node.workflow_id = workflow.workflow_id
            AND attempt.sent_at IS NOT NULL AND attempt.attempt_no > 1) AS retry_count,
        (SELECT COALESCE(SUM(attempt.input_tokens), 0)
          FROM ai_provider_attempts attempt
          JOIN ai_work_nodes node ON node.node_id = attempt.node_id
          WHERE node.workflow_id = workflow.workflow_id) AS input_tokens,
        (SELECT COALESCE(SUM(attempt.output_tokens), 0)
          FROM ai_provider_attempts attempt
          JOIN ai_work_nodes node ON node.node_id = attempt.node_id
          WHERE node.workflow_id = workflow.workflow_id) AS output_tokens,
        (SELECT COALESCE(SUM(attempt.cache_read_tokens), 0)
          FROM ai_provider_attempts attempt
          JOIN ai_work_nodes node ON node.node_id = attempt.node_id
          WHERE node.workflow_id = workflow.workflow_id) AS cache_read_tokens,
        (SELECT COALESCE(SUM(attempt.cache_write_tokens), 0)
          FROM ai_provider_attempts attempt
          JOIN ai_work_nodes node ON node.node_id = attempt.node_id
          WHERE node.workflow_id = workflow.workflow_id) AS cache_write_tokens,
        (SELECT GROUP_CONCAT(DISTINCT attempt.model_id)
          FROM ai_provider_attempts attempt
          JOIN ai_work_nodes node ON node.node_id = attempt.node_id
          WHERE node.workflow_id = workflow.workflow_id
            AND attempt.model_id <> '') AS models_csv,
        (SELECT COUNT(*) FROM ai_work_nodes node
          WHERE node.workflow_id = workflow.workflow_id
            AND node.status = 'FALLBACK') AS fallback_count,
        (SELECT COUNT(*) FROM ai_work_events event
          WHERE event.workflow_id = workflow.workflow_id
            AND event.severity IN ('WARNING', 'ERROR')) AS issue_count,
        (SELECT COUNT(*) FROM instance_outbox outbox
          WHERE outbox.workflow_id = workflow.workflow_id) AS delivery_count,
        (SELECT COUNT(*) FROM instance_outbox outbox
          WHERE outbox.workflow_id = workflow.workflow_id
            AND outbox.status = 'FAILED') AS delivery_failed_count,
        (SELECT COUNT(*) FROM instance_outbox outbox
          WHERE outbox.workflow_id = workflow.workflow_id
            AND outbox.status = 'PLATFORM_ACCEPTED_UNCONFIRMED'
            AND outbox.last_diagnostic_code NOT LIKE 'send_exception:%'
            AND outbox.last_diagnostic_code NOT LIKE
              'delivery_exception_after_platform_boundary:%') AS delivery_accepted_count,
        (SELECT COUNT(*) FROM instance_outbox outbox
          WHERE outbox.workflow_id = workflow.workflow_id
            AND outbox.status = 'PARTIALLY_ATTEMPTED') AS delivery_partial_count,
        (SELECT COUNT(*) FROM instance_outbox outbox
          WHERE outbox.workflow_id = workflow.workflow_id
            AND outbox.status IN ('PENDING', 'SENDING')) AS delivery_pending_count,
        (SELECT COUNT(*) FROM instance_outbox outbox
          WHERE outbox.workflow_id = workflow.workflow_id
            AND (
              outbox.status = 'UNKNOWN_AFTER_CRASH'
              OR (
                outbox.status = 'PLATFORM_ACCEPTED_UNCONFIRMED'
                AND (
                  outbox.last_diagnostic_code LIKE 'send_exception:%'
                  OR outbox.last_diagnostic_code LIKE
                    'delivery_exception_after_platform_boundary:%'
                )
              )
            )) AS delivery_unknown_count
        FROM ai_workflows AS workflow
        LEFT JOIN character_instances AS instance
          ON instance.profile_id = workflow.profile_id
         AND instance.instance_id = workflow.instance_id
        LEFT JOIN ai_tasks AS durable_task
          ON workflow.trigger_kind = 'DURABLE_TASK'
         AND durable_task.task_id = CAST(workflow.trigger_ref AS INTEGER)
         AND durable_task.workflow_id = workflow.workflow_id
        WHERE {" AND ".join(clauses)}
        ORDER BY workflow.started_at DESC, workflow.workflow_id DESC LIMIT ?"""

    @staticmethod
    def _issue_filter_sql(issue_type: str, issues_only: bool) -> tuple[str, list[Any]]:
        normalized = str(issue_type or "").strip().lower()
        if normalized == "fallback":
            return (
                "EXISTS (SELECT 1 FROM ai_work_nodes n WHERE n.workflow_id = workflow.workflow_id AND n.status = 'FALLBACK')",
                [],
            )
        if normalized == "retried":
            return (
                """EXISTS (SELECT 1 FROM ai_work_nodes n JOIN ai_provider_attempts a
                ON a.node_id = n.node_id WHERE n.workflow_id = workflow.workflow_id
                AND a.sent_at IS NOT NULL AND a.attempt_no > 1)""",
                [],
            )
        if normalized:
            return (
                """EXISTS (SELECT 1 FROM ai_work_events e
                WHERE e.workflow_id = workflow.workflow_id AND LOWER(e.code) = ?
                AND e.severity IN ('WARNING', 'ERROR'))""",
                [normalized],
            )
        if issues_only:
            return (
                """(workflow.status = 'FAILED'
                OR EXISTS (SELECT 1 FROM ai_work_nodes n WHERE n.workflow_id = workflow.workflow_id AND n.status IN ('FALLBACK','FAILED'))
                OR EXISTS (SELECT 1 FROM ai_work_events e WHERE e.workflow_id = workflow.workflow_id AND e.severity IN ('WARNING','ERROR'))
                OR EXISTS (SELECT 1 FROM instance_outbox o WHERE o.workflow_id = workflow.workflow_id AND o.status IN ('FAILED','PARTIALLY_ATTEMPTED','UNKNOWN_AFTER_CRASH')))""",
                [],
            )
        return "", []

    @staticmethod
    def _delivery_filter_sql(delivery_status: str) -> tuple[str, list[Any]]:
        normalized = str(delivery_status or "").strip().upper()
        if not normalized:
            return "", []
        mapping = {
            "PENDING": ("PENDING", "SENDING"),
            "PLATFORM_ACCEPTED_UNCONFIRMED": ("PLATFORM_ACCEPTED_UNCONFIRMED",),
            "PARTIALLY_ATTEMPTED": ("PARTIALLY_ATTEMPTED",),
            "FAILED": ("FAILED",),
            "UNKNOWN": ("UNKNOWN_AFTER_CRASH",),
            "CANCELLED": ("CANCELLED",),
            "NONE": (),
        }
        statuses = mapping.get(normalized, (normalized,))
        if normalized == "NONE":
            return (
                "NOT EXISTS (SELECT 1 FROM instance_outbox o WHERE o.workflow_id = workflow.workflow_id)",
                [],
            )
        placeholders = ",".join("?" for _ in statuses)
        return (
            f"EXISTS (SELECT 1 FROM instance_outbox o WHERE o.workflow_id = workflow.workflow_id AND o.status IN ({placeholders}))",
            list(statuses),
        )

    async def _attach_composition(self, records: list[dict[str, Any]]) -> None:
        ids = [int(item["workflow_id"]) for item in records]
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        rows = await self.db.fetch_all(
            f"""SELECT workflow_id, purpose, COUNT(*) AS count
            FROM ai_work_nodes WHERE workflow_id IN ({placeholders})
            AND node_role = 'BUSINESS_STAGE'
            GROUP BY workflow_id, purpose ORDER BY MIN(sequence)""",
            tuple(ids),
        )
        grouped: dict[int, list[dict[str, Any]]] = {value: [] for value in ids}
        for row in rows:
            grouped[int(row["workflow_id"])].append(
                {"purpose": str(row["purpose"]), "count": int(row["count"])}
            )
        for item in records:
            item["composition"] = grouped[int(item["workflow_id"])]

    async def list_ai_work_filter_values(
        self, *, profile_id: str, instance_id: str | None = None
    ) -> dict[str, list[str]]:
        where = "workflow.profile_id = ?"
        params: list[Any] = [str(profile_id)]
        if instance_id:
            where += " AND workflow.instance_id = ?"
            params.append(str(instance_id))
        purposes = await self.db.fetch_all(
            f"""SELECT DISTINCT workflow.primary_purpose AS value
            FROM ai_workflows workflow WHERE {where} ORDER BY value""",
            tuple(params),
        )
        models = await self.db.fetch_all(
            f"""SELECT DISTINCT attempt.model_id AS value
            FROM ai_workflows workflow
            JOIN ai_work_nodes node ON node.workflow_id = workflow.workflow_id
            JOIN ai_provider_attempts attempt ON attempt.node_id = node.node_id
            WHERE {where} AND attempt.model_id <> '' ORDER BY value""",
            tuple(params),
        )
        issue_codes = await self.db.fetch_all(
            f"""SELECT DISTINCT event.code AS value
            FROM ai_workflows workflow
            JOIN ai_work_events event ON event.workflow_id = workflow.workflow_id
            WHERE {where} AND event.code <> ''
              AND event.severity IN ('WARNING', 'ERROR') ORDER BY value""",
            tuple(params),
        )
        return {
            "purposes": [str(row["value"]) for row in purposes],
            "models": [str(row["value"]) for row in models],
            "issue_codes": [str(row["value"]) for row in issue_codes],
        }

    async def list_ai_work_nodes(self, workflow_id: int) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT * FROM ai_work_nodes WHERE workflow_id = ?
            ORDER BY sequence, node_id""",
            (int(workflow_id),),
        )
        return [self._record(row, json_columns=("input_json", "result_json")) for row in rows]

    async def list_ai_provider_attempts(
        self, *, workflow_id: int | None = None, node_id: int | None = None
    ) -> list[dict[str, Any]]:
        if workflow_id is not None:
            rows = await self.db.fetch_all(
                """SELECT attempt.* FROM ai_provider_attempts attempt
                JOIN ai_work_nodes node ON node.node_id = attempt.node_id
                WHERE node.workflow_id = ?
                ORDER BY node.sequence, attempt.round_no,
                attempt.attempt_no, attempt.attempt_id""",
                (int(workflow_id),),
            )
        elif node_id is not None:
            rows = await self.db.fetch_all(
                """SELECT * FROM ai_provider_attempts WHERE node_id = ?
                ORDER BY round_no, attempt_no, attempt_id""",
                (int(node_id),),
            )
        else:
            raise ValueError("workflow_id or node_id is required")
        return [self._attempt_record(row) for row in rows]

    async def get_ai_provider_attempt_by_ref(
        self, *, profile_id: str, work_ref: str, attempt_ref: str
    ) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            """SELECT attempt.* FROM ai_provider_attempts attempt
            JOIN ai_work_nodes node ON node.node_id = attempt.node_id
            JOIN ai_workflows workflow ON workflow.workflow_id = node.workflow_id
            WHERE workflow.profile_id = ? AND workflow.public_ref = ?
              AND attempt.public_ref = ?""",
            (str(profile_id), str(work_ref), str(attempt_ref)),
        )
        return self._attempt_record(row) if row is not None else None

    async def list_ai_work_events(self, workflow_id: int) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT * FROM ai_work_events WHERE workflow_id = ?
            ORDER BY sequence, event_id""",
            (int(workflow_id),),
        )
        return [self._record(row, json_columns=("details_json",)) for row in rows]

    async def list_ai_workflow_deliveries(self, workflow_id: int) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT outbox_id, status, payload_json, not_before_at,
            last_error_code, last_error, last_diagnostic_code, created_at, updated_at
            FROM instance_outbox WHERE workflow_id = ? ORDER BY outbox_id""",
            (int(workflow_id),),
        )
        return [self._record(row, json_columns=("payload_json",)) for row in rows]


__all__ = ["AiWorkRecordQueries"]
