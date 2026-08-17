"""Remove recalled source material from persisted model work records."""

from __future__ import annotations

import json
import sqlite3

from ..domain import InboundRecallHold


def redact_model_artifacts(
    conn: sqlite3.Connection,
    hold: InboundRecallHold,
    now_text: str,
) -> None:
    runs = _find_core_runs(conn, hold)
    workflow_ids = {int(run["workflow_id"]) for run in runs if int(run["workflow_id"] or 0) > 0}
    workflow_ids.update(_find_admission_workflows(conn, hold))
    workflow_ids.update(_find_model_visible_workflows(conn, hold))
    related_tasks = _find_related_tasks(conn, hold)
    related_task_ids = {int(row["task_id"]) for row in related_tasks}
    workflow_ids.update(
        int(row["workflow_id"]) for row in related_tasks if int(row["workflow_id"] or 0) > 0
    )
    workflow_ids.update(_find_durable_workflows(conn, hold, related_task_ids))
    redacted = json.dumps(
        {"redacted": True, "reason": "source_message_recalled"},
        separators=(",", ":"),
    )
    _redact_core_runs(conn, runs, redacted, now_text)
    _redact_workflows(conn, workflow_ids, redacted, now_text)
    affected_task_ids = _find_affected_task_ids(conn, workflow_ids, related_task_ids)
    _redact_tasks(conn, affected_task_ids, now_text)


def _find_core_runs(
    conn: sqlite3.Connection,
    hold: InboundRecallHold,
) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """SELECT run_id, workflow_id FROM instance_core_runs
            WHERE profile_id = ? AND instance_id = ? AND (
                CAST(COALESCE(json_extract(
                    request_json, '$.metadata.context_message_id'
                ), 0) AS INTEGER) = ?
                OR EXISTS (
                    SELECT 1 FROM json_each(
                        COALESCE(json_extract(
                            request_json, '$.metadata.context_message_ids'
                        ), '[]')
                    ) WHERE CAST(json_each.value AS INTEGER) = ?
                )
                OR EXISTS (
                    SELECT 1 FROM json_each(
                        COALESCE(json_extract(
                            request_json, '$.metadata.model_visible_message_ids'
                        ), '[]')
                    ) WHERE CAST(json_each.value AS INTEGER) = ?
                )
                OR EXISTS (
                    SELECT 1 FROM json_each(
                        COALESCE(json_extract(
                            request_json,
                            '$.metadata.model_visible_summary_coverage'
                        ), '[]')
                    ) coverage
                    WHERE ? BETWEEN
                      CAST(json_extract(coverage.value,
                        '$.covered_from_message_id') AS INTEGER)
                      AND CAST(json_extract(coverage.value,
                        '$.covered_through_message_id') AS INTEGER)
                )
            )""",
            (
                hold.profile_id,
                hold.instance_id,
                hold.ledger_message_id,
                hold.ledger_message_id,
                hold.ledger_message_id,
                hold.ledger_message_id,
            ),
        )
    )


def _find_model_visible_workflows(
    conn: sqlite3.Connection,
    hold: InboundRecallHold,
) -> set[int]:
    rows = conn.execute(
        """SELECT DISTINCT node.workflow_id
        FROM ai_work_nodes node
        LEFT JOIN ai_provider_attempts attempt ON attempt.node_id = node.node_id
        JOIN ai_workflows workflow ON workflow.workflow_id = node.workflow_id
        WHERE workflow.profile_id = ? AND workflow.instance_id = ?
          AND (
            EXISTS (
                SELECT 1 FROM json_each(
                    COALESCE(json_extract(
                        node.input_json,
                        '$.model_visible_message_ids'
                    ), '[]')
                ) source
                WHERE CAST(source.value AS INTEGER) = ?
            )
            OR EXISTS (
                SELECT 1 FROM json_each(
                    COALESCE(json_extract(
                        attempt.request_json,
                        '$.prompt_document.source_message_ids'
                    ), '[]')
                ) source
                WHERE CAST(source.value AS INTEGER) = ?
            )
            OR EXISTS (
                SELECT 1 FROM json_each(
                    COALESCE(json_extract(
                        node.input_json,
                        '$.model_visible_summary_coverage'
                    ), '[]')
                ) coverage
                WHERE ? BETWEEN
                  CAST(json_extract(coverage.value,
                    '$.covered_from_message_id') AS INTEGER)
                  AND CAST(json_extract(coverage.value,
                    '$.covered_through_message_id') AS INTEGER)
            )
            OR EXISTS (
                SELECT 1 FROM json_each(
                    COALESCE(json_extract(
                        attempt.request_json,
                        '$.prompt_document.source_summary_coverage'
                    ), '[]')
                ) coverage
                WHERE ? BETWEEN
                  CAST(json_extract(coverage.value,
                    '$.covered_from_message_id') AS INTEGER)
                  AND CAST(json_extract(coverage.value,
                    '$.covered_through_message_id') AS INTEGER)
            )
          )""",
        (
            hold.profile_id,
            hold.instance_id,
            hold.ledger_message_id,
            hold.ledger_message_id,
            hold.ledger_message_id,
            hold.ledger_message_id,
        ),
    )
    return {int(row["workflow_id"]) for row in rows}


def _find_admission_workflows(
    conn: sqlite3.Connection,
    hold: InboundRecallHold,
) -> set[int]:
    rows = conn.execute(
        """SELECT DISTINCT workflow.workflow_id
        FROM ai_workflows workflow
        WHERE workflow.profile_id = ? AND workflow.instance_id = ? AND (
            (
                workflow.primary_purpose = 'TURN_CLASSIFICATION'
                AND workflow.trigger_ref IN (
                    SELECT member.batch_id
                    FROM conversation_turn_buffer_members member
                    WHERE member.profile_id = ? AND member.instance_id = ?
                      AND member.message_id = ?
                )
            )
            OR (
                workflow.primary_purpose = 'GROUP_INTERJECTION'
                AND workflow.trigger_ref IN (
                    SELECT member.window_id
                    FROM group_flow_window_members member
                    WHERE member.profile_id = ? AND member.instance_id = ?
                      AND member.message_id = ?
                )
            )
        )""",
        (
            hold.profile_id,
            hold.instance_id,
            hold.profile_id,
            hold.instance_id,
            hold.ledger_message_id,
            hold.profile_id,
            hold.instance_id,
            hold.ledger_message_id,
        ),
    )
    return {int(row["workflow_id"]) for row in rows}


def _find_related_tasks(
    conn: sqlite3.Connection,
    hold: InboundRecallHold,
) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """SELECT DISTINCT task.task_id, task.workflow_id
            FROM ai_tasks task
            WHERE task.profile_id = ? AND task.instance_id = ? AND (
                (
                    task.task_type = 'KNOWLEDGE_FORMATION'
                    AND EXISTS (
                        SELECT 1 FROM knowledge_batches batch
                        JOIN knowledge_batch_messages member
                          ON member.batch_id = batch.batch_id
                        WHERE batch.ai_task_id = task.task_id
                          AND member.profile_id = ? AND member.instance_id = ?
                          AND member.message_id = ?
                    )
                )
                OR (
                    task.task_type = 'DIALOGUE_SUMMARY'
                    AND CAST(COALESCE(json_extract(
                        task.input_json, '$.payload.target_message_id'
                    ), 0) AS INTEGER) >= ?
                )
                OR (
                    task.task_type = 'VISION_DESCRIPTION'
                    AND EXISTS (
                        SELECT 1 FROM json_each(
                            task.input_json, '$.payload.asset_ids'
                        ) input
                        JOIN media_asset_message_links link
                          ON link.asset_id = CAST(input.value AS TEXT)
                        WHERE link.profile_id = ? AND link.instance_id = ?
                          AND link.message_id = ?
                    )
                )
                OR (
                    task.task_type = 'MAIN_CORE'
                    AND (
                        CAST(COALESCE(json_extract(
                            task.input_json,
                            '$.payload.metadata.context_message_id'
                        ), 0) AS INTEGER) = ?
                        OR EXISTS (
                            SELECT 1 FROM json_each(
                                COALESCE(json_extract(
                                    task.input_json,
                                    '$.payload.metadata.context_message_ids'
                                ), '[]')
                            ) input_message
                            WHERE CAST(input_message.value AS INTEGER) = ?
                        )
                    )
                )
            )""",
            (
                hold.profile_id,
                hold.instance_id,
                hold.profile_id,
                hold.instance_id,
                hold.ledger_message_id,
                hold.ledger_message_id,
                hold.profile_id,
                hold.instance_id,
                hold.ledger_message_id,
                hold.ledger_message_id,
                hold.ledger_message_id,
            ),
        )
    )


def _find_durable_workflows(
    conn: sqlite3.Connection,
    hold: InboundRecallHold,
    task_ids: set[int],
) -> set[int]:
    if not task_ids:
        return set()
    placeholders = ",".join("?" for _ in task_ids)
    rows = conn.execute(
        f"""SELECT workflow_id FROM ai_workflows
        WHERE profile_id = ? AND instance_id = ?
          AND trigger_kind = 'DURABLE_TASK'
          AND CAST(trigger_ref AS INTEGER) IN ({placeholders})""",
        (hold.profile_id, hold.instance_id, *sorted(task_ids)),
    )
    return {int(row["workflow_id"]) for row in rows}


def _redact_core_runs(
    conn: sqlite3.Connection,
    runs: list[sqlite3.Row],
    redacted: str,
    now_text: str,
) -> None:
    for run in runs:
        conn.execute(
            """UPDATE instance_core_runs SET request_json = ?, decision_json = NULL,
            status = CASE WHEN status = 'RUNNING' THEN 'SUPERSEDED' ELSE status END,
            error = CASE WHEN status = 'RUNNING'
                THEN 'source_message_recalled' ELSE error END,
            finished_at = CASE WHEN status = 'RUNNING'
                THEN COALESCE(finished_at, ?) ELSE finished_at END
            WHERE run_id = ?""",
            (redacted, now_text, int(run["run_id"])),
        )


def _redact_workflows(
    conn: sqlite3.Connection,
    workflow_ids: set[int],
    redacted: str,
    now_text: str,
) -> None:
    for workflow_id in sorted(workflow_ids):
        conn.execute(
            """UPDATE ai_workflows SET
            status = CASE WHEN status = 'RUNNING' THEN 'INTERRUPTED' ELSE status END,
            final_error_code = 'SOURCE_MESSAGE_RECALLED',
            final_message = '源消息撤回后，相关模型记录已脱敏',
            finished_at = COALESCE(finished_at, ?), updated_at = ?,
            version = version + 1
            WHERE workflow_id = ?""",
            (now_text, now_text, workflow_id),
        )
        _redact_workflow_children(conn, workflow_id, redacted, now_text)


def _redact_workflow_children(
    conn: sqlite3.Connection,
    workflow_id: int,
    redacted: str,
    now_text: str,
) -> None:
    conn.execute(
        """UPDATE ai_provider_attempts SET request_json = NULL,
        response_json = NULL, transport_json = '{}', evaluation_json = NULL,
        status = CASE WHEN status IN ('PREPARING','IN_FLIGHT')
            THEN 'INTERRUPTED' ELSE status END,
        error_code = 'SOURCE_MESSAGE_RECALLED',
        error_message = 'source_message_recalled',
        finished_at = CASE WHEN status IN ('PREPARING','IN_FLIGHT')
            THEN COALESCE(finished_at, ?) ELSE finished_at END
        WHERE node_id IN (
            SELECT node_id FROM ai_work_nodes WHERE workflow_id = ?
        )""",
        (now_text, workflow_id),
    )
    conn.execute(
        """UPDATE ai_work_nodes SET input_json = ?, result_json = NULL,
        status = CASE WHEN status = 'RUNNING' THEN 'INTERRUPTED' ELSE status END,
        error_code = 'SOURCE_MESSAGE_RECALLED',
        error_message = 'source_message_recalled',
        summary = '', finished_at = CASE WHEN status = 'RUNNING'
            THEN COALESCE(finished_at, ?) ELSE finished_at END
        WHERE workflow_id = ?""",
        (redacted, now_text, workflow_id),
    )
    conn.execute(
        """UPDATE ai_work_events SET summary = '源消息撤回后已脱敏',
        details_json = '{}' WHERE workflow_id = ?""",
        (workflow_id,),
    )


def _find_affected_task_ids(
    conn: sqlite3.Connection,
    workflow_ids: set[int],
    related_task_ids: set[int],
) -> set[int]:
    affected = set(related_task_ids)
    if not workflow_ids:
        return affected
    placeholders = ",".join("?" for _ in workflow_ids)
    affected.update(
        int(row["task_id"])
        for row in conn.execute(
            f"""SELECT task_id FROM ai_tasks
            WHERE workflow_id IN ({placeholders})
               OR caused_by_workflow_id IN ({placeholders})""",
            (*sorted(workflow_ids), *sorted(workflow_ids)),
        )
    )
    return affected


def _redact_tasks(
    conn: sqlite3.Connection,
    task_ids: set[int],
    now_text: str,
) -> None:
    if not task_ids:
        return
    payloads = {
        kind: json.dumps(
            {
                "schema_version": 1,
                "kind": kind,
                "payload": {"redacted": True, "reason": "source_message_recalled"},
            },
            separators=(",", ":"),
        )
        for kind in ("input", "checkpoint", "progress")
    }
    placeholders = ",".join("?" for _ in task_ids)
    conn.execute(
        """UPDATE ai_tasks SET input_json = ?, checkpoint_json = ?,
        result_json = NULL, progress_json = ?,
        last_error = CASE WHEN status NOT IN ('SUCCEEDED','FAILED','CANCELLED')
            THEN 'source_message_recalled' ELSE last_error END,
        status = CASE WHEN status NOT IN ('SUCCEEDED','FAILED','CANCELLED')
            THEN 'CANCELLED' ELSE status END,
        lease_owner = CASE WHEN status NOT IN ('SUCCEEDED','FAILED','CANCELLED')
            THEN NULL ELSE lease_owner END,
        lease_until = CASE WHEN status NOT IN ('SUCCEEDED','FAILED','CANCELLED')
            THEN NULL ELSE lease_until END,
        finished_at = CASE WHEN status NOT IN ('SUCCEEDED','FAILED','CANCELLED')
            THEN COALESCE(finished_at, ?) ELSE finished_at END,
        updated_at = ?
        WHERE task_id IN ("""
        + placeholders
        + ")",
        (
            payloads["input"],
            payloads["checkpoint"],
            payloads["progress"],
            now_text,
            now_text,
            *sorted(task_ids),
        ),
    )
    conn.execute(
        """UPDATE ai_task_audit SET details_json = '{}'
        WHERE task_id IN ("""
        + placeholders
        + ")",
        tuple(sorted(task_ids)),
    )


__all__ = ["redact_model_artifacts"]
