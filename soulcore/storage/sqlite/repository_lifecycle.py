from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from ...contracts.ai_task_payload import decode_task_payload, encode_task_payload
from ...contracts.runtime_limits import DURABLE_AI_MAX_ATTEMPTS
from ...features.ai.sqlite.support import AI_TASK_RETRY_HOURS
from ...features.knowledge.sqlite.support import (
    KNOWLEDGE_TASK_TYPE,
    KNOWLEDGE_TERMINAL_EXCLUDED_STATUSES,
    _estimate_knowledge_tokens,
)
from ...features.knowledge.sqlite.support import (
    _context_eligible_sql as knowledge_context_eligible_sql,
)
from .codec import _dt
from .dialogue_turns import dialogue_turn_key_sql

KNOWLEDGE_ACTIVE_TASK_STATUSES = (
    "SCHEDULED",
    "READY",
    "RUNNING",
    "PAUSE_REQUESTED",
    "PAUSED",
    "CANCEL_REQUESTED",
    "RETRY_WAIT",
    "RECOVERY_REQUIRED",
)
KNOWLEDGE_SCHEDULE_TURN_COUNT = 4
KNOWLEDGE_IMMEDIATE_TURN_COUNT = 24


def _ensure_processing_state(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    now: str | None,
) -> sqlite3.Row:
    state = conn.execute(
        """SELECT * FROM knowledge_processing_state
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    if state is not None:
        return state
    baseline = conn.execute(
        """SELECT COALESCE(MAX(message_id), 0) AS value
        FROM instance_messages WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()["value"]
    conn.execute(
        """INSERT INTO knowledge_processing_state(
            profile_id, instance_id, baseline_message_id,
            committed_through_message_id, desired_through_message_id,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            profile_id,
            instance_id,
            int(baseline or 0),
            int(baseline or 0),
            int(baseline or 0),
            now,
            now,
        ),
    )
    created = conn.execute(
        """SELECT * FROM knowledge_processing_state
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    assert created is not None
    return created


def _mark_terminal_outbound(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    baseline_message_id: int,
    now: str | None,
) -> None:
    placeholders = ",".join("?" for _ in KNOWLEDGE_TERMINAL_EXCLUDED_STATUSES)
    conn.execute(
        f"""INSERT OR IGNORE INTO knowledge_message_marks(
            profile_id, instance_id, message_id, outcome, reason, marked_at
        )
        SELECT profile_id, instance_id, message_id, 'TERMINAL_EXCLUDED',
            'delivery_terminal_failure', ?
        FROM instance_messages m
        WHERE m.profile_id = ? AND m.instance_id = ?
          AND m.message_id > ? AND m.direction = 'OUTBOUND'
          AND m.delivery_status IN ({placeholders})""",
        (
            now,
            profile_id,
            instance_id,
            baseline_message_id,
            *KNOWLEDGE_TERMINAL_EXCLUDED_STATUSES,
        ),
    )


def _unmarked_knowledge_rows(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    baseline_message_id: int,
) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            f"""SELECT m.message_id, m.direction, m.plain_text,
                m.components_json, m.occurred_at,
                {dialogue_turn_key_sql()} AS dialogue_turn_key,
                EXISTS (
                    SELECT 1 FROM instance_messages processed
                    JOIN knowledge_message_marks processed_mark
                      ON processed_mark.profile_id = processed.profile_id
                     AND processed_mark.instance_id = processed.instance_id
                     AND processed_mark.message_id = processed.message_id
                    WHERE processed.profile_id = m.profile_id
                      AND processed.instance_id = m.instance_id
                      AND processed_mark.outcome IN ('PROCESSED', 'NO_KNOWLEDGE')
                      AND {dialogue_turn_key_sql("processed")} = {dialogue_turn_key_sql()}
                ) AS turn_already_processed
            FROM instance_messages m
            LEFT JOIN knowledge_message_marks mark
              ON mark.profile_id = m.profile_id AND mark.instance_id = m.instance_id
             AND mark.message_id = m.message_id
            WHERE m.profile_id = ? AND m.instance_id = ?
              AND m.message_id > ? AND mark.message_id IS NULL
              AND m.knowledge_eligibility = 'ELIGIBLE'
              AND {knowledge_context_eligible_sql()}
            ORDER BY m.message_id""",
            (profile_id, instance_id, baseline_message_id),
        )
    )


def _knowledge_speaker_turn_count(
    rows: list[sqlite3.Row],
) -> int:
    """Count speaker turns without charging once per chat bubble."""

    return len(
        {str(row["dialogue_turn_key"]) for row in rows if not bool(row["turn_already_processed"])}
    )


def _coalesce_active_task(
    conn: sqlite3.Connection,
    active: sqlite3.Row,
    desired: int,
    due: str | None,
    due_dt: datetime,
    now_dt: datetime,
    now: str | None,
) -> sqlite3.Row:
    input_data = decode_task_payload("input", active["input_json"])
    input_data["desired_through_message_id"] = max(
        int(input_data.get("desired_through_message_id") or 0), desired
    )
    if active["status"] in {"SCHEDULED", "READY"}:
        next_status = "READY" if due_dt <= now_dt else "SCHEDULED"
        conn.execute(
            """UPDATE ai_tasks SET input_json = ?, due_at = ?, status = ?,
                updated_at = ?, version = version + 1 WHERE task_id = ?""",
            (
                encode_task_payload("input", input_data),
                due,
                next_status,
                now,
                int(active["task_id"]),
            ),
        )
    else:
        conn.execute(
            """UPDATE ai_tasks SET input_json = ?, updated_at = ?,
                version = version + 1 WHERE task_id = ?""",
            (encode_task_payload("input", input_data), now, int(active["task_id"])),
        )
    updated = conn.execute(
        "SELECT * FROM ai_tasks WHERE task_id = ?", (int(active["task_id"]),)
    ).fetchone()
    assert updated is not None
    return updated


def _create_knowledge_task(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    desired: int,
    due: str | None,
    immediate: bool,
    now: str | None,
) -> sqlite3.Row:
    previous = conn.execute(
        """SELECT COALESCE(MAX(generation), 0) AS value FROM ai_tasks
        WHERE profile_id = ? AND instance_id = ? AND task_type = ?""",
        (profile_id, instance_id, KNOWLEDGE_TASK_TYPE),
    ).fetchone()
    generation = int(previous["value"] or 0) + 1
    cursor = conn.execute(
        """INSERT INTO ai_tasks(
            profile_id, instance_id, task_type, task_class, capability,
            status, priority, due_at, mutex_key,
            idempotency_key, generation, input_json, checkpoint_json,
            retry_policy_json, recovery_policy, max_attempts,
            created_at, updated_at
        ) VALUES (?, ?, ?, 'BACKGROUND', 'text.completion', ?, -30, ?,
            ?, ?, ?, ?, ?, ?, 'RESUME_CHECKPOINT', ?, ?, ?)""",
        (
            profile_id,
            instance_id,
            KNOWLEDGE_TASK_TYPE,
            "READY" if immediate else "SCHEDULED",
            due,
            f"knowledge-formation:{instance_id}",
            f"knowledge-formation:{instance_id}",
            generation,
            encode_task_payload("input", {"desired_through_message_id": desired}),
            encode_task_payload("checkpoint", {}),
            encode_task_payload("retry_policy", {"delays_hours": list(AI_TASK_RETRY_HOURS)}),
            DURABLE_AI_MAX_ATTEMPTS,
            now,
            now,
        ),
    )
    task_id = int(cursor.lastrowid)
    created = conn.execute("SELECT * FROM ai_tasks WHERE task_id = ?", (task_id,)).fetchone()
    assert created is not None
    return created


class KnowledgeTaskSql:
    @staticmethod
    def _refresh_knowledge_task_sql(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        *,
        now_dt: datetime,
        force: bool = False,
    ) -> sqlite3.Row | None:
        """Coalesce new ledger work into the one durable formation task.

        Message marks, rather than a high-water cursor, are authoritative. This
        deliberately re-scans older unmarked rows so an outbound PENDING row
        promoted to an eligible delivery status cannot be lost forever.
        """

        now = _dt(now_dt)
        state = _ensure_processing_state(conn, profile_id, instance_id, now)
        baseline = int(state["baseline_message_id"])
        _mark_terminal_outbound(conn, profile_id, instance_id, baseline, now)
        rows = _unmarked_knowledge_rows(conn, profile_id, instance_id, baseline)
        if not rows:
            return None
        token_count = sum(
            _estimate_knowledge_tokens(row["plain_text"])
            + _estimate_knowledge_tokens(row["components_json"])
            for row in rows
        )
        turn_count = _knowledge_speaker_turn_count(rows)
        if not force and turn_count < KNOWLEDGE_SCHEDULE_TURN_COUNT and token_count < 512:
            conn.execute(
                """UPDATE knowledge_processing_state SET last_message_at = ?,
                    desired_through_message_id = MAX(desired_through_message_id, ?),
                    updated_at = ? WHERE profile_id = ? AND instance_id = ?""",
                (now, int(rows[-1]["message_id"]), now, profile_id, instance_id),
            )
            return None
        immediate = force or turn_count >= KNOWLEDGE_IMMEDIATE_TURN_COUNT or token_count >= 4096
        due_dt = now_dt if immediate else now_dt + timedelta(minutes=5)
        due = _dt(due_dt)
        desired = int(rows[-1]["message_id"])
        conn.execute(
            """UPDATE knowledge_processing_state SET last_message_at = ?,
                last_trigger_at = ?,
                desired_through_message_id = MAX(desired_through_message_id, ?),
                updated_at = ? WHERE profile_id = ? AND instance_id = ?""",
            (now, now, desired, now, profile_id, instance_id),
        )
        active_placeholders = ",".join("?" for _ in KNOWLEDGE_ACTIVE_TASK_STATUSES)
        active = conn.execute(
            f"""SELECT * FROM ai_tasks WHERE profile_id = ? AND instance_id = ?
              AND task_type = ? AND status IN ({active_placeholders})
            ORDER BY generation DESC, task_id DESC LIMIT 1""",
            (
                profile_id,
                instance_id,
                KNOWLEDGE_TASK_TYPE,
                *KNOWLEDGE_ACTIVE_TASK_STATUSES,
            ),
        ).fetchone()
        if active is not None:
            updated = _coalesce_active_task(conn, active, desired, due, due_dt, now_dt, now)
            conn.execute(
                """UPDATE knowledge_processing_state SET active_task_id = ?,
                    updated_at = ? WHERE profile_id = ? AND instance_id = ?""",
                (int(active["task_id"]), now, profile_id, instance_id),
            )
            return updated

        created = _create_knowledge_task(
            conn, profile_id, instance_id, desired, due, immediate, now
        )
        task_id = int(created["task_id"])
        conn.execute(
            """UPDATE knowledge_processing_state SET active_task_id = ?,
                updated_at = ? WHERE profile_id = ? AND instance_id = ?""",
            (task_id, now, profile_id, instance_id),
        )
        return created


class ContextBackupSql:
    async def publish_context_backup(self) -> str | None:
        """Publish a stable snapshot without reclassifying the committed write."""

        path = await self.db.publish_backup_after_commit(operation="context_commit")
        return str(path) if path is not None else None
