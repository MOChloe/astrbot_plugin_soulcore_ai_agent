from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any

from ....contracts.models import (
    MessageRetractionAction,
    MessageRetractionStatus,
    PlatformMessageFragment,
)
from ....storage.sqlite.codec import _dt, _now, _parse
from .message_retraction_recovery import MessageRetractionRecoveryMixin
from .message_retraction_transactions import (
    _action,
    _refresh_ledger_retraction_eligibility,
    _resolved_target_fragment_rows,
)

_RETRACTION_TERMINAL = {
    MessageRetractionStatus.RETRACTED,
    MessageRetractionStatus.FAILED,
    MessageRetractionStatus.UNKNOWN_AFTER_CRASH,
    MessageRetractionStatus.CANCELLED,
}
_RETRACTION_TRANSITIONS = {
    MessageRetractionStatus.PENDING: {
        MessageRetractionStatus.SENDING,
        MessageRetractionStatus.FAILED,
        MessageRetractionStatus.CANCELLED,
    },
    MessageRetractionStatus.SENDING: {
        MessageRetractionStatus.RETRACTED,
        MessageRetractionStatus.FAILED,
        MessageRetractionStatus.UNKNOWN_AFTER_CRASH,
    },
}


class MessageRetractionRecords(MessageRetractionRecoveryMixin):
    async def create_retraction_action(
        self,
        profile_id: str,
        instance_id: str,
        *,
        source_run_id: int,
        expression_batch_id: str,
        step_ordinal: int,
        idempotency_key: str,
        target_message_ref: str | None = None,
        target_output_ordinal: int | None = None,
        delay_after_previous_seconds: int = 0,
        not_before_at: datetime | None = None,
        now: datetime | None = None,
    ) -> MessageRetractionAction:
        current = now or _now()
        target_ref = str(target_message_ref or "").strip() or None
        target_output = int(target_output_ordinal) if target_output_ordinal is not None else None
        if (target_ref is None) == (target_output is None):
            raise ValueError("exactly one retraction target is required")
        if target_output is not None and target_output < 1:
            raise ValueError("target_output_ordinal is 1-based and must be positive")
        step = int(step_ordinal)
        delay = int(delay_after_previous_seconds)
        if step < 0:
            raise ValueError("step_ordinal cannot be negative")
        if not 0 <= delay <= 120:
            raise ValueError("retraction delay must be between zero and 120 seconds")
        key = str(idempotency_key or "").strip()
        batch = str(expression_batch_id or "").strip()
        if not key or not batch:
            raise ValueError("expression batch and idempotency key are required")

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            existing = conn.execute(
                """SELECT * FROM message_retraction_actions WHERE profile_id = ?
                AND instance_id = ? AND idempotency_key = ?""",
                (profile_id, instance_id, key),
            ).fetchone()
            if existing is not None:
                expected = (int(source_run_id), batch, step, target_ref, target_output)
                actual = (
                    int(existing["source_run_id"]),
                    str(existing["expression_batch_id"]),
                    int(existing["step_ordinal"]),
                    existing["target_message_ref"],
                    existing["target_output_ordinal"],
                )
                if actual != expected:
                    raise ValueError("retraction action replay conflicts with stored identity")
                return existing
            if target_ref is not None:
                self._validate_existing_target(conn, profile_id, instance_id, target_ref, current)
                duplicate = conn.execute(
                    """SELECT 1 FROM message_retraction_actions
                    WHERE profile_id = ? AND instance_id = ? AND target_message_ref = ?""",
                    (profile_id, instance_id, target_ref),
                ).fetchone()
            else:
                self._validate_output_target(
                    conn,
                    profile_id,
                    instance_id,
                    int(source_run_id),
                    batch,
                    step,
                    int(target_output),
                )
                duplicate = conn.execute(
                    """SELECT 1 FROM message_retraction_actions
                    WHERE profile_id = ? AND instance_id = ? AND source_run_id = ?
                      AND target_output_ordinal = ?""",
                    (profile_id, instance_id, int(source_run_id), int(target_output)),
                ).fetchone()
            if duplicate is not None:
                raise ValueError("retraction target already has a durable action")
            now_text = _dt(current)
            cursor = conn.execute(
                """INSERT INTO message_retraction_actions(
                    profile_id, instance_id, source_run_id, expression_batch_id,
                    step_ordinal, target_message_ref, target_output_ordinal,
                    delay_after_previous_seconds, not_before_at, idempotency_key,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    profile_id,
                    instance_id,
                    int(source_run_id),
                    batch,
                    step,
                    target_ref,
                    target_output,
                    delay,
                    _dt(not_before_at or current),
                    key,
                    now_text,
                    now_text,
                ),
            )
            row = conn.execute(
                "SELECT * FROM message_retraction_actions WHERE action_id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
            assert row is not None
            return row

        return _action(await self.uow.run(operation))

    async def get_retraction_action(
        self, profile_id: str, instance_id: str, action_id: int
    ) -> MessageRetractionAction | None:
        row = await self.db.fetch_one(
            """SELECT * FROM message_retraction_actions
            WHERE profile_id = ? AND instance_id = ? AND action_id = ?""",
            (profile_id, instance_id, int(action_id)),
        )
        return _action(row) if row else None

    async def list_retraction_actions(
        self,
        profile_id: str,
        instance_id: str,
        *,
        expression_batch_id: str | None = None,
        status: MessageRetractionStatus | str | None = None,
        limit: int = 100,
    ) -> list[MessageRetractionAction]:
        clauses = ["profile_id = ?", "instance_id = ?"]
        params: list[Any] = [profile_id, instance_id]
        if expression_batch_id is not None:
            clauses.append("expression_batch_id = ?")
            params.append(str(expression_batch_id))
        if status is not None:
            clauses.append("status = ?")
            params.append(MessageRetractionStatus(str(status)).value)
        params.append(max(1, min(int(limit), 500)))
        rows = await self.db.fetch_all(
            f"""SELECT * FROM message_retraction_actions
            WHERE {" AND ".join(clauses)} ORDER BY action_id DESC LIMIT ?""",
            params,
        )
        return [_action(row) for row in rows]

    async def list_due_retraction_actions(
        self,
        *,
        now: datetime | None = None,
        limit: int = 100,
    ) -> list[MessageRetractionAction]:
        rows = await self.db.fetch_all(
            """SELECT action.* FROM message_retraction_actions action
            JOIN instance_expression_batches batch
              ON batch.batch_id = action.expression_batch_id
            JOIN role_profiles profile ON profile.profile_id = action.profile_id
            WHERE action.status = 'PENDING' AND batch.status = 'ACTIVE'
              AND profile.enabled = 1
              AND (action.not_before_at IS NULL OR action.not_before_at <= ?)
              AND NOT EXISTS (
                SELECT 1 FROM instance_outbox predecessor
                WHERE predecessor.expression_batch_id = action.expression_batch_id
                  AND predecessor.expression_step_ordinal < action.step_ordinal
                  AND predecessor.status IN ('PENDING', 'SENDING')
              )
              AND NOT EXISTS (
                SELECT 1 FROM message_retraction_actions predecessor
                WHERE predecessor.expression_batch_id = action.expression_batch_id
                  AND predecessor.step_ordinal < action.step_ordinal
                  AND predecessor.status IN ('PENDING', 'SENDING')
              )
              AND NOT EXISTS (
                SELECT 1
                FROM instance_expression_batches previous
                JOIN instance_outbox predecessor
                  ON predecessor.expression_batch_id = previous.batch_id
                WHERE previous.profile_id = batch.profile_id
                  AND previous.instance_id = batch.instance_id
                  AND previous.source_run_id = batch.source_run_id
                  AND previous.segment_index < batch.segment_index
                  AND predecessor.status IN ('PENDING', 'SENDING')
              )
              AND NOT EXISTS (
                SELECT 1
                FROM instance_expression_batches previous
                JOIN message_retraction_actions predecessor
                  ON predecessor.expression_batch_id = previous.batch_id
                WHERE previous.profile_id = batch.profile_id
                  AND previous.instance_id = batch.instance_id
                  AND previous.source_run_id = batch.source_run_id
                  AND previous.segment_index < batch.segment_index
                  AND predecessor.status IN ('PENDING', 'SENDING')
              )
            ORDER BY action.not_before_at, action.action_id LIMIT ?""",
            (_dt(now or _now()), max(1, min(int(limit), 500))),
        )
        return [_action(row) for row in rows]

    async def claim_retraction_action(
        self,
        profile_id: str,
        instance_id: str,
        action_id: int,
        *,
        now: datetime | None = None,
    ) -> MessageRetractionAction:
        current = now or _now()

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            row = conn.execute(
                """SELECT * FROM message_retraction_actions
                WHERE profile_id = ? AND instance_id = ? AND action_id = ?""",
                (profile_id, instance_id, int(action_id)),
            ).fetchone()
            if row is None:
                raise KeyError((profile_id, instance_id, action_id))
            if MessageRetractionStatus(row["status"]) is not MessageRetractionStatus.PENDING:
                raise ValueError("retraction action status conflict")
            fragments = _resolved_target_fragment_rows(conn, row)
            if not fragments:
                raise ValueError("retraction target has no platform fragments")
            now_text = _dt(current)
            for fragment in fragments:
                conn.execute(
                    """INSERT INTO message_retraction_fragment_attempts(
                        action_id, message_ref, status, created_at, updated_at
                    ) VALUES (?, ?, 'PENDING', ?, ?)
                    ON CONFLICT(action_id, message_ref) DO NOTHING""",
                    (int(action_id), str(fragment["message_ref"]), now_text, now_text),
                )
            changed = conn.execute(
                """UPDATE message_retraction_actions SET status = 'SENDING',
                    attempted_at = COALESCE(attempted_at, ?), updated_at = ?
                WHERE action_id = ? AND status = 'PENDING'""",
                (now_text, now_text, int(action_id)),
            ).rowcount
            if changed != 1:
                raise ValueError("retraction action status conflict")
            updated = conn.execute(
                "SELECT * FROM message_retraction_actions WHERE action_id = ?",
                (int(action_id),),
            ).fetchone()
            assert updated is not None
            return updated

        return _action(await self.uow.run(operation))

    async def claim_retraction_fragment(
        self,
        profile_id: str,
        instance_id: str,
        action_id: int,
        message_ref: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        now_text = _dt(now or _now())

        def operation(conn: sqlite3.Connection) -> bool:
            action = conn.execute(
                """SELECT status FROM message_retraction_actions
                WHERE profile_id = ? AND instance_id = ? AND action_id = ?""",
                (profile_id, instance_id, int(action_id)),
            ).fetchone()
            if action is None:
                raise KeyError((profile_id, instance_id, action_id))
            if action["status"] != MessageRetractionStatus.SENDING.value:
                return False
            changed = conn.execute(
                """UPDATE message_retraction_fragment_attempts
                SET status = 'SENDING', attempted_at = ?, updated_at = ?
                WHERE action_id = ? AND message_ref = ? AND status = 'PENDING'""",
                (now_text, now_text, int(action_id), str(message_ref)),
            ).rowcount
            if changed:
                conn.execute(
                    """UPDATE instance_message_fragments
                    SET retraction_status = 'SENDING', updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND message_ref = ?""",
                    (now_text, profile_id, instance_id, str(message_ref)),
                )
            return changed == 1

        return await self.uow.run(operation)

    async def settle_retraction_fragment(
        self,
        profile_id: str,
        instance_id: str,
        action_id: int,
        message_ref: str,
        status: MessageRetractionStatus | str,
        *,
        error_code: str = "",
        now: datetime | None = None,
    ) -> bool:
        target = MessageRetractionStatus(str(status))
        if target not in {
            MessageRetractionStatus.RETRACTED,
            MessageRetractionStatus.FAILED,
            MessageRetractionStatus.UNKNOWN_AFTER_CRASH,
        }:
            raise ValueError("retraction fragment settlement must be terminal")
        now_text = _dt(now or _now())

        def operation(conn: sqlite3.Connection) -> bool:
            fragment = conn.execute(
                """SELECT fragment.ledger_message_id
                FROM message_retraction_fragment_attempts attempt
                JOIN message_retraction_actions action ON action.action_id = attempt.action_id
                JOIN instance_message_fragments fragment ON fragment.message_ref = attempt.message_ref
                WHERE attempt.action_id = ? AND attempt.message_ref = ?
                  AND action.profile_id = ? AND action.instance_id = ?
                  AND action.status = 'SENDING'""",
                (int(action_id), str(message_ref), profile_id, instance_id),
            ).fetchone()
            if fragment is None:
                return False
            changed = conn.execute(
                """UPDATE message_retraction_fragment_attempts
                SET status = ?, completed_at = ?, error_code = ?, updated_at = ?
                WHERE action_id = ? AND message_ref = ? AND status = 'SENDING'""",
                (
                    target.value,
                    now_text,
                    str(error_code)[:120],
                    now_text,
                    int(action_id),
                    str(message_ref),
                ),
            ).rowcount
            if changed != 1:
                return False
            conn.execute(
                """UPDATE instance_message_fragments SET retraction_status = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND message_ref = ?""",
                (target.value, now_text, profile_id, instance_id, str(message_ref)),
            )
            _refresh_ledger_retraction_eligibility(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                ledger_message_id=int(fragment["ledger_message_id"]),
            )
            return True

        return await self.uow.run(operation)

    async def release_retraction_action(
        self,
        profile_id: str,
        instance_id: str,
        action_id: int,
        *,
        now: datetime | None = None,
    ) -> bool:
        now_text = _dt(now or _now())

        def operation(conn: sqlite3.Connection) -> bool:
            active = conn.execute(
                """SELECT 1 FROM message_retraction_fragment_attempts
                WHERE action_id = ? AND status = 'SENDING' LIMIT 1""",
                (int(action_id),),
            ).fetchone()
            if active is not None:
                return False
            return (
                conn.execute(
                    """UPDATE message_retraction_actions SET status = 'PENDING', updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND action_id = ?
                      AND status = 'SENDING'""",
                    (now_text, profile_id, instance_id, int(action_id)),
                ).rowcount
                == 1
            )

        return await self.uow.run(operation)

    async def finalize_retraction_action(
        self,
        profile_id: str,
        instance_id: str,
        action_id: int,
        *,
        now: datetime | None = None,
    ) -> MessageRetractionAction:
        current = now or _now()

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            row = conn.execute(
                """SELECT * FROM message_retraction_actions
                WHERE profile_id = ? AND instance_id = ? AND action_id = ?""",
                (profile_id, instance_id, int(action_id)),
            ).fetchone()
            if row is None:
                raise KeyError((profile_id, instance_id, action_id))
            if row["status"] != MessageRetractionStatus.SENDING.value:
                return row
            attempts = conn.execute(
                """SELECT status, error_code FROM message_retraction_fragment_attempts
                WHERE action_id = ? ORDER BY message_ref""",
                (int(action_id),),
            ).fetchall()
            if not attempts:
                raise RuntimeError("claimed retraction has no physical fragment attempts")
            if any(item["status"] in {"PENDING", "SENDING"} for item in attempts):
                return row
            statuses = {str(item["status"]) for item in attempts}
            if MessageRetractionStatus.UNKNOWN_AFTER_CRASH.value in statuses:
                target = MessageRetractionStatus.UNKNOWN_AFTER_CRASH
            elif statuses == {MessageRetractionStatus.RETRACTED.value}:
                target = MessageRetractionStatus.RETRACTED
            elif MessageRetractionStatus.FAILED.value in statuses:
                target = MessageRetractionStatus.FAILED
            else:
                target = MessageRetractionStatus.CANCELLED
            errors = [str(item["error_code"] or "") for item in attempts if item["error_code"]]
            return self._settle_action_row(
                conn,
                row,
                target,
                _dt(current),
                error_code=(errors[0] if errors else ""),
                update_fragments=False,
            )

        return _action(await self.uow.run(operation))

    async def next_retraction_action_due_at(self) -> datetime | None:
        row = await self.db.fetch_one(
            """SELECT action.not_before_at, action.created_at
            FROM message_retraction_actions action
            JOIN instance_expression_batches batch
              ON batch.batch_id = action.expression_batch_id
            JOIN role_profiles profile ON profile.profile_id = action.profile_id
            WHERE action.status = 'PENDING' AND batch.status = 'ACTIVE'
              AND profile.enabled = 1
            ORDER BY COALESCE(action.not_before_at, action.created_at), action.action_id
            LIMIT 1"""
        )
        if row is None:
            return None
        return _parse(row["not_before_at"]) or _parse(row["created_at"])

    async def resolve_retraction_target_fragments(
        self,
        profile_id: str,
        instance_id: str,
        action_id: int,
    ) -> list[PlatformMessageFragment]:
        row = await self.db.fetch_one(
            """SELECT * FROM message_retraction_actions
            WHERE profile_id = ? AND instance_id = ? AND action_id = ?""",
            (profile_id, instance_id, int(action_id)),
        )
        if row is None:
            raise KeyError((profile_id, instance_id, action_id))
        if row["target_message_ref"] is not None:
            fragment = await self.get_message_fragment(
                profile_id, instance_id, str(row["target_message_ref"])
            )
            return [fragment] if fragment is not None else []
        source_batch = await self.db.fetch_one(
            """SELECT source_run_id FROM instance_expression_batches
            WHERE profile_id = ? AND instance_id = ? AND batch_id = ?""",
            (profile_id, instance_id, str(row["expression_batch_id"])),
        )
        if source_batch is None:
            return []
        global_ordinal = int(row["target_output_ordinal"])
        batches = await self.db.fetch_all(
            """SELECT batch_id, output_count FROM instance_expression_batches
            WHERE profile_id = ? AND instance_id = ? AND source_run_id = ?
            ORDER BY segment_index""",
            (profile_id, instance_id, int(source_batch["source_run_id"])),
        )
        consumed = 0
        for batch in batches:
            output_count = int(batch["output_count"] or 0)
            if global_ordinal <= consumed + output_count:
                return await self.list_message_fragments_for_expression_output(
                    profile_id,
                    instance_id,
                    str(batch["batch_id"]),
                    global_ordinal - consumed,
                )
            consumed += output_count
        return []

    async def transition_retraction_action(
        self,
        profile_id: str,
        instance_id: str,
        action_id: int,
        status: MessageRetractionStatus | str,
        *,
        expected_status: MessageRetractionStatus | str | None = None,
        error_code: str = "",
        now: datetime | None = None,
    ) -> MessageRetractionAction:
        target = MessageRetractionStatus(str(status))
        expected = MessageRetractionStatus(str(expected_status)) if expected_status else None
        current = now or _now()

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            row = conn.execute(
                """SELECT * FROM message_retraction_actions
                WHERE profile_id = ? AND instance_id = ? AND action_id = ?""",
                (profile_id, instance_id, int(action_id)),
            ).fetchone()
            if row is None:
                raise KeyError((profile_id, instance_id, action_id))
            current_status = MessageRetractionStatus(row["status"])
            if current_status is target:
                return row
            if expected is not None and current_status is not expected:
                raise ValueError("retraction action status conflict")
            if target not in _RETRACTION_TRANSITIONS.get(current_status, set()):
                raise ValueError(f"invalid retraction transition: {current_status} -> {target}")
            now_text = _dt(current)
            if target is MessageRetractionStatus.SENDING:
                raise ValueError("claim_retraction_action must start platform retraction")
            if current_status is MessageRetractionStatus.SENDING:
                conn.execute(
                    """UPDATE message_retraction_fragment_attempts
                    SET status = ?, completed_at = ?, error_code = ?, updated_at = ?
                    WHERE action_id = ? AND status IN ('PENDING', 'SENDING')""",
                    (
                        target.value,
                        now_text,
                        str(error_code)[:120],
                        now_text,
                        int(action_id),
                    ),
                )
            return self._settle_action_row(
                conn,
                row,
                target,
                now_text,
                error_code=str(error_code),
                update_fragments=True,
            )

        return _action(await self.uow.run(operation))


__all__ = ["MessageRetractionRecords"]
