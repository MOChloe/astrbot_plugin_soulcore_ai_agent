from __future__ import annotations

import sqlite3
from datetime import datetime

from ....contracts.models import MessageDirection, MessageRetractionAction, MessageRetractionStatus
from ....storage.sqlite.codec import _dt, _now, _parse
from ....storage.sqlite.expression_batch_lifecycle import sync_expression_batch_status
from .group_first_attempt import resolve_retract_only_group_window

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


def _action(row: sqlite3.Row) -> MessageRetractionAction:
    return MessageRetractionAction(
        action_id=int(row["action_id"]),
        profile_id=str(row["profile_id"]),
        instance_id=str(row["instance_id"]),
        source_run_id=int(row["source_run_id"]),
        expression_batch_id=str(row["expression_batch_id"]),
        step_ordinal=int(row["step_ordinal"]),
        idempotency_key=str(row["idempotency_key"]),
        status=MessageRetractionStatus(row["status"]),
        target_message_ref=(
            str(row["target_message_ref"]) if row["target_message_ref"] is not None else None
        ),
        target_output_ordinal=(
            int(row["target_output_ordinal"]) if row["target_output_ordinal"] is not None else None
        ),
        delay_after_previous_seconds=int(row["delay_after_previous_seconds"]),
        not_before_at=_parse(row["not_before_at"]),
        attempted_at=_parse(row["attempted_at"]),
        completed_at=_parse(row["completed_at"]),
        error_code=str(row["error_code"]),
        created_at=_parse(row["created_at"]),
        updated_at=_parse(row["updated_at"]),
    )


def _resolved_target_fragment_rows(
    conn: sqlite3.Connection, action: sqlite3.Row
) -> list[sqlite3.Row]:
    """Resolve a logical retraction target to its immutable physical fragments."""

    if action["target_message_ref"] is not None:
        row = conn.execute(
            """SELECT * FROM instance_message_fragments
            WHERE profile_id = ? AND instance_id = ? AND message_ref = ?""",
            (
                action["profile_id"],
                action["instance_id"],
                action["target_message_ref"],
            ),
        ).fetchone()
        return [row] if row is not None else []
    target_ordinal = int(action["target_output_ordinal"])
    batches = conn.execute(
        """SELECT batch_id, output_count FROM instance_expression_batches
        WHERE profile_id = ? AND instance_id = ? AND source_run_id = ?
        ORDER BY segment_index""",
        (
            action["profile_id"],
            action["instance_id"],
            int(action["source_run_id"]),
        ),
    ).fetchall()
    consumed = 0
    for batch in batches:
        output_count = int(batch["output_count"] or 0)
        if target_ordinal <= consumed + output_count:
            return list(
                conn.execute(
                    """SELECT fragment.* FROM instance_message_fragments fragment
                    JOIN instance_messages message
                      ON message.profile_id = fragment.profile_id
                     AND message.instance_id = fragment.instance_id
                     AND message.message_id = fragment.ledger_message_id
                    WHERE message.profile_id = ? AND message.instance_id = ?
                      AND message.expression_batch_id = ?
                      AND message.expression_ordinal = ?
                    ORDER BY fragment.fragment_ordinal""",
                    (
                        action["profile_id"],
                        action["instance_id"],
                        str(batch["batch_id"]),
                        target_ordinal - consumed - 1,
                    ),
                ).fetchall()
            )
        consumed += output_count
    return []


def _refresh_ledger_retraction_eligibility(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    ledger_message_id: int,
) -> None:
    rows = conn.execute(
        """SELECT retraction_status FROM instance_message_fragments
        WHERE profile_id = ? AND instance_id = ? AND ledger_message_id = ?
          AND direction = 'OUTBOUND'""",
        (profile_id, instance_id, int(ledger_message_id)),
    ).fetchall()
    statuses = {str(row["retraction_status"] or "") for row in rows}
    if MessageRetractionStatus.UNKNOWN_AFTER_CRASH.value in statuses:
        eligibility = "HELD"
        reason = "platform_message_retraction_unknown_after_crash"
    elif rows and statuses == {MessageRetractionStatus.RETRACTED.value}:
        eligibility = "EXCLUDED"
        reason = "platform_message_retraction_retracted"
    else:
        return
    conn.execute(
        """UPDATE instance_messages SET knowledge_eligibility = ?,
            knowledge_eligibility_reason = ?
        WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
        (eligibility, reason, profile_id, instance_id, int(ledger_message_id)),
    )


class MessageRetractionRecoveryMixin:
    async def recover_sending_retraction_actions(self) -> int:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            rows = conn.execute(
                "SELECT * FROM message_retraction_actions WHERE status = 'SENDING'"
            ).fetchall()
            for row in rows:
                sending = conn.execute(
                    """SELECT attempt.message_ref, fragment.ledger_message_id
                    FROM message_retraction_fragment_attempts attempt
                    JOIN instance_message_fragments fragment
                      ON fragment.message_ref = attempt.message_ref
                    WHERE attempt.action_id = ? AND attempt.status = 'SENDING'""",
                    (int(row["action_id"]),),
                ).fetchall()
                for attempt in sending:
                    conn.execute(
                        """UPDATE message_retraction_fragment_attempts
                        SET status = 'UNKNOWN_AFTER_CRASH', completed_at = ?,
                            error_code = 'worker_restarted_after_platform_call', updated_at = ?
                        WHERE action_id = ? AND message_ref = ? AND status = 'SENDING'""",
                        (now, now, int(row["action_id"]), str(attempt["message_ref"])),
                    )
                    conn.execute(
                        """UPDATE instance_message_fragments
                        SET retraction_status = 'UNKNOWN_AFTER_CRASH', updated_at = ?
                        WHERE message_ref = ?""",
                        (now, str(attempt["message_ref"])),
                    )
                    _refresh_ledger_retraction_eligibility(
                        conn,
                        profile_id=str(row["profile_id"]),
                        instance_id=str(row["instance_id"]),
                        ledger_message_id=int(attempt["ledger_message_id"]),
                    )
                attempts = conn.execute(
                    """SELECT status, error_code FROM message_retraction_fragment_attempts
                    WHERE action_id = ?""",
                    (int(row["action_id"]),),
                ).fetchall()
                statuses = {str(item["status"]) for item in attempts}
                if not attempts:
                    raise RuntimeError("claimed retraction has no physical fragment attempts")
                if MessageRetractionStatus.UNKNOWN_AFTER_CRASH.value in statuses:
                    conn.execute(
                        """UPDATE message_retraction_fragment_attempts
                        SET status = 'CANCELLED', completed_at = ?,
                            error_code = 'sibling_attempt_unknown_after_restart', updated_at = ?
                        WHERE action_id = ? AND status = 'PENDING'""",
                        (now, now, int(row["action_id"])),
                    )
                    self._settle_action_row(
                        conn,
                        row,
                        MessageRetractionStatus.UNKNOWN_AFTER_CRASH,
                        now,
                        error_code="worker_restarted_after_platform_call",
                        update_fragments=False,
                    )
                elif MessageRetractionStatus.PENDING.value in statuses:
                    conn.execute(
                        """UPDATE message_retraction_actions SET status = 'PENDING', updated_at = ?
                        WHERE action_id = ? AND status = 'SENDING'""",
                        (now, int(row["action_id"])),
                    )
                else:
                    target = (
                        MessageRetractionStatus.RETRACTED
                        if statuses == {MessageRetractionStatus.RETRACTED.value}
                        else MessageRetractionStatus.FAILED
                    )
                    self._settle_action_row(
                        conn,
                        row,
                        target,
                        now,
                        error_code="",
                        update_fragments=False,
                    )
            return len(rows)

        return await self.uow.run(operation)

    def _settle_action_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        status: MessageRetractionStatus,
        now: str,
        *,
        error_code: str,
        update_fragments: bool,
    ) -> sqlite3.Row:
        if status not in _RETRACTION_TERMINAL:
            raise ValueError("retraction action settlement must be terminal")
        conn.execute(
            """UPDATE message_retraction_actions SET status = ?, completed_at = ?,
                error_code = ?, updated_at = ? WHERE action_id = ?""",
            (status.value, now, str(error_code)[:120], now, int(row["action_id"])),
        )
        if update_fragments:
            self._update_target_fragments(conn, row, status, now)
        from .expression_outbox import defer_following_expression_step

        defer_following_expression_step(
            conn,
            str(row["expression_batch_id"]),
            int(row["step_ordinal"]),
            now,
        )
        sync_expression_batch_status(conn, str(row["expression_batch_id"]), now)
        resolve_retract_only_group_window(conn, str(row["expression_batch_id"]), now)
        updated = conn.execute(
            "SELECT * FROM message_retraction_actions WHERE action_id = ?",
            (int(row["action_id"]),),
        ).fetchone()
        assert updated is not None
        return updated

    @staticmethod
    def _validate_existing_target(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        message_ref: str,
        now: datetime,
    ) -> None:
        row = conn.execute(
            """SELECT * FROM instance_message_fragments
            WHERE profile_id = ? AND instance_id = ? AND message_ref = ?""",
            (profile_id, instance_id, message_ref),
        ).fetchone()
        if row is None:
            raise ValueError("retraction target is not available in this instance")
        if row["direction"] != MessageDirection.OUTBOUND.value:
            raise ValueError("only the assistant's outbound message may be retracted")
        if not bool(row["self_retraction_supported"]):
            raise ValueError("platform fragment does not support self retraction")
        deadline = _parse(row["retractable_until"])
        if deadline is not None and now >= deadline:
            raise ValueError("platform fragment retraction deadline has expired")
        if row["retraction_status"] in {
            MessageRetractionStatus.RETRACTED.value,
            MessageRetractionStatus.UNKNOWN_AFTER_CRASH.value,
        }:
            raise ValueError("platform fragment is no longer safely retractable")

    @staticmethod
    def _validate_output_target(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        source_run_id: int,
        expression_batch_id: str,
        step_ordinal: int,
        target_output_ordinal: int,
    ) -> None:
        current_batch = conn.execute(
            """SELECT segment_index FROM instance_expression_batches
            WHERE profile_id = ? AND instance_id = ? AND source_run_id = ?
              AND batch_id = ?""",
            (profile_id, instance_id, int(source_run_id), expression_batch_id),
        ).fetchone()
        if current_batch is None:
            raise ValueError("retraction expression batch is unavailable")
        batches = conn.execute(
            """SELECT batch_id, segment_index, output_count
            FROM instance_expression_batches
            WHERE profile_id = ? AND instance_id = ? AND source_run_id = ?
            ORDER BY segment_index""",
            (profile_id, instance_id, int(source_run_id)),
        ).fetchall()
        consumed = 0
        for batch in batches:
            output_count = int(batch["output_count"] or 0)
            if target_output_ordinal > consumed + output_count:
                consumed += output_count
                continue
            local_ordinal = target_output_ordinal - consumed - 1
            target = conn.execute(
                """SELECT expression_step_ordinal FROM instance_outbox
                WHERE profile_id = ? AND instance_id = ? AND expression_batch_id = ?
                  AND expression_ordinal = ?""",
                (profile_id, instance_id, str(batch["batch_id"]), local_ordinal),
            ).fetchone()
            if target is None:
                raise ValueError("retraction target output is unavailable")
            target_segment = int(batch["segment_index"])
            current_segment = int(current_batch["segment_index"])
            if target_segment > current_segment or (
                target_segment == current_segment
                and int(target["expression_step_ordinal"]) >= int(step_ordinal)
            ):
                raise ValueError("retraction target must be an earlier visible output")
            return
        raise ValueError("retraction target output is unavailable")

    @staticmethod
    def _update_target_fragments(
        conn: sqlite3.Connection,
        action: sqlite3.Row,
        status: MessageRetractionStatus,
        now: str,
    ) -> None:
        fragments = _resolved_target_fragment_rows(conn, action)
        message_ids: set[int] = set()
        for fragment in fragments:
            message_ref = str(fragment["message_ref"])
            message_ids.add(int(fragment["ledger_message_id"]))
            conn.execute(
                """UPDATE instance_message_fragments SET retraction_status = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND message_ref = ?""",
                (
                    status.value,
                    now,
                    action["profile_id"],
                    action["instance_id"],
                    message_ref,
                ),
            )
        for message_id in message_ids:
            _refresh_ledger_retraction_eligibility(
                conn,
                profile_id=str(action["profile_id"]),
                instance_id=str(action["instance_id"]),
                ledger_message_id=message_id,
            )


__all__ = [
    "MessageRetractionRecoveryMixin",
    "_action",
    "_refresh_ledger_retraction_eligibility",
    "_resolved_target_fragment_rows",
]
