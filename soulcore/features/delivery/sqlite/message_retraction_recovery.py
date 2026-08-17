from __future__ import annotations

import sqlite3
from datetime import datetime

from ....contracts.models import (
    MessageDirection,
    MessageRetractionStatus,
)
from ....storage.sqlite.codec import _dt, _now, _parse
from ....storage.sqlite.expression_batch_lifecycle import sync_expression_batch_status
from .group_first_attempt import resolve_retract_only_group_window
from .message_retraction_transactions import (
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


__all__ = ["MessageRetractionRecoveryMixin"]
