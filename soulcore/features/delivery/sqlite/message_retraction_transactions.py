from __future__ import annotations

import sqlite3

from ....contracts.models import (
    MessageRetractionAction,
    MessageRetractionStatus,
)
from ....storage.sqlite.codec import _parse

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


__all__ = [
    "_action",
    "_refresh_ledger_retraction_eligibility",
    "_resolved_target_fragment_rows",
]
