"Single-transaction settlement for a matched recall notification."

from __future__ import annotations

import json
import sqlite3

from ....storage.sqlite.background_projection import project_foreground_retraction_continuity_sql
from ....storage.sqlite.codec import _dt
from ....storage.sqlite.recall_file_transactions import RecallFileTransactions
from ..domain import (
    InboundRecallDecision,
    InboundRecallHold,
    InboundRecallSettlement,
    InboundRecallTarget,
)
from .admission import invalidate_active_admission
from .records import required_datetime
from .redaction import redact_model_artifacts


def invalidate_derived_content(
    conn: sqlite3.Connection,
    hold: InboundRecallHold,
    now_text: str,
) -> None:
    redact_model_artifacts(conn, hold, now_text)
    _invalidate_summaries_and_memories(conn, hold, now_text)
    _invalidate_player_profile(conn, hold, now_text)
    _invalidate_knowledge(conn, hold, now_text)
    _invalidate_media(conn, hold, now_text)


def _invalidate_summaries_and_memories(
    conn: sqlite3.Connection,
    hold: InboundRecallHold,
    now_text: str,
) -> None:
    values = (hold.profile_id, hold.instance_id, hold.ledger_message_id)
    conn.execute(
        """DELETE FROM dialogue_summaries WHERE profile_id = ? AND instance_id = ?
          AND covered_through_message_id >= ?""",
        values,
    )
    conn.execute(
        """UPDATE memories SET status = 'RETRACTED', updated_at = ?
        WHERE profile_id = ? AND instance_id = ? AND EXISTS (
            SELECT 1 FROM memory_revision_sources source
            WHERE source.memory_id = memories.memory_id
              AND source.profile_id = memories.profile_id
              AND source.instance_id = memories.instance_id
              AND source.revision = memories.current_revision
              AND source.message_id = ?
        )""",
        (now_text, *values),
    )
    conn.execute(
        """UPDATE knowledge_fact_entries SET status = 'RETRACTED', updated_at = ?
        WHERE profile_id = ? AND instance_id = ? AND EXISTS (
            SELECT 1 FROM knowledge_fact_revision_sources source
            WHERE source.knowledge_fact_id = knowledge_fact_entries.knowledge_fact_id
              AND source.profile_id = knowledge_fact_entries.profile_id
              AND source.instance_id = knowledge_fact_entries.instance_id
              AND source.revision = knowledge_fact_entries.current_revision
              AND source.message_id = ?
        )""",
        (now_text, *values),
    )


def _invalidate_player_profile(
    conn: sqlite3.Connection,
    hold: InboundRecallHold,
    now_text: str,
) -> None:
    message_ref = f"ledger-message:{hold.ledger_message_id}"
    conn.execute(
        """UPDATE player_profile_entry_revisions SET status = 'WITHDRAWN',
        withdrawal_evidence_json = json_array(json_object(
            'kind', 'MESSAGE_RECALLED', 'message_ref', ?
        )), withdrawn_at = ?, updated_at = ?
        WHERE profile_id = ? AND instance_id = ? AND status = 'ACTIVE'
          AND EXISTS (
            SELECT 1 FROM json_each(evidence_json)
            WHERE json_extract(json_each.value, '$.message_ref') = ?
          )""",
        (
            message_ref,
            now_text,
            now_text,
            hold.profile_id,
            hold.instance_id,
            message_ref,
        ),
    )


def _invalidate_knowledge(
    conn: sqlite3.Connection,
    hold: InboundRecallHold,
    now_text: str,
) -> None:
    values = (hold.profile_id, hold.instance_id, hold.ledger_message_id)
    conn.execute(
        """UPDATE knowledge_batches SET status = 'SUPERSEDED',
        output_json = NULL, rejection_json = '[]', error = 'source_message_recalled'
        WHERE profile_id = ? AND instance_id = ?
          AND status IN ('PREPARED','COMMITTED')
          AND EXISTS (
            SELECT 1 FROM knowledge_batch_messages member
            WHERE member.batch_id = knowledge_batches.batch_id
              AND member.message_id = ?
          )""",
        values,
    )
    conn.execute(
        """UPDATE knowledge_batch_messages SET projected_text = '',
        projection_truncated = 0
        WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
        values,
    )
    conn.execute(
        """INSERT INTO knowledge_message_marks(
            profile_id, instance_id, message_id, outcome, reason, marked_at
        ) VALUES (?, ?, ?, 'TERMINAL_EXCLUDED', 'source_message_recalled', ?)
        ON CONFLICT(profile_id, instance_id, message_id) DO UPDATE SET
            outcome = 'TERMINAL_EXCLUDED', batch_id = NULL,
            reason = 'source_message_recalled', marked_at = excluded.marked_at""",
        (*values, now_text),
    )
    conn.execute(
        """UPDATE knowledge_processing_state SET
            baseline_message_id = MIN(baseline_message_id, ?),
            committed_through_message_id = MIN(committed_through_message_id, ?),
            desired_through_message_id = MAX(desired_through_message_id, ?),
            active_task_id = NULL, processing_version = processing_version + 1,
            updated_at = ? WHERE profile_id = ? AND instance_id = ?""",
        (
            max(0, hold.ledger_message_id - 1),
            max(0, hold.ledger_message_id - 1),
            hold.ledger_message_id,
            now_text,
            hold.profile_id,
            hold.instance_id,
        ),
    )


def _invalidate_media(
    conn: sqlite3.Connection,
    hold: InboundRecallHold,
    now_text: str,
) -> None:
    assets = list(
        conn.execute(
            """SELECT asset.asset_id, asset.sha256, asset.ai_task_id
            FROM media_assets asset
            JOIN media_asset_message_links link ON link.asset_id = asset.asset_id
            WHERE link.profile_id = ? AND link.instance_id = ? AND link.message_id = ?""",
            (hold.profile_id, hold.instance_id, hold.ledger_message_id),
        )
    )
    for asset in assets:
        conn.execute("DELETE FROM media_projections WHERE asset_id = ?", (asset["asset_id"],))
        conn.execute(
            """DELETE FROM visual_observation_cache
            WHERE profile_id = ? AND instance_id = ? AND sha256 = ?""",
            (hold.profile_id, hold.instance_id, asset["sha256"]),
        )
        conn.execute(
            """UPDATE media_assets SET inspection_status = 'NOT_REQUIRED',
            last_error = 'source_message_recalled', updated_at = ? WHERE asset_id = ?""",
            (now_text, asset["asset_id"]),
        )
        if asset["ai_task_id"] is not None:
            conn.execute(
                """UPDATE ai_tasks SET status = 'CANCELLED', lease_owner = NULL,
                lease_until = NULL, last_error = 'source_message_recalled',
                finished_at = COALESCE(finished_at, ?), updated_at = ?,
                version = version + 1 WHERE task_id = ?
                  AND status NOT IN ('SUCCEEDED','FAILED','CANCELLED')""",
                (now_text, now_text, int(asset["ai_task_id"])),
            )


def finalize_notice_transaction(
    conn: sqlite3.Connection,
    target: InboundRecallTarget,
    decision: InboundRecallDecision,
    *,
    event_text: str,
    now_text: str,
    file_transactions: RecallFileTransactions,
) -> InboundRecallSettlement | None:
    if not _can_settle(conn, target):
        return None
    invalidate_derived_content(conn, target.hold, now_text)
    invalidate_active_admission(conn, target.hold, now_text, file_transactions)
    isolate_original(conn, target, now_text)
    project_foreground_retraction_continuity_sql(
        conn,
        profile_id=target.hold.profile_id,
        instance_id=target.hold.instance_id,
        source_message_id=target.hold.ledger_message_id,
        retraction_key=target.receipt_id,
        settled_at=now_text,
        metadata={
            "visibility": decision.visibility.value,
            "exposed_text": decision.exposed_text,
        },
    )
    event_message_id, inserted = _ensure_communication_event(
        conn,
        target,
        event_text=event_text,
        visibility=decision.visibility.value,
        now_text=now_text,
    )
    _complete_state(
        conn,
        target,
        decision,
        event_message_id=event_message_id,
        now_text=now_text,
    )
    _complete_receipt(conn, target, now_text)
    _advance_knowledge_epoch(conn, target, now_text)
    return InboundRecallSettlement(
        target.receipt_id,
        target.hold.profile_id,
        target.hold.instance_id,
        target.hold.ledger_message_id,
        event_message_id,
        str(event_text),
        decision.visibility,
        inserted,
        target.hold.scope,
        target.hold.direct_address,
        target.hold.route_umo,
    )


def _can_settle(conn: sqlite3.Connection, target: InboundRecallTarget) -> bool:
    receipt = conn.execute(
        "SELECT status FROM inbound_recall_receipts WHERE receipt_id = ?",
        (target.receipt_id,),
    ).fetchone()
    if receipt is None or str(receipt["status"]) != "PROCESSING":
        return False
    state = conn.execute(
        """SELECT status FROM inbound_message_recall_states
        WHERE profile_id = ? AND instance_id = ? AND ledger_message_id = ?""",
        (
            target.hold.profile_id,
            target.hold.instance_id,
            target.hold.ledger_message_id,
        ),
    ).fetchone()
    return state is not None and str(state["status"]) != "RECALLED"


def _ensure_communication_event(
    conn: sqlite3.Connection,
    target: InboundRecallTarget,
    *,
    event_text: str,
    visibility: str,
    now_text: str,
) -> tuple[int, bool]:
    event_key = f"inbound-recall-event:{target.receipt_id}"
    existing = conn.execute(
        """SELECT message_id FROM instance_messages
        WHERE profile_id = ? AND instance_id = ? AND idempotency_key = ?""",
        (target.hold.profile_id, target.hold.instance_id, event_key),
    ).fetchone()
    if existing is not None:
        return int(existing["message_id"]), False
    source = conn.execute(
        """SELECT sender_id, sender_name FROM instance_messages
        WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
        (
            target.hold.profile_id,
            target.hold.instance_id,
            target.hold.ledger_message_id,
        ),
    ).fetchone()
    metadata = json.dumps(
        {
            "communication_event": "inbound_message_recalled",
            "target_ledger_message_id": target.hold.ledger_message_id,
            "visibility": visibility,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    cursor = conn.execute(
        """INSERT INTO instance_messages(
            profile_id, instance_id, direction, role, sender_id, sender_name,
            plain_text, components_json, delivery_status, idempotency_key,
            metadata_json, occurred_at, created_at, knowledge_eligibility,
            knowledge_eligibility_reason
        ) VALUES (?, ?, 'INBOUND', 'system', ?, ?, ?, '[]', 'RECEIVED',
            ?, ?, ?, ?, 'EXCLUDED', 'inbound_recall_event')""",
        (
            target.hold.profile_id,
            target.hold.instance_id,
            str(source["sender_id"] if source is not None else ""),
            str(source["sender_name"] if source is not None else ""),
            str(event_text),
            event_key,
            metadata,
            required_datetime(target.notice.received_at),
            now_text,
        ),
    )
    return int(cursor.lastrowid), True


def _complete_state(
    conn: sqlite3.Connection,
    target: InboundRecallTarget,
    decision: InboundRecallDecision,
    *,
    event_message_id: int,
    now_text: str,
) -> None:
    conn.execute(
        """UPDATE inbound_message_recall_states SET status = 'RECALLED',
        lease_owner = NULL, lease_until = NULL, recall_received_at = ?,
        recall_platform_at = ?, recall_sender_id = ?, recall_operator_id = ?,
        probability_seen = ?, attention_sample = ?, read_sample = ?,
        read_fraction = ?, visibility = ?, exposed_text = ?,
        recall_event_message_id = ?, updated_at = ?
        WHERE profile_id = ? AND instance_id = ? AND ledger_message_id = ?""",
        (
            required_datetime(target.notice.received_at),
            _dt(target.notice.platform_occurred_at),
            target.notice.sender_id,
            target.notice.operator_id,
            decision.probability_seen,
            decision.attention_sample,
            decision.read_sample,
            decision.read_fraction,
            decision.visibility.value,
            decision.exposed_text,
            event_message_id,
            now_text,
            target.hold.profile_id,
            target.hold.instance_id,
            target.hold.ledger_message_id,
        ),
    )


def _complete_receipt(
    conn: sqlite3.Connection,
    target: InboundRecallTarget,
    now_text: str,
) -> None:
    conn.execute(
        """UPDATE inbound_recall_receipts SET status = 'RESOLVED',
        matched_ledger_message_id = ?, completed_at = ?, updated_at = ?
        WHERE receipt_id = ? AND status = 'PROCESSING'""",
        (target.hold.ledger_message_id, now_text, now_text, target.receipt_id),
    )


def _advance_knowledge_epoch(
    conn: sqlite3.Connection,
    target: InboundRecallTarget,
    now_text: str,
) -> None:
    conn.execute(
        """UPDATE knowledge_processing_state SET
        processing_version = processing_version + 1, updated_at = ?
        WHERE profile_id = ? AND instance_id = ?""",
        (now_text, target.hold.profile_id, target.hold.instance_id),
    )


def isolate_original(
    conn: sqlite3.Connection,
    target: InboundRecallTarget,
    now_text: str,
) -> None:
    hold = target.hold
    conn.execute(
        """UPDATE instance_messages SET plain_text = '', components_json = '[]',
        delivery_status = 'RETRACTED', knowledge_eligibility = 'EXCLUDED',
        knowledge_eligibility_reason = 'source_message_recalled'
        WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
        (hold.profile_id, hold.instance_id, hold.ledger_message_id),
    )
    conn.execute(
        """UPDATE instance_message_fragments SET content_projection = '',
        retraction_status = 'RETRACTED', updated_at = ?
        WHERE profile_id = ? AND instance_id = ? AND ledger_message_id = ?
          AND direction = 'INBOUND'""",
        (now_text, hold.profile_id, hold.instance_id, hold.ledger_message_id),
    )


__all__ = ["finalize_notice_transaction", "invalidate_derived_content"]
