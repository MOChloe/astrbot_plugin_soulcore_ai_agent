"""Withdrawal of knowledge, summaries, profiles, and media derived from a source."""

from __future__ import annotations

import sqlite3

from ..domain import InboundRecallHold
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


__all__ = ["invalidate_derived_content"]
