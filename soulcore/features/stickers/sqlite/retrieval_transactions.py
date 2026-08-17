from __future__ import annotations

from ....contracts.delivery_visibility import (
    is_dialogue_continuity_visible,
)
from .support import (
    _dt,
    _now,
    datetime,
    sqlite3,
)


def disable_sticker_item_for_instance_in_transaction(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    item_id: str,
    *,
    now: datetime | None = None,
) -> None:
    """Validate and stage one instance-only disable on an existing transaction."""

    row = conn.execute(
        """SELECT 1 FROM sticker_items i
        JOIN sticker_assets a ON a.sticker_asset_id = i.asset_id
        JOIN sticker_libraries l ON l.library_id = i.library_id
        JOIN character_instances current
          ON current.profile_id = ? AND current.instance_id = ?
        WHERE i.item_id = ? AND i.profile_id = current.profile_id
          AND i.status = 'ACTIVE' AND a.file_status = 'AVAILABLE'
          AND ((l.library_kind = 'CORE' AND l.scope = current.scope)
            OR (l.library_kind = 'PRIVATE'
              AND l.instance_id = current.instance_id))""",
        (profile_id, instance_id, item_id),
    ).fetchone()
    if row is None:
        raise KeyError((profile_id, instance_id, item_id))
    timestamp = _dt(now or _now())
    conn.execute(
        """INSERT INTO sticker_instance_item_states(
            profile_id, instance_id, item_id, disabled_at, updated_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(profile_id, instance_id, item_id) DO UPDATE SET
            disabled_at = excluded.disabled_at,
            updated_at = excluded.updated_at""",
        (profile_id, instance_id, item_id, timestamp, timestamp),
    )


def record_sticker_usage_in_transaction(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    *,
    item_id: str,
    run_id: int | str,
    sticker_ref: str,
    compact_projection: str,
    delivery_status: str,
    now: str,
    outbox_id: int | None = None,
    expression_ordinal: int | None = None,
    message_id: int | None = None,
) -> int:
    """Write one idempotent usage inside the delivery settlement transaction."""

    status = str(delivery_status).strip().upper()
    if not is_dialogue_continuity_visible("OUTBOUND", status):
        raise ValueError("sticker usage may only be recorded after accepted delivery")
    run = str(run_id)
    ref = conn.execute(
        """SELECT item_id FROM sticker_run_candidates WHERE sticker_ref = ?
        AND profile_id = ? AND instance_id = ? AND run_id = ?
        AND item_id = ? AND (
          expires_at > ?
          OR EXISTS (
            SELECT 1 FROM instance_outbox delivery
            WHERE delivery.outbox_id = ?
              AND delivery.profile_id = sticker_run_candidates.profile_id
              AND delivery.instance_id = sticker_run_candidates.instance_id
              AND CAST(delivery.origin_run_id AS TEXT) = sticker_run_candidates.run_id
              AND delivery.status IN (
                'PENDING', 'SENDING', 'PLATFORM_ACCEPTED_UNCONFIRMED'
              )
          )
          OR (? IS NOT NULL AND EXISTS (
            SELECT 1 FROM instance_messages message
            WHERE message.profile_id = sticker_run_candidates.profile_id
              AND message.instance_id = sticker_run_candidates.instance_id
              AND message.message_id = ? AND message.direction = 'OUTBOUND'
              AND message.delivery_status IN (
                'PENDING', 'PLATFORM_ACCEPTED_UNCONFIRMED'
              )
          ))
        )""",
        (
            sticker_ref,
            profile_id,
            instance_id,
            run,
            item_id,
            now,
            outbox_id,
            message_id,
            message_id,
        ),
    ).fetchone()
    if ref is None:
        raise ValueError("sticker reference is not deliverable")
    cursor = conn.execute(
        """INSERT INTO sticker_usages(
            profile_id, instance_id, item_id, run_id, sticker_ref,
            compact_projection, delivery_status, outbox_id,
            expression_ordinal, message_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(
            profile_id, instance_id, outbox_id, expression_ordinal, sticker_ref
        ) WHERE outbox_id IS NOT NULL AND expression_ordinal IS NOT NULL
        DO NOTHING""",
        (
            profile_id,
            instance_id,
            item_id,
            run,
            sticker_ref,
            compact_projection,
            status,
            outbox_id,
            expression_ordinal,
            message_id,
            now,
        ),
    )
    if int(cursor.rowcount) == 1:
        conn.execute(
            """UPDATE sticker_items SET usage_count = usage_count + 1,
                last_used_at = ?, updated_at = ? WHERE item_id = ?""",
            (now, now, item_id),
        )
        return int(cursor.lastrowid)
    existing = conn.execute(
        """SELECT usage_id FROM sticker_usages
        WHERE profile_id = ? AND instance_id = ? AND outbox_id = ?
          AND expression_ordinal = ? AND sticker_ref = ?""",
        (profile_id, instance_id, outbox_id, expression_ordinal, sticker_ref),
    ).fetchone()
    if existing is None:
        raise RuntimeError("sticker usage idempotency lookup failed")
    return int(existing["usage_id"])


__all__ = [
    "disable_sticker_item_for_instance_in_transaction",
    "record_sticker_usage_in_transaction",
]
