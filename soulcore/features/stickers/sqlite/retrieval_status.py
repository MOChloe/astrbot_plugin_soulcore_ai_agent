from __future__ import annotations

from ....contracts.delivery_visibility import is_dialogue_continuity_visible
from .support import StickerItem, StickerItemStatus, _dt, _now, datetime, sqlite3

LIVE_STICKER_RUN_REF_CONDITION = """(
  ref.expires_at > ?
  OR EXISTS (
    SELECT 1 FROM instance_outbox delivery
    WHERE delivery.profile_id = ref.profile_id
      AND delivery.instance_id = ref.instance_id
      AND CAST(delivery.origin_run_id AS TEXT) = ref.run_id
      AND delivery.status IN ('PENDING', 'SENDING')
  )
)"""


def has_live_sticker_run_ref(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    item_id: str,
    now: str,
) -> bool:
    return (
        conn.execute(
            f"""SELECT 1 FROM sticker_run_candidates ref
            WHERE ref.profile_id = ? AND ref.item_id = ?
              AND {LIVE_STICKER_RUN_REF_CONDITION}
            LIMIT 1""",
            (profile_id, item_id, now),
        ).fetchone()
        is not None
    )


def _owned_item_status_row(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    item_id: str,
) -> sqlite3.Row:
    row = conn.execute(
        """SELECT i.source_kind, i.cluster_id, i.status, i.asset_id,
            asset.file_status
        FROM sticker_items i JOIN sticker_libraries l ON l.library_id = i.library_id
        JOIN sticker_assets asset ON asset.sticker_asset_id = i.asset_id
        JOIN character_instances current
          ON current.profile_id = ? AND current.instance_id = ?
        WHERE i.item_id = ? AND i.profile_id = ? AND (
          (l.library_kind = 'CORE' AND l.scope = current.scope)
          OR (l.library_kind = 'PRIVATE' AND l.instance_id = current.instance_id)
        )""",
        (profile_id, instance_id, item_id, profile_id),
    ).fetchone()
    if row is None:
        raise KeyError((profile_id, instance_id, item_id))
    return row


def _require_status_transition(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    profile_id: str,
    item_id: str,
    desired: StickerItemStatus,
    now: str,
) -> None:
    current = str(row["status"])

    def live_ref() -> bool:
        return has_live_sticker_run_ref(
            conn,
            profile_id=profile_id,
            item_id=item_id,
            now=now,
        )

    if desired is StickerItemStatus.DELETED:
        if current != StickerItemStatus.ARCHIVED.value:
            raise ValueError("sticker must be archived before deletion")
        if live_ref():
            raise ValueError("sticker still has an active run reference")
    elif current == StickerItemStatus.DELETED.value:
        _restore_deleted_asset(conn, row, now=now)


def _restore_deleted_asset(conn: sqlite3.Connection, row: sqlite3.Row, *, now: str) -> None:
    file_status = str(row["file_status"])
    if file_status == "RELEASE_PENDING":
        restored = conn.execute(
            """UPDATE sticker_assets SET file_status = 'AVAILABLE',
            updated_at = ? WHERE sticker_asset_id = ?
              AND file_status = 'RELEASE_PENDING'""",
            (now, row["asset_id"]),
        )
        if restored.rowcount != 1:
            raise ValueError("sticker file release already started; reimport required")
    elif file_status != "AVAILABLE":
        raise ValueError("sticker file was released; reimport required")


def _update_cluster_status_counts(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    desired: StickerItemStatus,
    now: str,
) -> None:
    before_active = row["status"] in {"ACTIVE", "NEEDS_REVIEW"}
    after_active = desired.value in {"ACTIVE", "NEEDS_REVIEW"}
    if before_active == after_active:
        return
    delta = 1 if after_active else -1
    auto_delta = delta if row["source_kind"] != "PLAYER" else 0
    conn.execute(
        """UPDATE sticker_clusters SET active_count = MAX(0, active_count + ?),
            auto_count = MAX(0, auto_count + ?), updated_at = ? WHERE cluster_id = ?""",
        (delta, auto_delta, now, row["cluster_id"]),
    )


def _release_deleted_asset(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    desired: StickerItemStatus,
    now: str,
) -> None:
    if desired is not StickerItemStatus.DELETED:
        return
    conn.execute(
        """UPDATE sticker_assets SET file_status = 'RELEASE_PENDING',
            updated_at = ? WHERE sticker_asset_id = ? AND file_status = 'AVAILABLE'
            AND NOT EXISTS (
                SELECT 1 FROM sticker_items other
                WHERE other.asset_id = sticker_assets.sticker_asset_id
                  AND other.status <> 'DELETED'
            )""",
        (now, row["asset_id"]),
    )


class StickerStatusRecords:
    async def set_sticker_item_status(
        self,
        profile_id: str,
        instance_id: str,
        item_id: str,
        status: StickerItemStatus | str,
    ) -> StickerItem:
        desired = StickerItemStatus(str(status).upper())
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> None:
            row = _owned_item_status_row(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                item_id=item_id,
            )
            _require_status_transition(
                conn,
                row,
                profile_id=profile_id,
                item_id=item_id,
                desired=desired,
                now=now,
            )
            _update_cluster_status_counts(conn, row, desired=desired, now=now)
            conn.execute(
                "UPDATE sticker_items SET status = ?, updated_at = ? WHERE item_id = ?",
                (desired.value, now, item_id),
            )
            _release_deleted_asset(conn, row, desired=desired, now=now)

        await self.uow.run(operation)
        item = await self.get_sticker_item(profile_id, instance_id, item_id)
        assert item is not None
        return item

    async def mark_stickers_for_persona_review(
        self,
        profile_id: str,
        instance_id: str,
        persona_fingerprint: str,
    ) -> int:
        def operation(conn: sqlite3.Connection) -> int:
            restored = conn.execute(
                """UPDATE sticker_items SET status = 'ACTIVE', updated_at = ?
                WHERE profile_id = ? AND library_id IN (
                    SELECT l.library_id FROM sticker_libraries l
                    JOIN character_instances current
                      ON current.profile_id = ? AND current.instance_id = ?
                    WHERE l.profile_id = current.profile_id AND l.library_kind = 'CORE'
                      AND l.scope = current.scope
                ) AND COALESCE(json_extract(metadata_json, '$.persona_bound'), 0) = 1
                  AND status = 'NEEDS_REVIEW'
                  AND COALESCE(json_extract(metadata_json, '$.persona_fingerprint'), '') = ?""",
                (_dt(_now()), profile_id, profile_id, instance_id, persona_fingerprint),
            ).rowcount
            invalidated = conn.execute(
                """UPDATE sticker_items SET status = 'NEEDS_REVIEW', updated_at = ?
                WHERE profile_id = ? AND library_id IN (
                    SELECT l.library_id FROM sticker_libraries l
                    JOIN character_instances current
                      ON current.profile_id = ? AND current.instance_id = ?
                    WHERE l.profile_id = current.profile_id AND l.library_kind = 'CORE'
                      AND l.scope = current.scope
                ) AND COALESCE(json_extract(metadata_json, '$.persona_bound'), 0) = 1
                  AND status = 'ACTIVE'
                  AND COALESCE(json_extract(metadata_json, '$.persona_fingerprint'), '') <> ?""",
                (_dt(_now()), profile_id, profile_id, instance_id, persona_fingerprint),
            ).rowcount
            return int(restored) + int(invalidated)

        return int(await self.db.call(operation, transaction=True))


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
    "LIVE_STICKER_RUN_REF_CONDITION",
    "StickerStatusRecords",
    "disable_sticker_item_for_instance_in_transaction",
    "has_live_sticker_run_ref",
    "record_sticker_usage_in_transaction",
]
