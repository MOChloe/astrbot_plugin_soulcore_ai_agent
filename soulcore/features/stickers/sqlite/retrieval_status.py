from __future__ import annotations

from .retrieval_support import has_live_sticker_run_ref
from .support import StickerItem, StickerItemStatus, _dt, _now, sqlite3


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


__all__ = ["StickerStatusRecords"]
