from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any

from ....contracts.delivery_visibility import (
    DIALOGUE_CONTINUITY_OUTBOUND_STATUSES,
    sql_status_values,
)
from ..domain import MediaAsset, MediaInspectionStatus
from ..inspection import _dt, _now


class MediaReleasePlanningCommands:
    db: Any
    uow: Any

    async def mark_summary_covered_media_for_release(
        self, profile_id: str, instance_id: str, summary_id: int
    ) -> list[MediaAsset]:
        """Atomically mark releasable originals after a committed summary."""

        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            summary = conn.execute(
                """SELECT covered_through_message_id FROM dialogue_summaries
                WHERE summary_id = ? AND profile_id = ? AND instance_id = ?""",
                (int(summary_id), profile_id, instance_id),
            ).fetchone()
            if summary is None:
                raise KeyError((profile_id, instance_id, summary_id))
            return self._mark_summary_media_release_sql(
                conn,
                profile_id,
                instance_id,
                int(summary_id),
                int(summary["covered_through_message_id"]),
                now,
            )

        rows = await self.uow.run(operation)
        if rows:
            await self.db.publish_backup_after_commit()
        return [self._media_asset(row) for row in rows]

    @classmethod
    def _mark_summary_media_release_sql(
        cls,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        summary_id: int,
        covered_through_message_id: int,
        now: str,
    ) -> list[sqlite3.Row]:
        through = int(covered_through_message_id)
        outbound_statuses = sql_status_values(DIALOGUE_CONTINUITY_OUTBOUND_STATUSES)
        rows = list(
            conn.execute(
                f"""SELECT DISTINCT asset.* FROM media_assets asset
                JOIN media_asset_message_links link ON link.asset_id = asset.asset_id
                JOIN character_instances character
                  ON character.profile_id = asset.profile_id
                 AND character.instance_id = asset.instance_id
                JOIN scope_configs config
                  ON config.profile_id = character.profile_id
                 AND config.scope = character.scope
                JOIN instance_messages message
                  ON message.profile_id = link.profile_id
                 AND message.instance_id = link.instance_id
                 AND message.message_id = link.message_id
                WHERE asset.profile_id = ? AND asset.instance_id = ?
                  AND asset.file_status = 'AVAILABLE'
                  AND asset.inspection_status = 'READY'
                  AND NOT EXISTS (
                    SELECT 1 FROM media_retention_holds hold
                    WHERE hold.asset_id = asset.asset_id AND hold.released_at IS NULL
                  )
                  AND (
                    asset.mime_type NOT LIKE 'image/%'
                    OR julianday(?) >= julianday((
                      SELECT MAX(touch.created_at)
                      FROM media_asset_message_links touch
                      WHERE touch.asset_id = asset.asset_id
                    )) + config.media_original_retention_days
                  )
                  AND link.message_id <= ?
                  AND NOT EXISTS (
                    SELECT 1 FROM media_asset_message_links newer
                    WHERE newer.asset_id = asset.asset_id AND newer.message_id > ?
                  )
                  AND (
                    (asset.origin = 'USER_INPUT' AND message.direction = 'INBOUND'
                     AND message.delivery_status = 'RECEIVED')
                    OR
                    (asset.origin = 'GENERATED' AND message.direction = 'OUTBOUND'
                     AND message.delivery_status IN ({outbound_statuses}))
                  )""",
                (profile_id, instance_id, now, through, through),
            )
        )
        ids = [row["asset_id"] for row in rows]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"""UPDATE media_assets SET file_status = 'RELEASE_PENDING',
                        summary_covered_by = ?, updated_at = ?
                    WHERE asset_id IN ({placeholders})""",
                (int(summary_id), now, *ids),
            )
            for asset_id in ids:
                cls._media_cleanup_event_sql(
                    conn,
                    asset_id,
                    profile_id,
                    instance_id,
                    "SUMMARY_RELEASE",
                    "PENDING",
                    "summary_covered",
                    {},
                    now,
                )
            rows = list(
                conn.execute(
                    f"SELECT * FROM media_assets WHERE asset_id IN ({placeholders})",
                    ids,
                )
            )
        return rows

    async def mark_media_release_if_already_summarized(self, asset_id: str) -> list[MediaAsset]:
        asset = await self.get_media_asset(asset_id)
        if asset is None or asset.inspection_status is not MediaInspectionStatus.READY:
            return []
        row = await self.db.fetch_one(
            """SELECT summary.summary_id FROM media_asset_message_links link
            JOIN dialogue_summaries summary
              ON summary.profile_id = link.profile_id
             AND summary.instance_id = link.instance_id
            WHERE link.asset_id = ?
            GROUP BY summary.summary_id, summary.covered_through_message_id
            HAVING summary.covered_through_message_id >= MAX(link.message_id)
            ORDER BY summary.version DESC LIMIT 1""",
            (asset_id,),
        )
        if row is None:
            return []
        return await self.mark_summary_covered_media_for_release(
            asset.profile_id, asset.instance_id, int(row["summary_id"])
        )

    async def mark_expired_media_for_release(
        self, *, now: datetime | None = None, limit: int = 100
    ) -> list[MediaAsset]:
        current = _dt(now or _now())

        def operation(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            rows = list(
                conn.execute(
                    """SELECT * FROM media_assets asset WHERE file_status = 'AVAILABLE'
                AND expires_at IS NOT NULL AND expires_at <= ?
                AND NOT EXISTS (
                    SELECT 1 FROM media_retention_holds hold
                    WHERE hold.asset_id = asset.asset_id AND hold.released_at IS NULL
                )
                AND NOT (
                    asset.origin = 'GENERATED'
                    AND asset.delivery_status = 'SELECTED'
                    AND EXISTS (
                      SELECT 1 FROM instance_outbox delivery
                      WHERE delivery.profile_id = asset.profile_id
                        AND delivery.instance_id = asset.instance_id
                        AND delivery.origin_run_id = asset.core_run_id
                        AND delivery.status IN ('PENDING', 'SENDING')
                    )
                )
                ORDER BY expires_at, asset_id LIMIT ?""",
                    (current, max(1, min(int(limit), 1000))),
                )
            )
            for row in rows:
                conn.execute(
                    """UPDATE media_assets SET file_status = 'RELEASE_PENDING',
                    updated_at = ? WHERE asset_id = ? AND file_status = 'AVAILABLE'""",
                    (current, row["asset_id"]),
                )
                self._media_cleanup_event_sql(
                    conn,
                    row["asset_id"],
                    row["profile_id"],
                    row["instance_id"],
                    "TTL_RELEASE",
                    "PENDING",
                    "expired",
                    {},
                    current,
                )
            if not rows:
                return []
            ids = [row["asset_id"] for row in rows]
            placeholders = ",".join("?" for _ in ids)
            return list(
                conn.execute(f"SELECT * FROM media_assets WHERE asset_id IN ({placeholders})", ids)
            )

        rows = await self.uow.run(operation)
        if rows:
            await self.db.publish_backup_after_commit()
        return [self._media_asset(row) for row in rows]

    async def mark_retention_elapsed_media_for_release(
        self, *, now: datetime | None = None, limit: int = 100
    ) -> list[MediaAsset]:
        """Mark summarized image originals after their rolling minimum retention elapses."""

        current = _dt(now or _now())
        outbound_statuses = sql_status_values(DIALOGUE_CONTINUITY_OUTBOUND_STATUSES)

        def operation(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            rows = list(
                conn.execute(
                    f"""SELECT asset.*, config.media_original_retention_days AS retention_days,
                    (
                      SELECT summary.summary_id FROM dialogue_summaries summary
                      WHERE summary.profile_id = asset.profile_id
                        AND summary.instance_id = asset.instance_id
                        AND summary.covered_through_message_id >= (
                          SELECT MAX(covered_link.message_id)
                          FROM media_asset_message_links covered_link
                          WHERE covered_link.asset_id = asset.asset_id
                        )
                      ORDER BY summary.version DESC LIMIT 1
                    ) AS eligible_summary_id
                    FROM media_assets asset
                    JOIN character_instances character
                      ON character.profile_id = asset.profile_id
                     AND character.instance_id = asset.instance_id
                    JOIN scope_configs config
                      ON config.profile_id = character.profile_id
                     AND config.scope = character.scope
                    JOIN instance_messages message
                      ON message.profile_id = asset.profile_id
                     AND message.instance_id = asset.instance_id
                     AND message.message_id = (
                       SELECT MAX(latest_link.message_id)
                       FROM media_asset_message_links latest_link
                       WHERE latest_link.asset_id = asset.asset_id
                     )
                    WHERE asset.file_status = 'AVAILABLE'
                      AND asset.inspection_status = 'READY'
                      AND asset.mime_type LIKE 'image/%'
                      AND NOT EXISTS (
                        SELECT 1 FROM media_retention_holds hold
                        WHERE hold.asset_id = asset.asset_id AND hold.released_at IS NULL
                      )
                      AND EXISTS (
                        SELECT 1 FROM dialogue_summaries summary
                        WHERE summary.profile_id = asset.profile_id
                          AND summary.instance_id = asset.instance_id
                          AND summary.covered_through_message_id >= (
                            SELECT MAX(covered_link.message_id)
                            FROM media_asset_message_links covered_link
                            WHERE covered_link.asset_id = asset.asset_id
                          )
                      )
                      AND julianday(?) >= julianday((
                        SELECT MAX(touch.created_at)
                        FROM media_asset_message_links touch
                        WHERE touch.asset_id = asset.asset_id
                      )) + config.media_original_retention_days
                      AND (
                        (asset.origin = 'USER_INPUT' AND message.direction = 'INBOUND'
                         AND message.delivery_status = 'RECEIVED')
                        OR
                        (asset.origin = 'GENERATED' AND message.direction = 'OUTBOUND'
                         AND message.delivery_status IN ({outbound_statuses}))
                      )
                    ORDER BY (
                      SELECT MAX(touch.created_at)
                      FROM media_asset_message_links touch
                      WHERE touch.asset_id = asset.asset_id
                    ), asset.asset_id
                    LIMIT ?""",
                    (current, max(1, min(int(limit), 1000))),
                )
            )
            released_ids: list[str] = []
            for row in rows:
                cursor = conn.execute(
                    """UPDATE media_assets SET file_status = 'RELEASE_PENDING',
                    summary_covered_by = ?, updated_at = ?
                    WHERE asset_id = ? AND file_status = 'AVAILABLE'""",
                    (int(row["eligible_summary_id"]), current, row["asset_id"]),
                )
                if cursor.rowcount != 1:
                    continue
                released_ids.append(str(row["asset_id"]))
                self._media_cleanup_event_sql(
                    conn,
                    row["asset_id"],
                    row["profile_id"],
                    row["instance_id"],
                    "SUMMARY_RELEASE",
                    "PENDING",
                    "minimum_retention_elapsed",
                    {"retention_days": int(row["retention_days"])},
                    current,
                )
            if not released_ids:
                return []
            placeholders = ",".join("?" for _ in released_ids)
            return list(
                conn.execute(
                    f"SELECT * FROM media_assets WHERE asset_id IN ({placeholders})",
                    released_ids,
                )
            )

        rows = await self.uow.run(operation)
        if rows:
            await self.db.publish_backup_after_commit()
        return [self._media_asset(row) for row in rows]


__all__ = ["MediaReleasePlanningCommands"]
