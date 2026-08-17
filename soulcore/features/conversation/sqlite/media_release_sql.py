from __future__ import annotations

import sqlite3
from typing import Any

from ....contracts.delivery_visibility import (
    DIALOGUE_CONTINUITY_OUTBOUND_STATUSES,
    sql_status_values,
)
from ....storage.sqlite.codec import _dump


def _media_cleanup_event_sql(
    conn: sqlite3.Connection,
    asset_id: str,
    profile_id: str,
    instance_id: str,
    action: str,
    status: str,
    reason: str,
    details: dict[str, Any],
    created_at: str,
) -> None:
    conn.execute(
        """INSERT INTO media_cleanup_events(
            asset_id, profile_id, instance_id, action, status,
            reason, details_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            asset_id,
            profile_id,
            instance_id,
            action,
            status,
            reason,
            _dump(details),
            created_at,
        ),
    )


def mark_summary_media_release_sql(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    summary_id: int,
    covered_through_message_id: int,
    now: str,
) -> list[sqlite3.Row]:
    outbound_statuses = sql_status_values(DIALOGUE_CONTINUITY_OUTBOUND_STATUSES)
    through = int(covered_through_message_id)
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
    if not ids:
        return rows
    placeholders = ",".join("?" for _ in ids)
    conn.execute(
        f"""UPDATE media_assets SET file_status = 'RELEASE_PENDING',
        summary_covered_by = ?, updated_at = ?
        WHERE asset_id IN ({placeholders})""",
        (int(summary_id), now, *ids),
    )
    for asset_id in ids:
        _media_cleanup_event_sql(
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
    return list(conn.execute(f"SELECT * FROM media_assets WHERE asset_id IN ({placeholders})", ids))


__all__ = ["mark_summary_media_release_sql"]
