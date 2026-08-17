from __future__ import annotations

import sqlite3
from typing import Any

from ..inspection import _dt, _now
from ..visual_cache import (
    VISUAL_OBSERVATION_CACHE_HIGH_WATER,
    VISUAL_OBSERVATION_CACHE_LOW_WATER,
    VISUAL_OBSERVATION_CONTRACT_VERSION,
    CachedVisualObservation,
)
from .asset_projection_records import MediaProjectionRecords


def _asset_is_model_visible_sql(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    asset_id: str,
) -> bool:
    row = conn.execute(
        """SELECT COUNT(*) AS inbound_links,
            SUM(CASE WHEN message.delivery_status = 'RECEIVED' THEN 1 ELSE 0 END)
                AS visible_links
        FROM media_asset_message_links link
        JOIN instance_messages message
          ON message.profile_id = link.profile_id
         AND message.instance_id = link.instance_id
         AND message.message_id = link.message_id
        WHERE link.profile_id = ? AND link.instance_id = ? AND link.asset_id = ?
          AND message.direction = 'INBOUND'""",
        (profile_id, instance_id, asset_id),
    ).fetchone()
    if row is None or int(row["inbound_links"] or 0) == 0:
        return True
    return int(row["visible_links"] or 0) == int(row["inbound_links"] or 0)


def _prune_visual_observation_cache_sql(
    conn: sqlite3.Connection,
    *,
    contract_version: int,
    high_water: int,
    low_water: int,
) -> int:
    conn.execute(
        "DELETE FROM visual_observation_cache WHERE contract_version <> ?",
        (int(contract_version),),
    )
    row = conn.execute(
        """SELECT COUNT(*) AS amount FROM (
            SELECT 1 FROM visual_observation_cache
            WHERE contract_version = ?
            GROUP BY profile_id, instance_id, sha256
        )""",
        (int(contract_version),),
    ).fetchone()
    total = int(row["amount"] if row is not None else 0)
    if total <= int(high_water):
        return 0

    remaining = total - int(low_water)
    removed = 0
    while remaining > 0:
        batch = min(remaining, 400)
        rows = conn.execute(
            """SELECT profile_id, instance_id, sha256
            FROM visual_observation_cache
            WHERE contract_version = ?
            GROUP BY profile_id, instance_id, sha256
            ORDER BY MAX(last_used_at) ASC, profile_id ASC, instance_id ASC, sha256 ASC
            LIMIT ?""",
            (int(contract_version), batch),
        ).fetchall()
        keys = [
            (str(item["profile_id"]), str(item["instance_id"]), str(item["sha256"]))
            for item in rows
        ]
        if not keys:
            break
        conn.executemany(
            """DELETE FROM visual_observation_cache
            WHERE profile_id = ? AND instance_id = ? AND sha256 = ?
              AND contract_version = ?""",
            ((*key, int(contract_version)) for key in keys),
        )
        removed += len(keys)
        remaining -= len(keys)
    return removed


class MediaVisualCacheRecords(MediaProjectionRecords):
    db: Any
    uow: Any

    async def get_cached_visual_observation(
        self,
        profile_id: str,
        instance_id: str,
        sha256: str,
        *,
        contract_version: int = VISUAL_OBSERVATION_CONTRACT_VERSION,
    ) -> CachedVisualObservation | None:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> CachedVisualObservation | None:
            row = conn.execute(
                """SELECT visible_facts, ocr_text,
                    subject_identity, scene_description, visual_style,
                    sticker_type, visible_text_state, safe, backend_id, model_id
                FROM visual_observation_cache
                WHERE profile_id = ? AND instance_id = ? AND sha256 = ?
                  AND contract_version = ?""",
                (
                    profile_id,
                    instance_id,
                    str(sha256),
                    int(contract_version),
                ),
            ).fetchone()
            if row is None:
                return None
            values = dict(row)
            values["safe"] = bool(values.get("safe"))
            observation = CachedVisualObservation.from_record(values)
            if observation is None:
                conn.execute(
                    """DELETE FROM visual_observation_cache
                    WHERE profile_id = ? AND instance_id = ? AND sha256 = ?
                      AND contract_version = ?""",
                    (
                        profile_id,
                        instance_id,
                        str(sha256),
                        int(contract_version),
                    ),
                )
                return None
            conn.execute(
                """UPDATE visual_observation_cache SET last_used_at = ?
                WHERE profile_id = ? AND instance_id = ? AND sha256 = ?
                  AND contract_version = ?""",
                (
                    now,
                    profile_id,
                    instance_id,
                    str(sha256),
                    int(contract_version),
                ),
            )
            return observation

        return await self.uow.run(operation)

    async def save_cached_visual_observation(
        self,
        profile_id: str,
        instance_id: str,
        asset_id: str,
        sha256: str,
        observation: CachedVisualObservation,
        *,
        contract_version: int = VISUAL_OBSERVATION_CONTRACT_VERSION,
        high_water: int = VISUAL_OBSERVATION_CACHE_HIGH_WATER,
        low_water: int = VISUAL_OBSERVATION_CACHE_LOW_WATER,
    ) -> int:
        version = int(contract_version)
        high = max(1, int(high_water))
        low = max(0, min(int(low_water), high))
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            target = conn.execute(
                """SELECT 1 FROM media_assets
                WHERE asset_id = ? AND profile_id = ? AND instance_id = ? AND sha256 = ?""",
                (asset_id, profile_id, instance_id, str(sha256)),
            ).fetchone()
            if target is None:
                raise KeyError((profile_id, instance_id, asset_id, sha256))
            if not _asset_is_model_visible_sql(conn, profile_id, instance_id, asset_id):
                raise ValueError("source message is unavailable")
            conn.execute(
                """INSERT INTO visual_observation_cache(
                    profile_id, instance_id, sha256,
                    contract_version, visible_facts, ocr_text,
                    subject_identity, scene_description, visual_style, sticker_type,
                    visible_text_state, safe, backend_id, model_id, cached_at, last_used_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    profile_id, instance_id, sha256, contract_version
                ) DO UPDATE SET
                    visible_facts = excluded.visible_facts,
                    ocr_text = excluded.ocr_text,
                    subject_identity = excluded.subject_identity,
                    scene_description = excluded.scene_description,
                    visual_style = excluded.visual_style,
                    sticker_type = excluded.sticker_type,
                    visible_text_state = excluded.visible_text_state,
                    safe = excluded.safe,
                    backend_id = excluded.backend_id,
                    model_id = excluded.model_id,
                    cached_at = excluded.cached_at,
                    last_used_at = excluded.last_used_at""",
                (
                    profile_id,
                    instance_id,
                    str(sha256),
                    version,
                    observation.visible_facts,
                    observation.ocr_text,
                    observation.subject_identity,
                    observation.cache_scene_payload(),
                    observation.visual_style,
                    observation.sticker_type,
                    observation.physical_text_state(),
                    int(observation.safe),
                    observation.backend_id,
                    observation.model_id,
                    now,
                    now,
                ),
            )
            conn.execute(
                """UPDATE visual_observation_cache SET last_used_at = ?
                WHERE profile_id = ? AND instance_id = ? AND sha256 = ?
                  AND contract_version = ?""",
                (
                    now,
                    profile_id,
                    instance_id,
                    str(sha256),
                    version,
                ),
            )
            return _prune_visual_observation_cache_sql(
                conn,
                contract_version=version,
                high_water=high,
                low_water=low,
            )

        return await self.uow.run(operation)

    async def prune_visual_observation_cache(
        self,
        *,
        contract_version: int = VISUAL_OBSERVATION_CONTRACT_VERSION,
        high_water: int = VISUAL_OBSERVATION_CACHE_HIGH_WATER,
        low_water: int = VISUAL_OBSERVATION_CACHE_LOW_WATER,
    ) -> int:
        high = max(1, int(high_water))
        low = max(0, min(int(low_water), high))
        return await self.uow.run(
            lambda conn: _prune_visual_observation_cache_sql(
                conn,
                contract_version=int(contract_version),
                high_water=high,
                low_water=low,
            )
        )


__all__ = ["MediaVisualCacheRecords"]
