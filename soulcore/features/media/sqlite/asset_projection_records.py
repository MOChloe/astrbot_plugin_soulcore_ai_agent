"""Message links, platform references and semantic media projections."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from ..domain import (
    MediaAsset,
    MediaInspectionStatus,
    MediaProjection,
    MediaProjectionStatus,
)
from ..inspection import _dt, _now


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


def _link_media_to_message_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    asset_id: str,
    message_id: int,
    relation: str,
    ordinal: int,
    now: str,
) -> None:
    normalized_relation = str(relation).strip().upper()
    if normalized_relation not in {"ATTACHMENT", "REFERENCE", "GENERATED_OUTPUT"}:
        raise ValueError("unsupported media-message relation")
    asset = conn.execute(
        """SELECT 1 FROM media_assets WHERE asset_id = ?
        AND profile_id = ? AND instance_id = ?
        AND file_status = 'AVAILABLE'""",
        (asset_id, profile_id, instance_id),
    ).fetchone()
    message = conn.execute(
        """SELECT 1 FROM instance_messages WHERE message_id = ?
        AND profile_id = ? AND instance_id = ?""",
        (int(message_id), profile_id, instance_id),
    ).fetchone()
    if asset is None or message is None:
        raise KeyError((profile_id, instance_id, asset_id, message_id))
    conn.execute(
        """INSERT INTO media_asset_message_links(
            asset_id, profile_id, instance_id, message_id,
            relation, ordinal, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(asset_id, message_id, relation) DO UPDATE SET
            ordinal = excluded.ordinal""",
        (
            asset_id,
            profile_id,
            instance_id,
            int(message_id),
            normalized_relation,
            max(0, int(ordinal)),
            now,
        ),
    )


def _register_platform_media_reference_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    platform_message_id: str,
    asset_id: str,
    ordinal: int,
    now: str,
) -> None:
    platform_id = str(platform_message_id or "").strip()
    if not platform_id:
        raise ValueError("platform_message_id cannot be empty")
    if (
        conn.execute(
            """SELECT 1 FROM media_assets WHERE asset_id = ?
            AND profile_id = ? AND instance_id = ?""",
            (asset_id, profile_id, instance_id),
        ).fetchone()
        is None
    ):
        raise KeyError((profile_id, instance_id, asset_id))
    conn.execute(
        """INSERT INTO platform_message_media_refs(
            profile_id, instance_id, platform_message_id,
            asset_id, ordinal, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(profile_id, instance_id, platform_message_id, asset_id)
        DO UPDATE SET ordinal = excluded.ordinal""",
        (
            profile_id,
            instance_id,
            platform_id,
            asset_id,
            max(0, int(ordinal)),
            now,
        ),
    )


class MediaProjectionRecords:
    async def link_media_to_message(
        self,
        profile_id: str,
        instance_id: str,
        asset_id: str,
        message_id: int,
        *,
        relation: str = "ATTACHMENT",
        ordinal: int = 0,
    ) -> None:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> None:
            _link_media_to_message_sql(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                asset_id=asset_id,
                message_id=message_id,
                relation=relation,
                ordinal=ordinal,
                now=now,
            )

        await self.uow.run(operation)
        await self.db.publish_backup_after_commit()

    async def register_platform_media_reference(
        self,
        profile_id: str,
        instance_id: str,
        platform_message_id: str,
        asset_id: str,
        *,
        ordinal: int = 0,
    ) -> None:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> None:
            _register_platform_media_reference_sql(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                platform_message_id=platform_message_id,
                asset_id=asset_id,
                ordinal=ordinal,
                now=now,
            )

        await self.uow.run(operation)
        await self.db.publish_backup_after_commit()

    async def resolve_platform_media_reference(
        self, profile_id: str, instance_id: str, platform_message_id: str
    ) -> list[MediaAsset]:
        rows = await self.db.fetch_all(
            """SELECT asset.* FROM platform_message_media_refs ref
            JOIN media_assets asset ON asset.asset_id = ref.asset_id
            WHERE ref.profile_id = ? AND ref.instance_id = ?
                AND ref.platform_message_id = ?
                AND asset.file_status = 'AVAILABLE'
            ORDER BY ref.ordinal, asset.asset_id""",
            (profile_id, instance_id, str(platform_message_id)),
        )
        return [self._media_asset(row) for row in rows]

    async def media_history_projections_for_messages(
        self, profile_id: str, instance_id: str, message_ids: Sequence[int]
    ) -> dict[int, list[dict[str, str]]]:
        ids = list(dict.fromkeys(int(item) for item in message_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = await self.db.fetch_all(
            f"""SELECT link.message_id, asset.asset_id, asset.origin,
                projection.history_projection, projection.visible_facts
            FROM media_asset_message_links link
            JOIN media_assets asset ON asset.asset_id = link.asset_id
            LEFT JOIN media_projections projection
              ON projection.asset_id = asset.asset_id
             AND projection.version = asset.current_projection_version
            WHERE link.profile_id = ? AND link.instance_id = ?
              AND link.message_id IN ({placeholders})
            ORDER BY link.message_id, link.ordinal, asset.asset_id""",
            (profile_id, instance_id, *ids),
        )
        output: dict[int, list[dict[str, str]]] = {}
        for row in rows:
            text = str(row["history_projection"] or row["visible_facts"] or "").strip()
            if not text:
                text = (
                    "对方发送了一张图片" if row["origin"] == "USER_INPUT" else "本人发送了一张图片"
                )
            output.setdefault(int(row["message_id"]), []).append(
                {"asset_id": str(row["asset_id"]), "text": text}
            )
        return output

    async def save_media_projection(
        self,
        asset_id: str,
        *,
        status: MediaProjectionStatus | str,
        visible_facts: str = "",
        history_projection: str = "",
        ocr_text: str = "",
        backend_id: str = "",
        model_id: str = "",
        ai_task_id: int | None = None,
        error: str | None = None,
    ) -> MediaProjection:
        normalized = MediaProjectionStatus(str(status))
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            asset = conn.execute(
                """SELECT current_projection_version, profile_id, instance_id
                FROM media_assets WHERE asset_id = ?""",
                (asset_id,),
            ).fetchone()
            if asset is None:
                raise KeyError(asset_id)
            if not _asset_is_model_visible_sql(
                conn,
                str(asset["profile_id"]),
                str(asset["instance_id"]),
                asset_id,
            ):
                raise ValueError("source message is unavailable")
            version = int(asset["current_projection_version"]) + 1
            cursor = conn.execute(
                """INSERT INTO media_projections(
                    asset_id, version, status, visible_facts,
                    history_projection, ocr_text,
                    backend_id, model_id, ai_task_id, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    asset_id,
                    version,
                    normalized.value,
                    str(visible_facts),
                    str(history_projection),
                    str(ocr_text),
                    str(backend_id),
                    str(model_id),
                    ai_task_id,
                    error,
                    now,
                ),
            )
            inspection = (
                MediaInspectionStatus.READY.value
                if normalized is MediaProjectionStatus.READY
                else MediaInspectionStatus.FAILED.value
                if normalized is MediaProjectionStatus.FAILED
                else MediaInspectionStatus.RUNNING.value
            )
            conn.execute(
                """UPDATE media_assets SET current_projection_version = ?,
                    inspection_status = ?, last_error = ?, updated_at = ?
                    WHERE asset_id = ?""",
                (version, inspection, error, now, asset_id),
            )
            row = conn.execute(
                "SELECT * FROM media_projections WHERE projection_id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
            assert row is not None
            return row

        projection = self._media_projection(await self.uow.run(operation))
        await self.db.publish_backup_after_commit()
        return projection

    async def get_latest_media_projection(self, asset_id: str) -> MediaProjection | None:
        row = await self.db.fetch_one(
            """SELECT * FROM media_projections WHERE asset_id = ?
            ORDER BY version DESC LIMIT 1""",
            (asset_id,),
        )
        return self._media_projection(row) if row else None

    async def mark_media_inspected(
        self,
        profile_id: str,
        instance_id: str,
        asset_id: str,
        *,
        ready: bool,
        error: str | None = None,
    ) -> MediaAsset:
        status = MediaInspectionStatus.READY.value if ready else MediaInspectionStatus.FAILED.value
        return await self._update_media_asset_fields(
            profile_id,
            instance_id,
            asset_id,
            {"inspection_status": status, "last_error": error},
        )

    async def mark_media_assets_selected(
        self,
        profile_id: str,
        instance_id: str,
        core_run_id: int,
        asset_ids: Sequence[str],
    ) -> list[MediaAsset]:
        selected = list(dict.fromkeys(str(item) for item in asset_ids))
        if len(selected) > 5:
            raise ValueError("at most five generated assets may be selected")
        if not selected:
            return []
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            placeholders = ",".join("?" for _ in selected)
            rows = list(
                conn.execute(
                    f"""SELECT * FROM media_assets WHERE profile_id = ?
                AND instance_id = ? AND core_run_id = ?
                AND origin = 'GENERATED' AND file_status = 'AVAILABLE'
                AND inspection_status = 'READY' AND asset_id IN ({placeholders})""",
                    (profile_id, instance_id, int(core_run_id), *selected),
                )
            )
            if {row["asset_id"] for row in rows} != set(selected):
                raise ValueError("selected assets must be inspected outputs of the current run")
            conn.execute(
                f"""UPDATE media_assets SET delivery_status = 'SELECTED',
                    updated_at = ?
                WHERE asset_id IN ({placeholders})""",
                (now, *selected),
            )
            return list(
                conn.execute(
                    f"SELECT * FROM media_assets WHERE asset_id IN ({placeholders})",
                    selected,
                )
            )

        rows = await self.uow.run(operation)
        await self.db.publish_backup_after_commit()
        by_id = {row["asset_id"]: self._media_asset(row) for row in rows}
        return [by_id[item] for item in selected]


__all__ = ["MediaProjectionRecords"]
