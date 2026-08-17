from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from ....storage.sqlite.runtime_file_cleanup import (
    finish_runtime_file_cleanup_guard_sql,
)
from ..domain import (
    InboundMediaRegistrationState,
    MediaAsset,
    MediaFileStatus,
    MediaInspectionStatus,
    MediaOrigin,
    MediaPurpose,
    StoredMediaFile,
)
from ..inspection import _dt, _dump, _now
from .asset_projection_records import (
    _link_media_to_message_sql,
    _register_platform_media_reference_sql,
)
from .visual_cache_records import MediaVisualCacheRecords

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_INBOUND_ATTACHMENT_BYTES = 50 * 1024 * 1024
MAX_ANIMATION_DURATION_MS = 30_000
MAX_ANIMATION_DECODED_PIXELS = 240_000_000
_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_MIME_ALIASES = {"image/jpg": "image/jpeg", "image/x-png": "image/png"}
_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_INBOUND_MEDIA_KINDS = frozenset({"audio", "file", "video"})


def _create_media_asset_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    stored: StoredMediaFile,
    origin: MediaOrigin,
    purpose: MediaPurpose,
    delivery_status: str,
    inspection_status: MediaInspectionStatus,
    core_run_id: int | None,
    ai_task_id: int | None,
    expires_at: datetime | None,
    metadata: dict[str, Any],
    revive_missing_file: bool,
    now: str,
) -> sqlite3.Row:
    owner = conn.execute(
        "SELECT 1 FROM character_instances WHERE profile_id = ? AND instance_id = ?",
        (profile_id, instance_id),
    ).fetchone()
    if owner is None:
        raise KeyError((profile_id, instance_id))
    existing = conn.execute(
        "SELECT * FROM media_assets WHERE asset_id = ?", (stored.asset_id,)
    ).fetchone()
    if existing is not None:
        expected = (
            profile_id,
            instance_id,
            origin.value,
            purpose.value,
            stored.sha256,
            stored.relative_path,
            int(stored.byte_size),
        )
        actual = (
            existing["profile_id"],
            existing["instance_id"],
            existing["origin"],
            existing["purpose"],
            existing["sha256"],
            existing["storage_relpath"],
            int(existing["byte_size"]),
        )
        if actual != expected:
            raise ValueError("asset_id already belongs to different media")
        if str(existing["file_status"]) == MediaFileStatus.MISSING.value:
            if not revive_missing_file:
                raise ValueError("missing media asset requires a verified file revival")
            restored_cursor = conn.execute(
                """UPDATE media_assets
                SET file_status = 'AVAILABLE', last_error = NULL, updated_at = ?
                WHERE asset_id = ? AND profile_id = ? AND instance_id = ?
                  AND sha256 = ? AND storage_relpath = ? AND byte_size = ?
                  AND file_status = 'MISSING'""",
                (
                    now,
                    stored.asset_id,
                    profile_id,
                    instance_id,
                    stored.sha256,
                    stored.relative_path,
                    int(stored.byte_size),
                ),
            )
            if restored_cursor.rowcount != 1:
                raise RuntimeError("missing media asset revival lost its ownership fence")
            restored = conn.execute(
                "SELECT * FROM media_assets WHERE asset_id = ?",
                (stored.asset_id,),
            ).fetchone()
            assert restored is not None
            return restored
        return existing
    conn.execute(
        """INSERT INTO media_assets(
            asset_id, profile_id, instance_id, origin, purpose,
            mime_type, file_extension, sha256, byte_size, width, height,
            frame_count, storage_relpath, file_status, delivery_status,
            inspection_status, core_run_id, ai_task_id, expires_at,
            metadata_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'AVAILABLE',
            ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            stored.asset_id,
            profile_id,
            instance_id,
            origin.value,
            purpose.value,
            stored.mime_type,
            stored.file_extension,
            stored.sha256,
            int(stored.byte_size),
            stored.width,
            stored.height,
            stored.frame_count,
            stored.relative_path,
            str(delivery_status or "NOT_SENT").strip().upper(),
            inspection_status.value,
            core_run_id,
            ai_task_id,
            _dt(expires_at),
            _dump(metadata),
            now,
            now,
        ),
    )
    row = conn.execute(
        "SELECT * FROM media_assets WHERE asset_id = ?", (stored.asset_id,)
    ).fetchone()
    assert row is not None
    return row


class MediaAssetRecords(MediaVisualCacheRecords):
    """Repository methods mixed into the existing instance-scoped Repository."""

    db: Any

    async def create_media_asset(
        self,
        profile_id: str,
        instance_id: str,
        stored: StoredMediaFile,
        *,
        origin: MediaOrigin | str,
        purpose: MediaPurpose | str,
        delivery_status: str = "NOT_SENT",
        inspection_status: MediaInspectionStatus | str = MediaInspectionStatus.PENDING,
        core_run_id: int | None = None,
        ai_task_id: int | None = None,
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        revive_missing_file: bool = False,
        cleanup_guard_id: int | None = None,
    ) -> MediaAsset:
        normalized_origin = MediaOrigin(str(origin))
        normalized_purpose = MediaPurpose(str(purpose))
        normalized_inspection = MediaInspectionStatus(str(inspection_status))
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            row = _create_media_asset_sql(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                stored=stored,
                origin=normalized_origin,
                purpose=normalized_purpose,
                delivery_status=delivery_status,
                inspection_status=normalized_inspection,
                core_run_id=core_run_id,
                ai_task_id=ai_task_id,
                expires_at=expires_at,
                metadata=dict(metadata or {}),
                revive_missing_file=revive_missing_file,
                now=now,
            )
            if cleanup_guard_id is not None:
                finish_runtime_file_cleanup_guard_sql(
                    conn,
                    cleanup_id=cleanup_guard_id,
                    profile_id=profile_id,
                    instance_id=instance_id,
                    stored=stored,
                )
            return row

        asset = self._media_asset(await self.uow.run(operation))
        await self.db.publish_backup_after_commit()
        return asset

    async def register_inbound_media_asset(
        self,
        profile_id: str,
        instance_id: str,
        stored: StoredMediaFile,
        *,
        message_id: int,
        ordinal: int,
        platform_message_id: str = "",
        delivery_status: str = "NOT_SENT",
        inspection_status: MediaInspectionStatus | str = MediaInspectionStatus.PENDING,
        metadata: dict[str, Any] | None = None,
    ) -> MediaAsset:
        """Commit one inbound asset and every required lookup edge atomically."""

        normalized_inspection = MediaInspectionStatus(str(inspection_status))
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            row = _create_media_asset_sql(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                stored=stored,
                origin=MediaOrigin.USER_INPUT,
                purpose=MediaPurpose.NORMAL_IMAGE,
                delivery_status=delivery_status,
                inspection_status=normalized_inspection,
                core_run_id=None,
                ai_task_id=None,
                expires_at=None,
                metadata=dict(metadata or {}),
                revive_missing_file=True,
                now=now,
            )
            _link_media_to_message_sql(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                asset_id=stored.asset_id,
                message_id=message_id,
                relation="ATTACHMENT",
                ordinal=ordinal,
                now=now,
            )
            if str(platform_message_id or "").strip():
                _register_platform_media_reference_sql(
                    conn,
                    profile_id=profile_id,
                    instance_id=instance_id,
                    platform_message_id=platform_message_id,
                    asset_id=stored.asset_id,
                    ordinal=ordinal,
                    now=now,
                )
            return row

        asset = self._media_asset(await self.uow.run(operation))
        await self.db.publish_backup_after_commit()
        return asset

    async def inspect_inbound_media_registration(
        self,
        profile_id: str,
        instance_id: str,
        stored: StoredMediaFile,
        *,
        message_id: int,
        ordinal: int,
        platform_message_id: str = "",
    ) -> tuple[InboundMediaRegistrationState, MediaAsset | None]:
        """Classify an uncertain atomic registration without changing its state."""

        platform_id = str(platform_message_id or "").strip()

        def operation(
            conn: sqlite3.Connection,
        ) -> tuple[InboundMediaRegistrationState, sqlite3.Row | None]:
            row = conn.execute(
                "SELECT * FROM media_assets WHERE asset_id = ?",
                (stored.asset_id,),
            ).fetchone()
            if row is None:
                return InboundMediaRegistrationState.UNOWNED, None
            exact_owner = (
                str(row["profile_id"]) == profile_id
                and str(row["instance_id"]) == instance_id
                and str(row["origin"]) == MediaOrigin.USER_INPUT.value
                and str(row["purpose"]) == MediaPurpose.NORMAL_IMAGE.value
                and str(row["sha256"]) == stored.sha256
                and str(row["storage_relpath"]) == stored.relative_path
                and int(row["byte_size"]) == int(stored.byte_size)
            )
            if not exact_owner:
                return InboundMediaRegistrationState.UNOWNED, None
            if str(row["file_status"]) != MediaFileStatus.AVAILABLE.value:
                return InboundMediaRegistrationState.OWNED_INCOMPLETE, row
            message_link = conn.execute(
                """SELECT 1 FROM media_asset_message_links
                WHERE asset_id = ? AND profile_id = ? AND instance_id = ?
                  AND message_id = ? AND relation = 'ATTACHMENT' AND ordinal = ?""",
                (
                    stored.asset_id,
                    profile_id,
                    instance_id,
                    int(message_id),
                    max(0, int(ordinal)),
                ),
            ).fetchone()
            platform_link = (
                True
                if not platform_id
                else conn.execute(
                    """SELECT 1 FROM platform_message_media_refs
                    WHERE profile_id = ? AND instance_id = ?
                      AND platform_message_id = ? AND asset_id = ? AND ordinal = ?""",
                    (
                        profile_id,
                        instance_id,
                        platform_id,
                        stored.asset_id,
                        max(0, int(ordinal)),
                    ),
                ).fetchone()
                is not None
            )
            state = (
                InboundMediaRegistrationState.COMMITTED
                if message_link is not None and platform_link
                else InboundMediaRegistrationState.OWNED_INCOMPLETE
            )
            return state, row

        state, row = await self.uow.run(operation)
        return state, self._media_asset(row) if row is not None else None

    async def register_generated_media_asset(
        self,
        profile_id: str,
        instance_id: str,
        stored: StoredMediaFile,
        *,
        core_run_id: int,
        ai_task_id: int | None = None,
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        revive_missing_file: bool = False,
        cleanup_guard_id: int | None = None,
    ) -> MediaAsset:
        return await self.create_media_asset(
            profile_id,
            instance_id,
            stored,
            origin=MediaOrigin.GENERATED,
            purpose=MediaPurpose.GENERATED_IMAGE,
            inspection_status=MediaInspectionStatus.PENDING,
            core_run_id=core_run_id,
            ai_task_id=ai_task_id,
            expires_at=expires_at or (_now() + timedelta(hours=24)),
            metadata=metadata,
            revive_missing_file=revive_missing_file,
            cleanup_guard_id=cleanup_guard_id,
        )

    async def get_media_asset(
        self,
        asset_id: str,
        *,
        profile_id: str | None = None,
        instance_id: str | None = None,
    ) -> MediaAsset | None:
        clauses = ["asset_id = ?"]
        parameters: list[Any] = [asset_id]
        if profile_id is not None:
            clauses.append("profile_id = ?")
            parameters.append(profile_id)
        if instance_id is not None:
            clauses.append("instance_id = ?")
            parameters.append(instance_id)
        row = await self.db.fetch_one(
            f"SELECT * FROM media_assets WHERE {' AND '.join(clauses)}", parameters
        )
        return self._media_asset(row) if row else None

    async def list_media_assets(
        self,
        profile_id: str,
        instance_id: str,
        *,
        origin: MediaOrigin | str | None = None,
        file_status: MediaFileStatus | str | None = None,
        core_run_id: int | None = None,
        mime_prefix: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MediaAsset]:
        clauses = ["profile_id = ?", "instance_id = ?"]
        parameters: list[Any] = [profile_id, instance_id]
        if origin is not None:
            clauses.append("origin = ?")
            parameters.append(str(origin))
        if file_status is not None:
            clauses.append("file_status = ?")
            parameters.append(str(file_status))
        if core_run_id is not None:
            clauses.append("core_run_id = ?")
            parameters.append(int(core_run_id))
        if mime_prefix is not None:
            normalized_mime_prefix = str(mime_prefix).strip().lower()
            if not normalized_mime_prefix.endswith("/"):
                raise ValueError("media MIME prefix must end with '/'")
            clauses.append("LOWER(mime_type) LIKE ?")
            parameters.append(f"{normalized_mime_prefix}%")
        parameters.extend((max(1, min(int(limit), 1000)), max(0, int(offset))))
        rows = await self.db.fetch_all(
            f"""SELECT * FROM media_assets WHERE {" AND ".join(clauses)}
            ORDER BY created_at DESC, asset_id DESC LIMIT ? OFFSET ?""",
            parameters,
        )
        return [self._media_asset(row) for row in rows]

    async def list_available_image_asset_ids_for_messages(
        self,
        profile_id: str,
        instance_id: str,
        message_ids: Sequence[int],
        *,
        limit: int = 20,
    ) -> list[str]:
        """Return controlled image IDs attached to a bounded message batch."""

        ids = list(dict.fromkeys(int(item) for item in message_ids if int(item) > 0))[:100]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = await self.db.fetch_all(
            f"""SELECT link.asset_id FROM media_asset_message_links link
            JOIN media_assets asset
              ON asset.asset_id = link.asset_id
             AND asset.profile_id = link.profile_id
             AND asset.instance_id = link.instance_id
            JOIN instance_messages message
              ON message.profile_id = link.profile_id
             AND message.instance_id = link.instance_id
             AND message.message_id = link.message_id
            WHERE link.profile_id = ? AND link.instance_id = ?
              AND link.message_id IN ({placeholders})
              AND asset.file_status = 'AVAILABLE'
              AND asset.mime_type LIKE 'image/%'
              AND message.direction = 'INBOUND'
              AND message.delivery_status = 'RECEIVED'
              AND NOT EXISTS (
                  SELECT 1 FROM media_asset_message_links visibility_link
                  JOIN instance_messages visibility_message
                    ON visibility_message.profile_id = visibility_link.profile_id
                   AND visibility_message.instance_id = visibility_link.instance_id
                   AND visibility_message.message_id = visibility_link.message_id
                  WHERE visibility_link.profile_id = link.profile_id
                    AND visibility_link.instance_id = link.instance_id
                    AND visibility_link.asset_id = link.asset_id
                    AND visibility_message.direction = 'INBOUND'
                    AND COALESCE(visibility_message.delivery_status, '') <> 'RECEIVED'
              )
            ORDER BY link.message_id, link.ordinal, link.asset_id LIMIT ?""",
            (
                profile_id,
                instance_id,
                *ids,
                max(1, min(100, int(limit))),
            ),
        )
        return list(dict.fromkeys(str(row["asset_id"]) for row in rows))

    async def list_available_attachment_refs_for_messages(
        self,
        profile_id: str,
        instance_id: str,
        message_ids: Sequence[int],
        *,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        """Rebuild safe non-image references after a durable grace-period release."""

        ids = list(dict.fromkeys(int(item) for item in message_ids if int(item) > 0))[:100]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = await self.db.fetch_all(
            f"""SELECT link.asset_id,
                COALESCE(json_extract(asset.metadata_json, '$.media_kind'), '') AS media_kind,
                COALESCE(json_extract(asset.metadata_json, '$.display_name'), '') AS display_name
            FROM media_asset_message_links link
            JOIN media_assets asset
              ON asset.asset_id = link.asset_id
             AND asset.profile_id = link.profile_id
             AND asset.instance_id = link.instance_id
            JOIN instance_messages message
              ON message.profile_id = link.profile_id
             AND message.instance_id = link.instance_id
             AND message.message_id = link.message_id
            WHERE link.profile_id = ? AND link.instance_id = ?
              AND link.message_id IN ({placeholders})
              AND asset.file_status = 'AVAILABLE'
              AND COALESCE(json_extract(asset.metadata_json, '$.media_kind'), '')
                  IN ('audio', 'file', 'video')
              AND message.direction = 'INBOUND'
              AND message.delivery_status = 'RECEIVED'
              AND NOT EXISTS (
                  SELECT 1 FROM media_asset_message_links visibility_link
                  JOIN instance_messages visibility_message
                    ON visibility_message.profile_id = visibility_link.profile_id
                   AND visibility_message.instance_id = visibility_link.instance_id
                   AND visibility_message.message_id = visibility_link.message_id
                  WHERE visibility_link.profile_id = link.profile_id
                    AND visibility_link.instance_id = link.instance_id
                    AND visibility_link.asset_id = link.asset_id
                    AND visibility_message.direction = 'INBOUND'
                    AND COALESCE(visibility_message.delivery_status, '') <> 'RECEIVED'
              )
            ORDER BY link.message_id, link.ordinal, link.asset_id LIMIT ?""",
            (
                profile_id,
                instance_id,
                *ids,
                max(1, min(100, int(limit))),
            ),
        )
        return [
            {
                "asset_id": str(row["asset_id"]),
                "kind": str(row["media_kind"]),
                "display_name": str(row["display_name"]),
            }
            for row in rows
        ]

    async def asset_is_model_visible(
        self, profile_id: str, instance_id: str, asset_id: str
    ) -> bool:
        """A recalled or grace-held inbound source may retain bytes but never reach a model."""

        row = await self.db.fetch_one(
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
            (profile_id, instance_id, str(asset_id)),
        )
        if row is None or int(row["inbound_links"] or 0) == 0:
            return True
        return int(row["visible_links"] or 0) == int(row["inbound_links"] or 0)

    async def media_asset_statistics(
        self,
        profile_id: str,
        instance_id: str,
        *,
        mime_prefix: str | None = None,
    ) -> dict[str, Any]:
        clauses = ["profile_id = ?", "instance_id = ?"]
        parameters: list[Any] = [profile_id, instance_id]
        if mime_prefix is not None:
            normalized_mime_prefix = str(mime_prefix).strip().lower()
            if not normalized_mime_prefix.endswith("/"):
                raise ValueError("media MIME prefix must end with '/'")
            clauses.append("LOWER(mime_type) LIKE ?")
            parameters.append(f"{normalized_mime_prefix}%")
        rows = await self.db.fetch_all(
            f"""SELECT file_status, inspection_status, COUNT(*) AS amount
            FROM media_assets WHERE {" AND ".join(clauses)}
            GROUP BY file_status, inspection_status""",
            parameters,
        )
        file_status: dict[str, int] = {}
        inspection_status: dict[str, int] = {}
        total = 0
        for row in rows:
            amount = int(row["amount"])
            total += amount
            file_key = str(row["file_status"])
            inspection_key = str(row["inspection_status"])
            file_status[file_key] = file_status.get(file_key, 0) + amount
            inspection_status[inspection_key] = inspection_status.get(inspection_key, 0) + amount
        return {
            "total": total,
            "file_status": file_status,
            "inspection_status": inspection_status,
        }
