from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .library_sql import ensure_sticker_library
from .support import StickerAsset, _dt, _load, _now, sqlite3, uuid


class StickerLibraryRecords:
    async def ensure_sticker_library(
        self,
        profile_id: str,
        instance_id: str,
        *,
        library_kind: str,
    ) -> dict[str, Any]:
        now = _dt(_now())
        row = await self.uow.run(
            lambda conn: ensure_sticker_library(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                library_kind=library_kind,
                now=now,
            )
        )
        return dict(row)

    async def get_sticker_asset(
        self, sticker_asset_id: str, *, profile_id: str | None = None
    ) -> StickerAsset | None:
        clauses = ["sticker_asset_id = ?"]
        values: list[Any] = [str(sticker_asset_id)]
        if profile_id is not None:
            clauses.append("profile_id = ?")
            values.append(str(profile_id))
        row = await self.db.fetch_one(
            f"SELECT * FROM sticker_assets WHERE {' AND '.join(clauses)}", values
        )
        return self._sticker_asset(row) if row is not None else None

    async def get_accessible_sticker_asset(
        self,
        sticker_asset_id: str,
        *,
        profile_id: str,
        instance_id: str,
    ) -> StickerAsset | None:
        """Return an asset only through a library visible to this chat instance."""

        row = await self.db.fetch_one(
            """SELECT asset.*
            FROM sticker_assets asset
            WHERE asset.sticker_asset_id = ?
              AND asset.profile_id = ?
              AND asset.file_status = 'AVAILABLE'
              AND EXISTS (
                SELECT 1
                FROM sticker_items item
                JOIN sticker_libraries library ON library.library_id = item.library_id
                JOIN character_instances character
                  ON character.profile_id = ? AND character.instance_id = ?
                WHERE item.asset_id = asset.sticker_asset_id
                  AND item.status <> 'DELETED'
                  AND library.profile_id = character.profile_id
                  AND (
                    (library.library_kind = 'CORE' AND library.scope = character.scope)
                    OR
                    (library.library_kind = 'PRIVATE' AND library.instance_id = character.instance_id)
                  )
              )""",
            (sticker_asset_id, profile_id, profile_id, instance_id),
        )
        return self._sticker_asset(row) if row is not None else None

    async def find_sticker_asset_by_sha(self, profile_id: str, sha256: str) -> StickerAsset | None:
        row = await self.db.fetch_one(
            """SELECT * FROM sticker_assets
            WHERE profile_id = ? AND canonical_sha256 = ? AND file_status = 'AVAILABLE'""",
            (profile_id, str(sha256)),
        )
        return self._sticker_asset(row) if row is not None else None

    async def create_sticker_asset(
        self,
        profile_id: str,
        stored: Any,
        *,
        duration_ms: int = 0,
    ) -> tuple[StickerAsset, bool]:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> tuple[str, bool]:
            existing = conn.execute(
                """SELECT sticker_asset_id, file_status FROM sticker_assets
                WHERE profile_id = ? AND canonical_sha256 = ?""",
                (profile_id, str(stored.sha256)),
            ).fetchone()
            if existing is not None:
                asset_id = str(existing["sticker_asset_id"])
                if str(existing["file_status"]) == "AVAILABLE":
                    return asset_id, False
                conn.execute(
                    """UPDATE sticker_assets SET storage_relpath = ?, mime_type = ?,
                    file_extension = ?, byte_size = ?, width = ?, height = ?,
                    is_animated = ?, frame_count = ?, duration_ms = ?,
                    file_status = 'AVAILABLE', updated_at = ?
                    WHERE sticker_asset_id = ?""",
                    (
                        str(stored.relative_path),
                        str(stored.mime_type),
                        str(stored.file_extension),
                        int(stored.byte_size),
                        int(stored.width or 0),
                        int(stored.height or 0),
                        int(int(stored.frame_count or 1) > 1),
                        max(1, int(stored.frame_count or 1)),
                        max(0, int(duration_ms)),
                        now,
                        asset_id,
                    ),
                )
                return asset_id, True
            asset_id = str(stored.asset_id or "sa_" + uuid.uuid4().hex)
            conn.execute(
                """INSERT INTO sticker_assets(
                    sticker_asset_id, profile_id, canonical_sha256, storage_relpath,
                    mime_type, file_extension, byte_size, width, height, is_animated,
                    frame_count, duration_ms, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    asset_id,
                    profile_id,
                    str(stored.sha256),
                    str(stored.relative_path),
                    str(stored.mime_type),
                    str(stored.file_extension),
                    int(stored.byte_size),
                    int(stored.width or 0),
                    int(stored.height or 0),
                    int(int(stored.frame_count or 1) > 1),
                    max(1, int(stored.frame_count or 1)),
                    max(0, int(duration_ms)),
                    now,
                    now,
                ),
            )
            return asset_id, True

        asset_id, created = await self.uow.run(operation)
        asset = await self.get_sticker_asset(asset_id, profile_id=profile_id)
        assert asset is not None
        return asset, created

    async def delete_unreferenced_sticker_asset(self, sticker_asset_id: str) -> bool:
        def operation(conn: sqlite3.Connection) -> int:
            used = conn.execute(
                "SELECT 1 FROM sticker_items WHERE asset_id = ? LIMIT 1",
                (sticker_asset_id,),
            ).fetchone()
            if used is not None:
                return 0
            return int(
                conn.execute(
                    "DELETE FROM sticker_assets WHERE sticker_asset_id = ?",
                    (sticker_asset_id,),
                ).rowcount
            )

        return bool(await self.uow.run(operation))

    async def list_pending_sticker_releases(self, *, limit: int = 100) -> list[StickerAsset]:
        rows = await self.db.fetch_all(
            """SELECT asset.* FROM sticker_assets asset
            WHERE asset.file_status IN ('RELEASE_PENDING', 'RELEASED')
              AND NOT EXISTS (
                SELECT 1 FROM sticker_items item
                WHERE item.asset_id = asset.sticker_asset_id
                  AND item.status <> 'DELETED'
              )
            ORDER BY asset.updated_at, asset.sticker_asset_id LIMIT ?""",
            (max(1, min(int(limit), 1000)),),
        )
        return [self._sticker_asset(row) for row in rows]

    async def claim_sticker_asset_release(
        self,
        sticker_asset_id: str,
    ) -> StickerAsset | None:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row | None:
            row = conn.execute(
                """SELECT * FROM sticker_assets asset
                WHERE asset.sticker_asset_id = ?
                  AND asset.file_status IN ('RELEASE_PENDING', 'RELEASED')
                  AND NOT EXISTS (
                    SELECT 1 FROM sticker_items item
                    WHERE item.asset_id = asset.sticker_asset_id
                      AND item.status <> 'DELETED'
                  )""",
                (sticker_asset_id,),
            ).fetchone()
            if row is None:
                return None
            if str(row["file_status"]) == "RELEASE_PENDING":
                conn.execute(
                    """UPDATE sticker_assets SET file_status = 'RELEASED', updated_at = ?
                    WHERE sticker_asset_id = ? AND file_status = 'RELEASE_PENDING'""",
                    (now, sticker_asset_id),
                )
            return conn.execute(
                "SELECT * FROM sticker_assets WHERE sticker_asset_id = ?",
                (sticker_asset_id,),
            ).fetchone()

        row = await self.uow.run(operation)
        return self._sticker_asset(row) if row is not None else None

    async def sticker_inventory_summary(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> dict[str, Any]:
            core = ensure_sticker_library(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                library_kind="CORE",
                now=now,
            )
            private = ensure_sticker_library(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                library_kind="PRIVATE",
                now=now,
            )
            ids = (str(core["library_id"]), str(private["library_id"]))
            row = conn.execute(
                """SELECT COUNT(*) total,
                SUM(CASE WHEN i.usage_type = 'AMBIENT' THEN 1 ELSE 0 END) ambient,
                SUM(CASE WHEN i.usage_type = 'REACTION' THEN 1 ELSE 0 END) reaction,
                SUM(CASE WHEN i.usage_type = 'SPECIFIC' THEN 1 ELSE 0 END) specific,
                SUM(CASE WHEN i.is_animated = 1 THEN 1 ELSE 0 END) animated,
                SUM(CASE WHEN i.ocr_text <> '' OR i.visible_text <> '' THEN 1 ELSE 0 END) text_count,
                COUNT(DISTINCT NULLIF(f.visual_group, '')) visual_groups
                FROM sticker_items i LEFT JOIN sticker_fingerprints f ON f.item_id = i.item_id
                WHERE i.library_id IN (?, ?) AND i.status = 'ACTIVE'""",
                ids,
            ).fetchone()
            core_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM sticker_items WHERE library_id = ? AND status = 'ACTIVE'",
                    (ids[0],),
                ).fetchone()[0]
            )
            coverage_rows = conn.execute(
                """SELECT i.semantic_key, i.emotion, i.speech_act, i.vibe_tags_json,
                i.search_keywords_json, i.compact_description, i.usage_count, i.metadata_json
                FROM sticker_items i
                WHERE i.library_id IN (?, ?) AND i.status = 'ACTIVE'
                ORDER BY i.created_at DESC LIMIT 1000""",
                ids,
            ).fetchall()
            return _inventory_view(ids, core_count, dict(row or {}), coverage_rows)

        return await self.uow.run(operation)


def _inventory_count(values: Mapping[str, Any], key: str) -> int:
    return int(values.get(key) or 0)


def _inventory_view(
    library_ids: tuple[str, str],
    core_count: int,
    values: Mapping[str, Any],
    coverage_rows: list[Any],
) -> dict[str, Any]:
    total = _inventory_count(values, "total")
    animated = _inventory_count(values, "animated")
    text_count = _inventory_count(values, "text_count")
    return {
        "core_library_id": library_ids[0],
        "private_library_id": library_ids[1],
        "core_count": core_count,
        "private_count": max(0, total - core_count),
        "total": total,
        "ambient": _inventory_count(values, "ambient"),
        "reaction": _inventory_count(values, "reaction"),
        "specific": _inventory_count(values, "specific"),
        "animated": animated,
        "static": max(0, total - animated),
        "text": text_count,
        "no_text": max(0, total - text_count),
        "visual_groups": _inventory_count(values, "visual_groups"),
        "coverage": [_coverage_view(row) for row in coverage_rows],
    }


def _coverage_view(row: Any) -> dict[str, Any]:
    metadata = _load(row["metadata_json"]) or {}
    intent = (
        metadata.get("collection_intent")
        if isinstance(metadata, Mapping) and isinstance(metadata.get("collection_intent"), Mapping)
        else {}
    )
    return {
        "semantic_key": str(row["semantic_key"] or "")[:200],
        "emotion": str(row["emotion"] or "")[:100],
        "speech_act": str(row["speech_act"] or "")[:100],
        "vibe_tags": " ".join(str(value) for value in (_load(row["vibe_tags_json"]) or ()))[:500],
        "search_keywords": " ".join(
            str(value) for value in (_load(row["search_keywords_json"]) or ())
        )[:1000],
        "compact_description": str(row["compact_description"] or "")[:500],
        "usage_count": max(0, int(row["usage_count"] or 0)),
        "collection_scope": str(intent.get("身份边界") or "")[:100],
    }


__all__ = ["StickerLibraryRecords"]
