from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from .item_acceptance import StickerAcceptanceContext, StickerAcceptanceTransaction
from .support import Mapping, StickerCheckRevision, StickerItem, _dt, _dump, _load, _now, math, uuid


class StickerClearTransaction:
    def __init__(self, *, profile_id: str, instance_id: str, scope: str, now: str) -> None:
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.scope = scope
        self.now = now

    def __call__(self, conn: sqlite3.Connection) -> dict[str, Any]:
        library_ids = self._library_ids(conn)
        item_ids = self._item_ids(conn, library_ids)
        candidate_rows = self._candidate_rows(conn, library_ids)
        candidate_ids = [str(row["candidate_id"]) for row in candidate_rows]
        source_ids = [
            str(row["source_asset_id"])
            for row in candidate_rows
            if str(row["source_kind"]) != "PLAYER"
        ]
        release_rows = self._release_rows(conn, source_ids)
        release_ids = [str(row["asset_id"]) for row in release_rows]
        deleted: dict[str, int] = {}
        self._delete_item_relations(conn, item_ids, deleted)
        self._delete_candidates(conn, candidate_ids, deleted)
        self._delete_library_inventory(conn, library_ids, item_ids, deleted)
        self._delete_trigger_states(conn, deleted)
        unreferenced_assets = self._unreferenced_assets(conn)
        self._mark_unreferenced_assets_release_pending(conn, unreferenced_assets, deleted)
        self._mark_media_release_pending(conn, release_ids)
        return {
            "cleared": deleted,
            "release_asset_ids": release_ids,
            "sticker_release_asset_ids": [
                str(row["sticker_asset_id"]) for row in unreferenced_assets
            ],
        }

    def _library_ids(self, conn: sqlite3.Connection) -> list[str]:
        rows = conn.execute(
            """SELECT library_id FROM sticker_libraries
            WHERE profile_id = ? AND (
              (library_kind = 'CORE' AND scope = ?)
              OR (library_kind = 'PRIVATE' AND instance_id = ?)
            )""",
            (self.profile_id, self.scope, self.instance_id),
        )
        return [str(row["library_id"]) for row in rows]

    @staticmethod
    def _item_ids(conn: sqlite3.Connection, library_ids: list[str]) -> list[str]:
        if not library_ids:
            return []
        marks = ",".join("?" for _ in library_ids)
        rows = conn.execute(
            f"SELECT item_id FROM sticker_items WHERE library_id IN ({marks})",
            library_ids,
        )
        return [str(row[0]) for row in rows]

    @staticmethod
    def _candidate_rows(conn: sqlite3.Connection, library_ids: list[str]) -> list[sqlite3.Row]:
        if not library_ids:
            return []
        marks = ",".join("?" for _ in library_ids)
        return list(
            conn.execute(
                f"""SELECT candidate_id, source_asset_id, source_kind
                FROM sticker_candidates WHERE target_library_id IN ({marks})""",
                library_ids,
            )
        )

    @staticmethod
    def _release_rows(conn: sqlite3.Connection, source_ids: list[str]) -> list[sqlite3.Row]:
        if not source_ids:
            return []
        marks = ",".join("?" for _ in source_ids)
        return list(
            conn.execute(
                f"""SELECT DISTINCT a.asset_id
                FROM media_assets a WHERE a.asset_id IN ({marks})
                  AND a.storage_relpath IS NOT NULL AND a.storage_relpath <> ''
                  AND NOT EXISTS (
                    SELECT 1 FROM media_asset_message_links link
                    WHERE link.asset_id = a.asset_id
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM media_retention_holds hold
                    WHERE hold.asset_id = a.asset_id AND hold.released_at IS NULL
                      AND hold.holder_kind <> 'STICKER_CANDIDATE'
                  )""",
                source_ids,
            )
        )

    @staticmethod
    def _delete_item_relations(
        conn: sqlite3.Connection, item_ids: list[str], deleted: dict[str, int]
    ) -> None:
        if not item_ids:
            return
        marks = ",".join("?" for _ in item_ids)
        for table in (
            "sticker_reinforcements",
            "sticker_usages",
            "sticker_run_candidates",
            "sticker_fingerprints",
        ):
            cursor = conn.execute(f"DELETE FROM {table} WHERE item_id IN ({marks})", item_ids)
            deleted[table] = int(cursor.rowcount)
        deleted["sticker_import_events"] = int(
            conn.execute(
                f"DELETE FROM sticker_import_events WHERE item_id IN ({marks})",
                item_ids,
            ).rowcount
        )

    @staticmethod
    def _delete_candidates(
        conn: sqlite3.Connection, candidate_ids: list[str], deleted: dict[str, int]
    ) -> None:
        if not candidate_ids:
            return
        marks = ",".join("?" for _ in candidate_ids)
        deleted["media_retention_holds"] = int(
            conn.execute(
                f"""DELETE FROM media_retention_holds
                WHERE holder_kind = 'STICKER_CANDIDATE' AND holder_id IN ({marks})""",
                candidate_ids,
            ).rowcount
        )
        deleted["sticker_import_events"] = deleted.get("sticker_import_events", 0) + int(
            conn.execute(
                f"DELETE FROM sticker_import_events WHERE candidate_id IN ({marks})",
                candidate_ids,
            ).rowcount
        )
        deleted["sticker_candidates"] = int(
            conn.execute(
                f"DELETE FROM sticker_candidates WHERE candidate_id IN ({marks})",
                candidate_ids,
            ).rowcount
        )

    @staticmethod
    def _delete_library_inventory(
        conn: sqlite3.Connection,
        library_ids: list[str],
        item_ids: list[str],
        deleted: dict[str, int],
    ) -> None:
        deleted["sticker_items"] = StickerClearTransaction._delete_by_ids(
            conn, "sticker_items", "item_id", item_ids
        )
        deleted["sticker_clusters"] = StickerClearTransaction._delete_by_ids(
            conn, "sticker_clusters", "library_id", library_ids
        )
        deleted["sticker_libraries"] = StickerClearTransaction._delete_by_ids(
            conn, "sticker_libraries", "library_id", library_ids
        )

    @staticmethod
    def _delete_by_ids(conn: sqlite3.Connection, table: str, column: str, values: list[str]) -> int:
        if not values:
            return 0
        marks = ",".join("?" for _ in values)
        return int(
            conn.execute(f"DELETE FROM {table} WHERE {column} IN ({marks})", values).rowcount
        )

    def _delete_trigger_states(self, conn: sqlite3.Connection, deleted: dict[str, int]) -> None:
        deleted["sticker_trigger_states"] = int(
            conn.execute(
                """DELETE FROM sticker_trigger_states WHERE profile_id = ?
                AND instance_id IN (
                  SELECT instance_id FROM character_instances
                  WHERE profile_id = ? AND scope = ?
                )""",
                (self.profile_id, self.profile_id, self.scope),
            ).rowcount
        )

    def _unreferenced_assets(self, conn: sqlite3.Connection) -> list[sqlite3.Row]:
        return list(
            conn.execute(
                """SELECT sticker_asset_id, storage_relpath FROM sticker_assets a
                WHERE a.profile_id = ? AND NOT EXISTS (
                  SELECT 1 FROM sticker_items i WHERE i.asset_id = a.sticker_asset_id
                )""",
                (self.profile_id,),
            )
        )

    def _mark_unreferenced_assets_release_pending(
        self,
        conn: sqlite3.Connection,
        assets: list[sqlite3.Row],
        deleted: dict[str, int],
    ) -> None:
        asset_ids = [str(row["sticker_asset_id"]) for row in assets]
        if not asset_ids:
            deleted["sticker_assets_release_pending"] = 0
            return
        marks = ",".join("?" for _ in asset_ids)
        deleted["sticker_assets_release_pending"] = int(
            conn.execute(
                f"""UPDATE sticker_assets SET file_status = 'RELEASE_PENDING',
                updated_at = ? WHERE sticker_asset_id IN ({marks})
                  AND file_status = 'AVAILABLE'""",
                (self.now, *asset_ids),
            ).rowcount
        )

    def _mark_media_release_pending(self, conn: sqlite3.Connection, release_ids: list[str]) -> None:
        if not release_ids:
            return
        marks = ",".join("?" for _ in release_ids)
        conn.execute(
            f"""UPDATE media_assets SET file_status = 'RELEASE_PENDING',
            last_error = 'administrator_rebuilt_sticker_library', updated_at = ?
            WHERE asset_id IN ({marks}) AND file_status NOT IN ('RELEASED','MISSING')""",
            (self.now, *release_ids),
        )


@dataclass(frozen=True, slots=True)
class StickerDescriptionContext:
    profile_id: str
    instance_id: str
    item_id: str
    description: str
    visible_text: str
    search_keywords: tuple[str, ...]
    metadata_update: Mapping[str, Any]
    expected_description: str
    now: str


class StickerDescriptionTransaction:
    def __init__(self, owner: Any, context: StickerDescriptionContext) -> None:
        self.owner = owner
        self.context = context

    def __call__(self, conn: sqlite3.Connection) -> dict[str, Any]:
        row = self._load_available_item(conn)
        keywords = self.context.search_keywords or tuple(_load(row["search_keywords_json"]) or ())
        metadata = self._merged_metadata(row)
        search_index = self.owner._normalize_sticker_semantic(
            " ".join(
                (
                    self.context.description,
                    self.context.visible_text,
                    str(row["semantic_key"] or ""),
                    str(row["emotion"] or ""),
                    str(row["speech_act"] or ""),
                    *(str(value) for value in keywords),
                )
            )
        )[:8000]
        self._update_item(conn, keywords, metadata, search_index)
        refreshed = self._load_refreshed_item(conn)
        return dict(refreshed)

    def _load_available_item(self, conn: sqlite3.Connection) -> sqlite3.Row:
        context = self.context
        row = conn.execute(
            """SELECT i.*, a.file_status
            FROM sticker_items i JOIN sticker_assets a ON a.sticker_asset_id = i.asset_id
            JOIN sticker_libraries l ON l.library_id = i.library_id
            JOIN character_instances current
              ON current.profile_id = ? AND current.instance_id = ?
            WHERE i.item_id = ? AND i.profile_id = ? AND (
              (l.library_kind = 'CORE' AND l.scope = current.scope)
              OR (l.library_kind = 'PRIVATE' AND l.instance_id = current.instance_id)
            )""",
            (
                context.profile_id,
                context.instance_id,
                context.item_id,
                context.profile_id,
            ),
        ).fetchone()
        if row is None:
            raise KeyError((context.profile_id, context.instance_id, context.item_id))
        if str(row["status"]) == "DELETED":
            raise ValueError("deleted sticker cannot be updated")
        if self._media_unavailable(row):
            raise ValueError("formal sticker media is unavailable")
        if str(row["compact_description"] or "") != context.expected_description:
            raise ValueError("sticker description changed during regeneration")
        return row

    @staticmethod
    def _media_unavailable(row: sqlite3.Row) -> bool:
        return str(row["file_status"]) != "AVAILABLE"

    def _merged_metadata(self, row: sqlite3.Row) -> dict[str, Any]:
        current = _load(row["metadata_json"]) or {}
        if not isinstance(current, Mapping):
            current = {}
        return {
            **dict(current),
            **dict(self.context.metadata_update),
            "description_refreshed_at": self.context.now,
        }

    def _update_item(
        self,
        conn: sqlite3.Connection,
        keywords: tuple[str, ...],
        metadata: dict[str, Any],
        search_index: str,
    ) -> None:
        context = self.context
        cursor = conn.execute(
            """UPDATE sticker_items SET compact_description = ?,
                visible_text = ?, search_keywords_json = ?, search_index = ?,
                metadata_json = ?, updated_at = ?
            WHERE item_id = ? AND profile_id = ?
              AND compact_description = ? AND status <> 'DELETED'""",
            (
                context.description,
                context.visible_text,
                _dump(list(keywords)),
                search_index,
                _dump(metadata),
                context.now,
                context.item_id,
                context.profile_id,
                context.expected_description,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("sticker description changed during regeneration")

    def _load_refreshed_item(self, conn: sqlite3.Connection) -> sqlite3.Row:
        context = self.context
        row = conn.execute(
            """SELECT i.*, l.library_kind, l.scope, f.phash, f.dhash, f.visual_group
            FROM sticker_items i JOIN sticker_libraries l ON l.library_id = i.library_id
            LEFT JOIN sticker_fingerprints f ON f.item_id = i.item_id
            WHERE i.item_id = ? AND i.profile_id = ?""",
            (context.item_id, context.profile_id),
        ).fetchone()
        if row is None:
            raise KeyError((context.profile_id, context.instance_id, context.item_id))
        return row


class StickerItemRecords:
    async def preflight_sticker_visual_capacity(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
        *,
        phash: str,
        dhash: str,
    ) -> dict[str, Any]:
        """Reject saturated perceptual groups before any model-backed Check.

        This is an inexpensive early gate.  The acceptance transaction repeats
        the same capacity rule as the authoritative concurrency boundary.
        """

        candidate = await self.db.fetch_one(
            """SELECT target_library_id, source_kind FROM sticker_candidates
            WHERE candidate_id = ? AND profile_id = ? AND instance_id = ?""",
            (candidate_id, profile_id, instance_id),
        )
        if candidate is None:
            raise KeyError((profile_id, instance_id, candidate_id))
        rows = await self.db.fetch_all(
            """SELECT f.visual_group, f.phash, f.dhash, i.source_kind
            FROM sticker_fingerprints f JOIN sticker_items i ON i.item_id = f.item_id
            WHERE f.library_id = ? AND i.status IN ('ACTIVE', 'NEEDS_REVIEW')""",
            (candidate["target_library_id"],),
        )
        visual_group = _nearest_visual_group(self, rows, phash=phash, dhash=dhash)
        total, auto_count = _visual_group_counts(rows, visual_group)
        is_auto = str(candidate["source_kind"]) != "PLAYER"
        return {
            "allowed": total < 6 and (not is_auto or auto_count < 4),
            "visual_group": visual_group,
            "total": total,
            "auto_count": auto_count,
        }

    async def clear_sticker_instance_data(
        self, profile_id: str, instance_id: str
    ) -> dict[str, Any]:
        instance = await self._profiles.get_character_instance(profile_id, instance_id)
        if instance is None:
            raise KeyError((profile_id, instance_id))
        transaction = StickerClearTransaction(
            profile_id=profile_id,
            instance_id=instance_id,
            scope=str(instance.scope),
            now=_dt(_now()),
        )
        return await self.uow.run(transaction)

    async def list_sticker_checks(
        self,
        profile_id: str,
        instance_id: str,
        *,
        candidate_id: str | None = None,
        limit: int = 100,
    ) -> list[StickerCheckRevision]:
        values: list[Any] = [profile_id, instance_id]
        clause = ""
        if candidate_id is not None:
            clause = " AND c.candidate_id = ?"
            values.append(candidate_id)
        values.append(max(1, min(500, int(limit))))
        rows = await self.db.fetch_all(
            f"""SELECT r.* FROM sticker_check_revisions r
            JOIN sticker_candidates c ON c.candidate_id = r.candidate_id
            WHERE c.profile_id = ? AND c.instance_id = ? {clause}
            ORDER BY r.check_id DESC LIMIT ?""",
            values,
        )
        return [self._sticker_check(row) for row in rows]

    async def page_sticker_checks(
        self,
        profile_id: str,
        instance_id: str,
        *,
        statuses: tuple[str, ...] | list[str] = (),
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        verdicts = tuple(dict.fromkeys(str(value).upper() for value in statuses))
        size = max(1, min(100, int(page_size)))
        current = max(1, int(page))
        clauses = ["c.profile_id = ?", "c.instance_id = ?"]
        values: list[Any] = [profile_id, instance_id]
        if verdicts:
            placeholders = ",".join("?" for _ in verdicts)
            clauses.append(f"r.verdict IN ({placeholders})")
            values.extend(verdicts)
        where = " AND ".join(clauses)
        count = await self.db.fetch_one(
            f"""SELECT COUNT(*) amount FROM sticker_check_revisions r
            JOIN sticker_candidates c ON c.candidate_id = r.candidate_id
            WHERE {where}""",
            values,
        )
        rows = await self.db.fetch_all(
            f"""SELECT r.* FROM sticker_check_revisions r
            JOIN sticker_candidates c ON c.candidate_id = r.candidate_id
            WHERE {where} ORDER BY r.check_id DESC LIMIT ? OFFSET ?""",
            (*values, size, (current - 1) * size),
        )
        total = int(count["amount"] or 0) if count else 0
        return {
            "items": [self._sticker_check(row) for row in rows],
            "total": total,
            "page": current,
            "page_size": size,
            "page_count": max(1, math.ceil(total / size)),
        }

    async def find_sticker_item_by_sha(
        self,
        profile_id: str,
        instance_id: str,
        sha256: str,
    ) -> StickerItem | None:
        row = await self.db.fetch_one(
            """SELECT i.*, l.library_kind, l.scope, f.phash, f.dhash, f.visual_group
            FROM sticker_items i JOIN sticker_libraries l ON l.library_id = i.library_id
            LEFT JOIN sticker_fingerprints f ON f.item_id = i.item_id
            WHERE i.profile_id = ? AND i.library_id IN (
                SELECT visible.library_id FROM sticker_libraries visible
                JOIN character_instances current
                  ON current.profile_id = ? AND current.instance_id = ?
                WHERE visible.profile_id = current.profile_id AND (
                  (visible.library_kind = 'CORE' AND visible.scope = current.scope)
                  OR (visible.library_kind = 'PRIVATE'
                      AND visible.instance_id = current.instance_id)
                )
            )
            AND i.canonical_sha256 = ? AND i.status <> 'DELETED'""",
            (profile_id, profile_id, instance_id, str(sha256).lower()),
        )
        return self._sticker_item(row) if row is not None else None

    async def accept_sticker_candidate(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
        *,
        reserved_asset_id: str,
        compact_description: str = "",
        compact_name: str = "",
        visible_text: str = "",
        ocr_text: str = "",
        usage_type: str = "",
        vibe_tags: list[str] | tuple[str, ...] = (),
        search_keywords: list[str] | tuple[str, ...] = (),
        search_index: str = "",
        semantic_key: str = "",
        emotion: str = "",
        speech_act: str = "",
        intensity: int = 0,
        persona_score: float = 0.0,
        phash: str = "",
        dhash: str = "",
        frame_hashes: list[str] | tuple[str, ...] = (),
        representative_frame_hashes: list[str] | tuple[str, ...] = (),
        visual_group: str = "",
        metadata: Mapping[str, Any] | None = None,
        item_id: str = "",
    ) -> tuple[StickerItem, bool]:
        context = StickerAcceptanceContext(
            profile_id=profile_id,
            instance_id=instance_id,
            candidate_id=candidate_id,
            reserved_asset_id=reserved_asset_id,
            identifier=str(item_id).strip() or "si_" + uuid.uuid4().hex,
            compact_description=compact_description,
            compact_name=compact_name,
            visible_text=visible_text,
            ocr_text=ocr_text,
            usage_type=usage_type,
            vibe_tags=tuple(vibe_tags),
            search_keywords=tuple(search_keywords),
            search_index=search_index,
            semantic_key=semantic_key,
            emotion=emotion,
            speech_act=speech_act,
            intensity=intensity,
            persona_score=persona_score,
            phash=phash,
            dhash=dhash,
            frame_hashes=tuple(frame_hashes),
            representative_frame_hashes=tuple(representative_frame_hashes),
            visual_group=visual_group,
            metadata=metadata or {},
            now=_dt(_now()),
        )
        stored_id, created = await self.uow.run(StickerAcceptanceTransaction(self, context))
        item = await self.get_sticker_item(profile_id, instance_id, stored_id)
        assert item is not None
        return item, created

    async def get_sticker_item(
        self,
        profile_id: str,
        instance_id: str,
        item_id: str,
    ) -> StickerItem | None:
        row = await self.db.fetch_one(
            """SELECT i.*, l.library_kind, l.scope, f.phash, f.dhash, f.visual_group
            FROM sticker_items i JOIN sticker_libraries l ON l.library_id = i.library_id
            LEFT JOIN sticker_fingerprints f ON f.item_id = i.item_id
            JOIN character_instances current
              ON current.profile_id = ? AND current.instance_id = ?
            WHERE i.item_id = ? AND i.profile_id = ? AND (
              (l.library_kind = 'CORE' AND l.scope = current.scope)
              OR (l.library_kind = 'PRIVATE' AND l.instance_id = current.instance_id)
            )""",
            (profile_id, instance_id, item_id, profile_id),
        )
        return self._sticker_item(row) if row is not None else None

    async def update_sticker_item_description(
        self,
        profile_id: str,
        instance_id: str,
        item_id: str,
        *,
        compact_description: str,
        visible_text: str = "",
        search_keywords: list[str] | tuple[str, ...] = (),
        metadata_update: Mapping[str, Any] | None = None,
        expected_description: str,
    ) -> StickerItem:
        description = str(compact_description or "").strip()[:100]
        if not description:
            raise ValueError("sticker description must not be empty")
        keywords = tuple(
            dict.fromkeys(
                str(value).strip()[:100] for value in search_keywords if str(value).strip()
            )
        )[:100]
        context = StickerDescriptionContext(
            profile_id=profile_id,
            instance_id=instance_id,
            item_id=item_id,
            description=description,
            visible_text=str(visible_text or "").strip()[:500],
            search_keywords=keywords,
            metadata_update=metadata_update or {},
            expected_description=str(expected_description or ""),
            now=_dt(_now()),
        )
        refreshed = await self.uow.run(StickerDescriptionTransaction(self, context))
        return self._sticker_item(refreshed)


def _nearest_visual_group(owner: Any, rows: list[Any], *, phash: str, dhash: str) -> str:
    visual_group = str(phash or dhash).split(".", 1)[0]
    nearest_distance = 10_000
    for row in rows:
        distance = min(
            owner._sticker_hash_distance(phash, str(row["phash"] or "")),
            owner._sticker_hash_distance(dhash, str(row["dhash"] or "")),
        )
        if distance <= 6 and distance < nearest_distance:
            nearest_distance = distance
            visual_group = str(row["visual_group"] or row["phash"] or visual_group)
    return visual_group


def _visual_group_counts(rows: list[Any], visual_group: str) -> tuple[int, int]:
    total = 0
    auto_count = 0
    for row in rows:
        if not visual_group or str(row["visual_group"] or "") != visual_group:
            continue
        total += 1
        if str(row["source_kind"]) != "PLAYER":
            auto_count += 1
    return total, auto_count


__all__ = ["StickerClearTransaction"]
