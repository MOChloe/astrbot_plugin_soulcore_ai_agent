"""Scoped sticker-library rebuild transaction."""

from __future__ import annotations

import sqlite3
from typing import Any


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


__all__ = ["StickerClearTransaction"]
