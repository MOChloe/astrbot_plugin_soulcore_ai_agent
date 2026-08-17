"""Atomic reset of every conversation-owned datum for one character instance."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from ....contracts.runtime_cleanup import STICKER_RELEASE_PATHS_KEY
from ....storage.sqlite.instance_runtime import seed_instance_runtime_rows
from ....storage.sqlite.runtime_file_cleanup import (
    enqueue_runtime_file_cleanup_sql,
    queue_owned_runtime_files_sql,
)
from ....storage.sqlite.tables import (
    INSTANCE_RESET_DELETE_TABLES,
    INSTANCE_RESET_INDIRECT_CASCADE_TABLES,
    INSTANCE_RESET_STICKER_INVENTORY_TABLES,
)


@dataclass(frozen=True, slots=True)
class InstanceResetTransaction:
    profile_id: str
    instance_id: str
    preserve_stickers: bool
    now: str

    def __call__(self, conn: sqlite3.Connection) -> dict[str, Any]:
        # The static inventory is already child-first. Deferral also keeps this
        # explicit reset atomic if a later schema adds a stricter cross-table edge.
        conn.execute("PRAGMA defer_foreign_keys = ON")
        transferred = (
            self._transfer_shared_sticker_inventory(conn)
            if not self.preserve_stickers
            else {"items": 0, "clusters": 0, "fingerprints": 0}
        )
        sticker_context = self._sticker_context(conn)
        skipped = (
            frozenset(INSTANCE_RESET_STICKER_INVENTORY_TABLES)
            if self.preserve_stickers
            else frozenset()
        )
        deleted: dict[str, Any] = {
            "shared_sticker_items_transferred": transferred["items"],
            "shared_sticker_clusters_transferred": transferred["clusters"],
            "shared_sticker_fingerprints_transferred": transferred["fingerprints"],
        }
        deleted.update(
            queue_owned_runtime_files_sql(
                conn,
                profile_id=self.profile_id,
                instance_id=self.instance_id,
                reason="INSTANCE_RUNTIME_RESET",
                now=self.now,
            )
        )
        for table in INSTANCE_RESET_DELETE_TABLES:
            if table in skipped or table in INSTANCE_RESET_INDIRECT_CASCADE_TABLES:
                continue
            if table == "sticker_clusters":
                self._refresh_sticker_clusters(conn, sticker_context["cluster_ids"])
                cursor = conn.execute(
                    """DELETE FROM sticker_clusters
                    WHERE profile_id = ? AND instance_id = ?
                      AND NOT EXISTS (
                        SELECT 1 FROM sticker_items item
                        WHERE item.cluster_id = sticker_clusters.cluster_id
                      )""",
                    (self.profile_id, self.instance_id),
                )
            else:
                cursor = conn.execute(
                    f"DELETE FROM {table} WHERE profile_id = ? AND instance_id = ?",
                    (self.profile_id, self.instance_id),
                )
            deleted[table] = int(cursor.rowcount)

        self._refresh_remaining_sticker_items(conn, sticker_context["item_ids"])
        if not self.preserve_stickers:
            self._refresh_sticker_libraries(conn, sticker_context["library_ids"])
            release_paths, released = self._delete_unreferenced_sticker_assets(
                conn, sticker_context["asset_ids"]
            )
            deleted["sticker_assets"] = released
            deleted[STICKER_RELEASE_PATHS_KEY] = release_paths

        self._reset_core_clock(conn, deleted)
        self._reset_initialization_state(conn, deleted)
        self._seed_empty_runtime_rows(conn)
        return deleted

    def _transfer_shared_sticker_inventory(self, conn: sqlite3.Connection) -> dict[str, int]:
        anchor = conn.execute(
            """SELECT other.instance_id
            FROM character_instances current
            JOIN character_instances other
              ON other.profile_id = current.profile_id
             AND other.scope = current.scope
             AND other.instance_id <> current.instance_id
            WHERE current.profile_id = ? AND current.instance_id = ?
            ORDER BY other.created_at, other.instance_id
            LIMIT 1""",
            (self.profile_id, self.instance_id),
        ).fetchone()
        if anchor is None:
            return {"items": 0, "clusters": 0, "fingerprints": 0}
        anchor_id = str(anchor["instance_id"])
        shared_rows = list(
            conn.execute(
                """SELECT item.item_id, item.cluster_id
                FROM sticker_items item
                JOIN sticker_libraries library ON library.library_id = item.library_id
                WHERE item.profile_id = ? AND item.instance_id = ?
                  AND library.library_kind = 'CORE'
                  AND (
                    EXISTS (
                      SELECT 1 FROM sticker_usages usage
                      WHERE usage.item_id = item.item_id AND usage.instance_id <> ?
                    )
                    OR EXISTS (
                      SELECT 1 FROM sticker_reinforcements reinforcement
                      WHERE reinforcement.item_id = item.item_id
                        AND reinforcement.instance_id <> ?
                    )
                    OR EXISTS (
                      SELECT 1 FROM sticker_run_candidates candidate
                      WHERE candidate.item_id = item.item_id AND candidate.instance_id <> ?
                    )
                    OR EXISTS (
                      SELECT 1 FROM sticker_import_events import_event
                      WHERE import_event.item_id = item.item_id
                        AND import_event.instance_id <> ?
                    )
                  )""",
                (
                    self.profile_id,
                    self.instance_id,
                    self.instance_id,
                    self.instance_id,
                    self.instance_id,
                    self.instance_id,
                ),
            )
        )
        if not shared_rows:
            return {"items": 0, "clusters": 0, "fingerprints": 0}
        item_ids = sorted({str(row["item_id"]) for row in shared_rows})
        cluster_ids = sorted({str(row["cluster_id"]) for row in shared_rows})
        item_marks = ",".join("?" for _ in item_ids)
        fingerprints = conn.execute(
            f"""UPDATE sticker_fingerprints SET instance_id = ?, updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND item_id IN ({item_marks})""",
            (
                anchor_id,
                self.now,
                self.profile_id,
                self.instance_id,
                *item_ids,
            ),
        )
        items = conn.execute(
            f"""UPDATE sticker_items SET instance_id = ?, updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND item_id IN ({item_marks})""",
            (
                anchor_id,
                self.now,
                self.profile_id,
                self.instance_id,
                *item_ids,
            ),
        )
        clusters = None
        if cluster_ids:
            cluster_marks = ",".join("?" for _ in cluster_ids)
            clusters = conn.execute(
                f"""UPDATE sticker_clusters SET instance_id = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?
                  AND cluster_id IN ({cluster_marks})""",
                (
                    anchor_id,
                    self.now,
                    self.profile_id,
                    self.instance_id,
                    *cluster_ids,
                ),
            )
        return {
            "items": int(items.rowcount),
            "clusters": int(clusters.rowcount if clusters is not None else 0),
            "fingerprints": int(fingerprints.rowcount),
        }

    def _sticker_context(self, conn: sqlite3.Connection) -> dict[str, set[str]]:
        owned_items = list(
            conn.execute(
                """SELECT item_id, asset_id, cluster_id, library_id
                FROM sticker_items WHERE profile_id = ? AND instance_id = ?""",
                (self.profile_id, self.instance_id),
            )
        )
        usage_item_ids = self._column_values(
            conn,
            """SELECT item_id FROM sticker_usages
            WHERE profile_id = ? AND instance_id = ?""",
        )
        item_ids = usage_item_ids
        if not self.preserve_stickers:
            item_ids |= {str(row["item_id"]) for row in owned_items}
            for table in ("sticker_reinforcements", "sticker_import_events"):
                item_ids |= self._column_values(
                    conn,
                    f"""SELECT item_id FROM {table}
                    WHERE profile_id = ? AND instance_id = ? AND item_id IS NOT NULL""",
                )
        cluster_ids = {str(row["cluster_id"]) for row in owned_items}
        library_ids = {str(row["library_id"]) for row in owned_items}
        asset_ids = {str(row["asset_id"]) for row in owned_items}
        if not self.preserve_stickers:
            cluster_ids |= self._column_values(
                conn,
                """SELECT cluster_id FROM sticker_clusters
                WHERE profile_id = ? AND instance_id = ?""",
            )
            library_ids |= self._column_values(
                conn,
                """SELECT library_id FROM sticker_libraries
                WHERE profile_id = ? AND instance_id = ?""",
            )
            library_ids |= self._column_values(
                conn,
                """SELECT target_library_id FROM sticker_candidates
                WHERE profile_id = ? AND instance_id = ?""",
            )
        return {
            "item_ids": item_ids,
            "cluster_ids": cluster_ids,
            "library_ids": library_ids,
            "asset_ids": asset_ids,
        }

    def _column_values(self, conn: sqlite3.Connection, sql: str) -> set[str]:
        return {
            str(row[0])
            for row in conn.execute(sql, (self.profile_id, self.instance_id))
            if row[0] is not None and str(row[0])
        }

    def _refresh_remaining_sticker_items(
        self, conn: sqlite3.Connection, item_ids: set[str]
    ) -> None:
        for item_id in item_ids:
            if self.preserve_stickers:
                conn.execute(
                    """UPDATE sticker_items SET
                        usage_count = (
                            SELECT COUNT(*) FROM sticker_usages usage
                            WHERE usage.item_id = sticker_items.item_id
                        ),
                        last_used_at = (
                            SELECT MAX(created_at) FROM sticker_usages usage
                            WHERE usage.item_id = sticker_items.item_id
                        ),
                        updated_at = ?
                    WHERE item_id = ?""",
                    (self.now, item_id),
                )
                continue
            conn.execute(
                """UPDATE sticker_items SET
                    usage_count = (
                        SELECT COUNT(*) FROM sticker_usages usage
                        WHERE usage.item_id = sticker_items.item_id
                    ),
                    last_used_at = (
                        SELECT MAX(created_at) FROM sticker_usages usage
                        WHERE usage.item_id = sticker_items.item_id
                    ),
                    reinforcement_score = COALESCE((
                        SELECT SUM(strength) FROM sticker_reinforcements reinforcement
                        WHERE reinforcement.item_id = sticker_items.item_id
                    ), 0),
                    import_count = MAX(1, (
                        SELECT COUNT(*) FROM sticker_import_events import_event
                        WHERE import_event.item_id = sticker_items.item_id
                    )),
                    updated_at = ?
                WHERE item_id = ?""",
                (self.now, item_id),
            )

    def _refresh_sticker_clusters(self, conn: sqlite3.Connection, cluster_ids: set[str]) -> None:
        for cluster_id in cluster_ids:
            conn.execute(
                """UPDATE sticker_clusters SET
                    active_count = (
                        SELECT COUNT(*) FROM sticker_items item
                        WHERE item.cluster_id = sticker_clusters.cluster_id
                          AND item.status IN ('ACTIVE', 'NEEDS_REVIEW')
                    ),
                    auto_count = (
                        SELECT COUNT(*) FROM sticker_items item
                        WHERE item.cluster_id = sticker_clusters.cluster_id
                          AND item.status IN ('ACTIVE', 'NEEDS_REVIEW')
                          AND item.source_kind <> 'PLAYER'
                    ),
                    updated_at = ?
                WHERE cluster_id = ?""",
                (self.now, cluster_id),
            )

    def _refresh_sticker_libraries(self, conn: sqlite3.Connection, library_ids: set[str]) -> None:
        for library_id in library_ids:
            active = int(
                conn.execute(
                    """SELECT COUNT(*) FROM sticker_items
                    WHERE library_id = ? AND status IN ('ACTIVE', 'NEEDS_REVIEW')""",
                    (library_id,),
                ).fetchone()[0]
            )
            conn.execute(
                """UPDATE sticker_library_states SET
                    last_error = CASE WHEN ? = 0 THEN '' ELSE last_error END,
                    updated_at = ?
                WHERE library_id = ?""",
                (active, self.now, library_id),
            )

    def _delete_unreferenced_sticker_assets(
        self, conn: sqlite3.Connection, asset_ids: set[str]
    ) -> tuple[list[str], int]:
        if not asset_ids:
            return [], 0
        marks = ",".join("?" for _ in asset_ids)
        rows = list(
            conn.execute(
                f"""SELECT sticker_asset_id, storage_relpath,
                    canonical_sha256, byte_size
                FROM sticker_assets asset
                WHERE sticker_asset_id IN ({marks})
                  AND NOT EXISTS (
                    SELECT 1 FROM sticker_items item
                    WHERE item.asset_id = asset.sticker_asset_id
                  )""",
                tuple(sorted(asset_ids)),
            )
        )
        released_ids = [str(row["sticker_asset_id"]) for row in rows]
        if not released_ids:
            return [], 0
        for row in rows:
            enqueue_runtime_file_cleanup_sql(
                conn,
                profile_id=self.profile_id,
                instance_id=self.instance_id,
                storage_kind="MEDIA",
                storage_relpath=str(row["storage_relpath"]),
                owner_id=str(row["sticker_asset_id"]),
                expected_sha256=str(row["canonical_sha256"]),
                expected_byte_size=int(row["byte_size"]),
                reason="INSTANCE_STICKER_RESET",
                now=self.now,
            )
        release_marks = ",".join("?" for _ in released_ids)
        cursor = conn.execute(
            f"DELETE FROM sticker_assets WHERE sticker_asset_id IN ({release_marks})",
            tuple(released_ids),
        )
        return [str(row["storage_relpath"]) for row in rows], int(cursor.rowcount)

    def _reset_core_clock(self, conn: sqlite3.Connection, deleted: dict[str, Any]) -> None:
        cursor = conn.execute(
            """UPDATE instance_core_state SET state_epoch = 0,
                activity_epoch = 0, low_frequency_mode = 0,
                low_frequency_reason = '', low_frequency_since = NULL,
                updated_at = ? WHERE profile_id = ? AND instance_id = ?""",
            (self.now, self.profile_id, self.instance_id),
        )
        deleted["instance_core_state_reset"] = int(cursor.rowcount)

    def _reset_initialization_state(
        self, conn: sqlite3.Connection, deleted: dict[str, Any]
    ) -> None:
        cursor = conn.execute(
            """UPDATE character_instances SET initialization_state = 'UNINITIALIZED',
                initialization_completed_at = NULL, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?
              AND initialization_state <> 'UNINITIALIZED'""",
            (self.now, self.profile_id, self.instance_id),
        )
        deleted["character_instance_initialization_reset"] = int(cursor.rowcount)

    def _seed_empty_runtime_rows(self, conn: sqlite3.Connection) -> None:
        seed_instance_runtime_rows(
            conn,
            self.profile_id,
            self.instance_id,
            self.now,
        )


__all__ = ["InstanceResetTransaction"]
