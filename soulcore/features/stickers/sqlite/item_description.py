from __future__ import annotations

from dataclasses import dataclass

from .support import Any, Mapping, _dump, _load, sqlite3


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
