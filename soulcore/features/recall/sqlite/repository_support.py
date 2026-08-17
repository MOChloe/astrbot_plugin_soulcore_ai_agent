"""Shared state queries, row decoding, and accounting for Recall SQLite."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ....storage.sqlite.engine import SqliteEngine


class RecallSqliteRepositorySupport:
    if TYPE_CHECKING:
        db: SqliteEngine

    async def projection_state(self, profile_id: str, instance_id: str) -> dict[str, int]:
        row = await self.db.fetch_one(
            """SELECT
                (SELECT COUNT(*) FROM recall_documents
                 WHERE profile_id = ? AND instance_id = ?) AS documents,
                (SELECT COUNT(*) FROM recall_index_outbox
                 WHERE profile_id = ? AND instance_id = ?
                   AND status IN ('PENDING', 'LEASED', 'FAILED')) AS pending,
                (SELECT COUNT(DISTINCT document_key) FROM recall_documents_fts
                 WHERE profile_id = ? AND instance_id = ?) AS fts_documents,
                (SELECT COUNT(*) FROM recall_documents_fts
                 WHERE profile_id = ? AND instance_id = ?) AS fts_rows""",
            (
                profile_id,
                instance_id,
                profile_id,
                instance_id,
                profile_id,
                instance_id,
                profile_id,
                instance_id,
            ),
        )
        return {
            "documents": int(row["documents"] or 0),
            "pending": int(row["pending"] or 0),
            "fts_documents": int(row["fts_documents"] or 0),
            "fts_rows": int(row["fts_rows"] or 0),
        }

    async def projection_work_state(self, profile_id: str, instance_id: str) -> dict[str, int]:
        """Return cheap state after this service verified FTS integrity."""

        row = await self.db.fetch_one(
            """SELECT
                (SELECT COUNT(*) FROM recall_documents
                 WHERE profile_id = ? AND instance_id = ?) AS documents,
                (SELECT COUNT(*) FROM recall_index_outbox
                 WHERE profile_id = ? AND instance_id = ?
                   AND status IN ('PENDING', 'LEASED', 'FAILED')) AS pending""",
            (profile_id, instance_id, profile_id, instance_id),
        )
        return {"documents": int(row["documents"] or 0), "pending": int(row["pending"] or 0)}

    @staticmethod
    def _refresh_generation_counts(
        conn: sqlite3.Connection, profile_id: str, instance_id: str, now: str
    ) -> None:
        conn.execute(
            """UPDATE recall_index_generations AS generation
            SET document_count = (
                    SELECT COUNT(*) FROM recall_documents AS document
                    WHERE document.profile_id = generation.profile_id
                      AND document.instance_id = generation.instance_id
                      AND document.dense_eligible = 1
                ),
                embedded_count = (
                    SELECT COUNT(*) FROM recall_embeddings AS embedding
                    JOIN recall_documents AS document
                      ON document.document_key = embedding.document_key
                    WHERE embedding.generation_id = generation.generation_id
                      AND document.source_fingerprint = embedding.content_hash
                ),
                updated_at = ?
            WHERE generation.profile_id = ? AND generation.instance_id = ?""",
            (now, profile_id, instance_id),
        )

    @staticmethod
    def _nullable_provider(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)

    @staticmethod
    def _records(rows: Any) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in rows)

    @classmethod
    def _record(cls, row: Any, *, json_columns: Sequence[str] = ()) -> dict[str, Any]:
        item = dict(row)
        for column in json_columns:
            value = item.pop(column, None)
            item[column.removesuffix("_json")] = cls._load_json(value)
        return item

    @classmethod
    def _document(cls, row: Any) -> dict[str, Any]:
        return cls._record(row, json_columns=("entity_names_json", "evidence_json"))

    @staticmethod
    def _load_json(value: Any) -> Any:
        try:
            return json.loads(str(value or ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            return []


__all__ = ["RecallSqliteRepositorySupport"]
