from __future__ import annotations

from ....storage.sqlite.core_mappers import CoreRecordMappers
from ....storage.sqlite.repository import SqliteRepository
from ...profiles.ports import ProfilesRepositoryPort
from .configuration import WebConfigurationRecords
from .research import WebResearchRecords
from .support import _dump, _load, sqlite3


class WebConfigurationSql:
    @staticmethod
    def _validate_ai_identifier(value: str, field: str) -> str:
        normalized = str(value or "").strip()
        if (
            not normalized
            or len(normalized) > 200
            # Provider IDs are opaque database keys and may contain '/'.
            or any(character in normalized for character in "\0\r\n")
        ):
            raise ValueError(f"invalid {field}")
        return normalized

    @staticmethod
    def _require_expected_version(
        row: sqlite3.Row, expected_version: int | None, label: str
    ) -> None:
        if expected_version is not None and int(row["version"]) != int(expected_version):
            raise ValueError(f"{label} version conflict")

    @staticmethod
    def _normalize_web_provider_priorities(
        conn: sqlite3.Connection,
        profile_id: str,
        now: str,
        moved_provider_id: str | None = None,
        requested_priority: int | None = None,
    ) -> None:
        rows = list(
            conn.execute(
                """SELECT provider_id, priority FROM web_search_providers
            WHERE profile_id = ? AND archived_at IS NULL
            ORDER BY priority, provider_id""",
                (profile_id,),
            )
        )
        ids = [str(row["provider_id"]) for row in rows]
        old = {str(row["provider_id"]): int(row["priority"]) for row in rows}
        if moved_provider_id is not None and moved_provider_id in ids:
            ids.remove(moved_provider_id)
            index = min(max(0, int(requested_priority or 1) - 1), len(ids))
            ids.insert(index, moved_provider_id)
        # The active-priority unique index requires a collision-free two phase move.
        conn.execute(
            """UPDATE web_search_providers SET priority = priority + 1000000
            WHERE profile_id = ? AND archived_at IS NULL""",
            (profile_id,),
        )
        for priority, provider_id in enumerate(ids, start=1):
            conn.execute(
                """UPDATE web_search_providers SET priority = ?,
                version = version + ?, updated_at = ? WHERE provider_id = ?""",
                (priority, int(old.get(provider_id) != priority), now, provider_id),
            )

    @staticmethod
    def _sync_web_provider_runtime(conn: sqlite3.Connection, provider_id: str, now: str) -> None:
        row = conn.execute(
            "SELECT * FROM web_search_providers WHERE provider_id = ?",
            (provider_id,),
        ).fetchone()
        if row is None:
            return
        active = bool(row["enabled"]) and row["archived_at"] is None
        metadata = {
            "profile_id": str(row["profile_id"]),
            "provider_id": provider_id,
            "provider_kind": str(row["provider_kind"]),
            "credential_id": str(row["credential_id"]),
            "read_enabled": bool(row["read_enabled"]),
            "config": _load(row["config_json"]) or {},
        }
        conn.execute(
            """UPDATE ai_backends SET display_name = ?, enabled = ?,
            metadata_json = ?, version = version + 1, updated_at = ?
            WHERE backend_id = ?""",
            (
                row["display_name"],
                int(active),
                _dump(metadata),
                now,
                row["backend_id"],
            ),
        )
        capabilities = (
            ("web.search", True),
            ("web.read", bool(row["read_enabled"])),
        )
        for capability, supported in capabilities:
            conn.execute(
                """INSERT INTO ai_capability_pools(
                    capability, backend_id, priority, enabled, config_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, '{}', ?, ?)
                ON CONFLICT(capability, backend_id) DO UPDATE SET
                    priority = excluded.priority, enabled = excluded.enabled,
                    version = ai_capability_pools.version + 1,
                    updated_at = excluded.updated_at""",
                (
                    capability,
                    row["backend_id"],
                    int(row["priority"]),
                    int(active and supported),
                    now,
                    now,
                ),
            )


class SqliteWebRepository(
    WebConfigurationRecords,
    WebResearchRecords,
    WebConfigurationSql,
    CoreRecordMappers,
    SqliteRepository,
):
    """SQLite implementation of web provider and research persistence."""

    def __init__(self, engine, profiles: ProfilesRepositoryPort) -> None:
        SqliteRepository.__init__(self, engine)
        self._profiles = profiles


__all__ = ["SqliteWebRepository"]
