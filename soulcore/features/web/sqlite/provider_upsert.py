from __future__ import annotations

from dataclasses import dataclass

from .support import Any, WebSearchProviderRecord, _dump, sqlite3


@dataclass(frozen=True, slots=True)
class WebProviderUpsertContext:
    provider: WebSearchProviderRecord
    provider_id: str
    profile_id: str
    provider_kind: str
    backend_id: str
    expected_version: int | None
    now: str


class WebProviderUpsertTransaction:
    def __init__(self, owner: Any, context: WebProviderUpsertContext) -> None:
        self.owner = owner
        self.context = context

    def __call__(self, conn: sqlite3.Connection) -> sqlite3.Row:
        self._require_profile(conn)
        row = conn.execute(
            "SELECT * FROM web_search_providers WHERE provider_id = ?",
            (self.context.provider_id,),
        ).fetchone()
        if row is None:
            self._insert_provider(conn)
        else:
            self._update_provider(conn, row)
        context = self.context
        self.owner._normalize_web_provider_priorities(
            conn,
            context.profile_id,
            context.now,
            context.provider_id,
            max(1, int(context.provider.priority)),
        )
        self.owner._sync_web_provider_runtime(conn, context.provider_id, context.now)
        result = conn.execute(
            "SELECT * FROM web_search_providers WHERE provider_id = ?",
            (context.provider_id,),
        ).fetchone()
        assert result is not None
        return result

    def _require_profile(self, conn: sqlite3.Connection) -> None:
        profile_id = self.context.profile_id
        row = conn.execute(
            "SELECT 1 FROM role_profiles WHERE profile_id = ?", (profile_id,)
        ).fetchone()
        if row is None:
            raise KeyError(profile_id)

    def _insert_provider(self, conn: sqlite3.Connection) -> None:
        context = self.context
        provider = context.provider
        backend = conn.execute(
            "SELECT 1 FROM ai_backends WHERE backend_id = ?",
            (context.backend_id,),
        ).fetchone()
        if backend is not None:
            raise ValueError("backend_id is already in use")
        conn.execute(
            """INSERT INTO ai_backends(
                backend_id, backend_kind, display_name, enabled,
                metadata_json, created_at, updated_at
            ) VALUES (?, 'WEB_RESEARCH', ?, ?, ?, ?, ?)""",
            (
                context.backend_id,
                provider.display_name or context.provider_kind,
                int(provider.enabled),
                _dump(
                    {
                        "profile_id": context.profile_id,
                        "provider_kind": context.provider_kind,
                    }
                ),
                context.now,
                context.now,
            ),
        )
        priority = self._next_priority(conn)
        conn.execute(
            """INSERT INTO web_search_providers(
                provider_id, profile_id, provider_kind, display_name,
                backend_id, credential_id, priority, enabled, read_enabled,
                config_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                context.provider_id,
                context.profile_id,
                context.provider_kind,
                str(provider.display_name or context.provider_kind),
                context.backend_id,
                str(provider.credential_id or ""),
                priority,
                int(provider.enabled),
                int(provider.read_enabled),
                _dump(provider.config),
                context.now,
                context.now,
            ),
        )

    def _next_priority(self, conn: sqlite3.Connection) -> int:
        return int(
            conn.execute(
                """SELECT COALESCE(MAX(priority), 0) + 1
            FROM web_search_providers
            WHERE profile_id = ? AND archived_at IS NULL""",
                (self.context.profile_id,),
            ).fetchone()[0]
        )

    def _update_provider(self, conn: sqlite3.Connection, row: sqlite3.Row) -> None:
        context = self.context
        provider = context.provider
        if str(row["profile_id"]) != context.profile_id:
            raise ValueError("web search provider belongs to another profile")
        if row["archived_at"] is not None:
            raise ValueError("archived web search provider cannot be modified")
        self.owner._require_expected_version(row, context.expected_version, "web search provider")
        if str(row["backend_id"]) != context.backend_id:
            raise ValueError("web search provider backend identity is immutable")
        conn.execute(
            """UPDATE web_search_providers SET provider_kind = ?,
            display_name = ?, credential_id = ?, enabled = ?,
            read_enabled = ?, config_json = ?, version = version + 1,
            updated_at = ? WHERE provider_id = ?""",
            (
                context.provider_kind,
                str(provider.display_name or context.provider_kind),
                str(provider.credential_id or ""),
                int(provider.enabled),
                int(provider.read_enabled),
                _dump(provider.config),
                context.now,
                context.provider_id,
            ),
        )
