from __future__ import annotations

from .provider_upsert import (
    WebProviderUpsertContext,
    WebProviderUpsertTransaction,
)
from .support import (
    Any,
    Mapping,
    RoleProfile,
    WebCallerKind,
    WebSearchIntensity,
    WebSearchKind,
    WebSearchProviderRecord,
    WebSearchPurpose,
    WebSearchSessionRecord,
    WebSearchSessionStatus,
    _coerce_datetime,
    _dt,
    _dump,
    _now,
    sqlite3,
    timedelta,
)


class WebConfigurationRecords:
    async def update_web_search_settings(
        self,
        profile_id: str,
        *,
        enabled: bool | None = None,
        intensity: WebSearchIntensity | str | None = None,
    ) -> RoleProfile:
        changes: dict[str, Any] = {}
        if enabled is not None:
            changes["web_search_enabled"] = bool(enabled)
        if intensity is not None:
            changes["web_search_intensity"] = WebSearchIntensity(str(intensity).upper()).value
        profile = await self._profiles.update_profile(profile_id, **changes)
        if changes:
            await self.db.publish_backup_after_commit()
        return profile

    async def upsert_web_search_provider(
        self,
        provider: WebSearchProviderRecord,
        *,
        expected_version: int | None = None,
    ) -> WebSearchProviderRecord:
        provider_id = self._validate_ai_identifier(provider.provider_id, "provider_id")
        profile_id = self._validate_ai_identifier(provider.profile_id, "profile_id")
        provider_kind = str(provider.provider_kind or "").strip().upper()
        allowed = {"TAVILY", "BOCHA", "BRAVE", "FIRECRAWL", "BAIDU_AI", "EXA"}
        if provider_kind not in allowed:
            raise ValueError("unsupported web search provider kind")
        can_read = provider_kind in {"TAVILY", "FIRECRAWL", "EXA"}
        if provider.read_enabled and not can_read:
            raise ValueError(f"{provider_kind} does not support page reading")
        backend_id = self._validate_ai_identifier(
            provider.backend_id or f"web:{profile_id}:{provider_id}",
            "backend_id",
        )
        context = WebProviderUpsertContext(
            provider=provider,
            provider_id=provider_id,
            profile_id=profile_id,
            provider_kind=provider_kind,
            backend_id=backend_id,
            expected_version=expected_version,
            now=_dt(_now()),
        )
        row = await self.uow.run(WebProviderUpsertTransaction(self, context))
        await self.db.publish_backup_after_commit()
        return self._web_provider(row)

    async def get_web_search_provider(
        self,
        profile_id: str,
        provider_id: str,
        *,
        include_archived: bool = False,
    ) -> WebSearchProviderRecord | None:
        sql = """SELECT * FROM web_search_providers
            WHERE profile_id = ? AND provider_id = ?"""
        if not include_archived:
            sql += " AND archived_at IS NULL"
        row = await self.db.fetch_one(sql, (profile_id, provider_id))
        return self._web_provider(row) if row else None

    async def list_web_search_providers(
        self, profile_id: str, *, include_archived: bool = False
    ) -> list[WebSearchProviderRecord]:
        sql = "SELECT * FROM web_search_providers WHERE profile_id = ?"
        if not include_archived:
            sql += " AND archived_at IS NULL"
        sql += " ORDER BY priority, provider_id"
        return [self._web_provider(row) for row in await self.db.fetch_all(sql, (profile_id,))]

    async def set_web_search_provider_enabled(
        self,
        profile_id: str,
        provider_id: str,
        enabled: bool,
        *,
        expected_version: int | None = None,
    ) -> WebSearchProviderRecord:
        current = await self.get_web_search_provider(profile_id, provider_id)
        if current is None:
            raise KeyError(provider_id)
        current.enabled = bool(enabled)
        return await self.upsert_web_search_provider(current, expected_version=expected_version)

    async def archive_web_search_provider(
        self,
        profile_id: str,
        provider_id: str,
        *,
        expected_version: int | None = None,
    ) -> WebSearchProviderRecord:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            row = conn.execute(
                """SELECT * FROM web_search_providers
                WHERE profile_id = ? AND provider_id = ?""",
                (profile_id, provider_id),
            ).fetchone()
            if row is None:
                raise KeyError(provider_id)
            self._require_expected_version(row, expected_version, "web search provider")
            conn.execute(
                """UPDATE web_search_providers SET enabled = 0, archived_at = ?,
                version = version + 1, updated_at = ? WHERE provider_id = ?""",
                (now, now, provider_id),
            )
            self._sync_web_provider_runtime(conn, provider_id, now)
            self._normalize_web_provider_priorities(conn, profile_id, now)
            return conn.execute(
                "SELECT * FROM web_search_providers WHERE provider_id = ?",
                (provider_id,),
            ).fetchone()

        result = self._web_provider(await self.uow.run(operation))
        await self.db.publish_backup_after_commit()
        return result

    async def create_web_search_session(
        self, session: WebSearchSessionRecord | Mapping[str, Any]
    ) -> WebSearchSessionRecord:
        if isinstance(session, Mapping):
            session = self._web_session_input(session)
        allowed_callers = {
            WebCallerKind.MAIN_CORE,
            WebCallerKind.BACKGROUND_AUTHOR,
            WebCallerKind.STICKER_COLLECTOR,
        }
        if session.caller_kind not in allowed_callers:
            raise ValueError("unsupported web research caller")
        stored_caller = session.caller_kind.value
        if not str(session.query or "").strip():
            raise ValueError("web search query cannot be empty")
        now = session.started_at or _now()
        expires_at = session.expires_at or (now + timedelta(hours=24))
        deadline_at = session.deadline_at or (now + timedelta(seconds=300))
        await self.db.call(
            lambda conn: conn.execute(
                """INSERT INTO web_search_sessions(
                    session_id, profile_id, instance_id, caller_kind,
                    effective_caller_kind, caller_id,
                    core_run_id, ai_task_id, purpose, query, depth, freshness,
                    status, deadline_at, partial_warning, provider_count,
                    result_count, diagnostics_json, error, started_at,
                    finished_at, expires_at, redacted_at, search_kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session.session_id,
                    session.profile_id,
                    session.instance_id,
                    stored_caller,
                    session.caller_kind.value,
                    session.caller_id,
                    session.core_run_id,
                    session.ai_task_id,
                    session.purpose.value,
                    str(session.query).strip(),
                    session.depth,
                    session.freshness,
                    session.status.value,
                    _dt(deadline_at),
                    session.partial_warning,
                    session.provider_count,
                    session.result_count,
                    _dump(session.diagnostics),
                    session.error,
                    _dt(now),
                    _dt(session.finished_at),
                    _dt(expires_at),
                    _dt(session.redacted_at),
                    session.search_kind.value,
                ),
            ),
            transaction=True,
        )
        result = await self.get_web_search_session(
            session.profile_id,
            session.instance_id,
            session.session_id,
            include_expired=True,
        )
        assert result is not None
        return result

    @staticmethod
    def _web_session_input(session: Mapping[str, Any]) -> WebSearchSessionRecord:
        return WebSearchSessionRecord(
            session_id=str(session.get("session_id") or ""),
            profile_id=str(session.get("profile_id") or ""),
            instance_id=str(session.get("instance_id") or ""),
            caller_kind=WebCallerKind(str(session.get("caller_kind") or "MAIN_CORE")),
            caller_id=str(session.get("caller_id") or ""),
            core_run_id=session.get("core_run_id"),
            ai_task_id=str(session.get("ai_task_id") or "") or None,
            purpose=WebSearchPurpose(str(session.get("purpose") or "ANSWER_USER")),
            query=str(session.get("query") or ""),
            search_kind=WebSearchKind(str(session.get("search_kind") or "WEB").upper()),
            depth=str(session.get("depth") or "auto"),
            freshness=str(session.get("freshness") or "auto"),
            status=WebSearchSessionStatus(str(session.get("status") or "RUNNING")),
            deadline_at=_coerce_datetime(session.get("deadline_at")),
        )

    async def complete_web_search_session(
        self, session_id: str, record: Mapping[str, Any]
    ) -> WebSearchSessionRecord:
        row = await self.db.fetch_one(
            "SELECT profile_id, instance_id FROM web_search_sessions WHERE session_id = ?",
            (session_id,),
        )
        if row is None:
            raise KeyError(session_id)
        status = str(record.get("status") or "FAILED").upper()
        normalized = {
            "SUCCEEDED": WebSearchSessionStatus.COMPLETED,
            "COMPLETED": WebSearchSessionStatus.COMPLETED,
            "PARTIAL": WebSearchSessionStatus.PARTIAL,
            "FAILED": WebSearchSessionStatus.FAILED,
            "CANCELLED": WebSearchSessionStatus.CANCELLED,
        }.get(status)
        if normalized is None:
            raise ValueError("unsupported web search completion status")
        provider_errors = dict(record.get("provider_errors") or {})
        return await self.finish_web_search_session(
            str(row["profile_id"]),
            str(row["instance_id"]),
            session_id,
            normalized,
            partial_warning=("Some web-search interfaces failed" if provider_errors else ""),
            provider_count=max(0, int(record.get("provider_count") or 0)),
            result_count=(
                int(record["result_count"]) if record.get("result_count") is not None else None
            ),
            diagnostics={
                "provider_errors": provider_errors,
                "elapsed_seconds": float(record.get("elapsed_seconds") or 0.0),
            },
            error=str(record.get("error") or ""),
        )

    async def get_web_search_session(
        self,
        profile_id: str,
        instance_id: str,
        session_id: str,
        *,
        include_expired: bool = False,
    ) -> WebSearchSessionRecord | None:
        sql = """SELECT * FROM web_search_sessions
            WHERE profile_id = ? AND instance_id = ? AND session_id = ?"""
        parameters: list[Any] = [profile_id, instance_id, session_id]
        if not include_expired:
            sql += " AND expires_at > ? AND redacted_at IS NULL"
            parameters.append(_dt(_now()))
        row = await self.db.fetch_one(sql, parameters)
        return self._web_session(row) if row else None

    async def list_web_search_sessions(
        self,
        profile_id: str,
        *,
        instance_id: str | None = None,
        status: WebSearchSessionStatus | str | None = None,
        search_kind: WebSearchKind | str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[WebSearchSessionRecord]:
        clauses = ["profile_id = ?"]
        parameters: list[Any] = [profile_id]
        if instance_id is not None:
            clauses.append("instance_id = ?")
            parameters.append(instance_id)
        if status is not None:
            clauses.append("status = ?")
            parameters.append(WebSearchSessionStatus(status).value)
        if search_kind is not None:
            clauses.append("search_kind = ?")
            parameters.append(WebSearchKind(str(search_kind).upper()).value)
        parameters.extend((max(1, min(int(limit), 100)), max(0, int(offset))))
        rows = await self.db.fetch_all(
            f"""SELECT * FROM web_search_sessions
            WHERE {" AND ".join(clauses)} ORDER BY started_at DESC
            LIMIT ? OFFSET ?""",
            parameters,
        )
        return [self._web_session(row) for row in rows]

    async def count_web_search_sessions(self, profile_id: str) -> int:
        row = await self.db.fetch_one(
            "SELECT COUNT(*) AS total FROM web_search_sessions WHERE profile_id = ?",
            (profile_id,),
        )
        return int(row["total"] if row else 0)
