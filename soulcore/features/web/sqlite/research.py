from __future__ import annotations

from .support import (
    Any,
    Mapping,
    WebImageSearchResultRecord,
    WebPageSnapshotRecord,
    WebSearchResultRecord,
    WebSearchSessionRecord,
    WebSearchSessionStatus,
    _dt,
    _dump,
    _load,
    _now,
    datetime,
    sqlite3,
    timedelta,
)


class WebResearchRecords:
    async def finish_web_search_session(
        self,
        profile_id: str,
        instance_id: str,
        session_id: str,
        status: WebSearchSessionStatus | str,
        *,
        partial_warning: str = "",
        provider_count: int = 0,
        result_count: int | None = None,
        diagnostics: dict[str, Any] | None = None,
        error: str = "",
    ) -> WebSearchSessionRecord:
        normalized = WebSearchSessionStatus(status)
        if normalized is WebSearchSessionStatus.RUNNING:
            raise ValueError("finish status cannot be RUNNING")
        now = _dt(_now())
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE web_search_sessions SET status = ?, partial_warning = ?,
                provider_count = ?, result_count = COALESCE(?, result_count),
                diagnostics_json = ?, error = ?, finished_at = ?
                WHERE profile_id = ? AND instance_id = ? AND session_id = ?
                    AND status = 'RUNNING'""",
                (
                    normalized.value,
                    partial_warning,
                    max(0, int(provider_count)),
                    None if result_count is None else max(0, int(result_count)),
                    _dump(diagnostics or {}),
                    error,
                    now,
                    profile_id,
                    instance_id,
                    session_id,
                ),
            ),
            transaction=True,
        )
        if cursor.rowcount != 1:
            raise ValueError("web search session is not running")
        result = await self.get_web_search_session(
            profile_id, instance_id, session_id, include_expired=True
        )
        assert result is not None
        return result

    async def save_web_search_results(
        self,
        profile_id: str,
        instance_id: str,
        session_id: str,
        results: list[WebSearchResultRecord],
    ) -> list[WebSearchResultRecord]:
        now = _now()

        def operation(conn: sqlite3.Connection) -> None:
            session = conn.execute(
                """SELECT 1 FROM web_search_sessions
                WHERE profile_id = ? AND instance_id = ? AND session_id = ?""",
                (profile_id, instance_id, session_id),
            ).fetchone()
            if session is None:
                raise KeyError(session_id)
            for item in results:
                if (
                    item.profile_id != profile_id
                    or item.instance_id != instance_id
                    or item.session_id != session_id
                ):
                    raise ValueError("web search result scope mismatch")
                conn.execute(
                    """INSERT INTO web_search_results(
                        resource_id, session_id, profile_id, instance_id, title,
                        canonical_url, domain, snippet, published_at, retrieved_at,
                        provider_id, provider_rank, cross_source_count, read_status,
                        metadata_json, expires_at, redacted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(resource_id) DO UPDATE SET
                        title = excluded.title, canonical_url = excluded.canonical_url,
                        domain = excluded.domain, snippet = excluded.snippet,
                        published_at = excluded.published_at,
                        provider_rank = excluded.provider_rank,
                        cross_source_count = excluded.cross_source_count,
                        read_status = excluded.read_status,
                        metadata_json = excluded.metadata_json""",
                    (
                        item.resource_id,
                        session_id,
                        profile_id,
                        instance_id,
                        item.title,
                        item.canonical_url,
                        item.domain,
                        item.snippet,
                        _dt(item.published_at),
                        _dt(item.retrieved_at or now),
                        item.provider_id,
                        max(0, int(item.provider_rank)),
                        max(1, int(item.cross_source_count)),
                        item.read_status.value,
                        _dump(item.metadata),
                        _dt(item.expires_at or now + timedelta(hours=24)),
                        _dt(item.redacted_at),
                    ),
                )
            conn.execute(
                """UPDATE web_search_sessions SET result_count = (
                    SELECT COUNT(*) FROM web_search_results WHERE session_id = ?
                ) WHERE session_id = ?""",
                (session_id, session_id),
            )

        await self.uow.run(operation)
        return await self.list_web_search_results(profile_id, instance_id, session_id)

    async def list_web_search_results(
        self,
        profile_id: str,
        instance_id: str,
        session_id: str,
        *,
        include_expired: bool = False,
    ) -> list[WebSearchResultRecord]:
        sql = """SELECT * FROM web_search_results WHERE profile_id = ?
            AND instance_id = ? AND session_id = ?"""
        parameters: list[Any] = [profile_id, instance_id, session_id]
        if not include_expired:
            sql += " AND expires_at > ? AND redacted_at IS NULL"
            parameters.append(_dt(_now()))
        sql += " ORDER BY provider_rank, resource_id"
        return [self._web_result(row) for row in await self.db.fetch_all(sql, parameters)]

    async def get_web_search_result(
        self,
        resource_id: str,
        profile_id: str,
        instance_id: str,
        run_scope: str,
    ) -> Mapping[str, Any] | None:
        row = await self.db.fetch_one(
            """SELECT result.*, session.core_run_id, session.ai_task_id
            FROM web_search_results result
            JOIN web_search_sessions session USING(session_id)
            WHERE result.resource_id = ? AND result.profile_id = ?
              AND result.instance_id = ? AND result.expires_at > ?
              AND result.redacted_at IS NULL""",
            (resource_id, profile_id, instance_id, _dt(_now())),
        )
        if row is None:
            return None
        stored_scope = str(row["core_run_id"] or row["ai_task_id"] or "")
        if stored_scope != str(run_scope):
            return None
        metadata = _load(row["metadata_json"]) or {}
        return {
            "resource_id": str(row["resource_id"]),
            "session_id": str(row["session_id"]),
            "profile_id": str(row["profile_id"]),
            "instance_id": str(row["instance_id"]),
            "run_scope": stored_scope,
            "title": str(row["title"]),
            "canonical_url": str(row["canonical_url"]),
            "domain": str(row["domain"]),
            "snippet": str(row["snippet"]),
            "provider": str(row["provider_id"]),
            "read_status": str(row["read_status"]),
            "source_providers": list(metadata.get("source_providers") or ()),
            "score": float(metadata.get("score") or 0.0),
        }

    async def save_web_image_results(
        self,
        profile_id: str,
        instance_id: str,
        session_id: str,
        results: list[WebImageSearchResultRecord],
    ) -> list[WebImageSearchResultRecord]:
        now = _now()

        def operation(conn: sqlite3.Connection) -> None:
            session = conn.execute(
                """SELECT search_kind FROM web_search_sessions
                WHERE profile_id = ? AND instance_id = ? AND session_id = ?""",
                (profile_id, instance_id, session_id),
            ).fetchone()
            if session is None:
                raise KeyError(session_id)
            if str(session["search_kind"]) != "IMAGE":
                raise ValueError("image results require an IMAGE search session")
            for item in results:
                if (item.profile_id, item.instance_id, item.session_id) != (
                    profile_id,
                    instance_id,
                    session_id,
                ):
                    raise ValueError("web image result scope mismatch")
                if not item.image_resource_id:
                    raise ValueError("image_resource_id cannot be empty")
                conn.execute(
                    """INSERT INTO web_image_search_results(
                        image_resource_id, session_id, profile_id, instance_id,
                        original_url, thumbnail_url, source_page_url, source_domain,
                        title, description, provider_id, provider_rank,
                        cross_source_count, width, height, mime_type, metadata_json,
                        retrieved_at, expires_at, redacted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(image_resource_id) DO UPDATE SET
                        original_url = excluded.original_url,
                        thumbnail_url = excluded.thumbnail_url,
                        source_page_url = excluded.source_page_url,
                        source_domain = excluded.source_domain,
                        title = excluded.title, description = excluded.description,
                        provider_id = excluded.provider_id,
                        provider_rank = excluded.provider_rank,
                        cross_source_count = excluded.cross_source_count,
                        width = excluded.width, height = excluded.height,
                        mime_type = excluded.mime_type,
                        metadata_json = excluded.metadata_json,
                        retrieved_at = excluded.retrieved_at,
                        expires_at = excluded.expires_at,
                        redacted_at = excluded.redacted_at""",
                    (
                        item.image_resource_id,
                        session_id,
                        profile_id,
                        instance_id,
                        item.original_url,
                        item.thumbnail_url,
                        item.source_page_url,
                        item.source_domain,
                        item.title,
                        item.description,
                        item.provider_id,
                        max(0, int(item.provider_rank)),
                        max(1, int(item.cross_source_count)),
                        item.width,
                        item.height,
                        item.mime_type,
                        _dump(item.metadata),
                        _dt(item.retrieved_at or now),
                        _dt(item.expires_at or now + timedelta(hours=24)),
                        _dt(item.redacted_at),
                    ),
                )
            conn.execute(
                """UPDATE web_search_sessions SET result_count = (
                    SELECT COUNT(*) FROM web_image_search_results WHERE session_id = ?
                ) WHERE session_id = ?""",
                (session_id, session_id),
            )

        await self.uow.run(operation)
        return await self.list_web_image_results(profile_id, instance_id, session_id)

    async def list_web_image_results(
        self,
        profile_id: str,
        instance_id: str,
        session_id: str,
        *,
        include_expired: bool = False,
    ) -> list[WebImageSearchResultRecord]:
        sql = """SELECT * FROM web_image_search_results WHERE profile_id = ?
            AND instance_id = ? AND session_id = ?"""
        parameters: list[Any] = [profile_id, instance_id, session_id]
        if not include_expired:
            sql += " AND expires_at > ? AND redacted_at IS NULL"
            parameters.append(_dt(_now()))
        sql += " ORDER BY provider_rank, image_resource_id"
        return [self._web_image_result(row) for row in await self.db.fetch_all(sql, parameters)]

    async def get_web_image_result(
        self,
        image_resource_id: str,
        profile_id: str,
        instance_id: str,
        run_scope: str,
    ) -> Mapping[str, Any] | None:
        row = await self.db.fetch_one(
            """SELECT result.*, session.core_run_id, session.ai_task_id
            FROM web_image_search_results result
            JOIN web_search_sessions session USING(session_id)
            WHERE result.image_resource_id = ? AND result.profile_id = ?
              AND result.instance_id = ? AND result.expires_at > ?
              AND result.redacted_at IS NULL AND session.search_kind = 'IMAGE'""",
            (image_resource_id, profile_id, instance_id, _dt(_now())),
        )
        if row is None:
            return None
        stored_scope = str(row["core_run_id"] or row["ai_task_id"] or "")
        if stored_scope != str(run_scope):
            return None
        metadata = _load(row["metadata_json"]) or {}
        return {
            "image_resource_id": str(row["image_resource_id"]),
            "session_id": str(row["session_id"]),
            "profile_id": str(row["profile_id"]),
            "instance_id": str(row["instance_id"]),
            "run_scope": stored_scope,
            "title": str(row["title"]),
            "description": str(row["description"]),
            "original_url": str(row["original_url"]),
            "thumbnail_url": str(row["thumbnail_url"]),
            "source_page_url": str(row["source_page_url"]),
            "source_domain": str(row["source_domain"]),
            "width": row["width"],
            "height": row["height"],
            "mime_type": str(row["mime_type"]),
            "provider_id": str(row["provider_id"]),
            "provider_rank": int(row["provider_rank"]),
            "cross_source_count": int(row["cross_source_count"]),
            "source_providers": list(metadata.get("source_providers") or ()),
            "score": float(metadata.get("score") or 0.0),
        }

    async def upsert_web_page_snapshot(
        self, snapshot: WebPageSnapshotRecord
    ) -> WebPageSnapshotRecord:
        now = snapshot.retrieved_at or _now()
        expires_at = snapshot.expires_at or (now + timedelta(hours=6))

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            resource = conn.execute(
                """SELECT 1 FROM web_search_results WHERE resource_id = ?
                AND profile_id = ? AND instance_id = ?""",
                (snapshot.resource_id, snapshot.profile_id, snapshot.instance_id),
            ).fetchone()
            if resource is None:
                raise KeyError(snapshot.resource_id)
            conn.execute(
                """INSERT INTO web_page_snapshots(
                    resource_id, profile_id, instance_id, content, content_hash,
                    token_estimate, status, error, metadata_json, retrieved_at,
                    expires_at, redacted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(resource_id) DO UPDATE SET content = excluded.content,
                    content_hash = excluded.content_hash,
                    token_estimate = excluded.token_estimate,
                    status = excluded.status, error = excluded.error,
                    metadata_json = excluded.metadata_json,
                    retrieved_at = excluded.retrieved_at,
                    expires_at = excluded.expires_at, redacted_at = excluded.redacted_at""",
                (
                    snapshot.resource_id,
                    snapshot.profile_id,
                    snapshot.instance_id,
                    snapshot.content,
                    snapshot.content_hash,
                    max(0, int(snapshot.token_estimate)),
                    snapshot.status.value,
                    snapshot.error,
                    _dump(snapshot.metadata),
                    _dt(now),
                    _dt(expires_at),
                    _dt(snapshot.redacted_at),
                ),
            )
            conn.execute(
                "UPDATE web_search_results SET read_status = ? WHERE resource_id = ?",
                (snapshot.status.value, snapshot.resource_id),
            )
            return conn.execute(
                "SELECT * FROM web_page_snapshots WHERE resource_id = ?",
                (snapshot.resource_id,),
            ).fetchone()

        return self._web_page(await self.uow.run(operation))

    async def get_web_page_snapshot(
        self,
        profile_id: str,
        instance_id: str,
        resource_id: str,
        *,
        include_expired: bool = False,
    ) -> WebPageSnapshotRecord | None:
        sql = """SELECT * FROM web_page_snapshots WHERE profile_id = ?
            AND instance_id = ? AND resource_id = ?"""
        parameters: list[Any] = [profile_id, instance_id, resource_id]
        if not include_expired:
            sql += " AND expires_at > ? AND redacted_at IS NULL"
            parameters.append(_dt(_now()))
        row = await self.db.fetch_one(sql, parameters)
        return self._web_page(row) if row else None

    async def cleanup_expired_web_research(self, *, now: datetime | None = None) -> dict[str, int]:
        """Redact private payloads while retaining non-sensitive audit rows."""

        current = _dt(now or _now())

        def operation(conn: sqlite3.Connection) -> dict[str, int]:
            pages = conn.execute(
                """UPDATE web_page_snapshots SET content = '', content_hash = '',
                token_estimate = 0, status = 'EXPIRED', redacted_at = ?
                WHERE expires_at <= ? AND redacted_at IS NULL""",
                (current, current),
            ).rowcount
            conn.execute(
                """UPDATE web_search_results SET read_status = 'EXPIRED'
                WHERE resource_id IN (
                    SELECT resource_id FROM web_page_snapshots
                    WHERE redacted_at = ?
                )""",
                (current,),
            )
            results = conn.execute(
                """UPDATE web_search_results SET title = '', canonical_url = '',
                snippet = '', metadata_json = '{}', redacted_at = ?
                WHERE expires_at <= ? AND redacted_at IS NULL""",
                (current, current),
            ).rowcount
            image_results = conn.execute(
                """UPDATE web_image_search_results SET original_url = '',
                thumbnail_url = '', source_page_url = '', title = '',
                description = '', metadata_json = '{}', redacted_at = ?
                WHERE expires_at <= ? AND redacted_at IS NULL""",
                (current, current),
            ).rowcount
            sessions = conn.execute(
                """UPDATE web_search_sessions SET query = '', status = 'EXPIRED',
                diagnostics_json = '{}', redacted_at = ?
                WHERE expires_at <= ? AND redacted_at IS NULL""",
                (current, current),
            ).rowcount
            return {
                "web_page_snapshots": int(pages),
                "web_search_results": int(results),
                "web_image_search_results": int(image_results),
                "web_search_sessions": int(sessions),
            }

        result = await self.uow.run(operation)
        if any(result.values()):
            await self.db.publish_backup_after_commit()
        return result
