from __future__ import annotations

from ..model_parameters import (
    DEFAULT_MODEL_MAX_CONTEXT_TOKENS,
    MINIMUM_MODEL_MAX_CONTEXT_TOKENS,
    TEXT_GENERATION_CAPABILITIES,
)
from .support import (
    Any,
    _dt,
    _dump,
    _load,
    _now,
    sqlite3,
)


class AiConfigurationRecords:
    async def upsert_ai_backend(
        self,
        backend_id: str,
        backend_kind: str,
        *,
        display_name: str = "",
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        backend_id = str(backend_id or "").strip()
        if not backend_id:
            raise ValueError("backend_id cannot be empty")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            row = conn.execute(
                "SELECT * FROM ai_backends WHERE backend_id = ?", (backend_id,)
            ).fetchone()
            if row is None:
                conn.execute(
                    """INSERT INTO ai_backends(
                        backend_id, backend_kind, display_name, enabled,
                        metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        backend_id,
                        str(backend_kind).upper(),
                        display_name,
                        int(enabled),
                        _dump(metadata or {}),
                        now,
                        now,
                    ),
                )
            else:
                if expected_version is not None and int(row["version"]) != int(expected_version):
                    raise ValueError("ai backend version conflict")
                conn.execute(
                    """UPDATE ai_backends SET backend_kind = ?, display_name = ?,
                    enabled = ?, metadata_json = ?, version = version + 1,
                    updated_at = ? WHERE backend_id = ?""",
                    (
                        str(backend_kind).upper(),
                        display_name,
                        int(enabled),
                        _dump(
                            metadata
                            if metadata is not None
                            else (_load(row["metadata_json"]) or {})
                        ),
                        now,
                        backend_id,
                    ),
                )
            result = conn.execute(
                "SELECT * FROM ai_backends WHERE backend_id = ?", (backend_id,)
            ).fetchone()
            assert result is not None
            return result

        return self._ai_backend(await self.uow.run(operation))

    @staticmethod
    def _insert_ai_api_package(
        conn: sqlite3.Connection,
        values: tuple[Any, ...],
    ) -> None:
        conn.execute(
            """INSERT INTO ai_api_packages(
                package_id, profile_id, protocol, display_name, base_url,
                credential_id, enabled, config_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            values,
        )

    def _update_ai_api_package(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        package_id: str,
        profile_id: str,
        protocol: str,
        display_name: str,
        base_url: str,
        credential_id: str,
        enabled: bool,
        config: dict[str, Any] | None,
        expected_version: int | None,
        now: str,
    ) -> None:
        if str(row["profile_id"]) != profile_id:
            raise ValueError("API package belongs to another AstrBot profile")
        if row["archived_at"] is not None:
            raise ValueError("archived API package cannot be modified")
        self._require_expected_version(row, expected_version, "API package")
        config_value = config if config is not None else (_load(row["config_json"]) or {})
        conn.execute(
            """UPDATE ai_api_packages SET protocol = ?, display_name = ?,
            base_url = ?, credential_id = ?, enabled = ?,
            config_json = ?, version = version + 1,
            updated_at = ? WHERE package_id = ?""",
            (
                protocol,
                display_name,
                base_url,
                credential_id,
                int(enabled),
                _dump(config_value),
                now,
                package_id,
            ),
        )
        for model in conn.execute(
            "SELECT backend_id FROM ai_api_models WHERE package_id = ?", (package_id,)
        ):
            self._sync_ai_api_model_runtime(conn, model["backend_id"], now)

    def _upsert_ai_api_package_sql(
        self,
        conn: sqlite3.Connection,
        *,
        package_id: str,
        profile_id: str,
        protocol: str,
        display_name: str,
        base_url: str,
        credential_id: str,
        enabled: bool,
        config: dict[str, Any] | None,
        expected_version: int | None,
        now: str,
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM ai_api_packages WHERE package_id = ?", (package_id,)
        ).fetchone()
        normalized = (
            str(display_name or "").strip(),
            str(base_url or "").strip(),
            str(credential_id or "").strip(),
        )
        if row is None:
            self._insert_ai_api_package(
                conn,
                (
                    package_id,
                    profile_id,
                    protocol,
                    *normalized,
                    int(enabled),
                    _dump(config or {}),
                    now,
                    now,
                ),
            )
        else:
            self._update_ai_api_package(
                conn,
                row,
                package_id=package_id,
                profile_id=profile_id,
                protocol=protocol,
                display_name=normalized[0],
                base_url=normalized[1],
                credential_id=normalized[2],
                enabled=enabled,
                config=config,
                expected_version=expected_version,
                now=now,
            )
        result = conn.execute(
            "SELECT * FROM ai_api_packages WHERE package_id = ?", (package_id,)
        ).fetchone()
        assert result is not None
        return result

    async def upsert_ai_api_package(
        self,
        package_id: str,
        *,
        profile_id: str = "default",
        protocol: str = "OPENAI_COMPATIBLE",
        display_name: str,
        base_url: str = "",
        credential_id: str = "",
        enabled: bool = True,
        config: dict[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        package_id = self._validate_ai_identifier(package_id, "package_id")
        profile_id = self._validate_ai_identifier(profile_id, "profile_id")
        normalized_protocol = str(protocol or "OPENAI_COMPATIBLE").strip().upper()
        if normalized_protocol not in {
            "OPENAI",
            "OPENAI_COMPATIBLE",
            "ANTHROPIC",
            "GEMINI",
            "CUSTOM_HTTP_IMAGE",
            "MINIMAX_TTS",
            "MIMO_TTS",
            "GPT_SOVITS_V2",
            "GSVI_TTS",
        }:
            raise ValueError("unsupported API package protocol")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            return self._upsert_ai_api_package_sql(
                conn,
                package_id=package_id,
                profile_id=profile_id,
                protocol=normalized_protocol,
                display_name=display_name,
                base_url=base_url,
                credential_id=credential_id,
                enabled=enabled,
                config=config,
                expected_version=expected_version,
                now=now,
            )

        return self._ai_api_package(await self.uow.run(operation))

    async def get_ai_api_package(
        self, package_id: str, *, profile_id: str | None = None, include_archived: bool = False
    ) -> dict[str, Any] | None:
        clauses = ["package_id = ?"]
        parameters: list[Any] = [str(package_id)]
        if profile_id is not None:
            clauses.append("profile_id = ?")
            parameters.append(str(profile_id))
        if not include_archived:
            clauses.append("archived_at IS NULL")
        row = await self.db.fetch_one(
            f"SELECT * FROM ai_api_packages WHERE {' AND '.join(clauses)}",
            parameters,
        )
        return self._ai_api_package(row) if row else None

    async def list_ai_api_packages(
        self, profile_id: str | None = None, *, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if profile_id is not None:
            clauses.append("profile_id = ?")
            parameters.append(str(profile_id))
        if not include_archived:
            clauses.append("archived_at IS NULL")
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        rows = await self.db.fetch_all(
            f"""SELECT * FROM ai_api_packages {where}
            ORDER BY display_name COLLATE NOCASE, package_id""",
            parameters,
        )
        return [self._ai_api_package(row) for row in rows]

    async def set_ai_api_package_enabled(
        self, package_id: str, enabled: bool, *, expected_version: int | None = None
    ) -> dict[str, Any]:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            row = conn.execute(
                "SELECT * FROM ai_api_packages WHERE package_id = ?", (package_id,)
            ).fetchone()
            if row is None or row["archived_at"] is not None:
                raise KeyError(package_id)
            self._require_expected_version(row, expected_version, "API package")
            conn.execute(
                """UPDATE ai_api_packages SET enabled = ?, version = version + 1,
                updated_at = ? WHERE package_id = ?""",
                (int(enabled), now, package_id),
            )
            for model in conn.execute(
                "SELECT backend_id FROM ai_api_models WHERE package_id = ?",
                (package_id,),
            ):
                self._sync_ai_api_model_runtime(conn, model["backend_id"], now)
            return conn.execute(
                "SELECT * FROM ai_api_packages WHERE package_id = ?", (package_id,)
            ).fetchone()

        return self._ai_api_package(await self.uow.run(operation))

    async def archive_ai_api_package(
        self, package_id: str, *, expected_version: int | None = None
    ) -> dict[str, Any]:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            row = conn.execute(
                "SELECT * FROM ai_api_packages WHERE package_id = ?", (package_id,)
            ).fetchone()
            if row is None:
                raise KeyError(package_id)
            self._require_expected_version(row, expected_version, "API package")
            conn.execute(
                """UPDATE ai_api_packages SET enabled = 0, archived_at = ?,
                version = version + 1, updated_at = ? WHERE package_id = ?""",
                (now, now, package_id),
            )
            conn.execute(
                """UPDATE ai_api_models SET enabled = 0,
                archived_at = COALESCE(archived_at, ?), version = version + 1,
                updated_at = ? WHERE package_id = ?""",
                (now, now, package_id),
            )
            for model in conn.execute(
                "SELECT backend_id FROM ai_api_models WHERE package_id = ?",
                (package_id,),
            ):
                self._sync_ai_api_model_runtime(conn, model["backend_id"], now)
            self._normalize_ai_api_priorities(conn, now)
            return conn.execute(
                "SELECT * FROM ai_api_packages WHERE package_id = ?", (package_id,)
            ).fetchone()

        return self._ai_api_package(await self.uow.run(operation))

    @staticmethod
    def _insert_ai_api_model(
        conn: sqlite3.Connection,
        package: sqlite3.Row,
        *,
        package_id: str,
        backend_id: str,
        model_key: str,
        display_name: str,
        capabilities: tuple[str, ...],
        priority: int,
        enabled: bool,
        config: dict[str, Any] | None,
        now: str,
    ) -> None:
        conflict = conn.execute(
            "SELECT 1 FROM ai_backends WHERE backend_id = ?", (backend_id,)
        ).fetchone()
        if conflict is not None:
            raise ValueError("backend_id is already owned by another runtime record")
        shown_name = str(display_name or model_key).strip()
        conn.execute(
            """INSERT INTO ai_backends(
                backend_id, backend_kind, display_name, enabled,
                metadata_json, created_at, updated_at
            ) VALUES (?, 'MODEL', ?, 0, '{}', ?, ?)""",
            (backend_id, shown_name, now, now),
        )
        conn.execute(
            """INSERT INTO ai_api_models(
                backend_id, package_id, model_key, display_name,
                capabilities_json, priority, enabled,
                config_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                backend_id,
                package_id,
                model_key,
                shown_name,
                _dump(capabilities),
                max(1, int(priority)),
                int(enabled),
                _dump(config or {}),
                now,
                now,
            ),
        )

    @staticmethod
    def _validate_ai_model_package_move(
        conn: sqlite3.Connection,
        existing: sqlite3.Row,
        package: sqlite3.Row,
        *,
        package_id: str,
        backend_id: str,
        now: str,
    ) -> None:
        if existing["package_id"] == package_id:
            return
        raise ValueError("backend_id belongs to another API package")

    def _update_ai_api_model(
        self,
        conn: sqlite3.Connection,
        existing: sqlite3.Row,
        package: sqlite3.Row,
        *,
        package_id: str,
        backend_id: str,
        model_key: str,
        display_name: str,
        capabilities: tuple[str, ...],
        enabled: bool,
        config: dict[str, Any] | None,
        expected_version: int | None,
        now: str,
    ) -> None:
        self._validate_ai_model_package_move(
            conn,
            existing,
            package,
            package_id=package_id,
            backend_id=backend_id,
            now=now,
        )
        if existing["archived_at"] is not None:
            raise ValueError("archived API model cannot be modified")
        self._require_expected_version(existing, expected_version, "API model")
        values = (
            model_key,
            str(display_name or model_key).strip(),
            config if config is not None else (_load(existing["config_json"]) or {}),
        )
        conn.execute(
            """UPDATE ai_api_models SET model_key = ?, display_name = ?,
            capabilities_json = ?, enabled = ?,
            config_json = ?, version = version + 1, updated_at = ?
            WHERE backend_id = ?""",
            (*values[:2], _dump(capabilities), int(enabled), _dump(values[2]), now, backend_id),
        )

    def _upsert_ai_api_model_sql(
        self,
        conn: sqlite3.Connection,
        *,
        package_id: str,
        backend_id: str,
        model_key: str,
        display_name: str,
        capabilities: tuple[str, ...],
        priority: int,
        enabled: bool,
        config: dict[str, Any] | None,
        expected_version: int | None,
        now: str,
    ) -> sqlite3.Row:
        package = conn.execute(
            "SELECT * FROM ai_api_packages WHERE package_id = ?", (package_id,)
        ).fetchone()
        if package is None or package["archived_at"] is not None:
            raise KeyError(package_id)
        existing = conn.execute(
            "SELECT * FROM ai_api_models WHERE backend_id = ?", (backend_id,)
        ).fetchone()
        effective_config = (
            dict(config)
            if config is not None
            else (_load(existing["config_json"]) or {})
            if existing is not None
            else {}
        )
        if set(capabilities).intersection(TEXT_GENERATION_CAPABILITIES):
            raw_context = effective_config.get(
                "max_context_tokens", DEFAULT_MODEL_MAX_CONTEXT_TOKENS
            )
            try:
                max_context_tokens = int(raw_context)
            except (TypeError, ValueError) as exc:
                raise ValueError("model max_context_tokens must be an integer") from exc
            if (
                isinstance(raw_context, bool)
                or max_context_tokens < MINIMUM_MODEL_MAX_CONTEXT_TOKENS
            ):
                raise ValueError("text and vision models require at least 128000 context tokens")
            effective_config["max_context_tokens"] = min(max_context_tokens, 10_000_000)
        if existing is None:
            self._insert_ai_api_model(
                conn,
                package,
                package_id=package_id,
                backend_id=backend_id,
                model_key=model_key,
                display_name=display_name,
                capabilities=capabilities,
                priority=priority,
                enabled=enabled,
                config=effective_config,
                now=now,
            )
        else:
            self._update_ai_api_model(
                conn,
                existing,
                package,
                package_id=package_id,
                backend_id=backend_id,
                model_key=model_key,
                display_name=display_name,
                capabilities=capabilities,
                enabled=enabled,
                config=effective_config,
                expected_version=expected_version,
                now=now,
            )
        self._normalize_ai_api_priorities(
            conn,
            now,
            moved_backend_id=backend_id,
            requested_priority=max(1, int(priority)),
        )
        self._sync_ai_api_model_runtime(conn, backend_id, now)
        result = conn.execute(
            "SELECT * FROM ai_api_models WHERE backend_id = ?", (backend_id,)
        ).fetchone()
        assert result is not None
        return result

    async def upsert_ai_api_model(
        self,
        package_id: str,
        backend_id: str,
        *,
        model_key: str,
        display_name: str = "",
        capabilities: tuple[str, ...] | list[str] = (),
        priority: int = 1,
        enabled: bool = True,
        config: dict[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        package_id = self._validate_ai_identifier(package_id, "package_id")
        backend_id = self._validate_ai_identifier(backend_id, "backend_id")
        model_key = str(model_key or "").strip()
        if not model_key:
            raise ValueError("model_key cannot be empty")
        normalized_caps = self._normalize_ai_capabilities(capabilities)
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            return self._upsert_ai_api_model_sql(
                conn,
                package_id=package_id,
                backend_id=backend_id,
                model_key=model_key,
                display_name=display_name,
                capabilities=normalized_caps,
                priority=priority,
                enabled=enabled,
                config=config,
                expected_version=expected_version,
                now=now,
            )

        return self._ai_api_model(await self.uow.run(operation))

    async def get_ai_api_model(
        self, backend_id: str, *, include_archived: bool = False
    ) -> dict[str, Any] | None:
        clauses = ["backend_id = ?"]
        if not include_archived:
            clauses.append("archived_at IS NULL")
        row = await self.db.fetch_one(
            f"SELECT * FROM ai_api_models WHERE {' AND '.join(clauses)}",
            (str(backend_id),),
        )
        return self._ai_api_model(row) if row else None

    async def list_ai_api_models(
        self,
        package_id: str | None = None,
        *,
        profile_id: str | None = None,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if package_id is not None:
            clauses.append("model.package_id = ?")
            parameters.append(str(package_id))
        if profile_id is not None:
            clauses.append("package.profile_id = ?")
            parameters.append(str(profile_id))
        if not include_archived:
            clauses.append("model.archived_at IS NULL")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = await self.db.fetch_all(
            f"""SELECT model.* FROM ai_api_models model
            JOIN ai_api_packages package USING(package_id) {where}
            ORDER BY model.priority ASC, model.backend_id ASC""",
            parameters,
        )
        return [self._ai_api_model(row) for row in rows]

    async def set_ai_api_model_enabled(
        self, backend_id: str, enabled: bool, *, expected_version: int | None = None
    ) -> dict[str, Any]:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            row = conn.execute(
                """SELECT model.* FROM ai_api_models model
                JOIN ai_api_packages package USING(package_id)
                WHERE model.backend_id = ?""",
                (backend_id,),
            ).fetchone()
            if row is None or row["archived_at"] is not None:
                raise KeyError(backend_id)
            self._require_expected_version(row, expected_version, "API model")
            conn.execute(
                """UPDATE ai_api_models SET enabled = ?, version = version + 1,
                updated_at = ? WHERE backend_id = ?""",
                (int(enabled), now, backend_id),
            )
            self._sync_ai_api_model_runtime(conn, backend_id, now)
            return conn.execute(
                "SELECT * FROM ai_api_models WHERE backend_id = ?", (backend_id,)
            ).fetchone()

        return self._ai_api_model(await self.uow.run(operation))

    async def archive_ai_api_model(
        self, backend_id: str, *, expected_version: int | None = None
    ) -> dict[str, Any]:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            row = conn.execute(
                """SELECT model.*, package.profile_id FROM ai_api_models model
                JOIN ai_api_packages package USING(package_id)
                WHERE model.backend_id = ?""",
                (backend_id,),
            ).fetchone()
            if row is None:
                raise KeyError(backend_id)
            self._require_expected_version(row, expected_version, "API model")
            conn.execute(
                """UPDATE ai_api_models SET enabled = 0, archived_at = ?,
                version = version + 1, updated_at = ? WHERE backend_id = ?""",
                (now, now, backend_id),
            )
            self._sync_ai_api_model_runtime(conn, backend_id, now)
            self._normalize_ai_api_priorities(conn, now, profile_id=str(row["profile_id"]))
            return conn.execute(
                "SELECT * FROM ai_api_models WHERE backend_id = ?", (backend_id,)
            ).fetchone()

        return self._ai_api_model(await self.uow.run(operation))
