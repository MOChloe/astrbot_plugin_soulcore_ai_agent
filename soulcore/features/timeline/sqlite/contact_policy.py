from __future__ import annotations

from functools import partial

from .support import (
    CONTACT_POLICY_FIELDS,
    CONTACT_POLICY_STORAGE_FIELDS,
    PLATFORM_CONTACT_POLICY_FIELDS,
    Any,
    Mapping,
    _dt,
    _now,
    sqlite3,
)


class ContactPolicyRecords:
    async def get_contact_policy(self, profile_id: str, scope: str) -> dict[str, Any] | None:
        self._validate_scope(scope)
        row = await self.db.fetch_one(
            """SELECT cp.*, NULLIF(prs.timezone, '') AS timezone
            FROM contact_policies cp
            JOIN profile_runtime_settings prs ON prs.profile_id = cp.profile_id
            WHERE cp.profile_id = ? AND cp.scope = ?""",
            (profile_id, scope),
        )
        return self._record(row, json_columns=()) if row else None

    async def update_contact_policy(
        self,
        profile_id: str,
        scope: str,
        patch: Mapping[str, Any],
        *,
        expected_version: int,
        expected_timezone_version: int | None = None,
    ) -> dict[str, Any] | None:
        self._validate_scope(scope)
        unknown = set(patch) - set(CONTACT_POLICY_FIELDS)
        if unknown:
            raise ValueError(f"unsupported contact policy fields: {sorted(unknown)}")
        current = await self.get_contact_policy(profile_id, scope)
        if current is None:
            raise KeyError((profile_id, scope))
        values = {name: current[name] for name in CONTACT_POLICY_FIELDS}
        values.update(patch)
        self._validate_contact_policy(values)
        table_patch = {name: value for name, value in patch.items() if name != "timezone"}
        assignments = [f"{name} = ?" for name in table_patch]
        params = [self._policy_sql_value(name, value) for name, value in table_patch.items()]
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> bool:
            if "timezone" in patch:
                if expected_timezone_version is None:
                    raise ValueError("expected_timezone_version is required when timezone changes")
                timezone_row = conn.execute(
                    """SELECT version FROM profile_runtime_settings
                    WHERE profile_id = ? AND version = ?""",
                    (profile_id, int(expected_timezone_version)),
                ).fetchone()
                if timezone_row is None:
                    return False
            prefix = f"{', '.join(assignments)}, " if assignments else ""
            cursor = conn.execute(
                f"UPDATE contact_policies SET {prefix}version = version + 1, "
                "updated_at = ? WHERE profile_id = ? AND scope = ? AND version = ?",
                (*params, now, profile_id, scope, int(expected_version)),
            )
            if cursor.rowcount != 1:
                return False
            if "timezone" in patch:
                conn.execute(
                    """UPDATE profile_runtime_settings SET timezone = ?,
                    version = version + 1, updated_at = ? WHERE profile_id = ?
                    AND version = ?""",
                    (
                        str(patch["timezone"] or "").strip(),
                        now,
                        profile_id,
                        int(expected_timezone_version),
                    ),
                )
            return True

        if patch and not await self.uow.run(operation):
            return None
        return await self.get_contact_policy(profile_id, scope)

    async def configure_quick_setup_contact(
        self,
        profile_id: str,
        *,
        proactive_enabled: bool,
        contact_patch: Mapping[str, Any],
        timezone: str | None,
        expected_versions: Mapping[str, Any],
    ) -> str:
        """Apply both guided contact defaults as one optimistic transaction.

        Platform and per-contact overrides intentionally remain outside this write: the
        guide owns only the role's private/group defaults and timezone.
        """

        unknown = set(contact_patch) - set(CONTACT_POLICY_STORAGE_FIELDS)
        if unknown:
            raise ValueError(f"unsupported quick contact fields: {sorted(unknown)}")
        now = _dt(_now())
        return await self.uow.run(
            partial(
                self._configure_quick_setup_contact_transaction,
                profile_id=profile_id,
                proactive_enabled=proactive_enabled,
                contact_patch=contact_patch,
                timezone=timezone,
                expected_versions=expected_versions,
                now=now,
            )
        )

    def _configure_quick_setup_contact_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        proactive_enabled: bool,
        contact_patch: Mapping[str, Any],
        timezone: str | None,
        expected_versions: Mapping[str, Any],
        now: str,
    ) -> str:
        contact_rows, scope_rows, timezone_row = _quick_contact_rows(conn, profile_id)
        contact_changes = self._quick_contact_changes(
            contact_rows,
            contact_patch,
            proactive_enabled=proactive_enabled,
            timezone=timezone,
            stored_timezone=str(timezone_row["timezone"] or ""),
        )
        scope_changes = {
            scope: int(bool(proactive_enabled)) != int(scope_rows[scope]["proactive_enabled"])
            for scope in ("private", "group")
        }
        timezone_changed = timezone is not None and str(timezone).strip() != str(
            timezone_row["timezone"] or ""
        )
        if not any(contact_changes.values()) and not any(
            (*scope_changes.values(), timezone_changed)
        ):
            return "idempotent"
        if _quick_expected_versions(expected_versions) != _quick_actual_versions(
            contact_rows, scope_rows, timezone_row
        ):
            return "conflict"
        self._write_quick_contact_changes(
            conn,
            profile_id=profile_id,
            proactive_enabled=proactive_enabled,
            contact_changes=contact_changes,
            scope_changes=scope_changes,
            timezone=timezone,
            timezone_changed=timezone_changed,
            now=now,
        )
        return "applied"

    def _quick_contact_changes(
        self,
        contact_rows: Mapping[str, Mapping[str, Any]],
        contact_patch: Mapping[str, Any],
        *,
        proactive_enabled: bool,
        timezone: str | None,
        stored_timezone: str,
    ) -> dict[str, dict[str, Any]]:
        changes: dict[str, dict[str, Any]] = {}
        effective_timezone = (
            (str(timezone).strip() or None)
            if timezone is not None
            else (stored_timezone.strip() or None)
        )
        for scope in ("private", "group"):
            desired = {name: contact_rows[scope][name] for name in CONTACT_POLICY_STORAGE_FIELDS}
            desired.update(contact_patch)
            desired["proactive_enabled"] = bool(proactive_enabled)
            self._validate_contact_policy({**desired, "timezone": effective_timezone})
            changes[scope] = {
                name: value
                for name, value in desired.items()
                if self._policy_sql_value(name, value) != contact_rows[scope][name]
            }
        return changes

    def _write_quick_contact_changes(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        proactive_enabled: bool,
        contact_changes: Mapping[str, Mapping[str, Any]],
        scope_changes: Mapping[str, bool],
        timezone: str | None,
        timezone_changed: bool,
        now: str,
    ) -> None:
        for scope in ("private", "group"):
            self._write_quick_contact_scope(
                conn,
                profile_id=profile_id,
                scope=scope,
                proactive_enabled=proactive_enabled,
                changes=contact_changes[scope],
                scope_changed=scope_changes[scope],
                now=now,
            )
        if timezone_changed:
            conn.execute(
                "UPDATE profile_runtime_settings SET timezone = ?, "
                "version = version + 1, updated_at = ? WHERE profile_id = ?",
                (str(timezone).strip(), now, profile_id),
            )

    def _write_quick_contact_scope(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        scope: str,
        proactive_enabled: bool,
        changes: Mapping[str, Any],
        scope_changed: bool,
        now: str,
    ) -> None:
        if changes:
            assignments = [f"{name} = ?" for name in changes]
            params = [self._policy_sql_value(name, value) for name, value in changes.items()]
            conn.execute(
                f"UPDATE contact_policies SET {', '.join(assignments)}, "
                "version = version + 1, updated_at = ? "
                "WHERE profile_id = ? AND scope = ?",
                (*params, now, profile_id, scope),
            )
        if scope_changed:
            conn.execute(
                "UPDATE scope_configs SET proactive_enabled = ?, "
                "version = version + 1, updated_at = ? "
                "WHERE profile_id = ? AND scope = ?",
                (int(bool(proactive_enabled)), now, profile_id, scope),
            )

    async def get_profile_timezone(self, profile_id: str) -> dict[str, Any]:
        row = await self.db.fetch_one(
            "SELECT * FROM profile_runtime_settings WHERE profile_id = ?", (profile_id,)
        )
        if row is None:
            raise KeyError(profile_id)
        return self._record(row, json_columns=())

    async def get_platform_contact_policy(
        self,
        profile_id: str,
        scope: str,
        platform_instance_id: str,
    ) -> dict[str, Any] | None:
        self._validate_scope(scope)
        row = await self.db.fetch_one(
            """SELECT * FROM platform_connection_policies
            WHERE profile_id = ? AND scope = ? AND platform_instance_id = ?""",
            (profile_id, scope, platform_instance_id),
        )
        return self._record(row, json_columns=()) if row else None

    async def upsert_platform_contact_policy(
        self,
        profile_id: str,
        scope: str,
        platform_instance_id: str,
        patch: Mapping[str, Any],
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any] | None:
        self._validate_scope(scope)
        if not str(platform_instance_id).strip():
            raise ValueError("platform_instance_id cannot be empty")
        unknown = set(patch) - set(PLATFORM_CONTACT_POLICY_FIELDS)
        if unknown:
            raise ValueError(f"unsupported platform policy fields: {sorted(unknown)}")
        current = await self.get_platform_contact_policy(profile_id, scope, platform_instance_id)
        now = _dt(_now())
        if current is None:
            if expected_version not in (None, 0):
                return None
            columns = [
                "profile_id",
                "scope",
                "platform_instance_id",
                *patch,
                "created_at",
                "updated_at",
            ]
            values = [
                profile_id,
                scope,
                platform_instance_id,
                *(self._policy_sql_value(k, v) for k, v in patch.items()),
                now,
                now,
            ]
            try:
                await self.db.call(
                    lambda conn: conn.execute(
                        f"INSERT INTO platform_connection_policies({', '.join(columns)}) "
                        f"VALUES ({','.join('?' for _ in columns)})",
                        values,
                    ),
                    transaction=True,
                )
            except sqlite3.IntegrityError:
                return None
        else:
            if expected_version is None:
                expected_version = int(current["version"])
            assignments = [f"{name} = ?" for name in patch]
            params = [self._policy_sql_value(k, v) for k, v in patch.items()]
            params.extend((now, profile_id, scope, platform_instance_id, expected_version))
            cursor = await self.db.call(
                lambda conn: conn.execute(
                    f"UPDATE platform_connection_policies SET {', '.join(assignments)}, "
                    "version = version + 1, updated_at = ? WHERE profile_id = ? "
                    "AND scope = ? AND platform_instance_id = ? AND version = ?",
                    params,
                ),
                transaction=True,
            )
            if cursor.rowcount != 1:
                return None
        return await self.get_platform_contact_policy(profile_id, scope, platform_instance_id)

    async def delete_platform_contact_policy(
        self,
        profile_id: str,
        scope: str,
        platform_instance_id: str,
        *,
        expected_version: int,
    ) -> bool:
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """DELETE FROM platform_connection_policies WHERE profile_id = ?
                AND scope = ? AND platform_instance_id = ? AND version = ?""",
                (profile_id, scope, platform_instance_id, expected_version),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1

    async def get_instance_contact_override(
        self,
        profile_id: str,
        instance_id: str,
    ) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            """SELECT * FROM instance_contact_overrides
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        )
        return self._record(row, json_columns=()) if row else None

    async def upsert_instance_contact_override(
        self,
        profile_id: str,
        instance_id: str,
        patch: Mapping[str, Any],
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any] | None:
        unknown = set(patch) - set(CONTACT_POLICY_STORAGE_FIELDS)
        if unknown:
            raise ValueError(f"unsupported instance contact fields: {sorted(unknown)}")
        current = await self.get_instance_contact_override(profile_id, instance_id)
        now = _dt(_now())
        if current is None:
            if expected_version not in (None, 0):
                return None
            columns = ["profile_id", "instance_id", *patch, "created_at", "updated_at"]
            values = [
                profile_id,
                instance_id,
                *(self._policy_sql_value(k, v) for k, v in patch.items()),
                now,
                now,
            ]
            try:
                await self.db.call(
                    lambda conn: conn.execute(
                        f"INSERT INTO instance_contact_overrides({', '.join(columns)}) "
                        f"VALUES ({','.join('?' for _ in columns)})",
                        values,
                    ),
                    transaction=True,
                )
            except sqlite3.IntegrityError:
                return None
        else:
            if expected_version is None:
                expected_version = int(current["version"])
            assignments = [f"{name} = ?" for name in patch]
            params = [self._policy_sql_value(k, v) for k, v in patch.items()]
            params.extend((now, profile_id, instance_id, expected_version))
            cursor = await self.db.call(
                lambda conn: conn.execute(
                    f"UPDATE instance_contact_overrides SET {', '.join(assignments)}, "
                    "version = version + 1, updated_at = ? WHERE profile_id = ? "
                    "AND instance_id = ? AND version = ?",
                    params,
                ),
                transaction=True,
            )
            if cursor.rowcount != 1:
                return None
        return await self.get_instance_contact_override(profile_id, instance_id)

    async def delete_instance_contact_override(
        self,
        profile_id: str,
        instance_id: str,
        *,
        expected_version: int,
    ) -> bool:
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """DELETE FROM instance_contact_overrides WHERE profile_id = ?
                AND instance_id = ? AND version = ?""",
                (profile_id, instance_id, expected_version),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1

    async def get_delivery_policy(
        self,
        profile_id: str,
        scope: str,
    ) -> dict[str, Any] | None:
        self._validate_scope(scope)
        row = await self.db.fetch_one(
            """SELECT * FROM scope_delivery_policies
            WHERE profile_id = ? AND scope = ?""",
            (profile_id, scope),
        )
        return self._record(row, json_columns=()) if row else None

    async def update_delivery_policy(
        self,
        profile_id: str,
        scope: str,
        patch: Mapping[str, Any],
        *,
        expected_version: int,
    ) -> dict[str, Any] | None:
        self._validate_scope(scope)
        unknown = set(patch) - {
            "group_send_qpm_limit",
            "send_qpm_limit",
        }
        if unknown:
            raise ValueError(f"unsupported delivery policy fields: {sorted(unknown)}")
        if "group_send_qpm_limit" in patch and int(patch["group_send_qpm_limit"]) < 1:
            raise ValueError("group_send_qpm_limit must be positive")
        if "send_qpm_limit" in patch and int(patch["send_qpm_limit"]) < 1:
            raise ValueError("send_qpm_limit must be positive")
        assignments = [f"{name} = ?" for name in patch]
        if assignments:
            params = [int(value) for value in patch.values()]
            params.extend((_dt(_now()), profile_id, scope, expected_version))
            cursor = await self.db.call(
                lambda conn: conn.execute(
                    f"UPDATE scope_delivery_policies SET {', '.join(assignments)}, "
                    "version = version + 1, updated_at = ? WHERE profile_id = ? "
                    "AND scope = ? AND version = ?",
                    params,
                ),
                transaction=True,
            )
            if cursor.rowcount != 1:
                return None
        return await self.get_delivery_policy(profile_id, scope)

    async def get_instance_delivery_override(
        self,
        profile_id: str,
        instance_id: str,
    ) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            """SELECT * FROM instance_delivery_overrides WHERE profile_id = ?
            AND instance_id = ?""",
            (profile_id, instance_id),
        )
        return self._record(row, json_columns=()) if row else None

    async def upsert_instance_delivery_override(
        self,
        profile_id: str,
        instance_id: str,
        *,
        send_qpm_limit: int | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        if send_qpm_limit is None:
            raise ValueError("at least one delivery override is required")
        if send_qpm_limit is not None and int(send_qpm_limit) < 1:
            raise ValueError("send_qpm_limit must be positive")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            current = conn.execute(
                """SELECT version FROM instance_delivery_overrides
                WHERE profile_id = ? AND instance_id = ?""",
                (profile_id, instance_id),
            ).fetchone()
            if current is None:
                if expected_version not in {None, 0}:
                    raise ValueError("instance delivery override version conflict")
                conn.execute(
                    """INSERT INTO instance_delivery_overrides(
                        profile_id, instance_id, send_qpm_limit, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)""",
                    (
                        profile_id,
                        instance_id,
                        send_qpm_limit,
                        now,
                        now,
                    ),
                )
            else:
                version = int(current["version"])
                if expected_version is not None and int(expected_version) != version:
                    raise ValueError("instance delivery override version conflict")
                conn.execute(
                    """UPDATE instance_delivery_overrides SET send_qpm_limit = ?,
                    version = version + 1, updated_at = ?
                    WHERE profile_id = ? AND instance_id = ?""",
                    (
                        send_qpm_limit,
                        now,
                        profile_id,
                        instance_id,
                    ),
                )
            row = conn.execute(
                """SELECT * FROM instance_delivery_overrides WHERE profile_id = ?
                AND instance_id = ?""",
                (profile_id, instance_id),
            ).fetchone()
            assert row is not None
            return row

        row = await self.uow.run(operation)
        return self._record(row, json_columns=())

    async def delete_instance_delivery_override(
        self,
        profile_id: str,
        instance_id: str,
        *,
        expected_version: int | None = None,
    ) -> bool:
        params: list[Any] = [profile_id, instance_id]
        sql = """DELETE FROM instance_delivery_overrides
        WHERE profile_id = ? AND instance_id = ?"""
        if expected_version is not None:
            sql += " AND version = ?"
            params.append(int(expected_version))
        cursor = await self.db.call(
            lambda conn: conn.execute(sql, params),
            transaction=True,
        )
        return cursor.rowcount == 1

    async def resolve_expression_pacing_policy(
        self,
        profile_id: str,
        instance_id: str,
        platform_instance_id: str | None = None,
    ) -> dict[str, Any]:
        instance = await self.db.fetch_one(
            """SELECT scope, platform_id FROM character_instances
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        )
        if instance is None:
            raise KeyError((profile_id, instance_id))
        scope = str(instance["scope"])
        platform_key = str(platform_instance_id or instance["platform_id"] or "")
        base = await self.get_delivery_policy(profile_id, scope)
        if base is None:
            raise KeyError((profile_id, scope))
        send_qpm_limit = int(base["send_qpm_limit"])
        account_limit: int | None = None
        sources = {"send_qpm_limit": "scope"}
        platform = (
            await self.get_platform_contact_policy(profile_id, scope, platform_key)
            if platform_key
            else None
        )
        if platform is not None:
            if platform.get("send_qpm_limit") is not None:
                send_qpm_limit = int(platform["send_qpm_limit"])
                sources["send_qpm_limit"] = "platform"
            if platform.get("account_send_qpm_limit") is not None:
                account_limit = int(platform["account_send_qpm_limit"])
        override = await self.get_instance_delivery_override(profile_id, instance_id)
        if override is not None and override.get("send_qpm_limit") is not None:
            send_qpm_limit = int(override["send_qpm_limit"])
            sources["send_qpm_limit"] = "instance"
        return {
            "profile_id": profile_id,
            "instance_id": instance_id,
            "scope": scope,
            "platform_instance_id": platform_key,
            "send_qpm_limit": max(1, send_qpm_limit),
            "account_send_qpm_limit": account_limit,
            "sources": sources,
        }

    async def resolve_contact_policy(
        self,
        profile_id: str,
        instance_id: str,
        platform_instance_id: str | None = None,
    ) -> dict[str, Any]:
        instance = await self.db.fetch_one(
            """SELECT scope, platform_id FROM character_instances
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        )
        if instance is None:
            raise KeyError((profile_id, instance_id))
        scope = str(instance["scope"])
        platform_key = str(platform_instance_id or instance["platform_id"] or "")
        base = await self.get_contact_policy(profile_id, scope)
        if base is None:
            raise KeyError((profile_id, scope))
        platform = (
            await self.get_platform_contact_policy(profile_id, scope, platform_key)
            if platform_key
            else None
        )
        override = await self.get_instance_contact_override(profile_id, instance_id)
        effective, sources = self._resolve_scalar_contact_fields(base, platform, override)
        self._resolve_contact_limit_fields(effective, sources, base, platform, override)
        self._validate_contact_policy(effective)
        effective.update(
            {
                "profile_id": profile_id,
                "instance_id": instance_id,
                "scope": scope,
                "platform_instance_id": platform_key,
                "sources": sources,
            }
        )
        return effective

    @staticmethod
    def _resolve_scalar_contact_fields(
        base: Mapping[str, Any],
        platform: Mapping[str, Any] | None,
        override: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        effective: dict[str, Any] = {}
        sources: dict[str, str] = {}
        limit_fields = {
            "daily_limit_mode",
            "daily_success_limit",
            "unanswered_limit_mode",
            "max_consecutive_unanswered",
        }
        for field in (item for item in CONTACT_POLICY_FIELDS if item not in limit_fields):
            value, source = base[field], "scope"
            if platform is not None and platform.get(field) is not None:
                value, source = platform[field], "platform"
            if override is not None and override.get(field) is not None:
                value, source = override[field], "instance"
            effective[field], sources[field] = value, source
        return effective, sources

    @staticmethod
    def _resolve_contact_limit_fields(
        effective: dict[str, Any],
        sources: dict[str, str],
        base: Mapping[str, Any],
        platform: Mapping[str, Any] | None,
        override: Mapping[str, Any] | None,
    ) -> None:
        for mode_field, value_field in (
            ("daily_limit_mode", "daily_success_limit"),
            ("unanswered_limit_mode", "max_consecutive_unanswered"),
        ):
            mode, value, source = base[mode_field], base[value_field], "scope"
            for candidate, candidate_source in ((platform, "platform"), (override, "instance")):
                if candidate is not None and candidate.get(mode_field) not in (None, "INHERIT"):
                    mode = candidate[mode_field]
                    value = candidate.get(value_field)
                    source = candidate_source
            effective[mode_field], effective[value_field] = mode, value
            sources[mode_field] = sources[value_field] = source

    async def get_contact_state(
        self,
        profile_id: str,
        instance_id: str,
    ) -> dict[str, Any]:
        row = await self.db.fetch_one(
            """SELECT * FROM instance_contact_state
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        )
        if row is None:
            raise KeyError((profile_id, instance_id))
        return self._record(
            row,
            json_columns=(
                "evidence_watermarks_json",
                "deferred_evidence_json",
                "evidence_snapshot_json",
            ),
        )


def _quick_contact_rows(
    conn: sqlite3.Connection, profile_id: str
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], sqlite3.Row]:
    contact_rows = {
        str(row["scope"]): dict(row)
        for row in conn.execute(
            "SELECT * FROM contact_policies WHERE profile_id = ?",
            (profile_id,),
        ).fetchall()
    }
    scope_rows = {
        str(row["scope"]): dict(row)
        for row in conn.execute(
            "SELECT scope, proactive_enabled, version FROM scope_configs WHERE profile_id = ?",
            (profile_id,),
        ).fetchall()
    }
    timezone_row = conn.execute(
        "SELECT timezone, version FROM profile_runtime_settings WHERE profile_id = ?",
        (profile_id,),
    ).fetchone()
    if (
        set(contact_rows) != {"private", "group"}
        or set(scope_rows) != {"private", "group"}
        or timezone_row is None
    ):
        raise KeyError(profile_id)
    return contact_rows, scope_rows, timezone_row


def _quick_expected_versions(value: Mapping[str, Any]) -> dict[str, int]:
    return {
        "private_contact": int(value.get("private_contact", -1)),
        "group_contact": int(value.get("group_contact", -1)),
        "private_scope": int(value.get("private_scope", -1)),
        "group_scope": int(value.get("group_scope", -1)),
        "timezone": int(value.get("timezone", -1)),
    }


def _quick_actual_versions(
    contact_rows: Mapping[str, Mapping[str, Any]],
    scope_rows: Mapping[str, Mapping[str, Any]],
    timezone_row: sqlite3.Row,
) -> dict[str, int]:
    return {
        "private_contact": int(contact_rows["private"]["version"]),
        "group_contact": int(contact_rows["group"]["version"]),
        "private_scope": int(scope_rows["private"]["version"]),
        "group_scope": int(scope_rows["group"]["version"]),
        "timezone": int(timezone_row["version"]),
    }
