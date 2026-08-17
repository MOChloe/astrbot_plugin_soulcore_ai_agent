from __future__ import annotations

from dataclasses import dataclass

from ....features.media.ports import (
    MAX_ANIMATION_DECODED_PIXELS,
    MAX_ANIMATION_DURATION_MS,
    MAX_ANIMATION_FRAMES,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_DECODED_PIXELS,
    StoredMediaFile,
)
from ....storage.sqlite.dialogue_turns import dialogue_progress_eligible_sql
from ....storage.sqlite.runtime_file_cleanup import (
    enqueue_runtime_file_cleanup_sql,
    finish_runtime_file_cleanup_guard_sql,
)
from .support import (
    Any,
    CharacterIdentityReference,
    Mapping,
    StickerConfig,
    _dt,
    _dump,
    _now,
    _parse,
    datetime,
    re,
    sqlite3,
    timedelta,
    uuid,
)


def _normalize_identity_reference_location(
    *, storage_relpath: str, mime_type: str, file_extension: str
) -> tuple[str, str, str]:
    relpath = str(storage_relpath).strip().replace("\\", "/")
    media_type = str(mime_type).strip().lower()
    extension = str(file_extension).strip().lower()
    if not relpath or relpath.startswith("/") or ".." in relpath.split("/"):
        raise ValueError("identity reference requires a safe relative storage path")
    if media_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise ValueError("unsupported identity reference format")
    return relpath, media_type, extension


def _validate_identity_reference_budget(
    *, byte_size: int, width: int, height: int, frame_count: int, duration_ms: int
) -> None:
    if int(byte_size) < 1:
        raise ValueError("identity reference byte size is invalid")
    if min(int(width), int(height)) < 1:
        raise ValueError("identity reference dimensions are invalid")
    if int(frame_count) < 1:
        raise ValueError("identity reference frame count is invalid")
    if int(duration_ms) < 0:
        raise ValueError("identity reference duration is invalid")
    if int(byte_size) > MAX_IMAGE_BYTES:
        raise ValueError("identity reference exceeds the encoded byte budget")
    decoded_pixels = int(width) * int(height)
    if decoded_pixels > MAX_IMAGE_DECODED_PIXELS:
        raise ValueError("identity reference exceeds the decoded pixel budget")
    if int(frame_count) > MAX_ANIMATION_FRAMES:
        raise ValueError("identity reference exceeds the animation frame budget")
    if int(duration_ms) > MAX_ANIMATION_DURATION_MS:
        raise ValueError("identity reference exceeds the animation duration budget")
    if decoded_pixels * int(frame_count) > MAX_ANIMATION_DECODED_PIXELS:
        raise ValueError("identity reference exceeds the animation decode budget")


@dataclass(frozen=True, slots=True)
class ReplaceIdentityReferenceSql:
    profile_id: str
    scope: str
    identifier: str
    asset: str
    relpath: str
    media_type: str
    extension: str
    digest: str
    byte_size: int
    width: int
    height: int
    frame_count: int
    duration_ms: int
    label: str
    description: str
    metadata: Mapping[str, Any]
    cleanup_guard_id: int
    now: str

    def __call__(self, conn: sqlite3.Connection) -> tuple[sqlite3.Row, bool]:
        owner = conn.execute(
            "SELECT 1 FROM scope_configs WHERE profile_id = ? AND scope = ?",
            (self.profile_id, self.scope),
        ).fetchone()
        if owner is None:
            raise KeyError((self.profile_id, self.scope))
        previous = conn.execute(
            """SELECT reference_id, asset_id, storage_relpath, sha256, byte_size
            FROM character_identity_references
            WHERE profile_id = ? AND scope = ?""",
            (self.profile_id, self.scope),
        ).fetchone()
        conn.execute(
            "DELETE FROM character_identity_references WHERE profile_id = ? AND scope = ?",
            (self.profile_id, self.scope),
        )
        conn.execute(
            """INSERT INTO character_identity_references(
                reference_id, profile_id, scope, asset_id, storage_relpath,
                mime_type, file_extension, sha256, byte_size, width, height,
                frame_count, duration_ms, label, identity_description,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.identifier,
                self.profile_id,
                self.scope,
                self.asset,
                self.relpath,
                self.media_type,
                self.extension,
                self.digest,
                self.byte_size,
                self.width,
                self.height,
                self.frame_count,
                self.duration_ms,
                self.label,
                self.description,
                _dump(dict(self.metadata)),
                self.now,
                self.now,
            ),
        )
        row = conn.execute(
            "SELECT * FROM character_identity_references WHERE reference_id = ?",
            (self.identifier,),
        ).fetchone()
        assert row is not None
        finish_runtime_file_cleanup_guard_sql(
            conn,
            cleanup_id=self.cleanup_guard_id,
            profile_id=self.profile_id,
            instance_id=f"character-identity:{self.scope}",
            stored=StoredMediaFile(
                asset_id=self.asset,
                relative_path=self.relpath,
                mime_type=self.media_type,
                file_extension=self.extension,
                sha256=self.digest,
                byte_size=self.byte_size,
                width=self.width,
                height=self.height,
                frame_count=self.frame_count,
            ),
        )
        replaced = previous is not None and str(previous["storage_relpath"]) != self.relpath
        if replaced:
            enqueue_runtime_file_cleanup_sql(
                conn,
                profile_id=self.profile_id,
                instance_id=f"character-identity:{self.scope}",
                storage_kind="MEDIA",
                storage_relpath=str(previous["storage_relpath"]),
                owner_id=str(previous["asset_id"]),
                expected_sha256=str(previous["sha256"]),
                expected_byte_size=int(previous["byte_size"]),
                reason="IDENTITY_REFERENCE_REPLACED",
                now=self.now,
            )
        return row, replaced


def delete_identity_reference_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    scope: str,
    reference_id: str = "",
    require_available: bool = True,
) -> sqlite3.Row:
    """Delete one locked identity-reference row and durably queue its file."""

    clauses = ["profile_id = ?", "scope = ?"]
    values: list[Any] = [profile_id, scope]
    if reference_id:
        clauses.append("reference_id = ?")
        values.append(reference_id)
    if require_available:
        clauses.append("file_status = 'AVAILABLE'")
    row = conn.execute(
        "SELECT * FROM character_identity_references WHERE " + " AND ".join(clauses),
        values,
    ).fetchone()
    if row is None:
        raise KeyError((profile_id, scope, reference_id))
    enqueue_runtime_file_cleanup_sql(
        conn,
        profile_id=profile_id,
        instance_id=f"character-identity:{scope}",
        storage_kind="MEDIA",
        storage_relpath=str(row["storage_relpath"]),
        owner_id=str(row["asset_id"]),
        expected_sha256=str(row["sha256"]),
        expected_byte_size=int(row["byte_size"]),
        reason="IDENTITY_REFERENCE_DELETED",
    )
    conn.execute(
        "DELETE FROM character_identity_references WHERE reference_id = ?",
        (str(row["reference_id"]),),
    )
    return row


class StickerConfigurationRecords:
    _STICKER_CONFIG_FIELDS = (
        "enabled",
        "player_collection_enabled",
        "web_collection_enabled",
        "generation_enabled",
        "trigger_mode",
        "turn_threshold",
        "elapsed_hours",
        "library_limit",
        "web_daily_limit",
        "generated_daily_limit",
        "requirements",
    )

    @classmethod
    def _validated_sticker_config_values(
        cls, current: StickerConfig, patch: Mapping[str, Any]
    ) -> dict[str, Any]:
        unknown = set(patch) - set(cls._STICKER_CONFIG_FIELDS)
        if unknown:
            raise ValueError("unsupported sticker config fields: " + ", ".join(sorted(unknown)))
        values = {field: getattr(current, field) for field in cls._STICKER_CONFIG_FIELDS}
        values.update(dict(patch))
        for key in (
            "enabled",
            "player_collection_enabled",
            "web_collection_enabled",
            "generation_enabled",
        ):
            values[key] = int(bool(values[key]))
        mode = str(values["trigger_mode"]).upper()
        if mode not in {"TURNS_ONLY", "TIME_ONLY", "ANY", "ALL"}:
            raise ValueError("invalid sticker trigger mode")
        values["trigger_mode"] = mode
        values["turn_threshold"] = int(values["turn_threshold"])
        values["elapsed_hours"] = float(values["elapsed_hours"])
        values["library_limit"] = int(values["library_limit"])
        values["web_daily_limit"] = int(values["web_daily_limit"])
        values["generated_daily_limit"] = int(values["generated_daily_limit"])
        if values["turn_threshold"] < 1 or values["elapsed_hours"] <= 0:
            raise ValueError("sticker trigger thresholds must be positive")
        if not 50 <= values["library_limit"] <= 10000:
            raise ValueError("sticker library_limit must be between 50 and 10000")
        if values["web_daily_limit"] < 0 or values["generated_daily_limit"] < 0:
            raise ValueError("sticker daily limits cannot be negative")
        values["requirements"] = str(values["requirements"] or "").strip()
        if len(values["requirements"]) > 4000:
            raise ValueError("sticker requirements must not exceed 4000 characters")
        return values

    @classmethod
    def _sticker_config_matches(cls, current: StickerConfig, values: Mapping[str, Any]) -> bool:
        return all(getattr(current, field) == values[field] for field in cls._STICKER_CONFIG_FIELDS)

    @staticmethod
    def _write_sticker_config(
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        scope: str,
        expected_version: int,
        values: Mapping[str, Any],
        now: str,
    ) -> int:
        cursor = conn.execute(
            """UPDATE sticker_configs SET enabled = ?, player_collection_enabled = ?,
                web_collection_enabled = ?, generation_enabled = ?,
                trigger_mode = ?, turn_threshold = ?,
                elapsed_hours = ?, library_limit = ?, web_daily_limit = ?,
                generated_daily_limit = ?, requirements = ?,
                version = version + 1, updated_at = ?
            WHERE profile_id = ? AND scope = ? AND version = ?""",
            (
                values["enabled"],
                values["player_collection_enabled"],
                values["web_collection_enabled"],
                values["generation_enabled"],
                values["trigger_mode"],
                values["turn_threshold"],
                values["elapsed_hours"],
                values["library_limit"],
                values["web_daily_limit"],
                values["generated_daily_limit"],
                values["requirements"],
                now,
                profile_id,
                scope,
                expected_version,
            ),
        )
        return int(cursor.rowcount)

    async def get_sticker_config(self, profile_id: str, scope: str) -> StickerConfig:
        self._validate_scope(scope)
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            owner = conn.execute(
                "SELECT 1 FROM scope_configs WHERE profile_id = ? AND scope = ?",
                (profile_id, scope),
            ).fetchone()
            if owner is None:
                raise KeyError((profile_id, scope))
            conn.execute(
                """INSERT OR IGNORE INTO sticker_configs(
                    profile_id, scope, created_at, updated_at
                ) VALUES (?, ?, ?, ?)""",
                (profile_id, scope, now, now),
            )
            row = conn.execute(
                "SELECT * FROM sticker_configs WHERE profile_id = ? AND scope = ?",
                (profile_id, scope),
            ).fetchone()
            assert row is not None
            return row

        return self._sticker_config(await self.uow.run(operation))

    async def update_sticker_config(
        self,
        profile_id: str,
        scope: str,
        patch: Mapping[str, Any],
        *,
        expected_version: int | None = None,
    ) -> StickerConfig:
        current = await self.get_sticker_config(profile_id, scope)
        values = self._validated_sticker_config_values(current, patch)
        compare_version = int(current.version if expected_version is None else expected_version)
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            rowcount = self._write_sticker_config(
                conn,
                profile_id=profile_id,
                scope=scope,
                expected_version=compare_version,
                values=values,
                now=now,
            )
            if rowcount == 1 and current.enabled and not bool(values["enabled"]):
                conn.execute(
                    """DELETE FROM sticker_run_candidates
                    WHERE profile_id = ? AND instance_id IN (
                        SELECT instance_id FROM character_instances
                        WHERE profile_id = ? AND scope = ?
                    )""",
                    (profile_id, profile_id, scope),
                )
            return rowcount

        rowcount = await self.db.call(operation, transaction=True)
        if rowcount != 1:
            raise ValueError("sticker config version conflict")
        return await self.get_sticker_config(profile_id, scope)

    async def update_sticker_configs_atomically(
        self,
        profile_id: str,
        patch: Mapping[str, Any],
        *,
        expected_versions: Mapping[str, int],
    ) -> tuple[dict[str, StickerConfig], dict[str, StickerConfig], bool]:
        """Apply one guided choice to private and group without partial state."""

        scopes = ("private", "group")
        if set(expected_versions) != set(scopes):
            raise ValueError("快速设置缺少私聊或群聊的当前配置版本")
        now = _dt(_now())

        def operation(
            conn: sqlite3.Connection,
        ) -> tuple[dict[str, sqlite3.Row], dict[str, sqlite3.Row], bool]:
            current_rows: dict[str, sqlite3.Row] = {}
            current_configs: dict[str, StickerConfig] = {}
            for scope in scopes:
                owner = conn.execute(
                    "SELECT 1 FROM scope_configs WHERE profile_id = ? AND scope = ?",
                    (profile_id, scope),
                ).fetchone()
                if owner is None:
                    raise KeyError((profile_id, scope))
                conn.execute(
                    """INSERT OR IGNORE INTO sticker_configs(
                        profile_id, scope, created_at, updated_at
                    ) VALUES (?, ?, ?, ?)""",
                    (profile_id, scope, now, now),
                )
                row = conn.execute(
                    "SELECT * FROM sticker_configs WHERE profile_id = ? AND scope = ?",
                    (profile_id, scope),
                ).fetchone()
                assert row is not None
                current_rows[scope] = row
                current_configs[scope] = self._sticker_config(row)

            changed = False
            for scope in scopes:
                current = current_configs[scope]
                expected = int(expected_versions[scope])
                if int(current.version) != expected:
                    raise ValueError("表情包配置已被其他请求修改，请重新读取后再保存")
                values = self._validated_sticker_config_values(current, patch)
                if self._sticker_config_matches(current, values):
                    continue
                if (
                    self._write_sticker_config(
                        conn,
                        profile_id=profile_id,
                        scope=scope,
                        expected_version=expected,
                        values=values,
                        now=now,
                    )
                    != 1
                ):
                    raise ValueError("表情包配置已被其他请求修改，请重新读取后再保存")
                if current.enabled and not bool(values["enabled"]):
                    conn.execute(
                        """DELETE FROM sticker_run_candidates
                        WHERE profile_id = ? AND instance_id IN (
                            SELECT instance_id FROM character_instances
                            WHERE profile_id = ? AND scope = ?
                        )""",
                        (profile_id, profile_id, scope),
                    )
                changed = True

            updated_rows = {
                scope: conn.execute(
                    "SELECT * FROM sticker_configs WHERE profile_id = ? AND scope = ?",
                    (profile_id, scope),
                ).fetchone()
                for scope in scopes
            }
            assert all(row is not None for row in updated_rows.values())
            return current_rows, updated_rows, changed

        current_rows, updated_rows, changed = await self.db.call(operation, transaction=True)
        return (
            {scope: self._sticker_config(row) for scope, row in current_rows.items()},
            {scope: self._sticker_config(row) for scope, row in updated_rows.items()},
            changed,
        )

    async def replace_character_identity_reference(
        self,
        profile_id: str,
        scope: str,
        *,
        asset_id: str,
        storage_relpath: str,
        mime_type: str,
        file_extension: str,
        sha256: str,
        byte_size: int,
        width: int,
        height: int,
        frame_count: int = 1,
        duration_ms: int = 0,
        label: str = "",
        identity_description: str = "",
        metadata: Mapping[str, Any] | None = None,
        reference_id: str = "",
        cleanup_guard_id: int,
    ) -> tuple[CharacterIdentityReference, bool]:
        """Atomically replace the scope-owned character identity reference."""

        self._validate_scope(scope)
        identifier = str(reference_id).strip() or "cir_" + uuid.uuid4().hex
        asset = str(asset_id).strip()
        relpath, media_type, extension = _normalize_identity_reference_location(
            storage_relpath=storage_relpath,
            mime_type=mime_type,
            file_extension=file_extension,
        )
        digest = str(sha256).strip().lower()
        if not asset:
            raise ValueError("identity reference requires a safe relative storage path")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("identity reference sha256 is invalid")
        _validate_identity_reference_budget(
            byte_size=byte_size,
            width=width,
            height=height,
            frame_count=frame_count,
            duration_ms=duration_ms,
        )
        clean_label = str(label or "").strip()[:100]
        clean_description = str(identity_description or "").strip()[:2000]
        now = _dt(_now())

        row, replaced = await self.uow.run(
            ReplaceIdentityReferenceSql(
                profile_id,
                scope,
                identifier,
                asset,
                relpath,
                media_type,
                extension,
                digest,
                int(byte_size),
                int(width),
                int(height),
                int(frame_count),
                int(duration_ms),
                clean_label,
                clean_description,
                dict(metadata or {}),
                int(cleanup_guard_id),
                now,
            )
        )
        return self._character_identity_reference(row), replaced

    async def get_character_identity_reference(
        self, profile_id: str, scope: str
    ) -> CharacterIdentityReference | None:
        self._validate_scope(scope)
        row = await self.db.fetch_one(
            """SELECT * FROM character_identity_references
            WHERE profile_id = ? AND scope = ? AND file_status = 'AVAILABLE'""",
            (profile_id, scope),
        )
        return self._character_identity_reference(row) if row is not None else None

    async def delete_character_identity_reference(
        self, profile_id: str, scope: str, reference_id: str
    ) -> dict[str, Any]:
        self._validate_scope(scope)

        def operation(conn: sqlite3.Connection) -> dict[str, Any]:
            row = delete_identity_reference_sql(
                conn,
                profile_id=profile_id,
                scope=scope,
                reference_id=reference_id,
                require_available=True,
            )
            return self._record(row, json_columns=("metadata_json",))

        return await self.uow.run(operation)

    async def get_sticker_trigger_state(
        self,
        profile_id: str,
        instance_id: str,
    ) -> dict[str, Any]:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            owner = conn.execute(
                "SELECT 1 FROM character_instances WHERE profile_id = ? AND instance_id = ?",
                (profile_id, instance_id),
            ).fetchone()
            if owner is None:
                raise KeyError((profile_id, instance_id))
            baseline = int(
                conn.execute(
                    f"""SELECT COALESCE(MAX(m.message_id), 0) FROM instance_messages m
                    WHERE m.profile_id = ? AND m.instance_id = ?
                      AND {dialogue_progress_eligible_sql("m")}""",
                    (profile_id, instance_id),
                ).fetchone()[0]
            )
            conn.execute(
                """INSERT OR IGNORE INTO sticker_trigger_states(
                    profile_id, instance_id, processed_through_message_id,
                    enabled_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (profile_id, instance_id, baseline, now, now, now),
            )
            row = conn.execute(
                "SELECT * FROM sticker_trigger_states WHERE profile_id = ? AND instance_id = ?",
                (profile_id, instance_id),
            ).fetchone()
            assert row is not None
            return row

        return self._sticker_trigger_state(await self.uow.run(operation))

    async def update_sticker_trigger_state(
        self,
        profile_id: str,
        instance_id: str,
        *,
        processed_through_message_id: int | None = None,
        frozen_through_message_id: int | None | object = ...,
        enabled_at: datetime | None | object = ...,
        last_success_at: datetime | None | object = ...,
        cooldown_until: datetime | None | object = ...,
        active_task_id: int | None | object = ...,
        last_error: str | None = None,
    ) -> dict[str, Any]:
        await self.get_sticker_trigger_state(profile_id, instance_id)
        updates: list[str] = []
        values: list[Any] = []
        if processed_through_message_id is not None:
            updates.append("processed_through_message_id = MAX(processed_through_message_id, ?)")
            values.append(max(0, int(processed_through_message_id)))
        for column, value in (
            ("frozen_through_message_id", frozen_through_message_id),
            ("enabled_at", enabled_at),
            ("last_success_at", last_success_at),
            ("cooldown_until", cooldown_until),
            ("active_task_id", active_task_id),
        ):
            if value is ...:
                continue
            updates.append(f"{column} = ?")
            values.append(_dt(value) if isinstance(value, datetime) else value)
        if last_error is not None:
            updates.append("last_error = ?")
            values.append(str(last_error))
        if updates:
            updates.extend(("version = version + 1", "updated_at = ?"))
            values.append(_dt(_now()))
            values.extend((profile_id, instance_id))
            await self.db.call(
                lambda conn: conn.execute(
                    f"""UPDATE sticker_trigger_states SET {", ".join(updates)}
                    WHERE profile_id = ? AND instance_id = ?""",
                    values,
                ),
                transaction=True,
            )
        row = await self.db.fetch_one(
            "SELECT * FROM sticker_trigger_states WHERE profile_id = ? AND instance_id = ?",
            (profile_id, instance_id),
        )
        assert row is not None
        return self._sticker_trigger_state(row)

    def _sticker_trigger_state(self, row: sqlite3.Row) -> dict[str, Any]:
        result = self._record(row, json_columns=())
        cooldown_until = _parse(result.get("cooldown_until"))
        if cooldown_until is not None and cooldown_until.tzinfo is None:
            raise RuntimeError("sticker cooldown timestamp must be timezone-aware")
        result["cooldown_until"] = cooldown_until
        return result

    async def complete_sticker_collection_task(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        *,
        succeeded: bool,
        frozen_through_message_id: int,
        error: str = "",
        now: datetime | None = None,
        deferred: bool = False,
    ) -> dict[str, Any]:
        current = now or _now()
        state = await self.get_sticker_trigger_state(profile_id, instance_id)
        if state.get("active_task_id") not in {None, int(task_id)}:
            raise ValueError("sticker collection task no longer owns trigger state")
        return await self.update_sticker_trigger_state(
            profile_id,
            instance_id,
            processed_through_message_id=(frozen_through_message_id if succeeded else None),
            frozen_through_message_id=None,
            last_success_at=(current if succeeded else ...),
            cooldown_until=(None if succeeded or deferred else current + timedelta(hours=6)),
            active_task_id=None,
            last_error="" if succeeded else str(error),
        )

    async def defer_sticker_collection_task(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        *,
        error: str = "",
    ) -> dict[str, Any]:
        state = await self.get_sticker_trigger_state(profile_id, instance_id)
        if state.get("active_task_id") not in {None, int(task_id)}:
            raise ValueError("sticker collection task no longer owns trigger state")
        return await self.update_sticker_trigger_state(
            profile_id,
            instance_id,
            frozen_through_message_id=None,
            cooldown_until=None,
            active_task_id=None,
            last_error=str(error),
        )

    async def wake_waiting_sticker_checks(
        self, profile_id: str, *, scope: str | None = None
    ) -> int:
        if scope is not None:
            self._validate_scope(scope)
        now = _dt(_now())
        scope_clause = ""
        values: list[Any] = [now, now, profile_id]
        if scope is not None:
            scope_clause = " AND i.scope = ?"
            values.append(scope)
        cursor = await self.db.call(
            lambda conn: conn.execute(
                f"""UPDATE sticker_candidates SET next_retry_at = ?, updated_at = ?
                WHERE profile_id = ? AND status = 'WAITING_CHECK' AND recoverable = 1
                  AND EXISTS (
                    SELECT 1 FROM character_instances i
                    WHERE i.profile_id = sticker_candidates.profile_id
                      AND i.instance_id = sticker_candidates.instance_id
                      {scope_clause}
                  )""",
                values,
            ),
            transaction=True,
        )
        return int(cursor.rowcount)
