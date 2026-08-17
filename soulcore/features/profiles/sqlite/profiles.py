from __future__ import annotations

from ....contracts.thinking import MainCoreThinkingPolicy, require_thinking_policy
from .support import (
    Any,
    Mapping,
    RoleProfile,
    ScopeConfig,
    WebSearchIntensity,
    _dt,
    _now,
    sqlite3,
)

_SCOPE_CONFIG_FIELDS = {
    "proactive_enabled",
    "extra_background",
    "world_texture_prompt",
    "media_original_retention_days",
    "min_wakeup_minutes",
    "max_wakeup_minutes",
    "low_frequency_min_wakeup_minutes",
    "low_frequency_max_wakeup_minutes",
    "max_context_tokens",
    "target_context_tokens",
}


def _validated_thinking_policy(value: object) -> MainCoreThinkingPolicy:
    try:
        return require_thinking_policy(value)
    except ValueError as exc:
        raise ValueError("unsupported thinking complexity") from exc


def _media_original_retention_days(value: object) -> int:
    days = int(value)
    if not 0 <= days <= 3650:
        raise ValueError("media_original_retention_days must be between 0 and 3650")
    return days


def _normalized_scope_patch(
    scope: str,
    patch: Mapping[str, Any],
    current: ScopeConfig,
) -> dict[str, Any]:
    values = {name: getattr(current, name) for name in _SCOPE_CONFIG_FIELDS}
    values.update(patch)
    if (
        int(values["min_wakeup_minutes"]) < 1
        or int(values["max_wakeup_minutes"]) < int(values["min_wakeup_minutes"])
        or int(values["low_frequency_min_wakeup_minutes"]) < 1
        or int(values["low_frequency_max_wakeup_minutes"])
        < int(values["low_frequency_min_wakeup_minutes"])
    ):
        raise ValueError("invalid scope wakeup interval")
    maximum = int(values["max_context_tokens"])
    target = int(values["target_context_tokens"])
    if maximum < 128000:
        raise ValueError("max_context_tokens must be at least 128000")
    if target < 20000:
        raise ValueError("target_context_tokens must be at least 20000")
    normalized = dict(patch)
    normalized["max_context_tokens"] = maximum
    normalized["target_context_tokens"] = min(target, maximum)
    normalized["media_original_retention_days"] = _media_original_retention_days(
        values["media_original_retention_days"]
    )
    return normalized


def _insert_profile_companion_rows(
    conn: sqlite3.Connection,
    profile: RoleProfile,
    thinking_policy: MainCoreThinkingPolicy,
    now: str,
) -> int:
    """Create the required one-to-one and per-scope rows with a new profile."""

    inserted = conn.execute(
        """INSERT INTO profile_runtime_settings(
            profile_id, created_at, updated_at
        ) VALUES (?, ?, ?)
        ON CONFLICT(profile_id) DO NOTHING""",
        (profile.profile_id, now, now),
    ).rowcount
    for scope in ("private", "group"):
        inserted += conn.execute(
            """INSERT INTO scope_configs(
                profile_id, scope, proactive_enabled,
                extra_background, min_wakeup_minutes,
                max_wakeup_minutes, low_frequency_min_wakeup_minutes,
                low_frequency_max_wakeup_minutes,
                max_context_tokens, target_context_tokens,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id, scope) DO NOTHING""",
            (
                profile.profile_id,
                scope,
                int(profile.proactive_enabled),
                profile.extra_background,
                profile.min_wakeup_minutes,
                profile.max_wakeup_minutes,
                profile.low_frequency_min_wakeup_minutes,
                profile.low_frequency_max_wakeup_minutes,
                thinking_policy.max_context_tokens,
                thinking_policy.target_context_tokens,
                now,
                now,
            ),
        ).rowcount
        inserted += conn.execute(
            """INSERT INTO contact_policies(
                profile_id, scope, proactive_enabled,
                check_min_minutes, check_max_minutes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id, scope) DO NOTHING""",
            (
                profile.profile_id,
                scope,
                int(profile.proactive_enabled),
                180,
                480,
                now,
                now,
            ),
        ).rowcount
        inserted += conn.execute(
            """INSERT INTO scope_delivery_policies(
                profile_id, scope, created_at, updated_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(profile_id, scope) DO NOTHING""",
            (profile.profile_id, scope, now, now),
        ).rowcount
        inserted += conn.execute(
            """INSERT INTO scope_state_gate_policies(
                profile_id, scope, created_at, updated_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(profile_id, scope) DO NOTHING""",
            (profile.profile_id, scope, now, now),
        ).rowcount
    return inserted


def _assert_profile_companion_rows(conn: sqlite3.Connection, profile_id: str) -> None:
    expected = {
        "profile_runtime_settings": 1,
        "scope_configs": 2,
        "contact_policies": 2,
        "scope_delivery_policies": 2,
        "scope_state_gate_policies": 2,
    }
    missing: list[str] = []
    for table, count in expected.items():
        row = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE profile_id = ?",
            (profile_id,),
        ).fetchone()
        if row is None or int(row[0]) != count:
            missing.append(table)
    if missing:
        raise RuntimeError(
            f"profile companion invariant is broken for {profile_id}: {', '.join(missing)}"
        )


class ProfileRecords:
    async def create_profile(self, profile: RoleProfile) -> RoleProfile:
        if not profile.profile_id.strip():
            raise ValueError("profile_id cannot be empty")
        thinking_policy = _validated_thinking_policy(profile.thinking_complexity)
        if (
            profile.min_wakeup_minutes < 1
            or profile.max_wakeup_minutes < profile.min_wakeup_minutes
            or profile.low_frequency_min_wakeup_minutes < 1
            or profile.low_frequency_max_wakeup_minutes < profile.low_frequency_min_wakeup_minutes
        ):
            raise ValueError("invalid wakeup interval")
        now_dt = _now()
        now = _dt(now_dt)

        def operation(conn: sqlite3.Connection) -> Mapping[str, Any]:
            conn.execute(
                """INSERT INTO role_profiles(
                    profile_id, name, enabled, quick_setup_decided, thinking_complexity,
                    background_life_enabled, background_life_version,
                    turn_buffer_enabled, image_generation_enabled,
                    file_artifacts_enabled,
                    web_search_enabled, web_search_intensity,
                    proactive_enabled, extra_background,
                    min_wakeup_minutes, max_wakeup_minutes,
                    low_frequency_min_wakeup_minutes, low_frequency_max_wakeup_minutes,
                    orphaned, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    profile.profile_id,
                    profile.name,
                    int(profile.enabled),
                    int(profile.quick_setup_decided),
                    str(profile.thinking_complexity),
                    int(profile.background_life_enabled),
                    int(profile.background_life_version),
                    int(profile.turn_buffer_enabled),
                    int(profile.image_generation_enabled),
                    int(profile.file_artifacts_enabled),
                    int(profile.web_search_enabled),
                    str(profile.web_search_intensity).upper(),
                    int(profile.proactive_enabled),
                    profile.extra_background,
                    profile.min_wakeup_minutes,
                    profile.max_wakeup_minutes,
                    profile.low_frequency_min_wakeup_minutes,
                    profile.low_frequency_max_wakeup_minutes,
                    int(profile.orphaned),
                    now,
                    now,
                ),
            )
            _insert_profile_companion_rows(conn, profile, thinking_policy, now)

        await self.uow.run(operation)
        return await self.get_profile(profile.profile_id)  # type: ignore[return-value]

    async def ensure_profile(self, profile_id: str, name: str = "") -> RoleProfile:
        existing = await self.get_profile(profile_id)
        if existing is None:
            try:
                return await self.create_profile(RoleProfile(profile_id=profile_id, name=name))
            except sqlite3.IntegrityError:
                existing = await self.get_profile(profile_id)
                if existing is None:
                    raise

        await self.uow.run(lambda conn: _assert_profile_companion_rows(conn, existing.profile_id))
        if name and existing.name != name:
            return await self.update_profile(profile_id, name=name)
        return existing

    async def get_profile(self, profile_id: str) -> RoleProfile | None:
        row = await self.db.fetch_one(
            "SELECT * FROM role_profiles WHERE profile_id = ?", (profile_id,)
        )
        return self._profile(row) if row else None

    async def get_scope_config(self, profile_id: str, scope: str) -> ScopeConfig | None:
        self._validate_scope(scope)
        row = await self.db.fetch_one(
            "SELECT * FROM scope_configs WHERE profile_id = ? AND scope = ?",
            (profile_id, scope),
        )
        return self._scope_config(row) if row else None

    async def list_scope_configs(self, profile_id: str) -> list[ScopeConfig]:
        rows = await self.db.fetch_all(
            """SELECT * FROM scope_configs WHERE profile_id = ?
            ORDER BY CASE scope WHEN 'private' THEN 0 ELSE 1 END""",
            (profile_id,),
        )
        return [self._scope_config(row) for row in rows]

    async def get_scope_config_version(self, profile_id: str, scope: str) -> int:
        self._validate_scope(scope)
        row = await self.db.fetch_one(
            "SELECT version FROM scope_configs WHERE profile_id = ? AND scope = ?",
            (profile_id, scope),
        )
        if row is None:
            raise KeyError((profile_id, scope))
        return int(row["version"])

    async def update_scope_config(
        self, profile_id: str, scope: str, patch: dict[str, Any]
    ) -> ScopeConfig:
        self._validate_scope(scope)
        unknown = set(patch) - _SCOPE_CONFIG_FIELDS
        if unknown:
            raise ValueError(f"unsupported scope config fields: {sorted(unknown)}")
        current = await self.get_scope_config(profile_id, scope)
        if current is None:
            raise KeyError((profile_id, scope))
        patch_to_write = _normalized_scope_patch(scope, patch, current)
        if patch:
            assignments: list[str] = []
            params: list[Any] = []
            for name, value in patch_to_write.items():
                assignments.append(f"{name} = ?")
                params.append(int(value) if name == "proactive_enabled" else value)
            assignments.append("version = version + 1")
            assignments.append("updated_at = ?")
            params.extend((_dt(_now()), profile_id, scope))
            await self.db.call(
                lambda conn: conn.execute(
                    f"UPDATE scope_configs SET {', '.join(assignments)} "
                    "WHERE profile_id = ? AND scope = ?",
                    params,
                ),
                transaction=True,
            )
            await self.db.publish_backup_after_commit()
        result = await self.get_scope_config(profile_id, scope)
        assert result is not None
        return result

    async def list_profiles(self, *, include_orphaned: bool = True) -> list[RoleProfile]:
        sql = "SELECT * FROM role_profiles"
        if not include_orphaned:
            sql += " WHERE orphaned = 0"
        sql += " ORDER BY profile_id"
        return [self._profile(row) for row in await self.db.fetch_all(sql)]

    async def update_profile(
        self,
        profile_id: str,
        changes: Mapping[str, Any] | None = None,
        **named_changes: Any,
    ) -> RoleProfile:
        changes = {**dict(changes or {}), **named_changes}
        allowed = {
            "name",
            "enabled",
            "quick_setup_decided",
            "background_life_enabled",
            "background_life_version",
            "turn_buffer_enabled",
            "image_generation_enabled",
            "file_artifacts_enabled",
            "web_search_enabled",
            "web_search_intensity",
            "proactive_enabled",
            "extra_background",
            "min_wakeup_minutes",
            "max_wakeup_minutes",
            "low_frequency_min_wakeup_minutes",
            "low_frequency_max_wakeup_minutes",
            "orphaned",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unsupported profile fields: {sorted(unknown)}")
        current = await self.get_profile(profile_id)
        if current is None:
            raise KeyError(profile_id)
        values = {name: getattr(current, name) for name in allowed}
        values.update(changes)
        intensity = str(values["web_search_intensity"] or "").strip().upper()
        if intensity not in {item.value for item in WebSearchIntensity}:
            raise ValueError("unsupported web search intensity")
        values["web_search_intensity"] = intensity
        if "web_search_intensity" in changes:
            changes["web_search_intensity"] = intensity
        if not self._valid_profile_intervals(values):
            raise ValueError("invalid wakeup interval")
        if not changes:
            return current
        assignments, parameters = self._profile_update_values(changes)
        assignments.append("updated_at = ?")
        parameters.extend((_dt(_now()), profile_id))
        await self.db.call(
            lambda conn: conn.execute(
                f"UPDATE role_profiles SET {', '.join(assignments)} WHERE profile_id = ?",
                parameters,
            ),
            transaction=True,
        )
        await self.db.publish_backup_after_commit()
        return await self.get_profile(profile_id)  # type: ignore[return-value]

    @staticmethod
    def _valid_profile_intervals(values: Mapping[str, Any]) -> bool:
        return all(
            (
                values["min_wakeup_minutes"] >= 1,
                values["max_wakeup_minutes"] >= values["min_wakeup_minutes"],
                values["low_frequency_min_wakeup_minutes"] >= 1,
                values["low_frequency_max_wakeup_minutes"]
                >= values["low_frequency_min_wakeup_minutes"],
            )
        )

    @staticmethod
    def _profile_update_values(changes: Mapping[str, Any]) -> tuple[list[str], list[Any]]:
        boolean_fields = {
            "enabled",
            "quick_setup_decided",
            "background_life_enabled",
            "turn_buffer_enabled",
            "image_generation_enabled",
            "file_artifacts_enabled",
            "web_search_enabled",
            "proactive_enabled",
            "orphaned",
        }
        assignments = [f"{key} = ?" for key in changes]
        parameters = [
            int(value) if key in boolean_fields else value for key, value in changes.items()
        ]
        return assignments, parameters

    async def mark_orphaned_except(self, active_profile_ids: set[str]) -> int:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            if active_profile_ids:
                placeholders = ",".join("?" for _ in active_profile_ids)
                cursor = conn.execute(
                    f"UPDATE role_profiles SET orphaned = 1, updated_at = ? "
                    f"WHERE profile_id NOT IN ({placeholders}) AND orphaned = 0",
                    (now, *sorted(active_profile_ids)),
                )
                conn.execute(
                    f"UPDATE role_profiles SET orphaned = 0, updated_at = ? "
                    f"WHERE profile_id IN ({placeholders}) AND orphaned = 1",
                    (now, *sorted(active_profile_ids)),
                )
            else:
                cursor = conn.execute(
                    "UPDATE role_profiles SET orphaned = 1, updated_at = ? WHERE orphaned = 0",
                    (now,),
                )
            return cursor.rowcount

        return await self.uow.run(operation)
