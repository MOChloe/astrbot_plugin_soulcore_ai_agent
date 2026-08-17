from __future__ import annotations

from collections.abc import Sequence

from ....contracts.runtime_limits import (
    DEFAULT_RESPONSE_POLISH_TIMEOUT_SECONDS,
    require_response_polish_timeout_seconds,
)
from ....contracts.thinking import require_thinking_policy
from .support import (
    Any,
    RoleProfile,
    _dt,
    _now,
    sqlite3,
)


def _write_thinking_preset_to_scopes(
    conn: sqlite3.Connection,
    profile_id: str,
    *,
    max_context_tokens: int,
    target_context_tokens: int,
    now: str,
) -> None:
    row = conn.execute(
        "SELECT COUNT(*) FROM scope_configs WHERE profile_id = ?",
        (profile_id,),
    ).fetchone()
    if row is None or int(row[0]) != 2:
        raise RuntimeError(f"profile scope invariant is broken for {profile_id}")
    conn.execute(
        """UPDATE scope_configs
        SET max_context_tokens = ?, target_context_tokens = ?,
            version = version + 1, updated_at = ?
        WHERE profile_id = ?
          AND (max_context_tokens <> ? OR target_context_tokens <> ?)""",
        (
            int(max_context_tokens),
            int(target_context_tokens),
            now,
            profile_id,
            int(max_context_tokens),
            int(target_context_tokens),
        ),
    )


class ProfileAdministration:
    _PASSIVE_NO_REPLY_NOTICE_PREFERENCE_PREFIX = "profile.passive_no_reply_notice_enabled:"
    _RESPONSE_POLISH_PREFERENCE_PREFIX = "profile.response_polish_enabled:"
    _RESPONSE_POLISH_TIMEOUT_PREFERENCE_PREFIX = "profile.response_polish_timeout_seconds:"

    async def get_console_preference(self, key: str) -> str:
        row = await self.db.fetch_one(
            "SELECT preference_value FROM console_preferences WHERE preference_key = ?",
            (str(key),),
        )
        return str(row["preference_value"] or "") if row is not None else ""

    async def get_console_preferences(self, keys: Sequence[str]) -> dict[str, str]:
        """Read several page preferences while holding the DB connection once."""

        requested = tuple(dict.fromkeys(str(value) for value in keys if str(value)))
        if not requested:
            return {}

        def operation(conn: sqlite3.Connection) -> dict[str, str]:
            result: dict[str, str] = {}
            for start in range(0, len(requested), 500):
                chunk = requested[start : start + 500]
                placeholders = ", ".join("?" for _ in chunk)
                rows = conn.execute(
                    "SELECT preference_key, preference_value FROM console_preferences "
                    f"WHERE preference_key IN ({placeholders})",
                    chunk,
                )
                result.update(
                    {str(row["preference_key"]): str(row["preference_value"] or "") for row in rows}
                )
            return result

        return await self.db.call(operation)

    async def set_console_preference(self, key: str, value: str) -> None:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO console_preferences(preference_key, preference_value, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(preference_key) DO UPDATE SET
                       preference_value = excluded.preference_value,
                       updated_at = excluded.updated_at""",
                (str(key), str(value), now),
            )

        await self.uow.run(operation)

    async def sync_profiles(self, profiles: list[dict[str, Any]]) -> list[RoleProfile]:
        active: set[str] = set()
        result: list[RoleProfile] = []
        for item in profiles:
            profile_id = str(
                item.get("profile_id") or item.get("id") or item.get("name") or ""
            ).strip()
            if not profile_id:
                continue
            active.add(profile_id)
            role = await self.ensure_profile(profile_id)
            name = str(item.get("name") or profile_id)
            if role.name != name or role.orphaned:
                role = await self.update_profile(profile_id, name=name, orphaned=False)
            result.append(role)
        await self.mark_orphaned_except(active)
        return result

    async def get_profile_soulcore_enabled(self, profile_id: str) -> bool:
        """Return the authoritative profile-wide SoulCore master switch."""

        profile = await self.get_profile(str(profile_id))
        if profile is None:
            raise KeyError(profile_id)
        return bool(profile.enabled)

    async def set_profile_soulcore_enabled(self, profile_id: str, enabled: bool) -> RoleProfile:
        """Set the authoritative profile-wide SoulCore master switch."""

        return await self.update_profile(str(profile_id), enabled=bool(enabled))

    async def set_profile_quick_setup_decided(self, profile_id: str, decided: bool) -> RoleProfile:
        """Persist whether the administrator has made the first-run setup choice."""

        return await self.update_profile(str(profile_id), quick_setup_decided=bool(decided))

    async def set_profile_thinking_complexity(
        self,
        profile_id: str,
        complexity: str,
    ) -> RoleProfile:
        """Atomically apply one role-owned tier and its real scope budgets."""

        selected_profile = str(profile_id)
        policy = require_thinking_policy(complexity)
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> None:
            updated = conn.execute(
                """UPDATE role_profiles
                SET thinking_complexity = ?, updated_at = ?
                WHERE profile_id = ?""",
                (policy.complexity.value, now, selected_profile),
            ).rowcount
            if updated != 1:
                raise KeyError(selected_profile)
            _write_thinking_preset_to_scopes(
                conn,
                selected_profile,
                max_context_tokens=policy.max_context_tokens,
                target_context_tokens=policy.target_context_tokens,
                now=now,
            )

        await self.uow.run(operation)
        await self.db.publish_backup_after_commit()
        profile = await self.get_profile(selected_profile)
        if profile is None:
            raise KeyError(selected_profile)
        return profile

    async def finish_profile_quick_setup(
        self,
        profile_id: str,
        *,
        thinking_complexity: str,
    ) -> RoleProfile:
        """Atomically apply the selected preset, finish the guide, and enable the role."""

        selected_profile = str(profile_id)
        policy = require_thinking_policy(thinking_complexity)
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> None:
            exists = conn.execute(
                "SELECT 1 FROM role_profiles WHERE profile_id = ?",
                (selected_profile,),
            ).fetchone()
            if exists is None:
                raise KeyError(selected_profile)
            main_model = conn.execute(
                """SELECT m.backend_id, m.config_json
                FROM ai_capability_pools AS pool
                JOIN ai_backends AS backend USING(backend_id)
                JOIN ai_api_models AS m USING(backend_id)
                JOIN ai_api_packages AS package USING(package_id)
                WHERE pool.capability = 'CHAT.COMPLETION'
                  AND pool.enabled = 1 AND backend.enabled = 1
                  AND m.enabled = 1 AND m.archived_at IS NULL
                  AND package.enabled = 1 AND package.archived_at IS NULL
                  AND package.profile_id = ?
                ORDER BY pool.priority ASC, pool.backend_id ASC
                LIMIT 1""",
                (selected_profile,),
            ).fetchone()
            if main_model is None:
                raise ValueError("请先让主力模型通过真实连接测试，再完成快速设置")
            conn.execute(
                """UPDATE role_profiles
                SET enabled = 1, quick_setup_decided = 1,
                    thinking_complexity = ?, updated_at = ?
                WHERE profile_id = ?""",
                (policy.complexity.value, now, selected_profile),
            )
            _write_thinking_preset_to_scopes(
                conn,
                selected_profile,
                max_context_tokens=policy.max_context_tokens,
                target_context_tokens=policy.target_context_tokens,
                now=now,
            )

        await self.uow.run(operation)
        await self.db.publish_backup_after_commit()
        profile = await self.get_profile(selected_profile)
        if profile is None:
            raise KeyError(selected_profile)
        return profile

    async def get_profile_turn_buffer_enabled(self, profile_id: str) -> bool:
        """Return whether inbound messages wait for the player to finish."""

        profile = await self.get_profile(str(profile_id))
        if profile is None:
            raise KeyError(profile_id)
        return bool(profile.turn_buffer_enabled)

    async def set_profile_turn_buffer_enabled(self, profile_id: str, enabled: bool) -> RoleProfile:
        """Persist the profile-wide inbound turn-buffer feature switch."""

        return await self.update_profile(str(profile_id), turn_buffer_enabled=bool(enabled))

    async def get_profile_image_generation_enabled(self, profile_id: str) -> bool:
        """Return whether Main Core may use the drawing feature for a profile."""

        profile = await self.get_profile(str(profile_id))
        if profile is None:
            raise KeyError(profile_id)
        return bool(profile.image_generation_enabled)

    async def set_profile_image_generation_enabled(
        self, profile_id: str, enabled: bool
    ) -> RoleProfile:
        """Persist the profile-wide drawing feature switch."""

        return await self.update_profile(str(profile_id), image_generation_enabled=bool(enabled))

    @classmethod
    def _response_polish_preference_key(cls, profile_id: str) -> str:
        value = str(profile_id or "").strip()
        if not value:
            raise ValueError("profile_id cannot be empty")
        return f"{cls._RESPONSE_POLISH_PREFERENCE_PREFIX}{value}"

    async def get_profile_response_polish_enabled(self, profile_id: str) -> bool:
        """Return the explicit opt-in switch for final response polishing."""

        if await self.get_profile(str(profile_id)) is None:
            raise KeyError(profile_id)
        value = await self.get_console_preference(self._response_polish_preference_key(profile_id))
        if value not in {"", "0", "1"}:
            raise ValueError("invalid persisted response polish switch")
        return value == "1"

    async def set_profile_response_polish_enabled(
        self, profile_id: str, enabled: bool
    ) -> RoleProfile:
        """Persist the opt-in without changing the baseline database schema."""

        profile = await self.get_profile(str(profile_id))
        if profile is None:
            raise KeyError(profile_id)
        await self.set_console_preference(
            self._response_polish_preference_key(profile_id),
            "1" if enabled else "0",
        )
        return profile

    @classmethod
    def _response_polish_timeout_preference_key(cls, profile_id: str) -> str:
        value = str(profile_id or "").strip()
        if not value:
            raise ValueError("profile_id cannot be empty")
        return f"{cls._RESPONSE_POLISH_TIMEOUT_PREFERENCE_PREFIX}{value}"

    async def get_profile_response_polish_timeout_seconds(self, profile_id: str) -> int:
        """Return the configured whole-request polish deadline for one role."""

        if await self.get_profile(str(profile_id)) is None:
            raise KeyError(profile_id)
        value = await self.get_console_preference(
            self._response_polish_timeout_preference_key(profile_id)
        )
        if not value:
            return DEFAULT_RESPONSE_POLISH_TIMEOUT_SECONDS
        return require_response_polish_timeout_seconds(value)

    async def set_profile_response_polish_timeout_seconds(
        self,
        profile_id: str,
        timeout_seconds: int,
    ) -> RoleProfile:
        """Persist the role-owned deadline without adding a runtime-schema column."""

        profile = await self.get_profile(str(profile_id))
        if profile is None:
            raise KeyError(profile_id)
        timeout = require_response_polish_timeout_seconds(timeout_seconds)
        await self.set_console_preference(
            self._response_polish_timeout_preference_key(profile_id),
            str(timeout),
        )
        return profile

    @classmethod
    def _passive_no_reply_notice_preference_key(cls, profile_id: str) -> str:
        value = str(profile_id or "").strip()
        if not value:
            raise ValueError("profile_id cannot be empty")
        return f"{cls._PASSIVE_NO_REPLY_NOTICE_PREFERENCE_PREFIX}{value}"

    async def get_profile_passive_no_reply_notice_enabled(self, profile_id: str) -> bool:
        """Return whether passive turns should emit ephemeral no-reply feedback."""

        if await self.get_profile(str(profile_id)) is None:
            raise KeyError(profile_id)
        value = await self.get_console_preference(
            self._passive_no_reply_notice_preference_key(profile_id)
        )
        if value not in {"", "0", "1"}:
            raise ValueError("invalid persisted passive no-reply notice switch")
        return value == "1"

    async def set_profile_passive_no_reply_notice_enabled(
        self, profile_id: str, enabled: bool
    ) -> RoleProfile:
        """Persist the opt-in without adding a runtime-schema column."""

        profile = await self.get_profile(str(profile_id))
        if profile is None:
            raise KeyError(profile_id)
        await self.set_console_preference(
            self._passive_no_reply_notice_preference_key(profile_id),
            "1" if enabled else "0",
        )
        return profile
