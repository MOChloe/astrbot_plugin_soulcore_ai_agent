from __future__ import annotations

from ....contracts.routes import COLD_BOOT_PERSISTABLE_MESSAGE_TYPES
from ....storage.sqlite.instance_runtime import seed_instance_runtime_rows
from .support import (
    Any,
    CharacterInstance,
    CoreState,
    InstanceInitializationState,
    Mapping,
    RouteReadiness,
    ScopeConfig,
    _dt,
    _now,
    sqlite3,
    stable_instance_id,
)


class InstanceRecords:
    async def reset_instance_readiness(self, profile_id: str) -> int:
        persistable = tuple(COLD_BOOT_PERSISTABLE_MESSAGE_TYPES)
        placeholders = ", ".join("?" for _ in persistable)
        cursor = await self.db.call(
            lambda conn: conn.execute(
                f"""UPDATE character_instances SET readiness = CASE
                    WHEN message_type IN ({placeholders}) THEN ? ELSE ? END,
                    updated_at = ? WHERE profile_id = ?""",
                (
                    *persistable,
                    RouteReadiness.READY.value,
                    RouteReadiness.ROUTE_NOT_READY.value,
                    _dt(_now()),
                    profile_id,
                ),
            ),
            transaction=True,
        )
        return int(cursor.rowcount)

    async def mark_instance_ready(self, profile_id: str, instance_id: str, ready: bool) -> bool:
        status = RouteReadiness.READY if ready else RouteReadiness.ROUTE_NOT_READY
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE character_instances SET readiness = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?""",
                (status.value, _dt(_now()), profile_id, instance_id),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1

    async def ensure_character_instance(
        self,
        profile_id: str,
        umo: str,
        platform_id: str = "",
        message_type: str = "",
        target_id: str = "",
        session_kind: str = "",
        ready: bool = True,
    ) -> CharacterInstance:
        """Create a blank conversation character from template defaults once."""

        role = await self.get_profile(profile_id)
        if role is None:
            raise KeyError(profile_id)
        raw = str(umo or "").strip()
        if not raw:
            raise ValueError("umo cannot be empty")
        if not all((platform_id, message_type, target_id)):
            platform_id, message_type, target_id = self._parse_umo(raw)
        scope = "group" if ("Group" in message_type or "Guild" in message_type) else "private"
        template = await self.get_scope_config(profile_id, scope)
        if template is None:
            raise KeyError((profile_id, scope))
        instance_id = stable_instance_id(
            raw,
            platform_id=platform_id,
            message_type=message_type,
            target_id=target_id,
        )
        now = _dt(_now())
        context = {
            "profile_id": profile_id,
            "instance_id": instance_id,
            "raw": raw,
            "platform_id": platform_id,
            "message_type": message_type,
            "target_id": target_id,
            "scope": scope,
            "session_kind": session_kind,
            "readiness": (
                RouteReadiness.READY.value if ready else RouteReadiness.ROUTE_NOT_READY.value
            ),
            "now": now,
        }
        await self.db.call(
            lambda conn: self._ensure_character_instance_sql(conn, context, template),
            transaction=True,
        )
        await self.db.publish_backup_after_commit()
        result = await self.get_character_instance(profile_id, instance_id)
        assert result is not None
        return result

    @staticmethod
    def _ensure_character_instance_sql(
        conn: sqlite3.Connection,
        context: Mapping[str, Any],
        template: ScopeConfig,
    ) -> None:
        profile_id = context["profile_id"]
        instance_id = context["instance_id"]
        now = context["now"]
        conn.execute(
            """INSERT INTO character_instances(
                    profile_id, instance_id, route_umo, platform_id, message_type,
                    target_id, scope, session_kind, readiness, initialization_state,
                    proactive_enabled, extra_background,
                    min_wakeup_minutes, max_wakeup_minutes,
                    low_frequency_min_wakeup_minutes,
                    low_frequency_max_wakeup_minutes,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id, instance_id) DO UPDATE SET
                    route_umo = excluded.route_umo,
                    platform_id = excluded.platform_id,
                    message_type = excluded.message_type,
                    target_id = excluded.target_id,
                    scope = excluded.scope,
                    session_kind = excluded.session_kind,
                    readiness = excluded.readiness,
                    updated_at = excluded.updated_at""",
            (
                profile_id,
                instance_id,
                context["raw"],
                context["platform_id"],
                context["message_type"],
                context["target_id"],
                context["scope"],
                context["session_kind"],
                context["readiness"],
                InstanceInitializationState.UNINITIALIZED.value,
                int(template.proactive_enabled),
                template.extra_background,
                template.min_wakeup_minutes,
                template.max_wakeup_minutes,
                template.low_frequency_min_wakeup_minutes,
                template.low_frequency_max_wakeup_minutes,
                now,
                now,
            ),
        )
        conn.execute(
            """INSERT OR IGNORE INTO instance_core_state(
                    profile_id, instance_id, updated_at
                ) VALUES (?, ?, ?)""",
            (profile_id, instance_id, now),
        )
        seed_instance_runtime_rows(conn, profile_id, instance_id, now)

    async def get_character_instance(
        self, profile_id: str, instance_id: str
    ) -> CharacterInstance | None:
        row = await self.db.fetch_one(
            "SELECT * FROM character_instances WHERE profile_id = ? AND instance_id = ?",
            (profile_id, instance_id),
        )
        return self._character_instance(row) if row else None

    async def list_character_instances(
        self, profile_id: str, scope: str | None = None
    ) -> list[CharacterInstance]:
        sql = "SELECT * FROM character_instances WHERE profile_id = ?"
        params: list[Any] = [profile_id]
        if scope is not None:
            if scope not in {"private", "group"}:
                raise ValueError("scope must be 'private' or 'group'")
            sql += " AND scope = ?"
            params.append(scope)
        sql += " ORDER BY updated_at DESC, instance_id"
        return [self._character_instance(row) for row in await self.db.fetch_all(sql, params)]

    async def update_character_instance(
        self, profile_id: str, instance_id: str, patch: dict[str, Any]
    ) -> CharacterInstance:
        allowed = {
            "proactive_enabled",
            "extra_background",
            "min_wakeup_minutes",
            "max_wakeup_minutes",
            "low_frequency_min_wakeup_minutes",
            "low_frequency_max_wakeup_minutes",
        }
        unknown = set(patch) - allowed
        if unknown:
            raise ValueError(f"unsupported instance fields: {sorted(unknown)}")
        current = await self.get_character_instance(profile_id, instance_id)
        if current is None:
            raise KeyError((profile_id, instance_id))
        values = {name: getattr(current, name) for name in allowed}
        values.update(patch)
        if (
            int(values["min_wakeup_minutes"]) < 1
            or int(values["max_wakeup_minutes"]) < int(values["min_wakeup_minutes"])
            or int(values["low_frequency_min_wakeup_minutes"]) < 1
            or int(values["low_frequency_max_wakeup_minutes"])
            < int(values["low_frequency_min_wakeup_minutes"])
        ):
            raise ValueError("invalid instance wakeup interval")
        assignments = []
        params: list[Any] = []
        for name, value in patch.items():
            assignments.append(f"{name} = ?")
            params.append(int(value) if name == "proactive_enabled" else value)
        if assignments:
            assignments.append("updated_at = ?")
            params.extend((_dt(_now()), profile_id, instance_id))
            await self.db.call(
                lambda conn: conn.execute(
                    f"UPDATE character_instances SET {', '.join(assignments)} "
                    "WHERE profile_id = ? AND instance_id = ?",
                    params,
                ),
                transaction=True,
            )
            await self.db.publish_backup_after_commit()
        result = await self.get_character_instance(profile_id, instance_id)
        assert result is not None
        return result

    async def get_instance_state(self, profile_id: str, instance_id: str) -> CoreState:
        row = await self.db.fetch_one(
            "SELECT * FROM instance_core_state WHERE profile_id = ? AND instance_id = ?",
            (profile_id, instance_id),
        )
        if row is None:
            raise KeyError((profile_id, instance_id))
        return self._state(row)

    async def compare_and_swap_instance_state(
        self,
        profile_id: str,
        instance_id: str,
        expected_state_epoch: int,
    ) -> CoreState | None:
        now = _dt(_now())
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE instance_core_state SET
                    state_epoch = state_epoch + 1,
                    updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND state_epoch = ?""",
                (
                    now,
                    profile_id,
                    instance_id,
                    expected_state_epoch,
                ),
            ),
            transaction=True,
        )
        return (
            await self.get_instance_state(profile_id, instance_id) if cursor.rowcount == 1 else None
        )

    async def bump_instance_activity(self, profile_id: str, instance_id: str) -> int:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                """UPDATE instance_core_state SET activity_epoch = activity_epoch + 1,
                    low_frequency_mode = 0, low_frequency_reason = '',
                    low_frequency_since = NULL, updated_at = ?
                    WHERE profile_id = ? AND instance_id = ?""",
                (now, profile_id, instance_id),
            )
            if cursor.rowcount != 1:
                raise KeyError((profile_id, instance_id))
            return int(
                conn.execute(
                    "SELECT activity_epoch FROM instance_core_state "
                    "WHERE profile_id = ? AND instance_id = ?",
                    (profile_id, instance_id),
                ).fetchone()[0]
            )

        return await self.uow.run(operation)
