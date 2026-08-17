"""Versioned management operations for the minimal background runtime."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ....storage.sqlite.background_projection import ensure_background_instance_sql
from ....storage.sqlite.codec import _dt, _load, _now, _parse
from ....storage.sqlite.foreground_activity import foreground_activity_is_active_sql
from ..domain import AUTHOR_ORDER, BackgroundAuthorKind

SetBackgroundEnabled = Callable[..., int]


class QuickSetupLifeMixin:
    db: Any
    uow: Any

    async def quick_setup_life_snapshot(self, profile_id: str) -> dict[str, Any]:
        return dict(
            await self.db.call(lambda conn: quick_setup_life_snapshot_sql(conn, profile_id))
        )

    async def quick_setup_configure_life(
        self,
        profile_id: str,
        *,
        enabled: bool,
        initial_direction: str,
        expected_version: int,
        expected_world_revision: int,
    ) -> dict[str, Any]:
        direction = str(initial_direction or "").strip()
        if len(direction) > 500:
            raise ValueError("开始时的想法最多填写 500 个字")
        now = _dt(_now())
        applied = bool(
            await self.uow.run(
                lambda conn: quick_setup_configure_life_sql(
                    conn,
                    profile_id,
                    enabled=enabled,
                    direction=direction,
                    expected_version=expected_version,
                    expected_world_revision=expected_world_revision,
                    now=now,
                    set_background_enabled=self._set_background_enabled_for_quick_setup,
                )
            )
        )
        if applied:
            await self.db.publish_backup_after_commit()
        return {
            "ok": True,
            "applied": applied,
            "life": await self.quick_setup_life_snapshot(profile_id),
        }

    @staticmethod
    def _set_background_enabled_for_quick_setup(
        conn: sqlite3.Connection,
        instance: sqlite3.Row,
        **values: Any,
    ) -> int:
        raise NotImplementedError


def quick_setup_life_snapshot_sql(
    conn: sqlite3.Connection,
    profile_id: str,
) -> dict[str, Any]:
    profile = conn.execute(
        """SELECT background_life_enabled, background_life_version
        FROM role_profiles WHERE profile_id = ?""",
        (profile_id,),
    ).fetchone()
    if profile is None:
        raise KeyError(profile_id)
    world = conn.execute(
        """SELECT revision, life_direction FROM world_definitions
        WHERE profile_id = ?""",
        (profile_id,),
    ).fetchone()
    default_enabled = bool(profile["background_life_enabled"])
    instances = conn.execute(
        """SELECT COUNT(*) AS total,
            COALESCE(SUM(CASE WHEN COALESCE(background.enabled, ?) = 1 THEN 1 ELSE 0 END), 0)
                AS enabled_total,
            COALESCE(SUM(CASE
                WHEN background.instance_id IS NOT NULL AND background.enabled != ?
                THEN 1 ELSE 0 END), 0)
                AS difference_total
        FROM character_instances instance
        LEFT JOIN background_instances background
          ON background.profile_id = instance.profile_id
         AND background.instance_id = instance.instance_id
        WHERE instance.profile_id = ?""",
        (int(default_enabled), int(default_enabled), profile_id),
    ).fetchone()
    started = conn.execute(
        """SELECT EXISTS(
            SELECT 1 FROM background_instances
            WHERE profile_id = ? AND initialization_state != 'UNINITIALIZED'
            UNION ALL
            SELECT 1 FROM background_story_sources WHERE profile_id = ?
            UNION ALL
            SELECT 1 FROM background_role_timeline_events WHERE profile_id = ?
            UNION ALL
            SELECT 1 FROM background_author_publications WHERE profile_id = ?
        ) AS value""",
        (profile_id, profile_id, profile_id, profile_id),
    ).fetchone()
    return {
        "enabled": default_enabled,
        "version": int(profile["background_life_version"]),
        "initial_direction": str(world["life_direction"] or "") if world is not None else "",
        "world_revision": int(world["revision"]) if world is not None else 0,
        "total_instances": int(instances["total"] or 0),
        "enabled_instances": int(instances["enabled_total"] or 0),
        "mixed": bool(instances["difference_total"]),
        "already_started": bool(started["value"]),
    }


def quick_setup_configure_life_sql(
    conn: sqlite3.Connection,
    profile_id: str,
    *,
    enabled: bool,
    direction: str,
    expected_version: int,
    expected_world_revision: int,
    now: str,
    set_background_enabled: SetBackgroundEnabled,
) -> bool:
    profile, world, character_rows, instance_rows = _quick_setup_life_rows(conn, profile_id)
    current_direction = str(world["life_direction"] or "") if world is not None else ""
    desired_direction = direction if enabled else current_direction
    current_world_revision = int(world["revision"]) if world is not None else 0
    if _quick_setup_life_is_current(
        profile,
        character_rows,
        instance_rows,
        enabled=enabled,
        current_direction=current_direction,
        desired_direction=desired_direction,
    ):
        return False
    _require_quick_setup_life_versions(
        profile,
        current_world_revision=current_world_revision,
        expected_version=expected_version,
        expected_world_revision=expected_world_revision,
    )
    _save_quick_setup_life_direction_sql(
        conn,
        profile_id,
        world,
        current_direction=current_direction,
        desired_direction=desired_direction,
        current_revision=current_world_revision,
        now=now,
    )
    _configure_life_instances(
        conn,
        profile_id,
        character_rows,
        enabled=enabled,
        direction=desired_direction,
        now=now,
        set_background_enabled=set_background_enabled,
    )
    _update_profile_life_default(conn, profile_id, enabled, expected_version, now)
    return True


def _quick_setup_life_rows(
    conn: sqlite3.Connection, profile_id: str
) -> tuple[sqlite3.Row, sqlite3.Row | None, list[sqlite3.Row], list[sqlite3.Row]]:
    profile = conn.execute(
        """SELECT background_life_enabled, background_life_version
        FROM role_profiles WHERE profile_id = ?""",
        (profile_id,),
    ).fetchone()
    if profile is None:
        raise KeyError(profile_id)
    world = conn.execute(
        """SELECT revision, life_direction FROM world_definitions
        WHERE profile_id = ?""",
        (profile_id,),
    ).fetchone()
    character_rows = conn.execute(
        """SELECT instance_id FROM character_instances
        WHERE profile_id = ? ORDER BY instance_id""",
        (profile_id,),
    ).fetchall()
    instance_rows = conn.execute(
        """SELECT * FROM background_instances
        WHERE profile_id = ? ORDER BY instance_id""",
        (profile_id,),
    ).fetchall()
    return profile, world, character_rows, instance_rows


def _quick_setup_life_is_current(
    profile: sqlite3.Row,
    character_rows: Sequence[sqlite3.Row],
    instance_rows: Sequence[sqlite3.Row],
    *,
    enabled: bool,
    current_direction: str,
    desired_direction: str,
) -> bool:
    return (
        bool(profile["background_life_enabled"]) == bool(enabled)
        and current_direction == desired_direction
        and _life_instances_match(
            character_rows,
            instance_rows,
            enabled=enabled,
            direction=desired_direction,
        )
    )


def _require_quick_setup_life_versions(
    profile: sqlite3.Row,
    *,
    current_world_revision: int,
    expected_version: int,
    expected_world_revision: int,
) -> None:
    if int(profile["background_life_version"]) != int(expected_version):
        raise ValueError("角色生活设置已变化，请刷新后重试")
    if current_world_revision != int(expected_world_revision):
        raise ValueError("角色生活设置已变化，请刷新后重试")


def _configure_life_instances(
    conn: sqlite3.Connection,
    profile_id: str,
    character_rows: Sequence[sqlite3.Row],
    *,
    enabled: bool,
    direction: str,
    now: str,
    set_background_enabled: SetBackgroundEnabled,
) -> None:
    for character in character_rows:
        ensure_background_instance_sql(conn, profile_id, str(character["instance_id"]), now)
    instances = conn.execute(
        """SELECT * FROM background_instances
        WHERE profile_id = ? ORDER BY instance_id""",
        (profile_id,),
    ).fetchall()
    for instance in instances:
        set_background_enabled(
            conn,
            instance,
            enabled=enabled,
            now=now,
            reason="quick_setup_life_changed",
            conflict_message="角色生活设置已变化，请刷新后重试",
        )
    _refresh_uninitialized_life_seeds_sql(
        conn,
        profile_id,
        enabled=enabled,
        direction=direction,
        now=now,
    )


def _update_profile_life_default(
    conn: sqlite3.Connection,
    profile_id: str,
    enabled: bool,
    expected_version: int,
    now: str,
) -> None:
    cursor = conn.execute(
        """UPDATE role_profiles
        SET background_life_enabled = ?,
            background_life_version = background_life_version + 1,
            updated_at = ?
        WHERE profile_id = ? AND background_life_version = ?""",
        (int(enabled), now, profile_id, int(expected_version)),
    )
    if cursor.rowcount != 1:
        raise ValueError("角色生活设置已变化，请刷新后重试")


def _life_instances_match(
    character_rows: Sequence[sqlite3.Row],
    instance_rows: Sequence[sqlite3.Row],
    *,
    enabled: bool,
    direction: str,
) -> bool:
    by_instance = {str(row["instance_id"]): row for row in instance_rows}
    if len(by_instance) != len(character_rows):
        return False
    for character in character_rows:
        instance = by_instance.get(str(character["instance_id"]))
        if instance is None or bool(instance["enabled"]) != bool(enabled):
            return False
        seed_changed = (
            enabled
            and str(instance["initialization_state"]) == "UNINITIALIZED"
            and str(instance["initial_life_direction"] or "") != direction
        )
        if seed_changed:
            return False
    return True


def _save_quick_setup_life_direction_sql(
    conn: sqlite3.Connection,
    profile_id: str,
    world: sqlite3.Row | None,
    *,
    current_direction: str,
    desired_direction: str,
    current_revision: int,
    now: str,
) -> None:
    if desired_direction == current_direction:
        return
    if world is None:
        conn.execute(
            """INSERT INTO world_definitions(
                profile_id, revision, life_direction, created_at, updated_at
            ) VALUES (?, 1, ?, ?, ?)""",
            (profile_id, desired_direction, now, now),
        )
        return
    cursor = conn.execute(
        """UPDATE world_definitions
        SET life_direction = ?, revision = revision + 1, updated_at = ?
        WHERE profile_id = ? AND revision = ?""",
        (desired_direction, now, profile_id, current_revision),
    )
    if cursor.rowcount != 1:
        raise ValueError("角色生活设置已变化，请刷新后重试")


def _refresh_uninitialized_life_seeds_sql(
    conn: sqlite3.Connection,
    profile_id: str,
    *,
    enabled: bool,
    direction: str,
    now: str,
) -> None:
    if not enabled:
        return
    seed_cursor = conn.execute(
        """UPDATE background_instances
        SET initial_life_direction = ?, config_version = config_version + 1,
            updated_at = ?
        WHERE profile_id = ? AND initialization_state = 'UNINITIALIZED'
          AND initial_life_direction != ?""",
        (direction, now, profile_id, direction),
    )
    if not seed_cursor.rowcount:
        return
    conn.execute(
        """UPDATE background_author_states
        SET next_due_at = ?, hard_due_at = ?,
            schedule_version = schedule_version + 1, updated_at = ?
        WHERE profile_id = ? AND author_kind = 'WORLD'
          AND EXISTS(
            SELECT 1 FROM background_instances instance
            WHERE instance.profile_id = background_author_states.profile_id
              AND instance.instance_id = background_author_states.instance_id
              AND instance.enabled = 1
              AND instance.initialization_state = 'UNINITIALIZED'
          )""",
        (now, now, now, profile_id),
    )


_CONFIG_FIELDS = frozenset(
    {
        "default_backend_id",
        "initial_life_direction",
        "proactive_frame_prewarm_enabled",
        "ordinary_min_minutes",
        "ordinary_max_minutes",
        "keyframe_every_ordinary",
        "keyframe_max_minutes",
        "story_source_min_minutes",
        "story_source_max_minutes",
        "life_direction_min_minutes",
        "life_direction_max_minutes",
        "world_min_minutes",
        "world_max_minutes",
    }
)
_BOOLEAN_CONFIG_FIELDS = frozenset({"proactive_frame_prewarm_enabled"})
_INTEGER_CONFIG_FIELDS = (
    _CONFIG_FIELDS
    - _BOOLEAN_CONFIG_FIELDS
    - {
        "default_backend_id",
        "initial_life_direction",
    }
)
_TERMINAL_TASK_STATUSES = frozenset({"DEFERRED", "SUCCEEDED", "FAILED", "CANCELLED"})


def _json_mapping(value: object) -> dict[str, Any]:
    loaded = _load(value) if isinstance(value, str) else value
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _changed_config_values(
    instance: sqlite3.Row,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        name: value
        for name, value in values.items()
        if (
            bool(instance[name]) != bool(value)
            if name in _BOOLEAN_CONFIG_FIELDS
            else int(instance[name]) != int(value)
            if name in _INTEGER_CONFIG_FIELDS
            else str(instance[name] or "") != str(value)
        )
    }


def _changed_backend_overrides(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    overrides: Mapping[str, str],
) -> dict[str, str]:
    changed: dict[str, str] = {}
    for kind, backend_id in overrides.items():
        state = conn.execute(
            """SELECT backend_id FROM background_author_states
            WHERE profile_id = ? AND instance_id = ? AND author_kind = ?""",
            (profile_id, instance_id, kind),
        ).fetchone()
        if state is None:
            raise KeyError((profile_id, instance_id, kind))
        if str(state["backend_id"] or "") != backend_id:
            changed[kind] = backend_id
    return changed


def _cancel_tasks_sql(
    conn: sqlite3.Connection,
    task_ids: Sequence[int],
    *,
    reason: str,
    now: str,
) -> None:
    ids = tuple(dict.fromkeys(int(value) for value in task_ids))
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""SELECT task_id, status FROM ai_tasks
        WHERE task_id IN ({placeholders})""",
        ids,
    ).fetchall()
    for row in rows:
        status = str(row["status"])
        if status in _TERMINAL_TASK_STATUSES:
            continue
        target = (
            "CANCEL_REQUESTED"
            if status in {"RUNNING", "PAUSE_REQUESTED", "CANCEL_REQUESTED"}
            else "CANCELLED"
        )
        conn.execute(
            """UPDATE ai_tasks
            SET status = ?, last_error = ?, finished_at = CASE
                    WHEN ? = 'CANCELLED' THEN ? ELSE finished_at END,
                lease_owner = CASE
                    WHEN ? = 'CANCELLED' THEN NULL ELSE lease_owner END,
                lease_until = CASE
                    WHEN ? = 'CANCELLED' THEN NULL ELSE lease_until END,
                updated_at = ?, version = version + 1
            WHERE task_id = ? AND status = ?""",
            (
                target,
                reason,
                target,
                now,
                target,
                target,
                now,
                int(row["task_id"]),
                status,
            ),
        )


def _set_background_enabled_sql(
    conn: sqlite3.Connection,
    instance: sqlite3.Row,
    *,
    enabled: bool,
    now: str,
    reason: str,
    conflict_message: str = "背景设置已变化，请刷新后重试",
) -> int:
    """Apply one enable transition while preserving all generated life content."""

    profile_id = str(instance["profile_id"])
    instance_id = str(instance["instance_id"])
    current_version = int(instance["config_version"])
    if bool(instance["enabled"]) == bool(enabled):
        return current_version
    rows = conn.execute(
        """SELECT active_task_id FROM background_author_states
        WHERE profile_id = ? AND instance_id = ? AND active_task_id IS NOT NULL""",
        (profile_id, instance_id),
    ).fetchall()
    _cancel_tasks_sql(
        conn,
        tuple(int(row["active_task_id"]) for row in rows),
        reason=reason,
        now=now,
    )
    conn.execute(
        """UPDATE background_author_states
        SET status = 'IDLE', active_task_id = NULL,
            generation = generation + 1,
            schedule_version = schedule_version + 1,
            last_error = '', updated_at = ?
        WHERE profile_id = ? AND instance_id = ?""",
        (now, profile_id, instance_id),
    )
    if enabled:
        opening = conn.execute(
            """SELECT keyframe_completed
            FROM background_initialization_openings
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        ).fetchone()
        target = (
            "ORDINARY"
            if str(instance["initialization_step"]) == "READY"
            else (
                (
                    "ORDINARY"
                    if opening is not None and bool(opening["keyframe_completed"])
                    else "KEYFRAME"
                )
                if str(instance["initialization_step"]) == "ORDINARY_CURRENT"
                else str(instance["initialization_step"])
            )
        )
        conn.execute(
            """UPDATE background_author_states
            SET next_due_at = ?, hard_due_at = ?,
                schedule_version = schedule_version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND author_kind = ?""",
            (now, now, now, profile_id, instance_id, target),
        )
    cursor = conn.execute(
        """UPDATE background_instances
        SET enabled = ?, config_version = config_version + 1,
            disabled_at = CASE WHEN ? = 0 THEN ? ELSE disabled_at END,
            resumed_at = CASE WHEN ? = 1 THEN ? ELSE resumed_at END,
            updated_at = ?
        WHERE profile_id = ? AND instance_id = ? AND config_version = ?""",
        (
            int(enabled),
            int(enabled),
            now,
            int(enabled),
            now,
            now,
            profile_id,
            instance_id,
            current_version,
        ),
    )
    if cursor.rowcount != 1:
        raise ValueError(conflict_message)
    return current_version + 1


def _clear_background_generated_state_sql(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    *,
    now: str,
) -> None:
    conn.execute(
        """DELETE FROM background_story_sources
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    )
    conn.execute(
        """DELETE FROM background_role_timeline_events
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    )
    conn.execute(
        """UPDATE background_role_current_views
        SET revision = revision + 1, narrative_time = '', location = '', doing = '',
            body_state = '', mood = '', intention = '',
            current_concern = '', as_of = ?,
            source = 'INITIALIZATION', source_event_id = NULL,
            source_publication_id = NULL, updated_at = ?
        WHERE profile_id = ? AND instance_id = ?""",
        (now, now, profile_id, instance_id),
    )
    conn.execute(
        """DELETE FROM background_author_publications
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    )
    conn.execute(
        """UPDATE background_author_states
        SET state_version = state_version + 1,
            schedule_version = schedule_version + 1,
            state_json = '{}',
            status = 'IDLE',
            next_due_at = CASE WHEN author_kind = 'WORLD' THEN ? ELSE NULL END,
            hard_due_at = CASE WHEN author_kind = 'WORLD' THEN ? ELSE NULL END,
            last_started_at = NULL, last_success_at = NULL,
            last_publication_id = NULL, active_task_id = NULL,
            generation = generation + 1, failure_count = 0,
            last_error = '', force_generation = force_generation + 1,
            updated_at = ?
        WHERE profile_id = ? AND instance_id = ?""",
        (now, now, now, profile_id, instance_id),
    )


def invalidate_profile_seed_sql(
    conn: sqlite3.Connection,
    profile_id: str,
    now: str,
) -> int:
    """Fence old world drafts inside the seed mutation transaction."""

    rows = conn.execute(
        """SELECT active_task_id FROM background_author_states
        WHERE profile_id = ? AND active_task_id IS NOT NULL""",
        (profile_id,),
    ).fetchall()
    _cancel_tasks_sql(
        conn,
        tuple(int(row["active_task_id"]) for row in rows),
        reason="background_world_seed_changed",
        now=now,
    )
    changed = conn.execute(
        """UPDATE background_instances
        SET config_version = config_version + 1, updated_at = ?
        WHERE profile_id = ?""",
        (now, profile_id),
    ).rowcount
    conn.execute(
        """UPDATE background_author_states
        SET status = 'IDLE', active_task_id = NULL,
            schedule_version = schedule_version + 1,
            last_error = '', updated_at = ?
        WHERE profile_id = ? AND active_task_id IS NOT NULL""",
        (now, profile_id),
    )
    conn.execute(
        """UPDATE background_author_states
        SET status = 'IDLE', active_task_id = NULL,
            generation = generation + 1,
            next_due_at = ?, hard_due_at = ?,
            schedule_version = schedule_version + 1,
            last_error = '', updated_at = ?
        WHERE profile_id = ? AND author_kind = 'WORLD'""",
        (now, now, now, profile_id),
    )
    return int(changed)


def _project_workspace_authors(
    profile_id: str,
    instance_id: str,
    author_rows: Sequence[sqlite3.Row],
) -> list[dict[str, Any]]:
    by_kind = {str(row["author_kind"]): row for row in author_rows}
    authors: list[dict[str, Any]] = []
    for kind in AUTHOR_ORDER:
        row = by_kind.get(kind.value)
        if row is None:
            raise KeyError((profile_id, instance_id, kind.value))
        item = dict(row)
        item["state"] = _json_mapping(item.pop("state_json"))
        authors.append(item)
    return authors


def _load_workspace_story_sources(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in conn.execute(
            """SELECT story_source_id, public_ref, module_text,
                created_at, updated_at
            FROM background_story_sources
            WHERE profile_id = ? AND instance_id = ?
            ORDER BY story_source_id DESC
            LIMIT 100""",
            (profile_id, instance_id),
        )
    ]


def _load_workspace_timeline(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT event_id, public_ref, source, content,
            frame_start_at, frame_end_at, leftover_text, created_at
        FROM background_role_timeline_events
        WHERE profile_id = ? AND instance_id = ?
        ORDER BY frame_end_at DESC, event_id DESC LIMIT 100""",
        (profile_id, instance_id),
    ).fetchall()
    return [dict(row) for row in rows]


def _load_workspace_current_view(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
) -> dict[str, Any]:
    row = conn.execute(
        """SELECT * FROM background_role_current_views
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    if row is None:
        return {}
    return dict(row)


def _load_background_workspace_sql(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
) -> dict[str, Any]:
    instance = conn.execute(
        """SELECT * FROM background_instances
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    if instance is None:
        raise KeyError((profile_id, instance_id))
    author_rows = conn.execute(
        """SELECT * FROM background_author_states
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchall()
    return {
        "instance": dict(instance),
        "authors": _project_workspace_authors(
            profile_id,
            instance_id,
            author_rows,
        ),
        "story_sources": _load_workspace_story_sources(conn, profile_id, instance_id),
        "timeline": _load_workspace_timeline(conn, profile_id, instance_id),
        "current_view": _load_workspace_current_view(conn, profile_id, instance_id),
    }


class BackgroundAdminActions(QuickSetupLifeMixin):
    @staticmethod
    def _set_background_enabled_for_quick_setup(
        conn: sqlite3.Connection,
        instance: sqlite3.Row,
        **values: Any,
    ) -> int:
        return _set_background_enabled_sql(conn, instance, **values)

    async def invalidate_profile_seed(self, profile_id: str) -> int:
        now = _dt(_now())
        return int(
            await self.uow.run(lambda conn: invalidate_profile_seed_sql(conn, profile_id, now))
        )

    async def load_background_workspace(
        self,
        profile_id: str,
        instance_id: str,
    ) -> dict[str, Any]:
        await self.ensure_instance(profile_id, instance_id)
        return dict(
            await self.db.call(
                lambda conn: _load_background_workspace_sql(conn, profile_id, instance_id)
            )
        )

    async def set_background_enabled(
        self,
        profile_id: str,
        instance_id: str,
        *,
        enabled: bool,
        expected_version: int,
    ) -> int:
        await self.ensure_instance(profile_id, instance_id)
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            instance = self._require_version(
                conn,
                profile_id,
                instance_id,
                expected_version,
            )
            return _set_background_enabled_sql(
                conn,
                instance,
                enabled=enabled,
                now=now,
                reason="background_enabled_changed",
            )

        return int(await self.uow.run(operation))

    async def save_background_config(
        self,
        profile_id: str,
        instance_id: str,
        *,
        patch: Mapping[str, Any],
        backend_overrides: Mapping[str, str] | None,
        expected_version: int,
    ) -> int:
        await self.ensure_instance(profile_id, instance_id)
        unknown = set(patch) - _CONFIG_FIELDS
        if unknown:
            raise ValueError(f"不支持的背景设置：{sorted(unknown)}")
        values: dict[str, Any] = {}
        for key, value in patch.items():
            if key in _BOOLEAN_CONFIG_FIELDS:
                if not isinstance(value, bool):
                    raise ValueError(f"{key} 必须是布尔值")
                values[key] = int(value)
            elif key in _INTEGER_CONFIG_FIELDS:
                values[key] = int(value)
            else:
                values[key] = str(value or "").strip()
        overrides = {
            BackgroundAuthorKind(str(kind).upper()).value: str(value or "").strip()
            for kind, value in dict(backend_overrides or {}).items()
        }
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            instance = self._require_version(
                conn,
                profile_id,
                instance_id,
                expected_version,
            )
            self._validate_config({**dict(instance), **values})
            changed_values = _changed_config_values(instance, values)
            changed_overrides = _changed_backend_overrides(
                conn,
                profile_id,
                instance_id,
                overrides,
            )
            if not changed_values and not changed_overrides:
                return int(expected_version)
            task_ids = self._active_task_ids(conn, profile_id, instance_id)
            _cancel_tasks_sql(
                conn,
                task_ids,
                reason="background_config_changed",
                now=now,
            )
            conn.execute(
                """UPDATE background_author_states
                SET status = 'IDLE', active_task_id = NULL,
                    generation = generation + 1,
                    schedule_version = schedule_version + 1,
                    last_error = '', updated_at = ?
                WHERE profile_id = ? AND instance_id = ?
                  AND active_task_id IS NOT NULL""",
                (now, profile_id, instance_id),
            )
            assignments = [f"{name} = ?" for name in changed_values]
            parameters: list[Any] = [changed_values[name] for name in changed_values]
            assignments.extend(("config_version = config_version + 1", "updated_at = ?"))
            parameters.extend((now, profile_id, instance_id, int(expected_version)))
            cursor = conn.execute(
                f"""UPDATE background_instances SET {", ".join(assignments)}
                WHERE profile_id = ? AND instance_id = ? AND config_version = ?""",
                parameters,
            )
            if cursor.rowcount != 1:
                raise ValueError("背景设置已变化，请刷新后重试")
            for kind, backend_id in changed_overrides.items():
                conn.execute(
                    """UPDATE background_author_states
                    SET backend_id = ?, schedule_version = schedule_version + 1,
                        updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND author_kind = ?""",
                    (backend_id, now, profile_id, instance_id, kind),
                )
            return int(expected_version) + 1

        return int(await self.uow.run(operation))

    async def force_background_authors(
        self,
        profile_id: str,
        instance_id: str,
        *,
        author_kinds: Sequence[BackgroundAuthorKind | str],
        expected_version: int,
    ) -> dict[str, Any]:
        await self.ensure_instance(profile_id, instance_id)
        kinds = tuple(dict.fromkeys(BackgroundAuthorKind(value).value for value in author_kinds))
        if not kinds:
            raise ValueError("至少选择一个创作层")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> dict[str, Any]:
            instance = self._require_version(
                conn,
                profile_id,
                instance_id,
                expected_version,
            )
            if not bool(instance["enabled"]):
                raise ValueError("背景推演已关闭")
            placeholders = ",".join("?" for _ in kinds)
            active_rows = conn.execute(
                f"""SELECT active_task_id FROM background_author_states
                WHERE profile_id = ? AND instance_id = ?
                  AND author_kind IN ({placeholders})
                  AND active_task_id IS NOT NULL""",
                (profile_id, instance_id, *kinds),
            ).fetchall()
            _cancel_tasks_sql(
                conn,
                tuple(int(row["active_task_id"]) for row in active_rows),
                reason="background_manual_wake",
                now=now,
            )
            conn.execute(
                f"""UPDATE background_author_states
                SET status = 'IDLE', active_task_id = NULL,
                    schedule_version = schedule_version + 1,
                    last_error = '', updated_at = ?
                WHERE profile_id = ? AND instance_id = ?
                  AND author_kind IN ({placeholders})
                  AND active_task_id IS NOT NULL""",
                (now, profile_id, instance_id, *kinds),
            )
            conn.execute(
                f"""UPDATE background_author_states
                SET status = 'IDLE', active_task_id = NULL,
                    generation = generation + 1,
                    next_due_at = ?, hard_due_at = ?,
                    force_generation = force_generation + 1,
                    schedule_version = schedule_version + 1,
                    last_error = '', updated_at = ?
                WHERE profile_id = ? AND instance_id = ?
                  AND author_kind IN ({placeholders})""",
                (now, now, now, profile_id, instance_id, *kinds),
            )
            return {
                "config_version": int(expected_version),
                "active_task_ids": [],
            }

        return dict(await self.uow.run(operation))

    async def reset_background(
        self,
        profile_id: str,
        instance_id: str,
        *,
        expected_version: int,
    ) -> int:
        await self.ensure_instance(profile_id, instance_id)
        reset_at = _now()
        now = _dt(reset_at)

        def operation(conn: sqlite3.Connection) -> int:
            instance = self._require_version(
                conn,
                profile_id,
                instance_id,
                expected_version,
            )
            if int(instance["foreground_lease_count"] or 0) > 0:
                lease_until = _parse(instance["foreground_lease_until"])
                foreground_active = foreground_activity_is_active_sql(
                    conn,
                    profile_id=profile_id,
                    instance_id=instance_id,
                    now=now,
                )
                if lease_until is None or lease_until > reset_at or foreground_active:
                    raise ValueError("角色正在处理前台对话，暂时不能重置背景生活")
                cleared = conn.execute(
                    """UPDATE background_instances
                    SET foreground_lease_owner = NULL,
                        foreground_lease_token = NULL,
                        foreground_lease_until = NULL,
                        foreground_lease_holders_json = '{}',
                        foreground_lease_count = 0
                    WHERE profile_id = ? AND instance_id = ?
                      AND foreground_lease_count > 0""",
                    (profile_id, instance_id),
                )
                if cleared.rowcount != 1:
                    raise ValueError("前台租约状态已变化，请稍后重试")
            task_ids = self._active_task_ids(conn, profile_id, instance_id)
            _cancel_tasks_sql(
                conn,
                task_ids,
                reason="background_reset",
                now=now,
            )
            message_tail = conn.execute(
                """SELECT COALESCE(MAX(message_id), 0) AS tail
                FROM instance_messages
                WHERE profile_id = ? AND instance_id = ?
                  AND role IN ('user', 'assistant')""",
                (profile_id, instance_id),
            ).fetchone()
            run_tail = conn.execute(
                """SELECT COALESCE(MAX(run_id), 0) AS tail
                FROM instance_core_runs
                WHERE profile_id = ? AND instance_id = ? AND status = 'COMPLETED'
                  AND source IN ('FOREGROUND_MESSAGE', 'DEFERRED_MESSAGE')""",
                (profile_id, instance_id),
            ).fetchone()
            _clear_background_generated_state_sql(
                conn,
                profile_id,
                instance_id,
                now=now,
            )
            cursor = conn.execute(
                """UPDATE background_instances
                SET initialization_state = 'INITIALIZING',
                    initialization_step = 'WORLD',
                    ordinary_since_keyframe = 0,
                    continuity_version = continuity_version + 1,
                    simulated_through_at = NULL,
                    foreground_message_cursor = ?,
                    foreground_run_cursor = ?,
                    publication_version = publication_version + 1,
                    timeline_version = timeline_version + 1,
                    view_version = view_version + 1,
                    config_version = config_version + 1,
                    updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND config_version = ?""",
                (
                    int(message_tail["tail"] or 0),
                    int(run_tail["tail"] or 0),
                    now,
                    profile_id,
                    instance_id,
                    int(expected_version),
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("背景设置已变化，请刷新后重试")
            conn.execute(
                """INSERT INTO background_initialization_openings(
                    profile_id, instance_id, anchor_at, keyframe_completed,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 0, ?, ?)
                ON CONFLICT(profile_id, instance_id) DO UPDATE SET
                    anchor_at = excluded.anchor_at,
                    keyframe_completed = 0,
                    updated_at = excluded.updated_at""",
                (profile_id, instance_id, now, now, now),
            )
            return int(expected_version) + 1

        return int(await self.uow.run(operation))

    @staticmethod
    def _require_version(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        expected_version: int,
    ) -> sqlite3.Row:
        row = conn.execute(
            """SELECT * FROM background_instances
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        ).fetchone()
        if row is None:
            raise KeyError((profile_id, instance_id))
        if int(row["config_version"]) != int(expected_version):
            raise ValueError("背景设置已变化，请刷新后重试")
        return row

    @staticmethod
    def _active_task_ids(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
    ) -> tuple[int, ...]:
        rows = conn.execute(
            """SELECT active_task_id FROM background_author_states
            WHERE profile_id = ? AND instance_id = ? AND active_task_id IS NOT NULL""",
            (profile_id, instance_id),
        ).fetchall()
        return tuple(int(row["active_task_id"]) for row in rows)

    @staticmethod
    def _validate_config(values: Mapping[str, Any]) -> None:
        for name in _INTEGER_CONFIG_FIELDS:
            value = int(values[name])
            if value < 1 or value > 525600:
                raise ValueError(f"{name} 必须在 1 到 525600 之间")
        if int(values["keyframe_every_ordinary"]) > 100:
            raise ValueError("keyframe_every_ordinary 必须在 1 到 100 之间")
        for prefix in ("ordinary", "story_source", "life_direction", "world"):
            if int(values[f"{prefix}_min_minutes"]) > int(values[f"{prefix}_max_minutes"]):
                raise ValueError(f"{prefix} 的最短间隔不能大于最长间隔")


__all__ = ["BackgroundAdminActions", "invalidate_profile_seed_sql"]
