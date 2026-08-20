"""Explicit, transactional forward migrations between published schema identities."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from types import MappingProxyType

from .baseline import INSTANCE_CHAT_POLICIES_SQL
from .current import (
    CURRENT_SCHEMA_VERSION,
    METADATA_SQL,
    OLDEST_MIGRATABLE_SCHEMA_VERSION,
    SCHEMA_TABLE,
    SchemaIdentity,
    SchemaRecoveryReason,
    SchemaRecoveryRequired,
    current_schema_identity,
    database_uses_current_schema,
    database_uses_schema_identity,
    read_schema_created_at,
    schema_identity_for_version,
    write_schema_identity,
)

MigrationStep = Callable[[sqlite3.Connection], None]
MigrationStepHook = Callable[[sqlite3.Connection, SchemaIdentity], None]


def _migrate_prompt_cache_evidence_v1_to_v2(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        "SELECT backend_id, evidence_json FROM ai_prompt_cache_capabilities"
    ).fetchall()
    for backend_id, raw_evidence in rows:
        try:
            evidence = json.loads(str(raw_evidence))
        except (TypeError, ValueError) as exc:
            raise sqlite3.IntegrityError("prompt-cache evidence is not valid JSON") from exc
        if not isinstance(evidence, Mapping):
            raise sqlite3.IntegrityError("prompt-cache evidence is not an object")
        if (
            evidence.get("schema_version") == 2
            and isinstance(evidence.get("latest"), Mapping)
            and isinstance(evidence.get("quality"), Mapping)
        ):
            continue
        migrated = {
            "schema_version": 2,
            "latest": dict(evidence),
            "quality": {
                "families": {},
                "status": "OBSERVING",
                "reason": "",
                "anomaly_count": 0,
                "last_read_tokens": 0,
                "last_write_tokens": 0,
            },
        }
        connection.execute(
            "UPDATE ai_prompt_cache_capabilities SET evidence_json = ? WHERE backend_id = ?",
            (
                json.dumps(migrated, ensure_ascii=False, separators=(",", ":")),
                str(backend_id),
            ),
        )


def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    # Version 2 makes the metadata table stable for all later forward
    # migrations and moves the last persisted prompt-cache envelope conversion
    # out of runtime code. Both changes share this one transactional boundary.
    _migrate_prompt_cache_evidence_v1_to_v2(connection)
    connection.execute(f"DROP TABLE {SCHEMA_TABLE}")
    connection.execute(METADATA_SQL)


def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
    """Materialize the tier preset for rows left at the old ghost defaults.

    Only untouched 128k/64k rows are repaired. Explicit administrator values
    remain authoritative; selecting a tier after this migration intentionally
    applies that preset to both scopes through the normal save transaction.
    """

    presets = (
        ("极简", 128_000, 20_000),
        ("轻量", 128_000, 40_000),
        ("均衡", 128_000, 60_000),
        ("标准", 128_000, 80_000),
        ("深入", 160_000, 100_000),
        ("极致", 200_000, 150_000),
    )
    for complexity, maximum, target in presets:
        connection.execute(
            """UPDATE scope_configs
            SET max_context_tokens = ?, target_context_tokens = ?,
                version = version + 1, updated_at = CURRENT_TIMESTAMP
            WHERE max_context_tokens = 128000 AND target_context_tokens = 64000
              AND profile_id IN (
                  SELECT profile_id FROM role_profiles WHERE thinking_complexity = ?
              )""",
            (maximum, target, complexity),
        )


def _migrate_v3_to_v4(connection: sqlite3.Connection) -> None:
    """Add the per-private-chat display-name precedence without losing names."""

    connection.execute("ALTER TABLE instance_chat_policies RENAME TO instance_chat_policies_v3")
    connection.execute(INSTANCE_CHAT_POLICIES_SQL)
    connection.execute(
        """INSERT INTO instance_chat_policies(
            profile_id, instance_id, soulcore_enabled, image_send_enabled,
            private_fallback_player_name, private_name_override_enabled,
            version, created_at, updated_at
        )
        SELECT profile_id, instance_id, soulcore_enabled, image_send_enabled,
            private_fallback_player_name, 0, version, created_at, updated_at
        FROM instance_chat_policies_v3"""
    )
    connection.execute("DROP TABLE instance_chat_policies_v3")


def _migrate_v4_to_v5(connection: sqlite3.Connection) -> None:
    """Remove the retired per-conversation background enable state."""

    now_sql = "CURRENT_TIMESTAMP"
    connection.execute(
        f"""UPDATE role_profiles AS profile
        SET background_life_enabled = 1,
            background_life_version = background_life_version + 1,
            updated_at = {now_sql}
        WHERE background_life_enabled = 0
          AND EXISTS (
            SELECT 1 FROM background_instances background
            WHERE background.profile_id = profile.profile_id
              AND background.enabled = 1
          )"""
    )
    connection.execute(
        f"""UPDATE ai_workflows
        SET status = 'CANCELLED', final_error_code = 'BACKGROUND_POLICY_CHANGED',
            final_message = 'background enable policy became role-wide',
            finished_at = {now_sql}, updated_at = {now_sql}, version = version + 1
        WHERE status = 'RUNNING' AND workflow_id IN (
            SELECT workflow_id FROM ai_tasks
            WHERE task_type = 'BACKGROUND_AUTHOR'
              AND workflow_id IS NOT NULL
              AND status NOT IN (
                'RUNNING','PAUSE_REQUESTED','CANCEL_REQUESTED',
                'DEFERRED','SUCCEEDED','FAILED','CANCELLED'
              )
        )"""
    )
    connection.execute(
        f"""UPDATE ai_tasks
        SET status = CASE
                WHEN status IN ('RUNNING','PAUSE_REQUESTED','CANCEL_REQUESTED')
                THEN 'CANCEL_REQUESTED' ELSE 'CANCELLED' END,
            last_error = 'background enable policy became role-wide',
            finished_at = CASE
                WHEN status IN ('RUNNING','PAUSE_REQUESTED','CANCEL_REQUESTED')
                THEN finished_at ELSE {now_sql} END,
            lease_owner = CASE
                WHEN status IN ('RUNNING','PAUSE_REQUESTED','CANCEL_REQUESTED')
                THEN lease_owner ELSE NULL END,
            lease_until = CASE
                WHEN status IN ('RUNNING','PAUSE_REQUESTED','CANCEL_REQUESTED')
                THEN lease_until ELSE NULL END,
            updated_at = {now_sql}, version = version + 1
        WHERE task_type = 'BACKGROUND_AUTHOR'
          AND status NOT IN ('DEFERRED','SUCCEEDED','FAILED','CANCELLED')"""
    )
    connection.execute(
        f"""UPDATE background_author_states
        SET status = 'IDLE', active_task_id = NULL,
            generation = generation + 1,
            schedule_version = schedule_version + 1,
            last_error = '', updated_at = {now_sql}"""
    )
    connection.execute(
        f"""UPDATE background_instances
        SET config_version = config_version + 1, updated_at = {now_sql}"""
    )
    connection.execute(
        f"""UPDATE character_instances AS character
        SET initialization_state = CASE
                WHEN (
                    SELECT profile.background_life_enabled
                    FROM role_profiles profile
                    WHERE profile.profile_id = character.profile_id
                ) = 1
                AND (
                    SELECT background.initialization_state
                    FROM background_instances background
                    WHERE background.profile_id = character.profile_id
                      AND background.instance_id = character.instance_id
                ) IN ('UNINITIALIZED','INITIALIZING')
                THEN (
                    SELECT background.initialization_state
                    FROM background_instances background
                    WHERE background.profile_id = character.profile_id
                      AND background.instance_id = character.instance_id
                )
                ELSE 'READY' END,
            initialization_completed_at = CASE
                WHEN (
                    SELECT profile.background_life_enabled
                    FROM role_profiles profile
                    WHERE profile.profile_id = character.profile_id
                ) = 1
                AND (
                    SELECT background.initialization_state
                    FROM background_instances background
                    WHERE background.profile_id = character.profile_id
                      AND background.instance_id = character.instance_id
                ) IN ('UNINITIALIZED','INITIALIZING')
                THEN NULL ELSE COALESCE(initialization_completed_at, {now_sql}) END,
            updated_at = {now_sql}
        WHERE EXISTS (
            SELECT 1 FROM background_instances background
            WHERE background.profile_id = character.profile_id
              AND background.instance_id = character.instance_id
        )"""
    )
    connection.execute(
        f"""UPDATE background_author_states AS author
        SET next_due_at = {now_sql}, hard_due_at = {now_sql},
            schedule_version = schedule_version + 1, updated_at = {now_sql}
        WHERE EXISTS (
            SELECT 1
            FROM background_instances background
            JOIN role_profiles profile ON profile.profile_id = background.profile_id
            LEFT JOIN background_initialization_openings opening
              ON opening.profile_id = background.profile_id
             AND opening.instance_id = background.instance_id
            WHERE background.profile_id = author.profile_id
              AND background.instance_id = author.instance_id
              AND profile.background_life_enabled = 1
              AND author.author_kind = CASE background.initialization_step
                WHEN 'WORLD' THEN 'WORLD'
                WHEN 'LIFE_DIRECTION' THEN 'LIFE_DIRECTION'
                WHEN 'STORY_SOURCE' THEN 'STORY_SOURCE'
                WHEN 'ORDINARY_CURRENT' THEN CASE
                    WHEN COALESCE(opening.keyframe_completed, 0) = 1
                    THEN 'ORDINARY' ELSE 'KEYFRAME' END
                WHEN 'READY' THEN 'ORDINARY'
              END
        )"""
    )
    connection.execute("ALTER TABLE background_instances DROP COLUMN enabled")
    connection.execute("ALTER TABLE background_instances DROP COLUMN disabled_at")
    connection.execute("ALTER TABLE background_instances DROP COLUMN resumed_at")


MIGRATION_STEPS = MappingProxyType(
    {
        1: _migrate_v1_to_v2,
        2: _migrate_v2_to_v3,
        3: _migrate_v3_to_v4,
        4: _migrate_v4_to_v5,
    }
)


def migration_registry_is_contiguous() -> bool:
    expected = tuple(range(OLDEST_MIGRATABLE_SCHEMA_VERSION, CURRENT_SCHEMA_VERSION))
    return tuple(MIGRATION_STEPS) == expected and tuple(sorted(MIGRATION_STEPS)) == expected


def migrate_to_current(
    connection: sqlite3.Connection,
    source: SchemaIdentity,
    *,
    after_step: MigrationStepHook | None = None,
) -> None:
    """Advance an exact published identity; rollback the whole chain on failure."""

    registered = schema_identity_for_version(source.version)
    if registered != source or source.version >= CURRENT_SCHEMA_VERSION:
        raise ValueError("migration source is not an older published schema identity")
    if not migration_registry_is_contiguous():
        raise RuntimeError("SQLite migration registry is not contiguous")
    created_at = read_schema_created_at(connection)
    if created_at is None:
        raise ValueError("published schema identity has no database creation time")

    try:
        connection.execute("BEGIN IMMEDIATE")
        identity = source
        while identity.version < CURRENT_SCHEMA_VERSION:
            step = MIGRATION_STEPS.get(identity.version)
            target = schema_identity_for_version(identity.version + 1)
            if step is None or target is None:
                raise RuntimeError("SQLite migration chain is incomplete")
            step(connection)
            write_schema_identity(connection, target, created_at=created_at)
            if after_step is not None:
                after_step(connection, target)
            if not database_uses_schema_identity(connection, target):
                raise RuntimeError("SQLite migration produced an unregistered structure")
            identity = target
        integrity = [
            str(row[0]).strip().lower() for row in connection.execute("PRAGMA integrity_check")
        ]
        if integrity != ["ok"]:
            raise sqlite3.DatabaseError("post-migration integrity check failed")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise sqlite3.IntegrityError("post-migration foreign-key check failed")
        if identity != current_schema_identity() or not database_uses_current_schema(connection):
            raise RuntimeError("SQLite migration did not reach the current identity")
        connection.commit()
    except BaseException as exc:
        connection.rollback()
        raise SchemaRecoveryRequired(
            SchemaRecoveryReason.MIGRATION_FAILED,
            "the forward database migration was rolled back",
        ) from exc


__all__ = ["MIGRATION_STEPS", "migrate_to_current", "migration_registry_is_contiguous"]
