from __future__ import annotations

from ....storage.sqlite.core_mappers import CoreRecordMappers
from ....storage.sqlite.repository import SqliteRepository
from ...profiles.ports import ProfilesRepositoryPort
from ..lifecycle import FILE_ARTIFACTS_DISABLED_REASON
from ..ports import FileWorkCallbackPort
from .jobs import FileJobRecords
from .queries import FileQueries
from .release import FileReleaseCommands
from .support import (
    RoleProfile,
    _dt,
    _now,
    sqlite3,
)


class FileSettingsRecords:
    async def get_profile_file_artifacts_enabled(self, profile_id: str) -> bool:
        profile = await self._profiles.get_profile(str(profile_id))
        if profile is None:
            raise KeyError(profile_id)
        return bool(profile.file_artifacts_enabled)

    async def set_profile_file_artifacts_enabled(
        self, profile_id: str, enabled: bool
    ) -> RoleProfile:
        normalized_profile_id = str(profile_id)
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> None:
            cursor = conn.execute(
                """UPDATE role_profiles SET file_artifacts_enabled = ?, updated_at = ?
                WHERE profile_id = ?""",
                (int(bool(enabled)), now, normalized_profile_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(normalized_profile_id)
            if enabled:
                self._resume_feature_paused_tasks(conn, normalized_profile_id, now)
            else:
                self._pause_feature_tasks(conn, normalized_profile_id, now)

        await self.uow.run(operation)
        await self.db.publish_backup_after_commit()
        profile = await self._profiles.get_profile(normalized_profile_id)
        assert profile is not None
        return profile

    @classmethod
    def _pause_feature_tasks(cls, conn: sqlite3.Connection, profile_id: str, now: str) -> None:
        conn.execute(
            f"""UPDATE ai_tasks SET
                status = CASE WHEN status = 'RUNNING' THEN 'PAUSE_REQUESTED' ELSE 'PAUSED' END,
                last_error = ?,
                lease_owner = CASE WHEN status = 'RUNNING' THEN lease_owner ELSE NULL END,
                lease_until = CASE WHEN status = 'RUNNING' THEN lease_until ELSE NULL END,
                updated_at = ?, finished_at = NULL, version = version + 1
            WHERE profile_id = ?
              AND status IN ('SCHEDULED', 'READY', 'RUNNING',
                             'RETRY_WAIT', 'RECOVERY_REQUIRED')
              AND ({cls._feature_task_predicate()})""",
            (FILE_ARTIFACTS_DISABLED_REASON, now, profile_id),
        )

    @classmethod
    def _resume_feature_paused_tasks(
        cls, conn: sqlite3.Connection, profile_id: str, now: str
    ) -> None:
        conn.execute(
            f"""UPDATE ai_tasks SET
                status = CASE
                    WHEN status = 'PAUSE_REQUESTED' AND lease_owner IS NOT NULL THEN 'RUNNING'
                    ELSE 'READY'
                END,
                last_error = NULL,
                lease_owner = CASE
                    WHEN status = 'PAUSE_REQUESTED' AND lease_owner IS NOT NULL THEN lease_owner
                    ELSE NULL
                END,
                lease_until = CASE
                    WHEN status = 'PAUSE_REQUESTED' AND lease_owner IS NOT NULL THEN lease_until
                    ELSE NULL
                END,
                due_at = CASE WHEN status = 'PAUSED' THEN ? ELSE due_at END,
                updated_at = ?, finished_at = NULL, version = version + 1
            WHERE profile_id = ? AND status IN ('PAUSED', 'PAUSE_REQUESTED')
              AND last_error = ?
              AND ({cls._feature_task_predicate()})""",
            (now, now, profile_id, FILE_ARTIFACTS_DISABLED_REASON),
        )
        conn.execute(
            """UPDATE instance_wakeups SET due_at = ?, last_error = NULL,
                updated_at = ?, version = version + 1
            WHERE profile_id = ? AND source = 'PLUGIN_WAKE' AND status = 'PENDING'
              AND last_error = ?
              AND EXISTS (
                  SELECT 1 FROM main_core_work_file_bindings binding
                  WHERE binding.profile_id = instance_wakeups.profile_id
                    AND binding.instance_id = instance_wakeups.instance_id
                    AND binding.work_ref = json_extract(
                        instance_wakeups.payload_json, '$.work_ref'
                    )
              )""",
            (now, now, profile_id, FILE_ARTIFACTS_DISABLED_REASON),
        )

    @staticmethod
    def _feature_task_predicate() -> str:
        return """task_type = 'FILE_ARTIFACT_GENERATION' OR (
            task_type = 'MAIN_CORE'
            AND json_extract(input_json, '$.payload.source') = 'PLUGIN_WAKE'
            AND EXISTS (
                SELECT 1 FROM main_core_work_file_bindings binding
                WHERE binding.profile_id = ai_tasks.profile_id
                  AND binding.instance_id = ai_tasks.instance_id
                  AND binding.work_ref = json_extract(
                      ai_tasks.input_json, '$.payload.metadata.work_ref'
                  )
            )
        )"""


class _FileRecords(
    FileSettingsRecords,
    FileJobRecords,
    FileQueries,
    FileReleaseCommands,
):
    pass


class SqliteFileRepository(
    _FileRecords,
    CoreRecordMappers,
    SqliteRepository,
):
    """SQLite implementation of file settings, tasks, and release storage."""

    def __init__(
        self,
        engine,
        profiles: ProfilesRepositoryPort,
        *,
        work_callback: FileWorkCallbackPort,
    ) -> None:
        SqliteRepository.__init__(self, engine)
        self._profiles = profiles
        self._work_callback = work_callback


__all__ = ["SqliteFileRepository"]
