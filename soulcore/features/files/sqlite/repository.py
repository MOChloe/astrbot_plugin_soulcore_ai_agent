from __future__ import annotations

from dataclasses import dataclass

from ....contracts.ai_task_payload import decode_task_payload
from ....storage.sqlite.core_mappers import CoreRecordMappers
from ....storage.sqlite.repository import SqliteRepository
from ...profiles.ports import ProfilesRepositoryPort
from ..lifecycle import (
    FILE_ARTIFACT_RETENTION,
    FILE_ARTIFACTS_DISABLED_REASON,
    FileArtifactsDisabled,
)
from ..ports import FileWorkCallbackPort
from .queries import FileQueries
from .release import FileReleaseCommands
from .support import Any, Mapping, RoleProfile, _dt, _dump, _now, _parse, sqlite3, uuid


@dataclass(frozen=True, slots=True)
class FileJobCompletionContext:
    task_id: int
    lease_token: int
    artifact: Mapping[str, Any]
    now: str


class FileJobCompletionTransaction:
    def __init__(
        self, context: FileJobCompletionContext, work_callback: FileWorkCallbackPort
    ) -> None:
        self.context = context
        self.work_callback = work_callback

    def __call__(self, conn: sqlite3.Connection) -> sqlite3.Row:
        job = self._load_current_job(conn)
        existing = conn.execute(
            "SELECT * FROM important_todos WHERE source_job_id = ?",
            (job["job_id"],),
        ).fetchone()
        if existing is not None:
            self._release_media_holds(conn, str(job["job_id"]))
            self._complete_work_callback(
                conn,
                job,
                str(existing["file_asset_id"] or ""),
                str(existing["todo_id"]),
            )
            return existing
        asset_id = f"file:{uuid.uuid4().hex}"
        todo_id = f"todo:{uuid.uuid4().hex}"
        self._insert_asset(conn, job, asset_id)
        self._insert_todo(conn, job, asset_id, todo_id)
        self._finish_job(conn, str(job["job_id"]))
        self._release_media_holds(conn, str(job["job_id"]))
        if not self._complete_work_callback(conn, job, asset_id, todo_id):
            raise ValueError("file job is missing its Main Core work binding")
        result = conn.execute(
            "SELECT * FROM important_todos WHERE todo_id = ?", (todo_id,)
        ).fetchone()
        assert result is not None
        return result

    def _complete_work_callback(
        self,
        conn: sqlite3.Connection,
        job: sqlite3.Row,
        asset_id: str,
        todo_id: str,
    ) -> bool:
        completed_at = _parse(self.context.now)
        if completed_at is None:
            raise ValueError("file completion requires an aware timestamp")
        return bool(
            self.work_callback.complete_file_job(
                conn,
                job_id=str(job["job_id"]),
                status="SUCCEEDED",
                resource_ref=asset_id,
                result_kind="FILE_ARTIFACT",
                result_summary="Controlled file artifact generation completed.",
                todo_id=todo_id,
                now=completed_at,
            )
        )

    def _load_current_job(self, conn: sqlite3.Connection) -> sqlite3.Row:
        context = self.context
        job = conn.execute(
            """SELECT j.*, p.file_artifacts_enabled,
                t.status AS task_status, t.lease_token AS task_lease_token
            FROM file_generation_jobs j JOIN ai_tasks t
              ON t.task_id = j.ai_task_id
            JOIN role_profiles p ON p.profile_id = j.profile_id
            WHERE j.ai_task_id = ?""",
            (context.task_id,),
        ).fetchone()
        if job is None or int(job["task_lease_token"]) != context.lease_token:
            raise ValueError("file task lease is no longer current")
        if not bool(job["file_artifacts_enabled"]):
            raise FileArtifactsDisabled("file artifacts are disabled for this profile")
        if str(job["task_status"]) != "RUNNING":
            raise ValueError("file task lease is no longer current")
        return job

    def _insert_asset(self, conn: sqlite3.Connection, job: sqlite3.Row, asset_id: str) -> None:
        context = self.context
        artifact = context.artifact
        generated_at = _parse(context.now)
        if generated_at is None or generated_at.tzinfo is None:
            raise ValueError("file completion requires an aware timestamp")
        expires_at = _dt(generated_at + FILE_ARTIFACT_RETENTION)
        conn.execute(
            """INSERT INTO file_assets(
                asset_id, profile_id, instance_id, job_id, file_format,
                display_name, mime_type, storage_relpath, sha256, byte_size,
                char_count, page_count, metadata_json, expires_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                asset_id,
                job["profile_id"],
                job["instance_id"],
                job["job_id"],
                artifact["file_format"],
                artifact["display_name"],
                artifact["mime_type"],
                artifact["storage_relpath"],
                artifact["sha256"],
                int(artifact["byte_size"]),
                int(artifact.get("char_count") or 0),
                int(artifact.get("page_count") or 0),
                _dump(artifact.get("metadata") or {}),
                expires_at,
                context.now,
                context.now,
            ),
        )

    def _insert_todo(
        self,
        conn: sqlite3.Connection,
        job: sqlite3.Row,
        asset_id: str,
        todo_id: str,
    ) -> None:
        context = self.context
        artifact = context.artifact
        conn.execute(
            """INSERT INTO important_todos(
                todo_id, profile_id, instance_id, kind, source_job_id,
                file_asset_id, payload_json, status,
                available_at, idempotency_key, created_at, updated_at
            ) VALUES (?, ?, ?, 'FILE_READY', ?, ?, ?, 'PENDING', ?, ?, ?, ?)""",
            (
                todo_id,
                job["profile_id"],
                job["instance_id"],
                job["job_id"],
                asset_id,
                _dump(
                    {
                        "display_name": artifact["display_name"],
                        "file_format": artifact["file_format"],
                    }
                ),
                context.now,
                f"file-ready:{job['job_id']}",
                context.now,
                context.now,
            ),
        )

    def _finish_job(self, conn: sqlite3.Connection, job_id: str) -> None:
        conn.execute(
            """UPDATE file_generation_jobs SET status = 'SUCCEEDED',
            safe_error_code = '', safe_error_message = '', finished_at = ?,
            updated_at = ?, version = version + 1 WHERE job_id = ?""",
            (self.context.now, self.context.now, job_id),
        )

    def _release_media_holds(self, conn: sqlite3.Connection, job_id: str) -> None:
        conn.execute(
            """UPDATE media_retention_holds SET released_at = ?
            WHERE holder_kind = 'FILE_GENERATION_JOB' AND holder_id = ?
              AND released_at IS NULL""",
            (self.context.now, job_id),
        )


class FileJobRecords:
    async def get_file_generation_job_for_task(self, task_id: int) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            """SELECT j.*, t.input_json, t.status AS task_status,
                t.lease_token, t.attempts, t.max_attempts,
                x.todo_id, x.file_asset_id
            FROM file_generation_jobs j JOIN ai_tasks t ON t.task_id = j.ai_task_id
            LEFT JOIN important_todos x ON x.source_job_id = j.job_id
            WHERE j.ai_task_id = ?""",
            (int(task_id),),
        )
        if row is None:
            return None
        record = self._record(row, json_columns=())
        record["input"] = decode_task_payload("input", row["input_json"])
        record.pop("input_json", None)
        return record

    async def mark_file_generation_job_running(self, task_id: int, *, lease_token: int) -> bool:
        now = _dt(_now())
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE file_generation_jobs SET status = 'RUNNING',
                    version = version + 1, updated_at = ?
                WHERE ai_task_id = ? AND status IN ('QUEUED', 'RUNNING', 'RECOVERY_REQUIRED')
                  AND EXISTS (SELECT 1 FROM ai_tasks t WHERE t.task_id = ?
                    AND t.status = 'RUNNING' AND t.lease_token = ?)
                  AND EXISTS (SELECT 1 FROM role_profiles p
                    WHERE p.profile_id = file_generation_jobs.profile_id
                      AND p.file_artifacts_enabled = 1)""",
                (now, int(task_id), int(task_id), int(lease_token)),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1

    async def complete_file_generation_job(
        self,
        task_id: int,
        *,
        lease_token: int,
        artifact: Mapping[str, Any],
    ) -> dict[str, Any]:
        context = FileJobCompletionContext(
            task_id=int(task_id),
            lease_token=int(lease_token),
            artifact=artifact,
            now=_dt(_now()),
        )
        row = await self.uow.run(FileJobCompletionTransaction(context, self._work_callback))
        await self.db.publish_backup_after_commit()
        return self._record(row, json_columns=("payload_json",))

    async def reconcile_terminal_file_jobs(self) -> int:
        """Create one visible failed todo after the durable task truly terminates."""

        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            rows = list(
                conn.execute(
                    """SELECT j.*, t.status AS task_status, t.last_error
                FROM file_generation_jobs j JOIN ai_tasks t ON t.task_id = j.ai_task_id
                LEFT JOIN important_todos x ON x.source_job_id = j.job_id
                WHERE j.status NOT IN ('SUCCEEDED', 'FAILED', 'CANCELLED')
                  AND t.status IN ('FAILED', 'CANCELLED') AND x.todo_id IS NULL"""
                )
            )
            for row in rows:
                todo_id = f"todo:{uuid.uuid4().hex}"
                safe_code = (
                    "GENERATION_CANCELLED"
                    if row["task_status"] == "CANCELLED"
                    else "GENERATION_FAILED"
                )
                safe_message = (
                    "文件生成已取消。"
                    if row["task_status"] == "CANCELLED"
                    else "文件生成最终失败，未产生可发送文件。"
                )
                conn.execute(
                    """UPDATE file_generation_jobs SET status = ?,
                        safe_error_code = ?, safe_error_message = ?, finished_at = ?,
                        updated_at = ?, version = version + 1 WHERE job_id = ?""",
                    (
                        "CANCELLED" if row["task_status"] == "CANCELLED" else "FAILED",
                        safe_code,
                        safe_message,
                        now,
                        now,
                        row["job_id"],
                    ),
                )
                conn.execute(
                    """UPDATE media_retention_holds SET released_at = ?
                    WHERE holder_kind = 'FILE_GENERATION_JOB' AND holder_id = ?
                      AND released_at IS NULL""",
                    (now, row["job_id"]),
                )
                conn.execute(
                    """INSERT INTO important_todos(
                        todo_id, profile_id, instance_id, kind, source_job_id,
                        payload_json, status, available_at,
                        idempotency_key, created_at, updated_at
                    ) VALUES (?, ?, ?, 'FILE_FAILED', ?, ?, 'PENDING', ?, ?, ?, ?)""",
                    (
                        todo_id,
                        row["profile_id"],
                        row["instance_id"],
                        row["job_id"],
                        _dump(
                            {
                                "error_code": safe_code,
                                "message": safe_message,
                                "display_name": row["display_name"],
                                "file_format": row["file_format"],
                            }
                        ),
                        now,
                        f"file-failed:{row['job_id']}",
                        now,
                        now,
                    ),
                )
                callback_handled = bool(
                    self._work_callback.complete_file_job(
                        conn,
                        job_id=str(row["job_id"]),
                        status=("CANCELLED" if row["task_status"] == "CANCELLED" else "FAILED"),
                        resource_ref="",
                        result_kind=(
                            "FILE_GENERATION_CANCELLED"
                            if row["task_status"] == "CANCELLED"
                            else "FILE_GENERATION_FAILED"
                        ),
                        result_summary=safe_message,
                        todo_id=todo_id,
                        now=_now(),
                    )
                )
                if not callback_handled:
                    raise ValueError("file job is missing its Main Core work binding")
            return len(rows)

        return await self.uow.run(operation)


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
