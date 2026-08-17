from __future__ import annotations

from dataclasses import dataclass

from ..lifecycle import FILE_ARTIFACT_RETENTION, FileArtifactsDisabled
from ..ports import FileWorkCallbackPort
from .support import Any, Mapping, _dt, _dump, _parse, sqlite3, uuid


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
