from __future__ import annotations

from ....contracts.ai_task_payload import decode_task_payload
from .job_completion import (
    FileJobCompletionContext,
    FileJobCompletionTransaction,
)
from .support import (
    Any,
    Mapping,
    _dt,
    _dump,
    _now,
    sqlite3,
    uuid,
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
