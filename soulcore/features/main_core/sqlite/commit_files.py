from __future__ import annotations

from ....contracts.ai_task_payload import encode_task_payload
from .support import Any, sqlite3, uuid


class FileRequestWriter:
    def __init__(self, owner: Any, context: Any) -> None:
        self.owner = owner
        self.context = context

    def create(self, conn: sqlite3.Connection, index: int, request: dict[str, Any]) -> str:
        prepared = self._prepare(conn, request)
        job_id = f"filejob:{uuid.uuid4().hex}"
        key = f"file-request:{self.context.run_id}:{index}"
        task_input = self._task_input(job_id, request, prepared)
        task_id = self._insert_task(conn, key, task_input)
        self._insert_job(conn, key, job_id, task_id, task_input)
        self._add_media_holds(conn, job_id, prepared["image_asset_ids"])
        task_row = conn.execute("SELECT * FROM ai_tasks WHERE task_id = ?", (task_id,)).fetchone()
        assert task_row is not None
        self.owner._audit_ai_task(
            conn,
            task_row,
            "CREATE",
            to_status="READY",
            actor_type="MAIN_CORE",
            actor_id=str(self.context.run_id),
            created_at=self.context.now,
        )
        return job_id

    def _prepare(self, conn: sqlite3.Connection, request: dict[str, Any]) -> dict[str, Any]:
        image_asset_ids = self._unique_strings(request.get("image_asset_ids") or [])
        if len(image_asset_ids) > 5:
            raise ValueError("a file request may contain at most five images")
        file_format = self._validated_file_format(request)
        if image_asset_ids and file_format != "PDF":
            raise ValueError("only PDF file requests may contain images")
        self._validate_document_delegation(request)
        context_message_id = int(request.get("context_message_id") or 0)
        raw_message_ids = request.get("context_message_ids") or [context_message_id]
        context_message_ids = self._unique_positive_ints(raw_message_ids, limit=100)
        self._validate_images(conn, image_asset_ids, context_message_ids)
        layout_preference = self._validated_layout_preference(request)
        return {
            "file_format": file_format,
            "image_asset_ids": image_asset_ids,
            "layout_preference": layout_preference,
        }

    @staticmethod
    def _validated_file_format(request: dict[str, Any]) -> str:
        file_format = str(request.get("file_format") or "").upper()
        if file_format not in {"MD", "TXT", "PDF"}:
            raise ValueError("invalid file format")
        return file_format

    @staticmethod
    def _validate_document_delegation(request: dict[str, Any]) -> None:
        required_delegation = (
            "purpose",
            "audience",
            "requirements",
            "source_materials",
            "voice",
            "fact_policy",
        )
        if any(not str(request.get(name) or "").strip() for name in required_delegation):
            raise ValueError("file request is missing its durable document delegation")
        if str(request.get("fact_policy") or "") not in {
            "基于事实材料",
            "允许目标内创作",
        }:
            raise ValueError("invalid document fact policy")

    @staticmethod
    def _validated_layout_preference(request: dict[str, Any]) -> str:
        layout_preference = str(request.get("layout_preference") or "自动安排").strip()
        if layout_preference not in {"自动安排", "清晰正式", "杂志式图文", "数据概览"}:
            raise ValueError("invalid document layout preference")
        return layout_preference

    @staticmethod
    def _unique_strings(values: list[Any] | tuple[Any, ...]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = str(value or "").strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result

    @staticmethod
    def _unique_positive_ints(values: Any, *, limit: int) -> list[int]:
        result: list[int] = []
        seen: set[int] = set()
        for value in values:
            normalized = int(value or 0)
            if normalized > 0 and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
            if len(result) >= limit:
                break
        return result

    def _validate_images(
        self,
        conn: sqlite3.Connection,
        asset_ids: list[str],
        message_ids: list[int],
    ) -> None:
        if not asset_ids:
            return
        context = self.context
        placeholders = ",".join("?" for _ in asset_ids)
        message_placeholders = ",".join("?" for _ in message_ids) or "NULL"
        rows = list(
            conn.execute(
                f"""SELECT a.asset_id FROM media_assets a
            WHERE a.profile_id = ? AND a.instance_id = ?
              AND a.asset_id IN ({placeholders})
              AND a.file_status = 'AVAILABLE' AND a.mime_type LIKE 'image/%'
              AND (a.core_run_id = ? OR EXISTS (
                SELECT 1 FROM media_asset_message_links link
                WHERE link.asset_id = a.asset_id
                  AND link.profile_id = a.profile_id
                  AND link.instance_id = a.instance_id
                  AND link.message_id IN ({message_placeholders})
              ))""",
                (
                    context.profile_id,
                    context.instance_id,
                    *asset_ids,
                    context.run_id,
                    *message_ids,
                ),
            )
        )
        if {str(row["asset_id"]) for row in rows} != set(asset_ids):
            raise ValueError("document images must be owned current-run or current-message assets")

    @staticmethod
    def _task_input(
        job_id: str, request: dict[str, Any], prepared: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "job_id": job_id,
            "file_format": prepared["file_format"],
            "display_name": str(request.get("display_name") or ""),
            "purpose": str(request.get("purpose") or ""),
            "audience": str(request.get("audience") or ""),
            "requirements": str(request.get("requirements") or ""),
            "source_materials": str(request.get("source_materials") or ""),
            "voice": str(request.get("voice") or ""),
            "fact_policy": str(request.get("fact_policy") or ""),
            "image_asset_ids": prepared["image_asset_ids"],
            "layout_preference": prepared["layout_preference"],
        }

    def _insert_task(self, conn: sqlite3.Connection, key: str, task_input: dict[str, Any]) -> int:
        context = self.context
        cursor = conn.execute(
            """INSERT INTO ai_tasks(
                workflow_id, profile_id, instance_id, task_type, task_class, capability,
                status, priority, due_at, mutex_key,
                idempotency_key, generation, input_json, checkpoint_json,
                retry_policy_json, recovery_policy, max_attempts,
                created_at, updated_at
            ) VALUES ((SELECT workflow_id FROM instance_core_runs WHERE run_id = ?),
                ?, ?, 'FILE_ARTIFACT_GENERATION', 'BACKGROUND',
                'text.completion', 'READY', 40, ?, ?, ?, 1, ?, ?,
                ?, 'RESTART_SAFE', 3, ?, ?)""",
            (
                context.run_id,
                context.profile_id,
                context.instance_id,
                context.now,
                f"file-artifact:{context.instance_id}",
                key,
                encode_task_payload("input", task_input),
                encode_task_payload("checkpoint", {}),
                encode_task_payload("retry_policy", {"delays_hours": [0.02, 0.1, 0.5]}),
                context.now,
                context.now,
            ),
        )
        return int(cursor.lastrowid)

    def _insert_job(
        self,
        conn: sqlite3.Connection,
        key: str,
        job_id: str,
        task_id: int,
        task_input: dict[str, Any],
    ) -> None:
        context = self.context
        conn.execute(
            """INSERT INTO file_generation_jobs(
                job_id, profile_id, instance_id, source_run_id, ai_task_id,
                file_format, display_name, status, idempotency_key,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'QUEUED', ?, ?, ?)""",
            (
                job_id,
                context.profile_id,
                context.instance_id,
                context.run_id,
                task_id,
                task_input["file_format"],
                task_input["display_name"],
                key,
                context.now,
                context.now,
            ),
        )

    def _add_media_holds(self, conn: sqlite3.Connection, job_id: str, asset_ids: list[str]) -> None:
        context = self.context
        for asset_id in asset_ids:
            conn.execute(
                """INSERT INTO media_retention_holds(
                    profile_id, instance_id, asset_id, holder_kind,
                    holder_id, created_at
                ) VALUES (?, ?, ?, 'FILE_GENERATION_JOB', ?, ?)
                ON CONFLICT(asset_id, holder_kind, holder_id) DO NOTHING""",
                (
                    context.profile_id,
                    context.instance_id,
                    asset_id,
                    job_id,
                    context.now,
                ),
            )
