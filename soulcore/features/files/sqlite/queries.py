from __future__ import annotations

from .support import (
    Any,
    Mapping,
    OutboxStatus,
    Sequence,
    _dt,
    _load,
    _now,
    sqlite3,
)

ParsedOutbox = tuple[dict[str, Any], dict[str, Any], set[str], set[str]]


def _parse_file_outboxes(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[ParsedOutbox], dict[str, ParsedOutbox]]:
    parsed_outboxes: list[ParsedOutbox] = []
    outbox_by_key: dict[str, ParsedOutbox] = {}
    for raw_outbox in rows:
        outbox = dict(raw_outbox)
        payload = _load(outbox["payload_json"]) or {}
        todo_ids = {str(value) for value in payload.get("important_todo_ids") or [] if str(value)}
        component_assets = {
            str(value.get("asset_id") or "")
            for value in payload.get("components") or []
            if isinstance(value, Mapping) and str(value.get("type") or "") == "file_artifact"
        }
        parsed = (outbox, payload, todo_ids, component_assets)
        parsed_outboxes.append(parsed)
        outbox_by_key[str(outbox.get("idempotency_key") or "")] = parsed
    return parsed_outboxes, outbox_by_key


def _related_file_outbox(
    parsed_outboxes: Sequence[ParsedOutbox], todo_id: str, asset_id: str
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    for outbox, payload, todo_ids, component_assets in parsed_outboxes:
        if (todo_id and todo_id in todo_ids) or (asset_id and asset_id in component_assets):
            return outbox, payload
    return None, {}


def _paired_announcement(
    outbox_by_key: Mapping[str, ParsedOutbox],
    related_outbox: Mapping[str, Any] | None,
    related_payload: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, bool]:
    dependency_key = str(related_payload.get("depends_on_idempotency_key") or "").strip()
    if not dependency_key or related_outbox is None:
        return None, False
    paired = outbox_by_key.get(dependency_key)
    artifact_key = str(related_outbox.get("idempotency_key") or "")
    valid = (
        paired is not None
        and str(paired[1].get("file_delivery_role") or "") == "ANNOUNCEMENT"
        and str(paired[1].get("file_followup_idempotency_key") or "").strip() == artifact_key
    )
    return (paired[0] if valid and paired else None), not valid


def _file_delete_block_reason(
    record: Mapping[str, Any],
    *,
    asset_id: str,
    pair_invalid: bool,
    workflow_outboxes: Sequence[Mapping[str, Any]],
    outbox_attempts: int,
) -> str:
    delivery_status = str(record.get("delivery_status") or "")
    todo_status = str(record.get("todo_status") or "")
    if not asset_id:
        return "文件尚未生成"
    if str(record.get("file_status") or "") == "RELEASED":
        return "文件已经删除"
    if delivery_status == "DELIVERED":
        return "文件已经送达，不能作为未投递文件删除"
    if pair_invalid:
        return "文件说明与附件的配对状态不完整"
    if delivery_status in {
        "PLATFORM_ACCEPTED_UNCONFIRMED",
        "PARTIALLY_ATTEMPTED",
        "UNKNOWN_AFTER_CRASH",
    } or _has_unsafe_outbox(workflow_outboxes):
        return "平台投递结果未知，不能确认文件尚未送达"
    if outbox_attempts > 0:
        return "该文件已经发生过平台投递尝试，不能确认尚未送达"
    if todo_status in {"SELECTED", "DELIVERY_PENDING"} and not workflow_outboxes:
        return "文件已被运行选中且投递尚未结算，当前不能删除"
    if todo_status not in {"PENDING", "SELECTED", "DELIVERY_PENDING", "CANCELLED"}:
        return "当前待办状态不允许删除"
    return ""


def _has_unsafe_outbox(outboxes: Sequence[Mapping[str, Any]]) -> bool:
    unsafe = {
        OutboxStatus.SENDING.value,
        OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED.value,
        OutboxStatus.PARTIALLY_ATTEMPTED.value,
        OutboxStatus.UNKNOWN_AFTER_CRASH.value,
    }
    return any(str(value.get("status") or "") in unsafe for value in outboxes)


def _workflow_outboxes(
    related: dict[str, Any] | None, announcement: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], int]:
    outboxes = [value for value in (related, announcement) if value is not None]
    attempts = max((int(value.get("attempts") or 0) for value in outboxes), default=0)
    return outboxes, attempts


def _decorate_file_record(
    record: dict[str, Any],
    parsed_outboxes: Sequence[ParsedOutbox],
    outbox_by_key: Mapping[str, ParsedOutbox],
) -> dict[str, Any]:
    todo_id = str(record.get("todo_id") or "")
    asset_id = str(record.get("asset_id") or "")
    related_outbox, related_payload = _related_file_outbox(parsed_outboxes, todo_id, asset_id)
    announcement, pair_invalid = _paired_announcement(
        outbox_by_key, related_outbox, related_payload
    )
    workflow_outboxes, attempts = _workflow_outboxes(related_outbox, announcement)
    block_reason = _file_delete_block_reason(
        record,
        asset_id=asset_id,
        pair_invalid=pair_invalid,
        workflow_outboxes=workflow_outboxes,
        outbox_attempts=attempts,
    )
    record.update(
        {
            "outbox_id": (related_outbox or {}).get("outbox_id"),
            "outbox_status": str((related_outbox or {}).get("status") or "") or None,
            "outbox_attempts": attempts,
            "announcement_outbox_id": (announcement or {}).get("outbox_id"),
            "announcement_outbox_status": str((announcement or {}).get("status") or "") or None,
            "can_delete": not block_reason,
            "allowed_actions": ["delete"] if not block_reason else [],
            "delete_block_reason": block_reason,
        }
    )
    return record


class FileQueries:
    async def list_pending_important_file_todos(
        self, profile_id: str, instance_id: str, *, limit: int = 3
    ) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT t.todo_id, t.kind, t.payload_json, t.status,
                t.available_at, f.display_name, f.file_format, f.asset_id,
                f.storage_relpath, f.sha256, f.byte_size, f.mime_type
            FROM important_todos t JOIN role_profiles p
              ON p.profile_id = t.profile_id AND p.file_artifacts_enabled = 1
            LEFT JOIN file_assets f
              ON f.asset_id = t.file_asset_id
            WHERE t.profile_id = ? AND t.instance_id = ? AND t.status = 'PENDING'
              AND t.available_at <= ? ORDER BY t.available_at, t.created_at LIMIT ?""",
            (profile_id, instance_id, _dt(_now()), max(1, min(int(limit), 3))),
        )
        return [self._record(row, json_columns=("payload_json",)) for row in rows]

    async def get_file_assets_for_todos(
        self, profile_id: str, instance_id: str, todo_ids: Sequence[str]
    ) -> list[dict[str, Any]]:
        ids = list(dict.fromkeys(str(item) for item in todo_ids if str(item)))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = await self.db.fetch_all(
            f"""SELECT t.todo_id, t.kind, t.payload_json, t.status AS todo_status,
                f.* FROM important_todos t LEFT JOIN file_assets f
                  ON f.asset_id = t.file_asset_id
                WHERE t.profile_id = ? AND t.instance_id = ?
                  AND t.todo_id IN ({placeholders})""",
            (profile_id, instance_id, *ids),
        )
        return [self._record(row, json_columns=("payload_json", "metadata_json")) for row in rows]

    async def settle_file_todos(
        self,
        profile_id: str,
        instance_id: str,
        todo_ids: Sequence[str],
        *,
        status: str,
        error: str = "",
    ) -> int:
        normalized = str(status).upper()
        if normalized not in {
            "PENDING",
            "DELIVERY_PENDING",
            "DELIVERY_UNKNOWN",
            "COMPLETED",
            "CANCELLED",
        }:
            raise ValueError("unsupported file todo settlement")
        ids = list(dict.fromkeys(str(item) for item in todo_ids if str(item)))
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                f"""UPDATE important_todos SET status = ?,
                    selected_run_id = CASE WHEN ? = 'PENDING' THEN NULL ELSE selected_run_id END,
                    selected_activity_epoch = CASE WHEN ? = 'PENDING' THEN NULL ELSE selected_activity_epoch END,
                    delivery_outbox_id = CASE
                        WHEN ? = 'PENDING' THEN NULL ELSE delivery_outbox_id END,
                    resolved_at = CASE WHEN ? IN ('COMPLETED', 'CANCELLED') THEN ? ELSE NULL END,
                    version = version + 1, updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND todo_id IN ({placeholders})
                  AND status IN ('PENDING', 'SELECTED', 'DELIVERY_PENDING')""",
                (
                    normalized,
                    normalized,
                    normalized,
                    normalized,
                    normalized,
                    now,
                    now,
                    profile_id,
                    instance_id,
                    *ids,
                ),
            )
            delivery = {
                "PENDING": "NOT_SELECTED",
                "DELIVERY_PENDING": "OUTBOX_PENDING",
                "DELIVERY_UNKNOWN": "PLATFORM_ACCEPTED_UNCONFIRMED",
                "COMPLETED": "DELIVERED",
                "CANCELLED": "FAILED",
            }[normalized]
            conn.execute(
                f"""UPDATE file_assets SET delivery_status = ?, last_error = ?,
                    updated_at = ? WHERE asset_id IN (
                        SELECT file_asset_id FROM important_todos
                        WHERE profile_id = ? AND instance_id = ?
                          AND todo_id IN ({placeholders})
                    )""",
                (delivery, str(error or "")[:600], now, profile_id, instance_id, *ids),
            )
            if normalized in {"DELIVERY_UNKNOWN", "COMPLETED", "CANCELLED"}:
                conn.execute(
                    f"""UPDATE file_assets SET file_status = 'RELEASE_PENDING',
                        last_error = 'expired_by_retention', updated_at = ?
                    WHERE file_status = 'AVAILABLE' AND expires_at <= ?
                      AND asset_id IN (
                        SELECT file_asset_id FROM important_todos
                        WHERE profile_id = ? AND instance_id = ?
                          AND todo_id IN ({placeholders})
                    )""",
                    (now, now, profile_id, instance_id, *ids),
                )
            return int(cursor.rowcount)

        return await self.uow.run(operation)

    async def list_file_artifact_records(
        self, profile_id: str, instance_id: str, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT j.job_id, j.ai_task_id, j.file_format,
                j.display_name AS requested_display_name,
                j.status AS job_status, j.safe_error_code,
                j.safe_error_message, j.created_at, j.updated_at, j.finished_at,
                t.status AS task_status, f.asset_id, f.display_name,
                f.mime_type, f.storage_relpath, f.byte_size, f.char_count,
                f.page_count, f.file_status, f.delivery_status, f.last_error,
                f.expires_at, f.released_at, x.todo_id, x.kind AS todo_kind,
                x.status AS todo_status, x.resolved_at
            FROM file_generation_jobs j
            JOIN ai_tasks t ON t.task_id = j.ai_task_id
            LEFT JOIN file_assets f ON f.job_id = j.job_id
            LEFT JOIN important_todos x ON x.source_job_id = j.job_id
            WHERE j.profile_id = ? AND j.instance_id = ?
            ORDER BY j.created_at DESC LIMIT ?""",
            (profile_id, instance_id, max(1, min(int(limit), 500))),
        )
        outbox_rows = await self.db.fetch_all(
            """SELECT outbox_id, payload_json, status, attempts, last_error,
                idempotency_key, updated_at
            FROM instance_outbox WHERE profile_id = ? AND instance_id = ?
            ORDER BY outbox_id DESC""",
            (profile_id, instance_id),
        )
        parsed_outboxes, outbox_by_key = _parse_file_outboxes(outbox_rows)
        return [
            _decorate_file_record(
                self._record(row, json_columns=()), parsed_outboxes, outbox_by_key
            )
            for row in rows
        ]
