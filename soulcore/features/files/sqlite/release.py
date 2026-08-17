from __future__ import annotations

from datetime import datetime, timedelta

from ..ports import PendingFileArtifactRelease
from .support import (
    Any,
    Mapping,
    OutboxStatus,
    _dt,
    _dump,
    _load,
    _now,
    _parse,
    sqlite3,
    uuid,
)

ParsedOutbox = tuple[sqlite3.Row, dict[str, Any], set[str], set[str]]
RelatedOutbox = tuple[sqlite3.Row, dict[str, Any], set[str]]


def _release_asset_row(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    asset_id: str,
) -> sqlite3.Row:
    row = conn.execute(
        """SELECT f.*, x.todo_id, x.status AS todo_status
        FROM file_assets f JOIN important_todos x ON x.file_asset_id = f.asset_id
        WHERE f.profile_id = ? AND f.instance_id = ? AND f.asset_id = ?""",
        (profile_id, instance_id, asset_id),
    ).fetchone()
    if row is None:
        raise ValueError("文件成果不存在或不属于当前实例")
    if row["delivery_status"] == "DELIVERED":
        raise ValueError("文件已经送达，不能作为未投递文件删除")
    if row["delivery_status"] in {
        "PLATFORM_ACCEPTED_UNCONFIRMED",
        "PARTIALLY_ATTEMPTED",
        "UNKNOWN_AFTER_CRASH",
    }:
        raise ValueError("平台投递结果未知，不能确认文件尚未送达")
    return row


def _parse_outbox(row: sqlite3.Row) -> ParsedOutbox:
    payload = _load(row["payload_json"]) or {}
    todo_ids = {str(value) for value in payload.get("important_todo_ids") or [] if str(value)}
    components = [
        item
        for item in payload.get("components") or []
        if isinstance(item, Mapping) and str(item.get("type") or "") == "file_artifact"
    ]
    asset_ids = {str(item.get("asset_id") or "") for item in components}
    return row, payload, todo_ids, asset_ids


def _release_outboxes(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
) -> list[ParsedOutbox]:
    rows = conn.execute(
        """SELECT * FROM instance_outbox
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    )
    return [_parse_outbox(row) for row in rows]


def _paired_release_outboxes(
    parsed: list[ParsedOutbox],
    target_todo_id: str,
    asset_id: str,
) -> list[RelatedOutbox]:
    related: list[RelatedOutbox] = [
        (row, payload, todos)
        for row, payload, todos, assets in parsed
        if target_todo_id in todos or asset_id in assets
    ]
    by_key = {str(item[0]["idempotency_key"]): item for item in parsed}
    announcements: dict[str, list[ParsedOutbox]] = {}
    for item in parsed:
        followup = str(item[1].get("file_followup_idempotency_key") or "").strip()
        if followup:
            announcements.setdefault(followup, []).append(item)
    related_ids = {int(item[0]["outbox_id"]) for item in related}
    for outbox, payload, _todos in tuple(related):
        dependency_key = str(payload.get("depends_on_idempotency_key") or "").strip()
        if not dependency_key:
            continue
        dependency = by_key.get(dependency_key)
        reverse = announcements.get(str(outbox["idempotency_key"]), [])
        _validate_release_pair(outbox, payload, dependency, reverse)
        assert dependency is not None
        announcement_id = int(dependency[0]["outbox_id"])
        if announcement_id not in related_ids:
            related.append((dependency[0], dependency[1], dependency[2]))
            related_ids.add(announcement_id)
    return related


def _validate_release_pair(
    outbox: sqlite3.Row,
    payload: dict[str, Any],
    dependency: ParsedOutbox | None,
    reverse: list[ParsedOutbox],
) -> None:
    if dependency is None or len(reverse) != 1:
        raise ValueError("文件说明与附件的配对状态不完整，当前不能删除")
    announcement, announcement_payload, _todos, _assets = dependency
    pairs_match = all(
        (
            int(reverse[0][0]["outbox_id"]) == int(announcement["outbox_id"]),
            str(announcement_payload.get("file_delivery_role") or "") == "ANNOUNCEMENT",
            str(announcement_payload.get("file_followup_idempotency_key") or "").strip()
            == str(outbox["idempotency_key"]),
            str(announcement["route_umo"] or "") == str(outbox["route_umo"] or ""),
            int(announcement["activity_epoch"] or 0) == int(outbox["activity_epoch"] or 0),
            int(announcement["origin_run_id"] or 0) == int(outbox["origin_run_id"] or 0),
        )
    )
    if not pairs_match:
        raise ValueError("文件说明与附件的配对状态不完整，当前不能删除")


def _validate_related_outboxes(related: list[RelatedOutbox], todo_status: str) -> None:
    for outbox, _payload, _todos in related:
        if int(outbox["attempts"] or 0) > 0:
            raise ValueError("该文件已经发生过平台投递尝试，不能确认尚未送达")
        blocked = {
            OutboxStatus.SENDING.value,
            OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED.value,
            OutboxStatus.PARTIALLY_ATTEMPTED.value,
            OutboxStatus.UNKNOWN_AFTER_CRASH.value,
        }
        if outbox["status"] in blocked:
            raise ValueError("文件已经进入平台投递且结果未确定，当前不能删除")
    if todo_status in {"SELECTED", "DELIVERY_PENDING"} and not related:
        raise ValueError("文件已被运行选中且投递尚未结算，当前不能删除")
    allowed = {"PENDING", "SELECTED", "DELIVERY_PENDING", "CANCELLED"}
    if todo_status not in allowed:
        raise ValueError("当前待办状态不允许删除")


def _cancel_related_outboxes(
    conn: sqlite3.Connection,
    related: list[RelatedOutbox],
    target_todo_id: str,
    now: str,
) -> set[str]:
    retry_todos: set[str] = set()
    for outbox, payload, todo_ids in related:
        if outbox["status"] == OutboxStatus.PENDING.value:
            cursor = conn.execute(
                """UPDATE instance_outbox SET status = 'FAILED',
                last_error = 'cancelled_by_file_artifact_admin', updated_at = ?
                WHERE outbox_id = ? AND status = 'PENDING' AND attempts = 0""",
                (now, outbox["outbox_id"]),
            )
            if cursor.rowcount != 1:
                raise ValueError("文件投递状态已经变化，请刷新后重试")
        if str(payload.get("file_delivery_role") or "") != "ANNOUNCEMENT":
            retry_todos.update(todo_ids - {target_todo_id})
    return retry_todos


def _requeue_companion_todos(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    retry_todos: set[str],
    now: str,
) -> None:
    if not retry_todos:
        return
    identifiers = sorted(retry_todos)
    placeholders = ",".join("?" for _ in identifiers)
    conn.execute(
        f"""UPDATE important_todos SET status = 'PENDING',
        selected_run_id = NULL, selected_activity_epoch = NULL,
        resolved_at = NULL, version = version + 1, updated_at = ?
        WHERE profile_id = ? AND instance_id = ? AND todo_id IN ({placeholders})
          AND status IN ('SELECTED', 'DELIVERY_PENDING')""",
        (now, profile_id, instance_id, *identifiers),
    )
    conn.execute(
        f"""UPDATE file_assets SET delivery_status = 'NOT_SELECTED',
        last_error = 'companion_outbox_cancelled_by_file_deletion', updated_at = ?
        WHERE asset_id IN (SELECT file_asset_id FROM important_todos
        WHERE profile_id = ? AND instance_id = ? AND todo_id IN ({placeholders}))""",
        (now, profile_id, instance_id, *identifiers),
    )
    for todo_id in identifiers:
        _schedule_companion_retry(conn, profile_id, instance_id, todo_id, now)


def _schedule_companion_retry(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    todo_id: str,
    now: str,
) -> None:
    conn.execute(
        """INSERT INTO instance_wakeups(
            profile_id, instance_id, source, due_at, reason,
            conversation_ref, idempotency_key, payload_json,
            status, intent_kind, created_at, updated_at
        ) SELECT ?, ?, 'PLUGIN_WAKE', ?, ?, ci.route_umo, ?, ?,
            'PENDING', 'PLUGIN_WAKE', ?, ? FROM character_instances ci
        WHERE ci.profile_id = ? AND ci.instance_id = ?""",
        (
            profile_id,
            instance_id,
            now,
            "关联文件删除后，仍有重要文件待办需要重新处理。",
            f"important-todo-retry:{todo_id}:{uuid.uuid4().hex}",
            _dump({"todo_id": todo_id}),
            now,
            now,
            profile_id,
            instance_id,
        ),
    )


def _mark_file_release_pending(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    profile_id: str,
    instance_id: str,
    asset_id: str,
    now: str,
) -> None:
    todo_id = str(row["todo_id"])
    cursor = conn.execute(
        """UPDATE important_todos SET status = 'CANCELLED', resolved_at = ?,
        version = version + 1, updated_at = ? WHERE todo_id = ?
        AND status IN ('PENDING', 'SELECTED', 'DELIVERY_PENDING')""",
        (now, now, todo_id),
    )
    if str(row["todo_status"] or "") != "CANCELLED" and cursor.rowcount != 1:
        raise ValueError("文件待办状态已经变化，请刷新后重试")
    conn.execute(
        """UPDATE file_assets SET file_status = 'RELEASE_PENDING',
        delivery_status = 'FAILED', last_error = 'deleted_by_admin',
        updated_at = ? WHERE asset_id = ?""",
        (now, asset_id),
    )
    conn.execute(
        """UPDATE instance_wakeups SET status = 'CANCELLED', lease_until = NULL,
        last_error = 'file_artifact_deleted_by_admin', updated_at = ?
        WHERE profile_id = ? AND instance_id = ? AND idempotency_key = ?
          AND status IN ('PENDING', 'CLAIMED')""",
        (now, profile_id, instance_id, f"important-todo:{todo_id}"),
    )


class FileReleaseCommands:
    async def prepare_expired_file_artifact_releases(
        self,
        *,
        expired_before: datetime | None = None,
        limit: int = 100,
    ) -> int:
        cutoff = _dt(expired_before or _now())
        bounded_limit = max(1, min(int(limit), 1000))

        def operation(conn: sqlite3.Connection) -> int:
            candidates = conn.execute(
                """SELECT f.profile_id, f.instance_id, f.asset_id,
                    t.todo_id, t.status AS todo_status, t.delivery_outbox_id
                FROM file_assets f
                JOIN important_todos t ON t.file_asset_id = f.asset_id
                WHERE f.file_status = 'AVAILABLE' AND f.expires_at <= ?
                ORDER BY f.expires_at, f.asset_id""",
                (cutoff,),
            ).fetchall()
            if not candidates:
                return 0

            active_by_instance: dict[tuple[str, str], tuple[set[int], set[str], set[str]]] = {}
            for row in candidates:
                scope = (str(row["profile_id"]), str(row["instance_id"]))
                if scope in active_by_instance:
                    continue
                outboxes = conn.execute(
                    """SELECT * FROM instance_outbox
                    WHERE profile_id = ? AND instance_id = ?
                      AND status IN ('PENDING', 'SENDING')""",
                    scope,
                ).fetchall()
                outbox_ids: set[int] = set()
                todo_ids: set[str] = set()
                asset_ids: set[str] = set()
                for outbox in outboxes:
                    parsed, _payload, todos, assets = _parse_outbox(outbox)
                    outbox_ids.add(int(parsed["outbox_id"]))
                    todo_ids.update(todos)
                    asset_ids.update(assets)
                active_by_instance[scope] = (outbox_ids, todo_ids, asset_ids)

            prepared = 0
            for row in candidates:
                if prepared >= bounded_limit:
                    break
                profile_id = str(row["profile_id"])
                instance_id = str(row["instance_id"])
                asset_id = str(row["asset_id"])
                todo_id = str(row["todo_id"])
                todo_status = str(row["todo_status"] or "")
                outbox_ids, active_todos, active_assets = active_by_instance[
                    (profile_id, instance_id)
                ]
                delivery_outbox_id = row["delivery_outbox_id"]
                actively_delivering = any(
                    (
                        delivery_outbox_id is not None and int(delivery_outbox_id) in outbox_ids,
                        todo_id in active_todos,
                        asset_id in active_assets,
                    )
                )
                if actively_delivering:
                    continue

                if todo_status == "PENDING":
                    conn.execute(
                        """UPDATE important_todos SET status = 'CANCELLED',
                            resolved_at = ?, version = version + 1, updated_at = ?
                        WHERE profile_id = ? AND instance_id = ? AND todo_id = ?
                          AND status = 'PENDING'""",
                        (cutoff, cutoff, profile_id, instance_id, todo_id),
                    )
                    conn.execute(
                        """UPDATE instance_wakeups SET status = 'CANCELLED',
                            lease_until = NULL,
                            last_error = 'file_artifact_retention_elapsed',
                            updated_at = ?
                        WHERE profile_id = ? AND instance_id = ?
                          AND idempotency_key = ?
                          AND status IN ('PENDING', 'CLAIMED')""",
                        (
                            cutoff,
                            profile_id,
                            instance_id,
                            f"important-todo:{todo_id}",
                        ),
                    )

                cursor = conn.execute(
                    """UPDATE file_assets SET file_status = 'RELEASE_PENDING',
                        delivery_status = CASE
                            WHEN delivery_status IN (
                                'NOT_SELECTED', 'SELECTED', 'OUTBOX_PENDING'
                            ) THEN 'FAILED'
                            ELSE delivery_status
                        END,
                        last_error = 'expired_by_retention', updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND asset_id = ?
                      AND file_status = 'AVAILABLE' AND expires_at <= ?""",
                    (cutoff, profile_id, instance_id, asset_id, cutoff),
                )
                prepared += int(cursor.rowcount)
            return prepared

        prepared = await self.uow.run(operation)
        if prepared:
            await self.db.publish_backup_after_commit()
        return prepared

    async def list_pending_file_artifact_releases(
        self,
        *,
        limit: int = 100,
    ) -> list[PendingFileArtifactRelease]:
        rows = await self.db.fetch_all(
            """SELECT profile_id, instance_id, asset_id, storage_relpath,
                byte_size, sha256, updated_at
            FROM file_assets WHERE file_status = 'RELEASE_PENDING'
            ORDER BY updated_at, asset_id LIMIT ?""",
            (max(1, min(int(limit), 1000)),),
        )
        return [
            PendingFileArtifactRelease(
                profile_id=str(row["profile_id"]),
                instance_id=str(row["instance_id"]),
                asset_id=str(row["asset_id"]),
                storage_relpath=str(row["storage_relpath"]),
                byte_size=int(row["byte_size"] or 0),
                sha256=str(row["sha256"]),
                updated_at=str(row["updated_at"]),
            )
            for row in rows
        ]

    async def settle_pending_file_artifact_release(
        self,
        profile_id: str,
        instance_id: str,
        asset_id: str,
        *,
        expected_updated_at: str,
        released: bool,
        error: str = "",
    ) -> str:
        generation = str(expected_updated_at or "").strip()
        if not generation:
            raise ValueError("pending file release generation is required")
        observed_at = _parse(generation)
        settled_at = _now()
        if observed_at is not None and settled_at <= observed_at:
            settled_at = observed_at + timedelta(microseconds=1)
        now = _dt(settled_at)

        def operation(conn: sqlite3.Connection) -> str:
            cursor = conn.execute(
                """UPDATE file_assets SET file_status = ?, released_at = ?,
                    last_error = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND asset_id = ?
                  AND file_status = 'RELEASE_PENDING' AND updated_at = ?""",
                (
                    "RELEASED" if released else "RELEASE_PENDING",
                    now if released else None,
                    str(error or "")[:600],
                    now,
                    profile_id,
                    instance_id,
                    str(asset_id),
                    generation,
                ),
            )
            if cursor.rowcount == 1:
                return "APPLIED"
            current = conn.execute(
                """SELECT file_status FROM file_assets
                WHERE profile_id = ? AND instance_id = ? AND asset_id = ?""",
                (profile_id, instance_id, str(asset_id)),
            ).fetchone()
            if current is None:
                return "SUPERSEDED_MISSING"
            if str(current["file_status"]) == "RELEASED":
                return "SUPERSEDED_RELEASED"
            return "SUPERSEDED_PENDING"

        result = await self.uow.run(operation)
        if result == "APPLIED" and released:
            await self.db.publish_backup_after_commit()
        return result

    async def prepare_file_artifact_release(
        self, profile_id: str, instance_id: str, asset_id: str
    ) -> dict[str, Any]:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> dict[str, Any]:
            row = _release_asset_row(conn, profile_id, instance_id, str(asset_id))
            if row["file_status"] == "RELEASED":
                return {**dict(row), "already_released": True}
            target_todo_id = str(row["todo_id"])
            parsed = _release_outboxes(conn, profile_id, instance_id)
            related = _paired_release_outboxes(parsed, target_todo_id, str(asset_id))
            todo_status = str(row["todo_status"] or "")
            _validate_related_outboxes(related, todo_status)
            retry_todos = _cancel_related_outboxes(conn, related, target_todo_id, now)
            _requeue_companion_todos(conn, profile_id, instance_id, retry_todos, now)
            _mark_file_release_pending(conn, row, profile_id, instance_id, str(asset_id), now)
            return {**dict(row), "already_released": False}

        result = await self.uow.run(operation)
        await self.db.publish_backup_after_commit()
        return result

    async def finalize_file_artifact_release(
        self,
        profile_id: str,
        instance_id: str,
        asset_id: str,
        *,
        released: bool,
        error: str = "",
    ) -> bool:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                """UPDATE file_assets SET file_status = ?, released_at = ?,
                    last_error = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND asset_id = ?
                  AND file_status = 'RELEASE_PENDING'""",
                (
                    "RELEASED" if released else "RELEASE_PENDING",
                    now if released else None,
                    str(error or "")[:600],
                    now,
                    profile_id,
                    instance_id,
                    str(asset_id),
                ),
            )
            if cursor.rowcount == 1:
                return True
            current = conn.execute(
                """SELECT file_status FROM file_assets
                WHERE profile_id = ? AND instance_id = ? AND asset_id = ?""",
                (profile_id, instance_id, str(asset_id)),
            ).fetchone()
            return bool(current is not None and str(current["file_status"]) == "RELEASED")

        finalized = await self.uow.run(operation)
        if released and finalized:
            await self.db.publish_backup_after_commit()
        return finalized
