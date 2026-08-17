from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ....contracts.delivery_visibility import (
    FOREGROUND_DELIVERY_BOUNDARY_PREPARED,
    FOREGROUND_DELIVERY_BOUNDARY_PREPARING,
    foreground_delivery_boundary,
    foreground_delivery_todo_ids,
    is_dialogue_continuity_visible,
    outbox_todo_ids,
)
from ..domain import (
    MediaAsset,
    MediaFileStatus,
    MediaInspectionStatus,
    MediaOrigin,
    MediaProjection,
    MediaProjectionStatus,
    MediaPurpose,
)
from ..inspection import _dt, _dump, _load, _now, _parse
from .release_planning import MediaReleasePlanningCommands

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_INBOUND_ATTACHMENT_BYTES = 50 * 1024 * 1024
MAX_ANIMATION_DURATION_MS = 30_000
MAX_ANIMATION_DECODED_PIXELS = 240_000_000
_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_MIME_ALIASES = {"image/jpg": "image/jpeg", "image/x-png": "image/png"}
_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_INBOUND_MEDIA_KINDS = frozenset({"audio", "file", "video"})


@dataclass(slots=True)
class _InterruptedForegroundPlan:
    messages: tuple[sqlite3.Row, ...]
    prepared_message_ids: set[int]
    uncertain_message_ids: set[int]
    prepared_media_refs: set[tuple[str, str, str]]
    uncertain_media_refs: set[tuple[str, str, str]]
    prepared_file_refs: set[tuple[str, str, str]]
    uncertain_file_refs: set[tuple[str, str, str]]
    prepared_todo_refs: set[tuple[str, str, str]]
    uncertain_todo_refs: set[tuple[str, str, str]]
    active_outbox_file_refs: set[tuple[str, str, str]]
    active_outbox_todo_refs: set[tuple[str, str, str]]


def _foreground_plan_targets(
    plan: _InterruptedForegroundPlan,
    *,
    prepared: bool,
) -> tuple[
    set[int],
    set[tuple[str, str, str]],
    set[tuple[str, str, str]],
    set[tuple[str, str, str]],
]:
    if prepared:
        return (
            plan.prepared_message_ids,
            plan.prepared_media_refs,
            plan.prepared_file_refs,
            plan.prepared_todo_refs,
        )
    return (
        plan.uncertain_message_ids,
        plan.uncertain_media_refs,
        plan.uncertain_file_refs,
        plan.uncertain_todo_refs,
    )


def _message_component_refs(row: sqlite3.Row) -> list[tuple[str, tuple[str, str, str]]]:
    components = _load(row["components_json"]) or []
    if not isinstance(components, list):
        return []
    refs: list[tuple[str, tuple[str, str, str]]] = []
    for component in components:
        if not isinstance(component, dict):
            continue
        asset_id = str(component.get("asset_id") or "")
        if not asset_id:
            continue
        refs.append(
            (
                str(component.get("type") or ""),
                (str(row["profile_id"]), str(row["instance_id"]), asset_id),
            )
        )
    return refs


class _RecoverInterruptedMediaDelivery:
    def __init__(
        self,
        *,
        stale_before: datetime | None,
        now: str,
        expires: str,
    ) -> None:
        self.stale_before = stale_before
        self.now = now
        self.expires = expires

    def __call__(self, conn: sqlite3.Connection) -> dict[str, int]:
        plan = self._build_plan(conn)
        assets = self._settle_media(conn, plan)
        files, todos = self._settle_files(conn, plan)
        messages = self._settle_messages(conn, plan)
        return {
            "messages": messages,
            "assets": assets,
            "files": files,
            "todos": todos,
        }

    def _build_plan(self, conn: sqlite3.Connection) -> _InterruptedForegroundPlan:
        message_query = """SELECT message.message_id, message.profile_id,
                message.instance_id, message.components_json, message.metadata_json
            FROM instance_messages message
            WHERE message.direction = 'OUTBOUND'
              AND message.delivery_status = 'PENDING'
              AND NOT EXISTS (
                SELECT 1 FROM instance_outbox outbox
                WHERE outbox.context_message_id = message.message_id
              )"""
        parameters: tuple[object, ...] = ()
        if self.stale_before is not None:
            message_query += " AND message.created_at <= ?"
            parameters = (_dt(self.stale_before),)
        messages = tuple(conn.execute(message_query, parameters))
        plan = _InterruptedForegroundPlan(
            messages=messages,
            prepared_message_ids=set(),
            uncertain_message_ids=set(),
            prepared_media_refs=set(),
            uncertain_media_refs=set(),
            prepared_file_refs=set(),
            uncertain_file_refs=set(),
            prepared_todo_refs=set(),
            uncertain_todo_refs=set(),
            active_outbox_file_refs=set(),
            active_outbox_todo_refs=set(),
        )
        for row in messages:
            self._classify_message(plan, row)
        self._add_active_outbox_ownership(conn, plan)
        self._add_linked_media(conn, plan)
        plan.prepared_media_refs.difference_update(plan.uncertain_media_refs)
        plan.prepared_file_refs.difference_update(plan.uncertain_file_refs)
        self._add_file_todos(conn, plan)
        plan.prepared_todo_refs.difference_update(plan.uncertain_todo_refs)
        plan.prepared_file_refs.difference_update(plan.active_outbox_file_refs)
        plan.uncertain_file_refs.difference_update(plan.active_outbox_file_refs)
        plan.prepared_todo_refs.difference_update(plan.active_outbox_todo_refs)
        plan.uncertain_todo_refs.difference_update(plan.active_outbox_todo_refs)
        return plan

    @staticmethod
    def _add_active_outbox_ownership(
        conn: sqlite3.Connection, plan: _InterruptedForegroundPlan
    ) -> None:
        for row in conn.execute(
            """SELECT profile_id, instance_id, payload_json
            FROM instance_outbox WHERE status IN ('PENDING', 'SENDING')"""
        ):
            payload = _load(row["payload_json"]) or {}
            if not isinstance(payload, dict):
                continue
            profile_id = str(row["profile_id"])
            instance_id = str(row["instance_id"])
            plan.active_outbox_todo_refs.update(
                (profile_id, instance_id, todo_id) for todo_id in outbox_todo_ids(payload)
            )
            components = payload.get("components")
            if not isinstance(components, list):
                continue
            plan.active_outbox_file_refs.update(
                (profile_id, instance_id, str(component.get("asset_id") or ""))
                for component in components
                if isinstance(component, dict)
                and str(component.get("type") or "") == "file_artifact"
                and str(component.get("asset_id") or "")
            )

    @staticmethod
    def _classify_message(plan: _InterruptedForegroundPlan, row: sqlite3.Row) -> None:
        message_id = int(row["message_id"])
        metadata = _load(row["metadata_json"]) or {}
        prepared = isinstance(metadata, dict) and foreground_delivery_boundary(metadata) in {
            FOREGROUND_DELIVERY_BOUNDARY_PREPARED,
            FOREGROUND_DELIVERY_BOUNDARY_PREPARING,
        }
        target_messages, target_media, target_files, target_todos = _foreground_plan_targets(
            plan,
            prepared=prepared,
        )
        target_messages.add(message_id)
        if isinstance(metadata, dict):
            target_todos.update(
                (
                    str(row["profile_id"]),
                    str(row["instance_id"]),
                    todo_id,
                )
                for todo_id in foreground_delivery_todo_ids(metadata)
            )
        for component_type, ref in _message_component_refs(row):
            if component_type == "image_asset":
                target_media.add(ref)
            elif component_type == "file_artifact":
                target_files.add(ref)

    @staticmethod
    def _add_linked_media(conn: sqlite3.Connection, plan: _InterruptedForegroundPlan) -> None:
        message_ids = tuple(int(row["message_id"]) for row in plan.messages)
        if not message_ids:
            return
        placeholders = ",".join("?" for _ in message_ids)
        for link in conn.execute(
            f"""SELECT link.profile_id, link.instance_id, link.asset_id,
                link.message_id FROM media_asset_message_links link
            WHERE link.message_id IN ({placeholders})""",
            message_ids,
        ):
            ref = (
                str(link["profile_id"]),
                str(link["instance_id"]),
                str(link["asset_id"]),
            )
            target = (
                plan.prepared_media_refs
                if int(link["message_id"]) in plan.prepared_message_ids
                else plan.uncertain_media_refs
            )
            target.add(ref)

    @staticmethod
    def _add_file_todos(conn: sqlite3.Connection, plan: _InterruptedForegroundPlan) -> None:
        for prepared, refs in (
            (True, plan.prepared_file_refs),
            (False, plan.uncertain_file_refs),
        ):
            target = plan.prepared_todo_refs if prepared else plan.uncertain_todo_refs
            for profile_id, instance_id, asset_id in refs:
                target.update(
                    (profile_id, instance_id, str(row["todo_id"]))
                    for row in conn.execute(
                        """SELECT todo_id FROM important_todos
                        WHERE profile_id = ? AND instance_id = ?
                          AND file_asset_id = ?""",
                        (profile_id, instance_id, asset_id),
                    )
                )

    def _settle_media(self, conn: sqlite3.Connection, plan: _InterruptedForegroundPlan) -> int:
        changed = 0
        for profile_id, instance_id, asset_id in sorted(plan.uncertain_media_refs):
            changed += int(
                conn.execute(
                    """UPDATE media_assets SET delivery_status = 'UNKNOWN_AFTER_CRASH',
                        expires_at = COALESCE(expires_at, ?),
                        last_error = 'delivery_interrupted_after_platform_boundary',
                        updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND asset_id = ?
                      AND delivery_status = 'SELECTED'
                      AND NOT EXISTS (
                        SELECT 1 FROM instance_outbox delivery
                        WHERE delivery.profile_id = media_assets.profile_id
                          AND delivery.instance_id = media_assets.instance_id
                          AND delivery.origin_run_id = media_assets.core_run_id
                          AND delivery.status IN ('PENDING', 'SENDING')
                      )""",
                    (self.expires, self.now, profile_id, instance_id, asset_id),
                ).rowcount
            )
        for profile_id, instance_id, asset_id in sorted(plan.prepared_media_refs):
            changed += int(
                conn.execute(
                    """UPDATE media_assets SET delivery_status = 'FAILED',
                        expires_at = COALESCE(expires_at, ?),
                        last_error = 'delivery_interrupted_before_platform_call',
                        updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND asset_id = ?
                      AND delivery_status = 'SELECTED'
                      AND NOT EXISTS (
                        SELECT 1 FROM instance_outbox delivery
                        WHERE delivery.profile_id = media_assets.profile_id
                          AND delivery.instance_id = media_assets.instance_id
                          AND delivery.origin_run_id = media_assets.core_run_id
                          AND delivery.status IN ('PENDING', 'SENDING')
                      )""",
                    (self.expires, self.now, profile_id, instance_id, asset_id),
                ).rowcount
            )
        if self.stale_before is None:
            changed += self._settle_orphan_media(conn)
        return changed

    def _settle_orphan_media(self, conn: sqlite3.Connection) -> int:
        return int(
            conn.execute(
                """UPDATE media_assets AS asset SET delivery_status = 'FAILED',
                    expires_at = COALESCE(expires_at, ?),
                    last_error = 'delivery_interrupted_before_foreground_ledger',
                    updated_at = ?
                WHERE asset.origin = 'GENERATED'
                  AND asset.delivery_status = 'SELECTED'
                  AND NOT EXISTS (
                    SELECT 1 FROM media_asset_message_links link
                    JOIN instance_messages message
                      ON message.profile_id = link.profile_id
                     AND message.instance_id = link.instance_id
                     AND message.message_id = link.message_id
                    WHERE link.asset_id = asset.asset_id
                      AND message.delivery_status = 'PENDING'
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM instance_outbox delivery
                    WHERE delivery.profile_id = asset.profile_id
                      AND delivery.instance_id = asset.instance_id
                      AND delivery.origin_run_id = asset.core_run_id
                      AND delivery.status IN ('PENDING', 'SENDING')
                  )""",
                (self.expires, self.now),
            ).rowcount
        )

    def _settle_files(
        self, conn: sqlite3.Connection, plan: _InterruptedForegroundPlan
    ) -> tuple[int, int]:
        files = 0
        todos = 0
        for profile_id, instance_id, todo_id in sorted(plan.uncertain_todo_refs):
            todos += int(
                conn.execute(
                    """UPDATE important_todos SET status = 'DELIVERY_UNKNOWN',
                        resolved_at = NULL, version = version + 1, updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND todo_id = ?
                      AND status IN ('SELECTED', 'DELIVERY_PENDING')
                      AND NOT EXISTS (
                        SELECT 1 FROM instance_outbox delivery
                        WHERE delivery.outbox_id = important_todos.delivery_outbox_id
                          AND delivery.status IN ('PENDING', 'SENDING')
                      )""",
                    (self.now, profile_id, instance_id, todo_id),
                ).rowcount
            )
        for profile_id, instance_id, todo_id in sorted(plan.prepared_todo_refs):
            todos += int(
                conn.execute(
                    """UPDATE important_todos SET status = 'PENDING',
                        selected_run_id = NULL, selected_activity_epoch = NULL,
                        delivery_outbox_id = NULL, resolved_at = NULL,
                        version = version + 1, updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND todo_id = ?
                      AND status IN ('SELECTED', 'DELIVERY_PENDING')
                      AND NOT EXISTS (
                        SELECT 1 FROM instance_outbox delivery
                        WHERE delivery.outbox_id = important_todos.delivery_outbox_id
                          AND delivery.status IN ('PENDING', 'SENDING')
                      )""",
                    (self.now, profile_id, instance_id, todo_id),
                ).rowcount
            )
        for profile_id, instance_id, asset_id in sorted(plan.uncertain_file_refs):
            files += int(
                conn.execute(
                    """UPDATE file_assets SET delivery_status = 'UNKNOWN_AFTER_CRASH',
                        last_error = 'delivery_interrupted_after_platform_boundary',
                        updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND asset_id = ?
                      AND delivery_status IN ('SELECTED', 'OUTBOX_PENDING')
                      AND NOT EXISTS (
                        SELECT 1 FROM important_todos todo
                        JOIN instance_outbox delivery
                          ON delivery.outbox_id = todo.delivery_outbox_id
                        WHERE todo.profile_id = file_assets.profile_id
                          AND todo.instance_id = file_assets.instance_id
                          AND todo.file_asset_id = file_assets.asset_id
                          AND delivery.status IN ('PENDING', 'SENDING')
                      )""",
                    (self.now, profile_id, instance_id, asset_id),
                ).rowcount
            )
            todos += int(
                conn.execute(
                    """UPDATE important_todos SET status = 'DELIVERY_UNKNOWN',
                        resolved_at = NULL, version = version + 1, updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND file_asset_id = ?
                      AND status IN ('SELECTED', 'DELIVERY_PENDING')
                      AND NOT EXISTS (
                        SELECT 1 FROM instance_outbox delivery
                        WHERE delivery.outbox_id = important_todos.delivery_outbox_id
                          AND delivery.status IN ('PENDING', 'SENDING')
                      )""",
                    (self.now, profile_id, instance_id, asset_id),
                ).rowcount
            )
        for profile_id, instance_id, asset_id in sorted(plan.prepared_file_refs):
            todos += int(
                conn.execute(
                    """UPDATE important_todos SET status = 'PENDING',
                        selected_run_id = NULL, selected_activity_epoch = NULL,
                        delivery_outbox_id = NULL, resolved_at = NULL,
                        version = version + 1, updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND file_asset_id = ?
                      AND status IN ('SELECTED', 'DELIVERY_PENDING')
                      AND NOT EXISTS (
                        SELECT 1 FROM instance_outbox delivery
                        WHERE delivery.outbox_id = important_todos.delivery_outbox_id
                          AND delivery.status IN ('PENDING', 'SENDING')
                      )""",
                    (self.now, profile_id, instance_id, asset_id),
                ).rowcount
            )
            files += int(
                conn.execute(
                    """UPDATE file_assets SET delivery_status = 'NOT_SELECTED',
                        last_error = 'delivery_interrupted_before_platform_call',
                        updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND asset_id = ?
                      AND delivery_status IN ('SELECTED', 'OUTBOX_PENDING')
                      AND NOT EXISTS (
                        SELECT 1 FROM important_todos todo
                        JOIN instance_outbox delivery
                          ON delivery.outbox_id = todo.delivery_outbox_id
                        WHERE todo.profile_id = file_assets.profile_id
                          AND todo.instance_id = file_assets.instance_id
                          AND todo.file_asset_id = file_assets.asset_id
                          AND delivery.status IN ('PENDING', 'SENDING')
                      )""",
                    (self.now, profile_id, instance_id, asset_id),
                ).rowcount
            )
        return files, todos

    def _settle_messages(self, conn: sqlite3.Connection, plan: _InterruptedForegroundPlan) -> int:
        changed = 0
        for row in plan.messages:
            message_id = int(row["message_id"])
            if message_id in plan.prepared_message_ids:
                metadata = _load(row["metadata_json"]) or {}
                assert isinstance(metadata, dict)
                metadata["send_error"] = "delivery_interrupted_before_platform_call"
                cursor = conn.execute(
                    """UPDATE instance_messages SET delivery_status = 'FAILED',
                        metadata_json = ?
                    WHERE message_id = ? AND direction = 'OUTBOUND'
                      AND delivery_status = 'PENDING'""",
                    (_dump(metadata), message_id),
                )
            else:
                cursor = conn.execute(
                    """UPDATE instance_messages
                    SET delivery_status = 'UNKNOWN_AFTER_CRASH'
                    WHERE message_id = ? AND direction = 'OUTBOUND'
                      AND delivery_status = 'PENDING'""",
                    (message_id,),
                )
            changed += int(cursor.rowcount)
        return changed


class MediaLifecycleCommands(MediaReleasePlanningCommands):
    async def finalize_media_delivery(
        self,
        profile_id: str,
        instance_id: str,
        asset_id: str,
        status: str,
        *,
        error: str | None = None,
    ) -> MediaAsset:
        """Finalize one selected image and its retention clock.

        An adapter acknowledgement has no end-to-end receipt, but it is enough
        to keep the image until its visible ledger message is summarized.
        Definite failure or crash uncertainty restores a 24-hour diagnostic TTL.
        """

        normalized = str(status or "").strip().upper()
        accepted = is_dialogue_continuity_visible("OUTBOUND", normalized)
        now_dt = _now()
        now = _dt(now_dt)

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            if accepted:
                cursor = conn.execute(
                    """UPDATE media_assets SET delivery_status = ?,
                    expires_at = NULL, last_error = ?, updated_at = ?
                    WHERE asset_id = ? AND profile_id = ? AND instance_id = ?""",
                    (normalized, error, now, asset_id, profile_id, instance_id),
                )
            else:
                cursor = conn.execute(
                    """UPDATE media_assets SET delivery_status = ?,
                    expires_at = COALESCE(expires_at, ?), last_error = ?,
                    updated_at = ? WHERE asset_id = ? AND profile_id = ?
                    AND instance_id = ?""",
                    (
                        normalized or "FAILED",
                        _dt(now_dt + timedelta(hours=24)),
                        error,
                        now,
                        asset_id,
                        profile_id,
                        instance_id,
                    ),
                )
            if cursor.rowcount != 1:
                raise KeyError((profile_id, instance_id, asset_id))
            row = conn.execute(
                "SELECT * FROM media_assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            assert row is not None
            return row

        asset = self._media_asset(await self.uow.run(operation))
        await self.db.publish_backup_after_commit()
        return asset

    async def recover_interrupted_media_delivery(
        self,
        *,
        stale_before: datetime | None = None,
    ) -> dict[str, int]:
        """Close foreground/outbox states that must never be blindly replayed."""

        current = _now()
        result = await self.uow.run(
            _RecoverInterruptedMediaDelivery(
                stale_before=stale_before,
                now=_dt(current),
                expires=_dt(current + timedelta(hours=24)),
            )
        )
        if any(result.values()):
            await self.db.publish_backup_after_commit()
        return result

    async def list_pending_media_releases(self, *, limit: int = 100) -> list[MediaAsset]:
        rows = await self.db.fetch_all(
            """SELECT * FROM media_assets WHERE file_status = 'RELEASE_PENDING'
            ORDER BY updated_at, asset_id LIMIT ?""",
            (max(1, min(int(limit), 1000)),),
        )
        return [self._media_asset(row) for row in rows]

    async def finalize_media_release(
        self, asset_id: str, *, success: bool, error: str | None = None
    ) -> MediaAsset:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            row = conn.execute(
                "SELECT * FROM media_assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            if row is None:
                raise KeyError(asset_id)
            if success:
                conn.execute(
                    """UPDATE media_assets SET file_status = 'RELEASED',
                    storage_relpath = NULL, released_at = ?, last_error = NULL,
                    updated_at = ? WHERE asset_id = ?""",
                    (now, now, asset_id),
                )
            else:
                conn.execute(
                    """UPDATE media_assets SET file_status = 'RELEASE_PENDING',
                    last_error = ?, updated_at = ? WHERE asset_id = ?""",
                    (str(error or "media_release_failed"), now, asset_id),
                )
            self._media_cleanup_event_sql(
                conn,
                asset_id,
                row["profile_id"],
                row["instance_id"],
                "FILE_RELEASE",
                "SUCCEEDED" if success else "FAILED",
                "",
                {"error": error} if error else {},
                now,
            )
            updated = conn.execute(
                "SELECT * FROM media_assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            assert updated is not None
            return updated

        asset = self._media_asset(await self.uow.run(operation))
        await self.db.publish_backup_after_commit()
        return asset

    async def mark_media_missing(self, asset_id: str, *, reason: str) -> MediaAsset:
        row = await self.get_media_asset(asset_id)
        if row is None:
            raise KeyError(asset_id)
        return await self._update_media_asset_fields(
            row.profile_id,
            row.instance_id,
            asset_id,
            {"file_status": MediaFileStatus.MISSING.value, "last_error": str(reason)},
        )

    async def list_media_cleanup_events(
        self, profile_id: str, instance_id: str, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT * FROM media_cleanup_events WHERE profile_id = ?
            AND instance_id = ? ORDER BY cleanup_id DESC LIMIT ?""",
            (profile_id, instance_id, max(1, min(int(limit), 1000))),
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["details"] = _load(item.pop("details_json")) or {}
            result.append(item)
        return result

    async def _update_media_asset_fields(
        self,
        profile_id: str,
        instance_id: str,
        asset_id: str,
        fields: dict[str, Any],
    ) -> MediaAsset:
        allowed = {"file_status", "delivery_status", "inspection_status", "last_error"}
        if not fields or set(fields) - allowed:
            raise ValueError("unsupported media asset update")
        assignments = [f"{key} = ?" for key in fields]
        values = list(fields.values())
        assignments.append("updated_at = ?")
        values.extend((_dt(_now()), asset_id, profile_id, instance_id))

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            cursor = conn.execute(
                f"""UPDATE media_assets SET {", ".join(assignments)}
                WHERE asset_id = ? AND profile_id = ? AND instance_id = ?""",
                values,
            )
            if cursor.rowcount != 1:
                raise KeyError((profile_id, instance_id, asset_id))
            row = conn.execute(
                "SELECT * FROM media_assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            assert row is not None
            return row

        asset = self._media_asset(await self.uow.run(operation))
        await self.db.publish_backup_after_commit()
        return asset

    @staticmethod
    def _media_cleanup_event_sql(
        conn: sqlite3.Connection,
        asset_id: str,
        profile_id: str,
        instance_id: str,
        action: str,
        status: str,
        reason: str,
        details: dict[str, Any],
        created_at: str,
    ) -> None:
        conn.execute(
            """INSERT INTO media_cleanup_events(
                asset_id, profile_id, instance_id, action, status,
                reason, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                asset_id,
                profile_id,
                instance_id,
                action,
                status,
                reason,
                _dump(details),
                created_at,
            ),
        )

    @staticmethod
    def _media_asset(row: sqlite3.Row) -> MediaAsset:
        return MediaAsset(
            asset_id=row["asset_id"],
            profile_id=row["profile_id"],
            instance_id=row["instance_id"],
            origin=MediaOrigin(row["origin"]),
            purpose=MediaPurpose(row["purpose"]),
            mime_type=row["mime_type"],
            file_extension=row["file_extension"],
            sha256=row["sha256"],
            byte_size=int(row["byte_size"]),
            width=row["width"],
            height=row["height"],
            frame_count=row["frame_count"],
            storage_relpath=row["storage_relpath"],
            file_status=MediaFileStatus(row["file_status"]),
            delivery_status=row["delivery_status"],
            inspection_status=MediaInspectionStatus(row["inspection_status"]),
            current_projection_version=int(row["current_projection_version"]),
            core_run_id=row["core_run_id"],
            ai_task_id=row["ai_task_id"],
            summary_covered_by=row["summary_covered_by"],
            expires_at=_parse(row["expires_at"]),
            released_at=_parse(row["released_at"]),
            last_error=row["last_error"],
            metadata=_load(row["metadata_json"]) or {},
            created_at=_parse(row["created_at"]),
            updated_at=_parse(row["updated_at"]),
        )

    @staticmethod
    def _media_projection(row: sqlite3.Row) -> MediaProjection:
        return MediaProjection(
            projection_id=int(row["projection_id"]),
            asset_id=row["asset_id"],
            version=int(row["version"]),
            status=MediaProjectionStatus(row["status"]),
            visible_facts=row["visible_facts"],
            history_projection=row["history_projection"],
            ocr_text=row["ocr_text"],
            backend_id=row["backend_id"],
            model_id=row["model_id"],
            ai_task_id=row["ai_task_id"],
            error=row["error"],
            created_at=_parse(row["created_at"]),
        )
