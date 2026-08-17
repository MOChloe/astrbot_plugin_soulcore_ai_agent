"""Atomic Outbox and foreground delivery settlement commands."""

from __future__ import annotations

from ....contracts.delivery_visibility import (
    FOREGROUND_DELIVERY_BOUNDARY_ENTERED,
    FOREGROUND_DELIVERY_BOUNDARY_PREPARING,
    foreground_delivery_boundary,
    foreground_delivery_todo_ids,
)
from ....storage.sqlite.background_projection import project_foreground_message_continuity_sql
from ....storage.sqlite.outbox_settlement_dependencies import (
    record_sticker_usage_in_transaction,
)
from .platform_fragments import (
    _foreground_fragment_content,
    _insert_platform_fragment,
)
from .support import (
    Any,
    OutboxStatus,
    _dt,
    _dump,
    _load,
    datetime,
    sqlite3,
    timedelta,
)


class _FinalizeForegroundDelivery:
    """Settle one attempted or definitively unattempted foreground send atomically."""

    _ATTEMPTED_MEDIA_ACCEPTABLE = frozenset(
        {
            "SELECTED",
            OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED.value,
            OutboxStatus.UNKNOWN_AFTER_CRASH.value,
        }
    )
    _ATTEMPTED_TODO_ACCEPTABLE = frozenset(
        {
            "SELECTED",
            "DELIVERY_PENDING",
            "DELIVERY_UNKNOWN",
        }
    )
    _ATTEMPTED_FILE_ACCEPTABLE = frozenset(
        {
            "SELECTED",
            "OUTBOX_PENDING",
            OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED.value,
            OutboxStatus.UNKNOWN_AFTER_CRASH.value,
        }
    )
    _ATTEMPTED_MESSAGE_ACCEPTABLE = frozenset(
        {
            OutboxStatus.PENDING.value,
            OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED.value,
            OutboxStatus.PARTIALLY_ATTEMPTED.value,
            OutboxStatus.UNKNOWN_AFTER_CRASH.value,
        }
    )

    def __init__(
        self,
        owner: Any,
        *,
        profile_id: str,
        instance_id: str,
        message_id: int,
        media_asset_ids: tuple[str, ...],
        todo_ids: tuple[str, ...],
        status: OutboxStatus,
        error: str,
        receipts: tuple[Any, ...],
        sticker_deliveries: tuple[dict[str, Any], ...],
        route_umo: str,
        now: str,
        now_dt: datetime,
    ) -> None:
        self.owner = owner
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.message_id = int(message_id)
        self.media_asset_ids = media_asset_ids
        self.todo_ids = todo_ids
        self.status = status
        self.error = error
        self.receipts = receipts
        self.sticker_deliveries = sticker_deliveries
        self.route_umo = route_umo
        self.now = now
        self.now_dt = now_dt

    def __call__(self, conn: sqlite3.Connection) -> None:
        message = self._validate_message(conn)
        if str(message["delivery_status"]).upper() == self.status.value:
            self._record_sticker_usages(conn, message)
            self._write_platform_fragments(conn, message)
            return
        self._validate_media(conn, message)
        file_asset_ids = self._validate_todos(conn, message)
        self._settle_media(conn)
        self._settle_todos(conn, file_asset_ids)
        metadata = _load(message["metadata_json"]) or {}
        if not isinstance(metadata, dict):
            raise ValueError("foreground message metadata must be an object")
        if self.status is OutboxStatus.FAILED:
            metadata["send_error"] = self.error
        conn.execute(
            """UPDATE instance_messages SET delivery_status = ?, metadata_json = ?
            WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
            (
                self.status.value,
                _dump(metadata),
                self.profile_id,
                self.instance_id,
                self.message_id,
            ),
        )
        settled = conn.execute(
            """SELECT * FROM instance_messages
            WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
            (self.profile_id, self.instance_id, self.message_id),
        ).fetchone()
        assert settled is not None
        self._record_sticker_usages(conn, settled)
        self._write_platform_fragments(conn, settled)
        if self.status is OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED:
            project_foreground_message_continuity_sql(conn, settled, settled_at=self.now)
            self.owner._refresh_knowledge_task_sql(
                conn,
                self.profile_id,
                self.instance_id,
                now_dt=self.now_dt,
            )

    def _record_sticker_usages(
        self,
        conn: sqlite3.Connection,
        message: sqlite3.Row,
    ) -> None:
        if self.status is not OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED:
            return
        for ordinal, delivery in enumerate(self.sticker_deliveries, start=1):
            record_sticker_usage_in_transaction(
                conn,
                self.profile_id,
                self.instance_id,
                item_id=str(delivery["item_id"]),
                run_id=delivery["run_id"],
                sticker_ref=str(delivery["sticker_ref"]),
                compact_projection=str(delivery["projection"]),
                delivery_status=self.status.value,
                outbox_id=-self.message_id,
                expression_ordinal=ordinal,
                message_id=int(message["message_id"]),
                now=self.now,
            )

    def _write_platform_fragments(
        self,
        conn: sqlite3.Connection,
        message: sqlite3.Row,
    ) -> None:
        if (
            self.status
            not in {
                OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED,
                OutboxStatus.PARTIALLY_ATTEMPTED,
            }
            or not self.receipts
        ):
            return
        kind, projection = _foreground_fragment_content(message)
        for receipt in self.receipts:
            platform_message_id = str(getattr(receipt, "platform_message_id", "") or "").strip()
            if not platform_message_id:
                continue
            _insert_platform_fragment(
                conn,
                profile_id=self.profile_id,
                instance_id=self.instance_id,
                message=message,
                route_umo=self.route_umo,
                kind=kind,
                projection=projection,
                accepted_at=self.now_dt,
                platform_message_id=platform_message_id,
                fragment_ordinal=max(0, int(getattr(receipt, "fragment_ordinal", 0) or 0)),
                platform_id=str(getattr(receipt, "platform_id", "") or "").strip(),
                platform_reference_id=str(
                    getattr(receipt, "platform_reference_id", "") or ""
                ).strip(),
                native_reply_supported=bool(getattr(receipt, "native_reply_supported", False)),
                member_mention_supported=bool(getattr(receipt, "member_mention_supported", False)),
                self_retraction_supported=bool(
                    getattr(receipt, "self_retraction_supported", False)
                ),
                returns_platform_message_id=bool(
                    getattr(receipt, "returns_platform_message_id", False)
                ),
                retractable_for_seconds=getattr(receipt, "retractable_for_seconds", None),
                now=self.now,
            )

    def _validate_message(self, conn: sqlite3.Connection) -> sqlite3.Row:
        message = conn.execute(
            """SELECT * FROM instance_messages
            WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
            (self.profile_id, self.instance_id, self.message_id),
        ).fetchone()
        if message is None:
            raise KeyError((self.profile_id, self.instance_id, self.message_id))
        if str(message["direction"]).upper() != "OUTBOUND":
            raise ValueError("foreground settlement requires an outbound message")
        current_status = str(message["delivery_status"]).upper()
        acceptable = (
            frozenset({OutboxStatus.PENDING.value, OutboxStatus.FAILED.value})
            if self.status is OutboxStatus.FAILED
            else self._ATTEMPTED_MESSAGE_ACCEPTABLE
        )
        if current_status not in acceptable:
            raise RuntimeError("foreground message is no longer settlement-eligible")
        metadata = _load(message["metadata_json"]) or {}
        boundary = foreground_delivery_boundary(metadata) if isinstance(metadata, dict) else None
        if (
            self.status is OutboxStatus.FAILED
            and boundary != FOREGROUND_DELIVERY_BOUNDARY_PREPARING
        ):
            raise RuntimeError(
                "foreground preparation is not owned or platform boundary already entered"
            )
        if (
            current_status == OutboxStatus.PENDING.value
            and self.status is not OutboxStatus.FAILED
            and boundary != FOREGROUND_DELIVERY_BOUNDARY_ENTERED
        ):
            raise RuntimeError("foreground platform boundary was not entered")
        if (
            current_status == OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED.value
            and self.status is OutboxStatus.UNKNOWN_AFTER_CRASH
        ):
            raise RuntimeError(
                "confirmed foreground platform acceptance cannot be downgraded to unknown"
            )
        return message

    def _validate_media(self, conn: sqlite3.Connection, message: sqlite3.Row) -> None:
        if not self.media_asset_ids:
            return
        placeholders = ",".join("?" for _ in self.media_asset_ids)
        rows = list(
            conn.execute(
                f"""SELECT asset.*,
                    EXISTS (
                        SELECT 1 FROM media_asset_message_links link
                        WHERE link.asset_id = asset.asset_id
                          AND link.profile_id = asset.profile_id
                          AND link.instance_id = asset.instance_id
                          AND link.message_id = ?
                    ) AS linked_to_message
                FROM media_assets asset
                WHERE asset.profile_id = ? AND asset.instance_id = ?
                  AND asset.asset_id IN ({placeholders})""",
                (
                    self.message_id,
                    self.profile_id,
                    self.instance_id,
                    *self.media_asset_ids,
                ),
            )
        )
        if {str(row["asset_id"]) for row in rows} != set(self.media_asset_ids):
            raise KeyError("foreground media ownership changed before settlement")
        if self.status is OutboxStatus.FAILED:
            component_ids = self._component_asset_ids(message, "image_asset")
            if not set(self.media_asset_ids).issubset(component_ids):
                raise RuntimeError("foreground media is not owned by the failed message")
        elif any(not bool(row["linked_to_message"]) for row in rows):
            raise RuntimeError("foreground media is not linked to the settled message")
        acceptable = (
            frozenset({"SELECTED", OutboxStatus.FAILED.value})
            if self.status is OutboxStatus.FAILED
            else self._ATTEMPTED_MEDIA_ACCEPTABLE
        )
        if any(str(row["delivery_status"]).upper() not in acceptable for row in rows):
            raise RuntimeError("foreground media is no longer settlement-eligible")

    def _validate_todos(self, conn: sqlite3.Connection, message: sqlite3.Row) -> tuple[str, ...]:
        if not self.todo_ids:
            return ()
        rows = self._todo_rows(conn)
        self._validate_todo_message_ownership(rows, message)
        self._validate_todo_states_and_assets(rows)
        file_asset_ids = self._todo_file_asset_ids(rows)
        self._validate_failed_file_assets(message, file_asset_ids)
        return file_asset_ids

    def _todo_rows(self, conn: sqlite3.Connection) -> list[sqlite3.Row]:
        placeholders = ",".join("?" for _ in self.todo_ids)
        return list(
            conn.execute(
                f"""SELECT todo.*, asset.profile_id AS asset_profile_id,
                    asset.instance_id AS asset_instance_id,
                    asset.delivery_status AS asset_delivery_status
                FROM important_todos todo
                LEFT JOIN file_assets asset ON asset.asset_id = todo.file_asset_id
                WHERE todo.profile_id = ? AND todo.instance_id = ?
                  AND todo.todo_id IN ({placeholders})""",
                (self.profile_id, self.instance_id, *self.todo_ids),
            )
        )

    def _validate_todo_message_ownership(
        self,
        rows: list[sqlite3.Row],
        message: sqlite3.Row,
    ) -> None:
        if {str(row["todo_id"]) for row in rows} != set(self.todo_ids):
            raise KeyError("foreground file todo ownership changed before settlement")
        metadata = _load(message["metadata_json"]) or {}
        owned_todo_ids = (
            set(foreground_delivery_todo_ids(metadata)) if isinstance(metadata, dict) else set()
        )
        if not set(self.todo_ids).issubset(owned_todo_ids):
            raise RuntimeError("foreground todo is not owned by the settled message")

    def _validate_todo_states_and_assets(self, rows: list[sqlite3.Row]) -> None:
        acceptable_todos = (
            frozenset({"PENDING", "SELECTED", "DELIVERY_PENDING"})
            if self.status is OutboxStatus.FAILED
            else self._ATTEMPTED_TODO_ACCEPTABLE
        )
        if any(str(row["status"]).upper() not in acceptable_todos for row in rows):
            raise RuntimeError("foreground file todo is no longer settlement-eligible")
        if any(not self._todo_asset_is_owned(row) for row in rows):
            raise RuntimeError("foreground file asset ownership changed before settlement")
        acceptable_files = (
            frozenset({"NOT_SELECTED", "SELECTED", "OUTBOX_PENDING"})
            if self.status is OutboxStatus.FAILED
            else self._ATTEMPTED_FILE_ACCEPTABLE
        )
        if any(
            row["file_asset_id"] is not None
            and str(row["asset_delivery_status"]).upper() not in acceptable_files
            for row in rows
        ):
            raise RuntimeError("foreground file asset is no longer settlement-eligible")

    def _todo_asset_is_owned(self, row: sqlite3.Row) -> bool:
        if row["file_asset_id"] is None:
            return str(row["kind"] or "").upper() == "FILE_FAILED"
        return (
            str(row["asset_profile_id"] or "") == self.profile_id
            and str(row["asset_instance_id"] or "") == self.instance_id
        )

    @staticmethod
    def _todo_file_asset_ids(rows: list[sqlite3.Row]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                str(row["file_asset_id"]) for row in rows if row["file_asset_id"] is not None
            )
        )

    def _validate_failed_file_assets(
        self,
        message: sqlite3.Row,
        file_asset_ids: tuple[str, ...],
    ) -> None:
        if self.status is OutboxStatus.FAILED:
            component_ids = self._component_asset_ids(message, "file_artifact")
            if not set(file_asset_ids).issubset(component_ids):
                raise RuntimeError("foreground file asset is not owned by the failed message")

    @staticmethod
    def _component_asset_ids(message: sqlite3.Row, component_type: str) -> set[str]:
        components = _load(message["components_json"]) or []
        if not isinstance(components, list):
            raise ValueError("foreground message components must be a list")
        return {
            str(component.get("asset_id") or "")
            for component in components
            if isinstance(component, dict)
            and str(component.get("type") or "") == component_type
            and str(component.get("asset_id") or "")
        }

    def _settle_media(self, conn: sqlite3.Connection) -> None:
        if not self.media_asset_ids:
            return
        placeholders = ",".join("?" for _ in self.media_asset_ids)
        accepted = self.status is OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED
        stored_status = (
            OutboxStatus.UNKNOWN_AFTER_CRASH
            if self.status is OutboxStatus.PARTIALLY_ATTEMPTED
            else self.status
        )
        if accepted:
            conn.execute(
                f"""UPDATE media_assets SET delivery_status = ?,
                    expires_at = NULL, last_error = NULL, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?
                  AND asset_id IN ({placeholders})""",
                (
                    stored_status.value,
                    self.now,
                    self.profile_id,
                    self.instance_id,
                    *self.media_asset_ids,
                ),
            )
        else:
            conn.execute(
                f"""UPDATE media_assets SET delivery_status = ?,
                    expires_at = COALESCE(expires_at, ?), last_error = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?
                  AND asset_id IN ({placeholders})""",
                (
                    stored_status.value,
                    _dt(self.now_dt + timedelta(hours=24)),
                    self.error,
                    self.now,
                    self.profile_id,
                    self.instance_id,
                    *self.media_asset_ids,
                ),
            )

    def _settle_todos(
        self,
        conn: sqlite3.Connection,
        file_asset_ids: tuple[str, ...],
    ) -> None:
        if not self.todo_ids:
            return
        todo_placeholders = ",".join("?" for _ in self.todo_ids)
        if self.status is OutboxStatus.FAILED:
            conn.execute(
                f"""UPDATE important_todos SET status = 'PENDING',
                    selected_run_id = NULL, selected_activity_epoch = NULL,
                    delivery_outbox_id = NULL, resolved_at = NULL,
                    version = CASE WHEN status = 'PENDING' THEN version ELSE version + 1 END,
                    updated_at = ?
                WHERE profile_id = ? AND instance_id = ?
                  AND todo_id IN ({todo_placeholders})""",
                (
                    self.now,
                    self.profile_id,
                    self.instance_id,
                    *self.todo_ids,
                ),
            )
            if not file_asset_ids:
                return
            asset_placeholders = ",".join("?" for _ in file_asset_ids)
            conn.execute(
                f"""UPDATE file_assets SET delivery_status = 'NOT_SELECTED',
                    last_error = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?
                  AND asset_id IN ({asset_placeholders})""",
                (
                    self.error,
                    self.now,
                    self.profile_id,
                    self.instance_id,
                    *file_asset_ids,
                ),
            )
            return
        conn.execute(
            f"""UPDATE important_todos SET status = 'DELIVERY_UNKNOWN',
                resolved_at = NULL,
                version = CASE WHEN status = 'DELIVERY_UNKNOWN' THEN version ELSE version + 1 END,
                updated_at = ?
            WHERE profile_id = ? AND instance_id = ?
              AND todo_id IN ({todo_placeholders})""",
            (
                self.now,
                self.profile_id,
                self.instance_id,
                *self.todo_ids,
            ),
        )
        if not file_asset_ids:
            return
        asset_placeholders = ",".join("?" for _ in file_asset_ids)
        conn.execute(
            f"""UPDATE file_assets SET delivery_status = ?,
                last_error = ?, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?
              AND asset_id IN ({asset_placeholders})""",
            (
                (
                    OutboxStatus.UNKNOWN_AFTER_CRASH.value
                    if self.status is OutboxStatus.PARTIALLY_ATTEMPTED
                    else self.status.value
                ),
                ("" if self.status is OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED else self.error),
                self.now,
                self.profile_id,
                self.instance_id,
                *file_asset_ids,
            ),
        )


__all__ = ["_FinalizeForegroundDelivery"]
