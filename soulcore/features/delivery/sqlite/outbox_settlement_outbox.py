"""Atomic Outbox and foreground delivery settlement commands."""

from __future__ import annotations

from ....contracts.delivery_visibility import (
    DIALOGUE_CONTINUITY_OUTBOUND_STATUSES,
)
from ....storage.sqlite.background_projection import project_foreground_message_continuity_sql
from ....storage.sqlite.contact_evidence_settlement import ContactEvidenceSettlement
from ....storage.sqlite.expression_batch_lifecycle import (
    sync_expression_batch_status,
)
from ....storage.sqlite.outbox_settlement_dependencies import (
    _link_media_to_message_sql,
    finalize_contact_attempt_sql,
    record_sticker_usage_in_transaction,
)
from .expression_outbox import defer_following_expression_step
from .outbox_settlement_shared import (
    _cancel_terminal_expression_suffix,
    _resolve_terminal_group_window,
)
from .platform_fragments import (
    _fragment_content,
    _insert_platform_fragment,
)
from .support import (
    Any,
    OutboxStatus,
    _dt,
    _dump,
    _load,
    _now,
    _parse,
    datetime,
    sqlite3,
    timedelta,
)


class _FinalizeOutboxDelivery:
    def __init__(
        self,
        *,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        status: OutboxStatus,
        error_code: str,
        error: str | None,
        diagnostic_code: str,
        context_message: dict[str, Any] | None,
        receipts: tuple[Any, ...],
        sticker_deliveries: tuple[dict[str, Any], ...],
        now: str,
        now_dt: datetime,
    ) -> None:
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.outbox_id = int(outbox_id)
        self.status = status
        self.error_code = str(error_code or "")
        self.error = error
        self.diagnostic_code = str(diagnostic_code or "")
        self.context_message = context_message
        self.receipts = receipts
        self.sticker_deliveries = sticker_deliveries
        self.now = now
        self.now_dt = now_dt

    def __call__(
        self,
        conn: sqlite3.Connection,
    ) -> tuple[bool, sqlite3.Row | None]:
        current = self._load_current(conn)
        updated = self._transition(conn, str(current["status"]))
        if not updated:
            return False, None
        message = self._write_context_ledger(conn, current)
        if message is not None and self.status is OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED:
            project_foreground_message_continuity_sql(conn, message, settled_at=self.now)
        self._record_sticker_usages(conn, current, message)
        self._write_platform_fragments(conn, current, message)
        self._settle_delivery_dependents(conn, current, message)
        if self.status is not OutboxStatus.PENDING:
            defer_following_expression_step(
                conn,
                current["expression_batch_id"],
                current["expression_step_ordinal"]
                if current["expression_step_ordinal"] is not None
                else current["expression_ordinal"],
                self.now,
            )
        _cancel_terminal_expression_suffix(
            conn,
            current,
            self.status,
            self.now,
        )
        sync_expression_batch_status(conn, current["expression_batch_id"], self.now)
        _resolve_terminal_group_window(conn, current, self.status, self.now)
        return True, message

    @staticmethod
    def _payload(outbox: sqlite3.Row) -> dict[str, Any]:
        payload = _load(outbox["payload_json"]) or {}
        if not isinstance(payload, dict):
            raise ValueError("outbox payload must be an object")
        return payload

    def _settle_delivery_dependents(
        self,
        conn: sqlite3.Connection,
        outbox: sqlite3.Row,
        message: sqlite3.Row | None,
    ) -> None:
        if self.status is OutboxStatus.PENDING:
            return
        payload = self._payload(outbox)
        self._settle_image_assets(conn, payload, message)
        self._settle_file_todos(conn, payload)
        self._settle_contact(conn, payload)

    @staticmethod
    def _image_components(payload: dict[str, Any]) -> tuple[tuple[str, int], ...]:
        components = payload.get("components") or []
        if not isinstance(components, list):
            raise ValueError("outbox components must be a list")
        result: list[tuple[str, int]] = []
        seen: set[str] = set()
        for ordinal, component in enumerate(components):
            if not isinstance(component, dict):
                raise ValueError("outbox component must be an object")
            if str(component.get("type") or "") != "image_asset":
                continue
            asset_id = str(component.get("asset_id") or "").strip()
            if not asset_id:
                raise ValueError("image_asset component requires asset_id")
            if asset_id not in seen:
                result.append((asset_id, ordinal))
                seen.add(asset_id)
        return tuple(result)

    def _settle_image_assets(
        self,
        conn: sqlite3.Connection,
        payload: dict[str, Any],
        message: sqlite3.Row | None,
    ) -> None:
        components = self._image_components(payload)
        if not components:
            return
        asset_ids = tuple(asset_id for asset_id, _ in components)
        placeholders = ",".join("?" for _ in asset_ids)
        rows = list(
            conn.execute(
                f"""SELECT asset_id, delivery_status FROM media_assets
                WHERE profile_id = ? AND instance_id = ?
                  AND asset_id IN ({placeholders})""",
                (self.profile_id, self.instance_id, *asset_ids),
            )
        )
        self._validate_image_asset_rows(rows, asset_ids)
        target, error, expires_at = self._image_settlement_state()
        self._link_image_assets(conn, components, message)
        self._update_image_assets(
            conn,
            asset_ids,
            placeholders,
            target=target,
            error=error,
            expires_at=expires_at,
        )

    @staticmethod
    def _validate_image_asset_rows(rows: list[sqlite3.Row], asset_ids: tuple[str, ...]) -> None:
        if {str(row["asset_id"]) for row in rows} != set(asset_ids):
            raise KeyError("outbox image ownership changed before settlement")

    def _image_settlement_state(self) -> tuple[str, str, str | None]:
        if self.status is OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED:
            target = OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED.value
            error = ""
            expires_at = None
        elif self.status in {
            OutboxStatus.PARTIALLY_ATTEMPTED,
            OutboxStatus.UNKNOWN_AFTER_CRASH,
        }:
            target = OutboxStatus.UNKNOWN_AFTER_CRASH.value
            error = self.error or self.diagnostic_code or "delivery_result_unknown"
            expires_at = _dt(self.now_dt + timedelta(hours=24))
        else:
            target = OutboxStatus.FAILED.value
            error = self.error or self.error_code or self.status.value.lower()
            expires_at = _dt(self.now_dt + timedelta(hours=24))
        return target, str(error), expires_at

    def _link_image_assets(
        self,
        conn: sqlite3.Connection,
        components: tuple[tuple[str, int], ...],
        message: sqlite3.Row | None,
    ) -> None:
        if message is None:
            return
        for asset_id, ordinal in components:
            _link_media_to_message_sql(
                conn,
                profile_id=self.profile_id,
                instance_id=self.instance_id,
                asset_id=asset_id,
                message_id=int(message["message_id"]),
                relation="GENERATED_OUTPUT",
                ordinal=ordinal,
                now=self.now,
            )

    def _update_image_assets(
        self,
        conn: sqlite3.Connection,
        asset_ids: tuple[str, ...],
        placeholders: str,
        *,
        target: str,
        error: str,
        expires_at: str | None,
    ) -> None:
        eligible = conn.execute(
            f"""SELECT COUNT(*) AS total FROM media_assets
            WHERE profile_id = ? AND instance_id = ? AND asset_id IN ({placeholders})
              AND delivery_status IN ('SELECTED', ?)""",
            (self.profile_id, self.instance_id, *asset_ids, target),
        ).fetchone()
        if eligible is None or int(eligible["total"]) != len(asset_ids):
            raise RuntimeError("outbox image is no longer settlement-eligible")
        if expires_at is None:
            conn.execute(
                f"""UPDATE media_assets SET delivery_status = ?, expires_at = NULL,
                    last_error = NULL, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?
                  AND asset_id IN ({placeholders})""",
                (target, self.now, self.profile_id, self.instance_id, *asset_ids),
            )
        else:
            conn.execute(
                f"""UPDATE media_assets SET delivery_status = ?,
                    expires_at = COALESCE(expires_at, ?), last_error = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?
                  AND asset_id IN ({placeholders})""",
                (
                    target,
                    expires_at,
                    str(error)[:600],
                    self.now,
                    self.profile_id,
                    self.instance_id,
                    *asset_ids,
                ),
            )

    @staticmethod
    def _todo_ids(payload: dict[str, Any]) -> tuple[str, ...]:
        values = payload.get("important_todo_ids") or []
        if not isinstance(values, list):
            raise ValueError("important_todo_ids must be a list")
        return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))

    def _dependent_todo_ids(
        self,
        conn: sqlite3.Connection,
        payload: dict[str, Any],
    ) -> tuple[str, ...]:
        todo_ids = list(self._todo_ids(payload))
        if self.status not in {OutboxStatus.FAILED, OutboxStatus.CANCELLED}:
            return tuple(todo_ids)
        followup_key = str(payload.get("file_followup_idempotency_key") or "").strip()
        if not followup_key:
            return tuple(todo_ids)
        followup = conn.execute(
            """SELECT payload_json FROM instance_outbox
            WHERE profile_id = ? AND instance_id = ? AND idempotency_key = ?""",
            (self.profile_id, self.instance_id, followup_key),
        ).fetchone()
        if followup is not None:
            followup_payload = _load(followup["payload_json"]) or {}
            if not isinstance(followup_payload, dict):
                raise ValueError("file follow-up outbox payload must be an object")
            todo_ids.extend(self._todo_ids(followup_payload))
        return tuple(dict.fromkeys(todo_ids))

    def _settle_file_todos(self, conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
        todo_ids = self._dependent_todo_ids(conn, payload)
        if not todo_ids:
            return
        placeholders = ",".join("?" for _ in todo_ids)
        rows = self._load_file_todo_rows(conn, todo_ids, placeholders)
        self._validate_file_todo_rows(rows, todo_ids)
        file_asset_ids = self._file_asset_ids(rows)
        file_target = self._settle_file_todo_rows(conn, todo_ids, placeholders)
        self._settle_file_asset_rows(conn, file_asset_ids, file_target)

    def _load_file_todo_rows(
        self, conn: sqlite3.Connection, todo_ids: tuple[str, ...], placeholders: str
    ) -> list[sqlite3.Row]:
        return list(
            conn.execute(
                f"""SELECT todo.*, asset.profile_id AS asset_profile_id,
                    asset.instance_id AS asset_instance_id,
                    asset.delivery_status AS asset_delivery_status
                FROM important_todos todo
                LEFT JOIN file_assets asset ON asset.asset_id = todo.file_asset_id
                WHERE todo.profile_id = ? AND todo.instance_id = ?
                  AND todo.todo_id IN ({placeholders})""",
                (self.profile_id, self.instance_id, *todo_ids),
            )
        )

    def _validate_file_todo_rows(self, rows: list[sqlite3.Row], todo_ids: tuple[str, ...]) -> None:
        if {str(row["todo_id"]) for row in rows} != set(todo_ids):
            raise KeyError("outbox file todo ownership changed before settlement")
        for row in rows:
            missing_failed_asset = (
                row["file_asset_id"] is None and str(row["kind"] or "").upper() != "FILE_FAILED"
            )
            foreign_asset = row["file_asset_id"] is not None and (
                str(row["asset_profile_id"] or "") != self.profile_id
                or str(row["asset_instance_id"] or "") != self.instance_id
            )
            if missing_failed_asset:
                raise RuntimeError("outbox todo lost its file asset")
            if foreign_asset:
                raise RuntimeError("outbox file asset ownership changed before settlement")

    @staticmethod
    def _file_asset_ids(rows: list[sqlite3.Row]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                str(row["file_asset_id"]) for row in rows if row["file_asset_id"] is not None
            )
        )

    def _settle_file_todo_rows(
        self, conn: sqlite3.Connection, todo_ids: tuple[str, ...], placeholders: str
    ) -> str:
        if self.status in {OutboxStatus.FAILED, OutboxStatus.CANCELLED}:
            conn.execute(
                f"""UPDATE important_todos SET status = 'PENDING',
                    selected_run_id = NULL, selected_activity_epoch = NULL,
                    delivery_outbox_id = NULL, resolved_at = NULL,
                    version = CASE WHEN status = 'PENDING' THEN version ELSE version + 1 END,
                    updated_at = ?
                WHERE profile_id = ? AND instance_id = ?
                  AND todo_id IN ({placeholders})""",
                (self.now, self.profile_id, self.instance_id, *todo_ids),
            )
            return "NOT_SELECTED"
        conn.execute(
            f"""UPDATE important_todos SET status = 'DELIVERY_UNKNOWN',
                resolved_at = NULL,
                version = CASE WHEN status = 'DELIVERY_UNKNOWN' THEN version ELSE version + 1 END,
                updated_at = ?
            WHERE profile_id = ? AND instance_id = ?
              AND todo_id IN ({placeholders})""",
            (self.now, self.profile_id, self.instance_id, *todo_ids),
        )
        return (
            OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED.value
            if self.status is OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED
            else OutboxStatus.UNKNOWN_AFTER_CRASH.value
        )

    def _settle_file_asset_rows(
        self, conn: sqlite3.Connection, file_asset_ids: tuple[str, ...], file_target: str
    ) -> None:
        if not file_asset_ids:
            return
        asset_placeholders = ",".join("?" for _ in file_asset_ids)
        conn.execute(
            f"""UPDATE file_assets SET delivery_status = ?, last_error = ?, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?
              AND asset_id IN ({asset_placeholders})""",
            (
                file_target,
                ""
                if file_target == OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED.value
                else str(self.error or self.error_code or self.diagnostic_code or "")[:600],
                self.now,
                self.profile_id,
                self.instance_id,
                *file_asset_ids,
            ),
        )

    def _settle_contact(self, conn: sqlite3.Connection, payload: dict[str, Any]) -> None:
        attempt_ref = str(payload.get("contact_attempt_ref") or "").strip()
        if not attempt_ref:
            return
        generation = int(payload.get("contact_generation") or 0)
        task_id = int(payload.get("ai_task_id") or 0) or None
        attempt = conn.execute(
            """SELECT status, task_id FROM contact_attempts WHERE profile_id = ?
            AND instance_id = ? AND attempt_ref = ? AND generation = ?""",
            (self.profile_id, self.instance_id, attempt_ref, generation),
        ).fetchone()
        if attempt is None or (task_id is not None and attempt["task_id"] not in (None, task_id)):
            return
        attempted_unknown = self.status in {
            OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED,
            OutboxStatus.PARTIALLY_ATTEMPTED,
            OutboxStatus.UNKNOWN_AFTER_CRASH,
        }
        if attempted_unknown:
            outcome = "ATTEMPTED_UNKNOWN"
        else:
            failure_mode = str(payload.get("contact_failure_mode") or "SKIP").upper()
            outcome = "SUPERSEDED" if failure_mode == "SKIP" else "FAILED"
        target = {
            "ATTEMPTED_UNKNOWN": "RELEASED",
            "SUPERSEDED": "STALE",
            "FAILED": "RELEASED",
        }[outcome]
        ContactEvidenceSettlement(
            profile_id=self.profile_id,
            instance_id=self.instance_id,
            attempt_ref=attempt_ref,
            generation=generation,
            result=outcome,
            target=target,
            point=self.now,
        )(conn)
        finalize_contact_attempt_sql(
            conn,
            profile_id=self.profile_id,
            instance_id=self.instance_id,
            attempt_ref=attempt_ref,
            generation=generation,
            attempted=attempted_unknown,
            success=False,
            answered=False,
            task_id=task_id,
            current=self.now_dt,
            now_text=self.now,
        )

    def _record_sticker_usages(
        self,
        conn: sqlite3.Connection,
        outbox: sqlite3.Row,
        message: sqlite3.Row | None,
    ) -> None:
        if self.status is not OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED:
            return
        expression_ordinal = int(outbox["expression_ordinal"] or 0)
        for delivery in self.sticker_deliveries:
            record_sticker_usage_in_transaction(
                conn,
                self.profile_id,
                self.instance_id,
                item_id=str(delivery["item_id"]),
                run_id=delivery["run_id"],
                sticker_ref=str(delivery["sticker_ref"]),
                compact_projection=str(delivery["projection"]),
                delivery_status=self.status.value,
                outbox_id=self.outbox_id,
                expression_ordinal=expression_ordinal,
                message_id=(int(message["message_id"]) if message is not None else None),
                now=self.now,
            )

    def _write_platform_fragments(
        self,
        conn: sqlite3.Connection,
        outbox: sqlite3.Row,
        message: sqlite3.Row | None,
    ) -> None:
        if (
            message is None
            or self.status
            not in {
                OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED,
                OutboxStatus.PARTIALLY_ATTEMPTED,
            }
            or not self.receipts
        ):
            return
        kind, projection = _fragment_content(outbox, message)
        accepted_at = _parse(self.now) or _now()
        for receipt in self.receipts:
            platform_message_id = str(receipt.platform_message_id or "").strip()
            if not platform_message_id:
                continue
            fragment_ordinal = max(0, int(receipt.fragment_ordinal or 0))
            platform_id = str(receipt.platform_id or "").strip()
            platform_reference_id = str(receipt.platform_reference_id or "").strip()
            _insert_platform_fragment(
                conn,
                profile_id=self.profile_id,
                instance_id=self.instance_id,
                message=message,
                route_umo=str(outbox["route_umo"]),
                kind=kind,
                projection=projection,
                accepted_at=accepted_at,
                platform_message_id=platform_message_id,
                fragment_ordinal=fragment_ordinal,
                platform_id=platform_id,
                platform_reference_id=platform_reference_id,
                native_reply_supported=bool(receipt.native_reply_supported),
                member_mention_supported=bool(receipt.member_mention_supported),
                self_retraction_supported=bool(receipt.self_retraction_supported),
                returns_platform_message_id=bool(receipt.returns_platform_message_id),
                retractable_for_seconds=receipt.retractable_for_seconds,
                now=self.now,
            )

    def _load_current(self, conn: sqlite3.Connection) -> sqlite3.Row:
        current = conn.execute(
            """SELECT * FROM instance_outbox
            WHERE profile_id = ? AND instance_id = ? AND outbox_id = ?""",
            (self.profile_id, self.instance_id, self.outbox_id),
        ).fetchone()
        if current is None:
            raise KeyError((self.profile_id, self.instance_id, self.outbox_id))
        return current

    def _transition(self, conn: sqlite3.Connection, current_status: str) -> bool:
        if current_status == self.status.value:
            return True
        if current_status != OutboxStatus.SENDING.value:
            return False
        cursor = conn.execute(
            """UPDATE instance_outbox SET status = ?, last_error_code = ?, last_error = ?,
            last_diagnostic_code = ?,
            updated_at = ? WHERE profile_id = ? AND instance_id = ?
            AND outbox_id = ? AND status = ?""",
            (
                self.status.value,
                self.error_code,
                self.error,
                self.diagnostic_code,
                self.now,
                self.profile_id,
                self.instance_id,
                self.outbox_id,
                OutboxStatus.SENDING.value,
            ),
        )
        return cursor.rowcount == 1

    def _write_context_ledger(
        self,
        conn: sqlite3.Connection,
        outbox: sqlite3.Row,
    ) -> sqlite3.Row | None:
        linked_message_id = outbox["context_message_id"]
        if linked_message_id is not None:
            linked = conn.execute(
                """SELECT * FROM instance_messages
                WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
                (self.profile_id, self.instance_id, int(linked_message_id)),
            ).fetchone()
            if linked is None:
                raise KeyError((self.profile_id, self.instance_id, int(linked_message_id)))
            if (
                self.context_message is not None
                and str(outbox["status"]) == OutboxStatus.SENDING.value
            ):
                current_metadata = _load(linked["metadata_json"]) or {}
                final_metadata = self.context_message.get("metadata") or {}
                if not isinstance(current_metadata, dict) or not isinstance(final_metadata, dict):
                    raise ValueError("outbox context metadata must be an object")
                conn.execute(
                    """UPDATE instance_messages SET delivery_status = ?, plain_text = ?,
                    components_json = ?, metadata_json = ?
                    WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
                    (
                        self.status.value,
                        str(self.context_message.get("plain_text") or ""),
                        _dump(list(self.context_message.get("components") or [])),
                        _dump({**current_metadata, **final_metadata}),
                        self.profile_id,
                        self.instance_id,
                        int(linked_message_id),
                    ),
                )
            else:
                conn.execute(
                    """UPDATE instance_messages SET delivery_status = ?
                    WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
                    (
                        self.status.value,
                        self.profile_id,
                        self.instance_id,
                        int(linked_message_id),
                    ),
                )
            return conn.execute(
                "SELECT * FROM instance_messages WHERE message_id = ?",
                (int(linked_message_id),),
            ).fetchone()
        message = self.context_message
        if message is None:
            return None
        if self.status.value not in DIALOGUE_CONTINUITY_OUTBOUND_STATUSES:
            return None
        key = str(message.get("idempotency_key") or f"outbox:{self.outbox_id}").strip()
        existing = conn.execute(
            """SELECT * FROM instance_messages WHERE profile_id = ?
            AND instance_id = ? AND idempotency_key = ?""",
            (self.profile_id, self.instance_id, key),
        ).fetchone()
        if existing is not None:
            return existing
        return self._insert_context_ledger(conn, message, key)

    def _insert_context_ledger(
        self,
        conn: sqlite3.Connection,
        message: dict[str, Any],
        key: str,
    ) -> sqlite3.Row:
        cursor = conn.execute(
            """INSERT INTO instance_messages(
                profile_id, instance_id, direction, role, internal_memo, sender_id,
                sender_name, plain_text, identity_template, components_json,
                delivery_status, idempotency_key, metadata_json,
                occurred_at, created_at, knowledge_eligibility,
                knowledge_eligibility_reason, expression_batch_id,
                expression_ordinal
            ) VALUES (?, ?, 'OUTBOUND', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.profile_id,
                self.instance_id,
                str(message.get("role") or "assistant").lower(),
                str(message.get("internal_memo") or "").strip(),
                str(message.get("sender_id") or "soulcore"),
                str(message.get("sender_name") or "SoulCore"),
                str(message.get("plain_text") or ""),
                str(message.get("identity_template") or ""),
                _dump(list(message.get("components") or [])),
                self.status.value,
                key,
                _dump(message.get("metadata") or {}),
                _dt(message.get("occurred_at") or _now()),
                self.now,
                str(message.get("knowledge_eligibility") or "ELIGIBLE").upper(),
                str(message.get("knowledge_eligibility_reason") or ""),
                message.get("expression_batch_id"),
                message.get("expression_ordinal"),
            ),
        )
        row = conn.execute(
            "SELECT * FROM instance_messages WHERE message_id = ?",
            (int(cursor.lastrowid),),
        ).fetchone()
        assert row is not None
        return row


__all__ = ["_FinalizeOutboxDelivery"]
