from __future__ import annotations

from ....contracts.delivery_visibility import (
    outbox_todo_ids,
)
from ....storage.sqlite.expression_batch_lifecycle import (
    settle_pending_outbox_row,
    sync_expression_batch_status,
)
from ...ai import current_ai_work_context
from ...identity import validate_identity_template
from .expression_interruption_cleanup import restore_cancelled_file_todos
from .outbox_settlement_foreground import _FinalizeForegroundDelivery
from .outbox_settlement_outbox import _FinalizeOutboxDelivery
from .outbox_settlement_shared import (
    _cancel_terminal_expression_suffix,
    _resolve_terminal_group_window,
)
from .support import (
    INSTANCE_OUTBOX_SELECT,
    Any,
    ConversationMessage,
    OutboxInterruptPolicy,
    OutboxItem,
    OutboxStatus,
    _dt,
    _dump,
    _load,
    _now,
    datetime,
    sqlite3,
)
from .todo_ownership import bind_outbox_todos
from .voice_artifacts import schedule_outbox_voice_artifact_cleanup_sql


class _TransitionOutbox:
    def __init__(
        self,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        status: OutboxStatus,
        previous: tuple[OutboxStatus, ...],
        *,
        error_code: str,
        error: str | None,
        diagnostic_code: str,
        now: str,
    ) -> None:
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.outbox_id = int(outbox_id)
        self.status = status
        self.previous = previous
        self.error_code = str(error_code or "")
        self.error = error
        self.diagnostic_code = str(diagnostic_code or "")
        self.now = now

    def __call__(self, conn: sqlite3.Connection) -> bool:
        row = conn.execute(
            """SELECT * FROM instance_outbox
            WHERE profile_id = ? AND instance_id = ? AND outbox_id = ?""",
            (self.profile_id, self.instance_id, self.outbox_id),
        ).fetchone()
        if row is None:
            return False
        current = OutboxStatus(str(row["status"]))
        if current is OutboxStatus.PENDING and self.status in {
            OutboxStatus.CANCELLED,
            OutboxStatus.FAILED,
        }:
            changed = self._settle_pending(conn, row)
        else:
            changed = self._transition_regular(conn)
        if not changed:
            return False
        self._settle_dependents(conn, row)
        return True

    def _settle_pending(self, conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
        reason = str(self.error or self.error_code or self.status.value.lower())
        changed = settle_pending_outbox_row(
            conn,
            row,
            status=self.status,
            reason=reason,
            error_code=self.error_code,
            now=self.now,
        )
        if not changed:
            return False
        conn.execute(
            """UPDATE instance_outbox SET last_diagnostic_code = ?
            WHERE profile_id = ? AND instance_id = ? AND outbox_id = ?""",
            (self.diagnostic_code, self.profile_id, self.instance_id, self.outbox_id),
        )
        restore_cancelled_file_todos(
            conn,
            self.profile_id,
            self.instance_id,
            [row],
            self.now,
            reason,
            load_payload=_load,
        )
        return True

    def _transition_regular(self, conn: sqlite3.Connection) -> bool:
        placeholders = ",".join("?" for _ in self.previous)
        changed = conn.execute(
            f"UPDATE instance_outbox SET status = ?, last_error_code = ?, last_error = ?, "
            f"last_diagnostic_code = ?, updated_at = ? "
            f"WHERE profile_id = ? AND instance_id = ? AND outbox_id = ? "
            f"AND status IN ({placeholders})",
            (
                self.status.value,
                self.error_code,
                self.error,
                self.diagnostic_code,
                self.now,
                self.profile_id,
                self.instance_id,
                self.outbox_id,
                *(item.value for item in self.previous),
            ),
        ).rowcount
        return changed == 1

    def _settle_dependents(self, conn: sqlite3.Connection, row: sqlite3.Row) -> None:
        if self.status not in {OutboxStatus.PENDING, OutboxStatus.SENDING}:
            schedule_outbox_voice_artifact_cleanup_sql(
                conn,
                profile_id=self.profile_id,
                instance_id=self.instance_id,
                outbox_id=self.outbox_id,
                reason=f"voice_outbox_{self.status.value.lower()}",
                now=self.now,
            )
        _cancel_terminal_expression_suffix(conn, row, self.status, self.now)
        sync_expression_batch_status(conn, row["expression_batch_id"], self.now)
        _resolve_terminal_group_window(conn, row, self.status, self.now)


class OutboxRecords:
    async def enqueue_instance_outbox(
        self,
        profile_id: str,
        instance_id: str,
        payload: dict[str, Any],
        idempotency_key: str,
        activity_epoch: int = 0,
        *,
        origin_kind: str = "SYSTEM_EVENT",
        origin_run_id: int | None = None,
        origin_task_id: int | None = None,
        origin_wakeup_id: int | None = None,
        origin_generation: int | None = None,
        workflow_id: int | None = None,
        expression_batch_id: str | None = None,
        expression_ordinal: int | None = None,
        expression_step_ordinal: int | None = None,
        not_before_at: datetime | None = None,
        interrupt_policy: OutboxInterruptPolicy | str = OutboxInterruptPolicy.PRESERVE,
        depends_on_idempotency_key: str | None = None,
    ) -> OutboxItem:
        instance = await self._profiles.get_character_instance(profile_id, instance_id)
        if instance is None:
            raise KeyError((profile_id, instance_id))
        normalized_payload = dict(payload)
        scope = str(instance.scope)
        if "content" in normalized_payload:
            normalized_payload["content"] = str(
                validate_identity_template(
                    str(normalized_payload.get("content") or ""), scope=scope
                )
            )
        if "internal_memo" in normalized_payload:
            normalized_payload["internal_memo"] = str(
                validate_identity_template(
                    str(normalized_payload.get("internal_memo") or ""), scope=scope
                )
            )
        if (expression_batch_id is None) != (expression_ordinal is None):
            raise ValueError("expression_batch_id and expression_ordinal must be provided together")
        if expression_ordinal is not None and int(expression_ordinal) < 0:
            raise ValueError("expression_ordinal cannot be negative")
        if expression_step_ordinal is None:
            expression_step_ordinal = expression_ordinal
        if expression_step_ordinal is not None and int(expression_step_ordinal) < 0:
            raise ValueError("expression_step_ordinal cannot be negative")
        interrupt_policy = OutboxInterruptPolicy(interrupt_policy)
        now = _dt(_now())
        trace = current_ai_work_context()
        if workflow_id is None and trace is not None:
            workflow_id = trace.workflow_id

        def operation(conn: sqlite3.Connection) -> int:
            inserted = conn.execute(
                """INSERT INTO instance_outbox(
                    profile_id, instance_id, workflow_id, route_umo, payload_json, status,
                    idempotency_key, activity_epoch, origin_kind, origin_run_id,
                    origin_task_id, origin_wakeup_id, origin_generation,
                    expression_batch_id, expression_ordinal, expression_step_ordinal,
                    not_before_at,
                    interrupt_policy, depends_on_idempotency_key,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id, instance_id, idempotency_key) DO NOTHING""",
                (
                    profile_id,
                    instance_id,
                    workflow_id,
                    instance.route_umo,
                    _dump(normalized_payload),
                    OutboxStatus.PENDING.value,
                    idempotency_key,
                    activity_epoch,
                    str(origin_kind),
                    origin_run_id,
                    origin_task_id,
                    origin_wakeup_id,
                    origin_generation,
                    expression_batch_id,
                    expression_ordinal,
                    expression_step_ordinal,
                    _dt(not_before_at) if not_before_at is not None else None,
                    interrupt_policy.value,
                    depends_on_idempotency_key,
                    now,
                    now,
                ),
            )
            outbox_id = int(
                conn.execute(
                    """SELECT outbox_id FROM instance_outbox
                WHERE profile_id = ? AND instance_id = ? AND idempotency_key = ?""",
                    (profile_id, instance_id, idempotency_key),
                ).fetchone()[0]
            )
            if inserted.rowcount:
                bind_outbox_todos(
                    conn,
                    profile_id=profile_id,
                    instance_id=instance_id,
                    outbox_id=outbox_id,
                    todo_ids=outbox_todo_ids(normalized_payload),
                    selected_run_id=origin_run_id,
                )
            return outbox_id

        outbox_id = await self.uow.run(operation)
        item = await self.get_instance_outbox(profile_id, instance_id, outbox_id)
        assert item is not None
        return item

    async def get_instance_outbox(
        self, profile_id: str, instance_id: str, outbox_id: int
    ) -> OutboxItem | None:
        row = await self.db.fetch_one(
            f"""SELECT {INSTANCE_OUTBOX_SELECT}
            FROM instance_outbox
            WHERE profile_id = ? AND instance_id = ? AND outbox_id = ?""",
            (profile_id, instance_id, outbox_id),
        )
        return self._outbox(row) if row else None

    async def get_instance_outbox_by_idempotency_key(
        self,
        profile_id: str,
        instance_id: str,
        idempotency_key: str,
    ) -> OutboxItem | None:
        row = await self.db.fetch_one(
            f"""SELECT {INSTANCE_OUTBOX_SELECT}
            FROM instance_outbox WHERE profile_id = ? AND instance_id = ?
              AND idempotency_key = ?""",
            (profile_id, instance_id, str(idempotency_key)),
        )
        return self._outbox(row) if row else None

    async def list_instance_outbox(
        self,
        profile_id: str,
        instance_id: str,
        *,
        status: OutboxStatus | None = None,
        limit: int = 50,
    ) -> list[OutboxItem]:
        sql = f"""SELECT {INSTANCE_OUTBOX_SELECT} FROM instance_outbox
            WHERE profile_id = ? AND instance_id = ?"""
        params: list[Any] = [profile_id, instance_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(status.value)
        sql += " ORDER BY outbox_id DESC LIMIT ?"
        params.append(limit)
        return [self._outbox(row) for row in await self.db.fetch_all(sql, params)]

    async def list_profile_recent_failed_outbox(
        self,
        profile_id: str,
        *,
        limit_per_instance: int = 20,
    ) -> list[OutboxItem]:
        """Return failed items that are still inside each contact's recent window."""

        bounded_limit = max(1, min(int(limit_per_instance), 100))
        rows = await self.db.fetch_all(
            f"""WITH recent AS (
                SELECT instance_outbox.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY instance_id ORDER BY outbox_id DESC
                    ) AS recent_rank
                FROM instance_outbox
                WHERE profile_id = ?
            )
            SELECT {INSTANCE_OUTBOX_SELECT}
            FROM recent
            WHERE recent_rank <= ? AND status = 'FAILED'
            ORDER BY instance_id, outbox_id DESC""",
            (profile_id, bounded_limit),
        )
        return [self._outbox(row) for row in rows]

    async def claim_instance_outbox(
        self, profile_id: str, instance_id: str, outbox_id: int
    ) -> bool:
        claimed, _ = await self.begin_instance_outbox_dispatch(
            profile_id,
            instance_id,
            outbox_id,
        )
        return claimed

    async def transition_instance_outbox(
        self,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        status: OutboxStatus | str,
        *,
        error_code: str = "",
        error: str | None = None,
        diagnostic_code: str = "",
    ) -> bool:
        status = OutboxStatus(status)
        allowed: dict[OutboxStatus, set[OutboxStatus]] = {
            OutboxStatus.SENDING: {OutboxStatus.PENDING},
            OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED: {OutboxStatus.SENDING},
            OutboxStatus.PARTIALLY_ATTEMPTED: {OutboxStatus.SENDING},
            OutboxStatus.FAILED: {OutboxStatus.PENDING, OutboxStatus.SENDING},
            OutboxStatus.UNKNOWN_AFTER_CRASH: {OutboxStatus.SENDING},
            OutboxStatus.PENDING: {OutboxStatus.FAILED, OutboxStatus.SENDING},
            OutboxStatus.CANCELLED: {OutboxStatus.PENDING, OutboxStatus.SENDING},
        }
        previous = tuple(sorted(allowed.get(status, set()), key=lambda item: item.value))
        if not previous:
            return False
        now = _dt(_now())
        return bool(
            await self.uow.run(
                _TransitionOutbox(
                    profile_id,
                    instance_id,
                    outbox_id,
                    status,
                    previous,
                    error_code=error_code,
                    error=error,
                    diagnostic_code=diagnostic_code,
                    now=now,
                )
            )
        )


class OutboxSettlementCommands:
    async def finalize_foreground_delivery(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
        *,
        media_asset_ids: tuple[str, ...] = (),
        todo_ids: tuple[str, ...] = (),
        status: OutboxStatus | str = OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED,
        error: str = "",
        receipts: tuple[Any, ...] = (),
        sticker_deliveries: tuple[dict[str, Any], ...] = (),
        route_umo: str = "",
    ) -> None:
        """Atomically persist an attempted or definitely unattempted foreground result."""

        normalized_status = OutboxStatus(status)
        if normalized_status not in {
            OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED,
            OutboxStatus.PARTIALLY_ATTEMPTED,
            OutboxStatus.UNKNOWN_AFTER_CRASH,
            OutboxStatus.FAILED,
        }:
            raise ValueError("unsupported foreground delivery settlement status")
        normalized_media = tuple(
            dict.fromkeys(str(value) for value in media_asset_ids if str(value))
        )
        normalized_todos = tuple(dict.fromkeys(str(value) for value in todo_ids if str(value)))
        now_dt = _now()
        await self.uow.run(
            _FinalizeForegroundDelivery(
                self,
                profile_id=str(profile_id),
                instance_id=str(instance_id),
                message_id=int(message_id),
                media_asset_ids=normalized_media,
                todo_ids=normalized_todos,
                status=normalized_status,
                error=str(error or "")[:1000],
                receipts=tuple(receipts),
                sticker_deliveries=tuple(sticker_deliveries),
                route_umo=str(route_umo or ""),
                now=_dt(now_dt),
                now_dt=now_dt,
            )
        )

    async def finalize_instance_outbox_delivery(
        self,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        status: OutboxStatus | str,
        *,
        error_code: str = "",
        error: str | None = None,
        diagnostic_code: str = "",
        context_message: dict[str, Any] | None = None,
        receipts: tuple[Any, ...] = (),
        sticker_deliveries: tuple[dict[str, Any], ...] = (),
    ) -> tuple[bool, ConversationMessage | None]:
        """Atomically finalize delivery and its optional context-ledger row.

        Publishing the stable backup is part of the safety contract. If that
        publication fails, the stale pre-send snapshot is removed so recovery
        can never turn an already-attempted delivery back into ``PENDING``.
        """

        status = OutboxStatus(status)
        allowed_from_sending = {
            OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED,
            OutboxStatus.PARTIALLY_ATTEMPTED,
            OutboxStatus.FAILED,
            OutboxStatus.PENDING,
            OutboxStatus.UNKNOWN_AFTER_CRASH,
            OutboxStatus.CANCELLED,
        }
        if status not in allowed_from_sending:
            raise ValueError("unsupported instance outbox final status")
        now_dt = _now()
        now = _dt(now_dt)
        operation = _FinalizeOutboxDelivery(
            profile_id=profile_id,
            instance_id=instance_id,
            outbox_id=outbox_id,
            status=status,
            error_code=error_code,
            error=error,
            diagnostic_code=diagnostic_code,
            context_message=context_message,
            receipts=tuple(receipts),
            sticker_deliveries=tuple(sticker_deliveries),
            now=now,
            now_dt=now_dt,
        )
        updated, row = await self.uow.run(operation)
        await self.publish_context_backup()
        return updated, self._conversation_message(row) if row else None
