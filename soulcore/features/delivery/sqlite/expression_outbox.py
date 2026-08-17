from __future__ import annotations

from ....storage.sqlite.expression_batch_lifecycle import sync_expression_batch_status
from .expression_pending_inbound import has_pending_expression_inbound
from .support import (
    Any,
    ConversationMessage,
    ExpressionBatch,
    ExpressionBatchStatus,
    OutboxItem,
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

ATTEMPTED_OUTBOX_STATUSES = (
    OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED.value,
    OutboxStatus.UNKNOWN_AFTER_CRASH.value,
)

EXPRESSION_OUTBOX_SELECT = """candidate.outbox_id, candidate.profile_id,
    candidate.instance_id, candidate.route_umo AS umo, candidate.payload_json,
    candidate.status, candidate.idempotency_key, candidate.attempts,
    candidate.activity_epoch, candidate.last_error_code, candidate.last_error,
    candidate.last_diagnostic_code, candidate.created_at, candidate.updated_at,
    candidate.expression_batch_id,
    candidate.expression_ordinal,
    COALESCE(candidate.expression_step_ordinal, candidate.expression_ordinal)
        AS expression_step_ordinal,
    candidate.not_before_at,
    candidate.interrupt_policy, candidate.depends_on_idempotency_key,
    candidate.context_message_id"""


def defer_following_expression_step(
    conn: sqlite3.Connection,
    batch_id: str | None,
    after_step_ordinal: int | None,
    now: str,
) -> None:
    if not batch_id or after_step_ordinal is None:
        return
    candidates = list(
        conn.execute(
            """SELECT 'OUTBOX' AS source, outbox_id AS row_id,
                COALESCE(expression_step_ordinal, expression_ordinal) AS step_ordinal,
                payload_json, NULL AS delay_seconds, not_before_at
            FROM instance_outbox
            WHERE expression_batch_id = ? AND status = 'PENDING'
              AND COALESCE(expression_step_ordinal, expression_ordinal) > ?
            UNION ALL
            SELECT 'RETRACT' AS source, action_id AS row_id, step_ordinal,
                NULL AS payload_json, delay_after_previous_seconds AS delay_seconds,
                not_before_at
            FROM message_retraction_actions
            WHERE expression_batch_id = ? AND status = 'PENDING' AND step_ordinal > ?
            ORDER BY step_ordinal LIMIT 1""",
            (batch_id, int(after_step_ordinal), batch_id, int(after_step_ordinal)),
        )
    )
    if not candidates:
        batch = conn.execute(
            """SELECT profile_id, instance_id, source_run_id, segment_index
            FROM instance_expression_batches WHERE batch_id = ?""",
            (str(batch_id),),
        ).fetchone()
        if batch is None:
            return
        candidates = list(
            conn.execute(
                """SELECT * FROM (
                    SELECT 'OUTBOX' AS source, item.outbox_id AS row_id,
                        next_batch.segment_index,
                        COALESCE(item.expression_step_ordinal, item.expression_ordinal)
                            AS step_ordinal,
                        item.payload_json, NULL AS delay_seconds, item.not_before_at
                    FROM instance_expression_batches next_batch
                    JOIN instance_outbox item
                      ON item.expression_batch_id = next_batch.batch_id
                    WHERE next_batch.profile_id = ? AND next_batch.instance_id = ?
                      AND next_batch.source_run_id = ?
                      AND next_batch.segment_index > ?
                      AND item.status = 'PENDING'
                    UNION ALL
                    SELECT 'RETRACT' AS source, action.action_id AS row_id,
                        next_batch.segment_index, action.step_ordinal,
                        NULL AS payload_json,
                        action.delay_after_previous_seconds AS delay_seconds,
                        action.not_before_at
                    FROM instance_expression_batches next_batch
                    JOIN message_retraction_actions action
                      ON action.expression_batch_id = next_batch.batch_id
                    WHERE next_batch.profile_id = ? AND next_batch.instance_id = ?
                      AND next_batch.source_run_id = ?
                      AND next_batch.segment_index > ?
                      AND action.status = 'PENDING'
                ) ORDER BY segment_index, step_ordinal LIMIT 1""",
                (
                    str(batch["profile_id"]),
                    str(batch["instance_id"]),
                    int(batch["source_run_id"]),
                    int(batch["segment_index"]),
                    str(batch["profile_id"]),
                    str(batch["instance_id"]),
                    int(batch["source_run_id"]),
                    int(batch["segment_index"]),
                ),
            )
        )
        if not candidates:
            return
    row = candidates[0]
    if row["source"] == "OUTBOX":
        payload = _load(row["payload_json"]) or {}
        delay = max(0, int(payload.get("delay_after_previous_seconds") or 0))
    else:
        delay = max(0, int(row["delay_seconds"] or 0))
    target = _dt((_parse(now) or _now()) + timedelta(seconds=delay))
    if row["not_before_at"] is not None and str(row["not_before_at"]) >= target:
        return
    if row["source"] == "OUTBOX":
        conn.execute(
            "UPDATE instance_outbox SET not_before_at = ?, updated_at = ? WHERE outbox_id = ?",
            (target, now, int(row["row_id"])),
        )
    else:
        conn.execute(
            """UPDATE message_retraction_actions SET not_before_at = ?, updated_at = ?
            WHERE action_id = ?""",
            (target, now, int(row["row_id"])),
        )


def _is_file_expression(row: sqlite3.Row) -> bool:
    payload = _load(row["payload_json"]) or {}
    if str(payload.get("expression_kind") or "").upper() == "FILE":
        return True
    if str(payload.get("file_delivery_role") or "").upper() == "ARTIFACT":
        return True
    return any(
        isinstance(component, dict) and str(component.get("type") or "").lower() == "file_artifact"
        for component in list(payload.get("components") or [])
    )


def _activity_epoch_allows(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    if str(row["interrupt_policy"]) == "PRESERVE":
        return True
    state = conn.execute(
        """SELECT activity_epoch FROM instance_core_state
        WHERE profile_id = ? AND instance_id = ?""",
        (str(row["profile_id"]), str(row["instance_id"])),
    ).fetchone()
    return state is not None and int(state["activity_epoch"]) == int(row["activity_epoch"])


def _batch_predecessors_finished(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    batch_id = row["expression_batch_id"]
    if batch_id is None:
        return True
    batch = conn.execute(
        """SELECT status, profile_id, instance_id, source_run_id, segment_index
        FROM instance_expression_batches WHERE batch_id = ?""",
        (str(batch_id),),
    ).fetchone()
    if batch is None or str(batch["status"]) != ExpressionBatchStatus.ACTIVE.value:
        return False
    blocked = conn.execute(
        """SELECT 1 FROM (
            SELECT COALESCE(expression_step_ordinal, expression_ordinal)
                AS step_ordinal, status
            FROM instance_outbox WHERE expression_batch_id = ?
            UNION ALL
            SELECT step_ordinal, status FROM message_retraction_actions
            WHERE expression_batch_id = ?
        ) WHERE step_ordinal < ? AND status IN (?, ?) LIMIT 1""",
        (
            str(batch_id),
            str(batch_id),
            int(row["expression_step_ordinal"]),
            OutboxStatus.PENDING.value,
            OutboxStatus.SENDING.value,
        ),
    ).fetchone()
    if blocked is not None:
        return False
    prior_segment = conn.execute(
        """SELECT 1 FROM (
            SELECT predecessor.status
            FROM instance_expression_batches previous
            JOIN instance_outbox predecessor
              ON predecessor.expression_batch_id = previous.batch_id
            WHERE previous.profile_id = ? AND previous.instance_id = ?
              AND previous.source_run_id = ? AND previous.segment_index < ?
            UNION ALL
            SELECT predecessor.status
            FROM instance_expression_batches previous
            JOIN message_retraction_actions predecessor
              ON predecessor.expression_batch_id = previous.batch_id
            WHERE previous.profile_id = ? AND previous.instance_id = ?
              AND previous.source_run_id = ? AND previous.segment_index < ?
        ) WHERE status IN (?, ?) LIMIT 1""",
        (
            str(batch["profile_id"]),
            str(batch["instance_id"]),
            int(batch["source_run_id"]),
            int(batch["segment_index"]),
            str(batch["profile_id"]),
            str(batch["instance_id"]),
            int(batch["source_run_id"]),
            int(batch["segment_index"]),
            OutboxStatus.PENDING.value,
            OutboxStatus.SENDING.value,
        ),
    ).fetchone()
    return prior_segment is None


def _dependency_finished(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    dependency_key = row["depends_on_idempotency_key"]
    if dependency_key is None:
        return True
    dependency = conn.execute(
        """SELECT status FROM instance_outbox WHERE profile_id = ?
        AND instance_id = ? AND idempotency_key = ?""",
        (
            str(row["profile_id"]),
            str(row["instance_id"]),
            str(dependency_key),
        ),
    ).fetchone()
    if _is_file_expression(row):
        return dependency is not None and str(dependency["status"]) in ATTEMPTED_OUTBOX_STATUSES
    if dependency is None:
        return True
    return str(dependency["status"]) not in {
        OutboxStatus.PENDING.value,
        OutboxStatus.SENDING.value,
    }


def _expression_row_claimable(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    now: str | None,
) -> bool:
    if str(row["status"]) != OutboxStatus.PENDING.value:
        return False
    enabled = conn.execute(
        "SELECT enabled FROM role_profiles WHERE profile_id = ?",
        (str(row["profile_id"]),),
    ).fetchone()
    if enabled is None or not bool(enabled["enabled"]):
        return False
    if now is not None and row["not_before_at"] is not None and str(row["not_before_at"]) > now:
        return False
    if has_pending_expression_inbound(conn, str(row["expression_batch_id"])):
        return False
    return (
        _activity_epoch_allows(conn, row)
        and _batch_predecessors_finished(conn, row)
        and _dependency_finished(conn, row)
    )


def _claimable_expression_rows(
    conn: sqlite3.Connection,
    *,
    now: str | None,
    limit: int,
) -> list[sqlite3.Row]:
    if limit <= 0:
        return []
    rows = conn.execute(
        f"""SELECT {EXPRESSION_OUTBOX_SELECT} FROM instance_outbox candidate
        JOIN instance_expression_batches batch
          ON batch.batch_id = candidate.expression_batch_id
        JOIN role_profiles profile ON profile.profile_id = candidate.profile_id
        WHERE candidate.expression_batch_id IS NOT NULL
          AND candidate.status = 'PENDING' AND batch.status = 'ACTIVE'
          AND profile.enabled = 1
        ORDER BY COALESCE(candidate.not_before_at, candidate.created_at),
          candidate.expression_batch_id,
          COALESCE(candidate.expression_step_ordinal, candidate.expression_ordinal)"""
    )
    result: list[sqlite3.Row] = []
    for row in rows:
        if _expression_row_claimable(conn, row, now=now):
            result.append(row)
            if len(result) >= limit:
                break
    return result


class _BeginInstanceOutboxDispatch:
    def __init__(
        self,
        *,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        context_message: dict[str, Any] | None,
        now: str,
    ) -> None:
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.outbox_id = int(outbox_id)
        self.context_message = context_message
        self.now = now

    def __call__(self, conn: sqlite3.Connection) -> tuple[bool, sqlite3.Row | None]:
        row = self._load_outbox(conn)
        if str(row["status"]) != OutboxStatus.PENDING.value:
            return False, self._linked_message(conn, row)
        if not self._is_claimable(conn, row):
            return False, None
        cursor = conn.execute(
            """UPDATE instance_outbox SET status = ?, attempts = attempts + 1,
            last_error_code = '', last_error = NULL, last_diagnostic_code = '',
            updated_at = ? WHERE outbox_id = ? AND status = ?""",
            (
                OutboxStatus.SENDING.value,
                self.now,
                self.outbox_id,
                OutboxStatus.PENDING.value,
            ),
        )
        if cursor.rowcount != 1:
            return False, None
        message = self._create_expression_ledger(conn, row)
        sync_expression_batch_status(conn, row["expression_batch_id"], self.now)
        return True, message

    def _load_outbox(self, conn: sqlite3.Connection) -> sqlite3.Row:
        row = conn.execute(
            """SELECT * FROM instance_outbox WHERE profile_id = ?
            AND instance_id = ? AND outbox_id = ?""",
            (self.profile_id, self.instance_id, self.outbox_id),
        ).fetchone()
        if row is None:
            raise KeyError((self.profile_id, self.instance_id, self.outbox_id))
        return row

    @staticmethod
    def _linked_message(
        conn: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> sqlite3.Row | None:
        message_id = row["context_message_id"]
        if message_id is None:
            return None
        return conn.execute(
            "SELECT * FROM instance_messages WHERE message_id = ?",
            (int(message_id),),
        ).fetchone()

    def _is_claimable(self, conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
        return _expression_row_claimable(conn, row, now=self.now)

    def _create_expression_ledger(
        self,
        conn: sqlite3.Connection,
        outbox: sqlite3.Row,
    ) -> sqlite3.Row | None:
        if outbox["expression_batch_id"] is None:
            return None
        existing = self._linked_message(conn, outbox)
        if existing is not None:
            return existing
        message = self.context_message
        payload = _load(outbox["payload_json"]) or {}
        if not bool(payload.get("context_record", True)):
            return None
        if message is None:
            raise ValueError("expression dispatch requires a context ledger message")
        existing = self._find_or_insert_ledger(conn, outbox, message)
        conn.execute(
            "UPDATE instance_outbox SET context_message_id = ? WHERE outbox_id = ?",
            (int(existing["message_id"]), self.outbox_id),
        )
        return existing

    def _find_or_insert_ledger(
        self,
        conn: sqlite3.Connection,
        outbox: sqlite3.Row,
        message: dict[str, Any],
    ) -> sqlite3.Row:
        key = str(message.get("idempotency_key") or f"outbox:{self.outbox_id}").strip()
        existing = conn.execute(
            """SELECT * FROM instance_messages WHERE profile_id = ?
            AND instance_id = ? AND idempotency_key = ?""",
            (self.profile_id, self.instance_id, key),
        ).fetchone()
        if existing is not None:
            return existing
        cursor = conn.execute(
            """INSERT INTO instance_messages(
                profile_id, instance_id, direction, role, internal_memo, sender_id,
                sender_name, plain_text, identity_template, components_json, delivery_status,
                idempotency_key, metadata_json, occurred_at, created_at,
                knowledge_eligibility, knowledge_eligibility_reason,
                expression_batch_id, expression_ordinal
            ) VALUES (?, ?, 'OUTBOUND', ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                key,
                _dump(message.get("metadata") or {}),
                _dt(message.get("occurred_at") or _now()),
                self.now,
                str(message.get("knowledge_eligibility") or "ELIGIBLE").upper(),
                str(message.get("knowledge_eligibility_reason") or ""),
                str(outbox["expression_batch_id"]),
                int(outbox["expression_ordinal"]),
            ),
        )
        row = conn.execute(
            "SELECT * FROM instance_messages WHERE message_id = ?",
            (int(cursor.lastrowid),),
        ).fetchone()
        assert row is not None
        return row


class ExpressionOutboxRecords:
    async def get_expression_batch(
        self,
        profile_id: str,
        instance_id: str,
        batch_id: str,
    ) -> ExpressionBatch | None:
        row = await self.db.fetch_one(
            """SELECT * FROM instance_expression_batches WHERE profile_id = ?
            AND instance_id = ? AND batch_id = ?""",
            (profile_id, instance_id, str(batch_id)),
        )
        return self._expression_batch(row) if row else None

    async def list_expression_batches(
        self,
        profile_id: str,
        instance_id: str,
        *,
        status: ExpressionBatchStatus | str | None = None,
        limit: int = 50,
    ) -> list[ExpressionBatch]:
        sql = """SELECT * FROM instance_expression_batches
        WHERE profile_id = ? AND instance_id = ?"""
        params: list[Any] = [profile_id, instance_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(ExpressionBatchStatus(status).value)
        sql += " ORDER BY created_at DESC, batch_id DESC LIMIT ?"
        params.append(max(0, int(limit)))
        return [self._expression_batch(row) for row in await self.db.fetch_all(sql, params)]

    async def list_due_expression_outbox(
        self,
        *,
        now: datetime | None = None,
        limit: int = 50,
    ) -> list[OutboxItem]:
        current = _dt(now or _now())
        rows = await self.db.call(
            lambda conn: _claimable_expression_rows(
                conn,
                now=current,
                limit=max(0, int(limit)),
            )
        )
        return [self._outbox(row) for row in rows]

    async def next_expression_outbox_due_at(self) -> datetime | None:
        rows = await self.db.call(lambda conn: _claimable_expression_rows(conn, now=None, limit=1))
        if not rows:
            return None
        return _parse(rows[0]["not_before_at"]) or _parse(rows[0]["created_at"]) or _now()

    async def defer_expression_batch_suffix(
        self,
        profile_id: str,
        instance_id: str,
        batch_id: str,
        from_ordinal: int,
        not_before_at: datetime,
        *,
        error: str | None = None,
    ) -> int:
        ordinal = int(from_ordinal)
        if ordinal < 0:
            raise ValueError("from_ordinal cannot be negative")
        if not_before_at.tzinfo is None:
            raise ValueError("not_before_at must be timezone-aware")
        return int(
            await self.uow.run(
                _DeferExpressionSuffix(
                    profile_id=profile_id,
                    instance_id=instance_id,
                    batch_id=str(batch_id),
                    ordinal=ordinal,
                    target=not_before_at,
                    error=error,
                )
            )
        )

    async def begin_instance_outbox_dispatch(
        self,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        *,
        context_message: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> tuple[bool, ConversationMessage | None]:
        operation = _BeginInstanceOutboxDispatch(
            profile_id=profile_id,
            instance_id=instance_id,
            outbox_id=outbox_id,
            context_message=context_message,
            now=_dt(now or _now()),
        )
        claimed, row = await self.uow.run(operation)
        return bool(claimed), self._conversation_message(row) if row else None


class _DeferExpressionSuffix:
    def __init__(
        self,
        *,
        profile_id: str,
        instance_id: str,
        batch_id: str,
        ordinal: int,
        target: datetime,
        error: str | None,
    ) -> None:
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.batch_id = batch_id
        self.ordinal = ordinal
        self.target = target
        self.error = error

    def __call__(self, conn: sqlite3.Connection) -> int:
        rows = list(
            conn.execute(
                """SELECT outbox_id, expression_ordinal, not_before_at
                FROM instance_outbox WHERE profile_id = ? AND instance_id = ?
                  AND expression_batch_id = ? AND expression_ordinal >= ?
                  AND status = ? ORDER BY expression_ordinal""",
                (
                    self.profile_id,
                    self.instance_id,
                    self.batch_id,
                    self.ordinal,
                    OutboxStatus.PENDING.value,
                ),
            )
        )
        if not rows:
            return 0
        current = _parse(rows[0]["not_before_at"]) or _now()
        if self.target <= current:
            return 0
        return self._shift_rows(conn, rows, current, self.target - current)

    def _shift_rows(
        self,
        conn: sqlite3.Connection,
        rows: list[sqlite3.Row],
        current: datetime,
        shift: timedelta,
    ) -> int:
        changed = 0
        now_text = _dt(_now())
        for row in rows:
            original = _parse(row["not_before_at"]) or current
            cursor = conn.execute(
                """UPDATE instance_outbox SET not_before_at = ?, last_error = ?,
                updated_at = ? WHERE outbox_id = ? AND status = ?""",
                (
                    _dt(original + shift),
                    self.error,
                    now_text,
                    int(row["outbox_id"]),
                    OutboxStatus.PENDING.value,
                ),
            )
            changed += int(cursor.rowcount)
        return changed


__all__ = [
    "ATTEMPTED_OUTBOX_STATUSES",
    "ExpressionOutboxRecords",
]
