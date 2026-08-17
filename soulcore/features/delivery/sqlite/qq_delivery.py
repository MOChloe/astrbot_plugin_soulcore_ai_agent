from __future__ import annotations

from ....storage.sqlite.expression_batch_lifecycle import sync_expression_batch_status
from .group_first_attempt import mark_group_first_attempt_started
from .outbox_settlement_outbox import _FinalizeOutboxDelivery
from .support import (
    Any,
    OutboxStatus,
    _dt,
    _dump,
    _load,
    _now,
    _parse,
    _wakeup_period,
    datetime,
    sqlite3,
    timedelta,
    timezone,
)


class QqDeliveryRecords:
    async def recover_sending_instance_outbox(self) -> int:
        now_dt = _now()
        now = _dt(now_dt)

        def operation(conn: sqlite3.Connection) -> int:
            sending = list(
                conn.execute(
                    """SELECT outbox_id, profile_id, instance_id, expression_batch_id,
                    context_message_id, payload_json
                FROM instance_outbox WHERE status = ?""",
                    (OutboxStatus.SENDING.value,),
                )
            )
            retryable: list[sqlite3.Row] = []
            unknown: list[sqlite3.Row] = []
            for row in sending:
                target = retryable if _expression_permits_prove_uncalled(conn, row) else unknown
                target.append(row)
            batches = {
                str(row["expression_batch_id"])
                for row in sending
                if row["expression_batch_id"] is not None
            }
            _recover_unattempted_expression_outbox(conn, retryable, now)
            for row in unknown:
                _settle_crashed_expression_permits(conn, row, now)
                updated, _ = _FinalizeOutboxDelivery(
                    profile_id=str(row["profile_id"]),
                    instance_id=str(row["instance_id"]),
                    outbox_id=int(row["outbox_id"]),
                    status=OutboxStatus.UNKNOWN_AFTER_CRASH,
                    error_code="DELIVERY_INTERRUPTED",
                    error="outbox_unknown_after_crash",
                    diagnostic_code="recovered_platform_call_unknown_after_crash",
                    context_message=None,
                    receipts=(),
                    sticker_deliveries=(),
                    now=now,
                    now_dt=now_dt,
                )(conn)
                if not updated:
                    raise RuntimeError("sending outbox recovery ownership changed")
            for batch_id in batches:
                sync_expression_batch_status(conn, batch_id, now)
            return len(sending)

        return int(await self.uow.run(operation))

    async def record_instance_inbound_delivery(
        self,
        profile_id: str,
        instance_id: str,
        message_id: str,
        *,
        received_at: datetime | None = None,
    ) -> bool:
        if await self._profiles.get_character_instance(profile_id, instance_id) is None:
            raise KeyError((profile_id, instance_id))
        message_id = str(message_id or "").strip()
        if not message_id:
            return False
        received = received_at or _now()
        if received.tzinfo is None:
            received = received.replace(tzinfo=timezone.utc)
        now = _dt(_now())
        await self.db.call(
            lambda conn: conn.execute(
                """INSERT INTO instance_delivery_state(
                    profile_id, instance_id, inbound_message_id,
                    inbound_received_at, passive_reply_uses,
                    wakeup_periods_json, last_status, updated_at
                ) VALUES (?, ?, ?, ?, 0, '{}', 'INBOUND_REFRESHED', ?)
                ON CONFLICT(profile_id, instance_id) DO UPDATE SET
                    inbound_message_id = excluded.inbound_message_id,
                    inbound_received_at = excluded.inbound_received_at,
                    passive_reply_uses = 0, wakeup_periods_json = '{}',
                    last_mode = NULL, last_status = 'INBOUND_REFRESHED',
                    last_error = NULL, updated_at = excluded.updated_at""",
                (profile_id, instance_id, message_id, _dt(received), now),
            ),
            transaction=True,
        )
        return True

    async def get_instance_delivery_state(
        self, profile_id: str, instance_id: str
    ) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            """SELECT * FROM instance_delivery_state
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        )
        return self._record(row, json_columns=("wakeup_periods_json",)) if row else None

    async def reserve_instance_qq_delivery(
        self,
        profile_id: str,
        instance_id: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = now or _now()
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        current = current.astimezone(timezone.utc)

        def operation(conn: sqlite3.Connection) -> dict[str, Any]:
            row = conn.execute(
                """SELECT * FROM instance_delivery_state
                WHERE profile_id = ? AND instance_id = ?""",
                (profile_id, instance_id),
            ).fetchone()
            if row is None or not row["inbound_message_id"] or not row["inbound_received_at"]:
                return {"mode": "unavailable", "reason": "no_real_inbound_message"}
            inbound_at = _parse(row["inbound_received_at"])
            assert inbound_at is not None
            uses = int(row["passive_reply_uses"] or 0)
            if current - inbound_at <= timedelta(minutes=60) and uses < 4:
                sequence = uses + 1
                conn.execute(
                    """UPDATE instance_delivery_state SET passive_reply_uses = ?,
                        last_mode = 'passive_reply', last_status = 'RESERVED',
                        last_error = NULL, updated_at = ?
                    WHERE profile_id = ? AND instance_id = ?""",
                    (sequence, _dt(current), profile_id, instance_id),
                )
                return {
                    "mode": "passive_reply",
                    "message_id": row["inbound_message_id"],
                    "msg_seq": sequence,
                }
            period = _wakeup_period(inbound_at, current)
            periods = _load(row["wakeup_periods_json"]) or {}
            if period and period not in periods:
                periods[period] = {"status": "RESERVED", "reserved_at": _dt(current)}
                conn.execute(
                    """UPDATE instance_delivery_state SET wakeup_periods_json = ?,
                        last_mode = 'wakeup', last_status = 'RESERVED',
                        last_error = NULL, updated_at = ?
                    WHERE profile_id = ? AND instance_id = ?""",
                    (_dump(periods), _dt(current), profile_id, instance_id),
                )
                return {"mode": "wakeup", "period": period}
            return {"mode": "unavailable", "reason": "wakeup_period_exhausted"}

        return await self.uow.run(operation)

    async def finalize_instance_qq_delivery(
        self,
        profile_id: str,
        instance_id: str,
        reservation: dict[str, Any],
        *,
        accepted: bool,
        attempted: bool = True,
        error: str | None = None,
    ) -> None:
        now = _now()
        mode = str(reservation.get("mode") or "")

        def operation(conn: sqlite3.Connection) -> None:
            row = conn.execute(
                """SELECT wakeup_periods_json FROM instance_delivery_state
                WHERE profile_id = ? AND instance_id = ?""",
                (profile_id, instance_id),
            ).fetchone()
            if row is None:
                return
            periods = _load(row["wakeup_periods_json"]) or {}
            period = reservation.get("period")
            if mode == "wakeup" and period in periods and not attempted:
                periods.pop(period, None)
            elif mode == "wakeup" and period in periods:
                periods[period] = {
                    **periods[period],
                    "status": (
                        "PLATFORM_ACCEPTED_UNCONFIRMED" if accepted else "ATTEMPTED_UNKNOWN"
                    ),
                    "finished_at": _dt(now),
                    "error": error,
                }
            if mode == "passive_reply" and not attempted:
                sequence = max(0, int(reservation.get("msg_seq") or 0))
                conn.execute(
                    """UPDATE instance_delivery_state
                    SET passive_reply_uses = passive_reply_uses - 1
                    WHERE profile_id = ? AND instance_id = ?
                      AND passive_reply_uses = ? AND last_status = 'RESERVED'""",
                    (profile_id, instance_id, sequence),
                )
            conn.execute(
                """UPDATE instance_delivery_state SET wakeup_periods_json = ?,
                    last_mode = ?, last_status = ?, last_error = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?""",
                (
                    _dump(periods),
                    mode,
                    (
                        "PLATFORM_ACCEPTED_UNCONFIRMED"
                        if accepted
                        else ("ROLLED_BACK" if not attempted else "ATTEMPTED_UNKNOWN")
                    ),
                    error,
                    _dt(now),
                    profile_id,
                    instance_id,
                ),
            )

        await self.uow.run(operation)


def _expression_permits_prove_uncalled(
    conn: sqlite3.Connection,
    outbox: sqlite3.Row,
) -> bool:
    if outbox["expression_batch_id"] is None:
        return False
    permits = conn.execute(
        """SELECT COUNT(*) AS total,
          SUM(CASE WHEN status IN (
            'RESERVED', 'RELEASED', 'FAILED_BEFORE_DISPATCH'
          ) THEN 1 ELSE 0 END) AS proven_uncalled
        FROM platform_send_permits
        WHERE profile_id = ? AND instance_id = ?
          AND origin_kind = 'EXPRESSION_ITEM' AND origin_id = ?""",
        (
            str(outbox["profile_id"]),
            str(outbox["instance_id"]),
            f"expression-outbox:{int(outbox['outbox_id'])}",
        ),
    ).fetchone()
    if permits is None:
        return False
    total = int(permits["total"] or 0)
    return total > 0 and int(permits["proven_uncalled"] or 0) == total


def _recover_unattempted_expression_outbox(
    conn: sqlite3.Connection,
    rows: list[sqlite3.Row],
    now: str,
) -> None:
    for row in rows:
        changed = conn.execute(
            """UPDATE instance_outbox SET status = 'PENDING',
            last_diagnostic_code = 'recovered_before_platform_call',
            updated_at = ? WHERE outbox_id = ? AND status = 'SENDING'""",
            (now, int(row["outbox_id"])),
        ).rowcount
        if changed != 1:
            raise RuntimeError("sending outbox recovery ownership changed")
        if row["context_message_id"] is not None:
            conn.execute(
                """UPDATE instance_messages SET delivery_status = 'PENDING'
                WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
                (
                    str(row["profile_id"]),
                    str(row["instance_id"]),
                    int(row["context_message_id"]),
                ),
            )


def _settle_crashed_expression_permits(
    conn: sqlite3.Connection,
    outbox: sqlite3.Row,
    now: str,
) -> None:
    if outbox["expression_batch_id"] is None:
        return
    identity = (
        str(outbox["profile_id"]),
        str(outbox["instance_id"]),
        f"expression-outbox:{int(outbox['outbox_id'])}",
    )
    started = conn.execute(
        """UPDATE platform_send_permits SET status = 'ATTEMPTED_UNKNOWN',
        detail = 'recovered_platform_call_unknown_after_crash', updated_at = ?
        WHERE profile_id = ? AND instance_id = ?
          AND origin_kind = 'EXPRESSION_ITEM' AND origin_id = ?
          AND status = 'DISPATCHING'""",
        (now, *identity),
    ).rowcount
    if started:
        payload = _load(outbox["payload_json"]) or {}
        if not isinstance(payload, dict):
            raise ValueError("outbox payload must be an object")
        group_window_id = str(payload.get("group_window_id") or "").strip()
        if group_window_id:
            mark_group_first_attempt_started(
                conn,
                profile_id=identity[0],
                instance_id=identity[1],
                group_window_id=group_window_id,
                now=now,
            )
    conn.execute(
        """UPDATE platform_send_permits SET status = 'RELEASED',
        detail = 'released_after_crashed_platform_attempt', updated_at = ?
        WHERE profile_id = ? AND instance_id = ?
          AND origin_kind = 'EXPRESSION_ITEM' AND origin_id = ?
          AND status = 'RESERVED'""",
        (now, *identity),
    )
