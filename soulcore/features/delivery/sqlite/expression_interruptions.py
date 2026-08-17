"""Durable player interruption decisions for Main Core expression batches."""

from __future__ import annotations

import hashlib
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ....storage.sqlite.expression_batch_lifecycle import (
    cancel_pending_expression_row,
    sync_expression_batch_status,
)
from ...group_flow import advance_group_activity_release_boundary
from .expression_interruption_cleanup import (
    is_file_expression,
    restore_cancelled_file_todos,
)
from .expression_interruption_events import (
    append_interruption_event,
    interruption_event_exists,
)
from .expression_outbox import ATTEMPTED_OUTBOX_STATUSES
from .expression_pending_inbound import list_pending_expression_inbound
from .support import (
    OutboxInterruptPolicy,
    OutboxStatus,
    _dt,
    _dump,
    _load,
    _now,
    _parse,
)
from .voice_artifacts import schedule_outbox_voice_artifact_cleanup_sql


class AdvanceHeldGroupActivity:
    def __init__(self, profile_id: str, instance_id: str, now: str) -> None:
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.now = now

    def __call__(self, conn: sqlite3.Connection) -> tuple[int, _InterruptionResult]:
        current = conn.execute(
            """SELECT core.activity_epoch, flow.activity_released_through_message_id
            FROM instance_core_state core
            LEFT JOIN group_flow_instance_state flow
              ON flow.profile_id = core.profile_id AND flow.instance_id = core.instance_id
            WHERE core.profile_id = ? AND core.instance_id = ?""",
            (self.profile_id, self.instance_id),
        ).fetchone()
        if current is None:
            raise KeyError((self.profile_id, self.instance_id))
        released_through = int(current["activity_released_through_message_id"] or 0)
        # Only a window that has passed participation admission may interrupt
        # the previous expression.  COLLECTING/JUDGING messages are the
        # durable judge hold and must never be released by this operation.
        window = conn.execute(
            """SELECT window.last_message_id,
              (SELECT member.message_id FROM group_flow_window_members member
               WHERE member.window_id = window.window_id
                 AND member.message_id > ?
               ORDER BY member.ordinal LIMIT 1) AS first_unreleased_message_id
            FROM group_flow_windows window
            WHERE window.profile_id = ? AND window.instance_id = ?
              AND window.status IN ('READY', 'RUNNING', 'WAITING_FIRST_ATTEMPT')
              AND window.last_message_id > ?
            ORDER BY window.first_message_id LIMIT 1""",
            (
                released_through,
                self.profile_id,
                self.instance_id,
                released_through,
            ),
        ).fetchone()
        if window is None or window["first_unreleased_message_id"] is None:
            return int(current["activity_epoch"]), _InterruptionResult()
        result = _apply_interruption(
            conn,
            profile_id=self.profile_id,
            instance_id=self.instance_id,
            inbound_message_id=int(window["first_unreleased_message_id"]),
            reason="group_first_attempt_released",
            now=self.now,
        )
        self._ensure_epoch_advanced(conn, int(current["activity_epoch"]), result)
        advance_group_activity_release_boundary(
            conn,
            profile_id=self.profile_id,
            instance_id=self.instance_id,
            through_message_id=int(window["last_message_id"]),
            now=self.now,
        )
        epoch = conn.execute(
            """SELECT activity_epoch FROM instance_core_state
            WHERE profile_id = ? AND instance_id = ?""",
            (self.profile_id, self.instance_id),
        ).fetchone()
        return int(epoch["activity_epoch"]), result

    def _ensure_epoch_advanced(
        self, conn: sqlite3.Connection, previous_epoch: int, result: _InterruptionResult
    ) -> None:
        after = conn.execute(
            """SELECT activity_epoch FROM instance_core_state
            WHERE profile_id = ? AND instance_id = ?""",
            (self.profile_id, self.instance_id),
        ).fetchone()
        if int(after["activity_epoch"]) > previous_epoch:
            return
        conn.execute(
            """UPDATE instance_core_state SET activity_epoch = activity_epoch + 1,
            low_frequency_mode = 0, low_frequency_reason = '',
            low_frequency_since = NULL, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?""",
            (self.now, self.profile_id, self.instance_id),
        )
        result.changed = True


INTERRUPT_ALGORITHM_VERSION = "expression-interrupt"
INTERRUPT_RANDOM_WINDOW_SECONDS = 10.0


def expression_interrupt_keep_probability(seconds_until_due: float) -> float:
    delta = max(0.0, float(seconds_until_due))
    if delta > INTERRUPT_RANDOM_WINDOW_SECONDS:
        return 0.0
    return 0.92 * math.exp(-((delta / 3.5) ** 1.35))


def stable_expression_interrupt_roll(
    outbox_id: int,
    inbound_message_id: int,
    *,
    algorithm_version: str = INTERRUPT_ALGORITHM_VERSION,
) -> float:
    """Map stable persisted identities to ``[0, 1)`` without restart re-rolls."""

    material = f"{int(outbox_id)}:{int(inbound_message_id)}:{algorithm_version}".encode()
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return value / float(1 << 64)


@dataclass(slots=True)
class _InterruptionResult:
    cancelled: int = 0
    event_count: int = 0
    processed: int = 0
    changed: bool = False

    def merge(self, other: _InterruptionResult) -> None:
        self.cancelled += other.cancelled
        self.event_count += other.event_count
        self.processed += other.processed
        self.changed = self.changed or other.changed


@dataclass(slots=True)
class _PreparedBatch:
    batch_id: str
    rows: list[sqlite3.Row]
    continuing_reasons: dict[int, str]


def _file_announcement(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> sqlite3.Row | None:
    batch_id = row["expression_batch_id"]
    if batch_id is None:
        return None
    payload = _load(row["payload_json"]) or {}
    announcement_key = str(payload.get("file_announcement_idempotency_key") or "").strip()
    if announcement_key:
        announcement = conn.execute(
            """SELECT * FROM instance_outbox WHERE profile_id = ? AND instance_id = ?
            AND expression_batch_id = ? AND idempotency_key = ?
            AND expression_ordinal < ?""",
            (
                str(row["profile_id"]),
                str(row["instance_id"]),
                str(batch_id),
                announcement_key,
                int(row["expression_ordinal"]),
            ),
        ).fetchone()
        if announcement is not None:
            return announcement
    for candidate in conn.execute(
        """SELECT * FROM instance_outbox WHERE expression_batch_id = ?
        AND expression_ordinal < ? ORDER BY expression_ordinal DESC""",
        (str(batch_id), int(row["expression_ordinal"])),
    ):
        payload = _load(candidate["payload_json"]) or {}
        if str(payload.get("file_followup_idempotency_key") or "") == str(row["idempotency_key"]):
            return candidate
    return None


def file_expression_announcement_attempted(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    include_sending: bool = False,
) -> bool:
    announcement = _file_announcement(conn, row)
    if announcement is None:
        return False
    statuses = set(ATTEMPTED_OUTBOX_STATUSES)
    if include_sending:
        statuses.add(OutboxStatus.SENDING.value)
    return str(announcement["status"]) in statuses


def _runtime_candidate(
    conn: sqlite3.Connection,
    rows: list[sqlite3.Row],
) -> sqlite3.Row | None:
    active = [
        row
        for row in rows
        if str(row["status"]) in {OutboxStatus.PENDING.value, OutboxStatus.SENDING.value}
    ]
    if not active:
        return None
    head = active[0]
    if (
        str(head["status"]) != OutboxStatus.PENDING.value
        or str(head["interrupt_policy"]) != OutboxInterruptPolicy.CANCEL_ON_PLAYER_MESSAGE.value
    ):
        return None
    dependency_key = str(head["depends_on_idempotency_key"] or "").strip()
    if not dependency_key:
        return head
    dependency = conn.execute(
        """SELECT status FROM instance_outbox WHERE profile_id = ? AND instance_id = ?
        AND idempotency_key = ?""",
        (str(head["profile_id"]), str(head["instance_id"]), dependency_key),
    ).fetchone()
    payload = _load(head["payload_json"]) or {}
    if is_file_expression(payload):
        return head if file_expression_announcement_attempted(conn, head) else None
    if dependency is None:
        return head
    return (
        head
        if str(dependency["status"])
        in {
            *ATTEMPTED_OUTBOX_STATUSES,
            OutboxStatus.CANCELLED.value,
            OutboxStatus.FAILED.value,
        }
        else None
    )


def _runtime_evidence(
    row: sqlite3.Row,
    payload: dict[str, Any],
    inbound_message_id: int,
    interrupted_at: datetime,
) -> tuple[dict[str, Any], bool]:
    existing = payload.get("interrupt_runtime")
    if (
        isinstance(existing, dict)
        and int(existing.get("message_id") or 0) == int(inbound_message_id)
        and str(existing.get("algorithm_version") or "") == INTERRUPT_ALGORITHM_VERSION
        and str(existing.get("decision") or "") in {"KEEP", "CANCEL"}
    ):
        return dict(existing), str(existing["decision"]) == "KEEP"
    due_at = _parse(row["not_before_at"]) or interrupted_at
    delta = max(0.0, (due_at - interrupted_at).total_seconds())
    probability = expression_interrupt_keep_probability(delta)
    roll = stable_expression_interrupt_roll(int(row["outbox_id"]), inbound_message_id)
    keep = roll < probability
    evidence = {
        "message_id": int(inbound_message_id),
        "interrupted_at": _dt(interrupted_at),
        "seconds_until_due": delta,
        "keep_probability": probability,
        "roll": roll,
        "decision": "KEEP" if keep else "CANCEL",
        "algorithm_version": INTERRUPT_ALGORITHM_VERSION,
    }
    return evidence, keep


def _protect_started_and_file_items(
    conn: sqlite3.Connection,
    rows: list[sqlite3.Row],
    now: str,
) -> dict[int, str]:
    reasons: dict[int, str] = {}
    for row in rows:
        status = str(row["status"])
        payload = _load(row["payload_json"]) or {}
        reason = ""
        if status == OutboxStatus.SENDING.value:
            reason = "ALREADY_SENDING"
        elif (
            status == OutboxStatus.PENDING.value
            and is_file_expression(payload)
            and file_expression_announcement_attempted(conn, row, include_sending=True)
        ):
            reason = "FILE_ANNOUNCEMENT_STARTED"
        if not reason:
            continue
        conn.execute(
            """UPDATE instance_outbox SET interrupt_policy = ?, updated_at = ?
            WHERE outbox_id = ? AND status IN (?, ?)""",
            (
                OutboxInterruptPolicy.PRESERVE.value,
                now,
                int(row["outbox_id"]),
                OutboxStatus.PENDING.value,
                OutboxStatus.SENDING.value,
            ),
        )
        reasons[int(row["outbox_id"])] = reason
    return reasons


def _rewire_surviving_dependencies(
    conn: sqlite3.Connection,
    batch_id: str,
    now: str,
) -> list[sqlite3.Row]:
    rows = list(
        conn.execute(
            """SELECT * FROM instance_outbox WHERE expression_batch_id = ?
            ORDER BY expression_ordinal""",
            (batch_id,),
        )
    )
    previous_key: str | None = None
    unsafe_files: list[sqlite3.Row] = []
    for row in rows:
        if str(row["status"]) in {
            OutboxStatus.CANCELLED.value,
            OutboxStatus.FAILED.value,
        }:
            continue
        payload = _load(row["payload_json"]) or {}
        if is_file_expression(payload):
            announcement = _file_announcement(conn, row)
            if announcement is None or str(announcement["status"]) in {
                OutboxStatus.CANCELLED.value,
                OutboxStatus.FAILED.value,
            }:
                if str(row["status"]) == OutboxStatus.PENDING.value:
                    unsafe_files.append(row)
                continue
            dependency_key = str(announcement["idempotency_key"])
        else:
            dependency_key = previous_key
        if str(row["status"]) == OutboxStatus.PENDING.value:
            conn.execute(
                """UPDATE instance_outbox SET depends_on_idempotency_key = ?, updated_at = ?
                WHERE outbox_id = ? AND status = ?""",
                (
                    dependency_key,
                    now,
                    int(row["outbox_id"]),
                    OutboxStatus.PENDING.value,
                ),
            )
        previous_key = str(row["idempotency_key"])
    return unsafe_files


def _is_interruptible_pending(row: sqlite3.Row) -> bool:
    return (
        str(row["status"]) == OutboxStatus.PENDING.value
        and str(row["interrupt_policy"]) == OutboxInterruptPolicy.CANCEL_ON_PLAYER_MESSAGE.value
    )


def _refreshed_outbox(conn: sqlite3.Connection, outbox_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM instance_outbox WHERE outbox_id = ?", (int(outbox_id),)
    ).fetchone()
    assert row is not None
    return row


def _apply_runtime_decision(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    inbound_message_id: int,
    interrupted_at: datetime,
    now: str,
) -> bool:
    payload = _load(row["payload_json"]) or {}
    runtime, keep = _runtime_evidence(row, payload, inbound_message_id, interrupted_at)
    payload["interrupt_runtime"] = runtime
    policy = (
        OutboxInterruptPolicy.PRESERVE.value
        if keep
        else OutboxInterruptPolicy.CANCEL_ON_PLAYER_MESSAGE.value
    )
    conn.execute(
        """UPDATE instance_outbox SET payload_json = ?, interrupt_policy = ?, updated_at = ?
        WHERE outbox_id = ? AND status = ?""",
        (
            _dump(payload),
            policy,
            now,
            int(row["outbox_id"]),
            OutboxStatus.PENDING.value,
        ),
    )
    return keep


def _cancel_interruptible_rows(
    conn: sqlite3.Connection,
    rows: list[sqlite3.Row],
    *,
    runtime_candidate_id: int | None,
    inbound_message_id: int,
    interrupted_at: datetime,
    continuing_reasons: dict[int, str],
    reason: str,
    now: str,
) -> list[sqlite3.Row]:
    cancelled_rows: list[sqlite3.Row] = []
    for row in rows:
        if not _is_interruptible_pending(row):
            continue
        outbox_id = int(row["outbox_id"])
        if runtime_candidate_id == outbox_id:
            keep = _apply_runtime_decision(
                conn,
                row,
                inbound_message_id=inbound_message_id,
                interrupted_at=interrupted_at,
                now=now,
            )
            if keep:
                continuing_reasons[outbox_id] = "NEAR_DUE_RUNTIME_KEEP"
                continue
        if cancel_pending_expression_row(conn, row, reason=reason, now=now):
            cancelled_rows.append(_refreshed_outbox(conn, outbox_id))
    return cancelled_rows


def _cancel_unsafe_files(
    conn: sqlite3.Connection,
    batch_id: str,
    *,
    now: str,
) -> list[sqlite3.Row]:
    cancelled: list[sqlite3.Row] = []
    unsafe_files = _rewire_surviving_dependencies(conn, batch_id, now)
    for row in unsafe_files:
        if cancel_pending_expression_row(
            conn, row, reason="cancelled_file_announcement_dependency", now=now
        ):
            cancelled.append(_refreshed_outbox(conn, int(row["outbox_id"])))
    if unsafe_files:
        _rewire_surviving_dependencies(conn, batch_id, now)
    return cancelled


def _apply_interruption_to_batch(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    prepared: _PreparedBatch,
    runtime_candidate_id: int | None,
    inbound_message_id: int,
    interrupted_at: datetime,
    reason: str,
    now: str,
) -> _InterruptionResult:
    cancelled_rows = _cancel_interruptible_rows(
        conn,
        prepared.rows,
        runtime_candidate_id=runtime_candidate_id,
        inbound_message_id=inbound_message_id,
        interrupted_at=interrupted_at,
        continuing_reasons=prepared.continuing_reasons,
        reason=reason,
        now=now,
    )
    cancelled_rows.extend(_cancel_unsafe_files(conn, prepared.batch_id, now=now))
    for row in cancelled_rows:
        schedule_outbox_voice_artifact_cleanup_sql(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            outbox_id=int(row["outbox_id"]),
            reason="voice_expression_interrupted",
            now=now,
        )
    restore_cancelled_file_todos(
        conn,
        profile_id,
        instance_id,
        cancelled_rows,
        now,
        reason,
        load_payload=_load,
    )
    sync_expression_batch_status(conn, prepared.batch_id, now)
    event_added = append_interruption_event(
        conn,
        profile_id=profile_id,
        instance_id=instance_id,
        batch_id=prepared.batch_id,
        inbound_message_id=inbound_message_id,
        continuing_reasons=prepared.continuing_reasons,
        cancelled_rows=cancelled_rows,
        now=now,
    )
    return _InterruptionResult(
        cancelled=len(cancelled_rows),
        event_count=int(event_added),
        changed=bool(cancelled_rows or prepared.continuing_reasons or event_added),
    )


def _active_batch_ids(conn: sqlite3.Connection, profile_id: str, instance_id: str) -> list[str]:
    return [
        str(row["batch_id"])
        for row in conn.execute(
            """SELECT batch_id FROM instance_expression_batches
            WHERE profile_id = ? AND instance_id = ? AND status = 'ACTIVE'
            ORDER BY created_at, batch_id""",
            (profile_id, instance_id),
        )
    ]


def _prepare_batches(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    inbound_message_id: int,
    now: str,
) -> list[_PreparedBatch]:
    prepared: list[_PreparedBatch] = []
    for batch_id in _active_batch_ids(conn, profile_id, instance_id):
        if interruption_event_exists(conn, profile_id, instance_id, batch_id, inbound_message_id):
            continue
        rows = list(
            conn.execute(
                """SELECT * FROM instance_outbox WHERE profile_id = ? AND instance_id = ?
                AND expression_batch_id = ? ORDER BY expression_ordinal""",
                (profile_id, instance_id, batch_id),
            )
        )
        if not rows:
            continue
        reasons = _protect_started_and_file_items(conn, rows, now)
        refreshed = list(
            conn.execute(
                """SELECT * FROM instance_outbox WHERE expression_batch_id = ?
                ORDER BY expression_ordinal""",
                (batch_id,),
            )
        )
        prepared.append(_PreparedBatch(batch_id, refreshed, reasons))
    return prepared


def _select_runtime_candidate(
    conn: sqlite3.Connection, prepared: list[_PreparedBatch]
) -> int | None:
    candidates = [
        candidate
        for batch in prepared
        if (candidate := _runtime_candidate(conn, batch.rows)) is not None
    ]
    if not candidates:
        return None
    row = min(
        candidates,
        key=lambda item: (
            str(item["not_before_at"] or item["created_at"]),
            str(item["expression_batch_id"]),
            int(item["expression_ordinal"]),
        ),
    )
    return int(row["outbox_id"])


def _ensure_activity_fence(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    now: str,
) -> bool:
    row = conn.execute(
        """SELECT MAX(activity_epoch) AS activity_epoch
        FROM instance_expression_batches
        WHERE profile_id = ? AND instance_id = ? AND status = 'ACTIVE'""",
        (profile_id, instance_id),
    ).fetchone()
    if row is None or row["activity_epoch"] is None:
        return False
    cursor = conn.execute(
        """UPDATE instance_core_state SET activity_epoch = ?, low_frequency_mode = 0,
        low_frequency_reason = '', low_frequency_since = NULL, updated_at = ?
        WHERE profile_id = ? AND instance_id = ? AND activity_epoch < ?""",
        (
            int(row["activity_epoch"]) + 1,
            now,
            profile_id,
            instance_id,
            int(row["activity_epoch"]) + 1,
        ),
    )
    return cursor.rowcount == 1


def _apply_interruption(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    inbound_message_id: int,
    reason: str,
    now: str,
) -> _InterruptionResult:
    inbound = conn.execute(
        """SELECT message_id, occurred_at, created_at FROM instance_messages
        WHERE profile_id = ? AND instance_id = ? AND message_id = ?
        AND direction = 'INBOUND' AND role = 'user'""",
        (profile_id, instance_id, int(inbound_message_id)),
    ).fetchone()
    if inbound is None:
        raise ValueError("interrupting player message does not belong to the expression instance")
    interrupted_at = _parse(inbound["occurred_at"]) or _parse(inbound["created_at"]) or _now()
    fenced = _ensure_activity_fence(conn, profile_id=profile_id, instance_id=instance_id, now=now)
    prepared = _prepare_batches(
        conn,
        profile_id=profile_id,
        instance_id=instance_id,
        inbound_message_id=inbound_message_id,
        now=now,
    )
    runtime_candidate_id = _select_runtime_candidate(conn, prepared)
    result = _InterruptionResult(changed=fenced)
    for batch in prepared:
        result.merge(
            _apply_interruption_to_batch(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                prepared=batch,
                runtime_candidate_id=runtime_candidate_id,
                inbound_message_id=int(inbound_message_id),
                interrupted_at=interrupted_at,
                reason=reason,
                now=now,
            )
        )
    return result


class _CancelExpressionInterruption:
    def __init__(
        self,
        *,
        profile_id: str,
        instance_id: str,
        reason: str,
        inbound_message_id: int,
        now: str,
    ) -> None:
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.reason = reason
        self.inbound_message_id = int(inbound_message_id)
        self.now = now

    def __call__(self, conn: sqlite3.Connection) -> _InterruptionResult:
        return _apply_interruption(
            conn,
            profile_id=self.profile_id,
            instance_id=self.instance_id,
            inbound_message_id=self.inbound_message_id,
            reason=self.reason,
            now=self.now,
        )


class _AdvanceActivityAndInterrupt:
    def __init__(
        self, profile_id: str, instance_id: str, inbound_message_id: int, now: str
    ) -> None:
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.inbound_message_id = int(inbound_message_id)
        self.now = now

    def __call__(self, conn: sqlite3.Connection) -> tuple[int, _InterruptionResult]:
        cursor = conn.execute(
            """UPDATE instance_core_state SET activity_epoch = activity_epoch + 1,
            low_frequency_mode = 0, low_frequency_reason = '', low_frequency_since = NULL,
            updated_at = ? WHERE profile_id = ? AND instance_id = ?""",
            (self.now, self.profile_id, self.instance_id),
        )
        if cursor.rowcount != 1:
            raise KeyError((self.profile_id, self.instance_id))
        result = _apply_interruption(
            conn,
            profile_id=self.profile_id,
            instance_id=self.instance_id,
            inbound_message_id=self.inbound_message_id,
            reason="superseded_by_new_inbound_activity",
            now=self.now,
        )
        epoch = int(
            conn.execute(
                """SELECT activity_epoch FROM instance_core_state
                WHERE profile_id = ? AND instance_id = ?""",
                (self.profile_id, self.instance_id),
            ).fetchone()[0]
        )
        return epoch, result


class _RecoverPendingInterruptions:
    def __init__(self, *, limit: int, now: str) -> None:
        self.limit = max(1, min(int(limit), 1000))
        self.now = now

    def __call__(self, conn: sqlite3.Connection) -> _InterruptionResult:
        messages = list_pending_expression_inbound(conn, limit=self.limit)
        result = _InterruptionResult()
        for row in messages:
            result.processed += 1
            result.merge(
                _apply_interruption(
                    conn,
                    profile_id=str(row["profile_id"]),
                    instance_id=str(row["instance_id"]),
                    inbound_message_id=int(row["message_id"]),
                    reason="recovered_pending_player_interruption",
                    now=self.now,
                )
            )
        return result


class ExpressionInterruptionRecords:
    async def advance_activity_and_interrupt_expressions(
        self, profile_id: str, instance_id: str, inbound_message_id: int
    ) -> int:
        epoch, result = await self.uow.run(
            _AdvanceActivityAndInterrupt(profile_id, instance_id, inbound_message_id, _dt(_now()))
        )
        if result.changed:
            await self.publish_context_backup()
        return int(epoch)

    async def advance_group_held_activity(self, profile_id: str, instance_id: str) -> int:
        epoch, result = await self.uow.run(
            AdvanceHeldGroupActivity(profile_id, instance_id, _dt(_now()))
        )
        if result.changed:
            await self.publish_context_backup()
        return int(epoch)

    async def cancel_unstarted_expression_items(
        self,
        profile_id: str,
        instance_id: str,
        *,
        reason: str = "superseded_by_player_message",
        interrupted_by_message_id: int | None = None,
    ) -> int:
        if interrupted_by_message_id is None:
            raise ValueError("expression interruption requires the inbound ledger message id")
        operation = _CancelExpressionInterruption(
            profile_id=profile_id,
            instance_id=instance_id,
            reason=str(reason),
            inbound_message_id=int(interrupted_by_message_id),
            now=_dt(_now()),
        )
        result = await self.uow.run(operation)
        if result.changed:
            await self.publish_context_backup()
        return int(result.cancelled)

    async def recover_pending_expression_interruptions(self, *, limit: int = 100) -> int:
        result = await self.uow.run(_RecoverPendingInterruptions(limit=limit, now=_dt(_now())))
        if result.changed:
            await self.publish_context_backup()
        return int(result.processed)

    async def get_expression_foreground_barrier(
        self,
        profile_id: str,
        instance_id: str,
        activity_epoch: int,
    ) -> dict[str, Any]:
        rows = await self.db.fetch_all(
            """SELECT outbox.outbox_id, outbox.status, outbox.not_before_at,
            batch.batch_id FROM instance_outbox outbox
            JOIN instance_expression_batches batch
              ON batch.batch_id = outbox.expression_batch_id
            WHERE outbox.profile_id = ? AND outbox.instance_id = ?
              AND batch.status = 'ACTIVE' AND batch.activity_epoch < ?
              AND outbox.status IN ('PENDING', 'SENDING')
              AND outbox.interrupt_policy = 'PRESERVE'
            ORDER BY outbox.expression_ordinal""",
            (profile_id, instance_id, int(activity_epoch)),
        )
        now = _now()
        next_checks = []
        for row in rows:
            if str(row["status"]) == OutboxStatus.SENDING.value:
                next_checks.append(now + timedelta(seconds=1))
            else:
                next_checks.append(max(now, _parse(row["not_before_at"]) or now))
        return {
            "blocked": bool(rows),
            "activity_epoch": int(activity_epoch),
            "batch_ids": list(dict.fromkeys(str(row["batch_id"]) for row in rows)),
            "outbox_ids": [int(row["outbox_id"]) for row in rows],
            "pending_outbox_ids": [
                int(row["outbox_id"])
                for row in rows
                if str(row["status"]) == OutboxStatus.PENDING.value
            ],
            "sending_outbox_ids": [
                int(row["outbox_id"])
                for row in rows
                if str(row["status"]) == OutboxStatus.SENDING.value
            ],
            "next_check_at": min(next_checks) if next_checks else None,
        }

    async def allows_preserved_expression_dispatch(
        self,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
    ) -> bool:
        row = await self.db.fetch_one(
            """SELECT 1 FROM instance_outbox outbox
            JOIN instance_expression_batches batch
              ON batch.batch_id = outbox.expression_batch_id
            WHERE outbox.profile_id = ? AND outbox.instance_id = ?
              AND outbox.outbox_id = ? AND batch.status = 'ACTIVE'
              AND outbox.status IN ('PENDING', 'SENDING')
              AND outbox.interrupt_policy = 'PRESERVE'""",
            (profile_id, instance_id, int(outbox_id)),
        )
        return row is not None


__all__ = [
    "ExpressionInterruptionRecords",
    "INTERRUPT_ALGORITHM_VERSION",
    "expression_interrupt_keep_probability",
    "file_expression_announcement_attempted",
    "stable_expression_interrupt_roll",
]
