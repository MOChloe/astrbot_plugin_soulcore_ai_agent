"""Durable grace-state operations for recall-capable inbound messages."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from ....contracts.inbound_recall import InboundRecallCommitFence
from ....storage.sqlite.background_projection import project_foreground_message_continuity_sql
from ....storage.sqlite.codec import _parse
from ....storage.sqlite.dialogue_turns import context_eligible_sql
from ....storage.sqlite.inbound_admission_transactions import InboundAdmissionTransactions
from ....storage.sqlite.repository import SqliteRepository
from ..algorithm import INBOUND_RECALL_ALGORITHM_VERSION, INBOUND_RECALL_GRACE_SECONDS
from ..domain import InboundRecallHold
from .orphan_recovery import RecoverInboundRecallOrphans
from .records import hold_from_row, normalize_scope, required_datetime


class InboundRecallGraceRepository(SqliteRepository):
    _inbound_admission: InboundAdmissionTransactions

    async def list_release_commit_fences(
        self,
        profile_id: str,
        instance_id: str,
        message_ids: list[int] | tuple[int, ...],
        *,
        activity_epoch: int,
    ) -> tuple[InboundRecallCommitFence, ...]:
        ids = tuple(dict.fromkeys(int(value) for value in message_ids if int(value) > 0))
        if not ids:
            return ()
        placeholders = ",".join("?" for _ in ids)
        rows = await self.db.fetch_all(
            f"""SELECT ledger_message_id, lease_token
            FROM inbound_message_recall_states
            WHERE profile_id = ? AND instance_id = ?
              AND status = 'RELEASED'
              AND ledger_message_id IN ({placeholders})
            ORDER BY ledger_message_id""",
            (profile_id, instance_id, *ids),
        )
        return tuple(
            InboundRecallCommitFence(
                ledger_message_id=int(row["ledger_message_id"]),
                lease_token=int(row["lease_token"]),
                activity_epoch=max(0, int(activity_epoch)),
            )
            for row in rows
        )

    async def mark_release_handoffs_dispatched(
        self,
        profile_id: str,
        instance_id: str,
        fences: tuple[InboundRecallCommitFence, ...],
        *,
        now: datetime,
    ) -> bool:
        if not fences:
            return True
        now_text = required_datetime(now)

        def operation(conn: sqlite3.Connection) -> bool:
            pending: list[InboundRecallCommitFence] = []
            for fence in fences:
                row = conn.execute(
                    """SELECT status, activity_epoch FROM inbound_message_recall_states
                    WHERE profile_id = ? AND instance_id = ?
                      AND ledger_message_id = ? AND lease_token = ?""",
                    (
                        profile_id,
                        instance_id,
                        fence.ledger_message_id,
                        fence.lease_token,
                    ),
                ).fetchone()
                if row is None:
                    return False
                status = str(row["status"])
                if status == "DISPATCHED":
                    if int(row["activity_epoch"] or 0) != fence.activity_epoch:
                        return False
                    continue
                if status != "RELEASED":
                    return False
                pending.append(fence)
            for fence in pending:
                changed = conn.execute(
                    """UPDATE inbound_message_recall_states
                    SET status = 'DISPATCHED', activity_epoch = ?,
                        dispatched_at = ?, updated_at = ?
                    WHERE profile_id = ? AND instance_id = ?
                      AND ledger_message_id = ? AND status = 'RELEASED'
                      AND lease_token = ?""",
                    (
                        fence.activity_epoch,
                        now_text,
                        now_text,
                        profile_id,
                        instance_id,
                        fence.ledger_message_id,
                        fence.lease_token,
                    ),
                ).rowcount
                if changed != 1:
                    raise RuntimeError("inbound-recall ownership changed during durable handoff")
            return True

        return bool(await self.uow.run(operation))

    async def messages_are_model_visible(
        self,
        profile_id: str,
        instance_id: str,
        message_ids: list[int] | tuple[int, ...],
    ) -> bool:
        ids = tuple(dict.fromkeys(int(value) for value in message_ids if int(value) > 0))
        if not ids:
            return True
        placeholders = ",".join("?" for _ in ids)
        row = await self.db.fetch_one(
            f"""SELECT COUNT(*) AS total,
                SUM(CASE WHEN delivery_status = 'RECEIVED' THEN 1 ELSE 0 END) AS visible
            FROM instance_messages
            WHERE profile_id = ? AND instance_id = ?
              AND message_id IN ({placeholders})""",
            (profile_id, instance_id, *ids),
        )
        return bool(
            row is not None
            and int(row["total"] or 0) == len(ids)
            and int(row["visible"] or 0) == len(ids)
        )

    async def register_hold(
        self,
        *,
        profile_id: str,
        instance_id: str,
        ledger_message_id: int,
        platform_instance_id: str,
        route_umo: str,
        platform_message_id: str,
        scope: str,
        direct_address: bool,
        received_at: datetime,
        original_plain_text: str,
        original_components: list[dict[str, Any]],
        lease_owner: str,
        lease_token: int,
    ) -> InboundRecallHold | None:
        now_text = required_datetime(received_at)
        grace_text = required_datetime(
            received_at + timedelta(seconds=INBOUND_RECALL_GRACE_SECONDS)
        )
        components_json = json.dumps(
            original_components,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        def operation(conn: sqlite3.Connection) -> sqlite3.Row | None:
            if not self._inbound_admission.complete(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                message_id=int(ledger_message_id),
                lease_owner=lease_owner,
                lease_token=int(lease_token),
                now=datetime.now(UTC),
                status="HANDED_OFF",
            ):
                return None
            previous = conn.execute(
                f"""SELECT occurred_at FROM instance_messages
                WHERE profile_id = ? AND instance_id = ? AND message_id <> ?
                  AND occurred_at <= ? AND {context_eligible_sql()}
                  AND (direction = 'OUTBOUND' OR role = 'user')
                ORDER BY occurred_at DESC, message_id DESC LIMIT 1""",
                (profile_id, instance_id, int(ledger_message_id), now_text),
            ).fetchone()
            conn.execute(
                """INSERT INTO inbound_message_recall_states(
                    profile_id, instance_id, ledger_message_id, platform_instance_id,
                    route_umo, platform_message_id, scope, direct_address,
                    received_at, grace_until, previous_activity_at, status,
                    algorithm_version, original_plain_text, original_components_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'HELD', ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id, instance_id, ledger_message_id) DO NOTHING""",
                (
                    profile_id,
                    instance_id,
                    int(ledger_message_id),
                    platform_instance_id,
                    route_umo,
                    platform_message_id,
                    normalize_scope(scope),
                    int(bool(direct_address)),
                    now_text,
                    grace_text,
                    str(previous["occurred_at"]) if previous is not None else None,
                    INBOUND_RECALL_ALGORITHM_VERSION,
                    str(original_plain_text or ""),
                    components_json,
                    now_text,
                    now_text,
                ),
            )
            row = conn.execute(
                """SELECT * FROM inbound_message_recall_states
                WHERE profile_id = ? AND instance_id = ? AND ledger_message_id = ?""",
                (profile_id, instance_id, int(ledger_message_id)),
            ).fetchone()
            assert row is not None
            return row

        row = await self.uow.run(operation)
        return hold_from_row(row) if row is not None else None

    async def claim_due(
        self,
        *,
        now: datetime,
        worker_id: str,
        limit: int = 20,
        lease_seconds: int = 60,
    ) -> tuple[InboundRecallHold, ...]:
        now_text = required_datetime(now)
        lease_until = required_datetime(now + timedelta(seconds=max(5, int(lease_seconds))))
        owner = str(worker_id).strip()
        if not owner:
            raise ValueError("inbound recall worker_id cannot be empty")

        def operation(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            rows = list(
                conn.execute(
                    """SELECT * FROM inbound_message_recall_states state
                    WHERE state.status = 'HELD' AND state.grace_until <= ?
                      AND NOT EXISTS (
                        SELECT 1 FROM inbound_recall_receipts receipt
                        WHERE receipt.profile_id = state.profile_id
                          AND receipt.instance_id = state.instance_id
                          AND receipt.platform_instance_id = state.platform_instance_id
                          AND receipt.route_umo = state.route_umo
                          AND receipt.platform_message_id = state.platform_message_id
                          AND receipt.status IN ('UNMATCHED', 'PROCESSING')
                      )
                    ORDER BY state.grace_until, state.ledger_message_id LIMIT ?""",
                    (now_text, max(1, min(int(limit), 100))),
                )
            )
            claimed: list[sqlite3.Row] = []
            for row in rows:
                cursor = conn.execute(
                    """UPDATE inbound_message_recall_states SET status = 'CLAIMED',
                    lease_owner = ?, lease_until = ?, lease_token = lease_token + 1,
                    updated_at = ? WHERE profile_id = ? AND instance_id = ?
                    AND ledger_message_id = ? AND status = 'HELD'""",
                    (
                        owner,
                        lease_until,
                        now_text,
                        row["profile_id"],
                        row["instance_id"],
                        row["ledger_message_id"],
                    ),
                )
                if cursor.rowcount:
                    current = conn.execute(
                        """SELECT * FROM inbound_message_recall_states WHERE profile_id = ?
                        AND instance_id = ? AND ledger_message_id = ?""",
                        (row["profile_id"], row["instance_id"], row["ledger_message_id"]),
                    ).fetchone()
                    assert current is not None
                    claimed.append(current)
            return claimed

        return tuple(hold_from_row(row) for row in await self.uow.run(operation))

    async def release_claim(
        self,
        hold: InboundRecallHold,
        *,
        knowledge_reason: str,
        now: datetime,
    ) -> bool:
        now_text = required_datetime(now)

        def operation(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                """UPDATE inbound_message_recall_states SET status = 'RELEASED',
                lease_owner = NULL, lease_until = NULL, updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND ledger_message_id = ?
                  AND status = 'CLAIMED' AND lease_token = ?""",
                (
                    now_text,
                    hold.profile_id,
                    hold.instance_id,
                    hold.ledger_message_id,
                    hold.lease_token,
                ),
            )
            if cursor.rowcount != 1:
                return False
            changed = conn.execute(
                """UPDATE instance_messages SET delivery_status = 'RECEIVED',
                knowledge_eligibility = 'HELD', knowledge_eligibility_reason = ?
                WHERE profile_id = ? AND instance_id = ? AND message_id = ?
                  AND delivery_status = 'PENDING_RECALL_GRACE'""",
                (
                    str(knowledge_reason),
                    hold.profile_id,
                    hold.instance_id,
                    hold.ledger_message_id,
                ),
            )
            if changed.rowcount != 1:
                return False
            row = conn.execute(
                """SELECT * FROM instance_messages
                WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
                (hold.profile_id, hold.instance_id, hold.ledger_message_id),
            ).fetchone()
            if row is not None:
                project_foreground_message_continuity_sql(conn, row, settled_at=now_text)
            return True

        return bool(await self.uow.run(operation))

    async def defer_claim_for_shutdown(
        self,
        hold: InboundRecallHold,
        *,
        now: datetime,
    ) -> bool:
        now_text = required_datetime(now)
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE inbound_message_recall_states SET status = 'HELD',
                grace_until = ?, lease_owner = NULL, lease_until = NULL,
                last_error = 'dispatch_cancelled_for_shutdown', updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND ledger_message_id = ?
                  AND status = 'CLAIMED' AND lease_token = ?""",
                (
                    now_text,
                    now_text,
                    hold.profile_id,
                    hold.instance_id,
                    hold.ledger_message_id,
                    hold.lease_token,
                ),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1

    async def mark_dispatched(
        self,
        hold: InboundRecallHold,
        *,
        activity_epoch: int,
        now: datetime,
    ) -> bool:
        now_text = required_datetime(now)
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE inbound_message_recall_states SET status = 'DISPATCHED',
                activity_epoch = ?, dispatched_at = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND ledger_message_id = ?
                  AND status = 'RELEASED' AND lease_token = ?""",
                (
                    max(0, int(activity_epoch)),
                    now_text,
                    now_text,
                    hold.profile_id,
                    hold.instance_id,
                    hold.ledger_message_id,
                    hold.lease_token,
                ),
            ),
            transaction=True,
        )
        if cursor.rowcount == 1:
            return True
        row = await self.db.fetch_one(
            """SELECT 1 FROM inbound_message_recall_states
            WHERE profile_id = ? AND instance_id = ? AND ledger_message_id = ?
              AND status = 'DISPATCHED' AND lease_token = ? AND activity_epoch = ?""",
            (
                hold.profile_id,
                hold.instance_id,
                hold.ledger_message_id,
                hold.lease_token,
                max(0, int(activity_epoch)),
            ),
        )
        return row is not None

    async def retry_claim(
        self,
        hold: InboundRecallHold,
        *,
        retry_at: datetime,
        error: str,
    ) -> None:
        retry_text = required_datetime(retry_at)

        def operation(conn: sqlite3.Connection) -> None:
            changed = conn.execute(
                """UPDATE inbound_message_recall_states SET status = 'HELD',
                grace_until = ?, lease_owner = NULL, lease_until = NULL,
                last_error = ?, updated_at = ? WHERE profile_id = ? AND instance_id = ?
                  AND ledger_message_id = ? AND status IN ('CLAIMED', 'RELEASED')
                  AND lease_token = ?""",
                (
                    retry_text,
                    str(error or "")[:200],
                    retry_text,
                    hold.profile_id,
                    hold.instance_id,
                    hold.ledger_message_id,
                    hold.lease_token,
                ),
            ).rowcount
            if changed != 1:
                return
            conn.execute(
                """UPDATE instance_messages SET delivery_status = 'PENDING_RECALL_GRACE',
                knowledge_eligibility = 'HELD',
                knowledge_eligibility_reason = 'inbound_recall_grace'
                WHERE profile_id = ? AND instance_id = ? AND message_id = ?
                  AND delivery_status = 'RECEIVED'""",
                (hold.profile_id, hold.instance_id, hold.ledger_message_id),
            )

        await self.uow.run(operation)

    async def recover(self, *, now: datetime) -> int:
        now_text = required_datetime(now)
        stale_receipt_text = required_datetime(now - timedelta(seconds=90))

        def operation(conn: sqlite3.Connection) -> int:
            orphaned = RecoverInboundRecallOrphans(
                inbound_admission=self._inbound_admission,
                now=now,
                now_text=now_text,
            )(conn)
            rows = list(
                conn.execute(
                    """SELECT profile_id, instance_id, ledger_message_id,
                    status, lease_token
                    FROM inbound_message_recall_states
                    WHERE (status = 'CLAIMED' AND lease_until <= ?)
                       OR status = 'RELEASED'""",
                    (now_text,),
                )
            )
            recovered = 0
            for row in rows:
                if str(row["status"]) == "CLAIMED":
                    state_changed = conn.execute(
                        """UPDATE inbound_message_recall_states SET status = 'HELD',
                        grace_until = MIN(grace_until, ?), lease_owner = NULL,
                        lease_until = NULL, last_error = 'release_lease_expired',
                        updated_at = ? WHERE profile_id = ? AND instance_id = ?
                          AND ledger_message_id = ? AND status = 'CLAIMED'
                          AND lease_token = ? AND lease_until <= ?""",
                        (
                            now_text,
                            now_text,
                            row["profile_id"],
                            row["instance_id"],
                            row["ledger_message_id"],
                            row["lease_token"],
                            now_text,
                        ),
                    ).rowcount
                else:
                    state_changed = conn.execute(
                        """UPDATE inbound_message_recall_states SET status = 'HELD',
                        grace_until = MIN(grace_until, ?), lease_owner = NULL,
                        lease_until = NULL, last_error = 'release_lease_expired',
                        updated_at = ? WHERE profile_id = ? AND instance_id = ?
                          AND ledger_message_id = ? AND status = 'RELEASED'
                          AND lease_token = ?""",
                        (
                            now_text,
                            now_text,
                            row["profile_id"],
                            row["instance_id"],
                            row["ledger_message_id"],
                            row["lease_token"],
                        ),
                    ).rowcount
                if state_changed != 1:
                    continue
                conn.execute(
                    """UPDATE instance_messages SET delivery_status = 'PENDING_RECALL_GRACE',
                    knowledge_eligibility = 'HELD',
                    knowledge_eligibility_reason = 'inbound_recall_grace'
                    WHERE profile_id = ? AND instance_id = ? AND message_id = ?
                      AND delivery_status = 'RECEIVED'""",
                    (row["profile_id"], row["instance_id"], row["ledger_message_id"]),
                )
                recovered += 1
            repaired = self._repair_committed_dispatch_messages(conn, now_text)
            receipts = conn.execute(
                """UPDATE inbound_recall_receipts SET status = 'UNMATCHED',
                matched_ledger_message_id = NULL, updated_at = ?
                WHERE status = 'PROCESSING' AND updated_at <= ?""",
                (now_text, stale_receipt_text),
            ).rowcount
            return int(orphaned) + int(recovered) + int(repaired) + int(receipts)

        return int(await self.uow.run(operation))

    @staticmethod
    def _repair_committed_dispatch_messages(conn: sqlite3.Connection, now_text: str) -> int:
        """Repair only terminal handoffs proven to have reached a successful Core commit."""

        scopes = list(
            conn.execute(
                """SELECT DISTINCT message.profile_id, message.instance_id
                FROM instance_messages message
                JOIN inbound_message_recall_states state
                  ON state.profile_id = message.profile_id
                 AND state.instance_id = message.instance_id
                 AND state.ledger_message_id = message.message_id
                WHERE state.status = 'DISPATCHED'
                  AND state.committed_full_at IS NOT NULL
                  AND message.delivery_status = 'PENDING_RECALL_GRACE'
                  AND message.knowledge_eligibility = 'HELD'
                  AND message.knowledge_eligibility_reason = 'inbound_recall_grace'"""
            )
        )
        repaired = conn.execute(
            """UPDATE instance_messages AS message
            SET delivery_status = 'RECEIVED', knowledge_eligibility = 'ELIGIBLE',
                knowledge_eligibility_reason = ''
            WHERE delivery_status = 'PENDING_RECALL_GRACE'
              AND knowledge_eligibility = 'HELD'
              AND knowledge_eligibility_reason = 'inbound_recall_grace'
              AND EXISTS (
                SELECT 1 FROM inbound_message_recall_states state
                WHERE state.profile_id = message.profile_id
                  AND state.instance_id = message.instance_id
                  AND state.ledger_message_id = message.message_id
                  AND state.status = 'DISPATCHED'
                  AND state.committed_full_at IS NOT NULL
              )"""
        ).rowcount
        if repaired:
            for scope in scopes:
                conn.execute(
                    """UPDATE knowledge_processing_state SET
                    processing_version = processing_version + 1, updated_at = ?
                    WHERE profile_id = ? AND instance_id = ?""",
                    (now_text, scope["profile_id"], scope["instance_id"]),
                )
        return int(repaired)

    async def next_due_at(self) -> datetime | None:
        row = await self.db.fetch_one(
            """SELECT MIN(grace_until) AS due_at FROM inbound_message_recall_states
            WHERE status = 'HELD'"""
        )
        if row is None or not row["due_at"]:
            return None
        return _parse(row["due_at"])


__all__ = ["InboundRecallGraceRepository"]
