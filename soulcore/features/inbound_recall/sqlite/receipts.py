"""Durable OneBot recall receipt operations."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from ....storage.sqlite.codec import _parse
from ....storage.sqlite.repository import SqliteRepository
from ..domain import InboundRecallHold, InboundRecallTarget, OneBotRecallNotice
from .matching import begin_notice as match_notice
from .records import aware_datetime, hold_from_row, notice_from_row, required_datetime


class InboundRecallReceiptRepository(SqliteRepository):
    async def begin_notice(
        self,
        *,
        profile_id: str,
        instance_id: str,
        platform_instance_id: str,
        route_umo: str,
        notice: OneBotRecallNotice,
    ) -> InboundRecallTarget | None:
        return await self.uow.run(
            lambda conn: match_notice(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                platform_instance_id=platform_instance_id,
                route_umo=route_umo,
                notice=notice,
                insert=True,
            )
        )

    async def claim_unmatched_for_hold(
        self,
        hold: InboundRecallHold,
    ) -> InboundRecallTarget | None:
        def operation(conn: sqlite3.Connection) -> InboundRecallTarget | None:
            row = conn.execute(
                """SELECT * FROM inbound_recall_receipts
                WHERE profile_id = ? AND instance_id = ? AND platform_instance_id = ?
                  AND route_umo = ? AND platform_message_id = ? AND status = 'UNMATCHED'
                ORDER BY received_at LIMIT 1""",
                (
                    hold.profile_id,
                    hold.instance_id,
                    hold.platform_instance_id,
                    hold.route_umo,
                    hold.platform_message_id,
                ),
            ).fetchone()
            if row is None:
                return None
            return match_notice(
                conn,
                profile_id=hold.profile_id,
                instance_id=hold.instance_id,
                platform_instance_id=hold.platform_instance_id,
                route_umo=hold.route_umo,
                notice=notice_from_row(row),
                insert=False,
            )

        return await self.uow.run(operation)

    async def claim_matching_receipts(
        self,
        *,
        limit: int = 20,
    ) -> tuple[InboundRecallTarget, ...]:
        def operation(conn: sqlite3.Connection) -> list[InboundRecallTarget]:
            rows = list(
                conn.execute(
                    """SELECT receipt.* FROM inbound_recall_receipts receipt
                    JOIN inbound_message_recall_states state
                      ON state.profile_id = receipt.profile_id
                     AND state.instance_id = receipt.instance_id
                     AND state.platform_instance_id = receipt.platform_instance_id
                     AND state.route_umo = receipt.route_umo
                     AND state.platform_message_id = receipt.platform_message_id
                    WHERE receipt.status = 'UNMATCHED' AND state.status != 'RECALLED'
                    ORDER BY receipt.received_at LIMIT ?""",
                    (max(1, min(int(limit), 100)),),
                )
            )
            targets: list[InboundRecallTarget] = []
            for row in rows:
                target = match_notice(
                    conn,
                    profile_id=str(row["profile_id"]),
                    instance_id=str(row["instance_id"]),
                    platform_instance_id=str(row["platform_instance_id"]),
                    route_umo=str(row["route_umo"]),
                    notice=notice_from_row(row),
                    insert=False,
                )
                if target is not None:
                    targets.append(target)
            return targets

        return tuple(await self.uow.run(operation))

    async def get_processing_target(self, receipt_id: str) -> InboundRecallTarget | None:
        row = await self.db.fetch_one(
            """SELECT state.*,
                receipt.receipt_id AS target_receipt_id,
                receipt.notice_type AS target_notice_type,
                receipt.sender_id AS target_sender_id,
                receipt.operator_id AS target_operator_id,
                receipt.received_at AS target_recall_received_at,
                receipt.platform_occurred_at AS target_platform_occurred_at
            FROM inbound_recall_receipts receipt
            JOIN inbound_message_recall_states state
              ON state.profile_id = receipt.profile_id
             AND state.instance_id = receipt.instance_id
             AND state.ledger_message_id = receipt.matched_ledger_message_id
            WHERE receipt.receipt_id = ? AND receipt.status = 'PROCESSING'""",
            (str(receipt_id),),
        )
        if row is None:
            return None
        return InboundRecallTarget(
            str(row["target_receipt_id"]),
            OneBotRecallNotice(
                str(row["target_notice_type"]),
                str(row["platform_message_id"]),
                str(row["target_sender_id"]),
                str(row["target_operator_id"]),
                aware_datetime(row["target_recall_received_at"]),
                _parse(row["target_platform_occurred_at"]),
            ),
            hold_from_row(row),
        )

    async def retry_receipt(self, receipt_id: str, *, now: datetime) -> None:
        await self.db.call(
            lambda conn: conn.execute(
                """UPDATE inbound_recall_receipts SET status = 'UNMATCHED',
                matched_ledger_message_id = NULL, updated_at = ?
                WHERE receipt_id = ? AND status = 'PROCESSING'""",
                (required_datetime(now), str(receipt_id)),
            ),
            transaction=True,
        )

    async def cleanup_expired_receipts(self, *, now: datetime) -> int:
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """DELETE FROM inbound_recall_receipts
                WHERE status IN ('UNMATCHED', 'IGNORED') AND expires_at <= ?""",
                (required_datetime(now),),
            ),
            transaction=True,
        )
        return int(cursor.rowcount)


__all__ = ["InboundRecallReceiptRepository"]
