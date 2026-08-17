"""Composed SQLite repository for inbound recall state and settlement."""

from __future__ import annotations

from datetime import datetime

from ....storage.sqlite.inbound_admission_transactions import InboundAdmissionTransactions
from ....storage.sqlite.recall_file_transactions import RecallFileTransactions
from ....storage.sqlite.repository import SqliteRepository
from ..domain import InboundRecallDecision, InboundRecallSettlement, InboundRecallTarget
from .grace import InboundRecallGraceRepository
from .receipts import InboundRecallReceiptRepository
from .records import required_datetime
from .settlement import finalize_notice_transaction


class SqliteInboundRecallRepository(
    InboundRecallGraceRepository,
    InboundRecallReceiptRepository,
):
    def __init__(
        self,
        engine,
        inbound_admission: InboundAdmissionTransactions,
        file_transactions: RecallFileTransactions,
    ) -> None:
        SqliteRepository.__init__(self, engine)
        self._inbound_admission = inbound_admission
        self._file_transactions = file_transactions

    async def finalize_notice(
        self,
        target: InboundRecallTarget,
        decision: InboundRecallDecision,
        *,
        event_text: str,
        now: datetime,
    ) -> InboundRecallSettlement | None:
        now_text = required_datetime(now)
        return await self.uow.run(
            lambda conn: finalize_notice_transaction(
                conn,
                target,
                decision,
                event_text=event_text,
                now_text=now_text,
                file_transactions=self._file_transactions,
            )
        )


__all__ = ["SqliteInboundRecallRepository"]
