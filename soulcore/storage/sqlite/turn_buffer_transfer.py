"""Atomic transfer from conversation input buffering to state-gate deferral."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Sequence
from datetime import datetime

from ...contracts.turn_buffer import DeferredTurnBufferMessage
from ...features.timeline.sqlite.deferred_batch_transactions import (
    DeferredBatchAppendContext,
    DeferredBatchAppendTransaction,
)
from .codec import _dt
from .repository import SqliteRepository


class TurnBufferGateTransferCommandRepository(SqliteRepository):
    async def transfer_turn_buffer_to_state_gate(
        self,
        profile_id: str,
        instance_id: str,
        batch_id: str,
        *,
        expected_generation: int,
        expected_version: int,
        lease_token: int,
        expected_activity_epoch: int,
        gate_generation: int,
        due_at: datetime,
        messages: Sequence[DeferredTurnBufferMessage],
        transferred_at: datetime,
    ) -> bool:
        now_text = _dt(transferred_at)
        assert now_text is not None

        def operation(conn: sqlite3.Connection) -> bool:
            if not self._owns_claim(
                conn,
                profile_id,
                instance_id,
                batch_id,
                expected_generation,
                expected_version,
                lease_token,
                expected_activity_epoch,
            ):
                return False
            creation_key = f"state-gate:{int(gate_generation)}"
            for message in messages:
                DeferredBatchAppendTransaction(
                    DeferredBatchAppendContext(
                        profile_id=profile_id,
                        instance_id=instance_id,
                        message_id=int(message.message_id),
                        due_at=due_at,
                        activity_epoch=int(expected_activity_epoch),
                        gate_generation=int(gate_generation),
                        creation_key=creation_key,
                        identifier=f"defer:{uuid.uuid4().hex}",
                        message_ref=message.message_ref,
                        idempotency_key=message.message_ref,
                        received_at=message.received_at,
                        now=now_text,
                    )
                )(conn)
            cursor = conn.execute(
                """UPDATE conversation_turn_buffer_batches SET status = 'RESOLVED',
                due_at = NULL, lease_owner = NULL, lease_until = NULL,
                lease_token = lease_token + 1,
                resolution_outcome = 'TRANSFERRED_TO_STATE_GATE',
                version = version + 1, updated_at = ?, resolved_at = ?
                WHERE profile_id = ? AND instance_id = ? AND batch_id = ?
                AND status = 'CLAIMED' AND generation = ? AND activity_epoch = ?
                AND version = ? AND lease_token = ?""",
                (
                    now_text,
                    now_text,
                    profile_id,
                    instance_id,
                    batch_id,
                    int(expected_generation),
                    int(expected_activity_epoch),
                    int(expected_version),
                    int(lease_token),
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("turn-buffer ownership changed during gate transfer")
            return True

        return bool(await self.uow.run(operation))

    @staticmethod
    def _owns_claim(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        batch_id: str,
        generation: int,
        version: int,
        lease_token: int,
        activity_epoch: int,
    ) -> bool:
        row = conn.execute(
            """SELECT 1 FROM conversation_turn_buffer_batches
            WHERE profile_id = ? AND instance_id = ? AND batch_id = ?
            AND status = 'CLAIMED' AND generation = ? AND activity_epoch = ?
            AND version = ? AND lease_token = ?""",
            (
                profile_id,
                instance_id,
                batch_id,
                int(generation),
                int(activity_epoch),
                int(version),
                int(lease_token),
            ),
        ).fetchone()
        return row is not None


__all__ = ["TurnBufferGateTransferCommandRepository"]
