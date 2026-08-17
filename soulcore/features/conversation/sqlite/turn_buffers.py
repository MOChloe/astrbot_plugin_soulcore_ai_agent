from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta

from ....storage.sqlite.codec import _dt, _now, _parse
from ....storage.sqlite.engine import SqliteEngine
from ....storage.sqlite.inbound_admission_transactions import InboundAdmissionTransactions
from ....storage.sqlite.repository import SqliteRepository
from ....storage.sqlite.repository_lifecycle import KnowledgeTaskSql
from ..turn_buffer import TurnBufferBatch, TurnBufferStatus
from .turn_buffer_append import AppendTurnBufferBatch, normalize_turn_buffer_append
from .turn_buffer_recovery import TurnBufferRecoveryMixin


class SqliteTurnBufferRepository(TurnBufferRecoveryMixin, KnowledgeTaskSql, SqliteRepository):
    def __init__(
        self,
        engine: SqliteEngine,
        inbound_admission: InboundAdmissionTransactions,
    ) -> None:
        super().__init__(engine)
        self._inbound_admission = inbound_admission

    async def get_turn_buffer_batch(
        self, profile_id: str, instance_id: str, batch_id: str
    ) -> TurnBufferBatch | None:
        row = await self.db.fetch_one(
            """SELECT * FROM conversation_turn_buffer_batches
            WHERE profile_id = ? AND instance_id = ? AND batch_id = ?""",
            (profile_id, instance_id, batch_id),
        )
        return await self._batch(row) if row else None

    async def get_active_turn_buffer_batch(
        self, profile_id: str, instance_id: str
    ) -> TurnBufferBatch | None:
        row = await self.db.fetch_one(
            """SELECT * FROM conversation_turn_buffer_batches
            WHERE profile_id = ? AND instance_id = ?
              AND status IN ('PENDING','CLASSIFYING','WAITING','CLAIMED')""",
            (profile_id, instance_id),
        )
        return await self._batch(row) if row else None

    async def append_or_refresh_turn_buffer_batch(
        self,
        profile_id: str,
        instance_id: str,
        *,
        message_ids: Sequence[int],
        activity_epoch: int,
        now: datetime,
        admission_message_id: int | None = None,
        admission_lease_owner: str | None = None,
        admission_lease_token: int | None = None,
    ) -> TurnBufferBatch:
        ids, now_text = normalize_turn_buffer_append(
            message_ids,
            activity_epoch,
            now,
            admission_message_id,
            admission_lease_owner,
            admission_lease_token,
        )
        batch_id = await self.uow.run(
            AppendTurnBufferBatch(
                inbound_admission=self._inbound_admission,
                profile_id=profile_id,
                instance_id=instance_id,
                message_ids=ids,
                activity_epoch=activity_epoch,
                now_text=now_text,
                admission_message_id=admission_message_id,
                admission_lease_owner=admission_lease_owner,
                admission_lease_token=admission_lease_token,
            )
        )
        result = await self.get_turn_buffer_batch(profile_id, instance_id, batch_id)
        assert result is not None
        return result

    async def claim_turn_buffer_batches_for_classification(
        self,
        *,
        now: datetime,
        limit: int = 10,
        lease_seconds: int = 30,
        worker_id: str = "turn-buffer-classifier",
    ) -> tuple[TurnBufferBatch, ...]:
        await self._recover_expired_classification_leases(now=now)
        return await self._claim(
            from_status="PENDING",
            to_status="CLASSIFYING",
            now=now,
            limit=limit,
            lease_seconds=lease_seconds,
            worker_id=worker_id,
            due=False,
        )

    async def defer_turn_buffer_classification(
        self,
        profile_id: str,
        instance_id: str,
        batch_id: str,
        *,
        expected_generation: int,
        expected_version: int,
        lease_token: int,
        reason: str,
    ) -> bool:
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE conversation_turn_buffer_batches SET status = 'PENDING',
                due_at = NULL, lease_owner = NULL, lease_until = NULL,
                lease_token = lease_token + 1, error_code = ?, version = version + 1,
                updated_at = ? WHERE profile_id = ? AND instance_id = ?
                AND batch_id = ? AND status = 'CLASSIFYING' AND generation = ?
                AND version = ? AND lease_token = ?""",
                (
                    str(reason).strip()[:80],
                    _dt(_now()),
                    profile_id,
                    instance_id,
                    batch_id,
                    int(expected_generation),
                    int(expected_version),
                    int(lease_token),
                ),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1

    async def record_turn_buffer_decision(
        self,
        profile_id: str,
        instance_id: str,
        batch_id: str,
        *,
        expected_generation: int,
        expected_version: int,
        lease_token: int,
        requested_delay_seconds: int | None,
        ai_elapsed_seconds: float | None,
        due_at: datetime,
        error_code: str = "",
    ) -> TurnBufferBatch | None:
        requested = None if requested_delay_seconds is None else int(requested_delay_seconds)
        elapsed = None if ai_elapsed_seconds is None else float(ai_elapsed_seconds)
        if requested is not None and not 0 <= requested <= 60:
            raise ValueError("requested turn-buffer delay must be between zero and sixty")
        if elapsed is not None and elapsed < 0:
            raise ValueError("AI elapsed time cannot be negative")
        if requested is None and not str(error_code).strip():
            raise ValueError("failed-open decision requires an error code")
        remaining = max(0.0, float(requested or 0) - float(elapsed or 0))
        now_text = _dt(_now())
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE conversation_turn_buffer_batches SET status = 'WAITING',
                requested_delay_seconds = ?, ai_elapsed_seconds = ?,
                remaining_delay_seconds = ?, due_at = ?, error_code = ?,
                lease_owner = NULL, lease_until = NULL, version = version + 1,
                updated_at = ? WHERE profile_id = ? AND instance_id = ?
                AND batch_id = ? AND status = 'CLASSIFYING' AND generation = ?
                AND version = ? AND lease_token = ?""",
                (
                    requested,
                    elapsed,
                    remaining,
                    _dt(due_at),
                    str(error_code).strip()[:80],
                    now_text,
                    profile_id,
                    instance_id,
                    batch_id,
                    int(expected_generation),
                    int(expected_version),
                    int(lease_token),
                ),
            ),
            transaction=True,
        )
        return (
            await self.get_turn_buffer_batch(profile_id, instance_id, batch_id)
            if cursor.rowcount == 1
            else None
        )

    async def claim_due_turn_buffer_batches(
        self,
        *,
        now: datetime,
        limit: int = 10,
        lease_seconds: int = 120,
        worker_id: str = "turn-buffer-admission",
    ) -> tuple[TurnBufferBatch, ...]:
        now_text = _dt(now)
        assert now_text is not None
        await self.db.call(
            lambda conn: conn.execute(
                """UPDATE conversation_turn_buffer_batches SET status = 'WAITING',
                lease_owner = NULL, lease_until = NULL, lease_token = lease_token + 1,
                version = version + 1, error_code = CASE WHEN error_code = ''
                    THEN 'admission_lease_expired' ELSE error_code END, updated_at = ?
                WHERE status = 'CLAIMED' AND lease_until <= ?""",
                (now_text, now_text),
            ),
            transaction=True,
        )
        return await self._claim(
            from_status="WAITING",
            to_status="CLAIMED",
            now=now,
            limit=limit,
            lease_seconds=lease_seconds,
            worker_id=worker_id,
            due=True,
        )

    async def renew_turn_buffer_batch_lease(
        self,
        profile_id: str,
        instance_id: str,
        batch_id: str,
        *,
        expected_status: TurnBufferStatus,
        expected_generation: int,
        lease_token: int,
        lease_owner: str,
        now: datetime,
        lease_seconds: int,
    ) -> bool:
        if expected_status not in {TurnBufferStatus.CLASSIFYING, TurnBufferStatus.CLAIMED}:
            raise ValueError("only active turn-buffer claims have renewable leases")
        owner = str(lease_owner).strip()
        if not owner:
            raise ValueError("turn-buffer lease_owner cannot be empty")
        now_text = _dt(now)
        lease_until = _dt(now + timedelta(seconds=max(1, int(lease_seconds))))
        assert now_text is not None and lease_until is not None
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE conversation_turn_buffer_batches SET lease_until = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND batch_id = ? AND status = ?
                AND generation = ? AND lease_token = ? AND lease_owner = ?""",
                (
                    lease_until,
                    now_text,
                    profile_id,
                    instance_id,
                    batch_id,
                    expected_status.value,
                    int(expected_generation),
                    int(lease_token),
                    owner,
                ),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1

    async def attach_turn_buffer_main_core_task(
        self,
        profile_id: str,
        instance_id: str,
        batch_id: str,
        *,
        expected_generation: int,
        expected_version: int,
        lease_token: int,
        main_core_task_ref: str,
    ) -> TurnBufferBatch | None:
        reference = str(main_core_task_ref).strip()
        if not reference:
            raise ValueError("main_core_task_ref cannot be empty")
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE conversation_turn_buffer_batches SET main_core_task_ref = ?,
                version = version + 1, updated_at = ? WHERE profile_id = ?
                AND instance_id = ? AND batch_id = ? AND status = 'CLAIMED'
                AND generation = ? AND version = ? AND lease_token = ?""",
                (
                    reference,
                    _dt(_now()),
                    profile_id,
                    instance_id,
                    batch_id,
                    int(expected_generation),
                    int(expected_version),
                    int(lease_token),
                ),
            ),
            transaction=True,
        )
        return (
            await self.get_turn_buffer_batch(profile_id, instance_id, batch_id)
            if cursor.rowcount == 1
            else None
        )

    async def resolve_turn_buffer_batch(
        self,
        profile_id: str,
        instance_id: str,
        batch_id: str,
        *,
        expected_generation: int,
        expected_version: int,
        lease_token: int,
        expected_activity_epoch: int,
        outcome: str,
        resolved_at: datetime,
    ) -> bool:
        normalized = str(outcome).strip().upper() or "COMPLETED"
        status = (
            "CANCELLED"
            if normalized == "CANCELLED"
            else "FAILED"
            if normalized == "FAILED"
            else "RESOLVED"
        )
        now_text = _dt(resolved_at)
        assert now_text is not None

        def operation(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                """UPDATE conversation_turn_buffer_batches SET status = ?, due_at = NULL,
                lease_owner = NULL, lease_until = NULL, lease_token = lease_token + 1,
                resolution_outcome = ?, version = version + 1, updated_at = ?,
                resolved_at = ? WHERE profile_id = ? AND instance_id = ?
                AND batch_id = ? AND status = 'CLAIMED' AND generation = ?
                AND activity_epoch = ? AND version = ? AND lease_token = ?""",
                (
                    status,
                    normalized[:80],
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
                return False
            if normalized not in {"DEFERRED", "TRANSFERRED_TO_STATE_GATE"}:
                changed = conn.execute(
                    """UPDATE instance_messages SET knowledge_eligibility = 'ELIGIBLE',
                    knowledge_eligibility_reason = '' WHERE profile_id = ? AND instance_id = ?
                    AND knowledge_eligibility = 'HELD'
                    AND knowledge_eligibility_reason = 'inbound_turn_buffer_pending'
                    AND message_id IN (SELECT message_id
                        FROM conversation_turn_buffer_members WHERE batch_id = ?)""",
                    (profile_id, instance_id, batch_id),
                ).rowcount
                if changed:
                    conn.execute(
                        """UPDATE knowledge_processing_state SET
                        processing_version = processing_version + 1, updated_at = ?
                        WHERE profile_id = ? AND instance_id = ?""",
                        (now_text, profile_id, instance_id),
                    )
            return True

        return bool(await self.uow.run(operation))

    async def release_turn_buffer_batch(
        self,
        profile_id: str,
        instance_id: str,
        batch_id: str,
        *,
        expected_generation: int,
        expected_version: int,
        lease_token: int,
        retry_at: datetime,
        reason: str,
    ) -> bool:
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE conversation_turn_buffer_batches SET status = 'WAITING',
                due_at = ?, lease_owner = NULL, lease_until = NULL,
                lease_token = lease_token + 1, error_code = ?, version = version + 1,
                updated_at = ? WHERE profile_id = ? AND instance_id = ?
                AND batch_id = ? AND status = 'CLAIMED' AND generation = ?
                AND version = ? AND lease_token = ?""",
                (
                    _dt(retry_at),
                    str(reason).strip()[:80],
                    _dt(_now()),
                    profile_id,
                    instance_id,
                    batch_id,
                    int(expected_generation),
                    int(expected_version),
                    int(lease_token),
                ),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1

    async def _claim(
        self,
        *,
        from_status: str,
        to_status: str,
        now: datetime,
        limit: int,
        lease_seconds: int,
        worker_id: str,
        due: bool,
    ) -> tuple[TurnBufferBatch, ...]:
        owner = str(worker_id).strip()
        if not owner:
            raise ValueError("turn-buffer worker_id cannot be empty")
        now_text = _dt(now)
        lease_until = _dt(now + timedelta(seconds=max(1, int(lease_seconds))))
        assert now_text is not None and lease_until is not None

        def operation(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
            due_sql = " AND due_at <= ?" if due else ""
            params: list[object] = [from_status]
            if due:
                params.append(now_text)
            params.append(max(0, int(limit)))
            rows = list(
                conn.execute(
                    f"""SELECT batch_id, profile_id, instance_id
                    FROM conversation_turn_buffer_batches AS batch
                    WHERE status = ?{due_sql} AND EXISTS (
                        SELECT 1 FROM role_profiles profile
                        WHERE profile.profile_id = batch.profile_id AND profile.enabled = 1
                    )
                    ORDER BY {"due_at," if due else ""} updated_at,
                    batch_id LIMIT ?""",
                    params,
                )
            )
            claimed: list[tuple[str, str, str]] = []
            for row in rows:
                cursor = conn.execute(
                    """UPDATE conversation_turn_buffer_batches SET status = ?,
                    lease_owner = ?, lease_until = ?, lease_token = lease_token + 1,
                    version = version + 1, updated_at = ? WHERE batch_id = ? AND status = ?""",
                    (to_status, owner, lease_until, now_text, row["batch_id"], from_status),
                )
                if cursor.rowcount:
                    claimed.append(
                        (str(row["profile_id"]), str(row["instance_id"]), str(row["batch_id"]))
                    )
            return claimed

        identities = await self.uow.run(operation)
        result: list[TurnBufferBatch] = []
        for profile_id, instance_id, batch_id in identities:
            batch = await self.get_turn_buffer_batch(profile_id, instance_id, batch_id)
            if batch is not None:
                result.append(batch)
        return tuple(result)

    async def _batch(self, row: sqlite3.Row) -> TurnBufferBatch:
        members = await self.db.fetch_all(
            """SELECT message_id FROM conversation_turn_buffer_members
            WHERE batch_id = ? ORDER BY ordinal""",
            (row["batch_id"],),
        )
        return TurnBufferBatch(
            batch_id=str(row["batch_id"]),
            profile_id=str(row["profile_id"]),
            instance_id=str(row["instance_id"]),
            generation=int(row["generation"]),
            activity_epoch=int(row["activity_epoch"]),
            status=TurnBufferStatus(row["status"]),
            message_ids=tuple(int(item["message_id"]) for item in members),
            requested_delay_seconds=(
                int(row["requested_delay_seconds"])
                if row["requested_delay_seconds"] is not None
                else None
            ),
            ai_elapsed_seconds=(
                float(row["ai_elapsed_seconds"]) if row["ai_elapsed_seconds"] is not None else None
            ),
            remaining_delay_seconds=(
                float(row["remaining_delay_seconds"])
                if row["remaining_delay_seconds"] is not None
                else None
            ),
            due_at=_parse(row["due_at"]),
            lease_owner=(str(row["lease_owner"]) if row["lease_owner"] is not None else None),
            lease_token=int(row["lease_token"]),
            lease_until=_parse(row["lease_until"]),
            main_core_task_ref=(
                str(row["main_core_task_ref"]) if row["main_core_task_ref"] is not None else None
            ),
            error_code=str(row["error_code"]),
            resolution_outcome=str(row["resolution_outcome"]),
            version=int(row["version"]),
            created_at=_parse(row["created_at"]),
            updated_at=_parse(row["updated_at"]),
            resolved_at=_parse(row["resolved_at"]),
        )


__all__ = ["SqliteTurnBufferRepository"]
