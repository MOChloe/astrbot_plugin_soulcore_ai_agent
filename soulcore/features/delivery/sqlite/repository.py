from __future__ import annotations

import uuid
from datetime import UTC, datetime

from ....storage.sqlite.background_projection import (
    acquire_foreground_lease_sql,
    lease_deadline,
    release_foreground_lease_sql,
    renew_foreground_lease_sql,
)
from ....storage.sqlite.core_mappers import CoreRecordMappers
from ....storage.sqlite.inbound_admission_transactions import InboundAdmissionTransactions
from ....storage.sqlite.repository import SqliteRepository
from ....storage.sqlite.repository_lifecycle import KnowledgeTaskSql
from ...profiles.ports import ProfilesRepositoryPort
from .expression_interruptions import ExpressionInterruptionRecords
from .expression_outbox import ExpressionOutboxRecords
from .expression_pacing import ExpressionPacingRecords
from .inbound_admission import InboundAdmissionResult
from .message_fragments import MessageFragmentRecords, MessageRetractionRecords
from .outbox import OutboxRecords
from .qpm import QpmRecords
from .qq_delivery import QqDeliveryRecords
from .support import Any, _dt, _now, _safe_dump, sqlite3
from .voice_artifacts import VoiceArtifactRecords
from .wakeups import DeliveryWakeupRecords


class ProfileDeliveryRecords:
    """Structured diagnostic log storage shared by all instance delivery paths."""

    async def append_log(
        self,
        profile_id: str,
        level: str,
        category: str,
        message: str,
        *,
        instance_id: str | None = None,
        details: Any = None,
        created_at: datetime | None = None,
    ) -> dict[str, Any]:
        level = str(level or "INFO").strip().upper() or "INFO"
        category = str(category or "general").strip() or "general"
        timestamp = _dt(created_at or _now())
        details_json = _safe_dump({} if details is None else details)

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            cursor = conn.execute(
                """INSERT INTO soulcore_logs(
                    profile_id, instance_id, level, category, message,
                    details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    profile_id,
                    instance_id,
                    level,
                    category,
                    str(message),
                    details_json,
                    timestamp,
                ),
            )
            log_id = int(cursor.lastrowid)
            conn.execute(
                """DELETE FROM soulcore_logs
                WHERE profile_id = ? AND log_id NOT IN (
                    SELECT log_id FROM soulcore_logs
                    WHERE profile_id = ?
                    ORDER BY log_id DESC LIMIT 1000
                )""",
                (profile_id, profile_id),
            )
            row = conn.execute("SELECT * FROM soulcore_logs WHERE log_id = ?", (log_id,)).fetchone()
            if row is None:
                raise RuntimeError("newly appended SoulCore log was unexpectedly pruned")
            return row

        row = await self.uow.run(operation)
        return self._record(row, json_columns=("details_json",))

    async def list_logs(
        self,
        profile_id: str,
        *,
        level: str | None = None,
        category: str | None = None,
        instance_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        clauses = ["profile_id = ?"]
        parameters: list[Any] = [profile_id]
        if level is not None:
            clauses.append("level = ?")
            parameters.append(str(level).strip().upper())
        if category is not None:
            clauses.append("category = ?")
            parameters.append(str(category).strip())
        if instance_id is not None:
            clauses.append("instance_id = ?")
            parameters.append(instance_id)
        parameters.append(limit)
        rows = await self.db.fetch_all(
            f"""SELECT * FROM soulcore_logs
            WHERE {" AND ".join(clauses)}
            ORDER BY log_id DESC LIMIT ?""",
            parameters,
        )
        return [self._record(row, json_columns=("details_json",)) for row in rows]

    async def clear_logs(self, profile_id: str) -> int:
        cursor = await self.db.call(
            lambda conn: conn.execute(
                "DELETE FROM soulcore_logs WHERE profile_id = ?", (profile_id,)
            ),
            transaction=True,
        )
        return int(cursor.rowcount)


class _ExpressionQueue(
    ExpressionInterruptionRecords,
    ExpressionOutboxRecords,
    OutboxRecords,
):
    pass


class _MessageDeliveryRecords(MessageRetractionRecords, MessageFragmentRecords):
    pass


class _DeliveryQueue(
    _ExpressionQueue,
    _MessageDeliveryRecords,
    DeliveryWakeupRecords,
    VoiceArtifactRecords,
):
    pass


class _DeliveryPolicy(
    ExpressionPacingRecords,
    ProfileDeliveryRecords,
    QpmRecords,
    QqDeliveryRecords,
):
    pass


class SqliteDeliveryRepository(
    _DeliveryQueue,
    _DeliveryPolicy,
    KnowledgeTaskSql,
    CoreRecordMappers,
    SqliteRepository,
):
    """SQLite implementation of delivery, QPM, wakeup, and outbox storage."""

    def __init__(
        self,
        engine,
        profiles: ProfilesRepositoryPort,
        inbound_admission: InboundAdmissionTransactions,
    ) -> None:
        SqliteRepository.__init__(self, engine)
        self._profiles = profiles
        self._inbound_admission = inbound_admission

    async def publish_context_backup(self) -> str | None:
        path = await self.db.publish_backup_after_commit(operation="delivery_context_commit")
        return str(path) if path is not None else None

    async def apply_inbound_admission(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
        *,
        group_scope: bool,
        lease_owner: str | None = None,
        lease_token: int | None = None,
    ) -> InboundAdmissionResult:
        from datetime import UTC, datetime

        result = await self.uow.run(
            lambda conn: self._inbound_admission.apply(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                message_id=int(message_id),
                now=datetime.now(UTC),
                group_scope=bool(group_scope),
                refresh_knowledge_task=self._refresh_knowledge_task_sql,
                lease_owner=lease_owner,
                lease_token=lease_token,
            )
        )
        if result.interruption_changed:
            await self.publish_context_backup()
        return result

    async def renew_inbound_admission(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
        *,
        lease_owner: str,
        lease_token: int,
        lease_seconds: int,
    ) -> bool:
        from datetime import UTC, datetime

        return bool(
            await self.uow.run(
                lambda conn: self._inbound_admission.renew(
                    conn,
                    profile_id=profile_id,
                    instance_id=instance_id,
                    message_id=int(message_id),
                    lease_owner=lease_owner,
                    lease_token=int(lease_token),
                    now=datetime.now(UTC),
                    lease_seconds=int(lease_seconds),
                )
            )
        )

    async def complete_inbound_admission(
        self,
        profile_id: str,
        instance_id: str,
        message_id: int,
        *,
        lease_owner: str,
        lease_token: int,
        status: str = "APPLIED",
    ) -> bool:
        from datetime import UTC, datetime

        return bool(
            await self.uow.run(
                lambda conn: self._inbound_admission.complete(
                    conn,
                    profile_id=profile_id,
                    instance_id=instance_id,
                    message_id=int(message_id),
                    lease_owner=lease_owner,
                    lease_token=int(lease_token),
                    now=datetime.now(UTC),
                    status=status,
                )
            )
        )

    async def acquire_foreground_lease(
        self,
        profile_id: str,
        instance_id: str,
        *,
        owner: str,
        lease_seconds: int,
    ) -> str:
        now_dt = datetime.now(UTC)
        now = now_dt.isoformat()
        token = uuid.uuid4().hex
        effective_token = await self.uow.run(
            lambda conn: acquire_foreground_lease_sql(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                owner=owner,
                token=token,
                lease_until=lease_deadline(now_dt, lease_seconds),
                now=now,
            )
        )
        return str(effective_token)

    async def renew_foreground_lease(
        self,
        profile_id: str,
        instance_id: str,
        *,
        owner: str,
        token: str,
        lease_seconds: int,
    ) -> bool:
        now_dt = datetime.now(UTC)
        return bool(
            await self.uow.run(
                lambda conn: renew_foreground_lease_sql(
                    conn,
                    profile_id=profile_id,
                    instance_id=instance_id,
                    owner=owner,
                    token=token,
                    lease_until=lease_deadline(now_dt, lease_seconds),
                    now=now_dt.isoformat(),
                )
            )
        )

    async def release_foreground_lease(
        self,
        profile_id: str,
        instance_id: str,
        *,
        owner: str,
        token: str,
    ) -> bool:
        return bool(
            await self.uow.run(
                lambda conn: release_foreground_lease_sql(
                    conn,
                    profile_id=profile_id,
                    instance_id=instance_id,
                    owner=owner,
                    token=token,
                    released_at=datetime.now(UTC).isoformat(),
                )
            )
        )


__all__ = ["SqliteDeliveryRepository"]
