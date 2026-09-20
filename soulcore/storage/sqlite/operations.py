from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ...contracts.turn_buffer import DeferredTurnBufferMessage
from ...features.delivery.sqlite.outbox import OutboxSettlementCommands
from ...features.delivery.sqlite.todo_ownership import bind_outbox_todos
from ...features.files.sqlite.release import FileReleaseCommands
from ...features.knowledge.sqlite.commit import KnowledgeCommitCommands
from ...features.main_core.sqlite.commit import CoreCommitCommands, InstanceCoreResultCommands
from ...features.main_core.sqlite.work_recovery_run import WorkRecoveryRunCommands
from ...features.profiles.ports import ProfilesRepositoryPort
from ...features.profiles.sqlite.management import ProfileRuntimeCommands as ProfileCleanupCommands
from ...features.stickers.service import StickerImportIntent, StickerInstanceDisableCommitter
from ...features.stickers.sqlite.candidate_transactions import commit_core_sticker_import_intent
from ...features.stickers.sqlite.retrieval import disable_sticker_item_for_instance_in_transaction
from ...features.timeline.sqlite.deferred_batch_transactions import (
    DeferredBatchAppendContext,
    DeferredBatchAppendTransaction,
)
from ...features.timeline.sqlite.intents import apply_character_intent_mutations_sql
from .codec import _dt, _dump, _now
from .core_mappers import CoreRecordMappers
from .engine import SqliteEngine
from .repository import SqliteRepository
from .repository_lifecycle import KnowledgeTaskSql
from .runtime_file_cleanup import RuntimeFileCleanupRecords
from .scope_configuration import ScopeConfigurationCommandRepository


class OutboxTodoBinder(Protocol):
    def __call__(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        todo_ids: Iterable[str],
        selected_run_id: int | None,
    ) -> None: ...


class StickerImportCommitter(Protocol):
    def __call__(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        instance_id: str,
        run_id: int,
        intent: StickerImportIntent,
        now: str,
    ) -> tuple[str, bool]: ...


@dataclass(frozen=True, slots=True)
class CoreCommitTransactions:
    """Route cross-feature writes through dependencies owned by composition."""

    outbox_todo_binder: OutboxTodoBinder
    sticker_import_committer: StickerImportCommitter
    sticker_disable_committer: StickerInstanceDisableCommitter

    def bind_todos(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        todo_ids: Iterable[str],
        selected_run_id: int | None,
    ) -> None:
        self.outbox_todo_binder(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            outbox_id=outbox_id,
            todo_ids=todo_ids,
            selected_run_id=selected_run_id,
        )

    def commit_sticker(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        instance_id: str,
        run_id: int,
        intent: StickerImportIntent,
        now: str,
    ) -> tuple[str, bool]:
        return self.sticker_import_committer(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            run_id=run_id,
            intent=intent,
            now=now,
        )

    def disable_sticker(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        item_id: str,
        *,
        now: datetime,
    ) -> None:
        self.sticker_disable_committer(
            conn,
            profile_id,
            instance_id,
            item_id,
            now=now,
        )


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


class _IntentCommandSupport:
    _apply_character_intent_mutations_sql = staticmethod(apply_character_intent_mutations_sql)


class _KnowledgeCommandSupport:
    @staticmethod
    def _knowledge_audit_sql(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        entity_type: str,
        entity_id: int | None,
        action: str,
        actor_type: str,
        actor_id: str,
        reason: str,
        details: dict[str, object],
        created_at: str | None,
    ) -> None:
        conn.execute(
            """INSERT INTO knowledge_audit(
                profile_id, instance_id, entity_type, entity_id, action,
                actor_type, actor_id, reason, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                profile_id,
                instance_id,
                entity_type,
                entity_id,
                action,
                actor_type,
                actor_id,
                reason,
                _dump(details),
                created_at,
            ),
        )


class _CoreCommandSupport(_IntentCommandSupport, CoreRecordMappers):
    @staticmethod
    def _audit_ai_task(
        conn: sqlite3.Connection,
        row: sqlite3.Row | Mapping[str, object],
        action: str,
        *,
        from_status: str | None = None,
        to_status: str | None = None,
        actor_type: str = "SYSTEM",
        actor_id: str = "",
        details: dict[str, object] | None = None,
        created_at: str | None = None,
    ) -> None:
        conn.execute(
            """INSERT INTO ai_task_audit(
                task_id, profile_id, instance_id, actor_type, actor_id,
                action, from_status, to_status, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["task_id"],
                row["profile_id"],
                row["instance_id"],
                actor_type,
                actor_id,
                action,
                from_status,
                to_status,
                _dump(details or {}),
                created_at or _dt(_now()),
            ),
        )


class _CoreResultSqliteCommands(InstanceCoreResultCommands, WorkRecoveryRunCommands):
    pass


class CoreResultCommandRepository(
    _CoreResultSqliteCommands,
    _CoreCommandSupport,
    KnowledgeTaskSql,
    SqliteRepository,
):
    """Atomic Main Core result commits for character instances."""

    def __init__(
        self,
        engine: SqliteEngine,
        profiles: ProfilesRepositoryPort,
        core_commit_transactions: CoreCommitTransactions,
    ) -> None:
        SqliteRepository.__init__(self, engine)
        self._profiles = profiles
        self._core_commit_transactions = core_commit_transactions


class KnowledgeBatchCommandRepository(
    KnowledgeCommitCommands,
    _KnowledgeCommandSupport,
    SqliteRepository,
):
    """Atomic knowledge batch, revision, and task commit."""


class OutboxSettlementCommandRepository(
    OutboxSettlementCommands,
    KnowledgeTaskSql,
    CoreRecordMappers,
    SqliteRepository,
):
    """Atomic outbox settlement and message-ledger update."""

    async def publish_context_backup(self) -> str | None:
        path = await self.db.publish_backup_after_commit(operation="outbox_settlement")
        return str(path) if path is not None else None


class FileSettlementCommandRepository(FileReleaseCommands, SqliteRepository):
    """Atomic file release preparation and final deletion settlement."""


class RuntimeCleanupCommandRepository(
    CoreCommitCommands,
    ProfileCleanupCommands,
    RuntimeFileCleanupRecords,
    CoreRecordMappers,
    SqliteRepository,
):
    """Atomic instance runtime privacy cleanup across owned tables."""

    def __init__(self, engine: SqliteEngine, profiles: ProfilesRepositoryPort) -> None:
        SqliteRepository.__init__(self, engine)
        self._profiles = profiles

    async def get_profile(self, profile_id: str) -> object:
        return await self._profiles.get_profile(profile_id)


@dataclass(frozen=True, slots=True)
class OperationRepositories:
    core_results: CoreResultCommandRepository
    knowledge_batches: KnowledgeBatchCommandRepository
    outbox_settlement: OutboxSettlementCommandRepository
    file_settlement: FileSettlementCommandRepository
    runtime_cleanup: RuntimeCleanupCommandRepository
    scope_configuration: ScopeConfigurationCommandRepository
    turn_buffer_gate_transfer: TurnBufferGateTransferCommandRepository

    @classmethod
    def create(
        cls,
        engine: SqliteEngine,
        profiles: ProfilesRepositoryPort,
    ) -> OperationRepositories:
        core_commit_transactions = CoreCommitTransactions(
            outbox_todo_binder=bind_outbox_todos,
            sticker_import_committer=commit_core_sticker_import_intent,
            sticker_disable_committer=disable_sticker_item_for_instance_in_transaction,
        )
        return cls(
            core_results=CoreResultCommandRepository(
                engine,
                profiles,
                core_commit_transactions,
            ),
            knowledge_batches=KnowledgeBatchCommandRepository(engine),
            outbox_settlement=OutboxSettlementCommandRepository(engine),
            file_settlement=FileSettlementCommandRepository(engine),
            runtime_cleanup=RuntimeCleanupCommandRepository(engine, profiles),
            scope_configuration=ScopeConfigurationCommandRepository(engine),
            turn_buffer_gate_transfer=TurnBufferGateTransferCommandRepository(engine),
        )


__all__ = [
    "CoreCommitTransactions",
    "CoreResultCommandRepository",
    "FileSettlementCommandRepository",
    "KnowledgeBatchCommandRepository",
    "OperationRepositories",
    "OutboxSettlementCommandRepository",
    "RuntimeCleanupCommandRepository",
    "ScopeConfigurationCommandRepository",
    "TurnBufferGateTransferCommandRepository",
]
