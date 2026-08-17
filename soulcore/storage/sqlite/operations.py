from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass

from ...features.delivery.sqlite.outbox import OutboxSettlementCommands
from ...features.delivery.sqlite.todo_ownership import bind_outbox_todos
from ...features.files.sqlite.release import FileReleaseCommands
from ...features.knowledge.sqlite.commit import KnowledgeCommitCommands
from ...features.main_core.sqlite.commit import (
    CoreCommitCommands,
    InstanceCoreResultCommands,
)
from ...features.main_core.sqlite.work_recovery_run import WorkRecoveryRunCommands
from ...features.profiles.ports import ProfilesRepositoryPort
from ...features.profiles.sqlite.runtime_clear import (
    ProfileRuntimeCommands as ProfileCleanupCommands,
)
from ...features.stickers.sqlite.candidate_transactions import (
    commit_core_sticker_import_intent,
)
from ...features.stickers.sqlite.retrieval import (
    disable_sticker_item_for_instance_in_transaction,
)
from ...features.timeline.sqlite.intents import (
    apply_character_intent_mutations_sql,
)
from .codec import _dt, _dump, _now
from .core_commit_transactions import CoreCommitTransactions
from .core_mappers import CoreRecordMappers
from .engine import SqliteEngine
from .repository import SqliteRepository
from .repository_lifecycle import KnowledgeTaskSql
from .runtime_file_cleanup import RuntimeFileCleanupRecords
from .scope_configuration import ScopeConfigurationCommandRepository
from .turn_buffer_transfer import TurnBufferGateTransferCommandRepository


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
    "CoreResultCommandRepository",
    "FileSettlementCommandRepository",
    "KnowledgeBatchCommandRepository",
    "OperationRepositories",
    "OutboxSettlementCommandRepository",
    "RuntimeCleanupCommandRepository",
    "ScopeConfigurationCommandRepository",
    "TurnBufferGateTransferCommandRepository",
]
