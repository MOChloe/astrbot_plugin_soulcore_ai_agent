"""Async SQLite repository for durable Main Core work checkpoints."""

from __future__ import annotations

from ....storage.sqlite.engine import SqliteEngine
from ....storage.sqlite.repository import SqliteRepository
from ..work_checkpoint import MainCoreWorkCheckpoint, WorkScope
from ..work_checkpoint_repository import (
    MAX_EVENT_PAGE_SIZE,
    ApplyWorkCallbackCommand,
    ClaimWorkCheckpointLeaseCommand,
    FreezeWorkCheckpointCommand,
    ReleaseWorkCheckpointLeaseCommand,
    RenewWorkCheckpointLeaseCommand,
    TransitionWorkCheckpointCommand,
    WorkCallbackStorageResult,
    WorkCheckpointEvent,
    WorkCheckpointMutationResult,
)
from ..work_checkpoint_storage_errors import (
    WorkCheckpointStorageErrorCode,
    storage_fail,
)
from .work_checkpoint_operations import WorkCheckpointSqliteOperations


class SqliteWorkCheckpointRepository(SqliteRepository):
    def __init__(self, engine: SqliteEngine) -> None:
        super().__init__(engine)
        self.operations = WorkCheckpointSqliteOperations()

    async def get_checkpoint(
        self, scope: WorkScope, work_ref: str
    ) -> MainCoreWorkCheckpoint | None:
        return await self.db.call(
            lambda conn: self.operations.get_checkpoint(conn, scope, work_ref)
        )

    async def freeze_checkpoint(
        self, command: FreezeWorkCheckpointCommand
    ) -> WorkCheckpointMutationResult:
        result = await self.uow.run(lambda conn: self.operations.freeze_checkpoint(conn, command))
        await self.db.publish_backup_after_commit()
        return result

    async def apply_callback(self, command: ApplyWorkCallbackCommand) -> WorkCallbackStorageResult:
        result = await self.uow.run(lambda conn: self.operations.apply_callback(conn, command))
        await self.db.publish_backup_after_commit()
        return result

    async def claim_lease(
        self, command: ClaimWorkCheckpointLeaseCommand
    ) -> WorkCheckpointMutationResult:
        result = await self.uow.run(lambda conn: self.operations.claim_lease(conn, command))
        await self.db.publish_backup_after_commit()
        return result

    async def renew_lease(
        self, command: RenewWorkCheckpointLeaseCommand
    ) -> WorkCheckpointMutationResult:
        result = await self.uow.run(lambda conn: self.operations.renew_lease(conn, command))
        await self.db.publish_backup_after_commit()
        return result

    async def release_lease(
        self, command: ReleaseWorkCheckpointLeaseCommand
    ) -> WorkCheckpointMutationResult:
        result = await self.uow.run(lambda conn: self.operations.release_lease(conn, command))
        await self.db.publish_backup_after_commit()
        return result

    async def transition_terminal(
        self, command: TransitionWorkCheckpointCommand
    ) -> WorkCheckpointMutationResult:
        result = await self.uow.run(lambda conn: self.operations.transition_terminal(conn, command))
        await self.db.publish_backup_after_commit()
        return result

    async def list_events(
        self, scope: WorkScope, work_ref: str, *, limit: int = 256
    ) -> tuple[WorkCheckpointEvent, ...]:
        _limit(limit, MAX_EVENT_PAGE_SIZE)
        return await self.db.call(
            lambda conn: self.operations.list_events(conn, scope, work_ref, limit)
        )


def _limit(value: int, maximum: int) -> None:
    if isinstance(value, bool) or not 1 <= value <= maximum:
        raise storage_fail(WorkCheckpointStorageErrorCode.OUT_OF_RANGE)


__all__ = ["SqliteWorkCheckpointRepository"]
