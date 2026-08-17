from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from ...shared.private_paths import ensure_private_directory, restrict_private_path
from .backup import SQLiteBackupManager, infer_backup_path
from .schema.current import (
    CURRENT_SCHEMA_VERSION,
    SchemaRecoveryReason,
    SchemaRecoveryRequired,
    create_current_schema,
    database_is_empty,
    database_uses_current_schema,
    database_uses_schema_identity,
    read_schema_identity,
    read_schema_version,
    require_current_schema_definition,
    require_integrity,
    schema_identity_for_version,
)
from .schema.migrations import migrate_to_current
from .write_fence import SQLiteWriteFence

T = TypeVar("T")
logger = logging.getLogger(__name__)
_ACTIVE_TRANSACTION: ContextVar[int | None] = ContextVar(
    "soulcore_sqlite_active_transaction", default=None
)
_WRITE_AUTHORIZER_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_ANALYZE,
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
        sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_CREATE_VTABLE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_TRIGGER,
        sqlite3.SQLITE_DROP_TEMP_VIEW,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_DROP_VTABLE,
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_REINDEX,
        sqlite3.SQLITE_SAVEPOINT,
        sqlite3.SQLITE_TRANSACTION,
        sqlite3.SQLITE_UPDATE,
    }
)
_READ_ONLY_PRAGMAS = frozenset(
    {
        "collation_list",
        "compile_options",
        "database_list",
        "data_version",
        "foreign_key_check",
        "foreign_key_list",
        "function_list",
        "index_info",
        "index_list",
        "index_xinfo",
        "integrity_check",
        "module_list",
        "pragma_list",
        "table_info",
        "table_list",
        "table_xinfo",
    }
)


@dataclass(frozen=True, slots=True)
class BackupPublicationWarning:
    occurred_at: str
    operation: str
    error_type: str
    invalidation_error_type: str


class SqliteEngine:
    """Single-connection async SQLite lifecycle with serialized operations."""

    def __init__(
        self,
        path: str | Path,
        *,
        backup_path: str | Path | None = None,
        file_artifact_root: str | Path | None = None,
    ) -> None:
        self.path = str(path)
        self._connection: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._last_backup_warning: BackupPublicationWarning | None = None
        self._secured_database_files: dict[Path, tuple[int, int]] = {}
        self._write_fence = None if self.path == ":memory:" else SQLiteWriteFence(path)
        if self.path == ":memory:":
            self.backup_manager = None
            self._restore_explicit_backup = False
        else:
            self._restore_explicit_backup = backup_path is not None
            resolved_backup_path = (
                Path(backup_path) if backup_path is not None else infer_backup_path(path)
            )
            self.backup_manager = (
                SQLiteBackupManager(
                    path,
                    resolved_backup_path,
                    file_artifact_root=file_artifact_root,
                )
                if resolved_backup_path is not None
                else None
            )
            if self.backup_manager is not None:
                self._write_fence = self.backup_manager.write_fence

    @property
    def is_open(self) -> bool:
        return self._connection is not None

    async def open(self) -> SqliteEngine:
        async with self._lifecycle_lock:
            return await self._open_locked()

    async def _open_locked(self) -> SqliteEngine:
        if self._connection is not None:
            return self
        await self._prepare_managed_storage()
        await self._restore_explicit_backup_if_needed()
        connection, changed = await self._open_connection_for_lifecycle()
        self._connection = connection
        return await self._publish_open_backup_or_close(changed)

    async def _prepare_managed_storage(self) -> None:
        if self.path != ":memory:":
            ensure_private_directory(Path(self.path).parent)
            self._harden_database_family()
        if self.backup_manager is None:
            return
        await self._run_blocking(self.backup_manager.harden_existing_backups)
        if self.backup_manager.backup_is_invalidated and not Path(self.path).is_file():
            raise SchemaRecoveryRequired(
                SchemaRecoveryReason.CORRUPT_DATABASE,
                "the primary database is missing and its recovery backup is invalidated",
                database_path=self.path,
            )

    async def _restore_explicit_backup_if_needed(self) -> None:
        if self.backup_manager is None or not self._restore_explicit_backup:
            return
        try:
            await self._run_blocking(self.backup_manager.restore_if_missing)
        except sqlite3.DatabaseError as exc:
            raise SchemaRecoveryRequired(
                SchemaRecoveryReason.CORRUPT_DATABASE,
                "the managed backup could not be restored safely",
                database_path=self.path,
            ) from exc

    async def _open_connection_for_lifecycle(self) -> tuple[sqlite3.Connection, bool]:
        try:
            return await self._run_blocking(
                self._open_initialized_connection,
                cancelled_result_disposer=lambda result: result[0].close(),
            )
        except SchemaRecoveryRequired as exc:
            if not exc.database_path:
                exc.database_path = self.path
            raise
        except sqlite3.DatabaseError as exc:
            if self.path == ":memory:":
                raise
            raise SchemaRecoveryRequired(
                SchemaRecoveryReason.CORRUPT_DATABASE,
                "the SQLite database could not be inspected safely",
                database_path=self.path,
            ) from exc

    async def _publish_open_backup_or_close(self, changed: bool) -> SqliteEngine:
        try:
            if self._managed_backup_needs_publication(changed):
                await self.publish_backup_after_commit(operation="database_open")
            return self
        except BaseException as open_error:
            try:
                await self._close_locked()
            except BaseException as close_error:
                raise open_error from close_error
            raise

    def _open_initialized_connection(self) -> tuple[sqlite3.Connection, bool]:
        if self._write_fence is None:
            result = self._open_initialized_connection_locked()
        else:
            with self._write_fence.hold():
                result = self._open_initialized_connection_locked()
        self._harden_database_family()
        return result

    def _open_initialized_connection_locked(self) -> tuple[sqlite3.Connection, bool]:
        connection: sqlite3.Connection | None = self._connect()
        connection.row_factory = sqlite3.Row
        try:
            self._configure_open_connection(connection)
            self._require_registered_schema_definition()
            require_integrity(connection)
            changed = self._initialize_schema_state(connection)
            connection.execute("PRAGMA journal_mode=WAL")
            return connection, changed
        except BaseException:
            if connection is not None:
                connection.close()
            raise

    @staticmethod
    def _configure_open_connection(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA secure_delete=ON")

    @staticmethod
    def _require_registered_schema_definition() -> None:
        try:
            require_current_schema_definition()
        except RuntimeError as exc:
            raise SchemaRecoveryRequired(
                SchemaRecoveryReason.MIGRATION_FAILED,
                "this SoulCore build has an unregistered SQLite definition",
            ) from exc

    def _initialize_schema_state(self, connection: sqlite3.Connection) -> bool:
        if database_is_empty(connection):
            create_current_schema(connection)
            return True
        if database_uses_current_schema(connection) and self._foreign_keys_are_valid(connection):
            return False
        version = read_schema_version(connection)
        if version is not None and version > CURRENT_SCHEMA_VERSION:
            raise SchemaRecoveryRequired(
                SchemaRecoveryReason.NEWER_SCHEMA,
                "database schema version is newer than this SoulCore build",
            )
        if not self._foreign_keys_are_valid(connection):
            raise SchemaRecoveryRequired(
                SchemaRecoveryReason.CORRUPT_DATABASE,
                "database foreign-key validation failed",
            )
        identity = read_schema_identity(connection)
        registered = schema_identity_for_version(version) if version is not None else None
        if (
            identity is not None
            and registered == identity
            and identity.version < CURRENT_SCHEMA_VERSION
            and database_uses_schema_identity(connection, identity)
        ):
            self._migrate_registered_schema(connection, identity)
            return True
        raise SchemaRecoveryRequired(
            SchemaRecoveryReason.STRUCTURE_MISMATCH,
            "database structure does not exactly match the SoulCore 1.x schema",
        )

    def _migrate_registered_schema(self, connection: sqlite3.Connection, identity: Any) -> None:
        manager = self._migration_backup_manager()
        try:
            manager.backup_before_migration(connection)
        except BaseException as exc:
            raise SchemaRecoveryRequired(
                SchemaRecoveryReason.MIGRATION_FAILED,
                "the pre-migration database snapshot could not be created",
            ) from exc
        migrate_to_current(connection, identity)
        if not database_uses_current_schema(connection) or not self._foreign_keys_are_valid(
            connection
        ):
            raise SchemaRecoveryRequired(
                SchemaRecoveryReason.MIGRATION_FAILED,
                "the migrated database did not pass final validation",
            )

    def _migration_backup_manager(self) -> SQLiteBackupManager:
        manager = self.backup_manager
        if manager is not None:
            return manager
        database = Path(self.path)
        return SQLiteBackupManager(
            database,
            database.with_name(f"{database.name}.backup"),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=5.0,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            self._harden_database_family()
        except BaseException:
            connection.close()
            raise
        return connection

    def _harden_database_family(self) -> None:
        if self.path == ":memory:":
            return
        database = Path(self.path)
        ensure_private_directory(database.parent)
        for path in (
            database,
            Path(f"{database}-wal"),
            Path(f"{database}-shm"),
            Path(f"{database}-journal"),
        ):
            try:
                stat = path.stat()
            except FileNotFoundError:
                self._secured_database_files.pop(path, None)
                continue
            if not path.is_file():
                raise OSError("managed SQLite family path is not a file")
            identity = (int(stat.st_dev), int(stat.st_ino))
            if self._secured_database_files.get(path) == identity:
                continue
            restrict_private_path(path, directory=False)
            secured = path.stat()
            self._secured_database_files[path] = (int(secured.st_dev), int(secured.st_ino))

    async def close(self) -> None:
        async with self._lifecycle_lock:
            await self._close_locked()

    async def _close_locked(self) -> None:
        async with self._lock:
            connection = self._connection
            if connection is None:
                return
            close_completed = threading.Event()

            def close_connection() -> None:
                connection.close()
                close_completed.set()

            try:
                await self._run_blocking(close_connection)
            finally:
                if close_completed.is_set():
                    self._connection = None

    async def backup(self) -> Path | None:
        """Create an atomic, consistent snapshot of the open database."""

        if self.backup_manager is None:
            return None
        return await self.call(self.backup_manager.backup_connection)

    async def publish_backup_after_commit(
        self,
        *,
        operation: str = "database_commit",
    ) -> Path | None:
        """Best-effort publication after a transaction is already committed.

        A backup failure cannot roll back the committed transaction, so this
        method invalidates the older recovery generation, emits an operational
        warning, and preserves the successful application result.
        """

        try:
            path = await self.backup()
        except asyncio.CancelledError as cancellation:
            (
                invalidation_error_type,
                _cleanup_cancellation,
            ) = await self._invalidate_backup_uncancellable()
            self._last_backup_warning = BackupPublicationWarning(
                occurred_at=datetime.now(UTC).isoformat(),
                operation=str(operation or "database_commit"),
                error_type=type(cancellation).__name__,
                invalidation_error_type=invalidation_error_type,
            )
            if invalidation_error_type:
                cancellation.add_note(
                    "managed backup invalidation also failed: " + invalidation_error_type
                )
            logger.warning(
                "committed database write kept, backup publication was cancelled and invalidated",
                extra={
                    "backup_operation": operation,
                    "backup_invalidated": self._backup_is_safely_invalidated(),
                },
            )
            raise
        except Exception as exc:
            (
                invalidation_error_type,
                cleanup_cancellation,
            ) = await self._invalidate_backup_uncancellable()
            self._last_backup_warning = BackupPublicationWarning(
                occurred_at=datetime.now(UTC).isoformat(),
                operation=str(operation or "database_commit"),
                error_type=type(exc).__name__,
                invalidation_error_type=invalidation_error_type,
            )
            logger.exception(
                "committed database write kept, managed backup publication failed",
                extra={
                    "backup_operation": operation,
                    "backup_invalidated": self._backup_is_safely_invalidated(),
                },
            )
            if cleanup_cancellation is not None:
                raise cleanup_cancellation from exc
            return None
        self._last_backup_warning = None
        return path

    async def invalidate_backup(self) -> None:
        if self.backup_manager is not None:
            await self._run_blocking(self.backup_manager.remove_backup)

    async def _invalidate_backup_uncancellable(
        self,
    ) -> tuple[str, asyncio.CancelledError | None]:
        task = asyncio.create_task(self.invalidate_backup())
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
        try:
            task.result()
        except BaseException as exc:
            logger.exception("managed backup invalidation failed")
            return type(exc).__name__, cancellation
        return "", cancellation

    def _backup_is_safely_invalidated(self) -> bool:
        manager = self.backup_manager
        return manager is None or manager.backup_is_invalidated

    def _managed_backup_needs_publication(self, schema_changed: bool) -> bool:
        manager = self.backup_manager
        if manager is None:
            return False
        if schema_changed:
            return True
        if manager.backup_is_invalidated:
            return True
        artifacts = manager.file_artifacts
        return artifacts is not None and not artifacts.has_database_generation(manager.backup_path)

    @property
    def last_backup_warning(self) -> BackupPublicationWarning | None:
        return self._last_backup_warning

    async def __aenter__(self) -> SqliteEngine:
        return await self.open()

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("Database is not open")
        return self._connection

    @staticmethod
    def _foreign_keys_are_valid(connection: sqlite3.Connection) -> bool:
        return connection.execute("PRAGMA foreign_key_check").fetchone() is None

    async def call(
        self,
        operation: Callable[[sqlite3.Connection], T],
        *,
        transaction: bool = False,
    ) -> T:
        if transaction and _ACTIVE_TRANSACTION.get() == id(self):
            raise RuntimeError(
                "SQLite transactions are non-reentrant; execute all SQL in one callback"
            )
        async with self._lock:
            connection = self._require_connection()

            def invoke() -> T:
                if not transaction:
                    return self._invoke_read_only(connection, operation)
                token = _ACTIVE_TRANSACTION.set(id(self))
                try:
                    if self._write_fence is None:
                        return self._invoke_transaction(connection, operation)
                    with self._write_fence.hold():
                        return self._invoke_transaction(connection, operation)
                finally:
                    _ACTIVE_TRANSACTION.reset(token)
                    self._harden_database_family()

            return await self._run_blocking(invoke)

    @staticmethod
    def _invoke_read_only(
        connection: sqlite3.Connection, operation: Callable[[sqlite3.Connection], T]
    ) -> T:
        def deny_writes(
            action: int,
            arg1: str | None,
            _arg2: str | None,
            _database: str | None,
            _trigger: str | None,
        ) -> int:
            if action == sqlite3.SQLITE_PRAGMA:
                return (
                    sqlite3.SQLITE_OK
                    if arg1 is not None and arg1.lower() in _READ_ONLY_PRAGMAS
                    else sqlite3.SQLITE_DENY
                )
            return sqlite3.SQLITE_DENY if action in _WRITE_AUTHORIZER_ACTIONS else sqlite3.SQLITE_OK

        connection.set_authorizer(deny_writes)
        try:
            return operation(connection)
        finally:
            connection.set_authorizer(None)

    @staticmethod
    def _invoke_transaction(
        connection: sqlite3.Connection, operation: Callable[[sqlite3.Connection], T]
    ) -> T:
        connection.execute("BEGIN IMMEDIATE")
        try:
            result = operation(connection)
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise

    @staticmethod
    async def _run_blocking(
        operation: Callable[[], T],
        *,
        cancelled_result_disposer: Callable[[T], None] | None = None,
    ) -> T:
        """Keep ownership until a submitted thread finishes, even after cancellation."""

        worker = asyncio.create_task(asyncio.to_thread(operation))
        cancellation: asyncio.CancelledError | None = None
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
            except BaseException:
                # Inspect the completed worker below so a previously observed
                # caller cancellation remains the primary outcome.
                break
        try:
            result = worker.result()
        except BaseException as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
        if cancellation is not None:
            if cancelled_result_disposer is not None:
                disposer = asyncio.create_task(asyncio.to_thread(cancelled_result_disposer, result))
                while not disposer.done():
                    try:
                        await asyncio.shield(disposer)
                    except asyncio.CancelledError:
                        continue
                try:
                    disposer.result()
                except BaseException as exc:
                    raise cancellation from exc
            raise cancellation
        return result

    async def fetch_all(self, sql: str, parameters: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return await self.call(lambda conn: list(conn.execute(sql, parameters)))

    async def fetch_one(self, sql: str, parameters: Sequence[Any] = ()) -> sqlite3.Row | None:
        return await self.call(lambda conn: conn.execute(sql, parameters).fetchone())


__all__ = ["SqliteEngine"]
