"""Restore one selected recovery point while retaining the pre-restore state."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ...shared.private_paths import create_private_directory
from .backup import RecoveryArchive, SQLiteBackupManager
from .recovery_publish import (
    JOURNAL_NAME,
    STAGE_PREFIX,
    publish_restore,
    recover_interrupted_restore,
    stage_path,
)
from .recovery_sources import RecoverySource, archive_payloads, prepare_source


def restore_source(manager: SQLiteBackupManager, source: RecoverySource) -> RecoveryArchive | None:
    """Caller must exclude running applications and hold manager.write_fence."""
    recover_interrupted_restore(manager.database_path)
    stage = stage_path(manager.database_path, f"{STAGE_PREFIX}{uuid4().hex}")
    create_private_directory(stage)
    try:
        replacements = prepare_source(manager, source, stage)
        payloads = archive_payloads(manager)
        family = (
            manager.database_path,
            manager.backup_path,
            manager.invalidation_marker_path,
            *(Path(f"{manager.database_path}{suffix}") for suffix in ("-wal", "-shm", "-journal")),
            *(path for path, _ in payloads),
        )
        has_current_data = any(
            path.is_file() or (path.is_dir() and any(path.rglob("*"))) for path in family
        )
        preserved = manager.archive_database_family(payloads) if has_current_data else None
        publish_restore(manager.database_path, stage, replacements)
        return preserved
    finally:
        # A failed rollback retains its journal and files for next startup.
        if not (manager.database_path.parent / JOURNAL_NAME).exists():
            manager._remove_managed_tree(stage)
