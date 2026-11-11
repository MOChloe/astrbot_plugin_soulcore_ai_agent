"""Discover retained recovery points and prepare them without touching live data."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import zipfile
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ...shared.private_paths import ensure_private_directory, restrict_private_tree
from .artifact_backup import FileArtifactBackupStore
from .backup import SQLiteBackupManager
from .recovery_archive import archive_members, copy_private, extract_verified
from .recovery_publish import payload_paths
from .schema.current import (
    CURRENT_SCHEMA_VERSION,
    database_uses_current_schema,
    database_uses_schema_identity,
    read_schema_identity,
    require_integrity,
    schema_identity_for_version,
)
from .schema.migrations import migrate_to_current


@dataclass(frozen=True)
class RecoverySource:
    path: Path
    member: str
    label: str
    stale: bool = False

    @property
    def id(self) -> str:
        key = f"{self.path.name}\n{self.member}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]

    def public_view(self) -> dict:
        stat = self.path.stat()
        return {
            "id": self.id,
            "name": self.path.name,
            "label": self.label,
            "created_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            "bytes": stat.st_size,
            "stale": self.stale,
        }


def list_sources(manager: SQLiteBackupManager) -> list[RecoverySource]:
    sources = []
    for path, label, stale in (
        (manager.backup_path, "滚动备份", manager.backup_is_invalidated),
        (manager.migration_snapshot_path, "升级前快照", False),
    ):
        manager._reject_link(path)
        if path.is_file():
            sources.append(RecoverySource(path, "", label, stale))
    recovery_dir = manager.backup_path.parent / "recovery"
    manager._reject_link(recovery_dir)
    archives = sorted(
        recovery_dir.glob(f"{manager.database_path.stem}-recovery-*.zip"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in archives[:30]:
        manager._reject_link(path)
        try:
            with zipfile.ZipFile(path) as archive:
                members = archive_members(archive)
        except (OSError, ValueError, zipfile.BadZipFile):
            continue
        variants = (
            (manager.database_path.name, "恢复包中的原数据库", False),
            (
                f"managed-backup-{manager.backup_path.name}",
                "恢复包中的滚动备份",
                f"managed-backup-{manager.backup_path.name}.invalidated" in members,
            ),
            ("state/pre-migration.sqlite3", "恢复包中的升级前快照", False),
        )
        for member, label, stale in variants:
            if member in members:
                sources.append(RecoverySource(path, member, label, stale))
    return sources


def archive_payloads(manager: SQLiteBackupManager) -> tuple[tuple[Path, str], ...]:
    paths = [(path, f"state/{name}") for name, path in payload_paths(manager.database_path).items()]
    paths.append((manager.migration_snapshot_path, "state/pre-migration.sqlite3"))
    if manager.file_artifacts is not None:
        paths.append((manager.file_artifacts.root, "state/managed-backup-artifacts"))
    return tuple(paths)


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    restrict_private_tree(source)
    ensure_private_directory(destination.parent)
    shutil.copytree(source, destination)
    restrict_private_tree(destination)


def _normalized_database(source: Path, destination: Path) -> None:
    # source is always a private staging copy, including any original WAL.
    # An empty or unknown database is never accepted as a recovery snapshot.
    with closing(sqlite3.connect(source)) as incoming:
        require_integrity(incoming)
        identity = read_schema_identity(incoming)
        if (
            identity is None
            or schema_identity_for_version(identity.version) != identity
            or not database_uses_schema_identity(incoming, identity)
            or identity.version > CURRENT_SCHEMA_VERSION
        ):
            raise ValueError("备份结构无法由当前版本安全还原；请保留文件并使用匹配的 SoulCore 版本")
        if incoming.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("备份的关联数据校验失败，原数据未被替换")
        from ...shared.private_paths import create_private_file

        create_private_file(destination)
        with closing(sqlite3.connect(destination, isolation_level=None)) as outgoing:
            outgoing.row_factory = sqlite3.Row
            incoming.backup(outgoing)
            outgoing.execute("PRAGMA foreign_keys=ON")
            if identity.version < CURRENT_SCHEMA_VERSION:
                migrate_to_current(outgoing, identity)
            require_integrity(outgoing)
            if (
                not database_uses_current_schema(outgoing)
                or outgoing.execute("PRAGMA foreign_key_check").fetchone() is not None
            ):
                raise ValueError("备份升级后校验失败，原数据未被替换")
            outgoing.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            outgoing.execute("PRAGMA journal_mode=DELETE")


def _restore_artifacts(store: FileArtifactBackupStore, snapshot: Path) -> None:
    prepared = store.prepare_restore(snapshot)
    if prepared is not None:
        quarantine = store.publish_restore(prepared)
        store.finish_restore(quarantine)


def prepare_source(
    manager: SQLiteBackupManager,
    source: RecoverySource,
    stage: Path,
) -> dict[Path, Path | None]:
    database = manager.database_path
    replacements: dict[Path, Path | None] = {}
    staged_files = stage / "file_artifacts"
    if source.member:
        unpacked = ensure_private_directory(stage / "archive")
        extract_verified(source.path, unpacked)
        snapshot = unpacked.joinpath(*source.member.split("/"))
        for name, target in payload_paths(database).items():
            archived = unpacked / "state" / name
            replacements[target] = archived if archived.exists() else None
        archived_files = unpacked / "state" / "file_artifacts"
        if source.member.startswith("managed-backup-"):
            archive_store = unpacked / "state" / "managed-backup-artifacts"
            if archive_store.exists():
                store = FileArtifactBackupStore(
                    live_root=staged_files, backup_path=stage / "managed.sqlite3"
                )
                archive_store.rename(store.root)
                _restore_artifacts(store, snapshot)
            else:
                _copy_tree(archived_files, staged_files)
        else:
            _copy_tree(archived_files, staged_files)
    else:
        snapshot = stage / "snapshot.sqlite3"
        copy_private(source.path, snapshot)
        if source.path == manager.backup_path and manager.file_artifacts is not None:
            store = FileArtifactBackupStore(live_root=staged_files, backup_path=manager.backup_path)
            if store.has_database_generation(snapshot):
                _restore_artifacts(store, snapshot)
            else:
                _copy_tree(database.parent / "file_artifacts", staged_files)
        else:
            _copy_tree(database.parent / "file_artifacts", staged_files)
    normalized = stage / "ready.sqlite3"
    _normalized_database(snapshot, normalized)
    # Validate every AVAILABLE file against its database identity even for old
    # recovery ZIPs without an artifact manifest. Missing required files stop
    # the restore; an apparently healthy database alone is insufficient.
    validator = FileArtifactBackupStore(
        live_root=staged_files, backup_path=stage / "validation.sqlite3"
    )
    validator.snapshot(normalized)
    ensure_private_directory(staged_files)
    replacements[database.parent / "file_artifacts"] = staged_files
    for suffix in ("-wal", "-shm", "-journal"):
        replacements[Path(f"{database}{suffix}")] = None
    replacements[database] = normalized
    return replacements
