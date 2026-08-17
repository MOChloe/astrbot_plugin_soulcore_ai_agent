from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
import zipfile
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ...shared.private_paths import (
    create_private_file,
    ensure_private_directory,
    restrict_private_path,
    restrict_private_tree,
    sync_directory,
)
from .artifact_backup import FileArtifactBackupStore, PreparedArtifactRestore
from .write_fence import SQLiteWriteFence


def infer_backup_path(database_path: str | Path) -> Path | None:
    """Return the stable data-root backup path for a plugin_data database."""
    path = Path(database_path)
    if str(path) == ":memory:":
        return None
    resolved = path.resolve(strict=False)
    parts = resolved.parts
    indexes = [index for index, part in enumerate(parts) if part.lower() == "plugin_data"]
    if not indexes:
        return None
    index = indexes[-1]
    if index + 1 >= len(parts):
        return None
    data_root = Path(*parts[:index])
    plugin_name = parts[index + 1]
    return data_root / "soulcore_backups" / plugin_name / resolved.name


class SQLiteBackupManager:
    """Create and restore atomic SQLite backups without copying WAL files."""

    def __init__(
        self,
        database_path: str | Path,
        backup_path: str | Path,
        *,
        file_artifact_root: str | Path | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.backup_path = Path(backup_path)
        self.write_fence = SQLiteWriteFence(self.database_path)
        self.file_artifacts = (
            FileArtifactBackupStore(
                live_root=file_artifact_root,
                backup_path=self.backup_path,
            )
            if file_artifact_root is not None
            else None
        )

    @property
    def migration_snapshot_path(self) -> Path:
        """Return the retained snapshot taken immediately before migration."""

        return self.backup_path.with_name(f"{self.database_path.stem}.pre-migration.sqlite3")

    @property
    def invalidation_marker_path(self) -> Path:
        """Return the fail-closed marker that makes a stale backup unrestorable."""

        return self.backup_path.with_name(f".{self.backup_path.name}.invalidated")

    @property
    def backup_is_invalidated(self) -> bool:
        return self.invalidation_marker_path.is_file()

    def harden_existing_backups(self) -> None:
        """Secure managed backup material left by an earlier plugin build."""

        parent = self.backup_path.parent
        if parent.exists():
            ensure_private_directory(parent)
            temporary_prefixes = (
                f".{self.backup_path.name}.",
                f".{self.migration_snapshot_path.name}.",
                f".{self.invalidation_marker_path.name}.",
            )
            for path in parent.iterdir():
                if not path.name.endswith(".tmp") or not path.name.startswith(temporary_prefixes):
                    continue
                self._reject_link(path)
                if not path.is_file():
                    raise OSError("managed backup temporary path is not a file")
                restrict_private_path(path, directory=False)
        for path in (
            self.backup_path,
            self.migration_snapshot_path,
            self.invalidation_marker_path,
        ):
            if not path.exists():
                continue
            self._reject_link(path)
            if not path.is_file():
                raise OSError("managed backup path is not a file")
            restrict_private_path(path, directory=False)
        recovery_dir = parent / "recovery"
        if recovery_dir.exists():
            restrict_private_tree(recovery_dir)
        if self.file_artifacts is not None:
            self.file_artifacts.harden_existing()

    def restore_if_missing(self) -> bool:
        """Restore the primary database only when it is absent."""
        with self.write_fence.hold():
            self._recover_restore_quarantine()
            if self.database_path.exists():
                return False
            if self.backup_is_invalidated:
                raise sqlite3.DatabaseError(
                    "managed backup was invalidated after a committed write"
                )
            if not self.backup_path.is_file():
                return False
            restrict_private_path(self.backup_path, directory=False)
            ensure_private_directory(self.database_path.parent)
            temporary = self._temporary_path(self.database_path)
            create_private_file(temporary)
            source: sqlite3.Connection | None = None
            destination: sqlite3.Connection | None = None
            prepared_artifacts: PreparedArtifactRestore | None = None
            try:
                source = sqlite3.connect(str(self.backup_path), timeout=5.0)
                source.execute("PRAGMA query_only=ON")
                destination = sqlite3.connect(str(temporary), timeout=5.0)
                source.backup(destination)
                self._verify(destination)
                destination.close()
                destination = None
                source.close()
                source = None
                if self.file_artifacts is not None:
                    prepared_artifacts = self.file_artifacts.prepare_restore(temporary)
                self._publish_restored_generation(temporary, prepared_artifacts)
                return True
            finally:
                if destination is not None:
                    destination.close()
                if source is not None:
                    source.close()
                if self.file_artifacts is not None:
                    self.file_artifacts.discard_prepared(prepared_artifacts)
                self._discard_temporary(temporary)

    def _publish_restored_generation(
        self,
        temporary: Path,
        prepared_artifacts: PreparedArtifactRestore | None,
    ) -> None:
        if self.file_artifacts is None or prepared_artifacts is None:
            self._publish_restored_primary(temporary)
            return
        quarantine: Path | None = None
        artifacts_published = False
        try:
            quarantine = self.file_artifacts.publish_restore(prepared_artifacts)
            artifacts_published = True
            self._publish_restored_primary(temporary)
        except BaseException as publish_error:
            if not artifacts_published:
                raise
            try:
                self.file_artifacts.rollback_restore(quarantine)
            except BaseException as rollback_error:
                raise BaseExceptionGroup(
                    "managed database and file artifact restore rollback failed",
                    [publish_error, rollback_error],
                ) from publish_error
            raise
        else:
            self.file_artifacts.finish_restore(quarantine)

    def _publish_restored_primary(self, temporary: Path) -> None:
        """Publish a verified snapshot without ever pairing it with stale sidecars."""

        quarantined: list[tuple[Path, Path]] = []
        try:
            for sidecar in self._sidecar_paths():
                if not sidecar.exists():
                    continue
                quarantine = self._restore_quarantine_path(sidecar)
                os.replace(sidecar, quarantine)
                quarantined.append((sidecar, quarantine))
            self._replace_temporary(temporary, self.database_path)
        except Exception as publish_error:
            rollback_errors: list[Exception] = []
            for sidecar, quarantine in reversed(quarantined):
                if quarantine.exists():
                    try:
                        os.replace(quarantine, sidecar)
                    except Exception as rollback_error:
                        rollback_errors.append(rollback_error)
            if rollback_errors:
                raise ExceptionGroup(
                    "managed backup publication and sidecar rollback both failed",
                    [publish_error, *rollback_errors],
                ) from publish_error
            raise
        else:
            for _sidecar, quarantine in quarantined:
                self._discard_temporary(quarantine)

    def _recover_restore_quarantine(self) -> None:
        if self.file_artifacts is not None:
            self.file_artifacts.recover_restore_quarantine(
                database_exists=self.database_path.exists()
            )
        for sidecar in self._sidecar_paths():
            quarantine = self._restore_quarantine_path(sidecar)
            if not quarantine.exists():
                continue
            if self.database_path.exists():
                self._discard_temporary(quarantine)
                continue
            os.replace(quarantine, sidecar)

    @staticmethod
    def _restore_quarantine_path(sidecar: Path) -> Path:
        return sidecar.with_name(f".{sidecar.name}.restore-quarantine")

    def backup_connection(self, source: sqlite3.Connection) -> Path:
        """Atomically publish a consistent snapshot of an open connection."""

        with self.write_fence.hold():
            return self._backup_connection_to(
                source,
                self.backup_path,
                include_file_artifacts=True,
            )

    def backup_before_migration(self, source: sqlite3.Connection) -> Path:
        """Publish a verified snapshot while the caller owns the write fence."""

        return self._backup_connection_to(source, self.migration_snapshot_path)

    def _backup_connection_to(
        self,
        source: sqlite3.Connection,
        target: Path,
        *,
        include_file_artifacts: bool = False,
    ) -> Path:
        ensure_private_directory(target.parent)
        temporary = self._temporary_path(target)
        create_private_file(temporary)
        destination: sqlite3.Connection | None = None
        try:
            destination = sqlite3.connect(str(temporary), timeout=5.0)
            source.backup(destination)
            self._verify(destination)
            destination.close()
            destination = None
            if include_file_artifacts and self.file_artifacts is not None:
                self.file_artifacts.snapshot(temporary)
            self._replace_temporary(temporary, target)
            if include_file_artifacts and self.file_artifacts is not None:
                self.file_artifacts.prune(target)
            if target == self.backup_path:
                self._clear_invalidation_marker()
            return target
        finally:
            if destination is not None:
                destination.close()
            self._discard_temporary(temporary)

    def remove_backup(self) -> None:
        """Invalidate first, then remove a stale snapshot when the OS permits."""

        self._publish_invalidation_marker()
        with suppress(FileNotFoundError):
            self.backup_path.unlink()

    def clear_primary_and_backup(self) -> None:
        """Remove the database family and every backup generation coupled to it."""

        for path in (
            self.database_path,
            Path(f"{self.database_path}-wal"),
            Path(f"{self.database_path}-shm"),
            self.backup_path,
            self.migration_snapshot_path,
            self.invalidation_marker_path,
        ):
            with suppress(FileNotFoundError):
                path.unlink()
        if self.file_artifacts is not None:
            self._remove_managed_tree(self.file_artifacts.root)

    def _sidecar_paths(self) -> tuple[Path, Path]:
        return Path(f"{self.database_path}-wal"), Path(f"{self.database_path}-shm")

    def archive_database_family(
        self,
        extra_paths: Iterable[tuple[Path, str]] = (),
    ) -> RecoveryArchive:
        """Create a hashed raw archive before an administrator clears data."""

        ensure_private_directory(self.backup_path.parent)
        recovery_dir = ensure_private_directory(self.backup_path.parent / "recovery")
        restrict_private_tree(recovery_dir)
        created_at = datetime.now(UTC)
        stamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
        target = recovery_dir / f"{self.database_path.stem}-recovery-{stamp}.zip"
        temporary = self._temporary_path(target)
        candidates = (
            (self.database_path, self.database_path.name),
            (Path(f"{self.database_path}-wal"), f"{self.database_path.name}-wal"),
            (Path(f"{self.database_path}-shm"), f"{self.database_path.name}-shm"),
            (self.backup_path, f"managed-backup-{self.backup_path.name}"),
            (
                self.invalidation_marker_path,
                f"managed-backup-{self.backup_path.name}.invalidated",
            ),
        )
        existing = [(path, name) for path, name in candidates if path.is_file()]
        extra_members = self._archive_extra_members(extra_paths)
        if not existing and not extra_members:
            raise FileNotFoundError("there is no SoulCore database family to archive")
        try:
            create_private_file(temporary)
            with zipfile.ZipFile(
                temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
            ) as archive:
                manifest = []
                for path, name in existing:
                    archive.write(path, arcname=name)
                    manifest.append(
                        {
                            "name": name,
                            "size": path.stat().st_size,
                            "sha256": self._file_sha256(path),
                        }
                    )
                for path, name in extra_members:
                    archive.write(path, arcname=name)
                    manifest.append(
                        {
                            "name": name,
                            "size": path.stat().st_size,
                            "sha256": self._file_sha256(path),
                        }
                    )
                archive.writestr(
                    "RECOVERY_MANIFEST.json",
                    json.dumps(
                        {
                            "format": "soulcore-sqlite-recovery-v1",
                            "created_at": created_at.isoformat(),
                            "files": manifest,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if temporary.stat().st_size <= 0:
                raise OSError("recovery archive verification failed")
            self._replace_temporary(temporary, target)
            return RecoveryArchive(target, digest)
        finally:
            self._discard_temporary(temporary)

    @classmethod
    def _archive_extra_members(
        cls,
        extra_paths: Iterable[tuple[Path, str]],
    ) -> list[tuple[Path, str]]:
        members: list[tuple[Path, str]] = []
        names: set[str] = set()
        for raw_path, raw_prefix in extra_paths:
            path = Path(raw_path)
            prefix = str(raw_prefix or "").strip().strip("/").replace("\\", "/")
            if not prefix or prefix.startswith("../") or "/../" in f"/{prefix}/":
                raise ValueError("invalid recovery archive prefix")
            cls._reject_link(path)
            if path.is_file():
                candidates = ((path, prefix),)
            elif path.is_dir():
                candidates = tuple(
                    (item, f"{prefix}/{item.relative_to(path).as_posix()}")
                    for item in sorted(path.rglob("*"))
                    if item.is_file()
                )
            else:
                continue
            for item, name in candidates:
                cls._reject_link(item)
                if name in names:
                    raise ValueError("duplicate recovery archive member")
                names.add(name)
                members.append((item, name))
        return members

    @staticmethod
    def _reject_link(path: Path) -> None:
        is_junction = getattr(path, "is_junction", lambda: False)
        if path.is_symlink() or is_junction():
            raise OSError("managed recovery path must not be a link or junction")

    @classmethod
    def _remove_managed_tree(cls, path: Path) -> None:
        if not path.exists():
            return
        cls._reject_link(path)
        if not path.is_dir():
            raise OSError("managed recovery path is not a directory")
        restrict_private_tree(path)
        shutil.rmtree(path)

    @staticmethod
    def _temporary_path(path: Path) -> Path:
        return path.with_name(f".{path.name}.{uuid4().hex}.tmp")

    @staticmethod
    def _discard_temporary(path: Path) -> None:
        # A failed atomic publication must keep its original exception. A
        # scanner may still hold the disposable file for a moment on Windows;
        # the unique name prevents it from blocking a later backup attempt.
        with suppress(OSError):
            path.unlink()

    @staticmethod
    def _replace_temporary(temporary: Path, target: Path) -> None:
        """Publish a closed file despite short-lived Windows sharing locks."""

        restrict_private_path(temporary, directory=False)
        ensure_private_directory(target.parent)
        retry_delays = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0)
        for attempt, delay in enumerate(retry_delays):
            if delay:
                time.sleep(delay)
            try:
                os.replace(temporary, target)
                restrict_private_path(target, directory=False)
                sync_directory(target.parent)
                return
            except PermissionError:
                if attempt == len(retry_delays) - 1:
                    raise

    def _publish_invalidation_marker(self) -> None:
        marker = self.invalidation_marker_path
        ensure_private_directory(marker.parent)
        temporary = self._temporary_path(marker)
        try:
            create_private_file(temporary)
            temporary.write_text(
                json.dumps(
                    {
                        "format": "soulcore-managed-backup-invalidation-v1",
                        "invalidated_at": datetime.now(UTC).isoformat(),
                    },
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            self._replace_temporary(temporary, marker)
        finally:
            self._discard_temporary(temporary)

    def _clear_invalidation_marker(self) -> None:
        try:
            self.invalidation_marker_path.unlink()
        except FileNotFoundError:
            return
        sync_directory(self.invalidation_marker_path.parent)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _verify(connection: sqlite3.Connection) -> None:
        row = connection.execute("PRAGMA integrity_check").fetchone()
        if row is None or str(row[0]).lower() != "ok":
            raise sqlite3.DatabaseError(f"backup integrity check failed: {row!r}")


@dataclass(frozen=True, slots=True)
class RecoveryArchive:
    path: Path
    sha256: str


__all__ = ["RecoveryArchive", "SQLiteBackupManager", "infer_backup_path"]
