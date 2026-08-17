from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from ...shared.private_paths import (
    create_private_directory,
    create_private_file,
    ensure_private_directory,
    restrict_private_path,
    restrict_private_tree,
    sync_directory,
)

_MANIFEST_FORMAT = "soulcore-file-artifact-backup-v1"
_CAPTURED_STATUSES = frozenset({"AVAILABLE", "RELEASE_PENDING"})
_REQUIRED_STATUSES = frozenset({"AVAILABLE"})
_SHA256_LENGTH = 64


def _filesystem_path(path: Path) -> Path:
    """Use the Win32 extended-path namespace for deeply nested backup roots."""

    if os.name != "nt":
        return path
    raw = str(path)
    if raw.startswith("\\\\?\\"):
        return path
    if raw.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + raw[2:])
    return Path("\\\\?\\" + raw)


@dataclass(frozen=True, slots=True)
class PreparedArtifactRestore:
    root: Path


class FileArtifactBackupStore:
    """Bind immutable file artifacts to one exact managed SQLite snapshot."""

    def __init__(
        self,
        *,
        live_root: str | Path,
        backup_path: str | Path,
    ) -> None:
        self.live_root = _filesystem_path(Path(live_root).resolve(strict=False))
        backup = _filesystem_path(Path(backup_path).resolve(strict=False))
        self.root = backup.parent / f"{backup.name}.file-artifacts"
        self.blob_root = self.root / "blobs"
        self.manifest_root = self.root / "manifests"
        self._verified_blobs: dict[str, tuple[int, int]] = {}
        self._permissions_hardened = False

    @property
    def restore_quarantine(self) -> Path:
        return self.live_root.with_name(f".{self.live_root.name}.restore-quarantine")

    def harden_existing(self) -> None:
        """Tighten permissions on backup generations created by older builds."""

        if self.root.exists():
            self._harden_backup_tree()

    def has_database_generation(self, database_snapshot: Path) -> bool:
        if not database_snapshot.is_file():
            return False
        try:
            self._harden_backup_tree()
            database_sha256 = self._file_sha256(database_snapshot)
            manifest = self._manifest_path(database_sha256)
            self._read_manifest(manifest, database_sha256)
        except (OSError, sqlite3.DatabaseError):
            return False
        return True

    def snapshot(self, database_snapshot: Path) -> None:
        self._harden_backup_tree()
        rows = self._artifact_rows(database_snapshot)
        entries: list[dict[str, Any]] = []
        for row in rows:
            status = str(row["file_status"] or "")
            if status not in _CAPTURED_STATUSES:
                continue
            relative = self._controlled_relative_path(str(row["storage_relpath"] or ""))
            expected_size = int(row["byte_size"] or 0)
            expected_sha256 = self._normalized_sha256(row["sha256"])
            if not self._blob_is_verified(expected_size, expected_sha256):
                source = self._live_path(relative)
                if not source.is_file():
                    if status in _REQUIRED_STATUSES:
                        raise OSError(
                            f"available file artifact is missing from managed storage: {relative}"
                        )
                    continue
                if not self._verify_file(source, expected_size, expected_sha256):
                    raise OSError(
                        f"file artifact bytes do not match the database snapshot: {relative}"
                    )
                self._materialize_blob(source, expected_size, expected_sha256)
            entries.append(
                {
                    "storage_relpath": relative.as_posix(),
                    "byte_size": expected_size,
                    "sha256": expected_sha256,
                }
            )

        database_sha256 = self._file_sha256(database_snapshot)
        manifest = {
            "format": _MANIFEST_FORMAT,
            "database_sha256": database_sha256,
            "artifacts": sorted(entries, key=lambda item: item["storage_relpath"]),
        }
        self._publish_json(self._manifest_path(database_sha256), manifest)

    def prepare_restore(self, database_snapshot: Path) -> PreparedArtifactRestore | None:
        self._harden_backup_tree()
        database_sha256 = self._file_sha256(database_snapshot)
        manifest_path = self._manifest_path(database_sha256)
        if not manifest_path.is_file():
            raise sqlite3.DatabaseError(
                "managed database backup has no matching file-artifact manifest"
            )
        manifest = self._read_manifest(manifest_path, database_sha256)
        rows = self._artifact_rows(database_snapshot)
        expected = self._expected_rows(rows)
        entries = self._validated_entries(manifest, expected)
        temporary = self.live_root.with_name(f".{self.live_root.name}.{uuid4().hex}.restore")
        try:
            create_private_directory(temporary)
            for relative, entry in entries.items():
                target = temporary / relative
                ensure_private_directory(target.parent)
                create_private_file(target)
                blob = self._blob_path(entry["sha256"])
                shutil.copyfile(blob, target)
                if not self._verify_file(
                    target,
                    int(entry["byte_size"]),
                    str(entry["sha256"]),
                ):
                    raise OSError(
                        f"restored file artifact failed verification: {relative.as_posix()}"
                    )
            return PreparedArtifactRestore(temporary)
        except BaseException:
            self._discard_tree(temporary)
            raise

    def publish_restore(self, prepared: PreparedArtifactRestore) -> Path | None:
        temporary = prepared.root.resolve(strict=False)
        if temporary.parent != self.live_root.parent or not temporary.name.startswith(
            f".{self.live_root.name}."
        ):
            raise ValueError("file artifact restore directory escaped its managed parent")
        if self.restore_quarantine.exists():
            raise OSError("a previous file artifact restore quarantine is still present")
        quarantine: Path | None = None
        try:
            if self.live_root.exists():
                self._replace_path(self.live_root, self.restore_quarantine)
                quarantine = self.restore_quarantine
            self._replace_path(temporary, self.live_root)
            return quarantine
        except BaseException as publish_error:
            try:
                if quarantine is not None and quarantine.exists() and not self.live_root.exists():
                    self._replace_path(quarantine, self.live_root)
            except BaseException as rollback_error:
                raise BaseExceptionGroup(
                    "file artifact publication and rollback both failed",
                    [publish_error, rollback_error],
                ) from publish_error
            raise

    def rollback_restore(self, quarantine: Path | None) -> None:
        errors: list[Exception] = []
        if self.live_root.exists():
            try:
                self._discard_tree(self.live_root)
            except Exception as exc:
                errors.append(exc)
        if quarantine is not None and quarantine.exists():
            try:
                self._replace_path(quarantine, self.live_root)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup("file artifact restore rollback failed", errors)

    def finish_restore(self, quarantine: Path | None) -> None:
        if quarantine is not None:
            self._discard_tree(quarantine)

    def recover_restore_quarantine(self, *, database_exists: bool) -> None:
        quarantine = self.restore_quarantine
        if not quarantine.exists():
            return
        if database_exists:
            self._discard_tree(quarantine)
            return
        if self.live_root.exists():
            self._discard_tree(self.live_root)
        self._replace_path(quarantine, self.live_root)

    def prune(self, current_database_snapshot: Path) -> None:
        self._harden_backup_tree()
        current_sha256 = self._file_sha256(current_database_snapshot)
        current = self._manifest_path(current_sha256)
        if not current.is_file():
            raise sqlite3.DatabaseError("file artifact backup generation was not published")
        manifest = self._read_manifest(current, current_sha256)
        referenced = {str(entry["sha256"]) for entry in manifest["artifacts"]}
        manifests = tuple(self.manifest_root.glob("*.json"))
        for path in manifests:
            if path != current:
                with suppress(OSError):
                    path.unlink()
        if self.blob_root.is_dir():
            for path in self.blob_root.rglob("*"):
                if path.is_file() and path.name not in referenced:
                    with suppress(OSError):
                        path.unlink()
            self._discard_empty_directories(self.blob_root)
        self._verified_blobs = {
            digest: signature
            for digest, signature in self._verified_blobs.items()
            if digest in referenced
        }

    def discard_prepared(self, prepared: PreparedArtifactRestore | None) -> None:
        if prepared is not None:
            self._discard_tree(prepared.root)

    def _validated_entries(
        self,
        manifest: dict[str, Any],
        expected: dict[Path, sqlite3.Row],
    ) -> dict[Path, dict[str, Any]]:
        entries: dict[Path, dict[str, Any]] = {}
        for raw in manifest["artifacts"]:
            relative, sha256, byte_size = self._validated_manifest_entry(raw, expected)
            if relative in entries:
                raise sqlite3.DatabaseError("file artifact backup manifest contains duplicates")
            entries[relative] = {
                "sha256": sha256,
                "byte_size": byte_size,
            }
        missing_required = [
            relative
            for relative, row in expected.items()
            if str(row["file_status"] or "") in _REQUIRED_STATUSES and relative not in entries
        ]
        if missing_required:
            raise sqlite3.DatabaseError(
                "file artifact backup generation is incomplete for available artifacts"
            )
        return entries

    def _validated_manifest_entry(
        self,
        raw: object,
        expected: dict[Path, sqlite3.Row],
    ) -> tuple[Path, str, int]:
        if not isinstance(raw, dict):
            raise sqlite3.DatabaseError("file artifact backup manifest entry is invalid")
        relative = self._controlled_relative_path(str(raw.get("storage_relpath") or ""))
        sha256 = self._normalized_sha256(raw.get("sha256"))
        try:
            byte_size = int(raw.get("byte_size") or 0)
        except (TypeError, ValueError) as exc:
            raise sqlite3.DatabaseError(
                "file artifact backup manifest contains an invalid size"
            ) from exc
        row = expected.get(relative)
        if row is None:
            raise sqlite3.DatabaseError(
                "file artifact backup manifest is not part of its database generation"
            )
        expected_identity = (self._normalized_sha256(row["sha256"]), int(row["byte_size"] or 0))
        if (sha256, byte_size) != expected_identity:
            raise sqlite3.DatabaseError(
                "file artifact backup manifest disagrees with its database generation"
            )
        if not self._verify_file(self._blob_path(sha256), byte_size, sha256):
            raise sqlite3.DatabaseError("file artifact backup blob is missing or corrupted")
        return relative, sha256, byte_size

    def _expected_rows(self, rows: list[sqlite3.Row]) -> dict[Path, sqlite3.Row]:
        expected: dict[Path, sqlite3.Row] = {}
        for row in rows:
            if str(row["file_status"] or "") not in _CAPTURED_STATUSES:
                continue
            relative = self._controlled_relative_path(str(row["storage_relpath"] or ""))
            if relative in expected:
                raise sqlite3.DatabaseError("database contains duplicate file artifact paths")
            expected[relative] = row
        return expected

    def _artifact_rows(self, database_snapshot: Path) -> list[sqlite3.Row]:
        connection = sqlite3.connect(str(database_snapshot), timeout=5.0)
        connection.row_factory = sqlite3.Row
        try:
            return self._artifact_rows_from_connection(connection)
        finally:
            connection.close()

    @staticmethod
    def _artifact_rows_from_connection(connection: sqlite3.Connection) -> list[sqlite3.Row]:
        return list(
            connection.execute(
                """SELECT asset_id, storage_relpath, sha256, byte_size, file_status
                FROM file_assets ORDER BY asset_id"""
            )
        )

    def _materialize_blob(self, source: Path, byte_size: int, sha256: str) -> None:
        target = self._blob_path(sha256)
        if self._blob_is_verified(byte_size, sha256):
            return
        ensure_private_directory(target.parent)
        temporary = target.with_name(f".blob-{uuid4().hex}.tmp")
        try:
            create_private_file(temporary)
            shutil.copyfile(source, temporary)
            if not self._verify_file(temporary, byte_size, sha256):
                raise OSError("file artifact changed while its backup was being created")
            try:
                self._replace_path(temporary, target)
            except FileExistsError:
                if not self._verify_file(target, byte_size, sha256):
                    raise
            self._remember_verified_blob(target, sha256)
        finally:
            with suppress(OSError):
                temporary.unlink()

    def _blob_is_verified(self, byte_size: int, sha256: str) -> bool:
        target = self._blob_path(sha256)
        try:
            stat = target.stat()
        except OSError:
            self._verified_blobs.pop(sha256, None)
            return False
        signature = (stat.st_size, stat.st_mtime_ns)
        if stat.st_size != byte_size:
            self._verified_blobs.pop(sha256, None)
            return False
        if self._verified_blobs.get(sha256) == signature:
            return True
        if not self._verify_file(target, byte_size, sha256):
            self._verified_blobs.pop(sha256, None)
            return False
        self._verified_blobs[sha256] = signature
        return True

    def _remember_verified_blob(self, path: Path, sha256: str) -> None:
        stat = path.stat()
        self._verified_blobs[sha256] = (stat.st_size, stat.st_mtime_ns)

    def _read_manifest(self, path: Path, database_sha256: str) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise sqlite3.DatabaseError("file artifact backup manifest could not be read") from exc
        if not isinstance(value, dict):
            raise sqlite3.DatabaseError("file artifact backup manifest is not an object")
        if value.get("format") != _MANIFEST_FORMAT:
            raise sqlite3.DatabaseError("file artifact backup manifest format is unsupported")
        if value.get("database_sha256") != database_sha256:
            raise sqlite3.DatabaseError("file artifact backup manifest generation mismatch")
        artifacts = value.get("artifacts")
        if not isinstance(artifacts, list):
            raise sqlite3.DatabaseError("file artifact backup manifest has no artifact list")
        return value

    def _publish_json(self, target: Path, value: dict[str, Any]) -> None:
        ensure_private_directory(target.parent)
        temporary = target.with_name(f".manifest-{uuid4().hex}.tmp")
        try:
            create_private_file(temporary)
            temporary.write_text(
                json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            self._replace_path(temporary, target)
        finally:
            with suppress(OSError):
                temporary.unlink()

    def _manifest_path(self, database_sha256: str) -> Path:
        normalized = self._normalized_sha256(database_sha256)
        return self.manifest_root / f"{normalized}.json"

    def _blob_path(self, sha256: str) -> Path:
        normalized = self._normalized_sha256(sha256)
        return self.blob_root / normalized[:2] / normalized

    def _live_path(self, relative: Path) -> Path:
        candidate = (self.live_root / relative).resolve(strict=False)
        try:
            candidate.relative_to(self.live_root)
        except ValueError as exc:
            raise ValueError("file artifact path escaped its managed root") from exc
        return candidate

    @staticmethod
    def _controlled_relative_path(value: str) -> Path:
        relative = Path(str(value or ""))
        if not str(value or "").strip() or relative.is_absolute() or ".." in relative.parts:
            raise sqlite3.DatabaseError("invalid managed file artifact path")
        return relative

    @staticmethod
    def _normalized_sha256(value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if len(normalized) != _SHA256_LENGTH or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise sqlite3.DatabaseError("invalid managed file artifact digest")
        return normalized

    @classmethod
    def _verify_file(cls, path: Path, byte_size: int, sha256: str) -> bool:
        try:
            return (
                byte_size > 0
                and path.is_file()
                and path.stat().st_size == byte_size
                and cls._file_sha256(path) == sha256
            )
        except OSError:
            return False

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _verify_database(connection: sqlite3.Connection) -> None:
        row = connection.execute("PRAGMA integrity_check").fetchone()
        if row is None or str(row[0]).lower() != "ok":
            raise sqlite3.DatabaseError(f"restored database integrity check failed: {row!r}")

    @classmethod
    def _discard_tree(cls, path: Path) -> None:
        if not path.exists():
            return
        if not path.is_dir():
            raise OSError("managed file artifact cleanup target is not a directory")
        restrict_private_tree(path)
        shutil.rmtree(path)

    @staticmethod
    def _discard_empty_directories(root: Path) -> None:
        directories = sorted(
            (path for path in root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for path in directories:
            with suppress(OSError):
                path.rmdir()

    @staticmethod
    def _replace_path(source: Path, target: Path) -> None:
        retry_delays = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0)
        for attempt, delay in enumerate(retry_delays):
            if delay:
                time.sleep(delay)
            try:
                os.replace(source, target)
                if target.is_dir():
                    restrict_private_tree(target)
                else:
                    restrict_private_path(target, directory=False)
                sync_directory(target.parent)
                return
            except PermissionError:
                if attempt == len(retry_delays) - 1:
                    raise

    def _harden_backup_tree(self) -> None:
        if self._permissions_hardened:
            return
        ensure_private_directory(self.root)
        restrict_private_tree(self.root)
        self._permissions_hardened = True


__all__ = ["FileArtifactBackupStore", "PreparedArtifactRestore"]
