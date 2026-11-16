"""Backup, import, and recovery of the managed SoulCore database."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import zipfile
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ...shared.data_paths import infer_media_root
from ...shared.private_paths import create_private_file
from .backup import RecoveryArchive, SQLiteBackupManager, infer_backup_path
from .database_repair import RepairReport, repair_database
from .recovery_archive import archive_members
from .recovery_publish import recover_interrupted_restore
from .recovery_restore import restore_source
from .recovery_sources import RecoverySource, archive_payloads, list_sources
from .schema.current import SchemaRecoveryReason, SchemaRecoveryRequired, create_current_schema

CONFIRMATION_TEXT = "确认恢复操作"
BACKUP_AND_REBUILD = "backup_and_rebuild"
REBUILD_WITHOUT_BACKUP = "rebuild_without_backup"
RESTORE_PREFIX = "restore:"
CREATE_BACKUP = "create_backup"
IMPORT_BACKUP = "import_backup"
REPAIR_DATABASE = "repair"
_ACTIONS = frozenset({BACKUP_AND_REBUILD, IMPORT_BACKUP, REPAIR_DATABASE})


@dataclass(frozen=True, slots=True)
class SchemaRecoveryResult:
    action: str
    backup_created: bool
    backup_name: str
    backup_sha256: str
    backup_location: str = ""
    repair_report: RepairReport | None = None

    def public_view(self) -> dict[str, Any]:
        return {
            "ok": True,
            "action": self.action,
            "backup_created": self.backup_created,
            "backup_name": self.backup_name,
            "backup_sha256": self.backup_sha256,
            "backup_location": self.backup_location,
            "restored": self.action == IMPORT_BACKUP or self.action.startswith(RESTORE_PREFIX),
            "repair_report": asdict(self.repair_report) if self.repair_report else None,
            "message": self._message(),
        }

    def _message(self) -> str:
        return {
            CREATE_BACKUP: "备份已保存，可下载到本机。",
            REPAIR_DATABASE: "数据库已修复，修复前的备份可以下载。",
            IMPORT_BACKUP: "备份已导入。",
            BACKUP_AND_REBUILD: "已保存备份并重新建立数据库。",
        }.get(self.action, "已从所选备份还原。")


class SchemaRecoveryCoordinator:
    """Manage one private database family across normal operation and startup recovery."""

    def __init__(self, database: SchemaRecoveryRequired | Path) -> None:
        self.failure = database if isinstance(database, SchemaRecoveryRequired) else None
        path = self.failure.database_path if self.failure is not None else database
        if not path:
            raise ValueError("schema recovery failure has no database path")
        self.database_path = Path(path).resolve(strict=False)
        self.data_dir = self.database_path.parent
        backup_path = infer_backup_path(self.database_path)
        if backup_path is None:
            backup_path = self.database_path.with_name(f"{self.database_path.name}.backup")
        self.backups = SQLiteBackupManager(
            self.database_path,
            backup_path,
            file_artifact_root=self.data_dir / "file_artifacts",
        )
        self._state = self._database_state() if self.failure else ""
        self._token = secrets.token_urlsafe(32) if self.can_replace else ""

    @property
    def can_replace(self) -> bool:
        return self.failure is None or self.failure.destructive_recovery_allowed

    @property
    def backup_directory(self) -> Path:
        return self.backups.backup_path.parent / "recovery"

    def view(self) -> dict[str, Any]:
        sources = list_sources(self.backups)
        current_state = self._database_state() if self.failure else ""
        if self.failure and current_state != self._state:
            self._state = current_state
            self._token = secrets.token_urlsafe(32) if self.can_replace else ""
        route = {
            "required": self.failure is not None,
            "backup_directory": str(self.backup_directory),
            "can_import": self.can_replace,
            "sources": [source.public_view() for source in sources],
            "recovery_token": self._token,
            "confirmation_text": CONFIRMATION_TEXT if self.can_replace else "",
            "actions": self._actions(sources) if self.can_replace else [],
        }
        if self.failure is None:
            return route
        title, message = _reason_copy(self.failure.reason)
        return {
            **route,
            "reason": self.failure.reason.value,
            "title": title,
            "message": message,
            "impact": (
                "还原会回到备份时刻。修复会保留原件，并从能够读取的记录重建数据库。"
                if self.can_replace
                else "当前无法处理消息。可先下载数据库备份，此问题禁止通过清空数据处理。"
            ),
        }

    def _actions(self, sources: list[RecoverySource]) -> list[dict[str, Any]]:
        actions = [
            {
                "action": RESTORE_PREFIX + source.id,
                "label": source.label,
                "description": (
                    f"{source.path.name} · {source.public_view()['created_at']}。"
                    + (
                        "备份已过期，可能缺少最近的变更。"
                        if source.stale
                        else "校验失败时保留当前数据。"
                    )
                ),
                "recommended": index == 0 and not source.stale,
            }
            for index, source in enumerate(sources)
        ]
        if self.database_path.is_file():
            actions.append(
                {
                    "action": REPAIR_DATABASE,
                    "label": "修复数据库",
                    "description": "先备份当前数据，再把可读且符合当前结构的记录导入新库。",
                    "recommended": False,
                }
            )
        if self.failure:
            actions.append(
                {
                    "action": BACKUP_AND_REBUILD,
                    "label": "备份全部数据并重建",
                    "description": "保存恢复包后，从空数据库重新开始。",
                    "recommended": False,
                }
            )
        return actions

    def validate(self, action: str, recovery_token: str, confirmation: str) -> None:
        if action == CREATE_BACKUP:
            return
        if not self.can_replace:
            raise ValueError("此数据库问题禁止通过清空数据处理")
        if action == REBUILD_WITHOUT_BACKUP:
            raise ValueError("已禁止不留备份直接清空；请选择备份还原或备份全部数据并重建")
        if action not in _ACTIONS and not action.startswith(RESTORE_PREFIX):
            raise ValueError("未知的数据库恢复动作")
        if action == BACKUP_AND_REBUILD and not self.failure:
            raise ValueError("当前数据库可用，请使用修复数据库")
        if not secrets.compare_digest(str(recovery_token or ""), self._token):
            raise ValueError("数据库恢复状态已经变化，请刷新页面后重新确认")
        if str(confirmation or "").strip() != CONFIRMATION_TEXT:
            raise ValueError(f"请输入“{CONFIRMATION_TEXT}”后再继续")

    def execute(
        self,
        *,
        action: str,
        recovery_token: str,
        confirmation: str,
        upload: Path | None = None,
    ) -> SchemaRecoveryResult:
        normalized = str(action or "").strip().lower()
        self.validate(normalized, recovery_token, confirmation)
        with self.backups.write_fence.hold():
            recover_interrupted_restore(self.database_path)
            if normalized == CREATE_BACKUP:
                return _result(
                    normalized, self.backups.archive_database_family(archive_payloads(self.backups))
                )
            if self.failure and self._database_state() != self._state:
                raise ValueError("SoulCore 数据在确认期间发生变化，请刷新页面后重新检查")
            result = self._apply_recovery(normalized, upload)
            self._token = secrets.token_urlsafe(32)
            return result

    def _apply_recovery(self, action: str, upload: Path | None) -> SchemaRecoveryResult:
        if action == REPAIR_DATABASE:
            archive, report = repair_database(self.backups)
            return _result(action, archive, report)
        if action == BACKUP_AND_REBUILD:
            return self._rebuild()
        source = (
            self._uploaded_source(upload)
            if action == IMPORT_BACKUP
            else self._get_source(action.removeprefix(RESTORE_PREFIX))
        )
        try:
            archive = restore_source(self.backups, source)
        except (SchemaRecoveryRequired, sqlite3.DatabaseError, zipfile.BadZipFile) as exc:
            raise ValueError("所选备份无法读取或校验，当前数据已保留。") from exc
        return _result(action, archive)

    def _get_source(self, source_id: str) -> RecoverySource:
        source = next((item for item in list_sources(self.backups) if item.id == source_id), None)
        if source is None:
            raise ValueError("找不到这份备份，请刷新后重新选择")
        return source

    def _uploaded_source(self, path: Path | None) -> RecoverySource:
        if path is None:
            raise ValueError("请选择要导入的备份文件")
        if path.suffix != ".zip":
            return RecoverySource(path, "", "导入的数据库快照")
        try:
            with zipfile.ZipFile(path) as archive:
                members = archive_members(archive)
        except zipfile.BadZipFile as exc:
            raise ValueError("无法读取这个备份 ZIP") from exc
        if self.database_path.name not in members:
            raise ValueError("恢复包缺少主数据库")
        return RecoverySource(path, self.database_path.name, "导入的恢复包")

    def download_path(self, source_id: str, backup_name: str) -> Path:
        if source_id:
            return self._get_source(source_id).path
        prefix = f"{self.database_path.stem}-recovery-"
        if (
            Path(backup_name).name != backup_name
            or not backup_name.startswith(prefix)
            or not backup_name.endswith(".zip")
        ):
            raise ValueError("备份文件名无效")
        path = self.backup_directory / backup_name
        self.backups._reject_link(path)
        if not path.is_file():
            raise ValueError("备份文件已经移走或删除")
        return path

    def _rebuild(self) -> SchemaRecoveryResult:
        payloads = self._managed_payload_paths()
        archive = self.backups.archive_database_family(
            (path, f"state/{label}") for path, label in payloads
        )
        try:
            self.backups.clear_primary_and_backup()
            for path, _label in payloads:
                self._remove_managed_path(path)
            # Only an explicit, archived rebuild may create an empty schema.
            create_private_file(self.database_path)
            with closing(sqlite3.connect(self.database_path, isolation_level=None)) as connection:
                create_current_schema(connection)
        except BaseException:
            self._state = self._database_state()
            self._token = secrets.token_urlsafe(32)
            raise
        return _result(BACKUP_AND_REBUILD, archive)

    def _managed_payload_paths(self) -> tuple[tuple[Path, str], ...]:
        return tuple(
            (path, label.removeprefix("state/")) for path, label in archive_payloads(self.backups)
        )

    def _media_root(self) -> Path:
        return infer_media_root(self.database_path)

    @staticmethod
    def _remove_managed_path(path: Path) -> None:
        if not path.exists():
            return
        SQLiteBackupManager._reject_link(path)
        if path.is_dir():
            SQLiteBackupManager._remove_managed_tree(path)
            return
        if not path.is_file():
            raise OSError("managed recovery path is neither a file nor a directory")
        path.unlink()

    def _database_state(self) -> str:
        paths = (
            self.database_path,
            Path(f"{self.database_path}-wal"),
            Path(f"{self.database_path}-shm"),
            Path(f"{self.database_path}-journal"),
            self.backups.backup_path,
            self.backups.invalidation_marker_path,
            *(source.path for source in list_sources(self.backups)),
            *(path for path, _label in self._managed_payload_paths()),
        )
        records: list[tuple[str, int | None, int | None]] = []
        for root in paths:
            if root.is_dir():
                SQLiteBackupManager._reject_link(root)
                for path in sorted(root.rglob("*")):
                    SQLiteBackupManager._reject_link(path)
                    if path.is_file():
                        stat = path.stat()
                        records.append((str(path), stat.st_size, stat.st_mtime_ns))
                continue
            try:
                stat = root.stat()
                records.append((str(root), stat.st_size, stat.st_mtime_ns))
            except FileNotFoundError:
                records.append((str(root), None, None))
        payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _result(
    action: str, archive: RecoveryArchive | None, report: RepairReport | None = None
) -> SchemaRecoveryResult:
    return SchemaRecoveryResult(
        action,
        archive is not None,
        archive.path.name if archive else "",
        archive.sha256 if archive else "",
        str(archive.path) if archive else "",
        report,
    )


def _reason_copy(reason: SchemaRecoveryReason) -> tuple[str, str]:
    return {
        SchemaRecoveryReason.STRUCTURE_MISMATCH: (
            "数据库结构与正式版不一致",
            "当前版本不能直接打开这份结构。请优先还原已知版本的备份；未知结构仍需保留原件后排查。",
        ),
        SchemaRecoveryReason.NEWER_SCHEMA: (
            "数据库来自更高版本",
            "请安装创建该数据库的同版或更高版 SoulCore；当前版本不会改写或清空它。",
        ),
        SchemaRecoveryReason.CORRUPT_DATABASE: (
            "数据库完整性检查失败",
            "主数据库缺失、损坏或无法验证。SoulCore 已停止启动，请先检查现有备份并还原。",
        ),
        SchemaRecoveryReason.MIGRATION_FAILED: (
            "数据库升级没有完成",
            "升级事务已完整回滚，原数据库仍保持升级前版本。请保留日志并修复升级错误后重试。",
        ),
    }[reason]


__all__ = [
    "BACKUP_AND_REBUILD",
    "CONFIRMATION_TEXT",
    "REBUILD_WITHOUT_BACKUP",
    "SchemaRecoveryCoordinator",
    "SchemaRecoveryResult",
]
