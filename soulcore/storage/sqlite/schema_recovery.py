"""Administrator-approved restoration or archived rebuild after startup refusal."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...shared.data_paths import infer_media_root
from ...shared.private_paths import create_private_file
from .backup import RecoveryArchive, SQLiteBackupManager, infer_backup_path
from .recovery_restore import restore_source
from .recovery_sources import archive_payloads, list_sources
from .schema.current import SchemaRecoveryReason, SchemaRecoveryRequired, create_current_schema

CONFIRMATION_TEXT = "确认恢复操作"
BACKUP_AND_REBUILD = "backup_and_rebuild"
REBUILD_WITHOUT_BACKUP = "rebuild_without_backup"
RESTORE_PREFIX = "restore:"
_ACTIONS = frozenset({BACKUP_AND_REBUILD})


@dataclass(frozen=True, slots=True)
class SchemaRecoveryResult:
    action: str
    backup_created: bool
    backup_name: str
    backup_sha256: str

    def public_view(self) -> dict[str, Any]:
        return {
            "ok": True,
            "action": self.action,
            "backup_created": self.backup_created,
            "backup_name": self.backup_name,
            "backup_sha256": self.backup_sha256,
            "backup_location": f"backups/recovery/{self.backup_name}" if self.backup_name else "",
            "restored": self.action.startswith(RESTORE_PREFIX),
            "message": (
                (
                    "已从所选备份还原；还原前的数据已保留在恢复包中。"
                    if self.backup_created
                    else "已从所选备份还原。"
                )
                if self.action.startswith(RESTORE_PREFIX)
                else f"已备份并重建。恢复包保存在插件数据目录的 backups/recovery/{self.backup_name}。"
            ),
        }


class SchemaRecoveryCoordinator:
    """Bind one destructive choice to the exact failed SoulCore data family."""

    def __init__(self, failure: SchemaRecoveryRequired) -> None:
        if not failure.database_path:
            raise ValueError("schema recovery failure has no database path")
        self.failure = failure
        self.database_path = Path(failure.database_path).resolve(strict=False)
        self.data_dir = self.database_path.parent
        backup_path = infer_backup_path(self.database_path)
        if backup_path is None:
            backup_path = self.database_path.with_name(f"{self.database_path.name}.backup")
        self.backups = SQLiteBackupManager(
            self.database_path,
            backup_path,
            file_artifact_root=self.data_dir / "file_artifacts",
        )
        self._sources = list_sources(self.backups)
        self._state = self._database_state()
        self._token = secrets.token_urlsafe(32) if failure.destructive_recovery_allowed else ""

    def view(self) -> dict[str, Any]:
        current_state = self._database_state()
        if current_state != self._state:
            self._state = current_state
            self._sources = list_sources(self.backups)
            self._token = (
                secrets.token_urlsafe(32) if self.failure.destructive_recovery_allowed else ""
            )
        title, message = _reason_copy(self.failure.reason)
        if not self.failure.destructive_recovery_allowed:
            return {
                "required": True,
                "reason": self.failure.reason.value,
                "title": title,
                "message": message,
                "impact": (
                    "SoulCore 当前不会处理消息或读取业务数据；原数据库保持不变，"
                    "此问题禁止通过清空数据处理。"
                ),
                "recovery_token": "",
                "confirmation_text": "",
                "actions": [],
            }
        return {
            "required": True,
            "reason": self.failure.reason.value,
            "title": title,
            "message": message,
            "impact": (
                "原数据保留中。优先选择备份还原；还原前会校验并保存当前现场。"
                "还原会回到备份时刻，之后的数据不会合并。仅数据库快照保留当前凭据和媒体，"
                "恢复包则还原其中的配套数据。若选择重建，将归档后清空全部本地数据。"
            ),
            "recovery_token": self._token,
            "confirmation_text": CONFIRMATION_TEXT,
            "backup_directory": "data/plugin_data/astrbot_plugin_soulcore/backups/recovery/",
            "sources": [source.public_view() for source in self._sources],
            "actions": [
                *[
                    {
                        "action": RESTORE_PREFIX + source.id,
                        "label": f"校验并还原：{source.label}",
                        "description": (
                            f"{source.path.name} · {source.public_view()['created_at']}。"
                            + (
                                "这份备份已标记过期，可能不含最近写入或包含已删除的数据。"
                                if source.stale
                                else "校验失败时保留当前数据。"
                            )
                        ),
                        "recommended": index == 0 and not source.stale,
                    }
                    for index, source in enumerate(self._sources)
                ],
                {
                    "action": BACKUP_AND_REBUILD,
                    "label": "备份全部数据并重建",
                    "description": "先保存带校验值的完整恢复包；备份失败时不会清空。",
                    "recommended": False,
                },
            ],
        }

    def execute(
        self,
        *,
        action: str,
        recovery_token: str,
        confirmation: str,
    ) -> SchemaRecoveryResult:
        if not self.failure.destructive_recovery_allowed:
            raise ValueError("此数据库问题禁止通过清空数据处理")
        normalized = str(action or "").strip().lower()
        source = next(
            (item for item in self._sources if normalized == RESTORE_PREFIX + item.id), None
        )
        if normalized == REBUILD_WITHOUT_BACKUP:
            raise ValueError("已禁止不留备份直接清空；请选择备份还原或备份全部数据并重建")
        if normalized not in _ACTIONS and source is None:
            raise ValueError("未知的数据库恢复动作")
        if not secrets.compare_digest(str(recovery_token or ""), self._token):
            raise ValueError("数据库恢复状态已经变化，请刷新页面后重新确认")
        if str(confirmation or "").strip() != CONFIRMATION_TEXT:
            raise ValueError(f"请输入“{CONFIRMATION_TEXT}”后再继续")
        with self.backups.write_fence.hold():
            if self._database_state() != self._state:
                raise ValueError("SoulCore 数据在确认期间发生变化，请刷新页面后重新检查")
            if source is not None:
                archive = restore_source(self.backups, source)
                self._token = secrets.token_urlsafe(32)
                return _restore_result(normalized, archive)
            return self._rebuild()

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
        return SchemaRecoveryResult(
            BACKUP_AND_REBUILD,
            True,
            archive.path.name,
            archive.sha256,
        )

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


def _restore_result(action: str, archive: RecoveryArchive | None) -> SchemaRecoveryResult:
    return SchemaRecoveryResult(
        action,
        archive is not None,
        archive.path.name if archive else "",
        archive.sha256 if archive else "",
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
