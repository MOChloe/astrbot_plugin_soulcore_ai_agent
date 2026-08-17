"""Administrator-approved reset for a database normal startup refused to open."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backup import SQLiteBackupManager, infer_backup_path
from .schema.current import SchemaRecoveryReason, SchemaRecoveryRequired

CONFIRMATION_TEXT = "清空并重建"
BACKUP_AND_REBUILD = "backup_and_rebuild"
REBUILD_WITHOUT_BACKUP = "rebuild_without_backup"
_ACTIONS = frozenset({BACKUP_AND_REBUILD, REBUILD_WITHOUT_BACKUP})


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
        self._state = self._database_state()
        self._token = secrets.token_urlsafe(32) if failure.destructive_recovery_allowed else ""

    def view(self) -> dict[str, Any]:
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
                "SoulCore 当前不会处理消息或读取业务数据。原数据未被自动修改；"
                "重建会清除数据库、凭据、媒体、语音、文件产物和缓存。"
            ),
            "recovery_token": self._token,
            "confirmation_text": CONFIRMATION_TEXT,
            "actions": [
                {
                    "action": BACKUP_AND_REBUILD,
                    "label": "备份全部数据并重建",
                    "description": "先保存带校验值的完整恢复包；备份失败时不会清空。",
                    "recommended": True,
                },
                {
                    "action": REBUILD_WITHOUT_BACKUP,
                    "label": "直接清空并重建",
                    "description": "不创建恢复包，永久清除 SoulCore 的全部本地数据。",
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
        if normalized not in _ACTIONS:
            raise ValueError("未知的数据库恢复动作")
        if not secrets.compare_digest(str(recovery_token or ""), self._token):
            raise ValueError("数据库恢复状态已经变化，请刷新页面后重新确认")
        if str(confirmation or "").strip() != CONFIRMATION_TEXT:
            raise ValueError(f"请输入“{CONFIRMATION_TEXT}”后再继续")
        backup_name = ""
        backup_sha256 = ""
        with self.backups.write_fence.hold():
            if self._database_state() != self._state:
                raise ValueError("SoulCore 数据在确认期间发生变化，请刷新页面后重新检查")
            payloads = self._managed_payload_paths()
            if normalized == BACKUP_AND_REBUILD:
                archive = self.backups.archive_database_family(
                    (path, f"state/{label}") for path, label in payloads
                )
                backup_name = archive.path.name
                backup_sha256 = archive.sha256
            try:
                self.backups.clear_primary_and_backup()
                for path, _label in payloads:
                    self._remove_managed_path(path)
            except BaseException:
                self._state = self._database_state()
                self._token = secrets.token_urlsafe(32)
                raise
        return SchemaRecoveryResult(
            normalized,
            bool(backup_name),
            backup_name,
            backup_sha256,
        )

    def _managed_payload_paths(self) -> tuple[tuple[Path, str], ...]:
        candidates = (
            (self.data_dir / "credentials.json", "credentials.json"),
            (self.data_dir / "file_artifacts", "file_artifacts"),
            (self.data_dir / "voice_artifacts", "voice_artifacts"),
            (self.data_dir / "recall", "recall"),
            (self._media_root(), "media"),
            (
                self.backups.migration_snapshot_path,
                "pre-migration.sqlite3",
            ),
        )
        if self.backups.file_artifacts is not None:
            candidates += ((self.backups.file_artifacts.root, "managed-backup-artifacts"),)
        unique: list[tuple[Path, str]] = []
        seen: set[Path] = set()
        for path, label in candidates:
            resolved = path.resolve(strict=False)
            if resolved in seen:
                continue
            seen.add(resolved)
            unique.append((resolved, label))
        return tuple(unique)

    def _media_root(self) -> Path:
        parts = self.database_path.parts
        indexes = [index for index, part in enumerate(parts) if part.lower() == "plugin_data"]
        if indexes:
            index = indexes[-1]
            plugin_name = (
                parts[index + 1]
                if index + 1 < len(parts)
                else "astrbot_plugin_soulcore_ai_agent"
            )
            return Path(*parts[:index]) / "soulcore_media" / plugin_name
        return self.data_dir / "soulcore_media" / self.database_path.stem

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
            self.backups.backup_path,
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


def _reason_copy(reason: SchemaRecoveryReason) -> tuple[str, str]:
    return {
        SchemaRecoveryReason.STRUCTURE_MISMATCH: (
            "数据库结构与正式版不一致",
            "正式版不会自动改写未知或预发布数据库。请先备份后建立全新的正式版数据。",
        ),
        SchemaRecoveryReason.NEWER_SCHEMA: (
            "数据库来自更高版本",
            "请安装创建该数据库的同版或更高版 SoulCore；当前版本不会改写或清空它。",
        ),
        SchemaRecoveryReason.CORRUPT_DATABASE: (
            "数据库完整性检查失败",
            "SoulCore 没有尝试覆盖损坏或无法验证的数据。",
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
