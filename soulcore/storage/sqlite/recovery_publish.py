"""Journalled publication of a staged database and its associated payloads.

The caller owns both the runtime fence and the SQLite write fence. A process
interruption before the commit marker rolls back on the next startup.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ...shared.private_paths import create_private_file, ensure_private_directory, sync_directory
from .backup import SQLiteBackupManager

JOURNAL_NAME = ".soulcore-restore.json"
STAGE_PREFIX = ".soulcore-restore-"


def stage_path(database: Path, name: str) -> Path:
    """Keep deep archive paths usable on Windows without MAX_PATH truncation."""
    if not re.fullmatch(re.escape(STAGE_PREFIX) + r"[0-9a-f]{32}", name):
        raise ValueError("数据库还原临时目录无效")
    path = (database.parent / name).absolute()
    value = str(path)
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value)


def payload_paths(database: Path) -> dict[str, Path]:
    root = database.parent
    return {
        "credentials.json": root / "credentials.json",
        "file_artifacts": root / "file_artifacts",
        "voice_artifacts": root / "voice_artifacts",
        "recall": root / "recall",
        "media": root / "soulcore_media" / database.stem,
    }


def _targets(database: Path) -> set[str]:
    root = database.parent
    return {
        *(path.relative_to(root).as_posix() for path in payload_paths(database).values()),
        database.name,
        *(database.name + suffix for suffix in ("-wal", "-shm", "-journal")),
    }


def _write_journal(path: Path, value: dict) -> None:
    temporary = SQLiteBackupManager._temporary_path(path)
    try:
        create_private_file(temporary)
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        SQLiteBackupManager._replace_temporary(temporary, path)
    finally:
        SQLiteBackupManager._discard_temporary(temporary)


def _remove(path: Path) -> None:
    SQLiteBackupManager._reject_link(path)
    if path.is_dir():
        SQLiteBackupManager._remove_managed_tree(path)
    elif path.exists():
        path.unlink()


def _sync_payload(path: Path) -> None:
    """Persist staged file contents before publishing a durable commit marker."""
    SQLiteBackupManager._reject_link(path)
    if path.is_dir():
        for child in path.iterdir():
            _sync_payload(child)
        sync_directory(path)
    else:
        if not path.is_file():
            raise ValueError("数据库还原暂存内容不是普通文件")
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())


def _read_journal(database: Path, journal: Path) -> tuple[Path, list[dict], bool]:
    SQLiteBackupManager._reject_link(journal)
    if journal.stat().st_size > 64 * 1024:
        raise ValueError("数据库还原日志过大，请保留现场")
    value = json.loads(journal.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("format") != 1
        or type(value.get("committed")) is not bool
    ):
        raise ValueError("数据库还原日志无效，请保留现场")
    stage_name = value.get("stage", "")
    if (
        not isinstance(stage_name, str)
        or Path(stage_name).name != stage_name
        or not stage_name.startswith(STAGE_PREFIX)
    ):
        raise ValueError("数据库还原日志目录无效")
    stage = stage_path(database, stage_name)
    SQLiteBackupManager._reject_link(stage)
    operations = _validate_operations(value.get("operations"), _targets(database))
    return stage, operations, value["committed"]


def _validate_operations(operations: object, targets: set[str]) -> list[dict]:
    if not isinstance(operations, list) or len(operations) > len(targets):
        raise ValueError("数据库还原日志操作无效")
    seen = set()
    for row in operations:
        if not isinstance(row, dict) or not isinstance(row.get("target"), str):
            raise ValueError("数据库还原日志目标无效")
        if (
            row["target"] not in targets
            or row["target"] in seen
            or type(row.get("existed")) is not bool
        ):
            raise ValueError("数据库还原日志目标无效")
        seen.add(row["target"])
    return operations


def _rollback(root: Path, stage: Path, operations: list[dict]) -> None:
    if not stage.is_dir():
        raise ValueError("数据库还原回滚目录缺失，请保留现场")
    ensure_private_directory(stage / "previous")
    for index, row in reversed(list(enumerate(operations))):
        target = root / row["target"]
        ensure_private_directory(target.parent)
        previous = stage / "previous" / str(index)
        SQLiteBackupManager._reject_link(previous)
        if previous.exists():
            _remove(target)
            os.replace(previous, target)
            sync_directory(target.parent)
            sync_directory(previous.parent)
        elif not row["existed"]:
            _remove(target)
            sync_directory(target.parent)


def recover_interrupted_restore(database: Path) -> bool:
    """Resolve a durable journal before SQLite is allowed to open the primary."""
    root = database.parent
    journal = root / JOURNAL_NAME
    if not journal.exists():
        return False
    stage, operations, committed = _read_journal(database, journal)
    if not committed:
        _rollback(root, stage, operations)
    # Remove the journal first: cleanup can be retried without replaying a
    # partially deleted rollback directory. Unreferenced stages are harmless.
    journal.unlink()
    sync_directory(root)
    _remove(stage)
    return True


def publish_restore(database: Path, stage: Path, replacements: dict[Path, Path | None]) -> None:
    root = database.parent
    if stage != stage_path(database, stage.name):
        raise ValueError("数据库还原临时目录无效")
    journal = root / JOURNAL_NAME
    if journal.exists():
        raise ValueError("请先处理上一次未完成的数据库还原")
    previous_root = ensure_private_directory(stage / "previous")
    operations = []
    for target, source in replacements.items():
        relative = target.relative_to(root).as_posix()
        if relative not in _targets(database):
            raise ValueError("数据库还原目标超出受管范围")
        ensure_private_directory(target.parent)
        SQLiteBackupManager._reject_link(target)
        if source is not None:
            source.resolve().relative_to(stage.resolve())
            SQLiteBackupManager._reject_link(source)
            if not source.exists():
                raise ValueError("数据库还原临时文件缺失")
            _sync_payload(source)
        operations.append({"target": relative, "existed": target.exists()})
    value = {"format": 1, "stage": stage.name, "operations": operations, "committed": False}
    _write_journal(journal, value)
    try:
        for index, (target, source) in enumerate(replacements.items()):
            if target.exists():
                os.replace(target, previous_root / str(index))
                sync_directory(previous_root)
                sync_directory(target.parent)
            if source is not None:
                os.replace(source, target)
                sync_directory(source.parent)
                sync_directory(target.parent)
        value["committed"] = True
        _write_journal(journal, value)
    except BaseException:
        recover_interrupted_restore(database)
        raise
    recover_interrupted_restore(database)
