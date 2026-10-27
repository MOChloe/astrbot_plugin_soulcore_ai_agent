"""Relocate recognized external payload directories before opening the database."""

from __future__ import annotations

from pathlib import Path

from astrbot.api import logger

from ...shared.data_paths import infer_backup_path, infer_media_root
from ...shared.private_paths import (
    _reject_link,
    ensure_private_directory,
    restrict_private_tree,
    sync_directory,
)
from .write_fence import SQLiteWriteFence


def relocate_managed_payloads(database_path: str | Path) -> None:
    """Atomically move whole trees; never merge, overwrite or restore a database.

    Relative media references and backup invalidation markers remain unchanged.
    A restart resumes after any completed directory rename. Conflicting trees
    stop startup with both copies intact instead of guessing which one to keep.
    """

    database = Path(database_path).absolute()
    data_dir = database.parent
    if data_dir.parent.name.lower() != "plugin_data":
        return
    ensure_private_directory(data_dir)
    _reject_link(database)
    backup = infer_backup_path(database)
    assert backup is not None
    data_root = data_dir.parent.parent
    pairs = (
        (data_root / "soulcore_media" / data_dir.name, infer_media_root(database)),
        (data_root / "soulcore_backups" / data_dir.name, backup.parent),
    )
    with SQLiteWriteFence(database).hold():
        pending = _pending_moves(pairs, data_root, data_dir)
        for source, destination in pending:
            ensure_private_directory(destination.parent)
            source.rename(destination)
            sync_directory(source.parent)
            sync_directory(destination.parent)
            logger.info("SoulCore moved managed %s storage into plugin_data", source.parent.name)


def _pending_moves(
    pairs: tuple[tuple[Path, Path], ...], data_root: Path, data_dir: Path
) -> list[tuple[Path, Path]]:
    pending: list[tuple[Path, Path]] = []
    for source, destination in pairs:
        _reject_link(source.parent)
        _reject_link(source)
        if not source.exists():
            continue
        _reject_link(destination.parent)
        _reject_link(destination)
        source.resolve().relative_to(data_root.resolve())
        destination.resolve(strict=False).relative_to(data_dir.resolve())
        if not source.is_dir():
            raise OSError("SoulCore payload migration source is not a directory")
        if destination.exists():
            raise FileExistsError(
                "SoulCore payload migration found both external and plugin_data directories; "
                "preserve both directories and resolve the conflict before restarting"
            )
        restrict_private_tree(source)
        pending.append((source, destination))
    return pending
