from __future__ import annotations

from ....storage.sqlite.repository import SqliteRepository
from ....storage.sqlite.runtime_file_cleanup import RuntimeFileCleanupRecords
from .assets import MediaAssetRecords
from .lifecycle import MediaLifecycleCommands


class SqliteMediaRepository(
    MediaAssetRecords,
    MediaLifecycleCommands,
    RuntimeFileCleanupRecords,
    SqliteRepository,
):
    """SQLite implementation of the media persistence boundary."""


__all__ = ["SqliteMediaRepository"]
