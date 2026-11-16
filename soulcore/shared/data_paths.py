"""One layout for persistent payloads inside the plugin's private data directory."""

from __future__ import annotations

from pathlib import Path

DATABASE_FILENAME = "soulcore.sqlite3"


def infer_media_root(database_path: str | Path) -> Path:
    database = Path(database_path).resolve(strict=False)
    return database.parent / "soulcore_media" / database.stem


def infer_backup_path(database_path: str | Path) -> Path | None:
    """Enable managed backups inside the namespace of a plugin_data database."""

    if str(database_path) == ":memory:":
        return None
    database = Path(database_path).resolve(strict=False)
    if not any(parent.name.lower() == "plugin_data" for parent in database.parents[1:]):
        return None
    return database.parent / "backups" / database.name
