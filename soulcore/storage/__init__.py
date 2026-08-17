"""Storage composition exports without eager repository assembly."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "MAX_IMAGE_BYTES": ("..features.media", "MAX_IMAGE_BYTES"),
    "MAX_INBOUND_ATTACHMENT_BYTES": ("..features.media", "MAX_INBOUND_ATTACHMENT_BYTES"),
    "MediaFileStore": ("..features.media", "MediaFileStore"),
    "MediaStorageCoordinator": ("..features.media", "MediaStorageCoordinator"),
    "RepositoryBundle": (".sqlite.bundle", "RepositoryBundle"),
    "SQLiteBackupManager": (".sqlite.backup", "SQLiteBackupManager"),
    "SqliteEngine": (".sqlite.engine", "SqliteEngine"),
    "SqliteUnitOfWork": (".sqlite.uow", "SqliteUnitOfWork"),
    "generate_media_asset_id": ("..features.media", "generate_media_asset_id"),
    "infer_backup_path": (".sqlite.backup", "infer_backup_path"),
    "infer_media_root": ("..features.media", "infer_media_root"),
    "inspect_animation_bytes": ("..features.media", "inspect_animation_bytes"),
    "inspect_image_bytes": ("..features.media", "inspect_image_bytes"),
    "inspect_inbound_attachment_bytes": ("..features.media", "inspect_inbound_attachment_bytes"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
