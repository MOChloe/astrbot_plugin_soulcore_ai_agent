"""Lazy SQLite composition exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "RepositoryBundle": (".bundle", "RepositoryBundle"),
    "SQLiteBackupManager": (".backup", "SQLiteBackupManager"),
    "SqliteEngine": (".engine", "SqliteEngine"),
    "SqliteUnitOfWork": (".uow", "SqliteUnitOfWork"),
    "infer_backup_path": (".backup", "infer_backup_path"),
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
