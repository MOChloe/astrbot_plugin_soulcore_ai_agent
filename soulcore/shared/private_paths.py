"""Fail-closed filesystem permissions for SoulCore private data."""

from __future__ import annotations

import os
from pathlib import Path


def ensure_private_directory(path: Path) -> Path:
    """Create a directory and restrict it to the current process owner."""

    missing: list[Path] = []
    candidate = path
    while True:
        _reject_link(candidate)
        if candidate.exists():
            if not candidate.is_dir():
                raise OSError("private directory path is not a directory")
        else:
            missing.append(candidate)
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    for created in reversed(missing):
        try:
            created.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError:
            _reject_link(created)
            if not created.is_dir():
                raise OSError("private directory path is not a directory") from None
        restrict_private_path(created, directory=True)
    if not missing:
        restrict_private_path(path, directory=True)
    return path


def create_private_directory(path: Path) -> Path:
    """Exclusively create an owner-only directory for a sensitive operation."""

    ensure_private_directory(path.parent)
    path.mkdir(mode=0o700, exist_ok=False)
    try:
        restrict_private_path(path, directory=True)
    except BaseException:
        path.rmdir()
        raise
    return path


def create_private_file(path: Path) -> Path:
    """Exclusively create an empty owner-only file for a sensitive payload."""

    ensure_private_directory(path.parent)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    try:
        restrict_private_path(path, directory=False)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def restrict_private_tree(root: Path) -> None:
    """Harden every existing path in a managed private tree without following links."""

    if not root.exists():
        return
    _reject_link(root)
    if root.is_file():
        restrict_private_path(root, directory=False)
        return
    if not root.is_dir():
        raise OSError("private data path is neither a file nor a directory")
    restrict_private_path(root, directory=True)
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            _reject_link(path)
            restrict_private_path(path, directory=True)
        for name in files:
            path = current_path / name
            _reject_link(path)
            if not path.is_file():
                raise OSError("private data tree contains a non-file entry")
            restrict_private_path(path, directory=False)


def restrict_private_path(path: Path, *, directory: bool) -> None:
    """Remove inherited access and grant full control only to the owner.

    POSIX uses ``0700`` for directories and ``0600`` for files. Windows uses
    a protected DACL with one full-control ACE for the current process owner;
    private directories pass that ACE to newly created children.
    """

    _reject_link(path)
    if os.name != "nt":
        os.chmod(path, 0o700 if directory else 0o600)
        return
    _windows_owner_only(path, inherit_to_children=directory)


def _reject_link(path: Path) -> None:
    is_junction = getattr(path, "is_junction", lambda: False)
    if path.is_symlink() or is_junction():
        raise OSError("private data path must not be a link or junction")


def sync_directory(path: Path) -> None:
    """Durably publish a directory entry where the platform supports it."""

    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _windows_owner_only(path: Path, *, inherit_to_children: bool) -> None:
    from .private_paths_windows import restrict_windows_owner_only

    restrict_windows_owner_only(path, inherit_to_children=inherit_to_children)


__all__ = [
    "create_private_directory",
    "create_private_file",
    "ensure_private_directory",
    "restrict_private_path",
    "restrict_private_tree",
    "sync_directory",
]
