"""Cross-runtime ownership fence for startup recovery and background workers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO

from ..shared.private_paths import (
    create_private_file,
    ensure_private_directory,
    restrict_private_path,
)


class RuntimeOwnershipError(RuntimeError):
    """Raised when another live SoulCore runtime owns the same data directory."""


class RuntimeOwnershipFence:
    """Hold one OS-backed byte lock for the complete application lifetime."""

    def __init__(self, path: Path, handle: BinaryIO) -> None:
        self.path = path
        self._handle: BinaryIO | None = handle

    @classmethod
    def acquire(cls, data_dir: Path) -> RuntimeOwnershipFence:
        ensure_private_directory(data_dir)
        path = data_dir / ".soulcore-runtime.lock"
        try:
            create_private_file(path)
        except FileExistsError:
            restrict_private_path(path, directory=False)
        handle = path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            _lock(handle)
        except OSError as exc:
            handle.close()
            raise RuntimeOwnershipError(f"another live SoulCore runtime owns {data_dir}") from exc
        return cls(path, handle)

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            handle.seek(0)
            _unlock(handle)
        finally:
            handle.close()


if os.name == "nt":
    import msvcrt

    def _lock(handle: BinaryIO) -> None:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(handle: BinaryIO) -> None:
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(handle: BinaryIO) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle: BinaryIO) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = ["RuntimeOwnershipError", "RuntimeOwnershipFence"]
