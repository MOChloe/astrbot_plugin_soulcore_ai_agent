from __future__ import annotations

import errno
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from ...shared.private_paths import (
    create_private_file,
    ensure_private_directory,
    restrict_private_path,
)


class SQLiteWriteFence:
    """Advisory cross-process fence shared by writers and database publication."""

    def __init__(self, database_path: str | Path) -> None:
        database = Path(database_path)
        self.path = database.with_name(f".{database.name}.write.lock")
        self._permissions_ready = False

    @contextmanager
    def hold(self) -> Iterator[None]:
        if not self._permissions_ready:
            ensure_private_directory(self.path.parent)
            try:
                create_private_file(self.path)
            except FileExistsError:
                restrict_private_path(self.path, directory=False)
            self._permissions_ready = True
        with self.path.open("a+b") as stream:
            self._ensure_lock_byte(stream)
            self._lock(stream)
            try:
                yield
            finally:
                self._unlock(stream)

    @staticmethod
    def _ensure_lock_byte(stream: BinaryIO) -> None:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)

    @staticmethod
    def _lock(stream: BinaryIO) -> None:
        if os.name != "nt":
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            return

        import msvcrt

        retryable = {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
        while True:
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                if exc.errno not in retryable:
                    raise
                time.sleep(0.05)

    @staticmethod
    def _unlock(stream: BinaryIO) -> None:
        stream.seek(0)
        if os.name != "nt":
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            return

        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


__all__ = ["SQLiteWriteFence"]
