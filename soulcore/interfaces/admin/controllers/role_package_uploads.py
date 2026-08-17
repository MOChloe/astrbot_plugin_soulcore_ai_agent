"""Bounded chunk transport for ``.soulcore-role`` uploads."""

from __future__ import annotations

import asyncio
import base64
import os
import time
import uuid
import zlib
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from ....features.role_package.domain import (
    MAX_ARCHIVE_BYTES,
    ROLE_PACKAGE_EXTENSION,
    UPLOAD_CHUNK_BYTES,
    RolePackageError,
)

_UPLOAD_TTL_SECONDS = 30 * 60


@dataclass(frozen=True, slots=True)
class CompletedRolePackageUpload:
    path: Path
    filename: str
    crc32: str


@dataclass(slots=True)
class _Upload:
    upload_id: str
    role_ref: str
    profile_id: str
    filename: str
    expected_size: int
    path: Path
    next_index: int = 0
    received_size: int = 0
    running_crc32: int = 0
    last_chunk_size: int = 0
    last_chunk_crc32: str = ""
    updated_at: float = 0.0


class RolePackageUploadTransport:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve(strict=False)
        self._sessions: dict[str, _Upload] = {}
        self._lock = asyncio.Lock()
        self._startup_cleanup_pending = True

    async def prepare_and_expire(self) -> None:
        """Run startup and lazy TTL cleanup from every role-package request."""

        async with self._lock:
            await asyncio.to_thread(self._prepare_root)
            self._expire()

    async def begin(
        self,
        *,
        role_ref: str,
        profile_id: str,
        expected_size: int,
        filename: str,
    ) -> dict[str, int | str | bool]:
        if expected_size < 1 or expected_size > MAX_ARCHIVE_BYTES:
            raise RolePackageError("角色包必须包含内容且不能超过 128 MiB。")
        clean_name = Path(str(filename or "")).name[:180]
        if not clean_name.casefold().endswith(ROLE_PACKAGE_EXTENSION):
            raise RolePackageError("请选择扩展名为 .soulcore-role 的角色包。")
        async with self._lock:
            await asyncio.to_thread(self._prepare_root)
            self._expire()
            self._discard_owner(profile_id)
            upload_id = "rpu_" + uuid.uuid4().hex
            path = self.root / f"{upload_id}.part"
            await asyncio.to_thread(_create_empty, path)
            self._sessions[upload_id] = _Upload(
                upload_id=upload_id,
                role_ref=role_ref,
                profile_id=profile_id,
                filename=clean_name,
                expected_size=int(expected_size),
                path=path,
                updated_at=time.monotonic(),
            )
        return {"ok": True, "upload_id": upload_id, "chunk_bytes": UPLOAD_CHUNK_BYTES}

    async def append(
        self,
        *,
        role_ref: str,
        profile_id: str,
        upload_id: str,
        chunk_index: int,
        encoded_chunk: str,
        expected_chunk_size: int,
        expected_chunk_crc32: str,
    ) -> dict[str, int | str | bool]:
        raw = _decode_chunk(encoded_chunk)
        if not raw or len(raw) > UPLOAD_CHUNK_BYTES:
            raise RolePackageError("角色包上传分块为空或过大，请重新选择文件。")
        checksum = _crc32(raw)
        if len(raw) != expected_chunk_size or checksum != expected_chunk_crc32.lower():
            raise RolePackageError("角色包上传分块校验失败，请重新上传。")
        async with self._lock:
            self._expire()
            session = self._require(upload_id, role_ref, profile_id)
            if chunk_index == session.next_index - 1:
                if len(raw) != session.last_chunk_size or checksum != session.last_chunk_crc32:
                    raise RolePackageError("重试分块与已经接收的内容不一致。")
                return _receipt(session, checksum)
            if chunk_index != session.next_index:
                raise RolePackageError("角色包上传分块顺序不完整，请重新上传。")
            if session.received_size + len(raw) > session.expected_size:
                raise RolePackageError("收到的角色包数据超过浏览器声明大小。")
            await asyncio.to_thread(_append_bytes, session.path, raw)
            session.received_size += len(raw)
            session.next_index += 1
            session.running_crc32 = zlib.crc32(raw, session.running_crc32)
            session.last_chunk_size = len(raw)
            session.last_chunk_crc32 = checksum
            session.updated_at = time.monotonic()
            return _receipt(session, checksum)

    async def finish(
        self,
        *,
        role_ref: str,
        profile_id: str,
        upload_id: str,
        expected_crc32: str,
    ) -> CompletedRolePackageUpload:
        async with self._lock:
            self._expire()
            session = self._require(upload_id, role_ref, profile_id)
            if session.received_size != session.expected_size:
                raise RolePackageError("角色包尚未完整上传。")
            checksum = _format_crc32(session.running_crc32)
            if checksum != str(expected_crc32 or "").strip().lower():
                self._sessions.pop(upload_id, None)
                _safe_unlink(session.path)
                raise RolePackageError("角色包传输校验失败，已删除临时文件。")
            await asyncio.to_thread(_sync_file, session.path)
            completed = session.path.with_suffix(ROLE_PACKAGE_EXTENSION)
            await asyncio.to_thread(os.replace, session.path, completed)
            self._sessions.pop(upload_id, None)
        return CompletedRolePackageUpload(completed, session.filename, checksum)

    async def abort(
        self,
        *,
        role_ref: str,
        profile_id: str,
        upload_id: str,
    ) -> None:
        async with self._lock:
            session = self._sessions.get(upload_id)
            if session is None:
                return
            self._require(upload_id, role_ref, profile_id)
            self._sessions.pop(upload_id, None)
            _safe_unlink(session.path)

    async def abort_owner(self, *, profile_id: str) -> None:
        """Discard the single in-flight upload owned by one existing role."""

        async with self._lock:
            self._discard_owner(profile_id)

    def _require(self, upload_id: str, role_ref: str, profile_id: str) -> _Upload:
        session = self._sessions.get(str(upload_id or ""))
        if session is None:
            raise RolePackageError("角色包上传已经失效，请重新选择文件。")
        if session.role_ref != role_ref or session.profile_id != profile_id:
            raise RolePackageError("角色包上传不属于当前角色。")
        return session

    def _discard_owner(self, profile_id: str) -> None:
        for upload_id, session in tuple(self._sessions.items()):
            if session.profile_id == profile_id:
                self._sessions.pop(upload_id, None)
                _safe_unlink(session.path)

    def _expire(self) -> None:
        cutoff = time.monotonic() - _UPLOAD_TTL_SECONDS
        for upload_id, session in tuple(self._sessions.items()):
            if session.updated_at < cutoff:
                self._sessions.pop(upload_id, None)
                _safe_unlink(session.path)

    def _prepare_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self._startup_cleanup_pending:
            return
        for path in self.root.iterdir():
            if path.is_file() and (
                path.name.startswith("rpu_") or path.name.startswith("preview_")
            ):
                _safe_unlink(path)
        self._startup_cleanup_pending = False


def _decode_chunk(value: str) -> bytes:
    try:
        encoded = str(value or "").strip()
        encoded += "=" * (-len(encoded) % 4)
        return base64.b64decode(encoded, altchars=b"-_", validate=True)
    except Exception as exc:
        raise RolePackageError("角色包上传分块无法读取。") from exc


def _receipt(session: _Upload, checksum: str) -> dict[str, int | str | bool]:
    return {
        "ok": True,
        "upload_id": session.upload_id,
        "received_bytes": session.received_size,
        "next_index": session.next_index,
        "chunk_crc32": checksum,
        "file_crc32": _format_crc32(session.running_crc32),
    }


def _crc32(value: bytes) -> str:
    return _format_crc32(zlib.crc32(value))


def _format_crc32(value: int) -> str:
    return f"{int(value) & 0xFFFFFFFF:08x}"


def _create_empty(path: Path) -> None:
    with path.open("xb"):
        pass


def _append_bytes(path: Path, data: bytes) -> None:
    with path.open("ab") as handle:
        handle.write(data)
        handle.flush()


def _sync_file(path: Path) -> None:
    # Windows rejects FlushFileBuffers (used by os.fsync) for read-only file
    # handles. Open without truncation but with write access on every platform.
    with path.open("rb+") as handle:
        os.fsync(handle.fileno())


def _safe_unlink(path: Path) -> None:
    with suppress(FileNotFoundError):
        path.unlink()


__all__ = [
    "CompletedRolePackageUpload",
    "RolePackageUploadTransport",
]
