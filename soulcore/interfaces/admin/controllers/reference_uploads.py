"""Chunked transport for administrator-selected identity references."""

from __future__ import annotations

import asyncio
import base64
import time
import uuid
import zlib
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from ....features.media.inspection import MAX_IMAGE_BYTES
from ..console_errors import ConsoleValidationError

REFERENCE_UPLOAD_CHUNK_BYTES = 128 * 1024
_REFERENCE_UPLOAD_SESSION_TTL_SECONDS = 30 * 60


@dataclass(frozen=True, slots=True)
class CompletedReferenceUpload:
    data: bytes
    filename: str
    label: str
    mime_type: str
    crc32: str


@dataclass(slots=True)
class _ActiveReferenceUpload:
    upload_id: str
    profile_id: str
    scope: str
    filename: str
    label: str
    mime_type: str
    expected_size: int
    path: Path
    next_index: int = 0
    received_size: int = 0
    running_crc32: int = 0
    last_chunk_size: int = 0
    last_chunk_crc32: str = ""
    updated_at: float = 0.0


class ReferenceUploadTransport:
    """Reassemble bounded transport chunks inside one owner-scoped session."""

    def __init__(
        self,
        media_root: str | Path,
        *,
        maximum_bytes: int = MAX_IMAGE_BYTES,
    ) -> None:
        storage_root = Path(media_root).resolve(strict=False)
        self.root = (storage_root / ".reference-uploads").resolve(strict=False)
        if not self.root.is_relative_to(storage_root):
            raise ValueError("reference upload directory escapes media root")
        self._sessions: dict[str, _ActiveReferenceUpload] = {}
        self._lock = asyncio.Lock()
        self._orphan_cleanup_pending = True
        self.maximum_bytes = max(1, int(maximum_bytes))

    async def begin(
        self,
        *,
        profile_id: str,
        scope: str,
        expected_size: int,
        filename: str,
        label: str,
        mime_type: str,
    ) -> dict[str, int | str | bool]:
        if expected_size < 1:
            raise ConsoleValidationError("请选择包含图片内容的角色立绘参考。")
        if expected_size > self.maximum_bytes:
            raise ConsoleValidationError(
                f"角色立绘参考不能超过 {self.maximum_bytes // (1024 * 1024)} MiB。"
            )
        async with self._lock:
            self._expire_sessions()
            await asyncio.to_thread(self._prepare_root)
            self._discard_owner_session(profile_id, scope)
            upload_id = "ru_" + uuid.uuid4().hex
            path = self.root / f"{upload_id}.part"
            await asyncio.to_thread(self._create_empty_file, path)
            self._sessions[upload_id] = _ActiveReferenceUpload(
                upload_id=upload_id,
                profile_id=profile_id,
                scope=scope,
                filename=Path(filename).name[:160],
                label=label.strip()[:80],
                mime_type=mime_type.strip()[:120],
                expected_size=expected_size,
                path=path,
                updated_at=time.monotonic(),
            )
        return {
            "ok": True,
            "upload_id": upload_id,
            "chunk_bytes": REFERENCE_UPLOAD_CHUNK_BYTES,
        }

    async def append(
        self,
        *,
        profile_id: str,
        scope: str,
        upload_id: str,
        chunk_index: int,
        encoded_chunk: str,
        expected_chunk_size: int | None = None,
        expected_chunk_crc32: str = "",
    ) -> dict[str, int | str | bool]:
        raw = self._decode_chunk(encoded_chunk)
        if not raw:
            raise ConsoleValidationError("角色立绘参考包含空的上传分块，请重新选择图片。")
        if len(raw) > REFERENCE_UPLOAD_CHUNK_BYTES:
            raise ConsoleValidationError("角色立绘参考的上传分块过大，请重新选择图片后重试。")
        checksum = self._crc32(raw)
        if expected_chunk_size is not None and len(raw) != expected_chunk_size:
            raise ConsoleValidationError("角色立绘参考的上传分块长度校验不一致，请重新上传。")
        normalized_expected_crc32 = str(expected_chunk_crc32 or "").strip().lower()
        if normalized_expected_crc32 and checksum != normalized_expected_crc32:
            raise ConsoleValidationError("角色立绘参考的上传分块内容校验不一致，请重新上传。")
        async with self._lock:
            self._expire_sessions()
            session = self._require_owner(upload_id, profile_id, scope)
            if chunk_index == session.next_index - 1:
                if len(raw) != session.last_chunk_size or checksum != session.last_chunk_crc32:
                    raise ConsoleValidationError(
                        "角色立绘参考的重试分块与已经接收的内容不一致，请重新上传。"
                    )
                return self._chunk_receipt(session, checksum)
            if chunk_index != session.next_index:
                raise ConsoleValidationError("角色立绘参考的上传分块顺序不完整，请重新上传。")
            if session.received_size + len(raw) > session.expected_size:
                raise ConsoleValidationError("角色立绘参考收到的图片数据超过原文件大小。")
            await asyncio.to_thread(self._append_bytes, session.path, raw)
            session.received_size += len(raw)
            session.next_index += 1
            session.running_crc32 = zlib.crc32(raw, session.running_crc32)
            session.last_chunk_size = len(raw)
            session.last_chunk_crc32 = checksum
            session.updated_at = time.monotonic()
            return self._chunk_receipt(session, checksum)

    async def finish(
        self,
        *,
        profile_id: str,
        scope: str,
        upload_id: str,
        expected_crc32: str = "",
    ) -> CompletedReferenceUpload:
        async with self._lock:
            self._expire_sessions()
            session = self._require_owner(upload_id, profile_id, scope)
            if session.received_size != session.expected_size:
                raise ConsoleValidationError("角色立绘参考尚未完整上传，请等待传输完成后重试。")
            checksum = self._format_crc32(session.running_crc32)
            normalized_expected_crc32 = str(expected_crc32 or "").strip().lower()
            if normalized_expected_crc32 and checksum != normalized_expected_crc32:
                self._sessions.pop(upload_id, None)
                self._safe_unlink(session.path)
                raise ConsoleValidationError(
                    "角色立绘参考在浏览器与服务器之间传输后内容校验不一致，"
                    "已停止保存，请重新选择原图。"
                )
            try:
                data = await asyncio.to_thread(session.path.read_bytes)
            finally:
                self._sessions.pop(upload_id, None)
                self._safe_unlink(session.path)
        if len(data) != session.expected_size:
            raise ConsoleValidationError("角色立绘参考的临时文件不完整，请重新上传。")
        if self._crc32(data) != checksum:
            raise ConsoleValidationError("角色立绘参考的临时文件内容校验不一致，请重新上传。")
        return CompletedReferenceUpload(
            data=data,
            filename=session.filename,
            label=session.label,
            mime_type=session.mime_type,
            crc32=checksum,
        )

    async def abort(self, *, profile_id: str, scope: str, upload_id: str) -> None:
        async with self._lock:
            session = self._sessions.get(upload_id)
            if session is None:
                return
            self._require_owner(upload_id, profile_id, scope)
            self._sessions.pop(upload_id, None)
            self._safe_unlink(session.path)

    @staticmethod
    def _decode_chunk(encoded_chunk: str) -> bytes:
        try:
            encoded = str(encoded_chunk or "").strip()
            encoded += "=" * (-len(encoded) % 4)
            return base64.b64decode(encoded, altchars=b"-_", validate=True)
        except Exception as exc:
            raise ConsoleValidationError("角色立绘参考的上传分块无法读取。") from exc

    @classmethod
    def _chunk_receipt(
        cls,
        session: _ActiveReferenceUpload,
        checksum: str,
    ) -> dict[str, int | str | bool]:
        return {
            "ok": True,
            "upload_id": session.upload_id,
            "received_bytes": session.received_size,
            "next_index": session.next_index,
            "chunk_crc32": checksum,
            "file_crc32": cls._format_crc32(session.running_crc32),
        }

    @staticmethod
    def _crc32(data: bytes) -> str:
        return ReferenceUploadTransport._format_crc32(zlib.crc32(data))

    @staticmethod
    def _format_crc32(value: int) -> str:
        return f"{int(value) & 0xFFFFFFFF:08x}"

    def _require_owner(
        self,
        upload_id: str,
        profile_id: str,
        scope: str,
    ) -> _ActiveReferenceUpload:
        session = self._sessions.get(upload_id)
        if session is None:
            raise ConsoleValidationError("角色立绘参考上传已失效，请重新选择图片。")
        if session.profile_id != profile_id or session.scope != scope:
            raise ConsoleValidationError("角色立绘参考上传不属于当前角色范围。")
        return session

    def _discard_owner_session(self, profile_id: str, scope: str) -> None:
        for upload_id, session in tuple(self._sessions.items()):
            if session.profile_id == profile_id and session.scope == scope:
                self._sessions.pop(upload_id, None)
                self._safe_unlink(session.path)

    def _expire_sessions(self) -> None:
        cutoff = time.monotonic() - _REFERENCE_UPLOAD_SESSION_TTL_SECONDS
        for upload_id, session in tuple(self._sessions.items()):
            if session.updated_at < cutoff:
                self._sessions.pop(upload_id, None)
                self._safe_unlink(session.path)

    def _prepare_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self._orphan_cleanup_pending:
            return
        for path in self.root.glob("ru_*.part"):
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                continue
        self._orphan_cleanup_pending = False

    @staticmethod
    def _create_empty_file(path: Path) -> None:
        with path.open("xb"):
            pass

    @staticmethod
    def _append_bytes(path: Path, data: bytes) -> None:
        with path.open("ab") as handle:
            handle.write(data)
            handle.flush()

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        with suppress(FileNotFoundError):
            path.unlink()


__all__ = [
    "CompletedReferenceUpload",
    "REFERENCE_UPLOAD_CHUNK_BYTES",
    "ReferenceUploadTransport",
]
