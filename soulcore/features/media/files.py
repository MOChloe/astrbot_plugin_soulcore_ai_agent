from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol, TypeVar

from .domain import (
    MediaAsset,
    StoredMediaFile,
)
from .inspection import (
    MAX_IMAGE_BYTES,
    MAX_INBOUND_ATTACHMENT_BYTES,
    inspect_image_bytes,
    inspect_inbound_attachment_bytes,
)

_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_MIME_ALIASES = {"image/jpg": "image/jpeg", "image/x-png": "image/png"}
_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_INBOUND_MEDIA_KINDS = frozenset({"audio", "file", "video"})
_T = TypeVar("_T")


class _Missing:
    pass


_MISSING = _Missing()


async def _drain_cancelled_owner(
    task: asyncio.Task[_T],
    cancellation: asyncio.CancelledError,
    *,
    operation: str,
) -> _T | _Missing:
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                if task.cancelled():
                    cancellation.add_note(f"{operation} was cancelled during cancellation drain")
                    return _MISSING
                try:
                    return task.result()
                except BaseException as exc:
                    cancellation.add_note(
                        f"{operation} failed during cancellation drain: {type(exc).__name__}"
                    )
                    return _MISSING
            # A repeated owner cancellation must remain pending until the
            # already-running thread or cleanup owner reaches a terminal state.
            continue
        except BaseException as exc:
            cancellation.add_note(
                f"{operation} failed during cancellation drain: {type(exc).__name__}"
            )
            return _MISSING


async def _run_cleanup(
    cleanup_after_cancel: Callable[[_T], Awaitable[None]],
    stored: _T,
) -> None:
    await cleanup_after_cancel(stored)


async def await_cancellation_safe_file_store(
    store_call: Callable[..., _T],
    *,
    cleanup_after_cancel: Callable[[_T], Awaitable[None]],
    **kwargs: Any,
) -> _T:
    """Drain a sync store thread and its cleanup before propagating cancellation."""

    owner = asyncio.create_task(asyncio.to_thread(store_call, **kwargs))
    try:
        return await asyncio.shield(owner)
    except asyncio.CancelledError as cancellation:
        stored = await _drain_cancelled_owner(
            owner,
            cancellation,
            operation="media file store",
        )
        if not isinstance(stored, _Missing):
            cleanup_owner = asyncio.create_task(_run_cleanup(cleanup_after_cancel, stored))
            await _drain_cancelled_owner(
                cleanup_owner,
                cancellation,
                operation="cancelled media file cleanup",
            )
        raise


def _temporary_sibling(target: Path) -> Path:
    """Keep atomic-write names short enough for nested Windows plugin data roots."""

    return target.parent / f".tmp-{uuid.uuid4().hex[:16]}"


def _filesystem_path(path: Path) -> Path:
    """Use the Win32 extended-path namespace for deeply nested data roots."""

    if os.name != "nt":
        return path
    raw = str(path)
    if raw.startswith("\\\\?\\"):
        return path
    if raw.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + raw[2:])
    return Path("\\\\?\\" + raw)


class StoredAssetFile(Protocol):
    storage_relpath: str
    byte_size: int
    sha256: str


class MediaFileStore:
    """Filesystem boundary: only opaque relative paths escape this class."""

    def __init__(self, root: str | Path, *, maximum_bytes: int = MAX_IMAGE_BYTES) -> None:
        self.root = Path(root).resolve(strict=False)
        self.maximum_bytes = int(maximum_bytes)

    def store_bytes(
        self,
        *,
        asset_id: str,
        profile_id: str,
        instance_id: str,
        data: bytes,
        declared_mime: str | None = None,
    ) -> StoredMediaFile:
        stored = self.plan_store_bytes(
            asset_id=asset_id,
            profile_id=profile_id,
            instance_id=instance_id,
            data=data,
            declared_mime=declared_mime,
        )
        self._write_planned_bytes(stored, data)
        return stored

    def plan_store_bytes(
        self,
        *,
        asset_id: str,
        profile_id: str,
        instance_id: str,
        data: bytes,
        declared_mime: str | None = None,
    ) -> StoredMediaFile:
        """Validate bytes and derive their final path without writing the file."""

        return self._plan_image_bytes(
            asset_id=asset_id,
            profile_id=profile_id,
            instance_id=instance_id,
            data=data,
            declared_mime=declared_mime,
            maximum_bytes=self.maximum_bytes,
            enforce_animation_limits=True,
        )

    def _plan_image_bytes(
        self,
        *,
        asset_id: str,
        profile_id: str,
        instance_id: str,
        data: bytes,
        declared_mime: str | None,
        maximum_bytes: int | None,
        enforce_animation_limits: bool,
        validate_decoded: bool = True,
    ) -> StoredMediaFile:
        if not _ASSET_ID_RE.fullmatch(asset_id):
            raise ValueError("asset_id must be an opaque safe identifier")
        mime, extension, width, height, frames = inspect_image_bytes(
            data,
            declared_mime=declared_mime,
            maximum_bytes=maximum_bytes,
            enforce_animation_limits=enforce_animation_limits,
            validate_decoded=validate_decoded,
        )
        digest = hashlib.sha256(data).hexdigest()
        profile_bucket = hashlib.sha256(profile_id.encode("utf-8")).hexdigest()[:16]
        instance_bucket = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:20]
        relative = (
            Path(profile_bucket)
            / instance_bucket
            / digest[:2]
            / f"{asset_id}-{digest[:16]}{extension}"
        )
        return StoredMediaFile(
            asset_id=asset_id,
            relative_path=relative.as_posix(),
            mime_type=mime,
            file_extension=extension,
            sha256=digest,
            byte_size=len(data),
            width=width,
            height=height,
            frame_count=frames,
        )

    def _write_planned_bytes(self, stored: StoredMediaFile, data: bytes) -> None:
        digest = hashlib.sha256(data).hexdigest()
        if digest != stored.sha256 or len(data) != int(stored.byte_size):
            raise ValueError("planned media bytes changed before storage")
        target = self._resolve(Path(stored.relative_path))
        filesystem_target = _filesystem_path(target)
        filesystem_target.parent.mkdir(parents=True, exist_ok=True)
        if filesystem_target.exists():
            if hashlib.sha256(filesystem_target.read_bytes()).hexdigest() != digest:
                raise FileExistsError(f"media asset path collision: {stored.asset_id}")
            return
        temporary = _temporary_sibling(filesystem_target)
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, filesystem_target)
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()

    def store_reference_bytes(
        self,
        *,
        asset_id: str,
        profile_id: str,
        instance_id: str,
        data: bytes,
    ) -> StoredMediaFile:
        """Store a bounded, fully validated identity reference without altering its bytes."""

        stored = self.plan_reference_bytes(
            asset_id=asset_id,
            profile_id=profile_id,
            instance_id=instance_id,
            data=data,
        )
        self.write_planned_bytes(stored, data)
        return stored

    def plan_reference_bytes(
        self,
        *,
        asset_id: str,
        profile_id: str,
        instance_id: str,
        data: bytes,
    ) -> StoredMediaFile:
        """Validate and name an identity reference before its durable guard is written."""

        return self._plan_image_bytes(
            asset_id=asset_id,
            profile_id=profile_id,
            instance_id=instance_id,
            data=data,
            declared_mime=None,
            maximum_bytes=self.maximum_bytes,
            enforce_animation_limits=True,
            validate_decoded=True,
        )

    def write_planned_bytes(self, stored: StoredMediaFile, data: bytes) -> None:
        """Publish bytes previously validated by a ``plan_*`` method."""

        self._write_planned_bytes(stored, data)

    def _store_image_bytes(
        self,
        *,
        asset_id: str,
        profile_id: str,
        instance_id: str,
        data: bytes,
        declared_mime: str | None,
        maximum_bytes: int | None,
        enforce_animation_limits: bool,
        validate_decoded: bool = True,
    ) -> StoredMediaFile:
        stored = self._plan_image_bytes(
            asset_id=asset_id,
            profile_id=profile_id,
            instance_id=instance_id,
            data=data,
            declared_mime=declared_mime,
            maximum_bytes=maximum_bytes,
            enforce_animation_limits=enforce_animation_limits,
            validate_decoded=validate_decoded,
        )
        self._write_planned_bytes(stored, data)
        return stored

    def store_sticker_bytes(
        self,
        *,
        asset_id: str,
        profile_id: str,
        data: bytes,
        declared_mime: str | None = None,
    ) -> StoredMediaFile:
        """Store a profile-owned sticker independently from chat media."""

        if not _ASSET_ID_RE.fullmatch(asset_id):
            raise ValueError("asset_id must be an opaque safe identifier")
        mime, extension, width, height, frames = inspect_image_bytes(
            data, declared_mime=declared_mime, maximum_bytes=self.maximum_bytes
        )
        digest = hashlib.sha256(data).hexdigest()
        profile_bucket = hashlib.sha256(profile_id.encode("utf-8")).hexdigest()[:16]
        relative = Path(profile_bucket) / "stickers" / digest[:2] / f"{asset_id}{extension}"
        target = self._resolve(relative)
        filesystem_target = _filesystem_path(target)
        filesystem_target.parent.mkdir(parents=True, exist_ok=True)
        if filesystem_target.exists():
            if hashlib.sha256(filesystem_target.read_bytes()).hexdigest() != digest:
                raise FileExistsError(f"sticker asset path collision: {asset_id}")
        else:
            temporary = _temporary_sibling(filesystem_target)
            try:
                with temporary.open("xb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, filesystem_target)
            finally:
                with suppress(FileNotFoundError):
                    temporary.unlink()
        return StoredMediaFile(
            asset_id=asset_id,
            relative_path=relative.as_posix(),
            mime_type=mime,
            file_extension=extension,
            sha256=digest,
            byte_size=len(data),
            width=width,
            height=height,
            frame_count=frames,
        )

    def store_inbound_attachment_bytes(
        self,
        *,
        asset_id: str,
        profile_id: str,
        instance_id: str,
        data: bytes,
        media_kind: str,
        declared_mime: str | None = None,
    ) -> StoredMediaFile:
        if not _ASSET_ID_RE.fullmatch(asset_id):
            raise ValueError("asset_id must be an opaque safe identifier")
        mime, extension = inspect_inbound_attachment_bytes(
            data,
            media_kind=media_kind,
            declared_mime=declared_mime,
            maximum_bytes=MAX_INBOUND_ATTACHMENT_BYTES,
        )
        digest = hashlib.sha256(data).hexdigest()
        profile_bucket = hashlib.sha256(profile_id.encode("utf-8")).hexdigest()[:16]
        instance_bucket = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:20]
        relative = (
            Path(profile_bucket)
            / instance_bucket
            / digest[:2]
            / f"{asset_id}-{digest[:16]}{extension}"
        )
        target = self._resolve(relative)
        filesystem_target = _filesystem_path(target)
        filesystem_target.parent.mkdir(parents=True, exist_ok=True)
        if filesystem_target.exists():
            if hashlib.sha256(filesystem_target.read_bytes()).hexdigest() != digest:
                raise FileExistsError(f"media asset path collision: {asset_id}")
        else:
            temporary = _temporary_sibling(filesystem_target)
            try:
                with temporary.open("xb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, filesystem_target)
            finally:
                with suppress(FileNotFoundError):
                    temporary.unlink()
        return StoredMediaFile(
            asset_id=asset_id,
            relative_path=relative.as_posix(),
            mime_type=mime,
            file_extension=extension,
            sha256=digest,
            byte_size=len(data),
        )

    def absolute_path(self, relative_path: str) -> Path:
        return _filesystem_path(self._resolve(Path(relative_path)))

    def verify(self, file: StoredMediaFile | MediaAsset | StoredAssetFile) -> bool:
        relative = file.relative_path if isinstance(file, StoredMediaFile) else file.storage_relpath
        if not relative:
            return False
        path = _filesystem_path(self._resolve(Path(relative)))
        if not path.is_file() or path.stat().st_size != int(file.byte_size):
            return False
        return hashlib.sha256(path.read_bytes()).hexdigest() == file.sha256

    def delete(self, relative_path: str | None) -> bool:
        if not relative_path:
            return False
        resolved = self._resolve(Path(relative_path))
        path = _filesystem_path(resolved)
        try:
            path.unlink()
            removed = True
        except FileNotFoundError:
            removed = False
        parent = resolved.parent
        while parent != self.root:
            try:
                _filesystem_path(parent).rmdir()
            except OSError:
                break
            parent = parent.parent
        return removed

    def delete_scope(self, profile_id: str, instance_id: str | None = None) -> int:
        """Delete a complete opaque scope bucket without trusting DB row counts.

        The bucket is derived from stable hashes, resolved beneath ``root`` and
        removed as one tree.  This also removes files orphaned before their
        metadata transaction completed.
        """

        profile_bucket = hashlib.sha256(profile_id.encode("utf-8")).hexdigest()[:16]
        relative = Path(profile_bucket)
        if instance_id is not None:
            instance_bucket = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:20]
            relative /= instance_bucket
        target = self._resolve(relative)
        filesystem_target = _filesystem_path(target)
        if not filesystem_target.exists():
            return 0
        file_count = sum(1 for item in filesystem_target.rglob("*") if item.is_file())
        shutil.rmtree(filesystem_target)
        parent = target.parent
        while parent != self.root:
            try:
                _filesystem_path(parent).rmdir()
            except OSError:
                break
            parent = parent.parent
        return file_count

    def _resolve(self, relative: Path) -> Path:
        if relative.is_absolute():
            raise ValueError("media path must be relative")
        target = (self.root / relative).resolve(strict=False)
        if not target.is_relative_to(self.root):
            raise ValueError("media path escapes storage root")
        return target
