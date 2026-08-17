"""Player-facing role-package export, preview, and atomic apply workflow."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

from ....features.background.service import (
    normalize_creative_boundary_input,
    normalize_world_lore_input,
)
from ....features.character_model.ports import CharacterModelRepositoryPort
from ....features.files.runtime_cleanup import drain_runtime_file_cleanup
from ....features.media import (
    MAX_ANIMATION_DECODED_PIXELS,
    MAX_ANIMATION_DURATION_MS,
    MAX_ANIMATION_FRAMES,
    MAX_IMAGE_BYTES,
    generate_media_asset_id,
    inspect_animation_bytes,
    inspect_image_bytes,
)
from ....features.media.ports import MediaRepositoryPort
from ....features.media.storage import MediaStorageCoordinator
from ....features.role_package.domain import (
    ROLE_PACKAGE_EXTENSION,
    ROLE_PACKAGE_MIME,
    ExportSession,
    ImportState,
    PortraitMutation,
    PreviewSession,
    RolePackageError,
)
from ....features.role_package.format import build_role_package, read_role_package
from ....features.role_package.model_patch import merge_import_state, sparse_role_document
from ....features.role_package.ports import RolePackageRepositoryPort
from ....features.role_package.preview import build_import_preview
from ..downloads import PageFileDownload
from .role_package_uploads import RolePackageUploadTransport

_EXPORT_TTL_SECONDS = 10 * 60
_PREVIEW_TTL_SECONDS = 30 * 60
_SAFE_FILENAME = re.compile(r"[<>:\"/\\|?*\x00-\x1f]+")
logger = logging.getLogger(__name__)


class RolePackageController:
    def __init__(
        self,
        repository: RolePackageRepositoryPort,
        *,
        character_models: CharacterModelRepositoryPort,
        media_repository: MediaRepositoryPort,
        media_storage: MediaStorageCoordinator,
        file_artifacts: Any,
        data_dir: str | Path,
        plugin_version: str,
        notify_background: Callable[[], Any] | None = None,
    ) -> None:
        self.repository = repository
        self.character_models = character_models
        self.media_repository = media_repository
        self.media_storage = media_storage
        self.file_artifacts = file_artifacts
        self.plugin_version = str(plugin_version)
        base = (Path(data_dir).resolve(strict=False) / "role_packages").resolve(strict=False)
        self.root = base
        self.exports_root = base / "exports"
        self.uploads = RolePackageUploadTransport(base / "uploads")
        self.notify_background = notify_background
        self._exports: dict[str, ExportSession] = {}
        self._previews: dict[str, PreviewSession] = {}
        self._lock = asyncio.Lock()
        self._apply_lock = asyncio.Lock()
        self._startup_cleanup_pending = True

    async def export_prepare(
        self,
        *,
        role_ref: str,
        profile_id: str,
    ) -> dict[str, Any]:
        await self._prepare_and_expire()
        await self.character_models.load(profile_id)
        snapshot = await self.repository.snapshot(profile_id)
        role = sparse_role_document(snapshot)
        portraits: dict[str, tuple[bytes, str, str]] = {}
        for scope in ("private", "group"):
            current = snapshot.portraits.get(scope)
            if current is None:
                continue
            label = "私聊立绘" if scope == "private" else "群聊立绘"
            if current.file_status != "AVAILABLE":
                raise RolePackageError(f"{label}尚未处于可用状态，已停止导出。")
            valid = await asyncio.to_thread(self.media_storage.store.verify, current)
            if not valid:
                raise RolePackageError(f"{label}文件缺失或校验失败，已停止导出。")
            path = self.media_storage.store.absolute_path(current.storage_relpath)
            raw = await asyncio.to_thread(path.read_bytes)
            if len(raw) != current.byte_size or hashlib.sha256(raw).hexdigest() != current.sha256:
                raise RolePackageError(f"{label}在导出时发生变化，已停止导出。")
            image = await asyncio.to_thread(_inspect_package_image, raw, current.mime_type)
            if image["mime_type"] != current.mime_type:
                raise RolePackageError(f"{label}记录的图片格式与文件不一致，已停止导出。")
            portraits[scope] = (raw, current.mime_type, current.label)

        token = "rpe_" + uuid.uuid4().hex
        path = self.exports_root / f"{token}{ROLE_PACKAGE_EXTENSION}"
        await asyncio.to_thread(
            build_role_package,
            path,
            title=snapshot.title,
            generator_version=self.plugin_version,
            role_document=role,
            portrait_assets=portraits,
        )
        filename = _download_filename(snapshot.title)
        session = ExportSession(
            token=token,
            role_ref=role_ref,
            profile_id=profile_id,
            path=path,
            filename=filename,
            expires_at=time.monotonic() + _EXPORT_TTL_SECONDS,
        )
        async with self._lock:
            self._exports[token] = session
        return {
            "ok": True,
            "download_token": token,
            "filename": filename,
            "expires_in_seconds": _EXPORT_TTL_SECONDS,
        }

    async def download(
        self,
        *,
        role_ref: str,
        profile_id: str,
        download_token: str,
    ) -> PageFileDownload:
        await self._prepare_and_expire()
        async with self._lock:
            session = self._exports.get(str(download_token or ""))
            if session is None or session.expires_at <= time.monotonic():
                raise RolePackageError("角色包下载已经过期，请重新导出。")
            if session.role_ref != role_ref or session.profile_id != profile_id:
                raise RolePackageError("角色包下载令牌不属于当前角色。")
            if not session.path.is_file():
                self._exports.pop(session.token, None)
                raise RolePackageError("角色包临时文件已经失效，请重新导出。")
        return PageFileDownload(
            path=session.path,
            filename=session.filename,
            content_type=ROLE_PACKAGE_MIME,
            headers={
                "Cache-Control": "private, no-store, max-age=0",
                "X-Content-Type-Options": "nosniff",
            },
        )

    async def upload(
        self,
        *,
        role_ref: str,
        profile_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        await self._prepare_and_expire()
        action = str(payload.get("action") or "").strip().lower()
        if action == "upload_begin":
            return await self.uploads.begin(
                role_ref=role_ref,
                profile_id=profile_id,
                expected_size=_integer(payload.get("file_size"), "file_size", minimum=1),
                filename=str(payload.get("filename") or ""),
            )
        if action == "chunk":
            return await self.uploads.append(
                role_ref=role_ref,
                profile_id=profile_id,
                upload_id=str(payload.get("upload_id") or ""),
                chunk_index=_integer(payload.get("chunk_index"), "chunk_index", minimum=0),
                encoded_chunk=str(payload.get("chunk_data") or ""),
                expected_chunk_size=_integer(payload.get("chunk_size"), "chunk_size", minimum=1),
                expected_chunk_crc32=str(payload.get("chunk_crc32") or ""),
            )
        if action == "abort":
            await self._abort(role_ref, profile_id, payload)
            return {"ok": True, "aborted": True}
        if action != "finish":
            raise RolePackageError("角色包上传操作无效。")
        completed = await self.uploads.finish(
            role_ref=role_ref,
            profile_id=profile_id,
            upload_id=str(payload.get("upload_id") or ""),
            expected_crc32=str(payload.get("file_crc32") or ""),
        )
        try:
            package = await asyncio.to_thread(read_role_package, completed.path)
            await asyncio.to_thread(_inspect_all_package_images, package)
            await self.character_models.load(profile_id)
            snapshot = await self.repository.snapshot(profile_id)
            state = _normalize_import_state(merge_import_state(package, snapshot), snapshot)
            preview = build_import_preview(package, snapshot, state)
        except Exception:
            _safe_unlink(completed.path)
            raise
        token = "rpi_" + uuid.uuid4().hex
        session = PreviewSession(
            token=token,
            role_ref=role_ref,
            profile_id=profile_id,
            package_path=completed.path,
            package_sha256=package.archive_sha256,
            snapshot=snapshot,
            state=state,
            preview=preview,
            expires_at=time.monotonic() + _PREVIEW_TTL_SECONDS,
        )
        async with self._lock:
            self._discard_profile_previews(profile_id)
            self._previews[token] = session
        return {
            "ok": True,
            "confirmation_token": token,
            "expires_in_seconds": _PREVIEW_TTL_SECONDS,
            "preview": preview,
        }

    async def apply(
        self,
        *,
        role_ref: str,
        profile_id: str,
        confirmation_token: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        async with self._apply_lock:
            await self._prepare_and_expire()
            session = await self._preview_session(role_ref, profile_id, confirmation_token)
            package = await asyncio.to_thread(read_role_package, session.package_path)
            if package.archive_sha256 != session.package_sha256:
                raise RolePackageError("暂存角色包在确认前发生变化，请重新上传。")
            image_info = await asyncio.to_thread(_inspect_all_package_images, package)
            mutations: dict[str, PortraitMutation] = {}
            planned: list[PortraitMutation] = []
            try:
                for scope in ("private", "group"):
                    if not session.state.portrait_changed.get(scope):
                        continue
                    action = session.state.portrait_actions[scope]
                    if action["action"] == "clear":
                        mutations[scope] = PortraitMutation(scope=scope, action="clear")
                        continue
                    asset = package.assets[scope]
                    stored = await asyncio.to_thread(
                        self.media_storage.store.plan_reference_bytes,
                        asset_id=generate_media_asset_id(),
                        profile_id=profile_id,
                        instance_id=f"character-identity:{scope}",
                        data=asset.data,
                    )
                    guard = await self.media_repository.guard_unregistered_media_file(
                        profile_id,
                        f"character-identity:{scope}",
                        stored,
                        reason="ROLE_PACKAGE_IDENTITY_REGISTRATION",
                    )
                    mutation = PortraitMutation(
                        scope=scope,
                        action="replace",
                        label=str(action.get("label") or ""),
                        stored=stored,
                        cleanup_guard_id=guard.cleanup_id,
                        duration_ms=int(image_info[scope]["duration_ms"]),
                    )
                    planned.append(mutation)
                    await asyncio.to_thread(
                        self.media_storage.store.write_planned_bytes,
                        stored,
                        asset.data,
                    )
                    mutations[scope] = mutation
                result = await self.repository.apply(
                    profile_id=profile_id,
                    expected=session.snapshot,
                    state=session.state,
                    portrait_mutations=mutations,
                    package_sha256=session.package_sha256,
                    idempotency_key=idempotency_key,
                )
            except Exception:
                await self._discard_planned_files(planned)
                raise
            if result.replayed:
                await self._discard_planned_files(planned)
            cleanup_pending = bool(result.cleanup_targets)
            try:
                cleanup_pending = await self._cleanup_replaced_portraits(result.cleanup_targets)
            except Exception:
                logger.exception(
                    "role package committed but replaced portrait cleanup could not start",
                    extra={"profile_id": profile_id},
                )
            if session.state.world_changed and self.notify_background is not None:
                try:
                    notice = self.notify_background()
                    if asyncio.iscoroutine(notice):
                        await notice
                except Exception:
                    logger.exception(
                        "role package committed but background scheduler notification failed",
                        extra={"profile_id": profile_id},
                    )
            async with self._lock:
                self._previews.pop(session.token, None)
            _safe_unlink(session.package_path)
            labels = {
                "character": "角色资料与行为",
                "world": "世界资料与边界",
                "portraits": "角色立绘",
            }
            return {
                "ok": True,
                "changed": result.changed,
                "replayed": result.replayed,
                "target_role_name": session.snapshot.title,
                "changed_sections": [labels[item] for item in result.changed_sections],
                "message": (
                    "角色包内容与当前角色一致，没有需要写入的变化。"
                    if not result.changed
                    else "角色包已经导入当前角色。"
                ),
                "cleanup_pending": cleanup_pending,
            }

    async def _abort(
        self,
        role_ref: str,
        profile_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        upload_id = str(payload.get("upload_id") or "")
        if upload_id:
            await self.uploads.abort(
                role_ref=role_ref,
                profile_id=profile_id,
                upload_id=upload_id,
            )
        else:
            await self.uploads.abort_owner(profile_id=profile_id)
        token = str(payload.get("confirmation_token") or "")
        if token:
            async with self._lock:
                session = self._previews.get(token)
                if session is None:
                    return
                if session.role_ref != role_ref or session.profile_id != profile_id:
                    raise RolePackageError("导入预览不属于当前角色。")
                self._previews.pop(token, None)
                _safe_unlink(session.package_path)
        else:
            async with self._lock:
                self._discard_profile_previews(profile_id)

    async def _preview_session(
        self,
        role_ref: str,
        profile_id: str,
        token: str,
    ) -> PreviewSession:
        async with self._lock:
            session = self._previews.get(str(token or ""))
            if session is None or session.expires_at <= time.monotonic():
                raise RolePackageError("导入预览已经过期，请重新上传角色包。")
            if session.role_ref != role_ref or session.profile_id != profile_id:
                raise RolePackageError("导入预览不属于当前角色。")
            return session

    async def _discard_planned_files(self, values: Sequence[PortraitMutation]) -> None:
        for mutation in values:
            stored = mutation.stored
            if stored is None:
                continue
            deleted = False
            try:
                await asyncio.to_thread(
                    self.media_storage.store.delete,
                    stored.relative_path,
                )
                deleted = True
            except Exception:
                # The durable guard intentionally remains for maintenance retry.
                continue
            if deleted:
                with suppress(Exception):
                    await self.media_repository.complete_runtime_file_cleanup(
                        mutation.cleanup_guard_id
                    )

    async def _cleanup_replaced_portraits(self, paths: Sequence[str]) -> bool:
        if not paths:
            return False
        result = await drain_runtime_file_cleanup(
            self.media_repository,
            media_store=self.media_storage.store,
            file_artifacts=self.file_artifacts,
            targets=tuple(("MEDIA", path) for path in paths),
            limit=len(paths),
            raise_on_failure=False,
        )
        return bool(result.failed or result.completed < result.attempted)

    async def _prepare_and_expire(self) -> None:
        await self.uploads.prepare_and_expire()
        async with self._lock:
            await asyncio.to_thread(self._prepare_roots)
            now = time.monotonic()
            for token, session in tuple(self._exports.items()):
                if session.expires_at <= now:
                    self._exports.pop(token, None)
                    _safe_unlink(session.path)
            for token, session in tuple(self._previews.items()):
                if session.expires_at <= now:
                    self._previews.pop(token, None)
                    _safe_unlink(session.package_path)

    def _prepare_roots(self) -> None:
        self.exports_root.mkdir(parents=True, exist_ok=True)
        if not self._startup_cleanup_pending:
            return
        for path in self.exports_root.glob(f"rpe_*{ROLE_PACKAGE_EXTENSION}"):
            _safe_unlink(path)
        self._startup_cleanup_pending = False

    def _discard_profile_previews(self, profile_id: str) -> None:
        for token, session in tuple(self._previews.items()):
            if session.profile_id == profile_id:
                self._previews.pop(token, None)
                _safe_unlink(session.package_path)


def _normalize_import_state(state: ImportState, snapshot: Any) -> ImportState:
    definition = dict(state.world_definition)
    if state.world_definition_present:
        definition = {
            key: (str(value).strip() if key != "expansion_policy" else str(value))
            for key, value in definition.items()
        }
    lore = state.lore
    if state.lore_present:
        normalized_lore: list[dict[str, Any]] = []
        for item in lore:
            normalized_lore.append(normalize_world_lore_input(**item))
        lore = tuple(normalized_lore)
    boundaries = state.boundaries
    if state.boundaries_present:
        normalized_boundaries: list[dict[str, Any]] = []
        for item in boundaries:
            normalized_boundaries.append(normalize_creative_boundary_input(**item))
        boundaries = tuple(normalized_boundaries)
    world_changed = (
        definition != snapshot.world_definition
        or _canonical_records(lore) != _canonical_records(snapshot.lore)
        or _canonical_records(boundaries) != _canonical_records(snapshot.boundaries)
    )
    changed = state.character_changed or world_changed or any(state.portrait_changed.values())
    return replace(
        state,
        world_definition=definition,
        lore=lore,
        boundaries=boundaries,
        world_changed=world_changed,
        changed=changed,
    )


def _canonical_records(values: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        sorted(
            json.dumps(dict(item), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for item in values
        )
    )


def _inspect_all_package_images(package: Any) -> dict[str, dict[str, Any]]:
    return {
        scope: _inspect_package_image(asset.data, asset.mime_type)
        for scope, asset in package.assets.items()
    }


def _inspect_package_image(data: bytes, declared_mime: str) -> dict[str, Any]:
    try:
        mime, extension, width, height, frames = inspect_image_bytes(
            data,
            declared_mime=declared_mime,
            maximum_bytes=MAX_IMAGE_BYTES,
            enforce_animation_limits=True,
            validate_decoded=True,
        )
        duration = (
            int(
                inspect_animation_bytes(
                    data,
                    maximum_frames=MAX_ANIMATION_FRAMES,
                    maximum_duration_ms=MAX_ANIMATION_DURATION_MS,
                    maximum_decoded_pixels=MAX_ANIMATION_DECODED_PIXELS,
                )["duration_ms"]
            )
            if frames > 1
            else 0
        )
    except (OSError, ValueError) as exc:
        raise RolePackageError("角色包立绘不是可完整读取的受支持图片。") from exc
    if mime != str(declared_mime).strip().lower():
        raise RolePackageError("角色包立绘的声明格式与实际内容不一致。")
    return {
        "mime_type": mime,
        "extension": extension,
        "width": width,
        "height": height,
        "frame_count": frames,
        "duration_ms": duration,
    }


def _integer(value: Any, field: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise RolePackageError(f"{field} 必须是不小于 {minimum} 的整数。")
    return value


def _download_filename(title: str) -> str:
    name = _SAFE_FILENAME.sub("_", str(title or "角色")).strip(" ._")[:80]
    return (name or "角色") + ROLE_PACKAGE_EXTENSION


def _safe_unlink(path: Path) -> None:
    with suppress(FileNotFoundError, PermissionError):
        path.unlink()


__all__ = ["RolePackageController"]
