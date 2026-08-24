"""Media assets and generated-file administrator controller."""

from __future__ import annotations

import asyncio
import base64
import re
from collections.abc import Mapping
from datetime import UTC
from typing import Any

from ....features.files.ports import FileRepositoryPort
from ....features.files.service import FileArtifactService
from ....features.media.ports import MediaRepositoryPort
from ....features.media.previews import render_still_webp
from ....features.media.storage import MediaStorageCoordinator
from ....shared.event_log import EventLogPort, record_event
from ..downloads import PageFileDownload
from ..presentation import jsonable

_IMAGE_EXTENSIONS = {
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


class MediaAdminController:
    def __init__(
        self,
        media_repository: MediaRepositoryPort,
        files_repository: FileRepositoryPort,
        event_repository: EventLogPort,
        file_artifacts: FileArtifactService,
        media_storage: MediaStorageCoordinator,
    ) -> None:
        self.media_repository = media_repository
        self.files_repository = files_repository
        self.event_repository = event_repository
        self.file_artifacts = file_artifacts
        self.media_storage = media_storage

    async def image_snapshot(
        self,
        profile_id: str,
        instance_id: str,
        *,
        page: int = 1,
        page_size: int = 100,
    ) -> dict[str, Any]:
        assert self.media_repository is not None
        size = max(1, min(int(page_size), 100))
        statistics = await self.media_repository.media_asset_statistics(
            profile_id, instance_id, mime_prefix="image/"
        )
        total = int(statistics["total"])
        page = _bounded_page(page, size, total)
        assets = await self.media_repository.list_media_assets(
            profile_id,
            instance_id,
            mime_prefix="image/",
            limit=size,
            offset=(page - 1) * size,
        )
        cleanup = await self.media_repository.list_media_cleanup_events(
            profile_id, instance_id, limit=100
        )
        counts = dict(statistics["file_status"])
        inspection = dict(statistics["inspection_status"])
        available_count = int(counts.get("AVAILABLE", 0))
        pending = sum(inspection.get(value, 0) for value in ("PENDING", "RUNNING"))
        unavailable = total - available_count
        serialized_assets = []
        for asset in assets:
            value = dict(jsonable(asset))
            value.pop("storage_relpath", None)
            value.pop("sha256", None)
            asset_available = str(asset.file_status.value) == "AVAILABLE"
            value.update(
                {
                    "action_ref": asset.asset_id,
                    "preview_available": asset_available,
                    "download_available": asset_available,
                    "download_filename": self._download_filename(asset),
                    "unavailable_reason": "" if asset_available else "原图已释放或当前不可用",
                }
            )
            serialized_assets.append(value)
        return {
            "profile_id": profile_id,
            "instance_id": instance_id,
            "overview": {
                "total": total,
                "file_status": counts,
                "inspection_status": inspection,
                "pending_cleanup": counts.get("RELEASE_PENDING", 0),
            },
            "counts": {
                "total": total,
                "available": available_count,
                "pending": pending,
                "unavailable": unavailable,
            },
            "assets": serialized_assets,
            "pagination": _pagination(page, size, total),
            "cleanup_events": jsonable(cleanup),
        }

    async def image_preview(
        self, profile_id: str, instance_id: str, asset_id: str
    ) -> dict[str, Any]:
        asset = await self._available_image(profile_id, instance_id, asset_id)
        assert asset.storage_relpath is not None
        path = self.media_storage.store.absolute_path(asset.storage_relpath)
        try:
            preview = await asyncio.to_thread(render_still_webp, path)
        except Exception as exc:
            raise ValueError("图片预览生成失败") from exc
        return {
            "asset_ref": asset.asset_id,
            "preview_data_url": "data:image/webp;base64,"
            + base64.b64encode(preview).decode("ascii"),
        }

    async def image_download(
        self, profile_id: str, instance_id: str, asset_id: str
    ) -> PageFileDownload:
        asset = await self._available_image(profile_id, instance_id, asset_id)
        assert asset.storage_relpath is not None
        return PageFileDownload(
            path=self.media_storage.store.absolute_path(asset.storage_relpath),
            filename=self._download_filename(asset),
            content_type=str(asset.mime_type),
            headers={
                "Cache-Control": "private, no-store, max-age=0",
                "Pragma": "no-cache",
                "X-Content-Type-Options": "nosniff",
            },
        )

    async def _available_image(self, profile_id: str, instance_id: str, asset_id: str) -> Any:
        normalized_id = str(asset_id or "").strip()
        if not normalized_id:
            raise ValueError("asset_id is required")
        asset = await self.media_repository.get_media_asset(
            normalized_id,
            profile_id=profile_id,
            instance_id=instance_id,
        )
        if asset is None or not str(asset.mime_type).lower().startswith("image/"):
            raise ValueError("图片不存在或不属于当前对话")
        if str(asset.file_status.value) != "AVAILABLE" or not asset.storage_relpath:
            raise ValueError("图片原文件已释放或当前不可用")
        try:
            verified = await asyncio.to_thread(self.media_storage.store.verify, asset)
        except Exception as exc:
            raise ValueError("图片文件校验失败") from exc
        if not verified:
            raise ValueError("图片文件缺失或校验失败")
        return asset

    @staticmethod
    def _download_filename(asset: Any) -> str:
        mime = str(asset.mime_type or "").strip().lower()
        extension = _IMAGE_EXTENSIONS.get(mime, ".img")
        created_at = asset.created_at
        stamp = created_at.astimezone(UTC).strftime("%Y%m%d-%H%M%S") if created_at else "image"
        token = re.sub(r"[^A-Za-z0-9_-]+", "", str(asset.asset_id or ""))[-12:] or "asset"
        return f"soulcore-image-{stamp}-{token}{extension}"

    async def file_artifact_snapshot(
        self,
        profile_id: str,
        instance_id: str,
        *,
        page: int = 1,
        page_size: int = 200,
    ) -> dict[str, Any]:
        assert self.files_repository is not None
        size = max(1, min(int(page_size), 500))
        summary = await self.files_repository.file_artifact_statistics(profile_id, instance_id)
        total = int(summary.get("total") or 0)
        page = _bounded_page(page, size, total)
        records = await self.files_repository.list_file_artifact_records(
            profile_id,
            instance_id,
            limit=size,
            offset=(page - 1) * size,
        )
        public_records = []
        for record in records:
            value = dict(record)
            value.pop("storage_relpath", None)
            public_records.append(jsonable(value))
        return {
            "profile_id": profile_id,
            "instance_id": instance_id,
            "summary": summary,
            "artifacts": public_records,
            "pagination": _pagination(page, size, total),
        }

    async def file_artifact_admin_action(
        self,
        profile_id: str,
        instance_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        assert self.files_repository is not None
        action = str(payload.get("action") or "").strip().lower()
        if action != "delete":
            raise ValueError("unsupported file artifact action")
        asset_id = str(payload.get("asset_id") or "").strip()
        if not asset_id:
            raise ValueError("asset_id is required")
        if str(payload.get("confirm_asset_id") or "") != asset_id:
            raise ValueError("confirm_asset_id must match the selected file")
        service = self.file_artifacts
        prepared = await self.files_repository.prepare_file_artifact_release(
            profile_id, instance_id, asset_id
        )
        if bool(prepared.get("already_released")):
            return {"ok": True, "deleted": False, "message": "文件已经删除"}
        try:
            existed = await asyncio.to_thread(
                service.release, str(prepared.get("storage_relpath") or "")
            )
        except Exception as exc:
            await self.files_repository.finalize_file_artifact_release(
                profile_id,
                instance_id,
                asset_id,
                released=False,
                error=f"{type(exc).__name__}:{exc}",
            )
            raise RuntimeError("实体文件删除失败，已保留待清理状态") from exc
        await self.files_repository.finalize_file_artifact_release(
            profile_id, instance_id, asset_id, released=True
        )
        await record_event(
            self.event_repository,
            profile_id=profile_id,
            instance_id=instance_id,
            level="INFO",
            category="file_artifact.admin",
            message="管理员已删除未投递文件成果",
            details={"asset_id": asset_id, "file_existed": bool(existed)},
        )
        return {
            "ok": True,
            "deleted": True,
            "message": "未投递文件、关联待办和待发送引用已删除",
        }


def _bounded_page(page: int, page_size: int, total: int) -> int:
    return max(1, min(int(page), max(1, (total + page_size - 1) // page_size)))


def _pagination(page: int, page_size: int, total: int) -> dict[str, int | bool]:
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "page_count": max(1, (total + page_size - 1) // page_size),
        "has_more": page * page_size < total,
    }
