"""Reusable media projection for group-window cleaning and model input."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .domain import MediaAsset
from .fingerprints import (
    MediaFingerprintSet,
    are_strongly_similar,
    fingerprint_media,
)
from .ports import GroupPreviewLocatorPort, MediaRepositoryPort


@dataclass(frozen=True, slots=True)
class GroupMediaProjection:
    asset_ids: tuple[str, ...]
    preview_urls: tuple[str, ...]


class GroupMediaProjectionService:
    def __init__(
        self, repository: MediaRepositoryPort, visual_service: GroupPreviewLocatorPort
    ) -> None:
        self.repository = repository
        self.visual_service = visual_service

    async def cluster_keys(
        self,
        profile_id: str,
        instance_id: str,
        asset_ids: list[str] | tuple[str, ...],
    ) -> tuple[str, ...]:
        keys = []
        for asset_id in dict.fromkeys(str(item) for item in asset_ids if str(item)):
            resolved = await self._resolved(profile_id, instance_id, asset_id)
            if resolved is None:
                continue
            asset, path, fingerprint = resolved
            value = f"sha256={asset.sha256};kind={asset.mime_type}"
            if fingerprint is not None:
                value += (
                    f";phash={fingerprint.phash};dhash={fingerprint.dhash}"
                    f";frames={fingerprint.source_frame_count}"
                )
            keys.append(value)
        return tuple(keys)

    async def project_window(
        self,
        profile_id: str,
        instance_id: str,
        asset_ids: list[str] | tuple[str, ...],
        *,
        limit: int = 5,
    ) -> GroupMediaProjection:
        representatives: list[tuple[str, MediaAsset, MediaFingerprintSet | None, str]] = []
        for asset_id in dict.fromkeys(str(item) for item in asset_ids if str(item)):
            resolved = await self._resolved(profile_id, instance_id, asset_id)
            if resolved is None:
                continue
            asset, path, fingerprint = resolved
            if self._is_duplicate(representatives, asset.sha256, fingerprint):
                continue
            representatives.append((asset_id, asset, fingerprint, path))
            if len(representatives) >= max(1, min(5, int(limit))):
                break
        return GroupMediaProjection(
            asset_ids=tuple(item[0] for item in representatives),
            preview_urls=(),
        )

    async def _resolved(
        self, profile_id: str, instance_id: str, asset_id: str
    ) -> tuple[MediaAsset, str, MediaFingerprintSet | None] | None:
        asset = await self.repository.get_media_asset(
            asset_id, profile_id=profile_id, instance_id=instance_id
        )
        if asset is None:
            return None
        path = await self.visual_service.asset_file_path(
            profile_id=profile_id, instance_id=instance_id, asset_id=asset_id
        )
        if not path:
            return None
        fingerprint = None
        if str(asset.mime_type).lower().startswith("image/"):
            try:
                fingerprint = await asyncio.to_thread(fingerprint_media, path)
            except Exception:
                fingerprint = None
        return asset, str(path), fingerprint

    @staticmethod
    def _is_duplicate(
        representatives: list[tuple[str, MediaAsset, MediaFingerprintSet | None, str]],
        sha256: str,
        fingerprint: MediaFingerprintSet | None,
    ) -> bool:
        return any(
            sha256 == known_asset.sha256
            or (
                fingerprint is not None
                and known_fingerprint is not None
                and are_strongly_similar(fingerprint, known_fingerprint)
            )
            for _asset_id, known_asset, known_fingerprint, _path in representatives
        )


__all__ = ["GroupMediaProjection", "GroupMediaProjectionService"]
