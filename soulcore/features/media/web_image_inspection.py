"""Safe download, storage, and inspection of selected web image results."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ...contracts.web import ImageSearchResult
from ...shared.event_log import record_event
from .domain import (
    MediaInspectionStatus,
    MediaOrigin,
    MediaProjectionStatus,
    MediaPurpose,
)
from .visual_cache import VisualCachePolicy

if TYPE_CHECKING:
    from .image_service import VisualExpressionService


@dataclass(slots=True)
class _Result:
    asset_ids: list[str] = field(default_factory=list)
    inspected: list[dict[str, Any]] = field(default_factory=list)
    parts: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class _Candidate:
    resource: ImageSearchResult
    resource_id: str
    primary: str
    thumbnail: str
    source_url: str
    title: str
    description: str
    download_errors: list[str] = field(default_factory=list)


async def inspect_web_search_images(
    service: VisualExpressionService,
    *,
    resources: Sequence[ImageSearchResult],
    profile_id: str,
    instance_id: str,
    core_run_id: str,
    main_core_supports_vision: bool = False,
    defer_inspection_to_sticker_check: bool = False,
) -> Mapping[str, Any]:
    await service.runtime_gate.require_enabled(profile_id, instance_id)
    try:
        run_id = int(core_run_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid Core Run for image inspection") from exc
    result = _Result()
    for resource in list(resources)[:3]:
        await _process_candidate(
            service,
            _candidate(resource),
            result,
            profile_id=profile_id,
            instance_id=instance_id,
            core_run_id=core_run_id,
            run_id=run_id,
            main_core_supports_vision=main_core_supports_vision,
            defer_sticker_check=defer_inspection_to_sticker_check,
        )
    if not result.asset_ids:
        _raise_empty(result.failures)
    return {
        "asset_ids": result.asset_ids,
        "inspected": result.inspected,
        "failures": result.failures,
        "content_parts": result.parts,
        "content": (
            "Selected web images were downloaded and inspected. Only these asset IDs may be sent."
        ),
    }


def _candidate(resource: ImageSearchResult) -> _Candidate:
    return _Candidate(
        resource=resource,
        resource_id=str(resource.image_resource_id or ""),
        primary=str(resource.image_url or ""),
        thumbnail=str(resource.thumbnail_url or ""),
        source_url=str(resource.source_url or ""),
        title=str(resource.title or ""),
        description=str(resource.description or ""),
    )


async def _process_candidate(
    service: VisualExpressionService,
    candidate: _Candidate,
    result: _Result,
    *,
    profile_id: str,
    instance_id: str,
    core_run_id: str,
    run_id: int,
    main_core_supports_vision: bool,
    defer_sticker_check: bool,
) -> None:
    downloaded = await _download(service, candidate)
    if downloaded is None:
        failure = _download_failure(candidate)
        result.failures.append(failure)
        await _log_download_failure(
            service,
            profile_id,
            instance_id,
            candidate,
            failure,
        )
        return
    data, declared, mode = downloaded
    asset_id = (
        "ma_web_"
        + hashlib.sha256(
            f"{profile_id}:{instance_id}:{core_run_id}:{candidate.resource_id}".encode()
        ).hexdigest()[:24]
    )
    stored = await _store(
        service,
        candidate,
        result,
        asset_id,
        data,
        declared,
        mode,
        profile_id,
        instance_id,
        run_id,
    )
    if stored is None:
        return
    if defer_sticker_check:
        result.asset_ids.append(asset_id)
        result.inspected.append(_inspected_candidate(candidate, asset_id))
        return
    await _inspect(
        service,
        candidate,
        result,
        asset_id,
        stored,
        data,
        profile_id,
        instance_id,
        main_core_supports_vision,
    )


async def _download(
    service: VisualExpressionService,
    candidate: _Candidate,
) -> tuple[bytes, str | None, str] | None:
    errors: list[str] = []
    for locator, mode in (
        (candidate.primary, "original"),
        (candidate.thumbnail, "thumbnail"),
    ):
        if not locator:
            continue
        try:
            data, declared = await service.read_locator(
                locator,
                referer=candidate.source_url or None,
            )
            return data, declared, mode
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            errors.append(
                "UNSUPPORTED_IMAGE" if "did not return an image" in str(exc) else type(exc).__name__
            )
    candidate.download_errors = errors
    return None


def _download_failure(candidate: _Candidate) -> dict[str, str]:
    errors = candidate.download_errors
    code = "UNSUPPORTED_IMAGE" if "UNSUPPORTED_IMAGE" in errors else "DOWNLOAD_FAILED"
    return {
        "image_resource_id": candidate.resource_id,
        "error": code,
        "stage": "response_type" if code == "UNSUPPORTED_IMAGE" else "download",
    }


async def _log_download_failure(
    service: VisualExpressionService,
    profile_id: str,
    instance_id: str,
    candidate: _Candidate,
    failure: Mapping[str, str],
) -> None:
    await record_event(
        service.event_log,
        profile_id=profile_id,
        instance_id=instance_id,
        level="ERROR",
        category="web.image.inspect",
        message="联网搜索图片下载失败",
        details={
            "image_resource_id": candidate.resource_id,
            "stage": failure["stage"],
            "attempt_errors": candidate.download_errors,
        },
    )


async def _store(
    service: VisualExpressionService,
    candidate: _Candidate,
    result: _Result,
    asset_id: str,
    data: bytes,
    declared: str | None,
    mode: str,
    profile_id: str,
    instance_id: str,
    run_id: int,
) -> Any | None:
    try:
        stored = await asyncio.to_thread(
            service.file_store.store_bytes,
            asset_id=asset_id,
            profile_id=profile_id,
            instance_id=instance_id,
            data=data,
            declared_mime=declared,
        )
        await service.media.create_media_asset(
            profile_id,
            instance_id,
            stored,
            origin=MediaOrigin.GENERATED,
            purpose=MediaPurpose.GENERATED_IMAGE,
            inspection_status=MediaInspectionStatus.PENDING,
            core_run_id=run_id,
            expires_at=datetime.now(UTC) + timedelta(hours=24),
            metadata={
                "image_resource_id": candidate.resource_id,
                "source_kind": "WEB",
                "source_page_url": candidate.source_url or candidate.primary,
                "search_title": candidate.title,
                "provider": str(candidate.resource.provider or ""),
                "download_mode": mode,
            },
        )
        return stored
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        result.failures.append(
            {
                "image_resource_id": candidate.resource_id,
                "error": "UNSUPPORTED_IMAGE",
                "stage": "decode",
            }
        )
        await record_event(
            service.event_log,
            profile_id=profile_id,
            instance_id=instance_id,
            level="ERROR",
            category="web.image.inspect",
            message="联网搜索图片格式检查失败",
            details={
                "image_resource_id": candidate.resource_id,
                "stage": "decode",
                "error": type(exc).__name__,
            },
        )
        return None


async def _inspect(
    service: VisualExpressionService,
    candidate: _Candidate,
    result: _Result,
    asset_id: str,
    stored: Any,
    data: bytes,
    profile_id: str,
    instance_id: str,
    main_core_vision: bool,
) -> None:
    try:
        if main_core_vision:
            await _project_for_main_core(
                service,
                candidate,
                result,
                asset_id,
                stored,
                data,
                profile_id,
                instance_id,
            )
        else:
            await _describe(service, candidate, result, asset_id, profile_id, instance_id)
        result.asset_ids.append(asset_id)
        result.inspected.append(_inspected_candidate(candidate, asset_id))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await _inspection_failure(
            service,
            candidate,
            result,
            asset_id,
            profile_id,
            instance_id,
            exc,
        )


def _inspected_candidate(candidate: _Candidate, asset_id: str) -> dict[str, Any]:
    """Keep the exact search-resource to controlled-asset binding."""

    return {
        "image_resource_id": candidate.resource_id,
        "media_asset_id": str(asset_id),
        "title": candidate.title,
        "description": candidate.description,
        "source_url": candidate.source_url,
    }


async def _project_for_main_core(
    service: VisualExpressionService,
    candidate: _Candidate,
    result: _Result,
    asset_id: str,
    stored: Any,
    data: bytes,
    profile_id: str,
    instance_id: str,
) -> None:
    await service.media.save_media_projection(
        asset_id,
        status=MediaProjectionStatus.READY,
        visible_facts="",
        history_projection="",
    )
    service.describe_in_background(
        profile_id=profile_id,
        instance_id=instance_id,
        asset_ids=[asset_id],
        cache_policy=VisualCachePolicy.USE,
    )
    result.parts.extend(
        [
            {
                "type": "text",
                "text": f"Search asset {asset_id} is attached; inspect its pixels before selection.",
            },
            {"type": "image", "mime_type": stored.mime_type, "data": data, "asset_id": asset_id},
        ]
    )


async def _describe(
    service: VisualExpressionService,
    candidate: _Candidate,
    result: _Result,
    asset_id: str,
    profile_id: str,
    instance_id: str,
) -> None:
    vision = await service.describe_asset(
        profile_id=profile_id,
        instance_id=instance_id,
        asset_id=asset_id,
        foreground=True,
        cache_policy=VisualCachePolicy.USE,
    )
    result.parts.append(
        {
            "type": "text",
            "text": f"Inspected search asset {asset_id}: {vision.visible_facts}",
        }
    )


async def _inspection_failure(
    service: VisualExpressionService,
    candidate: _Candidate,
    result: _Result,
    asset_id: str,
    profile_id: str,
    instance_id: str,
    exc: Exception,
) -> None:
    result.failures.append(
        {
            "image_resource_id": candidate.resource_id,
            "error": "VISION_UNAVAILABLE",
            "stage": "visual_inspection",
        }
    )
    await record_event(
        service.event_log,
        profile_id=profile_id,
        instance_id=instance_id,
        level="ERROR",
        category="web.image.inspect",
        message="联网搜索图片视觉检查失败",
        details={
            "image_resource_id": candidate.resource_id,
            "stage": "visual_inspection",
            "error": type(exc).__name__,
        },
    )


def _raise_empty(failures: Sequence[Mapping[str, str]]) -> None:
    from .errors import WebImageInspectionError

    codes = {str(item.get("error") or "") for item in failures}
    if "VISION_UNAVAILABLE" in codes:
        raise WebImageInspectionError(
            "VISION_UNAVAILABLE",
            "Selected images were downloaded safely, but no visual inspection backend completed successfully",
        )
    if "UNSUPPORTED_IMAGE" in codes:
        raise WebImageInspectionError(
            "UNSUPPORTED_IMAGE",
            "Selected image responses were not supported, valid image files",
        )
    raise WebImageInspectionError(
        "DOWNLOAD_FAILED",
        "Selected images could not be downloaded from their remote hosts",
    )


__all__ = ["inspect_web_search_images"]
