"""Bounded current-media projection for MainCore model input and inspection."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .fingerprints import MediaModelPreview, bounded_model_preview, original_model_preview
from .sticker_likelihood import asset_sticker_likelihood, classify_possible_sticker

_STICKER_HANDLING_GUIDANCE = (
    "表情包有时只是对方手痒、无聊、觉得可爱好玩或闲得没事而随手发来，可能没有明确意思；"
    "看不出对方此刻的心境时直接无视，看得出也只接住那种感觉，不围绕画面元素展开。"
)
_IMAGE_HANDLING_GUIDANCE = (
    "图片按普通聊天内容来接：简单回应、接梗、略过或自然转开都可以，不要逐项复述；"
    "内容被问起或某个细节值得聊时，再回应相关部分。"
)
_POSSIBLE_STICKER_NOTE = "可能是表情包"


@dataclass(frozen=True, slots=True)
class MainCoreMediaItem:
    asset_ref: str
    mime_type: str
    source_width: int
    source_height: int
    source_frame_count: int
    duration_ms: int
    preview_frames: tuple[tuple[int, int, int], ...]
    possible_sticker: bool = False
    sticker_evidence: tuple[str, ...] = ()
    preview_source_indexes: tuple[int, ...] = ()
    preview_contact_sheet: bool = False
    preview_layout_columns: int = 1


@dataclass(frozen=True, slots=True)
class MainCoreMediaProjection:
    image_urls: tuple[str, ...]
    items: tuple[MainCoreMediaItem, ...]

    def context_note(self) -> str:
        lines = [
            "当前图片（压缩预览）",
            _STICKER_HANDLING_GUIDANCE,
            _IMAGE_HANDLING_GUIDANCE,
        ]
        for item in self.items:
            semantic = f"｜{_POSSIBLE_STICKER_NOTE}" if item.possible_sticker else ""
            if item.source_frame_count <= 1:
                preview = _static_preview_summary(item)
                lines.append(f"- {item.asset_ref}｜静态图片{preview}{semantic}")
                continue
            timing = f"｜帧数={item.source_frame_count}"
            if item.duration_ms > 0:
                timing += f"｜总时长={item.duration_ms}ms"
            lines.append(
                f"- {item.asset_ref}｜动图{timing}；当前预览："
                f"{_animated_preview_summary(item)}{semantic}"
            )
        return "\n".join(lines)


def _static_preview_summary(item: MainCoreMediaItem) -> str:
    if not item.preview_frames:
        return ""
    _source_index, width, height = item.preview_frames[0]
    return f"｜预览={width}x{height}"


def _animated_preview_summary(item: MainCoreMediaItem) -> str:
    if item.preview_contact_sheet and item.preview_frames:
        _source_index, width, height = item.preview_frames[0]
        indexes = ",".join(str(value) for value in item.preview_source_indexes)
        return (
            f"分镜拼图 {width}x{height}，每行{item.preview_layout_columns}格，"
            f"按从左到右、从上到下排列，覆盖源帧[{indexes}]"
        )
    frames = ", ".join(
        f"源帧{index}:{width}x{height}" for index, width, height in item.preview_frames
    )
    return frames or "已提供代表帧"


async def project_main_core_media(
    media: Any,
    file_store: Any,
    *,
    profile_id: str,
    instance_id: str,
    asset_ids: Sequence[str],
    limit: int = 5,
) -> MainCoreMediaProjection:
    urls: list[str] = []
    items: list[MainCoreMediaItem] = []
    selected = list(dict.fromkeys(str(value) for value in asset_ids if str(value)))[
        : max(1, min(5, int(limit)))
    ]
    for asset_id in selected:
        asset, path = await _available_asset(media, file_store, profile_id, instance_id, asset_id)
        preview = await asyncio.to_thread(bounded_model_preview, path, asset.mime_type)
        metadata = dict(getattr(asset, "metadata", None) or {})
        prior_evidence = metadata.get("sticker_evidence") or ()
        if isinstance(prior_evidence, str):
            prior_evidence = (prior_evidence,)
        sticker = classify_possible_sticker(
            mime_type=preview.source_mime_type,
            width=preview.source_width,
            height=preview.source_height,
            frame_count=preview.source_frame_count,
            evidence=prior_evidence,
            previously_possible=metadata.get("possible_sticker") is True,
        )
        urls.extend(
            f"data:{frame.mime_type};base64,{base64.b64encode(frame.data).decode('ascii')}"
            for frame in preview.frames
        )
        items.append(
            MainCoreMediaItem(
                asset_ref=asset_id,
                mime_type=preview.source_mime_type,
                source_width=preview.source_width,
                source_height=preview.source_height,
                source_frame_count=preview.source_frame_count,
                duration_ms=int((asset.metadata or {}).get("duration_ms") or 0),
                preview_frames=tuple(
                    (frame.source_frame_index, frame.width, frame.height)
                    for frame in preview.frames
                ),
                possible_sticker=sticker.possible,
                sticker_evidence=sticker.evidence,
                preview_source_indexes=preview.representative_frame_indexes,
                preview_contact_sheet=preview.uses_contact_sheet,
                preview_layout_columns=(preview.frames[0].layout_columns if preview.frames else 1),
            )
        )
    return MainCoreMediaProjection(tuple(urls), tuple(items))


async def main_core_media_semantic_note(
    media: Any,
    *,
    profile_id: str,
    instance_id: str,
    asset_ids: Sequence[str],
    limit: int = 5,
) -> str:
    """Describe likely sticker semantics when MainCore receives no image bytes."""

    selected = list(dict.fromkeys(str(value) for value in asset_ids if str(value)))[
        : max(1, min(5, int(limit)))
    ]
    if not selected:
        return ""
    lines: list[str] = []
    for ordinal, asset_id in enumerate(selected, start=1):
        asset = await media.get_media_asset(
            asset_id,
            profile_id=profile_id,
            instance_id=instance_id,
        )
        if asset is not None and asset_sticker_likelihood(asset).possible:
            lines.append(f"- 第{ordinal}张：{_POSSIBLE_STICKER_NOTE}")
    return "\n".join((_STICKER_HANDLING_GUIDANCE, _IMAGE_HANDLING_GUIDANCE, *lines))


async def inspect_current_media(
    media: Any,
    file_store: Any,
    *,
    profile_id: str,
    instance_id: str,
    asset_id: str,
    frame_indexes: Sequence[int] | None = None,
) -> MediaModelPreview:
    asset, path = await _available_asset(media, file_store, profile_id, instance_id, asset_id)
    return await asyncio.to_thread(
        original_model_preview,
        path,
        asset.mime_type,
        frame_indexes=list(frame_indexes or []) or None,
    )


async def _available_asset(
    media: Any, file_store: Any, profile_id: str, instance_id: str, asset_id: str
) -> tuple[Any, Any]:
    asset = await media.get_media_asset(asset_id, profile_id=profile_id, instance_id=instance_id)
    if asset is None or not asset.storage_relpath or asset.file_status.value != "AVAILABLE":
        raise ValueError("current media asset is unavailable")
    path = file_store.absolute_path(asset.storage_relpath)
    if not await asyncio.to_thread(file_store.verify, asset):
        raise ValueError("current media asset is unavailable")
    return asset, path


__all__ = [
    "MainCoreMediaItem",
    "MainCoreMediaProjection",
    "inspect_current_media",
    "main_core_media_semantic_note",
    "project_main_core_media",
]
