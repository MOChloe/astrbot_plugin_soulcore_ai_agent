"""Run-scoped high-detail inspection command for current player images."""

from __future__ import annotations

import re
from typing import Any

from .command_context import _active, _record_command_outcome


def _inspection_request_error(collector: Any, reference: str) -> tuple[str, str] | None:
    allowed = set(collector.current_image_asset_ids)
    if reference not in allowed:
        return "NOT_ALLOWED", "error: 图片短引用不属于本轮当前图片"
    if collector.visual_service is None or not collector.main_core_supports_vision:
        return "DISABLED", "error: 当前无法查看图片高清细节"
    return None


def _frame_indexes_are_invalid(raw_indexes: list[Any]) -> bool:
    return len(raw_indexes) > 4 or any(
        not isinstance(value, int) or isinstance(value, bool) for value in raw_indexes
    )


async def _current_media_timing(collector: Any, reference: str) -> dict[str, Any]:
    timing: dict[str, Any] = {}
    try:
        timing = dict(
            await collector.visual_service.current_media_timing(
                profile_id=collector.profile_id,
                instance_id=collector.instance_id,
                asset_id=reference,
            )
        )
    except Exception:
        timing = {}
    return timing


def _requested_frame_indexes(
    raw_indexes: list[int],
    position: str,
    timing: dict[str, Any],
    frame_count: int,
    duration_ms: int,
) -> tuple[list[int], str]:
    if raw_indexes and position:
        return [], "error: 动图位置与旧帧序号不能同时填写"
    if not position:
        return list(raw_indexes), ""
    if not timing:
        return [], "error: 当前图片缺少可用于选择动图位置的帧信息"
    try:
        return _natural_frame_indexes(position, frame_count, duration_ms), ""
    except ValueError as exc:
        return [], f"error: {exc}"


def _inspection_result(
    preview: Any,
    reference: str,
    focus: str,
    position: str,
    frame_count: int,
    indexes: list[int],
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    requested_focus = str(focus or "").strip()
    if frame_count <= 1 and position:
        position_notice = "这是静态图片，已忽略“动图位置”并查看原图。"
    elif position:
        human_frames = "、".join(str(index + 1) for index in indexes)
        position_notice = f"“{position}”已按源图元数据映射到第 {human_frames} 帧。"
    else:
        position_notice = ""
    return {
        "content": {
            "ok": True,
            "asset_ref": reference,
            "source_mime_type": preview.source_mime_type,
            "source_width": preview.source_width,
            "source_height": preview.source_height,
            "source_frame_count": preview.source_frame_count,
            "frames": frames,
            "notice": "；".join(
                item
                for item in (
                    (
                        f"请重点查看：{requested_focus}。"
                        if requested_focus
                        else "请整体查看这张图。"
                    ),
                    position_notice,
                    "这些是高精度源图；对外表达只说画面内容，不复述图前的短引用。",
                )
                if item
            ),
        },
        "content_parts": [
            {
                "type": "image",
                "mime_type": frame.mime_type,
                "data": frame.data,
            }
            for frame in preview.frames
        ],
        "media_asset_ids": [],
    }


async def inspect_current_image(
    _event: Any,
    asset_ref: str,
    frame_indexes: list[int] | None = None,
    focus: str = "",
    animation_position: str = "",
) -> Any:
    collector = _active()
    public_reference = str(asset_ref or "").strip()
    reference = str(collector.model_reference_map.get(public_reference) or "").strip()
    if not reference:
        return "error: [[图片]] 使用了当前不可用的短引用"
    request_error = _inspection_request_error(collector, reference)
    if request_error is not None:
        error_code, message = request_error
        _record_command_outcome(collector, "inspect_current_image", ok=False, error=error_code)
        return message

    raw_indexes = list(frame_indexes or [])
    if _frame_indexes_are_invalid(raw_indexes):
        _record_command_outcome(
            collector, "inspect_current_image", ok=False, error="INVALID_FRAMES"
        )
        return "error: 动图帧序号最多填写四个整数"

    timing = await _current_media_timing(collector, reference)
    frame_count = max(1, int(timing.get("frame_count") or 1))
    duration_ms = max(0, int(timing.get("duration_ms") or 0))
    position = str(animation_position or "").strip()
    indexes, frame_error = _requested_frame_indexes(
        raw_indexes,
        position,
        timing,
        frame_count,
        duration_ms,
    )
    if frame_error:
        _record_command_outcome(
            collector, "inspect_current_image", ok=False, error="INVALID_FRAMES"
        )
        return frame_error

    try:
        preview = await collector.visual_service.inspect_current_media(
            profile_id=collector.profile_id,
            instance_id=collector.instance_id,
            asset_id=reference,
            frame_indexes=indexes,
        )
    except Exception as exc:
        _record_command_outcome(
            collector,
            "inspect_current_image",
            ok=False,
            error=type(exc).__name__,
        )
        return "error: 图片高清细节暂时无法读取"

    collector.inspected_current_image_asset_ids.add(reference)
    frames = [
        {
            "source_frame_index": frame.source_frame_index,
            "width": frame.width,
            "height": frame.height,
            "mime_type": frame.mime_type,
        }
        for frame in preview.frames
    ]
    _record_command_outcome(collector, "inspect_current_image", ok=True)
    return _inspection_result(
        preview,
        reference,
        focus,
        position,
        frame_count,
        indexes,
        frames,
    )


def _natural_frame_indexes(position: str, frame_count: int, duration_ms: int) -> list[int]:
    count = max(1, int(frame_count))
    if count <= 1:
        return []
    text = str(position or "").strip().casefold()
    if not text:
        return []
    values: list[int] = []
    for part in re.split(r"[,，、;；/]|(?:\s+和\s+)", text):
        item = part.strip()
        if not item:
            continue
        value = _natural_frame_index(item, count, duration_ms)
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("动图位置不能为空")
    if len(values) > 4:
        raise ValueError("一次最多查看四个动图位置")
    return values


def _natural_frame_index(item: str, count: int, duration_ms: int) -> int:
    ordinal = re.search(r"第\s*(\d+)\s*帧", item)
    seconds = re.search(r"(\d+(?:\.\d+)?)\s*秒", item)
    if ordinal:
        value = int(ordinal.group(1)) - 1
    elif seconds:
        if duration_ms <= 0:
            raise ValueError("这张动图没有可用总时长，不能按秒选择位置")
        ratio = float(seconds.group(1)) * 1000.0 / float(duration_ms)
        value = round(max(0.0, min(1.0, ratio)) * (count - 1))
    elif any(token in item for token in ("开头", "开始", "第一帧", "最初")):
        value = 0
    elif any(token in item for token in ("最后", "结尾", "末尾", "结束")):
        value = count - 1
    elif any(token in item for token in ("中间", "一半", "正中")):
        value = round((count - 1) / 2)
    else:
        raise ValueError("动图位置请写“开头”“中间”“最后”“第 3 帧”或“2 秒附近”")
    if value < 0 or value >= count:
        raise ValueError(f"这张动图只有 {count} 帧，所选帧超出范围")
    return value


__all__ = ["inspect_current_image"]
