"""Deterministic sticker selection and normalization helpers."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ...shared.token_meter import ConservativeTokenMeter
from .domain import StickerItem, StickerUsageType
from .retrieval import RankedSticker


def normalize_semantic_key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)[:120]


def usage_type(payload: Mapping[str, Any], *, semantic: str) -> StickerUsageType:
    raw = str(payload.get("usage_type") or "").strip().upper()
    try:
        return StickerUsageType(raw)
    except ValueError:
        if compact_text(payload.get("visible_text"), 500) or semantic:
            return StickerUsageType.SPECIFIC
        if compact_text(payload.get("emotion"), 48) or compact_text(payload.get("speech_act"), 48):
            return StickerUsageType.REACTION
        return StickerUsageType.AMBIENT


def filter_preference(rows: Sequence[StickerItem], preference: str) -> list[StickerItem]:
    normalized = compact_text(preference, 32).casefold()
    if normalized in {"", "任意", "随便", "any"}:
        return list(rows)
    if normalized in {"随手发", "无意义", "氛围", "ambient"}:
        return [row for row in rows if row.usage_type is StickerUsageType.AMBIENT]
    if normalized in {"带字", "有字", "文字", "text"}:
        return [row for row in rows if bool(row.ocr_text or row.visible_text)]
    if normalized in {"动图", "gif", "animated", "动态"}:
        return [row for row in rows if row.is_animated]
    raise ValueError("表情包偏好只能是任意、随手发、带字或动图")


def _sticker_lanes(
    values: Sequence[RankedSticker],
) -> tuple[list[RankedSticker], ...]:
    return (
        [item for item in values if item.relevance > 0 or item.affect_match > 0],
        [item for item in values if item.row.usage_type is StickerUsageType.AMBIENT],
        [item for item in values if item.row.is_animated],
        [item for item in values if item.row.ocr_text or item.row.visible_text],
        [item for item in values if item.reinforcement > 0],
        list(values),
    )


def _next_lane_sticker(
    lane: Sequence[RankedSticker],
    position: int,
    *,
    selected_ids: set[str],
    visual_groups: set[str],
) -> tuple[RankedSticker | None, int]:
    while position < len(lane):
        item = lane[position]
        position += 1
        if item.item_id in selected_ids:
            continue
        visual = str(item.visual_group or "")
        if visual and visual in visual_groups:
            continue
        return item, position
    return None, position


def interleaved_stickers(ranked: Sequence[RankedSticker], *, limit: int) -> list[RankedSticker]:
    """Mix contextual, ambient, persona and format lanes before token fitting."""

    maximum = max(0, int(limit))
    if maximum <= 0:
        return []
    lanes = _sticker_lanes(ranked)
    positions = [0] * len(lanes)
    selected: list[RankedSticker] = []
    selected_ids: set[str] = set()
    visual_groups: set[str] = set()
    while len(selected) < maximum:
        progressed = False
        for lane_index, lane in enumerate(lanes):
            item, positions[lane_index] = _next_lane_sticker(
                lane,
                positions[lane_index],
                selected_ids=selected_ids,
                visual_groups=visual_groups,
            )
            if item is None:
                continue
            selected.append(item)
            selected_ids.add(item.item_id)
            visual = str(item.visual_group or "")
            if visual:
                visual_groups.add(visual)
            progressed = True
            if len(selected) >= maximum:
                break
        if not progressed:
            break
    return selected


def fit_ranked_to_token_budget(
    ranked: Sequence[RankedSticker],
    *,
    token_limit: int,
    meter: ConservativeTokenMeter,
    describe: Callable[[StickerItem], str],
) -> list[RankedSticker]:
    selected: list[RankedSticker] = []
    used = 0
    for item in ranked:
        compact = str(describe(item.row) or "").strip()
        cost = meter.count_text(compact) + meter.MESSAGE_OVERHEAD
        if used + cost > token_limit:
            continue
        selected.append(item)
        used += cost
    return selected


def compact_text(value: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def sticker_intensity(row: StickerItem) -> int:
    return max(0, min(10, int(row.intensity)))


__all__ = [
    "compact_text",
    "filter_preference",
    "fit_ranked_to_token_budget",
    "interleaved_stickers",
    "normalize_semantic_key",
    "sticker_intensity",
    "usage_type",
]
