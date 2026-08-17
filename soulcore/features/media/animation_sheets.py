"""Time-dense animation contact sheets for model inspection."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

MAX_MODEL_PREVIEW_FRAMES = 32
MAX_ANIMATION_CONTACT_SHEET_EDGE = 1920
MAX_ANIMATION_CONTACT_SHEET_PIXELS = 1920 * 1080


@dataclass(frozen=True, slots=True)
class AnimationContactSheet:
    data: bytes
    mime_type: str
    width: int
    height: int
    source_frame_indexes: tuple[int, ...]
    columns: int


def animation_frame_limit(
    frame_count: int,
    maximum_frames: int = MAX_MODEL_PREVIEW_FRAMES,
) -> int:
    """Keep temporal coverage independent from the source image resolution."""

    return min(max(1, int(frame_count)), max(1, int(maximum_frames)))


def build_animation_contact_sheet(
    opened: Any,
    frame_indexes: tuple[int, ...],
    *,
    maximum_edge: int = MAX_ANIMATION_CONTACT_SHEET_EDGE,
    maximum_pixels: int = MAX_ANIMATION_CONTACT_SHEET_PIXELS,
) -> AnimationContactSheet:
    """Compose chronological frames into one roughly-1080p compressed image.

    Selected frames are resized before composition.  This keeps source
    resolution from reducing temporal coverage and avoids allocating a giant
    full-resolution canvas for large animations.
    """

    indexes = tuple(dict.fromkeys(int(value) for value in frame_indexes))
    if len(indexes) < 2:
        raise ValueError("an animation contact sheet requires at least two frames")
    edge = max(1, int(maximum_edge))
    pixel_budget = max(1, int(maximum_pixels))
    Image = _pillow_image()
    source_width = max(1, int(getattr(opened, "width", 1) or 1))
    source_height = max(1, int(getattr(opened, "height", 1) or 1))
    columns = _balanced_columns(len(indexes), source_width, source_height)
    rows = math.ceil(len(indexes) / columns)
    gap = _sheet_gap(edge, columns, rows)
    tile_width, tile_height = _bounded_tile_size(
        source_width,
        source_height,
        columns=columns,
        rows=rows,
        gap=gap,
        maximum_edge=edge,
        maximum_pixels=pixel_budget,
    )
    canvas = Image.new(
        "RGB",
        (
            columns * tile_width + (columns - 1) * gap,
            rows * tile_height + (rows - 1) * gap,
        ),
        (255, 255, 255),
    )
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    try:
        for ordinal, index in enumerate(indexes):
            opened.seek(index)
            frame = opened.convert("RGBA")
            frame.thumbnail((tile_width, tile_height), resampling)
            column = ordinal % columns
            row = ordinal // columns
            x = column * (tile_width + gap) + (tile_width - int(frame.width)) // 2
            y = row * (tile_height + gap) + (tile_height - int(frame.height)) // 2
            canvas.paste(frame, (x, y), frame)
    finally:
        opened.seek(0)
    data, mime_type = _encode_contact_sheet(canvas)
    return AnimationContactSheet(
        data=data,
        mime_type=mime_type,
        width=int(canvas.width),
        height=int(canvas.height),
        source_frame_indexes=indexes,
        columns=columns,
    )


def _sheet_gap(maximum_edge: int, columns: int, rows: int) -> int:
    if maximum_edge >= max(columns, rows) * 16:
        return 2
    if maximum_edge >= max(columns, rows) * 4:
        return 1
    return 0


def _bounded_tile_size(
    width: int,
    height: int,
    *,
    columns: int,
    rows: int,
    gap: int,
    maximum_edge: int,
    maximum_pixels: int,
) -> tuple[int, int]:
    available_width = maximum_edge - (columns - 1) * gap
    available_height = maximum_edge - (rows - 1) * gap
    if available_width < columns or available_height < rows:
        raise ValueError("maximum_edge is too small for the animation contact sheet")
    scale = min(
        1.0,
        available_width / (columns * width),
        available_height / (rows * height),
    )
    tile_width = max(1, int(width * scale))
    tile_height = max(1, int(height * scale))

    def canvas_pixels(tile_w: int, tile_h: int) -> int:
        canvas_width = columns * tile_w + (columns - 1) * gap
        canvas_height = rows * tile_h + (rows - 1) * gap
        return canvas_width * canvas_height

    while canvas_pixels(tile_width, tile_height) > maximum_pixels:
        ratio = math.sqrt(maximum_pixels / canvas_pixels(tile_width, tile_height))
        next_width = max(1, int(tile_width * ratio))
        next_height = max(1, int(tile_height * ratio))
        if next_width == tile_width and tile_width > 1:
            next_width -= 1
        if next_height == tile_height and tile_height > 1:
            next_height -= 1
        if (next_width, next_height) == (tile_width, tile_height):
            raise ValueError("maximum_pixels is too small for the animation contact sheet")
        tile_width, tile_height = next_width, next_height
    return tile_width, tile_height


def _encode_contact_sheet(canvas: Any) -> tuple[bytes, str]:
    output = io.BytesIO()
    try:
        canvas.save(output, format="WEBP", quality=80, method=6)
        return output.getvalue(), "image/webp"
    except OSError:
        output = io.BytesIO()
        canvas.save(output, format="JPEG", quality=80, optimize=True, progressive=True)
        return output.getvalue(), "image/jpeg"


def _balanced_columns(frame_count: int, width: int, height: int) -> int:
    best_columns = 1
    best_score = float("inf")
    for columns in range(1, frame_count + 1):
        rows = math.ceil(frame_count / columns)
        aspect = (columns * width) / max(1, rows * height)
        empty_ratio = (columns * rows - frame_count) / frame_count
        score = abs(math.log(max(aspect, 1e-9))) + empty_ratio * 0.1
        if score < best_score:
            best_columns = columns
            best_score = score
    return best_columns


def _pillow_image() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a plugin dependency
        raise RuntimeError("Pillow is required for animation contact sheets") from exc
    return Image


__all__ = [
    "MAX_ANIMATION_CONTACT_SHEET_EDGE",
    "MAX_ANIMATION_CONTACT_SHEET_PIXELS",
    "MAX_MODEL_PREVIEW_FRAMES",
    "AnimationContactSheet",
    "animation_frame_limit",
    "build_animation_contact_sheet",
]
