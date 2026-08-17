"""Bounded still-image previews for controlled media files."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_PREVIEW_MAX_EDGE = 1280
IMAGE_PREVIEW_MAX_BYTES = 512 * 1024


def render_still_webp(
    path: str | Path,
    *,
    max_edge: int = IMAGE_PREVIEW_MAX_EDGE,
    max_bytes: int = IMAGE_PREVIEW_MAX_BYTES,
) -> bytes:
    """Render one representative frame as a metadata-free bounded WebP."""

    edge = max(64, int(max_edge))
    byte_limit = max(16 * 1024, int(max_bytes))
    with Image.open(path) as source:
        frame_count = max(1, int(getattr(source, "n_frames", 1) or 1))
        source.seek(frame_count // 2 if frame_count > 1 else 0)
        frame = ImageOps.exif_transpose(source.copy()).convert("RGBA")

    frame.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    while True:
        for quality in (82, 72, 62, 52, 42, 32, 24, 16):
            output = io.BytesIO()
            frame.save(
                output,
                format="WEBP",
                quality=quality,
                method=4,
                exact=False,
            )
            data = output.getvalue()
            if len(data) <= byte_limit:
                return data
        width, height = frame.size
        if max(width, height) <= 96:
            raise ValueError("image preview cannot be encoded within the size limit")
        scale = max(0.5, min(0.85, (byte_limit / max(1, len(data))) ** 0.5 * 0.92))
        next_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        frame = frame.resize(next_size, Image.Resampling.LANCZOS)


__all__ = ["IMAGE_PREVIEW_MAX_BYTES", "IMAGE_PREVIEW_MAX_EDGE", "render_still_webp"]
