"""Prepare sticker-sized temporary images without mutating durable media assets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

STICKER_RESIZE_THRESHOLD_PX = 512
STICKER_TARGET_MIN_PX = 240
STICKER_TARGET_MAX_PX = 360
MAX_STICKER_RESIZE_FRAME_BYTES = 64 * 1024 * 1024

_FORMAT_SUFFIXES = {
    "GIF": ".gif",
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}


class StickerDeliveryPreparationError(RuntimeError):
    """An oversized sticker could not be prepared safely for delivery."""


def sticker_target_longest_edge(idempotency_key: str, asset_id: str) -> int:
    """Choose a retry-stable, message-specific target edge within the sticker range."""

    digest = hashlib.sha256(f"{idempotency_key}\0{asset_id}".encode()).digest()
    choices = STICKER_TARGET_MAX_PX - STICKER_TARGET_MIN_PX + 1
    return STICKER_TARGET_MIN_PX + int.from_bytes(digest[:8], "big") % choices


def prepare_sticker_delivery_image(
    source_path: str | Path,
    output_directory: str | Path,
    *,
    idempotency_key: str,
    asset_id: str,
) -> Path:
    """Return the source or a temporary, proportionally resized sticker image."""

    from PIL import Image, ImageOps, ImageSequence

    source = Path(source_path)
    output_root = Path(output_directory)
    try:
        if not source.is_file():
            raise ValueError("source image is unavailable")
        with Image.open(source) as opened:
            source_format = str(opened.format or "").upper()
            suffix = _FORMAT_SUFFIXES.get(source_format)
            if suffix is None:
                raise ValueError("unsupported sticker image format")
            if max(opened.size) <= STICKER_RESIZE_THRESHOLD_PX:
                return source

            target_edge = sticker_target_longest_edge(idempotency_key, asset_id)
            output_root.mkdir(parents=True, exist_ok=True)
            output_name = hashlib.sha256(
                f"{idempotency_key}\0{asset_id}\0{target_edge}".encode()
            ).hexdigest()[:24]
            output = output_root / f"sticker-{output_name}{suffix}"
            if int(getattr(opened, "n_frames", 1) or 1) > 1:
                frame_durations = (
                    _webp_frame_durations(source.read_bytes()) if source_format == "WEBP" else None
                )
                _save_animated(
                    opened,
                    output,
                    source_format,
                    _target_size(opened.size, target_edge),
                    frame_durations=frame_durations,
                    Image=Image,
                    ImageSequence=ImageSequence,
                )
            else:
                _save_static(
                    opened,
                    output,
                    source_format,
                    target_edge,
                    Image=Image,
                    ImageOps=ImageOps,
                )
        if not output.is_file():
            raise OSError("resized sticker was not written")
        return output
    except StickerDeliveryPreparationError:
        raise
    except Exception as exc:
        raise StickerDeliveryPreparationError(
            f"sticker_delivery_resize_failed:{type(exc).__name__}"
        ) from exc


def _target_size(source_size: tuple[int, int], target_edge: int) -> tuple[int, int]:
    width, height = source_size
    scale = target_edge / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _save_static(
    opened: Any,
    output: Path,
    source_format: str,
    target_edge: int,
    *,
    Image: Any,
    ImageOps: Any,
) -> None:
    frame = ImageOps.exif_transpose(opened).copy()
    frame = frame.resize(_target_size(frame.size, target_edge), Image.Resampling.LANCZOS)
    if source_format == "JPEG":
        frame.convert("RGB").save(
            output,
            format="JPEG",
            quality=85,
            optimize=True,
            progressive=True,
        )
        return
    if source_format == "WEBP":
        frame.convert(_color_mode(frame)).save(
            output,
            format="WEBP",
            quality=85,
            method=4,
        )
        return
    if source_format == "PNG":
        frame.save(output, format="PNG", optimize=True)
        return
    frame.save(output, format="GIF", optimize=True)


def _save_animated(
    opened: Any,
    output: Path,
    source_format: str,
    target_size: tuple[int, int],
    *,
    frame_durations: list[int] | None,
    Image: Any,
    ImageSequence: Any,
) -> None:
    frame_count = max(1, int(getattr(opened, "n_frames", 1) or 1))
    decoded_bytes = frame_count * int(target_size[0]) * int(target_size[1]) * 4
    if decoded_bytes > MAX_STICKER_RESIZE_FRAME_BYTES:
        raise ValueError("animated sticker resize memory budget exceeded")
    frames, durations, disposals, blends = _collect_animation_frames(
        opened,
        target_size,
        frame_durations=frame_durations,
        Image=Image,
        ImageSequence=ImageSequence,
    )
    common = {
        "save_all": True,
        "append_images": frames[1:],
        "duration": durations,
        "loop": int(opened.info.get("loop") or 0),
    }
    if source_format == "GIF":
        frames[0].save(output, format="GIF", disposal=disposals, optimize=False, **common)
        return
    if source_format == "WEBP":
        frames[0].save(output, format="WEBP", quality=85, method=4, **common)
        return
    frames[0].save(
        output,
        format="PNG",
        disposal=disposals,
        blend=blends,
        optimize=True,
        **common,
    )


def _collect_animation_frames(
    opened: Any,
    target_size: tuple[int, int],
    *,
    frame_durations: list[int] | None,
    Image: Any,
    ImageSequence: Any,
) -> tuple[list[Any], list[int], list[int], list[int]]:
    frames: list[Any] = []
    durations: list[int] = []
    disposals: list[int] = []
    blends: list[int] = []
    default_duration = int(opened.info.get("duration") or 0)
    for index, source_frame in enumerate(ImageSequence.Iterator(opened)):
        frame = source_frame.convert("RGBA").resize(target_size, Image.Resampling.LANCZOS)
        frames.append(frame)
        durations.append(
            int(
                frame_durations[index]
                if frame_durations is not None and index < len(frame_durations)
                else source_frame.info.get("duration") or default_duration
            )
        )
        disposals.append(
            int(
                getattr(source_frame, "disposal_method", None)
                or source_frame.info.get("disposal")
                or 2
            )
        )
        blends.append(int(source_frame.info.get("blend") or 0))
    if not frames:
        raise ValueError("animated sticker has no frames")
    if frame_durations is not None and len(frame_durations) != len(frames):
        raise ValueError("animated WebP frame timing is incomplete")
    return frames, durations, disposals, blends


def _color_mode(frame: Any) -> str:
    return "RGBA" if "A" in frame.getbands() or "transparency" in frame.info else "RGB"


def _webp_frame_durations(data: bytes) -> list[int]:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("animated WebP container is invalid")
    durations: list[int] = []
    position = 12
    while position + 8 <= len(data):
        chunk_type = data[position : position + 4]
        chunk_size = int.from_bytes(data[position + 4 : position + 8], "little")
        payload_start = position + 8
        payload_end = payload_start + chunk_size
        if payload_end > len(data):
            raise ValueError("animated WebP chunk is truncated")
        if chunk_type == b"ANMF":
            if chunk_size < 16:
                raise ValueError("animated WebP frame header is truncated")
            durations.append(
                int.from_bytes(data[payload_start + 12 : payload_start + 15], "little")
            )
        position = payload_end + (chunk_size & 1)
    if not durations:
        raise ValueError("animated WebP has no frame timing")
    return durations


__all__ = [
    "STICKER_RESIZE_THRESHOLD_PX",
    "STICKER_TARGET_MAX_PX",
    "STICKER_TARGET_MIN_PX",
    "StickerDeliveryPreparationError",
    "prepare_sticker_delivery_image",
    "sticker_target_longest_edge",
]
