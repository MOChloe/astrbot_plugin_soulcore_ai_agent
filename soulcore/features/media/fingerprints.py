"Bounded, deterministic image fingerprints and model-facing previews."

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .inspection import MAX_IMAGE_BYTES, inspect_image_bytes

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


MAX_REPRESENTATIVE_FRAMES = 4
MAX_MODEL_PREVIEW_EDGE = 1024
STRONG_FINGERPRINT_DISTANCE = 6

MediaSource = bytes | bytearray | memoryview | str | Path


@dataclass(frozen=True, slots=True)
class MediaFingerprintSet:
    """Multi-frame perceptual hashes for one static image or animation."""

    phash: str
    dhash: str
    frame_indexes: tuple[int, ...]
    source_frame_count: int


@dataclass(frozen=True, slots=True)
class MediaModelFrame:
    """One compressed frame safe to hand to a vision-model adapter."""

    data: bytes = field(repr=False)
    mime_type: str
    width: int
    height: int
    source_frame_index: int
    source_frame_indexes: tuple[int, ...] = ()
    layout_columns: int = 1


@dataclass(frozen=True, slots=True)
class MediaModelPreview:
    """A bounded static preview or chronological animation representation."""

    frames: tuple[MediaModelFrame, ...]
    source_mime_type: str
    source_width: int
    source_height: int
    source_frame_count: int
    animated: bool

    @property
    def payloads(self) -> tuple[tuple[bytes, str], ...]:
        return tuple((frame.data, frame.mime_type) for frame in self.frames)

    @property
    def representative_frame_indexes(self) -> tuple[int, ...]:
        values = (
            index
            for frame in self.frames
            for index in (frame.source_frame_indexes or (frame.source_frame_index,))
        )
        return tuple(dict.fromkeys(values))

    @property
    def uses_contact_sheet(self) -> bool:
        return len(self.frames) == 1 and len(self.representative_frame_indexes) > 1


def fingerprint_media(source: MediaSource) -> MediaFingerprintSet:
    """Return pHash/dHash values for at most four representative frames."""

    Image, _ImageOps = _pillow()
    data = _source_bytes(source)
    with Image.open(io.BytesIO(data)) as opened:
        frame_count = max(1, int(getattr(opened, "n_frames", 1) or 1))
        indexes = _representative_frame_indexes(frame_count, MAX_REPRESENTATIVE_FRAMES)
        phashes: list[str] = []
        dhashes: list[str] = []
        for index in indexes:
            opened.seek(index)
            frame = opened.convert("L")
            phashes.append(_phash(frame))
            dhashes.append(_dhash(frame))
    return MediaFingerprintSet(
        phash=".".join(phashes),
        dhash=".".join(dhashes),
        frame_indexes=indexes,
        source_frame_count=frame_count,
    )


def image_fingerprints(source: MediaSource) -> tuple[str, str]:
    """Return the canonical pHash/dHash tuple stored for sticker matching."""

    result = fingerprint_media(source)
    return result.phash, result.dhash


def fingerprint_distance(
    first: MediaFingerprintSet,
    second: MediaFingerprintSet,
) -> int | None:
    """Return a strong whole-image/frame-set distance, or ``None`` if incomparable."""

    first_animated = first.source_frame_count > 1
    second_animated = second.source_frame_count > 1
    if first_animated != second_animated:
        return None

    distances = (
        _frame_set_hash_distance(first.phash, second.phash),
        _frame_set_hash_distance(first.dhash, second.dhash),
    )
    comparable = [distance for distance in distances if distance is not None]
    return max(comparable) if comparable else None


def are_strongly_similar(
    first: MediaFingerprintSet,
    second: MediaFingerprintSet,
    *,
    maximum_distance: int = STRONG_FINGERPRINT_DISTANCE,
) -> bool:
    """Use the established strong-near threshold without semantic guesswork."""

    threshold = max(0, int(maximum_distance))
    distance = fingerprint_distance(first, second)
    return distance is not None and distance <= threshold


def bounded_model_preview(
    source: MediaSource,
    mime_type: str | None = None,
    *,
    maximum_edge: int | None = None,
    maximum_frames: int = MAX_MODEL_PREVIEW_FRAMES,
) -> MediaModelPreview:
    """Build a compressed preview while leaving the authoritative bytes untouched."""

    requested_edge = (
        None
        if maximum_edge is None
        else _bounded_positive(
            maximum_edge,
            MAX_ANIMATION_CONTACT_SHEET_EDGE,
            "maximum_edge",
        )
    )
    static_edge = min(requested_edge or MAX_MODEL_PREVIEW_EDGE, MAX_MODEL_PREVIEW_EDGE)
    animation_edge = requested_edge or MAX_ANIMATION_CONTACT_SHEET_EDGE
    frame_limit = _bounded_positive(maximum_frames, MAX_MODEL_PREVIEW_FRAMES, "maximum_frames")
    data = _source_bytes(source)
    detected_mime, _extension, width, height, frame_count = inspect_image_bytes(
        data,
        declared_mime=mime_type,
        maximum_bytes=MAX_IMAGE_BYTES,
    )
    Image, ImageOps = _pillow()
    with Image.open(io.BytesIO(data)) as opened:
        selected_limit = animation_frame_limit(frame_count, frame_limit)
        indexes = representative_frame_indexes(opened, selected_limit)
        frames: tuple[MediaModelFrame, ...]
        if frame_count > 1:
            sheet = build_animation_contact_sheet(
                opened,
                indexes,
                maximum_edge=animation_edge,
                maximum_pixels=min(
                    MAX_ANIMATION_CONTACT_SHEET_PIXELS,
                    animation_edge * animation_edge,
                ),
            )
            frames = (
                MediaModelFrame(
                    sheet.data,
                    sheet.mime_type,
                    sheet.width,
                    sheet.height,
                    sheet.source_frame_indexes[0],
                    source_frame_indexes=sheet.source_frame_indexes,
                    layout_columns=sheet.columns,
                ),
            )
        else:
            frames = tuple(
                _preview_frame(
                    opened,
                    index,
                    detected_mime=detected_mime,
                    maximum_edge=static_edge,
                    Image=Image,
                    ImageOps=ImageOps,
                )
                for index in indexes
            )
    return MediaModelPreview(
        frames=frames,
        source_mime_type=detected_mime,
        source_width=width,
        source_height=height,
        source_frame_count=frame_count,
        animated=frame_count > 1,
    )


def original_model_preview(
    source: MediaSource,
    mime_type: str | None = None,
    *,
    frame_indexes: tuple[int, ...] | list[int] | None = None,
    maximum_frames: int = MAX_REPRESENTATIVE_FRAMES,
) -> MediaModelPreview:
    """Return exact static bytes or native-size composited animation frames."""

    frame_limit = _bounded_positive(maximum_frames, MAX_REPRESENTATIVE_FRAMES, "maximum_frames")
    data = _source_bytes(source)
    detected_mime, _extension, width, height, frame_count = inspect_image_bytes(
        data,
        declared_mime=mime_type,
        maximum_bytes=MAX_IMAGE_BYTES,
    )
    if frame_count <= 1:
        if frame_indexes:
            raise ValueError("frame_indexes are only valid for animated image assets")
        return MediaModelPreview(
            frames=(MediaModelFrame(data, detected_mime, width, height, 0, (0,)),),
            source_mime_type=detected_mime,
            source_width=width,
            source_height=height,
            source_frame_count=1,
            animated=False,
        )

    Image, ImageOps = _pillow()
    with Image.open(io.BytesIO(data)) as opened:
        indexes = (
            _validated_frame_indexes(frame_indexes, frame_count, frame_limit)
            if frame_indexes
            else representative_frame_indexes(opened, frame_limit)
        )
        frames = tuple(
            _preview_frame(
                opened,
                index,
                detected_mime=detected_mime,
                maximum_edge=None,
                Image=Image,
                ImageOps=ImageOps,
            )
            for index in indexes
        )
    return MediaModelPreview(
        frames=frames,
        source_mime_type=detected_mime,
        source_width=width,
        source_height=height,
        source_frame_count=frame_count,
        animated=True,
    )


def _preview_frame(
    opened: Any,
    frame_index: int,
    *,
    detected_mime: str,
    maximum_edge: int | None,
    Image: Any,
    ImageOps: Any,
) -> MediaModelFrame:
    opened.seek(frame_index)
    if int(getattr(opened, "n_frames", 1) or 1) > 1:
        frame = opened.convert("RGBA")
    else:
        frame = ImageOps.exif_transpose(opened).copy()
    if maximum_edge is not None:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        frame.thumbnail((maximum_edge, maximum_edge), resampling)
    payload, output_mime = _encode_preview(frame, detected_mime)
    return MediaModelFrame(
        data=payload,
        mime_type=output_mime,
        width=int(frame.width),
        height=int(frame.height),
        source_frame_index=frame_index,
        source_frame_indexes=(frame_index,),
    )


def _encode_preview(frame: Any, source_mime: str) -> tuple[bytes, str]:
    output = io.BytesIO()
    if source_mime == "image/jpeg":
        frame.convert("RGB").save(
            output, format="JPEG", quality=85, optimize=True, progressive=True
        )
        return output.getvalue(), "image/jpeg"
    if source_mime == "image/webp":
        mode = _preview_mode(frame)
        frame.convert(mode).save(output, format="WEBP", quality=85, method=4)
        return output.getvalue(), "image/webp"
    mode = _preview_mode(frame)
    frame.convert(mode).save(output, format="PNG", optimize=True)
    return output.getvalue(), "image/png"


def _preview_mode(frame: Any) -> str:
    has_alpha = "A" in frame.getbands() or "transparency" in dict(frame.info or {})
    return "RGBA" if has_alpha else "RGB"


def _source_bytes(source: MediaSource) -> bytes:
    if isinstance(source, bytes):
        data = source
    elif isinstance(source, (bytearray, memoryview)):
        data = bytes(source)
    else:
        data = Path(source).read_bytes()
    if not data:
        raise ValueError("image payload is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"image exceeds {MAX_IMAGE_BYTES} byte limit")
    return data


def _representative_frame_indexes(frame_count: int, maximum_frames: int) -> tuple[int, ...]:
    count = max(1, int(frame_count))
    limit = min(count, max(1, int(maximum_frames)))
    if limit == 1:
        return (0,)
    if limit == 2:
        return tuple(sorted({0, count - 1}))
    if limit == 3:
        return tuple(sorted({0, count // 2, count - 1}))
    if limit == 4:
        return tuple(sorted({0, count // 3, (count * 2) // 3, count - 1}))
    return tuple(
        dict.fromkeys(round(offset * (count - 1) / (limit - 1)) for offset in range(limit))
    )


def representative_frame_indexes(opened: Any, maximum_frames: int) -> tuple[int, ...]:
    """Choose chronological GIF frames by playback time, including first and last."""

    frame_count = max(1, int(getattr(opened, "n_frames", 1) or 1))
    limit = min(frame_count, max(1, int(maximum_frames)))
    if frame_count == 1 or limit == 1:
        return (0,)
    durations: list[int] = []
    for index in range(frame_count):
        opened.seek(index)
        durations.append(max(1, int((opened.info or {}).get("duration") or 100)))
    total = sum(durations)
    targets = [total * offset / (limit - 1) for offset in range(limit)]
    indexes: list[int] = []
    elapsed = 0
    target_index = 0
    for index, duration in enumerate(durations):
        upper = elapsed + duration
        while target_index < len(targets) and targets[target_index] < upper:
            indexes.append(index)
            target_index += 1
        elapsed = upper
    indexes.extend((0, frame_count - 1))
    selected = set(indexes)
    for candidate in _representative_frame_indexes(frame_count, limit):
        if len(selected) >= limit:
            break
        selected.add(candidate)
    for candidate in range(frame_count):
        if len(selected) >= limit:
            break
        selected.add(candidate)
    opened.seek(0)
    return tuple(sorted(selected))


def _validated_frame_indexes(
    values: tuple[int, ...] | list[int], frame_count: int, maximum_frames: int
) -> tuple[int, ...]:
    if len(values) > maximum_frames:
        raise ValueError(f"at most {maximum_frames} frame indexes are allowed")
    indexes = tuple(dict.fromkeys(int(value) for value in values))
    if not indexes:
        raise ValueError("frame_indexes cannot be empty")
    if any(index < 0 or index >= frame_count for index in indexes):
        raise ValueError(f"frame index must be between 0 and {frame_count - 1}")
    return tuple(sorted(indexes))


def _frame_set_hash_distance(first: str, second: str) -> int | None:
    left_frames = _valid_hashes(first)
    right_frames = _valid_hashes(second)
    if not left_frames or not right_frames:
        return None

    def directed(source: tuple[str, ...], target: tuple[str, ...]) -> int | None:
        nearest = []
        for value in source:
            distances = [
                (int(value, 16) ^ int(candidate, 16)).bit_count()
                for candidate in target
                if len(value) == len(candidate)
            ]
            if not distances:
                return None
            nearest.append(min(distances))
        return max(nearest)

    left_distance = directed(left_frames, right_frames)
    right_distance = directed(right_frames, left_frames)
    if left_distance is None or right_distance is None:
        return None
    return max(left_distance, right_distance)


def _valid_hashes(value: str) -> tuple[str, ...]:
    valid: list[str] = []
    for part in str(value or "").split(".")[:MAX_REPRESENTATIVE_FRAMES]:
        if not part or len(part) != 16:
            continue
        try:
            int(part, 16)
        except ValueError:
            continue
        valid.append(part.lower())
    return tuple(valid)


def _bounded_positive(value: int, hard_maximum: int, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return min(parsed, int(hard_maximum))


def _dhash(image: Any) -> str:
    resized = image.resize((9, 8))
    flattened = getattr(resized, "get_flattened_data", None)
    pixels = list(flattened() if callable(flattened) else resized.getdata())
    bits = [pixels[y * 9 + x] > pixels[y * 9 + x + 1] for y in range(8) for x in range(8)]
    return f"{sum((1 << index) for index, bit in enumerate(bits) if bit):016x}"


def _phash(image: Any) -> str:
    resized = image.resize((32, 32))
    flattened = getattr(resized, "get_flattened_data", None)
    pixels = list(flattened() if callable(flattened) else resized.getdata())
    coefficients: list[float] = []
    for vertical in range(8):
        for horizontal in range(8):
            total = 0.0
            for y in range(32):
                cosine_y = math.cos((2 * y + 1) * vertical * math.pi / 64)
                for x in range(32):
                    total += (
                        pixels[y * 32 + x]
                        * math.cos((2 * x + 1) * horizontal * math.pi / 64)
                        * cosine_y
                    )
            coefficients.append(total)
    median = sorted(coefficients[1:])[len(coefficients[1:]) // 2]
    bits = [value > median for value in coefficients]
    return f"{sum((1 << index) for index, bit in enumerate(bits) if bit):016x}"


def _pillow() -> tuple[Any, Any]:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - Pillow is a plugin dependency
        raise RuntimeError("Pillow is required for media fingerprints") from exc
    return Image, ImageOps


__all__ = [
    "AnimationContactSheet",
    "MAX_ANIMATION_CONTACT_SHEET_EDGE",
    "MAX_ANIMATION_CONTACT_SHEET_PIXELS",
    "MAX_MODEL_PREVIEW_EDGE",
    "MAX_MODEL_PREVIEW_FRAMES",
    "MAX_REPRESENTATIVE_FRAMES",
    "MediaFingerprintSet",
    "MediaModelFrame",
    "MediaModelPreview",
    "STRONG_FINGERPRINT_DISTANCE",
    "animation_frame_limit",
    "are_strongly_similar",
    "bounded_model_preview",
    "build_animation_contact_sheet",
    "fingerprint_distance",
    "fingerprint_media",
    "image_fingerprints",
    "original_model_preview",
    "representative_frame_indexes",
]
