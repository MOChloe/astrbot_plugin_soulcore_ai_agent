"""Hard-bounded image previews for model-only continuation inputs.

Authoritative media bytes stay in their owning feature.  This module only
creates disposable previews that are safe to embed in a provider request.
"""

from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass, field
from typing import Any

MAX_MODEL_IMAGE_PREVIEW_EDGE = 1024
MAX_MODEL_IMAGE_PREVIEW_BYTES = 256 * 1024
MAX_MODEL_IMAGE_PREVIEW_TOTAL_BYTES = 256 * 1024
STRICT_MODEL_IMAGE_PREVIEW_TOTAL_BYTES = 96 * 1024
_MAX_MODEL_IMAGE_PIXELS = 40_000_000
_SUPPORTED_MIME_TYPES = frozenset({"image/gif", "image/jpeg", "image/png", "image/webp"})
_ENCODE_QUALITIES = (84, 76, 68, 58, 48, 38, 28)


@dataclass(frozen=True, slots=True)
class ModelImagePreview:
    data: bytes = field(repr=False)
    mime_type: str
    width: int = 0
    height: int = 0


def bounded_model_image_preview(
    data: bytes | bytearray | memoryview,
    mime_type: str,
    *,
    maximum_bytes: int = MAX_MODEL_IMAGE_PREVIEW_BYTES,
    maximum_edge: int = MAX_MODEL_IMAGE_PREVIEW_EDGE,
) -> ModelImagePreview:
    """Return a preview under both byte and edge limits.

    Small payloads are kept byte-for-byte.  This preserves already-bounded
    provider-normalized images without decoding and re-encoding them again.
    """

    raw = bytes(data)
    normalized_mime = str(mime_type or "").strip().lower()
    byte_limit = _positive_limit(maximum_bytes, "maximum_bytes")
    edge_limit = _positive_limit(maximum_edge, "maximum_edge")
    if normalized_mime not in _SUPPORTED_MIME_TYPES:
        raise ValueError("unsupported model image preview type")
    if not raw:
        raise ValueError("model image preview bytes are missing")
    if len(raw) <= byte_limit:
        return ModelImagePreview(raw, normalized_mime)

    Image, ImageOps = _pillow()
    with Image.open(io.BytesIO(raw)) as opened:
        width, height = (int(opened.width), int(opened.height))
        if width <= 0 or height <= 0 or width * height > _MAX_MODEL_IMAGE_PIXELS:
            raise ValueError("model image preview dimensions are unsafe")
        opened.seek(0)
        frame = ImageOps.exif_transpose(opened).convert("RGBA" if _has_alpha(opened) else "RGB")
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        frame.thumbnail((edge_limit, edge_limit), resampling)
        while True:
            encoded, output_mime = _encode_frame_under_limit(frame, byte_limit)
            if encoded is not None:
                return ModelImagePreview(
                    encoded,
                    output_mime,
                    int(frame.width),
                    int(frame.height),
                )
            longest = max(int(frame.width), int(frame.height))
            if longest <= 128:
                break
            ratio = max(128 / longest, 0.78)
            next_size = (
                max(1, int(frame.width * ratio)),
                max(1, int(frame.height * ratio)),
            )
            frame = frame.resize(next_size, resampling)
    raise ValueError(f"model image preview cannot fit within {byte_limit} bytes")


def rebudget_model_image_data_uris(
    values: tuple[str, ...],
    *,
    target_values: tuple[str, ...],
    maximum_total_bytes: int = STRICT_MODEL_IMAGE_PREVIEW_TOTAL_BYTES,
) -> tuple[str, ...]:
    """Shrink selected data-URI previews while preserving other model inputs."""

    targets = set(target_values)
    selected = [value for value in values if value in targets and value.startswith("data:image/")]
    if not selected:
        return values
    remaining = _positive_limit(maximum_total_bytes, "maximum_total_bytes")
    slots = len(selected)
    result: list[str] = []
    for value in values:
        if value not in targets or not value.startswith("data:image/"):
            result.append(value)
            continue
        mime_type, raw = decode_model_image_data_uri(value)
        allocation = max(1, remaining // max(1, slots))
        preview = bounded_model_image_preview(
            raw,
            mime_type,
            maximum_bytes=allocation,
        )
        result.append(encode_model_image_data_uri(preview.data, preview.mime_type))
        remaining -= len(preview.data)
        slots -= 1
    return tuple(result)


def encode_model_image_data_uri(data: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def decode_model_image_data_uri(value: str) -> tuple[str, bytes]:
    header, separator, payload = str(value).partition(",")
    if not separator or not header.startswith("data:") or not header.endswith(";base64"):
        raise ValueError("invalid model image data URI")
    mime_type = header[5:-7].strip().lower()
    if mime_type not in _SUPPORTED_MIME_TYPES:
        raise ValueError("unsupported model image data URI type")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid model image data URI payload") from exc
    if not raw:
        raise ValueError("model image data URI is empty")
    return mime_type, raw


def _encode_frame_under_limit(frame: Any, byte_limit: int) -> tuple[bytes | None, str]:
    has_alpha = "A" in frame.getbands()
    output_mime = "image/webp" if has_alpha else "image/jpeg"
    output_format = "WEBP" if has_alpha else "JPEG"
    prepared = frame.convert("RGBA" if has_alpha else "RGB")
    for quality in _ENCODE_QUALITIES:
        output = io.BytesIO()
        options = (
            {"quality": quality, "method": 4}
            if has_alpha
            else {
                "quality": quality,
                "optimize": True,
                "progressive": True,
            }
        )
        prepared.save(output, format=output_format, **options)
        encoded = output.getvalue()
        if len(encoded) <= byte_limit:
            return encoded, output_mime
    return None, output_mime


def _has_alpha(image: Any) -> bool:
    return "A" in image.getbands() or "transparency" in dict(image.info or {})


def _positive_limit(value: int, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _pillow() -> tuple[Any, Any]:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - Pillow is a plugin dependency
        raise RuntimeError("Pillow is required for model image previews") from exc
    return Image, ImageOps


__all__ = [
    "MAX_MODEL_IMAGE_PREVIEW_BYTES",
    "MAX_MODEL_IMAGE_PREVIEW_EDGE",
    "MAX_MODEL_IMAGE_PREVIEW_TOTAL_BYTES",
    "STRICT_MODEL_IMAGE_PREVIEW_TOTAL_BYTES",
    "ModelImagePreview",
    "bounded_model_image_preview",
    "decode_model_image_data_uri",
    "encode_model_image_data_uri",
    "rebudget_model_image_data_uris",
]
