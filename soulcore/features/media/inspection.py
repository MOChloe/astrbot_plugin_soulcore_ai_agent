from __future__ import annotations

import io
import json
import re
import struct
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_DECODED_PIXELS = 64_000_000
MAX_INBOUND_ATTACHMENT_BYTES = 50 * 1024 * 1024
MAX_ANIMATION_FRAMES = 300
MAX_ANIMATION_DURATION_MS = 30_000
MAX_ANIMATION_DECODED_PIXELS = 240_000_000
_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_MIME_ALIASES = {"image/jpg": "image/jpeg", "image/x-png": "image/png"}
_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_INBOUND_MEDIA_KINDS = frozenset({"audio", "file", "video"})


def _now() -> datetime:
    return datetime.now(UTC)


def _dt(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value is not None else None


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _load(value: str | None) -> Any:
    return json.loads(value) if value else None


def generate_media_asset_id() -> str:
    return "ma_" + uuid.uuid4().hex


def infer_media_root(database_path: str | Path) -> Path:
    """Choose a stable root outside plugin_data whenever its layout is known."""

    path = Path(database_path).resolve(strict=False)
    parts = path.parts
    indexes = [index for index, part in enumerate(parts) if part.lower() == "plugin_data"]
    if indexes:
        index = indexes[-1]
        plugin_name = (
            parts[index + 1]
            if index + 1 < len(parts)
            else "astrbot_plugin_soulcore_ai_agent"
        )
        return Path(*parts[:index]) / "soulcore_media" / plugin_name
    return path.parent / "soulcore_media" / path.stem


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    offset = 2
    sof = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            break
        length = int.from_bytes(data[offset : offset + 2], "big")
        if length < 2 or offset + length > len(data):
            break
        if marker in sof and length >= 7:
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            if width > 0 and height > 0:
                return width, height
        offset += length
    raise ValueError("JPEG image has no valid dimensions")


def inspect_image_bytes(
    data: bytes,
    *,
    declared_mime: str | None = None,
    maximum_bytes: int | None = MAX_IMAGE_BYTES,
    enforce_animation_limits: bool = True,
    validate_decoded: bool = True,
) -> tuple[str, str, int, int, int]:
    """Validate supported image magic and return mime, extension, width, height, frames."""

    if not data:
        raise ValueError("image payload is empty")
    if maximum_bytes is not None and len(data) > int(maximum_bytes):
        raise ValueError(f"image exceeds {int(maximum_bytes)} byte limit")
    mime, width, height, frames = _inspect_image_header(
        data,
        enforce_animation_limits=enforce_animation_limits,
    )
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if int(width) * int(height) > MAX_IMAGE_DECODED_PIXELS:
        raise ValueError("image decoded pixel budget exceeded")
    if validate_decoded and mime != "image/gif" and frames <= 1:
        _validate_decoded_image(data, width, height)
    _validate_declared_image_mime(declared_mime, mime)
    return mime, _IMAGE_TYPES[mime], width, height, frames


def _inspect_image_header(
    data: bytes,
    *,
    enforce_animation_limits: bool,
) -> tuple[str, int, int, int]:
    inspected = _png_header(data, enforce_animation_limits)
    if inspected is not None:
        return inspected
    inspected = _jpeg_header(data)
    if inspected is not None:
        return inspected
    for inspect_animated_header in (_gif_header, _webp_header):
        inspected = inspect_animated_header(data, enforce_animation_limits)
        if inspected is not None:
            return inspected
    raise ValueError("unsupported image format")


def _png_header(
    data: bytes,
    enforce_animation_limits: bool,
) -> tuple[str, int, int, int] | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 33 and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
        if _png_declares_animation(data):
            animation = _inspect_animation(data, enforce_animation_limits)
            return "image/png", width, height, int(animation["frame_count"])
        return "image/png", width, height, 1
    return None


def _png_declares_animation(data: bytes) -> bool:
    offset = 8
    while offset + 12 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        end = offset + 12 + length
        if end > len(data):
            raise ValueError("malformed PNG chunk table")
        chunk_type = data[offset + 4 : offset + 8]
        if chunk_type == b"acTL":
            if length != 8 or int.from_bytes(data[offset + 8 : offset + 12], "big") < 1:
                raise ValueError("malformed APNG animation control")
            return True
        if chunk_type == b"IEND":
            return False
        offset = end
    raise ValueError("PNG image has no complete IEND chunk")


def _gif_header(
    data: bytes,
    enforce_animation_limits: bool,
) -> tuple[str, int, int, int] | None:
    if data.startswith((b"GIF87a", b"GIF89a")) and len(data) >= 14:
        width, height = struct.unpack("<HH", data[6:10])
        animation = _inspect_animation(data, enforce_animation_limits)
        return "image/gif", width, height, int(animation["frame_count"])
    return None


def _jpeg_header(data: bytes) -> tuple[str, int, int, int] | None:
    if data.startswith(b"\xff\xd8") and data.rfind(b"\xff\xd9", 2) >= 2:
        width, height = _jpeg_dimensions(data)
        return "image/jpeg", width, height, 1
    return None


def _webp_header(
    data: bytes,
    enforce_animation_limits: bool,
) -> tuple[str, int, int, int] | None:
    if len(data) >= 30 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        width, height = _webp_dimensions(data)
        animated = data[12:16] == b"VP8X" and bool(data[20] & 0x02)
        frames = (
            int(_inspect_animation(data, enforce_animation_limits)["frame_count"])
            if animated
            else 1
        )
        return "image/webp", width, height, frames
    return None


def _inspect_animation(data: bytes, enforce_limits: bool) -> dict[str, int | bool]:
    return inspect_animation_bytes(
        data,
        maximum_frames=MAX_ANIMATION_FRAMES if enforce_limits else None,
        maximum_duration_ms=MAX_ANIMATION_DURATION_MS if enforce_limits else None,
        maximum_decoded_pixels=MAX_ANIMATION_DECODED_PIXELS if enforce_limits else None,
    )


def _webp_dimensions(data: bytes) -> tuple[int, int]:
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        return (
            int.from_bytes(data[24:27], "little") + 1,
            int.from_bytes(data[27:30], "little") + 1,
        )
    if chunk == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
        b0, b1, b2, b3 = data[21:25]
        return (
            1 + b0 + ((b1 & 0x3F) << 8),
            1 + ((b1 >> 6) | (b2 << 2) | ((b3 & 0x0F) << 10)),
        )
    if chunk == b"VP8 " and len(data) >= 30 and data[23:26] == b"\x9d\x01\x2a":
        return (
            int.from_bytes(data[26:28], "little") & 0x3FFF,
            int.from_bytes(data[28:30], "little") & 0x3FFF,
        )
    raise ValueError("unsupported or malformed WebP image")


def _validate_decoded_image(data: bytes, width: int, height: int) -> None:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            decoded_size = tuple(int(value) for value in image.size)
            image.verify()
    except ImportError as exc:  # pragma: no cover - Pillow is a dependency
        raise RuntimeError("Pillow is required for image validation") from exc
    except Exception as exc:
        raise ValueError(f"image decode validation failed: {type(exc).__name__}") from exc
    if decoded_size != (int(width), int(height)):
        raise ValueError("image header dimensions do not match decoded image")


def _validate_declared_image_mime(declared_mime: str | None, mime: str) -> None:
    raw_declared = str(declared_mime or "").split(";", 1)[0].strip().lower()
    normalized_declared = _MIME_ALIASES.get(
        raw_declared,
        raw_declared,
    )
    if normalized_declared and normalized_declared != mime:
        raise ValueError(f"declared MIME {normalized_declared!r} does not match {mime!r}")


def inspect_animation_bytes(
    data: bytes,
    *,
    maximum_frames: int | None = MAX_ANIMATION_FRAMES,
    maximum_duration_ms: int | None = MAX_ANIMATION_DURATION_MS,
    maximum_decoded_pixels: int | None = MAX_ANIMATION_DECODED_PIXELS,
) -> dict[str, int | bool]:
    """Fully decode GIF/WebP animation metadata without altering the original bytes.

    Animation headers do not provide enough trustworthy validation on their own.
    Pillow walks every frame so truncated frame tables and
    decompression bombs are rejected before the file enters the asset layer.
    The original animated file is still stored and sent unchanged. Frame count
    is not an independent validity rule: byte size, decoded pixels and total
    duration bound normal ingestion, while model projection separately emits a
    small representative preview.
    """

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a plugin dependency
        raise RuntimeError("Pillow is required for animated image validation") from exc
    total_duration = 0
    frame_count = 0
    decoded_pixels = 0
    with Image.open(io.BytesIO(data)) as image:
        while True:
            # Force decompression of the current frame; merely seeking the GIF
            # table is insufficient to catch truncated or malicious payloads.
            image.load()
            width, height = image.size
            frame_count += 1
            if maximum_frames is not None and frame_count > int(maximum_frames):
                raise ValueError(f"animated image exceeds {int(maximum_frames)} frame limit")
            decoded_pixels += int(width) * int(height)
            if maximum_decoded_pixels is not None and decoded_pixels > int(maximum_decoded_pixels):
                raise ValueError("animated image decoded pixel budget exceeded")
            duration = int(image.info.get("duration") or 0)
            # A missing GIF delay is legal.  Give it a conservative display
            # duration for total-runtime validation instead of treating it as
            # an infinitely fast animation.
            total_duration += max(20, duration)
            if maximum_duration_ms is not None and total_duration > int(maximum_duration_ms):
                raise ValueError(
                    f"animated image exceeds {int(maximum_duration_ms)}ms duration limit"
                )
            try:
                image.seek(image.tell() + 1)
            except EOFError:
                break
    return {
        "animated": frame_count > 1,
        "frame_count": frame_count,
        "duration_ms": total_duration,
    }


def inspect_inbound_attachment_bytes(
    data: bytes,
    *,
    media_kind: str,
    declared_mime: str | None = None,
    maximum_bytes: int = MAX_INBOUND_ATTACHMENT_BYTES,
) -> tuple[str, str]:
    """Validate an inbound non-image attachment and choose a safe extension.

    The file is never executed and its platform filename is never reused as a
    storage path.  Audio/video require recognizable container magic; arbitrary
    files are retained as inert ``.bin`` unless a small passive format can be
    identified confidently.
    """

    kind = str(media_kind or "").strip().lower()
    if kind not in _INBOUND_MEDIA_KINDS:
        raise ValueError("unsupported inbound media kind")
    if not data:
        raise ValueError("media payload is empty")
    if len(data) > int(maximum_bytes):
        raise ValueError(f"media exceeds {int(maximum_bytes)} byte limit")
    declared = str(declared_mime or "").split(";", 1)[0].strip().lower()

    detected = _detect_inbound_container(data, kind, declared)
    detected = _validate_inbound_kind(data, kind, detected)
    _validate_declared_media_kind(declared, kind)
    return detected


def _detect_inbound_container(data: bytes, kind: str, declared: str) -> tuple[str, str] | None:
    for detector in (_detect_simple_audio, _detect_riff_container):
        detected = detector(data)
        if detected is not None:
            return detected
    return _detect_muxed_container(data, kind, declared)


def _detect_simple_audio(data: bytes) -> tuple[str, str] | None:
    if data.startswith(b"#!AMR\n") or data.startswith(b"#!AMR-WB\n"):
        return "audio/amr", ".amr"
    if data.startswith(b"#!SILK_V3"):
        return "audio/silk", ".silk"
    if data.startswith(b"fLaC"):
        return "audio/flac", ".flac"
    if data.startswith(b"ID3"):
        return "audio/mpeg", ".mp3"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "audio/mpeg", ".mp3"
    return None


def _detect_riff_container(data: bytes) -> tuple[str, str] | None:
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav", ".wav"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return "video/x-msvideo", ".avi"
    return None


def _detect_muxed_container(data: bytes, kind: str, declared: str) -> tuple[str, str] | None:
    if data.startswith(b"OggS"):
        return (
            "video/ogg" if kind == "video" else "audio/ogg",
            ".ogv" if kind == "video" else ".ogg",
        )
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return (
            "audio/webm" if kind == "audio" else "video/webm",
            ".webm",
        )
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return (
            "audio/mp4" if kind == "audio" and declared.startswith("audio/") else "video/mp4",
            ".m4a" if kind == "audio" and declared.startswith("audio/") else ".mp4",
        )
    return None


def _validate_inbound_kind(
    data: bytes, kind: str, detected: tuple[str, str] | None
) -> tuple[str, str]:
    if kind == "audio":
        if detected is None or not detected[0].startswith("audio/"):
            raise ValueError("unsupported or malformed audio payload")
        return detected
    if kind == "video":
        if detected is None or not detected[0].startswith("video/"):
            raise ValueError("unsupported or malformed video payload")
        return detected
    if data.startswith(b"%PDF-"):
        return "application/pdf", ".pdf"
    if data.startswith(b"PK\x03\x04"):
        return "application/zip", ".zip"
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return "application/octet-stream", ".bin"
    return "text/plain", ".txt"


def _validate_declared_media_kind(declared: str, kind: str) -> None:
    expected_prefix = {"audio": "audio/", "video": "video/"}.get(kind)
    if declared and expected_prefix and not declared.startswith(expected_prefix):
        raise ValueError("declared MIME does not match media kind")
