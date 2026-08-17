"""Consume live AstrBot media components before crossing into SoulCore features."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import mimetypes
import subprocess
import urllib.parse
import urllib.request
import wave
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ...features.media.inbound import InboundMediaSource
from ...features.media.locator_io import download_public_attachment, validate_remote_url
from ...shared.ffmpeg_runtime import managed_ffmpeg_executable

_REMOTE_SCHEMES = frozenset({"http", "https"})
_INLINE_PREFIXES = ("base64://", "data:")
_AUDIO_DOWNLOAD_TIMEOUT_SECONDS = 30.0
_AUDIO_CONVERSION_TIMEOUT_SECONDS = 60.0
_MAX_AUDIO_DURATION_SECONDS = 10 * 60


@dataclass(frozen=True, slots=True)
class ResolvedInboundAudio:
    """Bounded in-memory speech input with no durable source identifier."""

    data: bytes = field(repr=False, compare=False)
    mime_type: str
    filename: str
    duration_seconds: float | None = None


async def resolve_inbound_media(
    component: Any | None,
    locator: str,
    *,
    max_bytes: int,
    sticker_evidence: Sequence[str] = (),
) -> InboundMediaSource:
    """Resolve one live component without delegating remote I/O to AstrBot.

    AstrBot's conversion helpers are deliberately never invoked: they are
    opaque executable methods that may download, follow redirects, transcode,
    or materialize unbounded data before returning.  SoulCore consumes only
    explicit component fields and the normalized locator.

    Policy-rejected references produce an empty source.  Operational and
    programming failures propagate to the existing image or attachment ingest
    boundary so it can retain the real error in the event log while emitting a
    controlled, model-visible failure marker.
    """

    evidence = tuple(dict.fromkeys(str(value) for value in sticker_evidence if str(value)))
    source = await _resolve_component(component, str(locator or "").strip(), max_bytes)
    return replace(source, sticker_evidence=evidence)


async def resolve_inbound_audio(
    component: Any | None,
    locator: str,
    *,
    max_bytes: int,
) -> ResolvedInboundAudio:
    """Resolve speech to bounded bytes without using AstrBot conversion helpers."""

    source = await _resolve_component(component, str(locator or "").strip(), max_bytes)
    data = source.data
    mime_type = str(source.mime_type or "").strip().lower()
    if not data and source.locator:
        data, declared = await asyncio.to_thread(
            download_public_attachment,
            source.locator,
            max_bytes=max_bytes,
            timeout_seconds=_AUDIO_DOWNLOAD_TIMEOUT_SECONDS,
        )
        if not mime_type:
            mime_type = str(declared or "").strip().lower()
    if not data or len(data) > max_bytes:
        raise ValueError("audio size is invalid")
    mime_type = mime_type or "application/octet-stream"
    data, mime_type = await normalize_inbound_audio(
        data,
        mime_type,
        max_bytes=max_bytes,
    )
    return ResolvedInboundAudio(
        data=data,
        mime_type=mime_type,
        filename=_generic_audio_filename(mime_type),
        duration_seconds=_component_duration_seconds(component),
    )


async def normalize_inbound_audio(
    data: bytes,
    mime_type: str,
    *,
    max_bytes: int,
) -> tuple[bytes, str]:
    """Convert adapter-specific speech containers through bounded byte streams."""

    kind = _audio_magic(data)
    if kind == "wav":
        return data, "audio/wav"
    if kind == "silk":
        converted = await asyncio.to_thread(_decode_silk_to_wav, data, max_bytes)
        return converted, "audio/wav"
    if kind == "amr":
        converted = await _ffmpeg_amr_to_wav(data, max_bytes=max_bytes)
        return converted, "audio/wav"
    return data, str(mime_type or "application/octet-stream")


async def _resolve_component(
    component: Any | None,
    locator: str,
    max_bytes: int,
) -> InboundMediaSource:
    if max_bytes <= 0:
        raise ValueError("media size limit is invalid")

    references = await _component_references(component, locator)
    for reference in references:
        if reference.startswith(_INLINE_PREFIXES):
            try:
                data, mime_type = _decode_inline(reference, max_bytes)
            except ValueError:
                continue
            return InboundMediaSource(
                data=data,
                mime_type=_component_mime(component, "") or mime_type,
            )

    for reference in references:
        parsed = urllib.parse.urlsplit(reference)
        if parsed.scheme in _REMOTE_SCHEMES:
            # Do not call AstrBot's converter for a remote source: its downloader
            # owns neither SoulCore's SSRF policy nor its redirect/size bounds.
            try:
                await asyncio.to_thread(validate_remote_url, reference)
            except ValueError:
                continue
            else:
                return InboundMediaSource(
                    locator=reference,
                    mime_type=_component_mime(component, reference),
                )

    temp_root = _astrbot_temp_root()
    local_paths = _existing_local_paths(references, temp_root)
    for path in local_paths:
        try:
            data = await asyncio.to_thread(_read_bounded_file, path, max_bytes)
        except ValueError:
            continue
        return InboundMediaSource(
            data=data,
            mime_type=_component_mime(component, str(path)),
        )

    return InboundMediaSource()


async def _component_references(component: Any | None, locator: str) -> tuple[str, ...]:
    values: list[str] = []
    if component is not None:
        astrbot_file = _is_astrbot_file(component)
        # ``path``/``file_`` are the adapter's already-materialized variants.
        # Prefer them over a parallel signed URL so no network conversion runs.
        for name in ("path", "file_", "url") if astrbot_file else ("path", "file_", "url", "file"):
            value = str(getattr(component, name, "") or "").strip()
            if value:
                values.append(value)
        if astrbot_file:
            value = str(await component.get_file(allow_return_url=True) or "").strip()
            if value:
                values.append(value)
    if locator:
        values.append(locator)
    return tuple(dict.fromkeys(values))


def _is_astrbot_file(component: Any) -> bool:
    try:
        from astrbot.core.message.components import File
    except ImportError:
        return False
    return isinstance(component, File)


def _astrbot_temp_root() -> Path | None:
    try:
        from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

        root = Path(get_astrbot_temp_path())
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None
    try:
        resolved = root.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_absolute() else None


def _existing_local_paths(
    references: Sequence[str],
    temp_root: Path | None,
) -> tuple[Path, ...]:
    result: list[Path] = []
    for reference in references:
        resolved = _local_path(reference, temp_root)
        if resolved is not None and resolved.is_file():
            result.append(resolved)
    return tuple(dict.fromkeys(result))


def _local_path(reference: str, temp_root: Path | None) -> Path | None:
    if temp_root is None:
        return None
    value = str(reference or "").strip()
    if not value or value.startswith(_INLINE_PREFIXES):
        return None
    if value.startswith(("\\\\", "//")):
        return None
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme in _REMOTE_SCHEMES:
        return None
    if parsed.scheme == "file":
        value = _file_uri_path(parsed)
        if value is None:
            return None
    elif parsed.scheme and not _is_windows_absolute_reference(value):
        return None
    resolved = _resolved_absolute_path(value)
    if resolved is None or (resolved != temp_root and temp_root not in resolved.parents):
        return None
    return resolved


def _file_uri_path(parsed: urllib.parse.SplitResult) -> str | None:
    if parsed.netloc:
        return None
    value = urllib.request.url2pathname(parsed.path)
    if len(value) >= 3 and value[0] in {"/", "\\"} and value[2] == ":":
        return value[1:]
    return value


def _is_windows_absolute_reference(value: str) -> bool:
    return len(value) >= 3 and value[0].isalpha() and value[1] == ":" and value[2] in {"/", "\\"}


def _resolved_absolute_path(value: str) -> Path | None:
    try:
        path = Path(value)
        if not path.is_absolute():
            return None
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


def _read_bounded_file(path: Path, max_bytes: int) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if not data or len(data) > max_bytes:
        raise ValueError("media size is invalid")
    return data


def _decode_inline(value: str, max_bytes: int) -> tuple[bytes, str | None]:
    mime_type: str | None = None
    encoded = value
    max_encoded_length = ((max_bytes + 2) // 3) * 4
    if value.startswith("base64://"):
        if len(value) > 9 + max_encoded_length:
            raise ValueError("media size is invalid")
        encoded = value[9:]
    elif value.startswith("data:"):
        separator = value.find(",", 0, 257)
        if separator < 0:
            raise ValueError("unsupported inline media source")
        header = value[:separator]
        if ";base64" not in header.lower():
            raise ValueError("unsupported inline media source")
        if len(value) - separator - 1 > max_encoded_length:
            raise ValueError("media size is invalid")
        encoded = value[separator + 1 :]
        mime_type = header[5:].split(";", 1)[0].strip() or None
    # Reject before allocating decoded bytes.  Strict base64 contains no
    # whitespace, so this upper bound is exact apart from final padding.
    if not encoded or len(encoded) > max_encoded_length:
        raise ValueError("media size is invalid")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid inline media source") from exc
    if not data or len(data) > max_bytes:
        raise ValueError("media size is invalid")
    return data, mime_type


def _component_mime(component: Any | None, path: str) -> str | None:
    if component is not None:
        for name in ("mime_type", "mimetype", "content_type"):
            value = str(getattr(component, name, "") or "").split(";", 1)[0].strip()
            if value:
                return value
    guessed, _ = mimetypes.guess_type(path)
    return guessed


def _component_duration_seconds(component: Any | None) -> float | None:
    if component is None:
        return None
    for name in ("duration_seconds", "duration"):
        value = getattr(component, name, None)
        if value in (None, ""):
            continue
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if 0 < duration <= 24 * 60 * 60:
            return duration
    return None


def _audio_magic(data: bytes) -> str:
    header = bytes(data[:16])
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wav"
    if header.startswith((b"#!AMR\n", b"#!AMR-WB\n")):
        return "amr"
    if header.startswith((b"#!SILK_V3", b"\x02#!SILK_V3")):
        return "silk"
    return ""


class _BoundedBytesIO(io.BytesIO):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = int(limit)

    def write(self, value: bytes | bytearray) -> int:
        if self.tell() + len(value) > self.limit:
            raise ValueError("converted audio is too large")
        return super().write(value)


def _decode_silk_to_wav(data: bytes, max_bytes: int) -> bytes:
    import pysilk

    source = data[1:] if data.startswith(b"\x02#!SILK_V3") else data
    pcm = _BoundedBytesIO(max(1, int(max_bytes) - 1024))
    pysilk.decode(io.BytesIO(source), pcm, 24000)
    output = _BoundedBytesIO(max_bytes)
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24000)
        target.writeframes(pcm.getvalue())
    return output.getvalue()


async def _ffmpeg_amr_to_wav(data: bytes, *, max_bytes: int) -> bytes:
    try:
        executable = await asyncio.to_thread(managed_ffmpeg_executable)
        return await _ffmpeg_executable_amr_to_wav(
            executable,
            data,
            max_bytes=max_bytes,
            runtime_label="managed FFmpeg",
        )
    except asyncio.CancelledError:
        raise
    except Exception as managed_error:
        try:
            return await _ffmpeg_cli_amr_to_wav(data, max_bytes=max_bytes)
        except asyncio.CancelledError:
            raise
        except Exception as cli_error:
            raise ValueError("AMR audio conversion failed") from ExceptionGroup(
                "no managed AMR converter succeeded",
                [managed_error, cli_error],
            )


async def _ffmpeg_cli_amr_to_wav(data: bytes, *, max_bytes: int) -> bytes:
    return await _ffmpeg_executable_amr_to_wav(
        "ffmpeg",
        data,
        max_bytes=max_bytes,
        runtime_label="ffmpeg CLI",
    )


async def _ffmpeg_executable_amr_to_wav(
    executable: str,
    data: bytes,
    *,
    max_bytes: int,
    runtime_label: str,
) -> bytes:
    process = await asyncio.create_subprocess_exec(
        str(executable),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "amr",
        "-i",
        "pipe:0",
        "-t",
        str(_MAX_AUDIO_DURATION_SECONDS),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "pipe:1",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(_AUDIO_CONVERSION_TIMEOUT_SECONDS):
            stdout, _stderr = await process.communicate(data)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.communicate()
        raise
    if process.returncode != 0 or not stdout or len(stdout) > max_bytes:
        raise ValueError(f"{runtime_label} audio conversion failed")
    if _audio_magic(stdout) != "wav":
        raise ValueError("AMR converter did not return WAV")
    return stdout


def _generic_audio_filename(mime_type: str) -> str:
    normalized = str(mime_type or "").split(";", 1)[0].strip().lower()
    extension = {
        "audio/aac": ".aac",
        "audio/amr": ".amr",
        "audio/flac": ".flac",
        "audio/m4a": ".m4a",
        "audio/mp4": ".m4a",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
        "audio/opus": ".opus",
        "audio/wav": ".wav",
        "audio/webm": ".webm",
        "audio/x-wav": ".wav",
    }.get(normalized)
    if not extension:
        guessed = mimetypes.guess_extension(normalized) if normalized else None
        extension = guessed if guessed and guessed.isascii() else ".bin"
    return f"voice{extension}"


__all__ = [
    "ResolvedInboundAudio",
    "normalize_inbound_audio",
    "resolve_inbound_audio",
    "resolve_inbound_media",
]
