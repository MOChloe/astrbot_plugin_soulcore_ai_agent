"""Bounded audio validation and conversion for outbound OneBot records."""

from __future__ import annotations

import asyncio
import contextlib
import io
import subprocess
import tempfile
import uuid
import wave
from collections.abc import Mapping
from pathlib import Path

from ...shared.ffmpeg_runtime import managed_ffmpeg_executable

MAX_VOICE_DURATION_SECONDS = 10 * 60
_CONVERSION_TIMEOUT_SECONDS = 60.0
_SOURCE_EXTENSIONS = {
    "audio/aac": ".aac",
    "audio/amr": ".amr",
    "audio/flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-wav": ".wav",
}
_RAW_PCM_MIME_TYPES = frozenset({"audio/pcm", "audio/pcm16"})
_RAW_PCM_SUFFIXES = frozenset({".pcm", ".pcm16"})


async def normalize_outbound_voice_audio(
    data: bytes,
    mime_type: str,
    *,
    filename: str,
    maximum_bytes: int,
    work_root: str | Path,
    audio_metadata: Mapping[str, object] | None = None,
) -> bytes:
    """Return one verified, uncompressed PCM WAV or fail closed."""

    payload = bytes(data)
    limit = max(1, int(maximum_bytes))
    if not payload or len(payload) > limit:
        raise ValueError("voice audio byte size is outside the allowed range")
    if _wav_magic(payload):
        validate_pcm_wav(payload, maximum_bytes=limit)
        return payload
    normalized_mime = str(mime_type or "").split(";", 1)[0].strip().lower()
    suffix = Path(str(filename or "")).suffix.lower()
    if _is_raw_pcm(normalized_mime, suffix):
        converted = _pcm_s16le_to_wav(
            payload,
            audio_metadata or {},
            maximum_bytes=limit,
        )
        validate_pcm_wav(converted, maximum_bytes=limit)
        return converted
    if normalized_mime not in _SOURCE_EXTENSIONS and suffix not in set(_SOURCE_EXTENSIONS.values()):
        raise ValueError("speech provider returned an unsupported audio container")
    root = Path(work_root).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    try:
        converted = await asyncio.to_thread(
            _managed_ffmpeg_to_pcm_wav,
            payload,
            normalized_mime,
            suffix,
            limit,
            root,
        )
        validate_pcm_wav(converted, maximum_bytes=limit)
    except asyncio.CancelledError:
        raise
    except Exception as managed_error:
        try:
            converted = await asyncio.to_thread(
                _ffmpeg_cli_to_pcm_wav,
                payload,
                normalized_mime,
                suffix,
                limit,
                root,
            )
        except asyncio.CancelledError:
            raise
        except Exception as cli_error:
            raise ValueError("outbound voice audio conversion failed") from ExceptionGroup(
                "no managed outbound audio converter succeeded",
                [managed_error, cli_error],
            )
    validate_pcm_wav(converted, maximum_bytes=limit)
    return converted


def _is_raw_pcm(mime_type: str, suffix: str) -> bool:
    if mime_type in _RAW_PCM_MIME_TYPES:
        return True
    return mime_type in {"", "application/octet-stream"} and suffix in _RAW_PCM_SUFFIXES


def _pcm_s16le_to_wav(
    data: bytes,
    metadata: Mapping[str, object],
    *,
    maximum_bytes: int,
) -> bytes:
    encoding = str(metadata.get("encoding") or "").strip().lower()
    if encoding != "pcm_s16le":
        raise ValueError("raw PCM encoding metadata is unavailable or unsupported")
    sample_rate = _required_pcm_integer(
        metadata,
        "sample_rate_hz",
        minimum=8_000,
        maximum=96_000,
    )
    channels = _required_pcm_integer(metadata, "channels", minimum=1, maximum=2)
    sample_width = _required_pcm_integer(
        metadata,
        "sample_width_bytes",
        minimum=2,
        maximum=2,
    )
    frame_width = channels * sample_width
    if not data or len(data) % frame_width:
        raise ValueError("raw PCM frame data is empty or truncated")
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(sample_width)
        target.setframerate(sample_rate)
        target.writeframes(data)
    converted = output.getvalue()
    if len(converted) > maximum_bytes:
        raise ValueError("wrapped voice WAV exceeds the byte limit")
    return converted


def _required_pcm_integer(
    metadata: Mapping[str, object],
    key: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(metadata[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"raw PCM {key} metadata is unavailable") from None
    if value < minimum or value > maximum:
        raise ValueError(f"raw PCM {key} metadata is unsupported")
    return value


def validate_pcm_wav(data: bytes, *, maximum_bytes: int) -> float:
    payload = bytes(data)
    if not payload or len(payload) > max(1, int(maximum_bytes)):
        raise ValueError("voice WAV byte size is outside the allowed range")
    if not _wav_magic(payload):
        raise ValueError("voice artifact is not a RIFF/WAVE file")
    try:
        with wave.open(io.BytesIO(payload), "rb") as source:
            if source.getcomptype() != "NONE":
                raise ValueError("voice WAV must use uncompressed PCM")
            channels = int(source.getnchannels())
            sample_width = int(source.getsampwidth())
            sample_rate = int(source.getframerate())
            frames = int(source.getnframes())
            if channels not in {1, 2}:
                raise ValueError("voice WAV channel count is unsupported")
            if sample_width not in {1, 2, 3, 4}:
                raise ValueError("voice WAV sample width is unsupported")
            if sample_rate < 8_000 or sample_rate > 96_000:
                raise ValueError("voice WAV sample rate is unsupported")
            if frames <= 0:
                raise ValueError("voice WAV has no audio frames")
            duration = frames / sample_rate
            if duration <= 0 or duration > MAX_VOICE_DURATION_SECONDS:
                raise ValueError("voice WAV duration is outside the allowed range")
            expected_frame_bytes = frames * channels * sample_width
            decoded = source.readframes(frames)
            if len(decoded) != expected_frame_bytes:
                raise ValueError("voice WAV frame data is truncated")
    except (EOFError, wave.Error) as exc:
        raise ValueError("voice WAV structure is invalid") from exc
    return duration


def _managed_ffmpeg_to_pcm_wav(
    data: bytes,
    mime_type: str,
    suffix: str,
    maximum_bytes: int,
    work_root: Path,
) -> bytes:
    return _ffmpeg_executable_to_pcm_wav(
        managed_ffmpeg_executable(),
        data,
        mime_type,
        suffix,
        maximum_bytes,
        work_root,
        runtime_label="managed FFmpeg",
    )


def _ffmpeg_cli_to_pcm_wav(
    data: bytes,
    mime_type: str,
    suffix: str,
    maximum_bytes: int,
    work_root: Path,
) -> bytes:
    return _ffmpeg_executable_to_pcm_wav(
        "ffmpeg",
        data,
        mime_type,
        suffix,
        maximum_bytes,
        work_root,
        runtime_label="ffmpeg CLI",
    )


def _ffmpeg_executable_to_pcm_wav(
    executable: str,
    data: bytes,
    mime_type: str,
    suffix: str,
    maximum_bytes: int,
    work_root: Path,
    *,
    runtime_label: str,
) -> bytes:
    with _conversion_paths(work_root, mime_type, suffix) as paths:
        paths.source.write_bytes(data)
        try:
            completed = subprocess.run(
                [
                    str(executable),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(paths.source),
                    "-t",
                    str(MAX_VOICE_DURATION_SECONDS),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(paths.target),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=_CONVERSION_TIMEOUT_SECONDS,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"{runtime_label} is unavailable") from exc
        if completed.returncode != 0:
            raise ValueError(f"{runtime_label} rejected outbound audio")
        return _read_bounded(paths.target, maximum_bytes)


class _ConversionPaths:
    def __init__(self, work_root: Path, mime_type: str, suffix: str) -> None:
        self.work_root = work_root
        self.mime_type = mime_type
        self.suffix = suffix
        self.directory: Path | None = None
        self.source: Path
        self.target: Path

    def __enter__(self) -> _ConversionPaths:
        directory = Path(tempfile.mkdtemp(prefix=".vc-", dir=str(self.work_root))).resolve()
        if directory.parent != self.work_root:
            raise RuntimeError("voice conversion directory escaped its managed root")
        self.directory = directory
        token = uuid.uuid4().hex[:16]
        extension = _source_extension(self.mime_type, self.suffix)
        self.source = directory / f"{token}{extension}"
        self.target = directory / f"{token}.wav"
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        for path in (getattr(self, "source", None), getattr(self, "target", None)):
            if isinstance(path, Path):
                path.unlink(missing_ok=True)
        if self.directory is not None:
            # Unexpected converter side files stay confined below the voice
            # root and are removed by the normal orphan sweep.
            with contextlib.suppress(OSError):
                self.directory.rmdir()


def _conversion_paths(work_root: Path, mime_type: str, suffix: str) -> _ConversionPaths:
    return _ConversionPaths(work_root, mime_type, suffix)


def _source_extension(mime_type: str, suffix: str) -> str:
    if suffix in set(_SOURCE_EXTENSIONS.values()):
        return suffix
    return _SOURCE_EXTENSIONS.get(mime_type, ".bin")


def _read_bounded(path: Path, maximum_bytes: int) -> bytes:
    if not path.is_file():
        raise ValueError("audio converter did not create a WAV file")
    if int(path.stat().st_size) < 1 or int(path.stat().st_size) > int(maximum_bytes):
        raise ValueError("converted voice WAV byte size is outside the allowed range")
    with path.open("rb") as source:
        data = source.read(int(maximum_bytes) + 1)
    if len(data) > int(maximum_bytes):
        raise ValueError("converted voice WAV exceeds the byte limit")
    return data


def _wav_magic(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


__all__ = [
    "MAX_VOICE_DURATION_SECONDS",
    "normalize_outbound_voice_audio",
    "validate_pcm_wav",
]
