"""Resolve the packaged FFmpeg executable without owning audio policy."""

from __future__ import annotations

from pathlib import Path


def managed_ffmpeg_executable() -> str:
    """Return the executable supplied by the declared imageio-ffmpeg runtime."""

    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        candidate = Path(get_ffmpeg_exe()).resolve(strict=True)
    except Exception as exc:
        raise RuntimeError("managed FFmpeg runtime is unavailable") from exc
    if not candidate.is_file():
        raise RuntimeError("managed FFmpeg runtime is not a file")
    return str(candidate)


__all__ = ["managed_ffmpeg_executable"]
