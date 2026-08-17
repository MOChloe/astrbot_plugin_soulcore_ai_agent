"""Controlled, short-lived audio files used only for outbound voice delivery."""

from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from ...contracts.runtime_cleanup import RuntimeFileCleanupEntry, RuntimeFileKind
from .audio_normalization import validate_pcm_wav

MAX_VOICE_ARTIFACT_BYTES = 20 * 1024 * 1024
VOICE_ARTIFACT_TTL = timedelta(minutes=15)
VOICE_FALLBACK_PAYLOAD_KEY = "_voice_delivery_fallback_reason"
VOICE_FALLBACK_REASONS = frozenset(
    {
        "PLATFORM_UNSUPPORTED",
        "ARTIFACT_SERVICE_UNAVAILABLE",
        "ARTIFACT_REPOSITORY_UNAVAILABLE",
        "SYNTHESIS_FAILED",
        "AUDIO_NORMALIZATION_FAILED",
        "AUDIO_ARTIFACT_INVALID",
        "PLATFORM_DELIVERY_FAILED",
    }
)

_MIME_EXTENSIONS = {
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
_EXTENSION_MIMES = {
    extension: mime_type for mime_type, extension in reversed(tuple(_MIME_EXTENSIONS.items()))
}
_SAFE_EXTENSION = re.compile(r"^\.[a-z0-9]{1,8}$")


@dataclass(frozen=True, slots=True)
class VoiceArtifact:
    """Integrity-fenced metadata; the audio bytes are never represented here."""

    storage_relpath: str = field(repr=False)
    mime_type: str
    filename: str
    sha256: str
    byte_size: int


class VoiceArtifactService:
    """Own one filesystem root for retryable outbound TTS artifacts."""

    def __init__(
        self,
        root: str | Path,
        *,
        maximum_bytes: int = MAX_VOICE_ARTIFACT_BYTES,
        ttl: timedelta = VOICE_ARTIFACT_TTL,
    ) -> None:
        self.root = Path(root).resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)
        self._filesystem_root = _filesystem_path(self.root)
        self.maximum_bytes = max(1, int(maximum_bytes))
        self.ttl = ttl
        if self.ttl.total_seconds() <= 0:
            raise ValueError("voice artifact TTL must be positive")

    def materialize(
        self,
        *,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        text: str,
        data: bytes,
        mime_type: str,
        filename: str = "",
    ) -> VoiceArtifact:
        payload = bytes(data)
        if not payload or len(payload) > self.maximum_bytes:
            raise ValueError("voice artifact byte size is outside the allowed range")
        normalized_mime, extension = self._audio_type(mime_type, filename)
        validate_pcm_wav(payload, maximum_bytes=self.maximum_bytes)
        audio_digest = hashlib.sha256(payload).hexdigest()
        text_digest = self.text_fingerprint(text)
        relative = Path(
            self._scope_component(profile_id),
            self._scope_component(instance_id),
            f"v{int(outbox_id)}-{text_digest}-{audio_digest[:24]}{extension}",
        )
        destination = self._controlled_path(relative.as_posix())
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".tmp-{uuid.uuid4().hex[:12]}")
        try:
            temporary.write_bytes(payload)
            with temporary.open("r+b") as durable:
                os.fsync(durable.fileno())
            if int(temporary.stat().st_size) != len(payload):
                raise RuntimeError("voice artifact write was incomplete")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return VoiceArtifact(
            relative.as_posix(),
            normalized_mime,
            self._safe_filename(filename, extension),
            audio_digest,
            len(payload),
        )

    def from_cleanup_entry(self, entry: RuntimeFileCleanupEntry) -> VoiceArtifact:
        if entry.storage_kind is not RuntimeFileKind.VOICE_ARTIFACT:
            raise ValueError("runtime cleanup entry is not a voice artifact")
        extension = Path(entry.storage_relpath).suffix.lower()
        if extension != ".wav":
            raise ValueError("voice artifact is not a normalized WAV")
        return VoiceArtifact(
            entry.storage_relpath,
            "audio/wav",
            "voice.wav",
            entry.expected_sha256,
            int(entry.expected_byte_size),
        )

    def belongs_to(
        self,
        artifact: VoiceArtifact,
        *,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        text: str,
    ) -> bool:
        path = Path(artifact.storage_relpath)
        expected_parent = Path(
            self._scope_component(profile_id),
            self._scope_component(instance_id),
        )
        prefix = f"v{int(outbox_id)}-{self.text_fingerprint(text)}-"
        return path.parent == expected_parent and path.name.startswith(prefix)

    def resolve(self, artifact: VoiceArtifact, *, touch: bool = False) -> Path:
        path = self.resolve_path(artifact.storage_relpath)
        self._verify(path, artifact.byte_size, artifact.sha256)
        if touch:
            os.utime(path, None)
        return path

    def resolve_path(self, storage_relpath: str) -> Path:
        path = self._controlled_path(storage_relpath)
        if not path.is_file():
            raise FileNotFoundError(path.name)
        return path

    def release_artifact(self, artifact: VoiceArtifact) -> bool:
        path = self._controlled_path(artifact.storage_relpath)
        if path.exists():
            self._verify(path, artifact.byte_size, artifact.sha256)
        return self.release(artifact.storage_relpath)

    def release(self, storage_relpath: str) -> bool:
        path = self._controlled_path(storage_relpath)
        existed = path.exists()
        if existed and not path.is_file():
            raise ValueError("controlled voice artifact path is not a file")
        path.unlink(missing_ok=True)
        self._prune_empty_parents(path.parent)
        return existed

    def purge_orphans(self, *, now: float | None = None) -> int:
        """Delete only stale files below the controlled root.

        The durable runtime cleanup queue remains authoritative.  This scan is
        the final safety net for a crash between the atomic file write and its
        cleanup-intent registration.
        """

        cutoff = float(time.time() if now is None else now) - self.ttl.total_seconds()
        removed = 0
        for path in tuple(self._filesystem_root.rglob("*")):
            if not path.is_file() or path.stat().st_mtime > cutoff:
                continue
            if path.name.startswith(".tmp-"):
                path.unlink(missing_ok=True)
                removed += 1
                continue
            if path.suffix.lower() not in _EXTENSION_MIMES:
                continue
            path.unlink(missing_ok=True)
            removed += 1
        for path in sorted(
            (item for item in self._filesystem_root.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            self._prune_empty_parents(path)
        return removed

    def _controlled_path(self, storage_relpath: str) -> Path:
        relative = Path(str(storage_relpath or ""))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("invalid controlled voice artifact path")
        path = (self.root / relative).resolve(strict=False)
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("voice artifact escaped its controlled root") from exc
        return _filesystem_path(path)

    def _prune_empty_parents(self, start: Path) -> None:
        parent = start
        while parent != self._filesystem_root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    @staticmethod
    def _verify(path: Path, expected_size: int, expected_sha256: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(path.name)
        if int(path.stat().st_size) != int(expected_size):
            raise RuntimeError("voice artifact byte size changed")
        if hashlib.sha256(path.read_bytes()).hexdigest() != str(expected_sha256):
            raise RuntimeError("voice artifact SHA-256 changed")

    @staticmethod
    def _scope_component(value: str) -> str:
        return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def text_fingerprint(text: str) -> str:
        return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _audio_type(mime_type: str, filename: str) -> tuple[str, str]:
        normalized = str(mime_type or "").split(";", 1)[0].strip().lower()
        if normalized in {"audio/wav", "audio/wave", "audio/x-wav"}:
            return "audio/wav", ".wav"
        suffix = Path(str(filename or "")).suffix.lower()
        if normalized == "application/octet-stream" and suffix == ".wav":
            return "audio/wav", ".wav"
        raise ValueError("voice artifact must be normalized to WAV before materialization")

    @staticmethod
    def _safe_filename(filename: str, extension: str) -> str:
        candidate = Path(str(filename or "")).name
        if not candidate or Path(candidate).suffix.lower() != extension:
            return f"voice{extension}"
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(candidate).stem).strip(" ._")
        if not stem:
            stem = "voice"
        suffix = Path(candidate).suffix.lower()
        if not _SAFE_EXTENSION.fullmatch(suffix):
            suffix = extension
        return f"{stem[:64]}{suffix}"


def _filesystem_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    raw = str(path)
    if raw.startswith("\\\\?\\"):
        return path
    if raw.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + raw[2:])
    return Path("\\\\?\\" + raw)


__all__ = [
    "MAX_VOICE_ARTIFACT_BYTES",
    "VOICE_ARTIFACT_TTL",
    "VOICE_FALLBACK_PAYLOAD_KEY",
    "VOICE_FALLBACK_REASONS",
    "VoiceArtifact",
    "VoiceArtifactService",
]
