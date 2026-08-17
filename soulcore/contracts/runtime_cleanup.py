"""Private result keys joining atomic runtime deletion to file cleanup."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

MEDIA_RELEASE_PATHS_KEY = "_media_release_paths"
FILE_ARTIFACT_RELEASE_PATHS_KEY = "_file_artifact_release_paths"
STICKER_RELEASE_PATHS_KEY = "_sticker_release_paths"
VOICE_ARTIFACT_RELEASE_PATHS_KEY = "_voice_artifact_release_paths"


class RuntimeFileKind(StrEnum):
    MEDIA = "MEDIA"
    FILE_ARTIFACT = "FILE_ARTIFACT"
    VOICE_ARTIFACT = "VOICE_ARTIFACT"


@dataclass(frozen=True, slots=True)
class RuntimeFileCleanupEntry:
    cleanup_id: int
    profile_id: str
    instance_id: str
    storage_kind: RuntimeFileKind
    storage_relpath: str
    owner_id: str
    expected_sha256: str
    expected_byte_size: int
    reason: str
    not_before_at: str
    attempt_count: int
    last_error: str


__all__ = [
    "FILE_ARTIFACT_RELEASE_PATHS_KEY",
    "MEDIA_RELEASE_PATHS_KEY",
    "RuntimeFileCleanupEntry",
    "RuntimeFileKind",
    "STICKER_RELEASE_PATHS_KEY",
    "VOICE_ARTIFACT_RELEASE_PATHS_KEY",
]
