"""Domain records for portable, profile-agnostic SoulCore role packages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..character_model import CharacterModel
from ..media import StoredMediaFile

ROLE_PACKAGE_EXTENSION = ".soulcore-role"
ROLE_PACKAGE_FORMAT = "soulcore-role-package"
ROLE_PACKAGE_FORMAT_VERSION = 1
ROLE_PACKAGE_CONTENT_MODE = "sparse_patch"
ROLE_PACKAGE_MIME = "application/vnd.soulcore.role-package+zip"

MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_EXPANDED_BYTES = 192 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
MAX_ROLE_JSON_BYTES = 128 * 1024 * 1024
MAX_PACKAGE_FILES = 4
UPLOAD_CHUNK_BYTES = 256 * 1024


class RolePackageError(ValueError):
    """A deterministic, user-correctable package validation failure."""

    def __init__(self, message: str, *, field: str = "") -> None:
        self.field = str(field or "")
        super().__init__(message)


class RolePackageConflict(RolePackageError):
    """The preview target or package changed before confirmation."""


@dataclass(frozen=True, slots=True)
class PackageAsset:
    scope: str
    path: str
    mime_type: str
    sha256: str
    byte_size: int
    data: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class ParsedRolePackage:
    title: str
    generator_version: str
    role: dict[str, Any]
    assets: dict[str, PackageAsset]
    archive_sha256: str
    archive_path: Path


@dataclass(frozen=True, slots=True)
class PortraitSnapshot:
    scope: str
    reference_id: str
    asset_id: str
    storage_relpath: str
    mime_type: str
    file_extension: str
    sha256: str
    byte_size: int
    width: int
    height: int
    frame_count: int
    duration_ms: int
    label: str
    identity_description: str
    file_status: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class RoleDatabaseSnapshot:
    title: str
    character_revision: int
    character_fingerprint: str
    character: CharacterModel
    world_revision: int
    world_definition: dict[str, Any]
    lore: tuple[dict[str, Any], ...]
    boundaries: tuple[dict[str, Any], ...]
    portraits: dict[str, PortraitSnapshot | None]


@dataclass(frozen=True, slots=True)
class PortraitMutation:
    scope: str
    action: str
    label: str = ""
    stored: StoredMediaFile | None = None
    cleanup_guard_id: int = 0
    duration_ms: int = 0


@dataclass(frozen=True, slots=True)
class ImportState:
    character: CharacterModel
    world_definition: dict[str, Any]
    lore: tuple[dict[str, Any], ...]
    boundaries: tuple[dict[str, Any], ...]
    world_definition_present: bool
    lore_present: bool
    boundaries_present: bool
    portrait_actions: dict[str, dict[str, Any]]
    changed: bool
    character_changed: bool
    world_changed: bool
    portrait_changed: dict[str, bool]


@dataclass(frozen=True, slots=True)
class ApplyResult:
    replayed: bool
    changed: bool
    character_revision: int
    world_revision: int
    changed_sections: tuple[str, ...]
    cleanup_targets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreviewSession:
    token: str
    role_ref: str
    profile_id: str
    package_path: Path
    package_sha256: str
    snapshot: RoleDatabaseSnapshot
    state: ImportState
    preview: dict[str, Any]
    expires_at: float


@dataclass(frozen=True, slots=True)
class ExportSession:
    token: str
    role_ref: str
    profile_id: str
    path: Path
    filename: str
    expires_at: float


__all__ = [name for name in globals() if not name.startswith("_")]
