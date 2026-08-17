"""Persistence boundary for file artifact administration and release."""

from pathlib import Path
from typing import Any, Protocol, TypedDict

from ...contracts.runtime_cleanup import RuntimeFileCleanupEntry, RuntimeFileKind


class FileRepositoryPort(Protocol):
    async def get_profile_file_artifacts_enabled(self, *args: object, **kwargs: object) -> Any: ...
    async def set_profile_file_artifacts_enabled(self, *args: object, **kwargs: object) -> Any: ...
    async def finalize_file_artifact_release(self, *args: object, **kwargs: object) -> Any: ...
    async def list_pending_file_artifact_releases(self, *args: object, **kwargs: object) -> Any: ...
    async def settle_pending_file_artifact_release(
        self, *args: object, **kwargs: object
    ) -> Any: ...
    async def list_file_artifact_records(self, *args: object, **kwargs: object) -> Any: ...
    async def prepare_expired_file_artifact_releases(
        self, *args: object, **kwargs: object
    ) -> Any: ...
    async def prepare_file_artifact_release(self, *args: object, **kwargs: object) -> Any: ...
    async def get_file_assets_for_todos(self, *args: object, **kwargs: object) -> Any: ...
    async def list_pending_important_file_todos(self, *args: object, **kwargs: object) -> Any: ...
    async def settle_file_todos(self, *args: object, **kwargs: object) -> Any: ...


class FileMediaAssetView(Protocol):
    mime_type: str
    width: int | None
    height: int | None


class FileMediaProjectionView(Protocol):
    history_projection: str
    visible_facts: str


class FileMediaRepositoryPort(Protocol):
    async def get_media_asset(
        self,
        asset_id: str,
        *,
        profile_id: str,
        instance_id: str,
    ) -> FileMediaAssetView | None: ...

    async def get_latest_media_projection(
        self,
        asset_id: str,
    ) -> FileMediaProjectionView | None: ...


class FileWorkCallbackPort(Protocol):
    def complete_file_job(self, *args: object, **kwargs: object) -> bool: ...


class PendingFileArtifactRelease(TypedDict):
    profile_id: str
    instance_id: str
    asset_id: str
    storage_relpath: str
    byte_size: int
    sha256: str
    updated_at: str


class FileArtifactReleaseRepositoryPort(Protocol):
    async def list_pending_file_artifact_releases(
        self,
        *,
        limit: int = 100,
    ) -> list[PendingFileArtifactRelease]: ...

    async def settle_pending_file_artifact_release(
        self,
        profile_id: str,
        instance_id: str,
        asset_id: str,
        *,
        expected_updated_at: str,
        released: bool,
        error: str = "",
    ) -> str: ...


class RuntimeFileCleanupRepositoryPort(Protocol):
    async def list_runtime_file_cleanup(
        self,
        *,
        limit: int = 100,
    ) -> tuple[RuntimeFileCleanupEntry, ...]: ...

    async def get_runtime_file_cleanup_by_path(
        self,
        storage_kind: RuntimeFileKind | str,
        storage_relpath: str,
    ) -> RuntimeFileCleanupEntry | None: ...

    async def claim_runtime_file_cleanup(self, cleanup_id: int) -> str | None: ...

    async def runtime_file_cleanup_is_owned(self, entry: RuntimeFileCleanupEntry) -> bool: ...

    async def complete_claimed_runtime_file_cleanup(
        self,
        cleanup_id: int,
        *,
        claim_token: str,
    ) -> bool: ...

    async def record_claimed_runtime_file_cleanup_failure(
        self,
        cleanup_id: int,
        *,
        claim_token: str,
        error: str,
    ) -> bool: ...


class FileArtifactStorePort(Protocol):
    def resolve_path(self, storage_relpath: str) -> Path: ...

    def release(self, storage_relpath: str) -> bool: ...


class RuntimeMediaStorePort(Protocol):
    def absolute_path(self, relative_path: str) -> Path: ...

    def delete(self, relative_path: str | None) -> bool: ...


__all__ = [
    "FileArtifactReleaseRepositoryPort",
    "FileArtifactStorePort",
    "FileMediaAssetView",
    "FileMediaProjectionView",
    "FileMediaRepositoryPort",
    "FileRepositoryPort",
    "FileWorkCallbackPort",
    "PendingFileArtifactRelease",
    "RuntimeFileCleanupRepositoryPort",
    "RuntimeMediaStorePort",
]
