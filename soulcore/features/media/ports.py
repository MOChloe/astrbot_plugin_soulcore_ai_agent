"""Narrow persistence boundary used by media orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol

from ...contracts.runtime_cleanup import RuntimeFileCleanupEntry
from .domain import (
    InboundMediaRegistrationState,
    MediaAsset,
    MediaFileStatus,
    MediaInspectionStatus,
    MediaOrigin,
    MediaPurpose,
    StoredMediaFile,
)
from .inspection import (
    MAX_ANIMATION_DECODED_PIXELS,
    MAX_ANIMATION_DURATION_MS,
    MAX_ANIMATION_FRAMES,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_DECODED_PIXELS,
)
from .visual_cache import CachedVisualObservation
from .visual_cache import (
    VisualCachePolicy as VisualCachePolicy,
)

__all__ = [
    "MAX_ANIMATION_DECODED_PIXELS",
    "MAX_ANIMATION_DURATION_MS",
    "MAX_ANIMATION_FRAMES",
    "MAX_IMAGE_BYTES",
    "MAX_IMAGE_DECODED_PIXELS",
    "StoredMediaFile",
]


class DatabasePath(Protocol):
    path: str | Path


class CharacterInstanceView(Protocol):
    scope: str
    route_umo: str


class ScopeConfigView(Protocol):
    world_texture_prompt: str
    media_original_retention_days: int


class CreativeBoundaryView(Protocol):
    @property
    def severity(self) -> str: ...

    def render(self) -> str: ...


class WorldDefinitionView(Protocol):
    @property
    def world_brief(self) -> str: ...

    @property
    def world_rules(self) -> str: ...

    @property
    def world_texture(self) -> str: ...

    @property
    def boundaries(self) -> Sequence[CreativeBoundaryView]: ...


class VisualWorldPort(Protocol):
    async def get_world_definition(self, profile_id: str) -> WorldDefinitionView: ...


class GroupPreviewLocatorPort(Protocol):
    async def asset_file_path(
        self, *, profile_id: str, instance_id: str, asset_id: str
    ) -> Path | None: ...


class VisualProfilesPort(Protocol):
    async def get_profile_image_generation_enabled(self, profile_id: str) -> bool: ...

    async def get_character_instance(
        self, profile_id: str, instance_id: str
    ) -> CharacterInstanceView | None: ...

    async def get_scope_config(self, profile_id: str, scope: str) -> ScopeConfigView | None: ...


class StickerCandidateStatusView(Protocol):
    @property
    def value(self) -> str: ...


class StickerCandidateView(Protocol):
    @property
    def source_asset_id(self) -> str: ...

    @property
    def status(self) -> StickerCandidateStatusView: ...


class StickerItemView(Protocol):
    asset_id: str


class StickerAssetView(Protocol):
    asset_id: str
    storage_relpath: str
    byte_size: int
    sha256: str


class StickerPromotionRepositoryPort(Protocol):
    async def get_sticker_candidate(
        self, profile_id: str, instance_id: str, candidate_id: str
    ) -> StickerCandidateView | None: ...

    async def find_sticker_asset_by_sha(
        self, profile_id: str, sha256: str
    ) -> StickerAssetView | None: ...

    async def create_sticker_asset(
        self, profile_id: str, stored: StoredMediaFile, *, duration_ms: int = 0
    ) -> tuple[StickerAssetView, bool]: ...

    async def delete_unreferenced_sticker_asset(self, sticker_asset_id: str) -> bool: ...

    async def list_pending_sticker_releases(
        self, *, limit: int = 100
    ) -> list[StickerAssetView]: ...

    async def claim_sticker_asset_release(
        self, sticker_asset_id: str
    ) -> StickerAssetView | None: ...

    async def accept_sticker_candidate(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
        *,
        reserved_asset_id: str,
        compact_description: str = "",
        compact_name: str = "",
        visible_text: str = "",
        ocr_text: str = "",
        usage_type: str = "",
        vibe_tags: list[str] | tuple[str, ...] = (),
        search_keywords: list[str] | tuple[str, ...] = (),
        search_index: str = "",
        semantic_key: str = "",
        emotion: str = "",
        speech_act: str = "",
        intensity: int = 0,
        persona_score: float = 0.0,
        phash: str = "",
        dhash: str = "",
        frame_hashes: list[str] | tuple[str, ...] = (),
        visual_group: str = "",
        metadata: Mapping[str, object] | None = None,
    ) -> tuple[object, bool]: ...

    async def quarantine_sticker_candidate(
        self,
        profile_id: str,
        instance_id: str,
        candidate_id: str,
        *,
        reason: str,
    ) -> object: ...


class MediaRepositoryPort(Protocol):
    db: DatabasePath

    async def asset_is_model_visible(
        self, profile_id: str, instance_id: str, asset_id: str
    ) -> bool: ...

    async def list_available_image_asset_ids_for_messages(
        self, *args: object, **kwargs: object
    ) -> object: ...

    async def list_available_attachment_refs_for_messages(
        self, *args: object, **kwargs: object
    ) -> object: ...

    async def create_media_asset(
        self,
        profile_id: str,
        instance_id: str,
        stored: StoredMediaFile,
        *,
        origin: MediaOrigin | str,
        purpose: MediaPurpose | str,
        delivery_status: str = "NOT_SENT",
        inspection_status: MediaInspectionStatus | str = MediaInspectionStatus.PENDING,
        core_run_id: int | None = None,
        ai_task_id: int | None = None,
        expires_at: datetime | None = None,
        metadata: dict[str, object] | None = None,
        revive_missing_file: bool = False,
        cleanup_guard_id: int | None = None,
    ) -> MediaAsset: ...

    async def register_inbound_media_asset(
        self,
        profile_id: str,
        instance_id: str,
        stored: StoredMediaFile,
        *,
        message_id: int,
        ordinal: int,
        platform_message_id: str = "",
        delivery_status: str = "NOT_SENT",
        inspection_status: MediaInspectionStatus | str = MediaInspectionStatus.PENDING,
        metadata: dict[str, object] | None = None,
    ) -> MediaAsset: ...

    async def inspect_inbound_media_registration(
        self,
        profile_id: str,
        instance_id: str,
        stored: StoredMediaFile,
        *,
        message_id: int,
        ordinal: int,
        platform_message_id: str = "",
    ) -> tuple[InboundMediaRegistrationState, MediaAsset | None]: ...

    async def get_media_asset(
        self,
        asset_id: str,
        *,
        profile_id: str | None = None,
        instance_id: str | None = None,
    ) -> MediaAsset | None: ...

    async def get_cached_visual_observation(
        self,
        profile_id: str,
        instance_id: str,
        sha256: str,
        *,
        contract_version: int,
    ) -> CachedVisualObservation | None: ...

    async def save_cached_visual_observation(
        self,
        profile_id: str,
        instance_id: str,
        asset_id: str,
        sha256: str,
        observation: CachedVisualObservation,
        *,
        contract_version: int,
    ) -> int: ...

    async def prune_visual_observation_cache(
        self,
        *,
        contract_version: int,
    ) -> int: ...

    async def list_media_assets(
        self,
        profile_id: str,
        instance_id: str,
        *,
        origin: MediaOrigin | str | None = None,
        file_status: MediaFileStatus | str | None = None,
        core_run_id: int | None = None,
        mime_prefix: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MediaAsset]: ...

    async def link_media_to_message(
        self,
        profile_id: str,
        instance_id: str,
        asset_id: str,
        message_id: int,
        *,
        relation: str = "ATTACHMENT",
        ordinal: int = 0,
    ) -> None: ...

    async def mark_summary_covered_media_for_release(
        self, profile_id: str, instance_id: str, summary_id: int
    ) -> list[MediaAsset]: ...

    async def mark_media_release_if_already_summarized(self, asset_id: str) -> list[MediaAsset]: ...

    async def mark_expired_media_for_release(
        self, *, now: datetime | None = None, limit: int = 100
    ) -> list[MediaAsset]: ...

    async def mark_retention_elapsed_media_for_release(
        self, *, now: datetime | None = None, limit: int = 100
    ) -> list[MediaAsset]: ...

    async def list_pending_media_releases(self, *, limit: int = 100) -> list[MediaAsset]: ...

    async def finalize_media_release(
        self, asset_id: str, *, success: bool, error: str | None = None
    ) -> MediaAsset: ...

    async def mark_media_missing(self, asset_id: str, *, reason: str) -> MediaAsset: ...

    async def get_latest_media_projection(self, *args: object, **kwargs: object) -> object: ...
    async def media_history_projections_for_messages(
        self,
        profile_id: str,
        instance_id: str,
        message_ids: Sequence[int],
    ) -> dict[int, list[dict[str, str]]]: ...
    async def register_generated_media_asset(self, *args: object, **kwargs: object) -> object: ...
    async def guard_unregistered_media_file(
        self,
        profile_id: str,
        instance_id: str,
        stored: StoredMediaFile,
        *,
        reason: str,
    ) -> RuntimeFileCleanupEntry: ...
    async def complete_runtime_file_cleanup(self, cleanup_id: int) -> bool: ...
    async def register_platform_media_reference(
        self, *args: object, **kwargs: object
    ) -> object: ...
    async def save_media_projection(self, *args: object, **kwargs: object) -> object: ...
    async def list_media_cleanup_events(self, *args: object, **kwargs: object) -> object: ...
    async def media_asset_statistics(self, *args: object, **kwargs: object) -> object: ...
    async def finalize_media_delivery(self, *args: object, **kwargs: object) -> object: ...
