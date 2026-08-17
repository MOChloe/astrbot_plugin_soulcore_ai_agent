from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

STICKER_CHECK_FAILURE_LIMIT = 5
DEFAULT_STICKER_REQUIREMENTS = (
    "不要水印、Logo、网址、账号、署名或平台角标；不要与表情内容无关的文字；不要真人照片。"
)


class StickerSourceKind(StrEnum):
    PLAYER = "PLAYER"
    WEB = "WEB"
    GENERATED = "GENERATED"
    UPLOAD = "UPLOAD"


@dataclass(frozen=True, slots=True)
class StickerCollectedAsset:
    """One exact collector result before automatic-admission metadata is frozen."""

    asset_id: str
    source_kind: StickerSourceKind
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError("sticker collected asset id cannot be empty")


@dataclass(frozen=True, slots=True)
class StickerCandidateSource:
    asset_id: str
    source_kind: StickerSourceKind
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.asset_id.strip():
            raise ValueError("sticker candidate asset id cannot be empty")
        if not isinstance(self.metadata.get("collection_intent"), Mapping):
            raise ValueError("automatic sticker candidate requires collection_intent")


@dataclass(frozen=True, slots=True)
class StickerImportIntent:
    """A run-scoped request that becomes durable only with the MainCore result."""

    source_ref: str
    source_kind: StickerSourceKind
    source_asset_id: str


def sticker_import_source_ref(
    profile_id: str,
    instance_id: str,
    run_id: int,
    source_kind: StickerSourceKind | str,
    source_asset_id: str,
) -> str:
    """Build the opaque proof binding one import source to one MainCore run."""

    kind = (
        source_kind
        if isinstance(source_kind, StickerSourceKind)
        else StickerSourceKind(str(source_kind).upper())
    )
    material = (
        f"{str(profile_id)}\0{str(instance_id)}\0{int(run_id)}\0"
        f"{kind.value.lower()}\0{str(source_asset_id)}"
    )
    return "ss_" + hashlib.sha256(material.encode()).hexdigest()[:20]


class StickerLibraryKind(StrEnum):
    CORE = "CORE"
    PRIVATE = "PRIVATE"


class StickerUsageType(StrEnum):
    AMBIENT = "AMBIENT"
    REACTION = "REACTION"
    SPECIFIC = "SPECIFIC"


class StickerCandidateStatus(StrEnum):
    PENDING = "PENDING"
    CHECKING = "CHECKING"
    WAITING_CHECK = "WAITING_CHECK"
    READY = "READY"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    QUARANTINED = "QUARANTINED"


class StickerItemStatus(StrEnum):
    ACTIVE = "ACTIVE"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    ARCHIVED = "ARCHIVED"
    DELETED = "DELETED"


class StickerIntakeKind(StrEnum):
    UPLOAD = "UPLOAD"
    SEARCH = "SEARCH"


class StickerIntakeStatus(StrEnum):
    UPLOADING = "UPLOADING"
    RUNNING = "RUNNING"
    REVIEW = "REVIEW"
    FINALIZING = "FINALIZING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class StickerIntakeEntryStatus(StrEnum):
    PENDING = "PENDING"
    UPLOADED = "UPLOADED"
    ANALYZING = "ANALYZING"
    READY = "READY"
    REJECTED = "REJECTED"
    DUPLICATE = "DUPLICATE"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"
    IMPORTED = "IMPORTED"


class StickerCheckVerdict(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    QUARANTINE = "QUARANTINE"


@dataclass(frozen=True, slots=True)
class StickerConfig:
    profile_id: str
    scope: str
    enabled: bool = False
    player_collection_enabled: bool = False
    web_collection_enabled: bool = False
    generation_enabled: bool = False
    trigger_mode: str = "TURNS_ONLY"
    turn_threshold: int = 20
    elapsed_hours: float = 24.0
    library_limit: int = 1000
    web_daily_limit: int = 4
    generated_daily_limit: int = 1
    requirements: str = DEFAULT_STICKER_REQUIREMENTS
    version: int = 1
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class StickerAsset:
    sticker_asset_id: str
    profile_id: str
    canonical_sha256: str
    storage_relpath: str
    mime_type: str
    file_extension: str
    byte_size: int
    width: int
    height: int
    is_animated: bool = False
    frame_count: int = 1
    duration_ms: int = 0
    file_status: str = "AVAILABLE"
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def asset_id(self) -> str:
        return self.sticker_asset_id

    @property
    def sha256(self) -> str:
        return self.canonical_sha256


@dataclass(frozen=True, slots=True)
class StickerCandidate:
    candidate_id: str
    profile_id: str
    instance_id: str
    source_kind: StickerSourceKind
    source_asset_id: str
    source_ref: str = ""
    target_library_id: str = ""
    status: StickerCandidateStatus = StickerCandidateStatus.PENDING
    import_count: int = 1
    persona_fingerprint: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    last_error: str = ""
    failure_stage: str = ""
    retry_count: int = 0
    next_retry_at: datetime | None = None
    recoverable: bool = False
    accepted_item_id: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class StickerCheckRevision:
    check_id: int
    candidate_id: str
    revision: int
    verdict: StickerCheckVerdict
    compact_name: str = ""
    compact_description: str = ""
    visible_text: str = ""
    usage_type: StickerUsageType = StickerUsageType.REACTION
    semantic_key: str = ""
    emotion: str = ""
    speech_act: str = ""
    intensity: int = 0
    persona_score: float = 0.0
    reason: str = ""
    backend_id: str = ""
    model_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class StickerItem:
    item_id: str
    profile_id: str
    instance_id: str
    asset_id: str
    canonical_sha256: str
    source_kind: StickerSourceKind
    library_id: str = ""
    library_kind: StickerLibraryKind = StickerLibraryKind.PRIVATE
    scope: str = ""
    usage_type: StickerUsageType = StickerUsageType.REACTION
    compact_name: str = ""
    compact_description: str = ""
    visible_text: str = ""
    ocr_text: str = ""
    vibe_tags: tuple[str, ...] = ()
    search_keywords: tuple[str, ...] = ()
    search_index: str = ""
    semantic_key: str = ""
    cluster_id: str = ""
    emotion: str = ""
    speech_act: str = ""
    intensity: int = 0
    persona_score: float = 0.0
    status: StickerItemStatus = StickerItemStatus.ACTIVE
    import_count: int = 1
    reinforcement_score: float = 0.0
    usage_count: int = 0
    last_used_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    phash: str = ""
    dhash: str = ""
    visual_group: str = ""
    mime_type: str = "image/png"
    is_animated: bool = False
    frame_count: int = 1
    duration_ms: int = 0
    representative_frame_hashes: tuple[str, ...] = ()
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class StickerRunRef:
    sticker_ref: str
    profile_id: str
    instance_id: str
    run_id: str
    item_id: str
    compact_description: str
    expires_at: datetime
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class StickerUsage:
    usage_id: int
    profile_id: str
    instance_id: str
    item_id: str
    run_id: str
    sticker_ref: str
    compact_projection: str
    delivery_status: str
    outbox_id: int | None = None
    expression_ordinal: int | None = None
    message_id: int | None = None
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class CharacterIdentityReference:
    """The configuration-owned image which defines who the character is.

    The reference deliberately does not belong to a friend/group instance and
    is not normal chat media. ``identity_description`` may describe stable
    appearance features, but must not turn the source style, pose or background
    into generation constraints.
    """

    reference_id: str
    profile_id: str
    scope: str
    asset_id: str
    storage_relpath: str
    mime_type: str
    file_extension: str
    sha256: str
    byte_size: int
    width: int
    height: int
    frame_count: int = 1
    duration_ms: int = 0
    label: str = ""
    identity_description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
