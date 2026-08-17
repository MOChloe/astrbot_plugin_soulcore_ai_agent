from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class MediaOrigin(StrEnum):
    USER_INPUT = "USER_INPUT"
    GENERATED = "GENERATED"
    STICKER_RESERVED = "STICKER_RESERVED"


class MediaPurpose(StrEnum):
    NORMAL_IMAGE = "NORMAL_IMAGE"
    GENERATED_IMAGE = "GENERATED_IMAGE"
    STICKER = "STICKER"


class MediaFileStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    RELEASE_PENDING = "RELEASE_PENDING"
    RELEASED = "RELEASED"
    MISSING = "MISSING"
    QUARANTINED = "QUARANTINED"


class MediaInspectionStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    READY = "READY"
    FAILED = "FAILED"
    NOT_REQUIRED = "NOT_REQUIRED"


class MediaProjectionStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    READY = "READY"
    FAILED = "FAILED"


class InboundMediaRegistrationState(StrEnum):
    COMMITTED = "COMMITTED"
    OWNED_INCOMPLETE = "OWNED_INCOMPLETE"
    UNOWNED = "UNOWNED"


@dataclass(slots=True)
class StoredMediaFile:
    asset_id: str
    relative_path: str
    mime_type: str
    file_extension: str
    sha256: str
    byte_size: int
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None


@dataclass(slots=True)
class MediaAsset:
    asset_id: str
    profile_id: str
    instance_id: str
    origin: MediaOrigin
    purpose: MediaPurpose
    mime_type: str
    file_extension: str
    sha256: str
    byte_size: int
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    storage_relpath: str | None = None
    file_status: MediaFileStatus = MediaFileStatus.AVAILABLE
    delivery_status: str = "NOT_SENT"
    inspection_status: MediaInspectionStatus = MediaInspectionStatus.PENDING
    current_projection_version: int = 0
    core_run_id: int | None = None
    ai_task_id: int | None = None
    summary_covered_by: int | None = None
    expires_at: datetime | None = None
    released_at: datetime | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class MediaProjection:
    projection_id: int
    asset_id: str
    version: int
    status: MediaProjectionStatus
    visible_facts: str = ""
    history_projection: str = ""
    ocr_text: str = ""
    backend_id: str = ""
    model_id: str = ""
    ai_task_id: int | None = None
    error: str | None = None
    created_at: datetime | None = None
