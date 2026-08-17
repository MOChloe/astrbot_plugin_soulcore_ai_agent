from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from ...contracts.web import WebCallerKind, WebReadStatus, WebSearchPurpose


class WebSearchSessionStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class WebSearchKind(StrEnum):
    WEB = "WEB"
    IMAGE = "IMAGE"


@dataclass(slots=True)
class WebSearchProviderRecord:
    provider_id: str
    profile_id: str
    provider_kind: str
    display_name: str = ""
    backend_id: str = ""
    credential_id: str = ""
    priority: int = 1
    enabled: bool = True
    read_enabled: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    version: int = 1
    archived_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class WebSearchSessionRecord:
    session_id: str
    profile_id: str
    instance_id: str
    caller_kind: WebCallerKind
    caller_id: str
    purpose: WebSearchPurpose
    query: str
    search_kind: WebSearchKind = WebSearchKind.WEB
    depth: str = "auto"
    freshness: str = "auto"
    status: WebSearchSessionStatus = WebSearchSessionStatus.RUNNING
    core_run_id: int | None = None
    ai_task_id: str | None = None
    deadline_at: datetime | None = None
    partial_warning: str = ""
    provider_count: int = 0
    result_count: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    expires_at: datetime | None = None
    redacted_at: datetime | None = None


@dataclass(slots=True)
class WebSearchResultRecord:
    resource_id: str
    session_id: str
    profile_id: str
    instance_id: str
    title: str
    canonical_url: str
    domain: str
    snippet: str = ""
    published_at: datetime | None = None
    retrieved_at: datetime | None = None
    provider_id: str = ""
    provider_rank: int = 0
    cross_source_count: int = 1
    read_status: WebReadStatus = WebReadStatus.NOT_READ
    metadata: dict[str, Any] = field(default_factory=dict)
    expires_at: datetime | None = None
    redacted_at: datetime | None = None


@dataclass(slots=True)
class WebPageSnapshotRecord:
    snapshot_id: int
    resource_id: str
    profile_id: str
    instance_id: str
    content: str
    content_hash: str
    status: WebReadStatus
    token_estimate: int = 0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_at: datetime | None = None
    expires_at: datetime | None = None
    redacted_at: datetime | None = None


@dataclass(slots=True)
class WebImageSearchResultRecord:
    image_resource_id: str
    session_id: str
    profile_id: str
    instance_id: str
    original_url: str
    thumbnail_url: str
    source_page_url: str
    source_domain: str = ""
    title: str = ""
    description: str = ""
    provider_id: str = ""
    provider_rank: int = 0
    cross_source_count: int = 1
    width: int | None = None
    height: int | None = None
    mime_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_at: datetime | None = None
    expires_at: datetime | None = None
    redacted_at: datetime | None = None
