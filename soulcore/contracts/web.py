"""Transport-neutral contracts for SoulCore web research."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def web_utc_now() -> datetime:
    return datetime.now(UTC)


class WebCallerKind(StrEnum):
    MAIN_CORE = "MAIN_CORE"
    BACKGROUND_AUTHOR = "BACKGROUND_AUTHOR"
    STICKER_COLLECTOR = "STICKER_COLLECTOR"


class WebDiagnosticOverride(StrEnum):
    """Explicit non-gameplay exception for administrator-owned provider probes."""

    ADMIN_PROVIDER_PROBE = "ADMIN_PROVIDER_PROBE"


class WebSearchPurpose(StrEnum):
    ANSWER_USER = "ANSWER_USER"
    SELF_EXPLORATION = "SELF_EXPLORATION"


class WebSearchDepth(StrEnum):
    AUTO = "auto"
    QUICK = "quick"
    BALANCED = "balanced"
    DEEP = "deep"


class WebSearchFreshness(StrEnum):
    AUTO = "auto"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    ANY = "any"


class WebSearchIntensity(StrEnum):
    ECONOMY = "ECONOMY"
    STANDARD = "STANDARD"
    DEEP = "DEEP"


class WebReadStatus(StrEnum):
    NOT_READ = "NOT_READ"
    READ = "READ"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True, slots=True)
class SearchRequest:
    profile_id: str
    instance_id: str
    caller_kind: WebCallerKind
    caller_id: str
    core_run_id: str = ""
    ai_task_id: str = ""
    purpose: WebSearchPurpose = WebSearchPurpose.ANSWER_USER
    query: str = ""
    depth: WebSearchDepth = WebSearchDepth.AUTO
    freshness: WebSearchFreshness = WebSearchFreshness.AUTO
    intensity: WebSearchIntensity = WebSearchIntensity.STANDARD
    operation_timeout_seconds: float = 300.0

    @property
    def run_scope(self) -> str:
        return self.core_run_id or self.ai_task_id


@dataclass(frozen=True, slots=True)
class ProviderSearchItem:
    title: str
    url: str
    snippet: str = ""
    published_at: str = ""
    favicon: str = ""
    provider_rank: int = 0


@dataclass(frozen=True, slots=True)
class ProviderSearchOutput:
    items: tuple[ProviderSearchItem, ...]
    provider_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class ProviderImageItem:
    image_url: str
    thumbnail_url: str = ""
    source_url: str = ""
    title: str = ""
    description: str = ""
    width: int = 0
    height: int = 0
    provider_rank: int = 0


@dataclass(frozen=True, slots=True)
class ProviderImageSearchOutput:
    items: tuple[ProviderImageItem, ...]
    provider_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class ProviderReadOutput:
    url: str
    content: str
    title: str = ""
    provider_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    resource_id: str
    title: str
    canonical_url: str
    domain: str
    snippet: str = ""
    published_at: str = ""
    retrieved_at: datetime = field(default_factory=web_utc_now)
    provider: str = ""
    provider_rank: int = 0
    cross_source_count: int = 1
    read_status: WebReadStatus = WebReadStatus.NOT_READ
    source_providers: tuple[str, ...] = ()
    score: float = 0.0

    def as_result_data(self) -> Mapping[str, Any]:
        return {
            "resource_id": self.resource_id,
            "title": self.title,
            "url": self.canonical_url,
            "domain": self.domain,
            "snippet": self.snippet,
            "published_at": self.published_at,
            "cross_source_count": self.cross_source_count,
            "read_status": self.read_status.value,
        }


@dataclass(frozen=True, slots=True)
class SearchResponse:
    session_id: str
    query: str
    purpose: WebSearchPurpose
    depth: WebSearchDepth
    results: tuple[WebSearchResult, ...]
    partial_warning: str = ""
    provider_errors: Mapping[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def as_result_data(self) -> Mapping[str, Any]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "purpose": self.purpose.value,
            "depth": self.depth.value,
            "results": [item.as_result_data() for item in self.results],
            "partial_warning": self.partial_warning,
        }


@dataclass(frozen=True, slots=True)
class ImageSearchResult:
    image_resource_id: str
    image_url: str
    thumbnail_url: str
    source_url: str
    title: str = ""
    description: str = ""
    width: int = 0
    height: int = 0
    provider: str = ""
    provider_rank: int = 0
    source_providers: tuple[str, ...] = ()
    retrieved_at: datetime = field(default_factory=web_utc_now)

    def as_result_data(self) -> Mapping[str, Any]:
        return {
            "image_resource_id": self.image_resource_id,
            "thumbnail_url": self.thumbnail_url,
            "source_url": self.source_url,
            "title": self.title,
            "description": self.description,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class ImageSearchResponse:
    session_id: str
    query: str
    purpose: WebSearchPurpose
    depth: WebSearchDepth
    results: tuple[ImageSearchResult, ...]
    partial_warning: str = ""
    provider_errors: Mapping[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def as_result_data(self) -> Mapping[str, Any]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "purpose": self.purpose.value,
            "depth": self.depth.value,
            "results": [item.as_result_data() for item in self.results],
            "partial_warning": self.partial_warning,
        }


@dataclass(frozen=True, slots=True)
class ReadRequest:
    profile_id: str
    instance_id: str
    caller_kind: WebCallerKind
    caller_id: str
    resource_ids: tuple[str, ...]
    core_run_id: str = ""
    ai_task_id: str = ""
    focus: str = ""
    operation_timeout_seconds: float = 300.0

    @property
    def run_scope(self) -> str:
        return self.core_run_id or self.ai_task_id


@dataclass(frozen=True, slots=True)
class WebPageContent:
    resource_id: str
    canonical_url: str
    title: str
    content: str
    provider: str
    retrieved_at: datetime = field(default_factory=web_utc_now)
    truncated: bool = False

    def as_result_data(self) -> Mapping[str, Any]:
        return {
            "resource_id": self.resource_id,
            "url": self.canonical_url,
            "title": self.title,
            "content": self.content,
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class ReadResponse:
    pages: tuple[WebPageContent, ...]
    errors: Mapping[str, str] = field(default_factory=dict)
    partial_warning: str = ""
    elapsed_seconds: float = 0.0

    def as_result_data(self) -> Mapping[str, Any]:
        return {
            "pages": [page.as_result_data() for page in self.pages],
            "errors": dict(self.errors),
            "partial_warning": self.partial_warning,
        }


class WebResearchError(RuntimeError):
    """Safe, model-readable web research failure."""

    def __init__(self, code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.code = str(code)
        self.safe_message = str(safe_message)


__all__ = [
    "ImageSearchResponse",
    "ImageSearchResult",
    "ProviderImageItem",
    "ProviderImageSearchOutput",
    "ProviderReadOutput",
    "ProviderSearchItem",
    "ProviderSearchOutput",
    "ReadRequest",
    "ReadResponse",
    "SearchRequest",
    "SearchResponse",
    "WebCallerKind",
    "WebPageContent",
    "WebReadStatus",
    "WebResearchError",
    "WebSearchDepth",
    "WebSearchFreshness",
    "WebSearchIntensity",
    "WebSearchPurpose",
    "WebSearchResult",
    "web_utc_now",
]
