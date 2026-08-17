"""Run-scoped, multi-provider web research orchestration."""

from __future__ import annotations

import inspect
import math
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from urllib.parse import urlsplit

from ...contracts.ai_models import (
    AICapabilityEffect,
    AICapabilityRequest,
    AIExecutionMode,
    AIRetryPolicy,
    AIWorkPurpose,
)
from ...contracts.web import (
    ImageSearchResponse,
    ImageSearchResult,
    ProviderImageSearchOutput,
    ProviderReadOutput,
    ProviderSearchOutput,
    ReadRequest,
    ReadResponse,
    SearchRequest,
    SearchResponse,
    WebCallerKind,
    WebPageContent,
    WebReadStatus,
    WebResearchError,
    WebSearchDepth,
    WebSearchIntensity,
    WebSearchPurpose,
    WebSearchResult,
)
from ..profiles.ports import ProfilesRepositoryPort
from .content import (
    canonicalize_url,
    enum_value,
    sanitize_untrusted_web_content,
    validate_public_web_url,
)
from .content import (
    clean_text as _clean_text,
)
from .content import freshness as parse_freshness
from .content import (
    freshness_bonus as _freshness_bonus,
)
from .content import (
    image_resource_id as _image_resource_id,
)
from .content import (
    prefer_domain_diversity as _prefer_domain_diversity,
)
from .content import (
    resource_id as _resource_id,
)
from .content import (
    source_quality_bonus as _source_quality_bonus,
)
from .content import (
    text_relevance as _text_relevance,
)
from .content import (
    validate_scope as _validate_scope,
)
from .domain import WebSearchProviderRecord, WebSearchResultRecord
from .limits import IMAGE_RESULT_LIMITS, WEB_INTENSITY_LIMITS, WebIntensityLimits
from .ports import WebAIManagerPort, WebRepositoryPort


class WebCommandContext:
    """Prevent commands from widening profile, instance, run, or provider scope."""

    def __init__(
        self,
        service: WebResearchService,
        *,
        profile_id: str,
        instance_id: str,
        caller_id: str,
        core_run_id: str,
        intensity: WebSearchIntensity = WebSearchIntensity.STANDARD,
        image_inspector: Any | None = None,
        caller_kind: WebCallerKind = WebCallerKind.MAIN_CORE,
    ) -> None:
        self.service = service
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.caller_id = caller_id
        self.core_run_id = core_run_id
        self.intensity = enum_value(
            WebSearchIntensity,
            intensity,
            WebSearchIntensity.STANDARD,
        )
        self.caller_kind = enum_value(WebCallerKind, caller_kind, WebCallerKind.MAIN_CORE)
        self.search_count = 0
        self.resource_ids: set[str] = set()
        self.image_search_count = 0
        self.read_count = 0
        self.image_inspector = image_inspector

    async def search_web(
        self,
        query: str,
        purpose: str,
        depth: str = "auto",
        freshness: str = "auto",
    ) -> SearchResponse:
        limits = WEB_INTENSITY_LIMITS[self.intensity]
        if self.search_count >= limits.searches_per_run:
            raise WebResearchError("SEARCH_LIMIT", "本轮网页搜索次数已经达到上限")
        self.search_count += 1
        response = await self.service.search(
            SearchRequest(
                profile_id=self.profile_id,
                instance_id=self.instance_id,
                caller_kind=self.caller_kind,
                caller_id=self.caller_id,
                core_run_id=self.core_run_id,
                purpose=enum_value(WebSearchPurpose, purpose, WebSearchPurpose.ANSWER_USER),
                query=query,
                depth=enum_value(WebSearchDepth, depth, WebSearchDepth.AUTO),
                freshness=parse_freshness(freshness),
                intensity=self.intensity,
            )
        )
        self.resource_ids.update(
            str(item.resource_id) for item in response.results if str(item.resource_id)
        )
        return response

    async def read_web_content(
        self,
        resource_ids: Sequence[str],
        focus: str = "",
    ) -> ReadResponse:
        ids = tuple(str(item).strip() for item in resource_ids if str(item).strip())
        if len(ids) > 3:
            raise WebResearchError("READ_LIMIT", "一次最多读取三项网页资料")
        limits = WEB_INTENSITY_LIMITS[self.intensity]
        if self.read_count + len(ids) > limits.reads_per_run:
            raise WebResearchError("READ_LIMIT", "本轮网页读取次数已经达到上限")
        self.read_count += len(ids)
        return await self.service.read(
            ReadRequest(
                profile_id=self.profile_id,
                instance_id=self.instance_id,
                caller_kind=self.caller_kind,
                caller_id=self.caller_id,
                core_run_id=self.core_run_id,
                resource_ids=ids,
                focus=focus,
            )
        )

    async def read_link(self, link: str, focus: str = "") -> ReadResponse:
        """Register one explicit public URL, then read it in this run scope."""

        results = await self.service.register_current_message_urls(
            profile_id=self.profile_id,
            instance_id=self.instance_id,
            core_run_id=self.core_run_id,
            urls=(str(link or "").strip(),),
        )
        ids = tuple(str(item.resource_id) for item in results if str(item.resource_id))
        self.resource_ids.update(ids)
        if len(ids) != 1:
            raise WebResearchError("INVALID_RESOURCES", "网址没有形成可读取的网页资料")
        return await self.read_web_content(ids, focus=focus)

    async def search_images(
        self,
        query: str,
        purpose: str,
        depth: str = "auto",
        freshness: str = "auto",
    ) -> ImageSearchResponse:
        limits = WEB_INTENSITY_LIMITS[self.intensity]
        if self.image_search_count >= limits.searches_per_run:
            raise WebResearchError(
                "IMAGE_SEARCH_LIMIT",
                "本轮图片搜索次数已经达到上限",
            )
        self.image_search_count += 1
        return await self.service.search_images(
            SearchRequest(
                profile_id=self.profile_id,
                instance_id=self.instance_id,
                caller_kind=self.caller_kind,
                caller_id=self.caller_id,
                core_run_id=self.core_run_id,
                purpose=enum_value(WebSearchPurpose, purpose, WebSearchPurpose.ANSWER_USER),
                query=query,
                depth=enum_value(WebSearchDepth, depth, WebSearchDepth.AUTO),
                freshness=parse_freshness(freshness),
                intensity=self.intensity,
            )
        )

    async def inspect_search_images(
        self,
        image_resource_ids: Sequence[str],
        main_core_supports_vision: bool = False,
    ) -> Mapping[str, Any]:
        if not callable(self.image_inspector):
            raise WebResearchError(
                "IMAGE_INSPECTION_UNAVAILABLE",
                "图片查看暂不可用",
            )
        await self.service.require_gameplay_enabled(self.profile_id)
        resources = await self.service.resolve_image_resources(
            profile_id=self.profile_id,
            instance_id=self.instance_id,
            run_scope=self.core_run_id,
            image_resource_ids=image_resource_ids,
        )
        result = self.image_inspector(
            resources=resources,
            profile_id=self.profile_id,
            instance_id=self.instance_id,
            core_run_id=self.core_run_id,
            main_core_supports_vision=bool(main_core_supports_vision),
        )
        output = await result if inspect.isawaitable(result) else result
        await self.service.require_gameplay_enabled(self.profile_id)
        return output


@dataclass(frozen=True, slots=True)
class _Provider:
    provider_id: str
    backend_id: str
    provider_kind: str
    priority: int
    supports_read: bool
    enabled: bool = True


class WebResearchService:
    def __init__(
        self,
        ai_manager: WebAIManagerPort,
        repository: WebRepositoryPort,
        profiles: ProfilesRepositoryPort,
    ) -> None:
        self.ai_manager = ai_manager
        self.repository = repository
        self.profiles = profiles
        self.resources: dict[str, dict[str, Any]] = {}
        self.search_cache: dict[tuple[str, ...], tuple[float, SearchResponse]] = {}
        self.image_search_cache: dict[tuple[str, ...], tuple[float, ImageSearchResponse]] = {}
        self.image_resources: dict[str, dict[str, Any]] = {}
        self.page_cache: dict[str, tuple[float, WebPageContent]] = {}

    async def search(self, request: SearchRequest) -> SearchResponse:
        from .pipelines import search

        await self.require_gameplay_enabled(request.profile_id)
        return await search(self, request)

    async def search_images(self, request: SearchRequest) -> ImageSearchResponse:
        from .pipelines import search_images

        await self.require_gameplay_enabled(request.profile_id)
        return await search_images(self, request)

    async def read(self, request: ReadRequest) -> ReadResponse:
        from .pipelines import read

        await self.require_gameplay_enabled(request.profile_id)
        return await read(self, request)

    async def require_gameplay_enabled(self, profile_id: str) -> None:
        """Re-read the profile parent switch at every live gameplay boundary."""

        profile = await self.profiles.get_profile(str(profile_id))
        if profile is None or not bool(profile.web_search_enabled):
            raise WebResearchError("WEB_RESEARCH_DISABLED", "当前角色已关闭联网查询")

    async def has_read_provider(self, profile_id: str) -> bool:
        """Return whether this profile currently has a usable reader.

        Search-only providers still provide trustworthy titles and snippets.
        Callers use this capability check to make full-page enrichment optional
        instead of turning a missing reader into a failed search workflow.
        """

        try:
            await self.require_gameplay_enabled(profile_id)
        except WebResearchError as exc:
            if exc.code == "WEB_RESEARCH_DISABLED":
                return False
            raise
        return any(
            provider.enabled and provider.supports_read
            for provider in await self.providers(str(profile_id))
        )

    async def has_search_provider(self, profile_id: str) -> bool:
        """Return whether at least one enabled search provider is routable."""

        try:
            await self.require_gameplay_enabled(profile_id)
        except WebResearchError as exc:
            if exc.code == "WEB_RESEARCH_DISABLED":
                return False
            raise
        return any(provider.enabled for provider in await self.providers(str(profile_id)))

    async def has_image_search_provider(self, profile_id: str) -> bool:
        """Return whether one enabled provider implements image search."""

        from .pipelines import IMAGE_SEARCH_PROVIDER_KINDS

        try:
            await self.require_gameplay_enabled(profile_id)
        except WebResearchError as exc:
            if exc.code == "WEB_RESEARCH_DISABLED":
                return False
            raise
        return any(
            provider.enabled and provider.provider_kind in IMAGE_SEARCH_PROVIDER_KINDS
            for provider in await self.providers(str(profile_id))
        )

    async def register_current_message_urls(
        self,
        *,
        profile_id: str,
        instance_id: str,
        core_run_id: str,
        urls: Sequence[str],
    ) -> tuple[WebSearchResult, ...]:
        """Register explicit player URLs as read-only resources for this run."""

        _validate_scope(profile_id, instance_id, core_run_id)
        results: list[WebSearchResult] = []
        for raw_url in tuple(urls)[:10]:
            url = validate_public_web_url(raw_url)
            resource_id = _resource_id(core_run_id, url)
            result = WebSearchResult(
                resource_id=resource_id,
                title=url,
                canonical_url=url,
                domain=urlsplit(url).hostname or "",
                provider="player_message",
                source_providers=("player_message",),
            )
            self.remember(result, profile_id, instance_id, core_run_id, "")
            results.append(result)
        return tuple(results)

    async def providers(self, profile_id: str) -> list[_Provider]:
        rows: Sequence[WebSearchProviderRecord] = (
            await self.repository.list_web_search_providers(profile_id) or ()
        )
        output: list[_Provider] = []
        for row in rows:
            if not row.enabled:
                continue
            provider_kind = str(row.provider_kind).strip().lower()
            provider_id = str(row.provider_id)
            backend_id = str(row.backend_id)
            if not provider_id or not backend_id:
                continue
            output.append(
                _Provider(
                    provider_id,
                    backend_id,
                    provider_kind,
                    int(row.priority),
                    bool(row.read_enabled),
                    True,
                )
            )
        return sorted(output, key=lambda item: (item.priority, item.provider_id))

    async def search_provider(
        self,
        provider: _Provider,
        request: SearchRequest,
        query: str,
        depth: WebSearchDepth,
        max_results: int,
        timeout: float,
    ) -> ProviderSearchOutput:
        await self.require_gameplay_enabled(request.profile_id)
        invocation = AICapabilityRequest(
            invocation_id=f"{request.run_scope}:web.search:{provider.provider_id}:{secrets.token_hex(4)}",
            capability="web.search",
            work_purpose=AIWorkPurpose.WEB_SEARCH,
            logical_stage_key=f"{request.run_scope}:web.search:{provider.provider_id}:{sha256(query.encode()).hexdigest()[:16]}",
            payload={
                "query": query,
                "depth": depth.value,
                "freshness": request.freshness.value,
                "max_results": max_results,
            },
            backend_ids=(provider.backend_id,),
            effect=AICapabilityEffect.READ_ONLY,
            execution_mode=AIExecutionMode.FOREGROUND_SYNC,
            profile_id=request.profile_id,
            instance_id=request.instance_id,
            owner_kind="WEB_SEARCH",
            owner_id=request.caller_id,
            idempotency_key=f"{request.run_scope}:search:{provider.provider_id}:{sha256(query.encode()).hexdigest()[:16]}",
            retry_policy=AIRetryPolicy(
                max_attempts=3,
                backend_timeout_seconds=min(
                    float(request.operation_timeout_seconds), max(1.0, timeout)
                ),
            ),
            metadata={
                "core_run_id": request.core_run_id,
                "ai_task_id": request.ai_task_id,
                "provider_kind": provider.provider_kind,
            },
        )
        result = await self.ai_manager.invoke_capability(invocation)
        await self.require_gameplay_enabled(request.profile_id)
        output = result.output
        if not isinstance(output, ProviderSearchOutput):
            raise WebResearchError(
                "OUTPUT_CONTRACT", "Web provider returned an invalid search result"
            )
        return output

    async def read_provider(
        self,
        provider: _Provider,
        request: ReadRequest,
        url: str,
        timeout: float,
    ) -> ProviderReadOutput:
        await self.require_gameplay_enabled(request.profile_id)
        invocation = AICapabilityRequest(
            invocation_id=f"{request.run_scope}:web.read:{provider.provider_id}:{secrets.token_hex(4)}",
            capability="web.read",
            work_purpose=AIWorkPurpose.WEB_READ,
            logical_stage_key=f"{request.run_scope}:web.read:{provider.provider_id}:{sha256(url.encode()).hexdigest()[:16]}",
            payload={"url": url, "focus": request.focus, "max_characters": 30000},
            backend_ids=(provider.backend_id,),
            effect=AICapabilityEffect.READ_ONLY,
            execution_mode=AIExecutionMode.FOREGROUND_SYNC,
            profile_id=request.profile_id,
            instance_id=request.instance_id,
            owner_kind="WEB_READ",
            owner_id=request.caller_id,
            idempotency_key=f"{request.run_scope}:read:{provider.provider_id}:{sha256(url.encode()).hexdigest()[:16]}",
            retry_policy=AIRetryPolicy(
                max_attempts=3,
                backend_timeout_seconds=min(
                    float(request.operation_timeout_seconds), max(1.0, timeout)
                ),
            ),
            metadata={
                "core_run_id": request.core_run_id,
                "ai_task_id": request.ai_task_id,
                "provider_kind": provider.provider_kind,
            },
        )
        result = await self.ai_manager.invoke_capability(invocation)
        await self.require_gameplay_enabled(request.profile_id)
        output = result.output
        if not isinstance(output, ProviderReadOutput):
            raise WebResearchError("OUTPUT_CONTRACT", "Web provider returned invalid page content")
        return output

    async def image_search_provider(
        self,
        provider: _Provider,
        request: SearchRequest,
        query: str,
        max_results: int,
        timeout: float,
    ) -> ProviderImageSearchOutput:
        await self.require_gameplay_enabled(request.profile_id)
        invocation = AICapabilityRequest(
            invocation_id=f"{request.run_scope}:web.image_search:{provider.provider_id}:{secrets.token_hex(4)}",
            capability="web.image_search",
            work_purpose=AIWorkPurpose.WEB_IMAGE_SEARCH,
            logical_stage_key=f"{request.run_scope}:web.image_search:{provider.provider_id}:{sha256(query.encode()).hexdigest()[:16]}",
            payload={
                "query": query,
                "freshness": request.freshness.value,
                "max_results": max_results,
            },
            backend_ids=(provider.backend_id,),
            effect=AICapabilityEffect.READ_ONLY,
            execution_mode=AIExecutionMode.FOREGROUND_SYNC,
            profile_id=request.profile_id,
            instance_id=request.instance_id,
            owner_kind="WEB_IMAGE_SEARCH",
            owner_id=request.caller_id,
            idempotency_key=f"{request.run_scope}:image-search:{provider.provider_id}:{sha256(query.encode()).hexdigest()[:16]}",
            retry_policy=AIRetryPolicy(
                max_attempts=3,
                backend_timeout_seconds=min(
                    float(request.operation_timeout_seconds), max(1.0, timeout)
                ),
            ),
            metadata={
                "core_run_id": request.core_run_id,
                "ai_task_id": request.ai_task_id,
                "provider_kind": provider.provider_kind,
            },
        )
        result = await self.ai_manager.invoke_capability(invocation)
        await self.require_gameplay_enabled(request.profile_id)
        output = result.output
        if not isinstance(output, ProviderImageSearchOutput):
            raise WebResearchError("OUTPUT_CONTRACT", "Image provider returned an invalid result")
        return output

    def merge_image_results(
        self,
        outputs: Sequence[tuple[_Provider, ProviderImageSearchOutput]],
        *,
        request: SearchRequest,
        session_id: str,
        limit: int,
    ) -> list[ImageSearchResult]:
        merged: dict[str, dict[str, Any]] = {}
        for provider, output in outputs:
            for item in output.items:
                try:
                    image_url = validate_public_web_url(item.image_url)
                    thumbnail_url = validate_public_web_url(item.thumbnail_url or item.image_url)
                    source_url = validate_public_web_url(item.source_url) if item.source_url else ""
                except WebResearchError:
                    continue
                key = canonicalize_url(image_url)
                current = merged.get(key)
                if current is None:
                    current = {
                        "image_url": image_url,
                        "thumbnail_url": thumbnail_url,
                        "source_url": source_url,
                        "title": _clean_text(item.title),
                        "description": _clean_text(item.description),
                        "width": max(0, item.width),
                        "height": max(0, item.height),
                        "provider": provider.provider_id,
                        "rank": max(1, item.provider_rank),
                        "providers": [provider.provider_id],
                    }
                    merged[key] = current
                else:
                    if provider.provider_id not in current["providers"]:
                        current["providers"].append(provider.provider_id)
                    if len(item.description) > len(current["description"]):
                        current["description"] = _clean_text(item.description)
                    if not current["source_url"] and source_url:
                        current["source_url"] = source_url
        ranked = sorted(
            merged.values(), key=lambda row: (-len(row["providers"]), row["rank"], row["image_url"])
        )
        output: list[ImageSearchResult] = []
        for row in ranked[:limit]:
            resource_id = _image_resource_id(request.run_scope, row["image_url"])
            result = ImageSearchResult(
                image_resource_id=resource_id,
                image_url=row["image_url"],
                thumbnail_url=row["thumbnail_url"],
                source_url=row["source_url"],
                title=row["title"],
                description=row["description"],
                width=row["width"],
                height=row["height"],
                provider=row["provider"],
                provider_rank=row["rank"],
                source_providers=tuple(row["providers"]),
            )
            self.image_resources[resource_id] = {
                **asdict(result),
                "profile_id": request.profile_id,
                "instance_id": request.instance_id,
                "run_scope": request.run_scope,
                "session_id": session_id,
            }
            output.append(result)
        return output

    async def resolve_image_resources(
        self,
        *,
        profile_id: str,
        instance_id: str,
        run_scope: str,
        image_resource_ids: Sequence[str],
    ) -> tuple[ImageSearchResult, ...]:
        """Resolve image candidates for inspection under their exact Run scope."""
        _validate_scope(profile_id, instance_id, run_scope)
        ids = tuple(
            dict.fromkeys(str(item).strip() for item in image_resource_ids if str(item).strip())
        )
        if not ids or len(ids) > 3:
            raise WebResearchError(
                "IMAGE_RESOURCE_LIMIT", "One to three image resources are required"
            )
        output: list[ImageSearchResult] = []
        for resource_id in ids:
            row = self.image_resources.get(resource_id)
            if row is None:
                row = await self.repository.get_web_image_result(
                    resource_id, profile_id, instance_id, run_scope
                )
            if not isinstance(row, Mapping):
                raise WebResearchError(
                    "IMAGE_RESOURCE_NOT_FOUND", "The image resource is unavailable"
                )
            if not _resource_in_scope(row, profile_id, instance_id, run_scope):
                raise WebResearchError(
                    "IMAGE_RESOURCE_SCOPE", "The image resource belongs to another run"
                )
            output.append(_image_result_from_row(resource_id, row))
        return tuple(output)

    def merge_results(
        self,
        outputs: Sequence[tuple[_Provider, ProviderSearchOutput]],
        *,
        request: SearchRequest,
        session_id: str,
        limit: int,
        token_budget: int,
    ) -> list[WebSearchResult]:
        merged: dict[str, dict[str, Any]] = {}
        for provider, output in outputs:
            for item in output.items:
                try:
                    url = validate_public_web_url(item.url)
                except WebResearchError:
                    continue
                key = canonicalize_url(url)
                current = merged.get(key)
                if current is None:
                    current = {
                        "title": item.title.strip() or key,
                        "url": key,
                        "snippet": _clean_text(item.snippet),
                        "published_at": item.published_at,
                        "provider": provider.provider_id,
                        "rank": max(1, item.provider_rank),
                        "providers": [provider.provider_id],
                        "rank_score": 1.0 / (60.0 + max(1, item.provider_rank)),
                    }
                    merged[key] = current
                else:
                    if provider.provider_id not in current["providers"]:
                        current["providers"].append(provider.provider_id)
                        current["rank_score"] += 1.0 / (60.0 + max(1, item.provider_rank))
                    if len(item.snippet) > len(current["snippet"]):
                        current["snippet"] = _clean_text(item.snippet)
                    current["rank"] = min(current["rank"], max(1, item.provider_rank))
        ranked = []
        for value in merged.values():
            cross = len(value["providers"])
            score = (
                value["rank_score"]
                + 0.15 * max(0, cross - 1)
                + _text_relevance(request.query, value["title"], value["snippet"])
                + _freshness_bonus(value["published_at"])
                + _source_quality_bonus(value["url"])
            )
            ranked.append((score, value))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["rank"], pair[1]["url"]))
        ranked = _prefer_domain_diversity(ranked)
        output: list[WebSearchResult] = []
        used_tokens = 0
        for score, value in ranked[:limit]:
            remaining_chars = max(0, (token_budget - used_tokens) * 3)
            if remaining_chars < 80:
                break
            snippet = value["snippet"][: min(1600, remaining_chars)]
            estimated = max(
                1, math.ceil((len(value["title"]) + len(snippet) + len(value["url"])) / 3)
            )
            if used_tokens + estimated > token_budget:
                break
            used_tokens += estimated
            resource_id = _resource_id(request.run_scope, value["url"])
            result = WebSearchResult(
                resource_id=resource_id,
                title=value["title"],
                canonical_url=value["url"],
                domain=urlsplit(value["url"]).hostname or "",
                snippet=snippet,
                published_at=value["published_at"],
                provider=value["provider"],
                provider_rank=value["rank"],
                cross_source_count=len(value["providers"]),
                source_providers=tuple(value["providers"]),
                score=score,
            )
            self.remember(
                result, request.profile_id, request.instance_id, request.run_scope, session_id
            )
            output.append(result)
        return output

    def remember(
        self,
        result: WebSearchResult,
        profile_id: str,
        instance_id: str,
        run_scope: str,
        session_id: str,
    ) -> None:
        self.resources[result.resource_id] = {
            **asdict(result),
            "profile_id": profile_id,
            "instance_id": instance_id,
            "run_scope": run_scope,
            "session_id": session_id,
        }

    async def scoped_resource(self, resource_id: str, request: ReadRequest) -> Mapping[str, Any]:
        row = self.resources.get(resource_id)
        if row is None:
            row = await self.repository.get_web_search_result(
                resource_id,
                request.profile_id,
                request.instance_id,
                request.run_scope,
            )
        if not isinstance(row, Mapping):
            raise WebResearchError("RESOURCE_NOT_FOUND", "The web resource is unavailable")
        if any(
            str(row.get(key) or "") != expected
            for key, expected in (
                ("profile_id", request.profile_id),
                ("instance_id", request.instance_id),
                ("run_scope", request.run_scope),
            )
        ):
            raise WebResearchError("RESOURCE_SCOPE", "The web resource belongs to another run")
        return row

    async def persistent_cached_page(
        self, resource_id: str, request: ReadRequest
    ) -> WebPageContent | None:
        row = await self.repository.get_web_page_snapshot(
            request.profile_id, request.instance_id, resource_id
        )
        if row is None:
            return None
        content = str(row.content or "")
        if not content or row.status is not WebReadStatus.READ:
            return None
        resource = self.resources.get(resource_id, {})
        metadata = row.metadata
        return WebPageContent(
            resource_id=resource_id,
            canonical_url=str(resource.get("canonical_url") or metadata.get("canonical_url") or ""),
            title=str(resource.get("title") or metadata.get("title") or ""),
            content=content,
            provider=str(metadata.get("provider_id") or "cache"),
            retrieved_at=row.retrieved_at or datetime.now(UTC),
            truncated=bool(metadata.get("truncated", False)),
        )

    @staticmethod
    def result_record(
        result: WebSearchResult,
        request: SearchRequest,
        *,
        session_id: str,
    ) -> WebSearchResultRecord:
        published_at: datetime | None = None
        if result.published_at:
            try:
                published_at = datetime.fromisoformat(
                    str(result.published_at).replace("Z", "+00:00")
                )
            except ValueError:
                published_at = None
        return WebSearchResultRecord(
            resource_id=result.resource_id,
            session_id=session_id,
            profile_id=request.profile_id,
            instance_id=request.instance_id,
            title=result.title,
            canonical_url=result.canonical_url,
            domain=result.domain,
            snippet=result.snippet,
            published_at=published_at,
            retrieved_at=result.retrieved_at,
            provider_id=result.provider,
            provider_rank=result.provider_rank,
            cross_source_count=max(1, result.cross_source_count),
            read_status=result.read_status,
            metadata={
                "run_scope": request.run_scope,
                "source_providers": list(result.source_providers),
                "score": result.score,
            },
        )


def _resource_in_scope(
    row: Mapping[str, Any],
    profile_id: str,
    instance_id: str,
    run_scope: str,
) -> bool:
    return all(
        str(row.get(key) or "") == expected
        for key, expected in (
            ("profile_id", profile_id),
            ("instance_id", instance_id),
            ("run_scope", run_scope),
        )
    )


def _image_result_from_row(
    resource_id: str,
    row: Mapping[str, Any],
) -> ImageSearchResult:
    return ImageSearchResult(
        image_resource_id=resource_id,
        image_url=str(row.get("image_url") or row.get("original_url") or ""),
        thumbnail_url=str(row.get("thumbnail_url") or ""),
        source_url=str(row.get("source_url") or row.get("source_page_url") or ""),
        title=str(row.get("title") or ""),
        description=str(row.get("description") or ""),
        width=int(row.get("width") or 0),
        height=int(row.get("height") or 0),
        provider=str(row.get("provider") or row.get("provider_id") or ""),
        provider_rank=int(row.get("provider_rank") or 0),
        source_providers=tuple(row.get("source_providers") or ()),
    )


__all__ = [
    "IMAGE_RESULT_LIMITS",
    "WEB_INTENSITY_LIMITS",
    "WebIntensityLimits",
    "WebResearchService",
    "WebCommandContext",
    "canonicalize_url",
    "sanitize_untrusted_web_content",
    "validate_public_web_url",
]
