"""Search, image-search, and page-read orchestration pipelines."""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from ...contracts.web import (
    ImageSearchResponse,
    ProviderReadOutput,
    ReadRequest,
    ReadResponse,
    SearchRequest,
    SearchResponse,
    WebCallerKind,
    WebPageContent,
    WebReadStatus,
    WebResearchError,
)
from .content import (
    resolved_depth,
    safe_error_code,
    sanitize_untrusted_web_content,
    validate_scope,
    validated_query,
)
from .domain import WebImageSearchResultRecord, WebPageSnapshotRecord
from .limits import IMAGE_RESULT_LIMITS, WEB_INTENSITY_LIMITS

if TYPE_CHECKING:
    from .research import WebResearchService


IMAGE_SEARCH_PROVIDER_KINDS = frozenset(
    {"tavily", "bocha", "brave", "firecrawl", "baidu_ai_search"}
)


async def search(service: WebResearchService, request: SearchRequest) -> SearchResponse:
    _validate_search_caller(request, image=False)
    query = validated_query(request.query)
    depth = resolved_depth(request.depth, request.purpose)
    limits = WEB_INTENSITY_LIMITS[request.intensity]
    cache_key = (
        request.profile_id,
        request.instance_id,
        request.run_scope,
        query,
        depth.value,
        request.freshness.value,
        request.intensity.value,
    )
    cached = service.search_cache.get(cache_key)
    if cached is not None and cached[0] > time.monotonic():
        return cached[1]
    providers = await service.providers(request.profile_id)
    if not providers:
        raise WebResearchError("NO_PROVIDER", "No web-search interface is enabled")
    started, deadline = _timing(request)
    session_id = "ws_" + secrets.token_hex(12)
    await _create_session(service, request, session_id, query, depth.value, deadline)
    try:
        outputs, errors = await _collect(
            providers,
            limits.provider_limits[depth],
            deadline,
            lambda provider, remaining: service.search_provider(
                provider,
                request,
                query,
                depth,
                limits.result_limit,
                remaining,
            ),
        )
    except WebResearchError as exc:
        await _cancel_disabled_session(service, session_id, exc, started)
        raise
    await _require_outputs(
        service,
        session_id,
        outputs,
        errors,
        started,
        timeout_code="TIMEOUT",
        failed_code="ALL_PROVIDERS_FAILED",
        message="All web-search interfaces failed",
        timeout_message="Web search exceeded its shared deadline",
    )
    results = service.merge_results(
        outputs,
        request=request,
        session_id=session_id,
        limit=limits.result_limit,
        token_budget=limits.data_budget_tokens,
    )
    await _require_enabled_for_session(service, request, session_id, started)
    await service.repository.save_web_search_results(
        request.profile_id,
        request.instance_id,
        session_id,
        [service.result_record(item, request, session_id=session_id) for item in results],
    )
    elapsed = await _complete(service, session_id, errors, started, len(results))
    response = SearchResponse(
        session_id,
        query,
        request.purpose,
        depth,
        tuple(results),
        "Some web-search interfaces failed; remaining results are shown." if errors else "",
        errors,
        elapsed,
    )
    service.search_cache[cache_key] = (time.monotonic() + 24 * 3600, response)
    return response


async def search_images(
    service: WebResearchService,
    request: SearchRequest,
) -> ImageSearchResponse:
    _validate_search_caller(request, image=True)
    query = validated_query(request.query)
    depth = resolved_depth(request.depth, request.purpose)
    limits = WEB_INTENSITY_LIMITS[request.intensity]
    result_limit = IMAGE_RESULT_LIMITS[request.intensity]
    cache_key = (
        request.profile_id,
        request.instance_id,
        request.run_scope,
        "IMAGE",
        query,
        depth.value,
        request.freshness.value,
        request.intensity.value,
    )
    cached = service.image_search_cache.get(cache_key)
    if cached is not None and cached[0] > time.monotonic():
        return cached[1]
    providers = [
        provider
        for provider in await service.providers(request.profile_id)
        if provider.provider_kind in IMAGE_SEARCH_PROVIDER_KINDS
    ]
    if not providers:
        raise WebResearchError("NO_IMAGE_PROVIDER", "No image-search interface is enabled")
    started, deadline = _timing(request)
    session_id = "wi_" + secrets.token_hex(12)
    await _create_session(
        service,
        request,
        session_id,
        query,
        depth.value,
        deadline,
        image=True,
    )
    try:
        outputs, errors = await _collect(
            providers,
            limits.provider_limits[depth],
            deadline,
            lambda provider, remaining: service.image_search_provider(
                provider,
                request,
                query,
                result_limit,
                remaining,
            ),
        )
    except WebResearchError as exc:
        await _cancel_disabled_session(service, session_id, exc, started)
        raise
    await _require_outputs(
        service,
        session_id,
        outputs,
        errors,
        started,
        timeout_code="TIMEOUT",
        failed_code="ALL_IMAGE_PROVIDERS_FAILED",
        message="All image-search interfaces failed",
    )
    results = service.merge_image_results(
        outputs,
        request=request,
        session_id=session_id,
        limit=result_limit,
    )
    await _require_enabled_for_session(service, request, session_id, started)
    await service.repository.save_web_image_results(
        request.profile_id,
        request.instance_id,
        session_id,
        [_image_record(item, request, session_id=session_id) for item in results],
    )
    elapsed = await _complete(service, session_id, errors, started, len(results))
    response = ImageSearchResponse(
        session_id,
        query,
        request.purpose,
        depth,
        tuple(results),
        "Some image-search interfaces failed; remaining results are shown." if errors else "",
        errors,
        elapsed,
    )
    service.image_search_cache[cache_key] = (time.monotonic() + 24 * 3600, response)
    return response


def _validate_search_caller(request: SearchRequest, *, image: bool) -> None:
    validate_scope(request.profile_id, request.instance_id, request.run_scope)
    allowed = (
        {WebCallerKind.MAIN_CORE, WebCallerKind.STICKER_COLLECTOR}
        if image
        else {
            WebCallerKind.MAIN_CORE,
            WebCallerKind.BACKGROUND_AUTHOR,
            WebCallerKind.STICKER_COLLECTOR,
        }
    )
    if request.caller_kind not in allowed:
        kind = "image-search" if image else "web"
        raise WebResearchError("CALLER_NOT_ENABLED", f"This {kind} caller is not enabled yet")


def _timing(request: Any) -> tuple[float, float]:
    started = time.monotonic()
    return started, started + max(1.0, float(request.operation_timeout_seconds))


async def _create_session(
    service: WebResearchService,
    request: SearchRequest,
    session_id: str,
    query: str,
    depth: str,
    deadline: float,
    *,
    image: bool = False,
) -> None:
    record = {
        "session_id": session_id,
        "profile_id": request.profile_id,
        "instance_id": request.instance_id,
        "caller_kind": request.caller_kind.value,
        "caller_id": request.caller_id,
        "core_run_id": request.core_run_id,
        "ai_task_id": request.ai_task_id,
        "purpose": request.purpose.value,
        "query": query,
        "depth": depth,
        "freshness": request.freshness.value,
        "status": "RUNNING",
        "deadline_at": datetime.fromtimestamp(
            time.time() + max(0.0, deadline - time.monotonic()),
            UTC,
        ),
    }
    if image:
        record["search_kind"] = "IMAGE"
    await service.repository.create_web_search_session(record)


async def _collect(
    providers: list[Any],
    wanted: int,
    deadline: float,
    invoke: Callable[[Any, float], Awaitable[Any]],
) -> tuple[list[tuple[Any, Any]], dict[str, str]]:
    outputs: list[tuple[Any, Any]] = []
    errors: dict[str, str] = {}
    cursor = 0
    while len(outputs) < wanted and cursor < len(providers):
        batch = providers[cursor : cursor + max(1, wanted - len(outputs))]
        cursor += len(batch)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            errors["deadline"] = "TIMEOUT"
            break
        values = await _invoke_batch(batch, remaining, invoke)
        for provider, value in zip(batch, values, strict=True):
            if isinstance(value, WebResearchError) and value.code == "WEB_RESEARCH_DISABLED":
                raise value
            if isinstance(value, BaseException):
                errors[provider.provider_id] = safe_error_code(value)
            elif value.items:
                outputs.append((provider, value))
            else:
                errors[provider.provider_id] = "EMPTY_OUTPUT"
    return outputs, errors


async def _invoke_batch(
    batch: list[Any],
    remaining: float,
    invoke: Callable[[Any, float], Awaitable[Any]],
) -> list[Any]:
    tasks = [invoke(provider, remaining) for provider in batch]
    try:
        async with asyncio.timeout(remaining):
            return list(await asyncio.gather(*tasks, return_exceptions=True))
    except TimeoutError:
        return [TimeoutError()] * len(batch)


async def _require_outputs(
    service: WebResearchService,
    session_id: str,
    outputs: list[Any],
    errors: dict[str, str],
    started: float,
    *,
    timeout_code: str,
    failed_code: str,
    message: str,
    timeout_message: str | None = None,
) -> None:
    if outputs:
        return
    await service.repository.complete_web_search_session(
        session_id,
        {
            "status": "FAILED",
            "provider_errors": errors,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    timed_out = errors.get("deadline") == "TIMEOUT"
    code = timeout_code if timed_out else failed_code
    safe_message = timeout_message if timed_out and timeout_message else message
    raise WebResearchError(code, safe_message)


async def _require_enabled_for_session(
    service: WebResearchService,
    request: SearchRequest,
    session_id: str,
    started: float,
) -> None:
    try:
        await service.require_gameplay_enabled(request.profile_id)
    except WebResearchError as exc:
        await _cancel_disabled_session(service, session_id, exc, started)
        raise


async def _cancel_disabled_session(
    service: WebResearchService,
    session_id: str,
    error: WebResearchError,
    started: float,
) -> None:
    if error.code != "WEB_RESEARCH_DISABLED":
        return
    await service.repository.complete_web_search_session(
        session_id,
        {
            "status": "CANCELLED",
            "error": error.code,
            "elapsed_seconds": time.monotonic() - started,
        },
    )


async def _complete(
    service: WebResearchService,
    session_id: str,
    errors: dict[str, str],
    started: float,
    result_count: int,
) -> float:
    elapsed = time.monotonic() - started
    await service.repository.complete_web_search_session(
        session_id,
        {
            "status": "PARTIAL" if errors else "SUCCEEDED",
            "provider_errors": errors,
            "result_count": result_count,
            "elapsed_seconds": elapsed,
        },
    )
    return elapsed


def _image_record(
    item: Any, request: SearchRequest, *, session_id: str
) -> WebImageSearchResultRecord:
    return WebImageSearchResultRecord(
        image_resource_id=item.image_resource_id,
        session_id=session_id,
        profile_id=request.profile_id,
        instance_id=request.instance_id,
        title=item.title,
        description=item.description,
        original_url=item.image_url,
        thumbnail_url=item.thumbnail_url,
        source_page_url=item.source_url,
        source_domain=urlsplit(item.source_url).hostname or "" if item.source_url else "",
        width=item.width,
        height=item.height,
        provider_id=item.provider,
        provider_rank=item.provider_rank,
        cross_source_count=max(1, len(item.source_providers)),
        metadata={
            "run_scope": request.run_scope,
            "source_providers": list(item.source_providers),
            "score": 0.0,
        },
        retrieved_at=item.retrieved_at,
    )


async def read(service: WebResearchService, request: ReadRequest) -> ReadResponse:
    validate_scope(request.profile_id, request.instance_id, request.run_scope)
    if request.caller_kind not in {
        WebCallerKind.MAIN_CORE,
        WebCallerKind.BACKGROUND_AUTHOR,
        WebCallerKind.STICKER_COLLECTOR,
    }:
        raise WebResearchError("CALLER_NOT_ENABLED", "This web caller is not enabled yet")
    if not request.resource_ids or len(request.resource_ids) > 3:
        raise WebResearchError("INVALID_RESOURCES", "One to three resource IDs are required")
    started, deadline = _timing(request)
    readable = [
        provider
        for provider in await service.providers(request.profile_id)
        if provider.supports_read
    ]
    if not readable:
        raise WebResearchError("NO_READER", "No webpage-reading interface is enabled")
    pages: list[WebPageContent] = []
    errors: dict[str, str] = {}
    for resource_id in request.resource_ids:
        try:
            pages.append(
                await _read_one(
                    service,
                    request,
                    str(resource_id),
                    readable,
                    deadline,
                )
            )
        except asyncio.CancelledError:
            raise
        except WebResearchError as exc:
            if exc.code == "WEB_RESEARCH_DISABLED":
                raise
            errors[str(resource_id)] = safe_error_code(exc)
        except BaseException as exc:
            errors[str(resource_id)] = safe_error_code(exc)
    if not pages:
        raise WebResearchError("READ_FAILED", "None of the requested webpages could be read")
    await service.require_gameplay_enabled(request.profile_id)
    return ReadResponse(
        tuple(pages),
        errors,
        "Some webpages could not be read." if errors else "",
        time.monotonic() - started,
    )


async def _read_one(
    service: WebResearchService,
    request: ReadRequest,
    resource_id: str,
    readable: list[Any],
    deadline: float,
) -> WebPageContent:
    resource = await service.scoped_resource(resource_id, request)
    cached = service.page_cache.get(resource_id)
    if cached is not None and cached[0] > time.monotonic():
        return cached[1]
    persistent = await service.persistent_cached_page(resource_id, request)
    if persistent is not None:
        service.page_cache[resource_id] = (time.monotonic() + 6 * 3600, persistent)
        return persistent
    output = await _read_from_provider(service, request, resource, readable, deadline)
    content, truncated = sanitize_untrusted_web_content(output.content, request.focus)
    if not content:
        raise WebResearchError("EMPTY_OUTPUT", "The webpage contains no usable text")
    page = WebPageContent(
        resource_id=resource_id,
        canonical_url=resource["canonical_url"],
        title=output.title or str(resource.get("title") or ""),
        content=content,
        provider=output.provider_id,
        truncated=truncated,
    )
    await _persist_page(service, request, resource, page)
    return page


async def _read_from_provider(
    service: WebResearchService,
    request: ReadRequest,
    resource: dict[str, Any],
    readable: list[Any],
    deadline: float,
) -> ProviderReadOutput:
    preferred = str(resource.get("provider") or "")
    choices = sorted(
        readable,
        key=lambda item: (0 if item.provider_id == preferred else 1, item.priority),
    )
    last_error: BaseException | None = None
    for provider in choices:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            output = await service.read_provider(
                provider,
                request,
                resource["canonical_url"],
                remaining,
            )
            if output.content.strip():
                return output
        except asyncio.CancelledError:
            raise
        except WebResearchError as exc:
            if exc.code == "WEB_RESEARCH_DISABLED":
                raise
            last_error = exc
        except BaseException as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise WebResearchError("EMPTY_OUTPUT", "No webpage content was returned")


async def _persist_page(
    service: WebResearchService,
    request: ReadRequest,
    resource: dict[str, Any],
    page: WebPageContent,
) -> None:
    await service.require_gameplay_enabled(request.profile_id)
    service.page_cache[page.resource_id] = (time.monotonic() + 6 * 3600, page)
    service.resources[page.resource_id]["read_status"] = WebReadStatus.READ.value
    await service.repository.upsert_web_page_snapshot(
        WebPageSnapshotRecord(
            snapshot_id=0,
            resource_id=page.resource_id,
            profile_id=request.profile_id,
            instance_id=request.instance_id,
            content=page.content,
            content_hash=sha256(page.content.encode()).hexdigest(),
            token_estimate=max(0, (len(page.content) + 2) // 3),
            status=WebReadStatus.READ,
            metadata={
                "canonical_url": resource["canonical_url"],
                "title": page.title,
                "provider_id": page.provider,
                "truncated": page.truncated,
                "run_scope": request.run_scope,
            },
        )
    )


__all__ = ["read", "search", "search_images"]
