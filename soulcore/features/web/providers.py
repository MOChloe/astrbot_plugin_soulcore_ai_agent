"""Whitelisted web-search provider adapters.

Only administrator-configured official endpoints are used.  The model never
supplies an endpoint, credential, profile, or conversation scope.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from ...contracts.ai_models import (
    AIBackendDescriptor,
    AICapabilityRequest,
    AIErrorCode,
    AIErrorInfo,
    AIInvocationError,
)
from ...contracts.web import (
    ProviderImageSearchOutput,
    ProviderReadOutput,
    ProviderSearchOutput,
)
from .normalization import (
    bearer_headers as _bearer_headers,
)
from .normalization import bounded_int
from .normalization import (
    image_items as _image_items,
)
from .normalization import (
    search_items as _items,
)
from .transport import (
    AiohttpWebJSONTransport,
    WebHTTPResponse,
    WebHTTPStatusError,
    WebJSONTransport,
    WebTransportError,
)

CredentialResolver = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class WebSearchAdapterConfig:
    credential_id: str


class BaseWebAdapter:
    provider_kind = ""
    capabilities = ("web.search",)
    image_features = None

    def __init__(
        self,
        config: WebSearchAdapterConfig,
        credential_resolver: CredentialResolver,
        transport: WebJSONTransport | None = None,
    ) -> None:
        self.config = config
        self.credential_resolver = credential_resolver
        self.transport = transport or AiohttpWebJSONTransport()

    @property
    def adapter_id(self) -> str:
        return f"web_{self.provider_kind}"

    def _credential(self, backend: AIBackendDescriptor) -> str:
        credential_id = backend.credential_id or self.config.credential_id
        try:
            value = str(self.credential_resolver(credential_id) or "").strip()
        except Exception as exc:
            raise AIInvocationError(
                AIErrorInfo(
                    AIErrorCode.AUTHENTICATION,
                    "The web-search credential is unavailable",
                    switch_backend=True,
                    open_circuit=True,
                    backend_id=backend.backend_id,
                    phase="prepare",
                ),
                cause=exc,
            ) from None
        if not value:
            raise AIInvocationError(
                AIErrorInfo(
                    AIErrorCode.AUTHENTICATION,
                    "The web-search credential is empty",
                    switch_backend=True,
                    open_circuit=True,
                    backend_id=backend.backend_id,
                    phase="prepare",
                )
            )
        return value

    async def invoke(self, request: AICapabilityRequest, backend: AIBackendDescriptor) -> Any:
        timeout = request.retry_policy.normalized().backend_timeout_seconds
        if request.capability == "web.search":
            return await self._invoke_search(request, backend, timeout)
        if request.capability == "web.image_search" and "web.image_search" in self.capabilities:
            return await self._invoke_image_search(request, backend, timeout)
        if request.capability == "web.read" and "web.read" in self.capabilities:
            return await self._invoke_read(request, backend, timeout)
        raise invalid_request(
            "This provider does not support the requested web capability", backend
        )

    async def _invoke_search(self, request, backend, timeout):
        query = str(request.payload.get("query") or "").strip()
        if not query:
            raise invalid_request("Search query is empty", backend)
        return await self.search(
            query=query,
            max_results=bounded_int(request.payload.get("max_results"), 8, 1, 20),
            depth=str(request.payload.get("depth") or "balanced"),
            freshness=str(request.payload.get("freshness") or "auto"),
            timeout_seconds=timeout,
            backend=backend,
        )

    async def _invoke_image_search(self, request, backend, timeout):
        query = str(request.payload.get("query") or "").strip()
        if not query:
            raise invalid_request("Image search query is empty", backend)
        return await self.image_search(
            query=query,
            max_results=bounded_int(request.payload.get("max_results"), 8, 1, 20),
            freshness=str(request.payload.get("freshness") or "auto"),
            timeout_seconds=timeout,
            backend=backend,
        )

    async def _invoke_read(self, request, backend, timeout):
        url = str(request.payload.get("url") or "").strip()
        if not url:
            raise invalid_request("Web resource URL is empty", backend)
        return await self.read(
            url=url,
            focus=str(request.payload.get("focus") or ""),
            max_characters=bounded_int(request.payload.get("max_characters"), 12000, 500, 30000),
            timeout_seconds=timeout,
            backend=backend,
        )

    async def search(self, **_: Any) -> ProviderSearchOutput:
        raise NotImplementedError

    async def read(self, **_: Any) -> ProviderReadOutput:
        raise NotImplementedError

    async def image_search(self, **_: Any) -> ProviderImageSearchOutput:
        raise NotImplementedError

    def classify_error(self, exc: BaseException, backend: AIBackendDescriptor) -> AIErrorInfo:
        if isinstance(exc, AIInvocationError):
            return exc.info
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            code = AIErrorCode.TIMEOUT
        elif isinstance(exc, WebHTTPStatusError):
            if exc.status_code == 401:
                code = AIErrorCode.AUTHENTICATION
            elif exc.status_code == 403:
                code = AIErrorCode.PERMISSION
            elif exc.status_code == 429:
                code = AIErrorCode.RATE_LIMIT
            elif exc.status_code >= 500:
                code = AIErrorCode.REMOTE_5XX
            else:
                code = AIErrorCode.INVALID_REQUEST
        elif isinstance(exc, (WebTransportError, ConnectionError, OSError)):
            code = AIErrorCode.NETWORK
        else:
            code = AIErrorCode.INTERNAL
        switch = code in {
            AIErrorCode.AUTHENTICATION,
            AIErrorCode.PERMISSION,
            AIErrorCode.RATE_LIMIT,
            AIErrorCode.REMOTE_5XX,
            AIErrorCode.NETWORK,
            AIErrorCode.TIMEOUT,
        }
        return AIErrorInfo(
            code,
            f"Web provider failed: {code.value}",
            retryable=code in {AIErrorCode.REMOTE_5XX, AIErrorCode.NETWORK},
            switch_backend=switch,
            open_circuit=code
            in {AIErrorCode.AUTHENTICATION, AIErrorCode.PERMISSION, AIErrorCode.RATE_LIMIT},
            backend_id=backend.backend_id,
            phase="web_provider",
            status_code=getattr(exc, "status_code", None),
        )

    async def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        response = await self.transport.request_json(
            method,
            url,
            headers=headers,
            payload=payload,
            params=params,
            timeout_seconds=timeout_seconds,
        )
        if not 200 <= response.status_code < 300:
            raise WebHTTPStatusError(response.status_code)
        return response.data


def invalid_request(message: str, backend: AIBackendDescriptor) -> AIInvocationError:
    return AIInvocationError(
        AIErrorInfo(
            AIErrorCode.INVALID_REQUEST,
            message,
            backend_id=backend.backend_id,
            phase="prepare",
        )
    )


class TavilyWebSearchAdapter(BaseWebAdapter):
    provider_kind = "tavily"
    capabilities = ("web.search", "web.read", "web.image_search")

    async def image_search(
        self,
        *,
        query: str,
        max_results: int,
        freshness: str,
        timeout_seconds: float,
        backend: AIBackendDescriptor,
    ) -> ProviderImageSearchOutput:
        payload: dict[str, Any] = {
            "query": query,
            "max_results": 1,
            "include_images": True,
            "include_image_descriptions": True,
            "search_depth": "basic",
            "topic": "general",
        }
        if freshness in {"day", "week", "month", "year"}:
            payload["time_range"] = freshness
        data = await self._request(
            "POST",
            "https://api.tavily.com/search",
            headers=_bearer_headers(self._credential(backend)),
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        return ProviderImageSearchOutput(
            _image_items(data.get("images") or (), limit=max_results), backend.backend_id
        )

    async def search(
        self,
        *,
        query: str,
        max_results: int,
        depth: str,
        freshness: str,
        timeout_seconds: float,
        backend: AIBackendDescriptor,
    ) -> ProviderSearchOutput:
        payload: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "include_favicon": True,
            "search_depth": "advanced" if depth == "deep" else "basic",
            "topic": "general",
        }
        if freshness in {"day", "week", "month", "year"}:
            payload["time_range"] = freshness
        data = await self._request(
            "POST",
            "https://api.tavily.com/search",
            headers=_bearer_headers(self._credential(backend)),
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        return ProviderSearchOutput(
            _items(
                data.get("results", ()),
                title="title",
                url="url",
                snippet="content",
                favicon="favicon",
                published="published_date",
            ),
            backend.backend_id,
        )

    async def read(
        self,
        *,
        url: str,
        focus: str,
        max_characters: int,
        timeout_seconds: float,
        backend: AIBackendDescriptor,
    ) -> ProviderReadOutput:
        data = await self._request(
            "POST",
            "https://api.tavily.com/extract",
            headers=_bearer_headers(self._credential(backend)),
            payload={"urls": [url], "extract_depth": "advanced" if focus else "basic"},
            timeout_seconds=timeout_seconds,
        )
        rows = data.get("results") or ()
        row = rows[0] if isinstance(rows, list) and rows else {}
        return ProviderReadOutput(
            str(row.get("url") or url),
            str(row.get("raw_content") or "")[:max_characters],
            provider_id=backend.backend_id,
        )


class BoChaWebSearchAdapter(BaseWebAdapter):
    provider_kind = "bocha"
    capabilities = ("web.search", "web.image_search")

    async def image_search(
        self,
        *,
        query: str,
        max_results: int,
        freshness: str,
        timeout_seconds: float,
        backend: AIBackendDescriptor,
    ) -> ProviderImageSearchOutput:
        payload: dict[str, Any] = {"query": query, "count": max_results, "summary": True}
        payload["freshness"] = {
            "day": "oneDay",
            "week": "oneWeek",
            "month": "oneMonth",
            "year": "oneYear",
        }.get(freshness, "noLimit")
        data = await self._request(
            "POST",
            "https://api.bochaai.com/v1/web-search",
            headers={
                **_bearer_headers(self._credential(backend)),
                "Accept-Encoding": "gzip, deflate",
            },
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        root = data.get("data") or {}
        rows = (root.get("images") or {}).get("value") or root.get("images") or ()
        return ProviderImageSearchOutput(_image_items(rows, limit=max_results), backend.backend_id)

    async def search(
        self,
        *,
        query: str,
        max_results: int,
        depth: str,
        freshness: str,
        timeout_seconds: float,
        backend: AIBackendDescriptor,
    ) -> ProviderSearchOutput:
        payload: dict[str, Any] = {"query": query, "count": max_results, "summary": True}
        payload["freshness"] = {
            "day": "oneDay",
            "week": "oneWeek",
            "month": "oneMonth",
            "year": "oneYear",
        }.get(freshness, "noLimit")
        data = await self._request(
            "POST",
            "https://api.bochaai.com/v1/web-search",
            headers={
                **_bearer_headers(self._credential(backend)),
                "Accept-Encoding": "gzip, deflate",
            },
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        rows = ((data.get("data") or {}).get("webPages") or {}).get("value") or ()
        return ProviderSearchOutput(
            _items(
                rows,
                title="name",
                url="url",
                snippet="snippet",
                favicon="siteIcon",
                published="datePublished",
            ),
            backend.backend_id,
        )


class BraveWebSearchAdapter(BaseWebAdapter):
    provider_kind = "brave"
    capabilities = ("web.search", "web.image_search")

    async def image_search(
        self,
        *,
        query: str,
        max_results: int,
        freshness: str,
        timeout_seconds: float,
        backend: AIBackendDescriptor,
    ) -> ProviderImageSearchOutput:
        params: dict[str, Any] = {
            "q": query,
            "count": max_results,
            "country": "US",
            "search_lang": "zh-hans",
            "safesearch": "strict",
        }
        data = await self._request(
            "GET",
            "https://api.search.brave.com/res/v1/images/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self._credential(backend),
            },
            params=params,
            timeout_seconds=timeout_seconds,
        )
        return ProviderImageSearchOutput(
            _image_items(data.get("results") or (), limit=max_results), backend.backend_id
        )

    async def search(
        self,
        *,
        query: str,
        max_results: int,
        depth: str,
        freshness: str,
        timeout_seconds: float,
        backend: AIBackendDescriptor,
    ) -> ProviderSearchOutput:
        params: dict[str, Any] = {
            "q": query,
            "count": max_results,
            "country": "US",
            "search_lang": "zh-hans",
        }
        if freshness in {"day", "week", "month", "year"}:
            params["freshness"] = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}[
                freshness
            ]
        data = await self._request(
            "GET",
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self._credential(backend),
            },
            params=params,
            timeout_seconds=timeout_seconds,
        )
        rows = (data.get("web") or {}).get("results") or ()
        return ProviderSearchOutput(
            _items(rows, title="title", url="url", snippet="description", published="page_age"),
            backend.backend_id,
        )


class FirecrawlWebSearchAdapter(BaseWebAdapter):
    provider_kind = "firecrawl"
    capabilities = ("web.search", "web.read", "web.image_search")

    async def image_search(
        self,
        *,
        query: str,
        max_results: int,
        freshness: str,
        timeout_seconds: float,
        backend: AIBackendDescriptor,
    ) -> ProviderImageSearchOutput:
        data = await self._request(
            "POST",
            "https://api.firecrawl.dev/v2/search",
            headers=_bearer_headers(self._credential(backend)),
            payload={"query": query, "limit": max_results, "sources": ["images"]},
            timeout_seconds=timeout_seconds,
        )
        rows = data.get("data") or ()
        if isinstance(rows, Mapping):
            rows = rows.get("images") or ()
        return ProviderImageSearchOutput(_image_items(rows, limit=max_results), backend.backend_id)

    async def search(
        self,
        *,
        query: str,
        max_results: int,
        depth: str,
        freshness: str,
        timeout_seconds: float,
        backend: AIBackendDescriptor,
    ) -> ProviderSearchOutput:
        data = await self._request(
            "POST",
            "https://api.firecrawl.dev/v2/search",
            headers=_bearer_headers(self._credential(backend)),
            payload={"query": query, "limit": max_results, "sources": ["web"]},
            timeout_seconds=timeout_seconds,
        )
        rows = data.get("data") or ()
        if isinstance(rows, Mapping):
            rows = rows.get("web") or ()
        return ProviderSearchOutput(
            _items(
                rows,
                title="title",
                url="url",
                snippet=("description", "snippet", "markdown"),
                published="publishedDate",
            ),
            backend.backend_id,
        )

    async def read(
        self,
        *,
        url: str,
        focus: str,
        max_characters: int,
        timeout_seconds: float,
        backend: AIBackendDescriptor,
    ) -> ProviderReadOutput:
        data = await self._request(
            "POST",
            "https://api.firecrawl.dev/v2/scrape",
            headers=_bearer_headers(self._credential(backend)),
            payload={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            timeout_seconds=timeout_seconds,
        )
        row = data.get("data") or {}
        return ProviderReadOutput(
            str(row.get("url") or url),
            str(row.get("markdown") or "")[:max_characters],
            title=str((row.get("metadata") or {}).get("title") or ""),
            provider_id=backend.backend_id,
        )


class BaiduWebSearchAdapter(BaseWebAdapter):
    provider_kind = "baidu_ai_search"
    capabilities = ("web.search", "web.image_search")

    async def image_search(
        self,
        *,
        query: str,
        max_results: int,
        freshness: str,
        timeout_seconds: float,
        backend: AIBackendDescriptor,
    ) -> ProviderImageSearchOutput:
        key = self._credential(backend)
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": query[:72]}],
            "search_source": "baidu_search_v2",
            "resource_type_filter": [{"type": "image", "top_k": min(30, max_results)}],
            "safe_search": True,
        }
        data = await self._request(
            "POST",
            "https://qianfan.baidubce.com/v2/ai_search/web_search",
            headers={
                "Authorization": f"Bearer {key}",
                "X-Appbuilder-Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        return ProviderImageSearchOutput(
            _image_items(data.get("references") or (), limit=max_results), backend.backend_id
        )

    async def search(
        self,
        *,
        query: str,
        max_results: int,
        depth: str,
        freshness: str,
        timeout_seconds: float,
        backend: AIBackendDescriptor,
    ) -> ProviderSearchOutput:
        key = self._credential(backend)
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": query[:72]}],
            "search_source": "baidu_search_v2",
            "resource_type_filter": [{"type": "web", "top_k": max_results}],
        }
        if freshness in {"week", "month", "year"}:
            payload["search_recency_filter"] = freshness
        data = await self._request(
            "POST",
            "https://qianfan.baidubce.com/v2/ai_search/web_search",
            headers={
                "Authorization": f"Bearer {key}",
                "X-Appbuilder-Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        return ProviderSearchOutput(
            _items(
                data.get("references") or (),
                title="title",
                url="url",
                snippet="content",
                favicon="icon",
                published="date",
            ),
            backend.backend_id,
        )


class ExaWebSearchAdapter(BaseWebAdapter):
    provider_kind = "exa"
    capabilities = ("web.search", "web.read")

    def _headers(self, backend: AIBackendDescriptor) -> Mapping[str, str]:
        return {"x-api-key": self._credential(backend), "Content-Type": "application/json"}

    async def search(
        self,
        *,
        query: str,
        max_results: int,
        depth: str,
        freshness: str,
        timeout_seconds: float,
        backend: AIBackendDescriptor,
    ) -> ProviderSearchOutput:
        payload: dict[str, Any] = {
            "query": query,
            "numResults": max_results,
            "type": "auto",
            "contents": {"text": {"maxCharacters": 500}},
        }
        freshness_days = {"day": 1, "week": 7, "month": 30, "year": 365}.get(freshness)
        if freshness_days:
            payload["startPublishedDate"] = (
                (datetime.now(UTC) - timedelta(days=freshness_days))
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z")
            )
        data = await self._request(
            "POST",
            "https://api.exa.ai/search",
            headers=self._headers(backend),
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        return ProviderSearchOutput(
            _items(
                data.get("results") or (),
                title="title",
                url="url",
                snippet=("text", "summary", "highlights"),
                published="publishedDate",
            ),
            backend.backend_id,
        )

    async def read(
        self,
        *,
        url: str,
        focus: str,
        max_characters: int,
        timeout_seconds: float,
        backend: AIBackendDescriptor,
    ) -> ProviderReadOutput:
        data = await self._request(
            "POST",
            "https://api.exa.ai/contents",
            headers=self._headers(backend),
            payload={"ids": [url], "text": {"maxCharacters": max_characters}},
            timeout_seconds=timeout_seconds,
        )
        rows = data.get("results") or ()
        row = rows[0] if isinstance(rows, list) and rows else {}
        return ProviderReadOutput(
            str(row.get("url") or url),
            str(row.get("text") or "")[:max_characters],
            title=str(row.get("title") or ""),
            provider_id=backend.backend_id,
        )


def build_web_search_adapter(
    provider_kind: str,
    credential_id: str,
    credential_resolver: CredentialResolver,
    transport: WebJSONTransport | None = None,
) -> BaseWebAdapter:
    classes = {
        "tavily": TavilyWebSearchAdapter,
        "bocha": BoChaWebSearchAdapter,
        "brave": BraveWebSearchAdapter,
        "firecrawl": FirecrawlWebSearchAdapter,
        "baidu_ai_search": BaiduWebSearchAdapter,
        "exa": ExaWebSearchAdapter,
    }
    try:
        cls = classes[str(provider_kind).strip().lower()]
    except KeyError as exc:
        raise ValueError("unsupported web-search provider type") from exc
    return cls(WebSearchAdapterConfig(credential_id), credential_resolver, transport)


__all__ = [
    "BaiduWebSearchAdapter",
    "BoChaWebSearchAdapter",
    "BraveWebSearchAdapter",
    "ExaWebSearchAdapter",
    "FirecrawlWebSearchAdapter",
    "TavilyWebSearchAdapter",
    "AiohttpWebJSONTransport",
    "WebHTTPResponse",
    "WebHTTPStatusError",
    "WebJSONTransport",
    "WebSearchAdapterConfig",
    "WebTransportError",
    "build_web_search_adapter",
]
