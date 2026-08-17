"""Cancellation-safe JSON transport for whitelisted web providers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class WebHTTPResponse:
    status_code: int
    data: Mapping[str, Any]
    headers: Mapping[str, str]


class WebJSONTransport(Protocol):
    async def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, object] | None,
        params: Mapping[str, object] | None,
        timeout_seconds: float,
    ) -> WebHTTPResponse: ...


class WebTransportError(RuntimeError):
    pass


class WebHTTPStatusError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"web provider returned HTTP {int(status_code)}")
        self.status_code = int(status_code)


class AiohttpWebJSONTransport:
    """Cancellation-safe transport using AstrBot's existing aiohttp runtime."""

    _MAX_RESPONSE_BYTES = 4 * 1024 * 1024

    async def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None,
        params: Mapping[str, Any] | None,
        timeout_seconds: float,
    ) -> WebHTTPResponse:
        # AstrBot ships aiohttp. Keep the import lazy so pure-domain tests and
        # packaging checks remain importable outside AstrBot.
        try:
            import aiohttp
        except ImportError as exc:  # pragma: no cover - AstrBot always supplies it
            raise WebTransportError("aiohttp is unavailable") from exc
        timeout = aiohttp.ClientTimeout(total=max(1.0, float(timeout_seconds)))
        try:
            async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                async with session.request(
                    str(method).upper(),
                    url,
                    headers=dict(headers),
                    json=dict(payload) if payload is not None else None,
                    params=dict(params) if params is not None else None,
                    allow_redirects=False,
                ) as response:
                    if not 200 <= response.status < 300:
                        # Provider error bodies may echo queries or credentials.
                        raise WebHTTPStatusError(response.status)
                    raw = bytearray()
                    async for chunk in response.content.iter_chunked(64 * 1024):
                        raw.extend(chunk)
                        if len(raw) > self._MAX_RESPONSE_BYTES:
                            raise WebTransportError("provider response exceeds 4 MiB")
                data = json.loads(bytes(raw).decode("utf-8")) if raw else {}
                if not isinstance(data, Mapping):
                    raise WebTransportError("provider response is not a JSON object")
                return WebHTTPResponse(
                    int(response.status),
                    dict(data),
                    {str(k): str(v) for k, v in response.headers.items()},
                )
        except asyncio.CancelledError:
            raise
        except WebHTTPStatusError:
            raise
        except (aiohttp.ClientError, OSError, UnicodeError, ValueError) as exc:
            raise WebTransportError(type(exc).__name__) from None


__all__ = [
    "AiohttpWebJSONTransport",
    "WebHTTPResponse",
    "WebHTTPStatusError",
    "WebJSONTransport",
    "WebTransportError",
]
