"""HTTP primitives shared by the OpenAI-compatible adapter."""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

from ...shared.http_security import (
    HTTPResponseTooLargeError,
    open_same_origin,
    read_limited,
)


@dataclass(frozen=True, slots=True)
class HTTPJSONResponse:
    status_code: int
    data: Mapping[str, Any]
    headers: Mapping[str, str] = field(default_factory=dict)
    raw_text: str = ""


class JSONTransport(Protocol):
    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> HTTPJSONResponse: ...


class OpenAITransportError(RuntimeError):
    pass


class OpenAIHTTPStatusError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        *,
        api_code: str = "",
        retry_after_seconds: float | None = None,
        provider_response: Any = None,
    ) -> None:
        super().__init__(f"OpenAI-compatible API returned HTTP {int(status_code)}")
        self.status_code = int(status_code)
        self.api_code = str(api_code or "")
        self.retry_after_seconds = retry_after_seconds
        self.provider_response = provider_response


class UrllibJSONTransport:
    """Standard-library transport; tests normally inject an in-memory fake."""

    def __init__(self, *, max_response_bytes: int = 32 * 1024 * 1024) -> None:
        maximum = int(max_response_bytes)
        if maximum < 1:
            raise ValueError("max_response_bytes must be positive")
        self.max_response_bytes = maximum

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        return await asyncio.to_thread(
            self._post,
            url,
            dict(headers),
            dict(payload),
            float(timeout_seconds),
            self.max_response_bytes,
        )

    @staticmethod
    def _post(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> HTTPJSONResponse:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with open_same_origin(request, timeout=timeout_seconds) as response:
                raw = read_limited(response, max_response_bytes)
                data = json.loads(raw.decode("utf-8")) if raw else {}
                if not isinstance(data, dict):
                    raise OpenAITransportError("API response is not a JSON object")
                return HTTPJSONResponse(
                    int(getattr(response, "status", 200)),
                    data,
                    {str(key): str(value) for key, value in response.headers.items()},
                    raw.decode("utf-8"),
                )
        except urllib.error.HTTPError as exc:
            raise _http_status_error(exc) from None
        except HTTPResponseTooLargeError:
            raise OpenAITransportError("provider_response_too_large") from None
        except (urllib.error.URLError, OSError) as exc:
            raise OpenAITransportError(type(exc).__name__) from None


def _http_status_error(exc: urllib.error.HTTPError) -> OpenAIHTTPStatusError:
    raw = exc.read(64 * 1024)
    api_code = ""
    value: Any = None
    raw_text = ""
    try:
        raw_text = raw.decode("utf-8") if raw else ""
        value = json.loads(raw_text) if raw_text else {}
        error = value.get("error") if isinstance(value, dict) else None
        if isinstance(error, dict):
            api_code = str(error.get("code") or error.get("type") or "")
    except (UnicodeDecodeError, ValueError):
        pass
    return OpenAIHTTPStatusError(
        exc.code,
        api_code=api_code,
        retry_after_seconds=_retry_after(dict(exc.headers.items()) if exc.headers else {}),
        provider_response=raw_text or value,
    )


def _retry_after(headers: Mapping[str, str]) -> float | None:
    value = next(
        (str(raw) for key, raw in headers.items() if str(key).lower() == "retry-after"),
        "",
    ).strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return _retry_after_date(value)


def _retry_after_date(value: str) -> float | None:
    try:
        parsed = parsedate_to_datetime(value)
        return max(0.0, parsed.timestamp() - time.time())
    except (TypeError, ValueError, OverflowError):
        return None


__all__ = [
    "HTTPJSONResponse",
    "JSONTransport",
    "OpenAIHTTPStatusError",
    "OpenAITransportError",
    "UrllibJSONTransport",
]
