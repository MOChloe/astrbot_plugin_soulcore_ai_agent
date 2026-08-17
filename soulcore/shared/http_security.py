"""Fail-closed HTTP policy shared by provider transports and admin validation."""

from __future__ import annotations

import ipaddress
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import BinaryIO, Protocol, cast
from urllib.parse import urlsplit


class HTTPResponseTooLargeError(RuntimeError):
    """Raised before an untrusted response can be buffered without a bound."""


class UnsafeHTTPRedirectError(urllib.error.URLError):
    """A redirect attempted to cross the configured provider trust boundary."""


class _ReadableResponse(Protocol):
    headers: Mapping[str, str]

    def read(self, amount: int = -1) -> bytes: ...


def require_secure_http_url(value: str, label: str = "URL") -> str:
    """Require HTTPS, except for explicit loopback-only local providers."""

    text = str(value or "").strip()
    try:
        parsed = urlsplit(text)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid HTTP(S) URL") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{label} must be a valid HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must not contain credentials")
    if parsed.fragment:
        raise ValueError(f"{label} must not contain a URL fragment")
    if parsed.scheme.lower() == "http" and not _is_loopback_host(parsed.hostname):
        raise ValueError(f"{label} must use HTTPS unless it targets loopback")
    return text


def same_origin(first_url: str, second_url: str) -> bool:
    try:
        first = urlsplit(first_url)
        second = urlsplit(second_url)
        first_port = first.port or (443 if first.scheme.lower() == "https" else 80)
        second_port = second.port or (443 if second.scheme.lower() == "https" else 80)
    except ValueError:
        return False
    return bool(
        first.scheme.lower() in {"http", "https"}
        and first.scheme.lower() == second.scheme.lower()
        and str(first.hostname or "").casefold() == str(second.hostname or "").casefold()
        and first_port == second_port
        and second.username is None
        and second.password is None
    )


class SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow provider redirects only inside the originally configured origin."""

    def __init__(self, initial_url: str) -> None:
        super().__init__()
        self._initial_url = require_secure_http_url(initial_url)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not same_origin(self._initial_url, str(newurl)):
            raise UnsafeHTTPRedirectError("cross_origin_provider_redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_same_origin(request: urllib.request.Request, *, timeout: float):
    url = require_secure_http_url(request.full_url)
    opener = urllib.request.build_opener(SameOriginRedirectHandler(url))
    return opener.open(request, timeout=float(timeout))


def read_limited(response: _ReadableResponse | BinaryIO, maximum_bytes: int) -> bytes:
    maximum = int(maximum_bytes)
    if maximum < 1:
        raise ValueError("maximum response size must be positive")
    headers = getattr(response, "headers", {})
    raw_length = _header(headers, "content-length")
    if raw_length:
        try:
            declared = int(raw_length)
        except ValueError:
            declared = -1
        if declared > maximum:
            raise HTTPResponseTooLargeError("provider_response_too_large")
    data = cast(_ReadableResponse, response).read(maximum + 1)
    if len(data) > maximum:
        raise HTTPResponseTooLargeError("provider_response_too_large")
    return data


def _header(headers: object, name: str) -> str:
    if not isinstance(headers, Mapping) and not hasattr(headers, "items"):
        return ""
    items_method = getattr(headers, "items", None)
    if not callable(items_method):
        return ""
    target = name.casefold()
    try:
        items = items_method()
    except Exception:
        return ""
    return next(
        (str(value) for key, value in items if str(key).casefold() == target),
        "",
    )


def _is_loopback_host(hostname: str) -> bool:
    host = str(hostname or "").strip().rstrip(".").casefold()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


__all__ = [
    "HTTPResponseTooLargeError",
    "SameOriginRedirectHandler",
    "UnsafeHTTPRedirectError",
    "open_same_origin",
    "read_limited",
    "require_secure_http_url",
    "same_origin",
]
