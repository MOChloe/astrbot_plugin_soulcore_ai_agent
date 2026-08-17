"""Bounded image locator I/O and representative-frame projection."""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .fingerprints import bounded_model_preview
from .inspection import MAX_IMAGE_BYTES


def download_http(
    owner: type[Any],
    url: str,
    *,
    referer: str | None = None,
) -> tuple[bytes, str | None]:
    data, declared = _download_remote(
        owner,
        url,
        headers=_headers(referer),
        max_bytes=MAX_IMAGE_BYTES,
        timeout_seconds=30,
    )
    normalized = str(declared or "").strip().lower()
    allowed_binary = {
        "application/octet-stream",
        "binary/octet-stream",
        "application/download",
        "application/x-download",
    }
    if normalized and not normalized.startswith("image/"):
        if normalized not in allowed_binary:
            raise ValueError("remote locator did not return an image")
        normalized = ""
    return data, normalized or None


def download_public_attachment(
    url: str,
    *,
    max_bytes: int,
    timeout_seconds: float,
) -> tuple[bytes, str | None]:
    return _download_remote(
        _DefaultURLValidator,
        url,
        headers={"User-Agent": "SoulCore/1.0", "Accept": "*/*"},
        max_bytes=max_bytes,
        timeout_seconds=timeout_seconds,
    )


def _headers(referer: str | None) -> dict[str, str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
    }
    safe = _safe_referer(referer)
    if safe:
        headers["Referer"] = safe
    return headers


def _safe_referer(value: str | None) -> str:
    parsed = urllib.parse.urlsplit(str(value or ""))
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        return ""
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path or "/",
            "",
            "",
        )
    )


class _HeaderLookup(Protocol):
    def get(self, name: str) -> str | None: ...


def _redirect_target(current: str, status: int, headers: _HeaderLookup) -> str:
    if status not in {301, 302, 303, 307, 308}:
        raise ValueError(f"remote media download failed with HTTP {status}")
    location = str(headers.get("Location") or "").strip()
    if not location:
        raise ValueError("remote media redirect has no target")
    return urllib.parse.urljoin(current, location)


def validate_remote_url(url: str) -> tuple[str, ...]:
    parsed = urllib.parse.urlsplit(str(url or ""))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("invalid remote image locator")
    if parsed.username or parsed.password:
        raise ValueError("remote image locator credentials are not allowed")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".local"):
        raise ValueError("local remote image host is not allowed")
    addresses = _resolved_addresses(
        hostname,
        parsed.port or (443 if parsed.scheme == "https" else 80),
    )
    for address in addresses:
        if not ipaddress.ip_address(address).is_global:
            raise ValueError("private or special remote image address is not allowed")
    return tuple(sorted(addresses))


def _resolved_addresses(hostname: str, port: int) -> set[str]:
    try:
        resolved = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError("remote image host cannot be resolved") from exc
    addresses: set[str] = set()
    for item in resolved:
        address = item[4][0]
        if not isinstance(address, str):
            raise ValueError("remote image host returned an invalid address")
        addresses.add(address)
    if not addresses:
        raise ValueError("remote image host cannot be resolved")
    return addresses


class _DefaultURLValidator:
    validate_remote_url = staticmethod(validate_remote_url)


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(
        self,
        host: str,
        port: int,
        *,
        pinned_address: str,
        timeout: float,
        source_address: tuple[str, int] | None = None,
    ) -> None:
        super().__init__(
            host,
            port,
            timeout=timeout,
            source_address=source_address,
        )
        self._pinned_address = pinned_address
        self._connection_source_address = source_address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._pinned_address, self.port),
            self.timeout,
            self._connection_source_address,
        )
        _verify_peer(self.sock, self._pinned_address)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        port: int,
        *,
        pinned_address: str,
        timeout: float,
        source_address: tuple[str, int] | None = None,
        context: ssl.SSLContext | None = None,
    ) -> None:
        if context is None:
            tls_context = ssl.create_default_context()
            tls_context.set_alpn_protocols(["http/1.1"])
            if tls_context.post_handshake_auth is not None:
                tls_context.post_handshake_auth = True
        else:
            tls_context = context
        super().__init__(
            host,
            port,
            timeout=timeout,
            source_address=source_address,
            context=tls_context,
        )
        self._pinned_address = pinned_address
        self._connection_source_address = source_address
        self._tls_context = tls_context

    def connect(self) -> None:
        raw_socket = socket.create_connection(
            (self._pinned_address, self.port),
            self.timeout,
            self._connection_source_address,
        )
        _verify_peer(raw_socket, self._pinned_address)
        try:
            self.sock = self._tls_context.wrap_socket(raw_socket, server_hostname=self.host)
        except BaseException:
            raw_socket.close()
            raise
        _verify_peer(self.sock, self._pinned_address)


def _verify_peer(sock: socket.socket, expected_address: str) -> None:
    peer_address = str(sock.getpeername()[0]).split("%", 1)[0]
    if ipaddress.ip_address(peer_address) != ipaddress.ip_address(expected_address):
        sock.close()
        raise ValueError("remote media connection address changed")


def _download_remote(
    owner: type[Any],
    url: str,
    *,
    headers: Mapping[str, str],
    max_bytes: int,
    timeout_seconds: float,
) -> tuple[bytes, str | None]:
    deadline = time.monotonic() + timeout_seconds
    current = str(url)
    for _ in range(4):
        addresses = owner.validate_remote_url(current)
        if not addresses:
            raise ValueError("remote media URL validation returned no addresses")
        response, connection = _open_validated(
            current,
            addresses=tuple(addresses),
            headers=headers,
            deadline=deadline,
        )
        try:
            if 300 <= response.status < 400:
                current = _redirect_target(current, response.status, response.headers)
                continue
            if not 200 <= response.status < 300:
                raise ValueError(f"remote media download failed with HTTP {response.status}")
            return _read_bounded_response(
                response,
                connection,
                max_bytes=max_bytes,
                deadline=deadline,
            )
        finally:
            response.close()
            connection.close()
    raise ValueError("remote media locator has too many redirects")


def _open_validated(
    url: str,
    *,
    addresses: Sequence[str],
    headers: Mapping[str, str],
    deadline: float,
) -> tuple[http.client.HTTPResponse, http.client.HTTPConnection]:
    parsed = urllib.parse.urlsplit(url)
    hostname = str(parsed.hostname)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    last_error: OSError | None = None
    for address in addresses:
        connection: http.client.HTTPConnection
        remaining = _remaining_timeout(deadline)
        if parsed.scheme == "https":
            connection = _PinnedHTTPSConnection(
                hostname,
                port,
                pinned_address=address,
                timeout=remaining,
            )
        else:
            connection = _PinnedHTTPConnection(
                hostname,
                port,
                pinned_address=address,
                timeout=remaining,
            )
        try:
            connection.request("GET", target, headers=dict(headers))
            return connection.getresponse(), connection
        except OSError as exc:
            connection.close()
            last_error = exc
    raise ValueError("remote media host could not be reached") from last_error


def _read_bounded_response(
    response: http.client.HTTPResponse,
    connection: http.client.HTTPConnection,
    *,
    max_bytes: int,
    deadline: float,
) -> tuple[bytes, str | None]:
    declared_length = response.headers.get("Content-Length")
    if declared_length:
        try:
            length = int(declared_length)
        except ValueError as exc:
            raise ValueError("remote media Content-Length is invalid") from exc
        if length > max_bytes:
            raise ValueError("remote media file is too large")
    chunks: list[bytes] = []
    size = 0
    while True:
        if connection.sock is not None:
            connection.sock.settimeout(_remaining_timeout(deadline))
        chunk = response.read(min(64 * 1024, max_bytes + 1 - size))
        if not chunk:
            break
        size += len(chunk)
        if size > max_bytes:
            raise ValueError("remote media file is too large")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise ValueError("remote media file is empty")
    declared = str(response.headers.get("Content-Type") or "").split(";", 1)[0]
    return data, declared or None


def _remaining_timeout(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("remote media download timed out")
    return remaining


def vision_payloads(data: bytes, mime_type: str) -> list[tuple[bytes, str]]:
    if str(mime_type).lower() not in {"image/gif", "image/webp"}:
        return [(data, mime_type)]
    preview = bounded_model_preview(data, mime_type)
    return list(preview.payloads) if preview.animated else [(data, mime_type)]


def vision_payload(data: bytes, mime_type: str) -> tuple[bytes, str]:
    return vision_payloads(data, mime_type)[0]


__all__ = [
    "download_http",
    "download_public_attachment",
    "validate_remote_url",
    "vision_payload",
    "vision_payloads",
]
