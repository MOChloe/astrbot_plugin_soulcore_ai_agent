"""In-memory HTTP transport used by SoulCore audio capability adapters."""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ...shared.http_security import HTTPResponseTooLargeError, open_same_origin, read_limited
from .openai_compatible import OpenAIHTTPStatusError, OpenAITransportError

_FORMAT_BY_AUDIO_MIME = {
    "audio/aac": "aac",
    "audio/flac": "flac",
    "audio/m4a": "m4a",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
    "audio/opus": "opus",
    "audio/pcm": "pcm",
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
}


@dataclass(frozen=True, slots=True)
class HTTPAudioResponse:
    status_code: int
    body: bytes = field(default=b"", repr=False, compare=False)
    headers: Mapping[str, str] = field(default_factory=dict)


class AudioHTTPTransport(Protocol):
    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> HTTPAudioResponse: ...

    async def post_multipart(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        fields: Mapping[str, str],
        files: Sequence[tuple[str, str, str, bytes]],
        timeout_seconds: float,
    ) -> HTTPAudioResponse: ...

    async def get_bytes(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HTTPAudioResponse: ...


class UrllibAudioTransport:
    """Small standard-library transport with injectable in-memory test doubles."""

    def __init__(self, *, max_response_bytes: int = 128 * 1024 * 1024) -> None:
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
    ) -> HTTPAudioResponse:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return await asyncio.to_thread(
            self._request,
            url,
            "POST",
            {**dict(headers), "Content-Type": "application/json"},
            body,
            float(timeout_seconds),
            self.max_response_bytes,
        )

    async def post_multipart(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        fields: Mapping[str, str],
        files: Sequence[tuple[str, str, str, bytes]],
        timeout_seconds: float,
    ) -> HTTPAudioResponse:
        boundary = "soulcore-audio-" + secrets.token_hex(16)
        body = bytearray()
        for name, value in fields.items():
            body.extend(
                (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
                    f"\r\n\r\n{value}\r\n"
                ).encode()
            )
        for name, filename, mime_type, data in files:
            safe_filename = _safe_filename(filename, mime_type)
            body.extend(
                (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                    f'filename="{safe_filename}"\r\nContent-Type: {mime_type}\r\n\r\n'
                ).encode()
            )
            body.extend(data)
            body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("ascii"))
        return await asyncio.to_thread(
            self._request,
            url,
            "POST",
            {**dict(headers), "Content-Type": f"multipart/form-data; boundary={boundary}"},
            bytes(body),
            float(timeout_seconds),
            self.max_response_bytes,
        )

    async def get_bytes(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HTTPAudioResponse:
        return await asyncio.to_thread(
            self._request,
            url,
            "GET",
            dict(headers),
            None,
            float(timeout_seconds),
            self.max_response_bytes,
        )

    @staticmethod
    def _request(
        url: str,
        method: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> HTTPAudioResponse:
        request = urllib.request.Request(
            url,
            data=body,
            headers=dict(headers),
            method=method,
        )
        try:
            with open_same_origin(request, timeout=timeout_seconds) as response:
                return HTTPAudioResponse(
                    int(getattr(response, "status", 200)),
                    read_limited(response, max_response_bytes),
                    {str(key): str(value) for key, value in response.headers.items()},
                )
        except urllib.error.HTTPError as exc:
            raw = exc.read(64 * 1024)
            api_code = _error_code(raw)
            raise OpenAIHTTPStatusError(
                exc.code,
                api_code=api_code,
                retry_after_seconds=_retry_after(
                    {str(key): str(value) for key, value in exc.headers.items()}
                    if exc.headers
                    else {}
                ),
                provider_response=({"error": {"code": api_code}} if api_code else None),
            ) from None
        except HTTPResponseTooLargeError:
            raise OpenAITransportError("provider_response_too_large") from None
        except (urllib.error.URLError, OSError) as exc:
            # URLs and request bodies may contain private local audio sources.
            raise OpenAITransportError(type(exc).__name__) from None


def _safe_filename(filename: str, mime_type: str) -> str:
    basename = re.split(r"[/\\]", str(filename or ""))[-1]
    basename = re.sub(r"[^A-Za-z0-9._-]", "_", basename).strip("._")
    if basename:
        return basename[:128]
    extension = _FORMAT_BY_AUDIO_MIME.get(str(mime_type).split(";", 1)[0].lower(), "bin")
    return f"audio.{extension}"


def _header(headers: Mapping[str, str], name: str) -> str:
    target = name.lower()
    return next((str(value) for key, value in headers.items() if str(key).lower() == target), "")


def _error_code(raw: bytes) -> str:
    try:
        value = json.loads(raw.decode("utf-8")) if raw else {}
    except (UnicodeDecodeError, ValueError):
        return ""
    if not isinstance(value, Mapping):
        return ""
    error = value.get("error")
    if isinstance(error, Mapping):
        return str(error.get("code") or error.get("type") or "")
    base = value.get("base_resp")
    if isinstance(base, Mapping):
        return str(base.get("status_code") or "")
    return ""


def _retry_after(headers: Mapping[str, str]) -> float | None:
    raw = _header(headers, "retry-after").strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


__all__ = ["AudioHTTPTransport", "HTTPAudioResponse", "UrllibAudioTransport"]
