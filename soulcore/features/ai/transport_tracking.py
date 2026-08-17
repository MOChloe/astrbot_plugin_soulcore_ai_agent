from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TransportSendHook = Callable[[Any], Awaitable[None]]

_transport_send_hook: ContextVar[TransportSendHook | None] = ContextVar(
    "soulcore_transport_send_hook",
    default=None,
)


@contextmanager
def bind_transport_send_hook(hook: TransportSendHook) -> Iterator[None]:
    token = _transport_send_hook.set(hook)
    try:
        yield
    finally:
        _transport_send_hook.reset(token)


async def mark_transport_send(provider_request: Any) -> None:
    hook = _transport_send_hook.get()
    if hook is not None:
        await hook(provider_request)


def json_transport_request(endpoint: str, payload: Any) -> dict[str, Any]:
    return {
        "method": "POST",
        "endpoint": _safe_endpoint(endpoint),
        "content_type": "application/json",
        "payload": _safe_value(payload),
    }


def multipart_transport_request(
    endpoint: str,
    fields: Any,
    files: Any,
) -> dict[str, Any]:
    return {
        "method": "POST",
        "endpoint": _safe_endpoint(endpoint),
        "content_type": "multipart/form-data",
        "fields": _safe_value(fields),
        "parts": [
            {
                "part_name": str(part_name),
                "filename": str(filename),
                "mime_type": str(mime_type),
                "size_bytes": len(bytes(data)),
                "sha256": hashlib.sha256(bytes(data)).hexdigest(),
            }
            for part_name, filename, mime_type, data in files
        ],
    }


def _safe_value(value: Any, *, field_name: str = "") -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {
            "binary": True,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]" if _secret_field(str(key)) else _safe_value(item, field_name=str(key))
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, field_name=field_name) for item in value]
    if isinstance(value, str) and (
        value.startswith("data:") or field_name.casefold() in {"data", "b64_json"}
    ):
        raw = value.encode("utf-8")
        return {
            "encoded_data": True,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return value


def _safe_endpoint(endpoint: str) -> str:
    parts = urlsplit(str(endpoint))
    safe_query = urlencode(
        [
            (
                key,
                "[REDACTED]" if _secret_field(key) else value,
            )
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
        ]
    )
    safe_netloc = parts.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parts.scheme, safe_netloc, parts.path, safe_query, ""))


def _secret_field(value: str) -> bool:
    compact = "".join(character for character in str(value).casefold() if character.isalnum())
    return compact in {"key", "auth", "token", "secret", "signature", "password", "passwd"} or any(
        marker in compact
        for marker in (
            "apikey",
            "authorization",
            "accesstoken",
            "refreshtoken",
            "authtoken",
            "clientsecret",
            "credential",
            "password",
            "passwd",
            "signature",
        )
    )


__all__ = [
    "bind_transport_send_hook",
    "json_transport_request",
    "mark_transport_send",
    "multipart_transport_request",
]
