"""Preserve QQ Official send receipts discarded by AstrBot media delivery."""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

_HTTP_STATE_ATTRIBUTE = "_soulcore_qq_outbound_receipt_state_v1"
_HTTP_WRAPPER_MARKER = "_soulcore_qq_outbound_receipt_wrapper_v1"


@dataclass(slots=True)
class QQOutboundReceiptCapture:
    """The addressable identifiers returned by one QQ send operation."""

    message_id: str = ""
    platform_reference_id: str = ""
    observations: list[tuple[str, str]] = field(default_factory=list)

    def observe(self, response: Any) -> None:
        message_id = extract_qq_message_id(response)
        reference_id = extract_qq_reference_id(response)
        if message_id:
            self.message_id = message_id
            self.observations.append((message_id, reference_id))
        if reference_id:
            self.platform_reference_id = reference_id


@contextmanager
def capture_qq_outbound_receipt(platform: Any) -> Iterator[QQOutboundReceiptCapture]:
    """Capture the authenticated QQ response within the current async task.

    AstrBot 4.26.x returns ``None`` from QQ Official ``send_by_session`` after
    caching the ordinary message id. Its underlying authenticated HTTP call did
    return ``ext_info.ref_idx`` though, so observe that response without
    replacing AstrBot's platform implementation. A ContextVar keeps concurrent
    sends on a shared QQ connection isolated.
    """

    capture = QQOutboundReceiptCapture()
    state = _install_http_observer(platform)
    if state is None:
        yield capture
        return
    active = state["active"]
    token = active.set(capture)
    try:
        yield capture
    finally:
        active.reset(token)


def extract_qq_message_id(response: Any) -> str:
    target = _qq_message_response(response)
    value = _field(target, "id") or _field(target, "message_id")
    return str(value or "").strip()


def extract_qq_reference_id(response: Any) -> str:
    from .qq_reference_ids import normalize_qq_reference_index

    target = _qq_message_response(response)
    ext_info = _field(target, "ext_info")
    value = _field(ext_info, "ref_idx")
    return normalize_qq_reference_index(value)


def _qq_message_response(response: Any) -> Any:
    """Unwrap the known message envelopes used by QQ SDK send methods."""

    current = response
    for _ in range(3):
        if _field(current, "id") or _field(current, "message_id") or _field(current, "ext_info"):
            return current
        nested = _field(current, "message") or _field(current, "data") or _field(current, "result")
        if nested is None or nested is current:
            break
        current = nested
    return current


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _install_http_observer(platform: Any) -> dict[str, Any] | None:
    client_getter = getattr(platform, "get_client", None)
    client = client_getter() if callable(client_getter) else getattr(platform, "client", None)
    api = getattr(client, "api", None)
    http = getattr(api, "_http", None)
    request = getattr(http, "request", None)
    if http is None or not callable(request):
        return None

    state = getattr(http, _HTTP_STATE_ATTRIBUTE, None)
    if not isinstance(state, dict) or not isinstance(state.get("active"), ContextVar):
        state = {"active": ContextVar(f"soulcore_qq_receipt_{id(http)}", default=None)}
        try:
            setattr(http, _HTTP_STATE_ATTRIBUTE, state)
        except (AttributeError, TypeError):
            return None

    if getattr(request, _HTTP_WRAPPER_MARKER, False):
        return state

    @wraps(request)
    async def observed_request(*args: Any, **kwargs: Any) -> Any:
        value = request(*args, **kwargs)
        response = await value if inspect.isawaitable(value) else value
        current_state = getattr(http, _HTTP_STATE_ATTRIBUTE, state)
        active = current_state.get("active") if isinstance(current_state, dict) else None
        capture = active.get() if isinstance(active, ContextVar) else None
        observer = getattr(capture, "observe", None)
        if callable(observer):
            observer(response)
        return response

    setattr(observed_request, _HTTP_WRAPPER_MARKER, True)
    try:
        http.request = observed_request
    except (AttributeError, TypeError):
        return None
    return state


__all__ = [
    "QQOutboundReceiptCapture",
    "capture_qq_outbound_receipt",
    "extract_qq_message_id",
    "extract_qq_reference_id",
]
