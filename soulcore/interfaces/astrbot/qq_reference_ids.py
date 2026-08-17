"""Capture QQ Official reference metadata before qq-botpy discards it."""

from __future__ import annotations

import gc
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from functools import wraps
from threading import RLock
from time import monotonic
from typing import Any

_CACHE_ATTRIBUTE = "_soulcore_qq_reference_id_cache"
_REPLY_CACHE_ATTRIBUTE = "_soulcore_qq_reply_reference_cache"
_WRAPPER_MARKER = "_soulcore_qq_reference_id_wrapper"
_CURRENT_WRAPPER_TOKEN = object()
_PARSER_NAMES = (
    "parse_c2c_message_create",
    "parse_group_at_message_create",
    "parse_group_message_create",
)
_PARSER_KEYS = tuple(name.removeprefix("parse_") for name in _PARSER_NAMES)
_DEFAULT_TTL_SECONDS = 600.0
_DEFAULT_MAX_ENTRIES = 4096


@dataclass(frozen=True, slots=True)
class QQReplyCapture:
    """Bounded quote payload preserved before qq-botpy drops it."""

    reference_id: str = ""
    content_projection: str = ""
    content_kind: str = "OTHER"
    attachment_count: int = 0

    @property
    def available(self) -> bool:
        return bool(
            self.reference_id or self.content_projection or self.content_kind in {"IMAGE", "FILE"}
        )


def install_qq_reference_id_capture() -> bool:
    """Idempotently wrap qq-botpy raw parsers, including already-created states."""

    state_type = _connection_state_type()
    if state_type is None:
        return False
    _cache_for(state_type)
    for name in _PARSER_NAMES:
        installed = getattr(state_type, name, None)
        if not callable(installed) or _is_current_wrapper(installed):
            continue
        original = _unwrap_previous_wrappers(installed)

        @wraps(original)
        def parser(self: Any, payload: Any, *, _original: Any = original) -> Any:
            _capture_safely(type(self), getattr(self, "api", None), payload)
            return _original(self, payload)

        setattr(parser, _WRAPPER_MARKER, _CURRENT_WRAPPER_TOKEN)
        setattr(state_type, name, parser)
    _wrap_state_initializer(state_type)
    _wrap_existing_state_parsers(state_type)
    return True


def bind_qq_reference_id_capture(context: Any) -> dict[str, int]:
    """Bind the QQ parser bridge to live clients after plugin hot install.

    AstrBot creates qq-botpy ``ConnectionState.parsers`` before a hot-installed
    plugin is imported. Replacing only the class methods therefore cannot affect
    the already-running websocket. Resolve the live platform clients through the
    AstrBot context and wrap their concrete parser dictionaries as well.
    """

    installed = install_qq_reference_id_capture()
    state_type = _connection_state_type()
    if not installed or state_type is None:
        return {"platforms": 0, "states": 0, "parser_slots": 0}
    platforms = 0
    states: set[int] = set()
    parser_slots = 0
    manager = getattr(context, "platform_manager", None)
    getter = getattr(manager, "get_insts", None)
    instances = getter() if callable(getter) else None
    if instances is None:
        instances = getattr(manager, "platform_insts", [])
    for platform in instances or []:
        client_getter = getattr(platform, "get_client", None)
        client = client_getter() if callable(client_getter) else getattr(platform, "client", None)
        connection = getattr(client, "_connection", None)
        state = getattr(connection, "state", None)
        if not isinstance(state, state_type):
            continue
        platforms += 1
        states.add(id(state))
        parser_slots += _wrap_parser_map(state_type, state)
        parser_map = getattr(connection, "parser", None)
        state_parsers = getattr(state, "parsers", None)
        if isinstance(parser_map, dict) and parser_map is not state_parsers:
            parser_slots += _wrap_parser_dict(state_type, state, parser_map)
    result = {"platforms": platforms, "states": len(states), "parser_slots": parser_slots}
    with suppress(Exception):
        from astrbot import logger

        if result["states"]:
            logger.info(
                "[SoulCore] QQ reference parser bridge live bind "
                "platforms=%s states=%s parser_slots=%s",
                result["platforms"],
                result["states"],
                result["parser_slots"],
            )
    return result


def _live_qq_states(
    context: Any, state_type: type[Any]
) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    states: dict[int, Any] = {}
    parser_maps: list[dict[str, Any]] = []
    for platform in context.platform_manager.get_insts():
        connection = getattr(platform.get_client(), "_connection", None)
        state = getattr(connection, "state", None)
        if isinstance(state, state_type):
            states[id(state)] = state
        parser_map = getattr(connection, "parser", None)
        if isinstance(parser_map, dict):
            parser_maps.append(parser_map)
    with suppress(Exception):
        for candidate in gc.get_objects():
            with suppress(Exception):
                if isinstance(candidate, state_type):
                    states[id(candidate)] = candidate
    return states, parser_maps


def _clear_qq_capture_caches(state_type: type[Any]) -> None:
    for attribute in (_CACHE_ATTRIBUTE, _REPLY_CACHE_ATTRIBUTE):
        cache = getattr(state_type, attribute, None)
        if isinstance(cache, dict):
            entries, lock = cache.get("entries"), cache.get("lock")
            if isinstance(entries, OrderedDict) and lock is not None:
                with lock:
                    entries.clear()
        with suppress(AttributeError):
            delattr(state_type, attribute)


def uninstall_qq_reference_id_capture(context: Any) -> None:
    """Remove this plugin's process-wide QQ parser hooks and sensitive caches."""

    state_type = _connection_state_type()
    if state_type is None:
        return
    for name in (*_PARSER_NAMES, "__init__"):
        installed = getattr(state_type, name, None)
        if callable(installed) and getattr(installed, _WRAPPER_MARKER, None):
            setattr(state_type, name, _unwrap_previous_wrappers(installed))
    states, parser_maps = _live_qq_states(context, state_type)
    for state in states.values():
        _unwrap_parser_map(state)
    for parser_map in parser_maps:
        _unwrap_parser_dict(parser_map)
    _clear_qq_capture_caches(state_type)


def event_platform_reference_id(event: Any) -> str:
    """Return the captured REFIDX for an AstrBot QQ event, or an empty string."""

    try:
        message = getattr(event, "message_obj", None)
        raw = getattr(message, "raw_message", None)
        raw_data = _raw_payload(raw)
        if raw_data is not None:
            _, direct = _reference_identity(raw_data)
            if direct:
                return direct
        api = getattr(raw, "_api", None)
        message_id = str(
            getattr(raw, "id", None) or getattr(message, "message_id", None) or ""
        ).strip()
        state_type = _connection_state_type()
        if state_type is None or api is None or not message_id:
            return ""
        return _resolve_cached(state_type, api, message_id)
    except Exception:
        return ""


def event_reply_reference_id(event: Any) -> str:
    """Return the quoted target captured from one QQ Official raw event."""

    capture = event_reply_capture(event)
    return capture.reference_id if capture is not None else ""


def prefer_qq_reference_locator(*values: Any) -> str:
    """Prefer a real QQ REFIDX over ordinary platform message identifiers."""

    candidates = [str(value or "").strip() for value in values]
    candidates = [value for value in candidates if value]
    for value in candidates:
        if value.upper().startswith("REFIDX"):
            return normalize_qq_reference_index(value)
    return candidates[0] if candidates else ""


def qq_reference_locator_candidates(*values: Any) -> tuple[str, ...]:
    """Return canonical REFIDX and raw suffix spellings without changing ordinary ids."""

    canonical: list[str] = []
    ordinary: list[str] = []
    for raw in values:
        value = str(raw or "").strip()
        if not value:
            continue
        if value.upper().startswith("REFIDX"):
            normalized = normalize_qq_reference_index(value)
            if normalized:
                canonical.extend((normalized, normalized[len("REFIDX_") :]))
        else:
            ordinary.append(value)
    return tuple(dict.fromkeys([*canonical, *ordinary]))


def normalize_qq_reference_index(value: Any) -> str:
    """Return QQ's canonical ``REFIDX_`` locator for an index field.

    Some QQ gateway revisions expose only the suffix in ``msg_idx``,
    ``ref_msg_idx`` or ``ext_info.ref_idx`` while quote events use the full
    ``REFIDX_`` spelling.  Only callers that already know a value came from one
    of those index fields may use this helper; ordinary message ids must remain
    untouched.
    """

    normalized = str(value or "").strip()
    if not normalized:
        return ""
    if normalized.upper().startswith("REFIDX"):
        return normalized
    return f"REFIDX_{normalized}"


def event_reply_capture(event: Any) -> QQReplyCapture | None:
    """Return a bounded quote captured from raw data or AstrBot's Reply component."""

    try:
        component_capture = _event_reply_component_capture(event)
        message = getattr(event, "message_obj", None)
        raw = getattr(message, "raw_message", None)
        raw_data = _raw_payload(raw)
        if raw_data is not None:
            _, direct = _reply_capture_identity(raw_data)
            if direct is not None and direct.available:
                return _merge_reply_captures(direct, component_capture)
        api = getattr(raw, "_api", None)
        message_id = str(
            getattr(raw, "id", None) or getattr(message, "message_id", None) or ""
        ).strip()
        state_type = _connection_state_type()
        if state_type is None or api is None or not message_id:
            return component_capture
        return _merge_reply_captures(
            _resolve_cached_reply(state_type, api, message_id), component_capture
        )
    except Exception:
        return None


def _merge_reply_captures(
    primary: QQReplyCapture | None, fallback: QQReplyCapture | None
) -> QQReplyCapture | None:
    if primary is None:
        return fallback
    if fallback is None:
        return primary
    projection = primary.content_projection or fallback.content_projection
    kind = primary.content_kind
    if kind == "OTHER" and fallback.content_kind != "OTHER":
        kind = fallback.content_kind
    return QQReplyCapture(
        reference_id=primary.reference_id or fallback.reference_id,
        content_projection=projection,
        content_kind=kind,
        attachment_count=max(primary.attachment_count, fallback.attachment_count),
    )


def _event_reply_component_capture(event: Any) -> QQReplyCapture | None:
    """Read only bounded semantic fields retained on AstrBot Reply components."""

    for component in _event_message_components(event):
        capture = _reply_component_capture(component)
        if capture is not None:
            return capture
    return None


def _event_message_components(event: Any) -> list[Any]:
    getter = getattr(event, "get_messages", None)
    try:
        components = list(getter() or []) if callable(getter) else []
    except Exception:
        components = []
    if not components:
        message = getattr(event, "message_obj", None)
        components = list(getattr(message, "message", None) or [])
    return components


def _reply_component_capture(component: Any) -> QQReplyCapture | None:
    component_type = (
        (
            str(component.get("type") or component.get("kind") or "")
            if isinstance(component, Mapping)
            else component.__class__.__name__
        )
        .strip()
        .lower()
    )
    if component_type != "reply":
        return None
    reference_id = str(
        _element_field(component, "id") or _element_field(component, "message_id") or ""
    ).strip()
    content = str(
        _element_field(component, "message_str")
        or _element_field(component, "text")
        or _element_field(component, "content")
        or ""
    ).strip()[:500]
    if not content:
        content = _reply_chain_projection(
            _element_field(component, "chain") or _element_field(component, "message_chain")
        )
    capture = QQReplyCapture(
        reference_id=reference_id,
        content_projection=content,
        content_kind="TEXT" if content else "OTHER",
    )
    return capture if capture.available else None


def _reply_chain_projection(value: Any) -> str:
    """Project AstrBot ``Reply.chain`` without retaining platform locators."""

    if not isinstance(value, (list, tuple)):
        return ""
    parts: list[str] = []
    for item in value[:20]:
        kind = (
            (
                str(item.get("type") or item.get("kind") or "")
                if isinstance(item, Mapping)
                else item.__class__.__name__
            )
            .strip()
            .lower()
        )
        if kind in {"plain", "text"}:
            text = str(_element_field(item, "text") or _element_field(item, "content") or "")
            if text.strip():
                parts.append(text.strip())
        elif kind == "image":
            parts.append("[图片]")
        elif kind in {"file", "record", "audio", "voice", "video"}:
            parts.append("[文件]")
        if sum(len(part) for part in parts) >= 500:
            break
    return " ".join(parts).strip()[:500]


def _raw_payload(raw: Any) -> Mapping[str, Any] | None:
    if isinstance(raw, Mapping):
        return raw
    raw_data = getattr(raw, "raw_data", None)
    if isinstance(raw_data, Mapping):
        return raw_data
    message_type = getattr(raw, "message_type", None)
    msg_elements = getattr(raw, "msg_elements", None)
    message_reference = getattr(raw, "message_reference", None)
    reference_id = getattr(message_reference, "message_id", None)
    if message_type is None and not msg_elements and not reference_id:
        return None
    payload: dict[str, Any] = {
        "id": getattr(raw, "id", None),
        "message_type": message_type,
        "msg_elements": msg_elements,
    }
    if reference_id:
        payload["message_reference"] = {"message_id": reference_id}
    return payload


def _connection_state_type() -> type[Any] | None:
    try:
        from botpy.connection import ConnectionState
    except (ImportError, ModuleNotFoundError):
        return None
    return ConnectionState


def _cache_for(state_type: type[Any]) -> dict[str, Any]:
    cache = getattr(state_type, _CACHE_ATTRIBUTE, None)
    if isinstance(cache, dict) and isinstance(cache.get("entries"), OrderedDict):
        return cache
    cache = {
        "entries": OrderedDict(),
        "lock": RLock(),
        "ttl_seconds": _DEFAULT_TTL_SECONDS,
        "max_entries": _DEFAULT_MAX_ENTRIES,
    }
    setattr(state_type, _CACHE_ATTRIBUTE, cache)
    return cache


def _reply_cache_for(state_type: type[Any]) -> dict[str, Any]:
    cache = getattr(state_type, _REPLY_CACHE_ATTRIBUTE, None)
    if isinstance(cache, dict) and isinstance(cache.get("entries"), OrderedDict):
        return cache
    cache = {
        "entries": OrderedDict(),
        "lock": RLock(),
        "ttl_seconds": _DEFAULT_TTL_SECONDS,
        "max_entries": _DEFAULT_MAX_ENTRIES,
    }
    setattr(state_type, _REPLY_CACHE_ATTRIBUTE, cache)
    return cache


def _capture_safely(state_type: type[Any], api: Any, payload: Any) -> None:
    try:
        message_id, reference_id = _reference_identity(payload)
        reply_message_id, reply_capture = _reply_capture_identity(payload)
        if api is None:
            return
        now = monotonic()
        if message_id and reference_id:
            _store_cached(_cache_for(state_type), api, message_id, reference_id, now)
        if reply_message_id and reply_capture is not None and reply_capture.available:
            _store_cached(
                _reply_cache_for(state_type),
                api,
                reply_message_id,
                reply_capture,
                now,
            )
    except Exception:
        return


def _store_cached(cache: dict[str, Any], api: Any, message_id: str, value: Any, now: float) -> None:
    key = (id(api), message_id)
    with cache["lock"]:
        entries = cache["entries"]
        _prune_expired(entries, now)
        entries[key] = (api, value, now + float(cache["ttl_seconds"]))
        entries.move_to_end(key)
        while len(entries) > int(cache["max_entries"]):
            entries.popitem(last=False)


def _resolve_cached(state_type: type[Any], api: Any, message_id: str) -> str:
    return str(_resolve_from_cache(_cache_for(state_type), api, message_id) or "").strip()


def _resolve_cached_reply(
    state_type: type[Any], api: Any, message_id: str
) -> QQReplyCapture | None:
    value = _resolve_from_cache(_reply_cache_for(state_type), api, message_id)
    if value is not None:
        capture = QQReplyCapture(
            reference_id=str(getattr(value, "reference_id", "") or "").strip(),
            content_projection=str(getattr(value, "content_projection", "") or ""),
            content_kind=str(getattr(value, "content_kind", "OTHER") or "OTHER"),
            attachment_count=int(getattr(value, "attachment_count", 0) or 0),
        )
        if capture.available:
            return capture
    return None


def _resolve_from_cache(cache: dict[str, Any], api: Any, message_id: str) -> Any:
    now = monotonic()
    key = (id(api), str(message_id or "").strip())
    with cache["lock"]:
        entries = cache["entries"]
        _prune_expired(entries, now)
        entry = entries.get(key)
        if entry is None:
            return ""
        cached_api, value, expires_at = entry
        if cached_api is not api or float(expires_at) <= now:
            entries.pop(key, None)
            return ""
        entries.move_to_end(key)
        return value


def _prune_expired(entries: OrderedDict[Any, Any], now: float) -> None:
    expired = [key for key, (_, _, expiry) in entries.items() if float(expiry) <= now]
    for key in expired:
        entries.pop(key, None)


def _reference_identity(payload: Any) -> tuple[str, str]:
    if not isinstance(payload, Mapping):
        return "", ""
    data = payload.get("d")
    if not isinstance(data, Mapping):
        data = payload
    message_id = str(data.get("id") or "").strip()
    scene = data.get("message_scene")
    if not message_id or not isinstance(scene, Mapping):
        return message_id, ""
    ext = scene.get("ext")
    if not isinstance(ext, (list, tuple)):
        return message_id, ""
    for item in ext:
        if not isinstance(item, str):
            continue
        name, separator, value = item.partition("=")
        if separator and name.strip() == "msg_idx" and value.strip():
            return message_id, normalize_qq_reference_index(value)
    return message_id, ""


def _reply_reference_identity(payload: Any) -> tuple[str, str]:
    """Extract only the target locator used by QQ quote events."""

    message_id, capture = _reply_capture_identity(payload)
    return message_id, capture.reference_id if capture is not None else ""


def _scene_ext_value(scene: Any, expected_name: str) -> str:
    if not isinstance(scene, Mapping):
        return ""
    ext = scene.get("ext")
    if not isinstance(ext, (list, tuple)):
        return ""
    for item in ext:
        if not isinstance(item, str):
            continue
        name, separator, value = item.partition("=")
        if separator and name.strip() == expected_name and value.strip():
            return normalize_qq_reference_index(value)
    return ""


def _element_field(element: Any, name: str) -> Any:
    return element.get(name) if isinstance(element, Mapping) else getattr(element, name, None)


def _quote_elements_projection(elements: list[Any]) -> tuple[str, str, list[str]]:
    reference_id = ""
    content = ""
    attachment_types: list[str] = []
    for element in elements:
        target = (
            _element_field(element, "msg_idx")
            or _element_field(element, "id")
            or _element_field(element, "message_id")
        )
        if target and not reference_id:
            reference_id = normalize_qq_reference_index(target)
        element_content = _element_field(element, "content")
        if element_content and not content:
            content = str(element_content).strip()[:500]
        attachments = _element_field(element, "attachments")
        if not isinstance(attachments, (list, tuple)):
            continue
        for attachment in attachments[:5]:
            content_type = _element_field(attachment, "content_type")
            attachment_types.append(str(content_type or "").strip().lower())
    return reference_id, content, attachment_types


def _reply_payload_data(payload: Any) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    data = payload.get("d")
    return data if isinstance(data, Mapping) else payload


def _direct_reply_reference_id(data: Mapping[str, Any]) -> str:
    reference = data.get("message_reference")
    if not isinstance(reference, Mapping):
        return ""
    return str(reference.get("message_id") or "").strip()


def _quoted_message_elements(data: Mapping[str, Any]) -> list[Any]:
    try:
        is_quoted_message = int(data.get("message_type") or 0) == 103
    except (TypeError, ValueError):
        is_quoted_message = False
    elements = data.get("msg_elements")
    if is_quoted_message and isinstance(elements, (list, tuple)):
        return list(elements)
    return []


def _quote_capture_projection(content: str, attachment_types: list[str]) -> tuple[str, str]:
    if any(item.startswith("image/") for item in attachment_types):
        return "IMAGE", content or "[图片]"
    if attachment_types:
        return "FILE", content or "[文件]"
    if content:
        return "TEXT", content
    return "OTHER", ""


def _reply_capture_identity(payload: Any) -> tuple[str, QQReplyCapture | None]:
    """Extract the target locator and bounded quote projection.

    QQ C2C/group quote payloads expose the quoted target as ``ref_msg_idx``
    in ``message_scene.ext`` or as ``msg_elements[*].msg_idx`` when
    ``message_type=103``. AstrBot 4.26.4-era qq-botpy discards both fields
    while constructing its message object. Other QQ payload revisions use
    ``id``/``message_id`` on the quoted element. Newer payloads may expose
    ``message_reference.message_id`` directly. Like Tencent's official
    quote middleware, keep quoted content as a fallback when the index misses.
    """

    data = _reply_payload_data(payload)
    if data is None:
        return "", None
    message_id = str(data.get("id") or "").strip()
    direct_reference_id = _direct_reply_reference_id(data)
    scene_reference_id = _scene_ext_value(data.get("message_scene"), "ref_msg_idx")
    quote_elements = _quoted_message_elements(data)
    element_reference_id, content, attachment_types = _quote_elements_projection(quote_elements)
    reference_id = prefer_qq_reference_locator(
        element_reference_id,
        scene_reference_id,
        direct_reference_id,
    )
    content_kind, projection = _quote_capture_projection(content, attachment_types)
    capture = QQReplyCapture(
        reference_id=reference_id,
        content_projection=projection,
        content_kind=content_kind,
        attachment_count=len(attachment_types),
    )
    return message_id, capture if capture.available else None


def _wrap_state_initializer(state_type: type[Any]) -> None:
    installed = getattr(state_type, "__init__", None)
    if not callable(installed) or _is_current_wrapper(installed):
        return
    original = _unwrap_previous_wrappers(installed)

    @wraps(original)
    def initializer(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        _wrap_parser_map(type(self), self)

    setattr(initializer, _WRAPPER_MARKER, _CURRENT_WRAPPER_TOKEN)
    state_type.__init__ = initializer


def _wrap_existing_state_parsers(state_type: type[Any]) -> None:
    with suppress(Exception):
        for candidate in gc.get_objects():
            with suppress(Exception):
                if isinstance(candidate, state_type):
                    _wrap_parser_map(state_type, candidate)


def _wrap_parser_map(state_type: type[Any], state: Any) -> int:
    parsers = getattr(state, "parsers", None)
    if not isinstance(parsers, dict):
        return 0
    return _wrap_parser_dict(state_type, state, parsers)


def _wrap_parser_dict(state_type: type[Any], state: Any, parsers: dict[str, Any]) -> int:
    wrapped = 0
    for key in _PARSER_KEYS:
        installed = parsers.get(key)
        if not callable(installed) or _is_current_wrapper(installed):
            continue
        original = _unwrap_previous_wrappers(installed)

        @wraps(original)
        def parser(payload: Any, *, _original: Any = original) -> Any:
            _capture_safely(state_type, getattr(state, "api", None), payload)
            return _original(payload)

        setattr(parser, _WRAPPER_MARKER, _CURRENT_WRAPPER_TOKEN)
        parsers[key] = parser
        wrapped += 1
    return wrapped


def _unwrap_parser_map(state: Any) -> None:
    for parsers in (
        getattr(state, "parsers", None),
        getattr(getattr(state, "_connection", None), "parser", None),
    ):
        if not isinstance(parsers, dict):
            continue
        for key in _PARSER_KEYS:
            installed = parsers.get(key)
            if callable(installed) and getattr(installed, _WRAPPER_MARKER, None):
                parsers[key] = _unwrap_previous_wrappers(installed)


def _unwrap_parser_dict(parsers: dict[str, Any]) -> None:
    for key in _PARSER_KEYS:
        installed = parsers.get(key)
        if callable(installed) and getattr(installed, _WRAPPER_MARKER, None):
            parsers[key] = _unwrap_previous_wrappers(installed)


def _is_current_wrapper(value: Any) -> bool:
    return getattr(value, _WRAPPER_MARKER, None) is _CURRENT_WRAPPER_TOKEN


def _unwrap_previous_wrappers(value: Any) -> Any:
    """Remove capture closures left by an earlier plugin module generation."""

    current = value
    for _ in range(16):
        if not getattr(current, _WRAPPER_MARKER, None):
            break
        wrapped = getattr(current, "__wrapped__", None)
        if not callable(wrapped):
            break
        current = wrapped
    return current


__all__ = [
    "QQReplyCapture",
    "bind_qq_reference_id_capture",
    "event_platform_reference_id",
    "event_reply_capture",
    "event_reply_reference_id",
    "install_qq_reference_id_capture",
    "uninstall_qq_reference_id_capture",
    "normalize_qq_reference_index",
    "prefer_qq_reference_locator",
    "qq_reference_locator_candidates",
]
