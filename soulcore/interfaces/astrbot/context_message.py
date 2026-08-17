"""Convert AstrBot events into stable context-ledger payloads."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from ...contracts.message_reference import safe_model_identity

PLATFORM_EMOJI_METADATA = "PLATFORM_EMOJI_METADATA"
PLATFORM_STICKER_SUMMARY = "PLATFORM_STICKER_SUMMARY"

_SUMMARY_MARKERS = (
    "动画表情",
    "商城表情",
    "表情包",
    "贴纸",
    "marketface",
    "sticker",
)
_EMOJI_FIELDS = (
    "emoji_id",
    "emojiId",
    "emoji_package_id",
    "emojiPackageId",
    "sticker_id",
    "stickerId",
)


def image_sticker_evidence(event: Any, image_components: Sequence[Any]) -> list[tuple[str, ...]]:
    """Return privacy-safe sticker evidence aligned with normalized image components.

    AstrBot's normalized ``Image`` component intentionally exposes very few
    fields.  Some adapters retain richer OneBot data on ``raw_message``.  Only
    closed evidence labels leave this boundary; raw summaries, IDs, keys and
    signed locators never do.
    """

    raw_evidence = [_value_evidence(item) for item in _raw_image_segments(event)]
    result: list[tuple[str, ...]] = []
    for index, component in enumerate(image_components):
        evidence = list(_value_evidence(component))
        if index < len(raw_evidence):
            evidence.extend(raw_evidence[index])
        result.append(tuple(dict.fromkeys(evidence)))
    return result


def _raw_image_segments(event: Any) -> list[Any]:
    message = getattr(event, "message_obj", None)
    raw = getattr(message, "raw_message", None)
    payloads = (raw, getattr(raw, "raw_data", None))
    for payload in payloads:
        segments = _field(payload, "message")
        if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes, bytearray)):
            continue
        images = [item for item in segments if str(_field(item, "type") or "").lower() == "image"]
        if images:
            return images
    return []


def _value_evidence(value: Any) -> tuple[str, ...]:
    data = _field(value, "data")
    sources = (value, data) if data is not None else (value,)
    evidence: list[str] = []
    for source in sources:
        summary = str(_field(source, "summary") or "").strip().casefold()
        if summary and any(marker in summary for marker in _SUMMARY_MARKERS):
            evidence.append(PLATFORM_STICKER_SUMMARY)
        if any(_present(_field(source, name)) for name in _EMOJI_FIELDS):
            evidence.append(PLATFORM_EMOJI_METADATA)
    return tuple(dict.fromkeys(evidence))


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _present(value: Any) -> bool:
    if value in (None, "", 0, "0", False):
        return False
    return bool(str(value).strip())


_SAFE_FIELDS = (
    "text",
    "url",
    "file",
    "name",
    "id",
    "qq",
    "sender_id",
    "message_id",
    "message_str",
)

_ASTRBOT_COMPONENT_PLACEHOLDER = re.compile(r"\[用户发送了(?:一个|一条|一张|一段)[^\]\r\n]*组件\]")
_QQ_INLINE_MENTION = re.compile(r"<@!?([A-Za-z0-9][A-Za-z0-9._:-]{1,159})>")
_QQ_OFFICIAL_PLATFORMS = frozenset({"qq_official", "qq_official_webhook"})
_VOICE_COMPONENT_KINDS = frozenset({"record", "audio", "voice"})
VOICE_MARKER = "（语音）"


def event_context_payload(event: Any) -> dict[str, Any]:
    """Return structured components without persisting opaque platform objects."""

    raw_components = _event_components(event)

    components: list[dict[str, Any]] = []
    image_urls: list[str] = []
    image_components: list[Any] = []
    inbound_media: list[dict[str, Any]] = []
    inbound_voice: list[dict[str, Any]] = []
    sensitive_locators: list[str] = []
    projected: list[str] = []
    ordered_projection: list[str] = []
    normalized_items: list[dict[str, Any]] = []
    self_id = _event_self_id(event)
    is_group = _event_is_group(event)
    for component in raw_components:
        for item in _normalized_component_payloads(event, component):
            normalized_items.append(item)
            projection = _project_component(item, component, self_id=self_id, is_group=is_group)
            component_index = len(components)
            ordered_projection.append(projection.text)
            if projection.text:
                projected.append(projection.text)
            if projection.image:
                image_components.append(component)
                image_urls.append(projection.locator)
            if projection.media:
                inbound_media.append(projection.media)
            if projection.voice:
                inbound_voice.append(
                    {
                        "component": component,
                        "locator": projection.locator,
                        "component_index": component_index,
                    }
                )
            if projection.locator:
                sensitive_locators.append(projection.locator)
            components.append(
                {"type": "plain", "text": VOICE_MARKER}
                if projection.voice
                else _ledger_component(item)
            )

    plain = _semantic_plain_text(event, normalized_items, projected, sensitive_locators)
    payload = {
        "plain_text": plain,
        "components": components,
        "image_urls": image_urls,
        "image_components": image_components,
        "image_sticker_evidence": image_sticker_evidence(event, image_components),
        # Live platform objects and locators exist only until the immediate
        # ingest pass.  They are deliberately kept outside ``components`` so
        # neither the ledger nor later context assembly can persist them.  The
        # parallel sticker evidence contains closed labels only, never raw IDs,
        # summaries, keys or signed URLs.
        "inbound_media": inbound_media,
    }
    if inbound_voice:
        # These two fields are live admission inputs only.  The component itself
        # never enters ``components`` or ``inbound_media`` and is removed before
        # the immutable conversation ledger append.
        payload["inbound_voice"] = inbound_voice
        payload["voice_ordered_projection"] = ordered_projection
    return payload


def _semantic_plain_text(
    event: Any,
    component_items: list[dict[str, Any]],
    projected: list[str],
    sensitive_locators: list[str],
) -> str:
    plain = str(getattr(event, "message_str", "") or "").strip()
    for locator in sensitive_locators:
        plain = plain.replace(locator, "[受控媒体资产]")
    contains_platform_placeholder = bool(_ASTRBOT_COMPONENT_PLACEHOLDER.search(plain))
    plain = _ASTRBOT_COMPONENT_PLACEHOLDER.sub("", plain).strip()
    if _is_qq_official_group_event(event):
        plain = _QQ_INLINE_MENTION.sub("[提及一位群成员]", plain)
    semantic_projection = " ".join(part for part in projected if part).strip()
    # AstrBot's display fallback may contain implementation labels such as
    # "at组件" and platform IDs.  Control components are always rebuilt from
    # SoulCore's semantic projection; ordinary mixed media keeps typed text.
    control_kinds = {str(item.get("type") or "").lower() for item in component_items}
    requires_rebuild = bool(control_kinds.intersection({"at", "atall", "face"}))
    if semantic_projection and (not plain or contains_platform_placeholder or requires_rebuild):
        return semantic_projection
    return plain


def _normalized_component_payloads(event: Any, component: Any) -> tuple[dict[str, Any], ...]:
    item = _component_payload(component)
    kind = str(item.get("type") or "").lower()
    text = str(item.get("text") or "")
    if (
        kind not in {"plain", "text"}
        or not text
        or not _is_qq_official_group_event(event)
        or _QQ_INLINE_MENTION.search(text) is None
    ):
        return (item,)

    result: list[dict[str, Any]] = []
    cursor = 0
    for match in _QQ_INLINE_MENTION.finditer(text):
        if prefix := text[cursor : match.start()]:
            result.append({"type": kind, "text": prefix})
        result.append(
            {
                "type": "at",
                "qq": match.group(1),
                "mention_role": "group_member",
            }
        )
        cursor = match.end()
    if suffix := text[cursor:]:
        result.append({"type": kind, "text": suffix})
    return tuple(result)


def _event_platform_name(event: Any) -> str:
    getter = getattr(event, "get_platform_name", None)
    try:
        platform_name = getter() if callable(getter) else ""
    except Exception:
        platform_name = ""
    return str(platform_name or "").strip().lower().replace("-", "_")


def _is_qq_official_group_event(event: Any) -> bool:
    return _event_platform_name(event) in _QQ_OFFICIAL_PLATFORMS and _event_is_group(event)


@dataclass(frozen=True, slots=True)
class _ProjectedComponent:
    text: str = ""
    locator: str = ""
    image: bool = False
    media: dict[str, Any] | None = None
    voice: bool = False


def _event_components(event: Any) -> list[Any]:
    getter = getattr(event, "get_messages", None)
    if callable(getter):
        try:
            components = list(getter() or [])
        except Exception:
            components = []
        if components:
            return components
    message_obj = getattr(event, "message_obj", None)
    return list(getattr(message_obj, "message", None) or [])


def _project_component(
    item: Mapping[str, Any], component: Any, *, self_id: str = "", is_group: bool = False
) -> _ProjectedComponent:
    kind = str(item.get("type") or "")
    if kind in {"plain", "text"} and item.get("text"):
        return _ProjectedComponent(text=str(item["text"]))
    if kind == "image":
        return _ProjectedComponent(
            text="[对方发送了一张图片]",
            locator=_media_locator(item),
            image=True,
        )
    if kind in _VOICE_COMPONENT_KINDS:
        return _ProjectedComponent(
            text=VOICE_MARKER,
            locator=_media_locator(item),
            voice=True,
        )
    if kind in {"file", "video"}:
        return _media_projection(kind, item, component)
    if kind == "reply":
        # Native reply content is resolved against SoulCore's own message-fragment
        # index.  It must not masquerade as newly typed player text or enter the
        # turn-buffer classifier through the platform display text.
        return _ProjectedComponent()
    if kind in {"forward", "node", "nodes"}:
        return _ProjectedComponent(text="[收到一条转发消息；SoulCore 0.7 不展开其中内容]")
    if kind == "atall":
        return _ProjectedComponent(text="[提及全体成员]")
    if kind == "at":
        target = str(item.get("qq") or item.get("sender_id") or "").strip()
        label = _mention_label(target, self_id=self_id, is_group=is_group)
        return _ProjectedComponent(text=f"[提及{label}]")
    if kind == "face":
        return _ProjectedComponent(text="[发送了一个平台表情]")
    return _ProjectedComponent()


def _mention_label(target: str, *, self_id: str, is_group: bool) -> str:
    if self_id and target == self_id:
        return "本人"
    return "一位群成员" if is_group else "一位联系人"


def _media_projection(
    kind: str,
    item: Mapping[str, Any],
    component: Any,
) -> _ProjectedComponent:
    locator = _media_locator(item)
    media_kind = kind
    name = _safe_display_name(item)
    if media_kind == "file":
        name = name or "未命名文件"
        text = f"[对方发送了文件：{name}]"
    elif media_kind == "video":
        text = "[对方发送了一段视频]"
    return _ProjectedComponent(
        text=text,
        locator=locator,
        media={
            "kind": media_kind,
            "component": component,
            "locator": locator,
            "display_name": name,
        },
    )


def _media_locator(item: Mapping[str, Any]) -> str:
    return str(item.get("url") or item.get("file") or "").strip()


def _event_self_id(event: Any) -> str:
    getter = getattr(event, "get_self_id", None)
    try:
        value = getter() if callable(getter) else ""
    except Exception:
        value = ""
    if value in (None, ""):
        value = getattr(getattr(event, "message_obj", None), "self_id", "")
    return str(value or "").strip()


def _event_is_group(event: Any) -> bool:
    getter = getattr(event, "get_message_type", None)
    try:
        value = getter() if callable(getter) else ""
    except Exception:
        value = ""
    if not value:
        value = getattr(getattr(event, "message_obj", None), "type", "")
    return "group" in str(getattr(value, "value", value) or "").casefold()


def event_sender(event: Any) -> tuple[str, str]:
    message_obj = getattr(event, "message_obj", None)
    sender = getattr(message_obj, "sender", None)
    sender_id = _first(sender, "user_id", "id", "qq", "openid")
    visible_name = _first_model_identity(
        sender,
        "nickname",
        "display_name",
        "card",
        "full_name",
        "name",
        "username",
    )
    if not visible_name:
        visible_name = _qq_official_author_username(event, message_obj)
    if not sender_id:
        sender_id = str(getattr(message_obj, "session_id", "") or "").strip()
    fallback = "一位群成员" if _event_is_group(event) else "对方"
    return sender_id, visible_name or fallback


def _qq_official_author_username(event: Any, message_obj: Any) -> str:
    """Read QQ's optional live username without retaining the raw platform object."""

    if _event_platform_name(event) not in _QQ_OFFICIAL_PLATFORMS:
        return ""
    raw_message = getattr(message_obj, "raw_message", None)
    author = (
        raw_message.get("author")
        if isinstance(raw_message, Mapping)
        else getattr(raw_message, "author", None)
    )
    return _first_model_identity(author, "username")


async def live_instance_display_names(
    context: Any,
    items: list[dict[str, Any]],
    *,
    timeout_seconds: float = 5.0,
) -> dict[tuple[str, str, str], str]:
    """Read mutable QQ names from adapters on every call, without caching."""

    requested: dict[str, set[str]] = {}
    for item in items:
        platform_id = str(item.get("platform_id") or "").strip()
        kind = str(
            item.get("scope") or item.get("session_kind") or item.get("message_type") or ""
        ).lower()
        if platform_id:
            requested.setdefault(platform_id, set()).add(
                "group" if kind in {"group", "guild"} else "private"
            )

    if not requested:
        return {}
    parts = await asyncio.gather(
        *(
            _read_platform_names(
                context,
                platform_id,
                scopes,
                timeout_seconds=timeout_seconds,
            )
            for platform_id, scopes in requested.items()
        )
    )
    return {key: value for part in parts for key, value in part.items()}


async def _read_platform_names(
    context: Any,
    platform_id: str,
    scopes: set[str],
    *,
    timeout_seconds: float,
) -> dict[tuple[str, str, str], str]:
    getter = getattr(context, "get_platform_inst", None)
    try:
        platform = getter(platform_id) if callable(getter) else None
    except Exception:
        return {}
    call_action = getattr(getattr(platform, "bot", None), "call_action", None)
    if not callable(call_action):
        return {}
    action_map = {
        "group": "get_group_list",
        "private": "get_friend_list",
    }
    parts = await asyncio.gather(
        *(
            _read_action_names(
                call_action,
                platform_id,
                scope,
                action_map[scope],
                timeout_seconds,
            )
            for scope in ("group", "private")
            if scope in scopes
        )
    )
    return {key: value for part in parts for key, value in part.items()}


async def _read_action_names(
    call_action: Any,
    platform_id: str,
    scope: str,
    action: str,
    timeout_seconds: float,
) -> dict[tuple[str, str, str], str]:
    try:
        async with asyncio.timeout(timeout_seconds):
            rows = await call_action(action)
    except Exception:
        return {}
    rows = _directory_rows(rows)
    result: dict[tuple[str, str, str], str] = {}
    for row in rows:
        target, name = _directory_identity(scope, row)
        if target and name:
            result[(platform_id, scope, target)] = name
    return result


def _directory_rows(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        value = value.get("data") or value.get("groups") or value.get("friends") or []
    if not isinstance(value, (list, tuple)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _directory_identity(scope: str, row: Mapping[str, Any]) -> tuple[str, str]:
    if scope == "group":
        target = row.get("group_id") or row.get("id")
        name = row.get("group_name") or row.get("name")
    else:
        target = row.get("user_id") or row.get("id")
        name = row.get("remark") or row.get("nickname") or row.get("name")
    return str(target or "").strip(), str(name or "").strip()


def instance_identity_labels(
    *,
    scope: str,
    target_id: str,
    instance_id: str,
    display_name: str = "",
) -> dict[str, str]:
    """Build stable human-readable labels without changing instance identity."""

    normalized_scope = "group" if str(scope).lower() == "group" else "private"
    target = str(target_id or "").strip()
    stable_instance = str(instance_id or "").strip()
    current_display_name = str(display_name or "").strip()
    if normalized_scope == "group":
        identifier = target or stable_instance
        if not target:
            identifier_label = f"会话ID：{identifier}"
        elif target.isdigit():
            identifier_label = f"群号：{identifier}"
        else:
            identifier_label = f"平台会话ID：{identifier}"
    else:
        identifier = target or stable_instance
        identifier_label = (
            f"QQ号：{identifier}" if identifier.isdigit() else f"QQ/用户ID：{identifier}"
        )

    identity_label = (
        f"{current_display_name}（{identifier_label}）"
        if current_display_name and current_display_name != identifier
        else identifier_label
    )
    return {
        "target_id": target or identifier,
        "identifier_label": identifier_label,
        "identity_label": identity_label,
        "display_name": current_display_name or identifier_label,
    }


def _component_payload(component: Any) -> dict[str, Any]:
    kind = component.__class__.__name__.lower()
    result: dict[str, Any] = {"type": kind}
    if is_dataclass(component):
        try:
            raw = asdict(component)  # type: ignore[arg-type]
        except Exception:
            raw = {}
        for key in _SAFE_FIELDS:
            value = raw.get(key)
            if value not in (None, "", [], {}):
                result[key] = _json_safe(value)
    for key in _SAFE_FIELDS:
        if key in result:
            continue
        value = getattr(component, key, None)
        if value not in (None, "", [], {}):
            result[key] = _json_safe(value)
    return result


def _ledger_component(item: Mapping[str, Any]) -> dict[str, Any]:
    """Remove transport locators while retaining a stable text projection."""

    kind = str(item.get("type") or "").lower()
    result = dict(item)
    if kind in {"image", "record", "audio", "voice", "file", "video"}:
        result.pop("url", None)
        result.pop("file", None)
        if kind == "file":
            if name := _safe_display_name(item):
                result["name"] = name
        else:
            # Adapter-generated image/video names are transport labels such as
            # ``media_image_<opaque>.gif`` rather than user-visible filenames.
            # Only an actual file attachment has filename semantics in chat.
            result.pop("name", None)
    return result


def is_voice_component_kind(value: object) -> bool:
    """Return whether one normalized AstrBot component is inbound speech."""

    return str(value or "").strip().lower() in _VOICE_COMPONENT_KINDS


def _safe_display_name(item: Mapping[str, Any]) -> str:
    raw = str(item.get("name") or "").strip()
    if not raw:
        raw = str(item.get("file") or "").strip()
    if not raw:
        return ""
    # ``PurePath`` on Windows-shaped input is platform-dependent.  Splitting
    # both separators first keeps only the leaf without interpreting the path.
    leaf = re.split(r"[\\/]", raw)[-1]
    leaf = "".join(ch for ch in leaf if ch >= " " and ch != "\x7f").strip()
    return leaf[:128]


def _first(value: Any, *names: str) -> str:
    for name in names:
        item = value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
        if item not in (None, ""):
            return str(item).strip()
    return ""


def _first_model_identity(value: Any, *names: str) -> str:
    for name in names:
        if visible_name := safe_model_identity(_first(value, name)):
            return visible_name
    return ""


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)
