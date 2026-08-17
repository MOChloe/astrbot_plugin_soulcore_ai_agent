from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .dto import EntryKind, SceneMode, SnapshotTheme, SocialSnapshotScene

MAX_PROJECTION_CHARS = 4096
MAX_PROJECTION_ITEMS = 64
MAX_ITEM_TEXT = 240
MAX_MEDIA_DESCRIPTION = 240
_THEME_LABELS = {
    SnapshotTheme.MOBILE_CHAT: "QQ",
    SnapshotTheme.WECHAT: "微信",
    SnapshotTheme.DINGTALK: "钉钉",
    SnapshotTheme.WEIBO_FEED: "微博",
    SnapshotTheme.X: "X",
    SnapshotTheme.XIAOHONGSHU: "小红书",
}
_MODE_LABELS = {
    SceneMode.PRIVATE_CHAT: "私聊",
    SceneMode.GROUP_CHAT: "群聊",
    SceneMode.FEED: "动态",
    SceneMode.NOTE: "笔记",
}
_URL = re.compile(r"(?i)\b(?:https?|file|data)://\S+")
_WINDOWS_PATH = re.compile(r"(?i)(?<!\w)[a-z]:[\\/](?:[^\s<>:\"|?*]+[\\/]?)+")
_POSIX_PATH = re.compile(r"(?<!\w)/(?:[^\s/]+/)+[^\s]*")
_LONG_HASH = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{32,}(?![0-9a-f])")


class ProjectionItemKind(StrEnum):
    SENT_CONTENT = "SENT_CONTENT"
    QUOTED_CONTENT = "QUOTED_CONTENT"
    UNSENT_DRAFT = "UNSENT_DRAFT"
    TIMESTAMP = "TIMESTAMP"


@dataclass(frozen=True, slots=True)
class SocialSnapshotProjectionItem:
    kind: ProjectionItemKind
    ordinal: int
    author: str
    text: str
    media_description: str
    time: str
    related_sent_ordinal: int | None = None

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind.value,
            "ordinal": self.ordinal,
            "author": self.author,
            "text": self.text,
            "media_description": self.media_description,
            "time": self.time,
        }
        if self.related_sent_ordinal is not None:
            payload["related_sent_ordinal"] = self.related_sent_ordinal
        return payload


@dataclass(frozen=True, slots=True)
class SocialSnapshotProjection:
    theme: str
    mode: str
    title: str
    disclosure: str
    items: tuple[SocialSnapshotProjectionItem, ...]
    omitted_items: int
    part_dimensions: tuple[tuple[int, int], ...]
    text: str

    def as_payload(self) -> dict[str, Any]:
        return {
            "theme": self.theme,
            "mode": self.mode,
            "title": self.title,
            "disclosure": self.disclosure,
            "items": [item.as_payload() for item in self.items],
            "omitted_items": self.omitted_items,
            "part_count": len(self.part_dimensions),
            "part_dimensions": [
                {"width": width, "height": height} for width, height in self.part_dimensions
            ],
            "text": self.text,
        }


def compile_social_snapshot_projection(
    scene: SocialSnapshotScene,
    *,
    media_descriptions: Mapping[str, str] | None = None,
    part_dimensions: Sequence[tuple[int, int]] = (),
) -> SocialSnapshotProjection:
    """Compile one stable semantic view from the exact normalized render scene."""

    if not isinstance(scene, SocialSnapshotScene):
        raise TypeError("projection requires a normalized SocialSnapshotScene")
    descriptions = dict(media_descriptions or {})
    forbidden = tuple(
        dict.fromkeys(
            [
                *(item.avatar_ref for item in scene.participants if item.avatar_ref),
                *(item.media_ref for item in scene.entries if item.media_ref),
            ]
        )
    )
    candidates = _projection_items(scene, descriptions, forbidden)
    draft = _draft_item(scene, forbidden)
    limit = MAX_PROJECTION_ITEMS - (1 if draft is not None else 0)
    selected = list(candidates[:limit])
    omitted = max(0, len(candidates) - len(selected))
    if draft is not None:
        selected.append(draft)
    dimensions = tuple((int(width), int(height)) for width, height in part_dimensions)
    text, rendered_items, omitted = _render_projection_text(scene, selected, omitted, forbidden)
    return SocialSnapshotProjection(
        theme=scene.theme.value,
        mode=scene.mode.value,
        title=_safe_text(scene.title, MAX_ITEM_TEXT, forbidden),
        disclosure=_safe_text(scene.disclosure, 40, forbidden),
        items=rendered_items,
        omitted_items=omitted,
        part_dimensions=dimensions,
        text=text,
    )


def _projection_items(
    scene: SocialSnapshotScene,
    descriptions: Mapping[str, str],
    forbidden: Sequence[str],
) -> list[SocialSnapshotProjectionItem]:
    items: list[SocialSnapshotProjectionItem] = []
    sent_ordinal = 0
    for entry in scene.entries:
        if entry.kind is EntryKind.TIMESTAMP:
            items.append(
                SocialSnapshotProjectionItem(
                    kind=ProjectionItemKind.TIMESTAMP,
                    ordinal=len(items) + 1,
                    author="",
                    text=_safe_text(entry.text or entry.time, MAX_ITEM_TEXT, forbidden),
                    media_description="",
                    time=_safe_text(entry.time, 40, forbidden),
                )
            )
            continue
        sent_ordinal += 1
        author = scene.participant(entry.author_id or "").display_name
        items.append(
            SocialSnapshotProjectionItem(
                kind=ProjectionItemKind.SENT_CONTENT,
                ordinal=sent_ordinal,
                author=_safe_text(author, 80, forbidden),
                text=_safe_text(entry.text, MAX_ITEM_TEXT, forbidden),
                media_description=_media_description(entry.media_ref, descriptions, forbidden),
                time=_safe_text(entry.time, 40, forbidden),
            )
        )
        if entry.quote is not None:
            items.append(
                SocialSnapshotProjectionItem(
                    kind=ProjectionItemKind.QUOTED_CONTENT,
                    ordinal=len(items) + 1,
                    author=_safe_text(entry.quote.sender, 80, forbidden),
                    text=_safe_text(entry.quote.text, MAX_ITEM_TEXT, forbidden),
                    media_description=_safe_text(
                        entry.quote.media_label, MAX_MEDIA_DESCRIPTION, forbidden
                    ),
                    time=_safe_text(entry.quote.time, 40, forbidden),
                    related_sent_ordinal=sent_ordinal,
                )
            )
    return items


def _draft_item(
    scene: SocialSnapshotScene, forbidden: Sequence[str]
) -> SocialSnapshotProjectionItem | None:
    if not scene.draft:
        return None
    return SocialSnapshotProjectionItem(
        kind=ProjectionItemKind.UNSENT_DRAFT,
        ordinal=1,
        author="",
        text=_safe_text(scene.draft, MAX_ITEM_TEXT, forbidden),
        media_description="",
        time="",
    )


def _media_description(
    media_ref: str | None,
    descriptions: Mapping[str, str],
    forbidden: Sequence[str],
) -> str:
    if media_ref is None:
        return ""
    description = descriptions.get(media_ref, "")
    if not str(description or "").strip():
        return "媒体内容无可用来源描述"
    return _safe_text(description, MAX_MEDIA_DESCRIPTION, forbidden)


def _safe_text(value: object, maximum: int, forbidden: Sequence[str]) -> str:
    text = " ".join(str(value or "").split())
    for secret in sorted((item for item in forbidden if item), key=len, reverse=True):
        text = text.replace(secret, "[受控媒体]")
    text = _URL.sub("[外部定位已移除]", text)
    text = _WINDOWS_PATH.sub("[本地定位已移除]", text)
    text = _POSIX_PATH.sub("[本地定位已移除]", text)
    text = _LONG_HASH.sub("[哈希已移除]", text)
    if len(text) <= maximum:
        return text
    return text[: max(0, maximum - 1)].rstrip() + "…"


def _render_projection_text(
    scene: SocialSnapshotScene,
    items: Sequence[SocialSnapshotProjectionItem],
    omitted: int,
    forbidden: Sequence[str],
) -> tuple[str, tuple[SocialSnapshotProjectionItem, ...], int]:
    header = (
        f"社交截图内容（{scene.disclosure}；界面：{_THEME_LABELS[scene.theme]}"
        f"{_MODE_LABELS[scene.mode]}；标题：{_safe_text(scene.title, 120, forbidden)}）"
    )
    draft_lines = [
        _render_item(item) for item in items if item.kind is ProjectionItemKind.UNSENT_DRAFT
    ]
    tail = draft_lines or ["未发送草稿：无。"]
    available = MAX_PROJECTION_CHARS - len("\n".join([header, *tail])) - 2
    body: list[str] = []
    rendered_items: list[SocialSnapshotProjectionItem] = []
    for item in items:
        if item.kind is ProjectionItemKind.UNSENT_DRAFT:
            continue
        line = _render_item(item)
        if len(line) + 1 > available:
            omitted += 1
            continue
        body.append(line)
        rendered_items.append(item)
        available -= len(line) + 1
    if omitted:
        marker = f"还有{omitted}项未在这份简要内容中展开。"
        if len(marker) + 1 <= available:
            body.append(marker)
    rendered = "\n".join([header, *(body or ["已发送／已发布内容：无。"]), *tail])
    draft_items = [item for item in items if item.kind is ProjectionItemKind.UNSENT_DRAFT]
    return rendered[:MAX_PROJECTION_CHARS], (*rendered_items, *draft_items), omitted


def _render_item(item: SocialSnapshotProjectionItem) -> str:
    if item.kind is ProjectionItemKind.TIMESTAMP:
        return f"时间标记：{item.text}。"
    if item.kind is ProjectionItemKind.UNSENT_DRAFT:
        return f"未发送草稿（明确未发送、不是已发生对话）：{item.text}"
    content = item.text or "（无文字）"
    if item.media_description:
        content += f"；媒体：{item.media_description}"
    if item.kind is ProjectionItemKind.QUOTED_CONTENT:
        return (
            f"引用内容（属于第{item.related_sent_ordinal}条，不计作新发送消息）："
            f"{item.author}：{content}"
        )
    timing = f"（{item.time}）" if item.time else ""
    return f"已发送／已发布内容{item.ordinal}{timing}：{item.author}：{content}"


__all__ = [
    "MAX_PROJECTION_CHARS",
    "MAX_PROJECTION_ITEMS",
    "ProjectionItemKind",
    "SocialSnapshotProjection",
    "SocialSnapshotProjectionItem",
    "compile_social_snapshot_projection",
]
