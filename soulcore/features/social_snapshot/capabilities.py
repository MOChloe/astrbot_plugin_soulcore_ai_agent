from __future__ import annotations

from dataclasses import dataclass

from .dto import EntryKind, SceneMode, SnapshotTheme


@dataclass(frozen=True, slots=True)
class ThemeCapability:
    modes: frozenset[SceneMode]
    entry_kinds: frozenset[EntryKind]
    supports_quotes: bool
    supports_draft: bool
    supports_auto_height: bool


CHAT_ENTRIES = frozenset({EntryKind.TIMESTAMP, EntryKind.MESSAGE, EntryKind.IMAGE})
FEED_ENTRIES = frozenset({EntryKind.POST, EntryKind.COMMENT, EntryKind.REPOST})

THEME_CAPABILITIES: dict[SnapshotTheme, ThemeCapability] = {
    SnapshotTheme.MOBILE_CHAT: ThemeCapability(
        modes=frozenset({SceneMode.PRIVATE_CHAT, SceneMode.GROUP_CHAT}),
        entry_kinds=CHAT_ENTRIES,
        supports_quotes=True,
        supports_draft=True,
        supports_auto_height=True,
    ),
    SnapshotTheme.WECHAT: ThemeCapability(
        modes=frozenset({SceneMode.PRIVATE_CHAT, SceneMode.GROUP_CHAT}),
        entry_kinds=CHAT_ENTRIES,
        supports_quotes=True,
        supports_draft=True,
        supports_auto_height=False,
    ),
    SnapshotTheme.DINGTALK: ThemeCapability(
        modes=frozenset({SceneMode.PRIVATE_CHAT, SceneMode.GROUP_CHAT}),
        entry_kinds=CHAT_ENTRIES | {EntryKind.FILE},
        supports_quotes=True,
        supports_draft=True,
        supports_auto_height=False,
    ),
    SnapshotTheme.WEIBO_FEED: ThemeCapability(
        modes=frozenset({SceneMode.FEED}),
        entry_kinds=FEED_ENTRIES,
        supports_quotes=False,
        supports_draft=False,
        supports_auto_height=False,
    ),
    SnapshotTheme.X: ThemeCapability(
        modes=frozenset({SceneMode.FEED}),
        entry_kinds=FEED_ENTRIES,
        supports_quotes=False,
        supports_draft=False,
        supports_auto_height=False,
    ),
    SnapshotTheme.XIAOHONGSHU: ThemeCapability(
        modes=frozenset({SceneMode.NOTE}),
        entry_kinds=frozenset({EntryKind.POST, EntryKind.COMMENT}),
        supports_quotes=False,
        supports_draft=True,
        supports_auto_height=False,
    ),
}
