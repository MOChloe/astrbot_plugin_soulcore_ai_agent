from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SnapshotTheme(StrEnum):
    MOBILE_CHAT = "mobile_chat"
    WECHAT = "wechat"
    DINGTALK = "dingtalk"
    WEIBO_FEED = "weibo_feed"
    X = "x"
    XIAOHONGSHU = "xiaohongshu"


class SceneMode(StrEnum):
    PRIVATE_CHAT = "private_chat"
    GROUP_CHAT = "group_chat"
    FEED = "feed"
    NOTE = "note"


class EntryKind(StrEnum):
    TIMESTAMP = "timestamp"
    MESSAGE = "message"
    IMAGE = "image"
    FILE = "file"
    POST = "post"
    COMMENT = "comment"
    REPOST = "repost"


class ParticipantSide(StrEnum):
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class CompactPerson:
    id: str
    name: str
    avatar: str | None = None
    side: str = ParticipantSide.LEFT
    badge: str = ""
    color: str = "#7f8c9a"


@dataclass(frozen=True, slots=True)
class CompactQuote:
    sender: str
    text: str = ""
    media_label: str = ""
    time: str = ""


@dataclass(frozen=True, slots=True)
class CompactItem:
    k: str
    by: str | None = None
    text: str = ""
    time: str = ""
    media: str | None = None
    quote: CompactQuote | None = None


@dataclass(frozen=True, slots=True)
class CompactUi:
    subtitle: str = ""
    clock: str = "00:00"
    battery_percent: int = 100
    battery_charging: bool = False
    width: int = 873
    height: int | None = None
    segment_height: int = 1920


@dataclass(frozen=True, slots=True)
class SocialSnapshotRequest:
    theme: str
    mode: str
    title: str
    people: tuple[CompactPerson, ...]
    items: tuple[CompactItem, ...]
    draft: str = ""
    ui: CompactUi = field(default_factory=CompactUi)


@dataclass(frozen=True, slots=True)
class SnapshotParticipant:
    participant_id: str
    display_name: str
    avatar_ref: str | None
    side: ParticipantSide
    badge: str
    color: str


@dataclass(frozen=True, slots=True)
class SnapshotQuote:
    sender: str
    text: str
    media_label: str
    time: str


@dataclass(frozen=True, slots=True)
class SnapshotEntry:
    kind: EntryKind
    author_id: str | None
    text: str
    time: str
    media_ref: str | None
    quote: SnapshotQuote | None


@dataclass(frozen=True, slots=True)
class SnapshotUi:
    subtitle: str
    clock: str
    battery_percent: int
    battery_charging: bool
    width: int
    height: int | None
    segment_height: int


@dataclass(frozen=True, slots=True)
class SocialSnapshotScene:
    theme: SnapshotTheme
    mode: SceneMode
    title: str
    participants: tuple[SnapshotParticipant, ...]
    entries: tuple[SnapshotEntry, ...]
    draft: str
    ui: SnapshotUi
    disclosure: str
    request_fingerprint: str

    def participant(self, participant_id: str) -> SnapshotParticipant:
        for participant in self.participants:
            if participant.participant_id == participant_id:
                return participant
        raise KeyError(participant_id)


class SocialSnapshotErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    UNSUPPORTED_THEME = "UNSUPPORTED_THEME"
    UNSUPPORTED_MODE = "UNSUPPORTED_MODE"
    UNSUPPORTED_ENTRY = "UNSUPPORTED_ENTRY"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    ASSET_MISSING = "ASSET_MISSING"
    ASSET_INVALID = "ASSET_INVALID"
    ASSET_TOO_LARGE = "ASSET_TOO_LARGE"
    FONT_UNAVAILABLE = "FONT_UNAVAILABLE"
    RENDER_FAILED = "RENDER_FAILED"


class SocialSnapshotError(ValueError):
    """A redaction-safe domain or rendering failure."""

    def __init__(self, code: SocialSnapshotErrorCode, message: str) -> None:
        self.code = code
        super().__init__(message)


def invalid(message: str) -> SocialSnapshotError:
    return SocialSnapshotError(SocialSnapshotErrorCode.INVALID_REQUEST, message)


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
