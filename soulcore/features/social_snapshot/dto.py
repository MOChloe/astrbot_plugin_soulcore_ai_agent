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
