from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from enum import StrEnum
from typing import TypeVar

from .capabilities import THEME_CAPABILITIES
from .dto import (
    CompactItem,
    CompactPerson,
    CompactQuote,
    CompactUi,
    EntryKind,
    ParticipantSide,
    SceneMode,
    SnapshotEntry,
    SnapshotParticipant,
    SnapshotQuote,
    SnapshotTheme,
    SnapshotUi,
    SocialSnapshotRequest,
    SocialSnapshotScene,
)
from .errors import SocialSnapshotError, SocialSnapshotErrorCode, invalid

MAX_PARTICIPANTS = 12
MAX_ENTRIES = 60
MAX_ITEM_TEXT = 500
MAX_TOTAL_TEXT = 8_000
MAX_TITLE = 120
MAX_DRAFT = 500
MAX_ASSET_REFS = 12
MAX_MEDIA_REFS = 5
MAX_REQUEST_BYTES = 32 * 1024
MIN_WIDTH = 640
MAX_WIDTH = 1080
MIN_HEIGHT = 960
MAX_HEIGHT = 2400
MAX_PARTS = 5
MAX_PART_PIXELS = 4_000_000
MAX_TOTAL_PIXELS = 16_000_000
DISCLOSURE = "AI演绎"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_ASSET_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_EnumT = TypeVar("_EnumT", bound=StrEnum)


def _enum(enum_type: type[_EnumT], raw: str) -> _EnumT:
    try:
        return enum_type(raw)
    except (TypeError, ValueError) as exc:
        raise invalid("request contains an unsupported enum value") from exc


def _bounded_text(value: object, *, label: str, maximum: int, required: bool = False) -> str:
    if not isinstance(value, str):
        raise invalid(f"{label} must be text")
    text = value.strip()
    if required and not text:
        raise invalid(f"{label} is required")
    if len(text) > maximum or any(
        ord(character) < 32 and character not in "\n\t" for character in text
    ):
        raise invalid(f"{label} is invalid or too long")
    return text


def _opaque_ref(value: object, *, label: str, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not _ASSET_REF.fullmatch(value):
        raise invalid(
            f"{label} must be a controlled media asset ID; omit it to use the default visual"
        )
    return value


def _participant(raw: CompactPerson) -> SnapshotParticipant:
    if not isinstance(raw, CompactPerson):
        raise invalid("person type is invalid")
    participant_id = _bounded_text(raw.id, label="person id", maximum=64, required=True)
    if not _IDENTIFIER.fullmatch(participant_id):
        raise invalid("person id is invalid")
    side = _enum(ParticipantSide, raw.side)
    if not isinstance(raw.color, str) or not _COLOR.fullmatch(raw.color):
        raise invalid("person color is invalid")
    return SnapshotParticipant(
        participant_id=participant_id,
        display_name=_bounded_text(raw.name, label="person name", maximum=80, required=True),
        avatar_ref=_opaque_ref(raw.avatar, label="scene.people[].avatar"),
        side=side,
        badge=_bounded_text(raw.badge, label="person badge", maximum=24),
        color=raw.color.lower(),
    )


def _entry_author(kind: EntryKind, author_id: object, people: set[str]) -> str | None:
    if kind is EntryKind.TIMESTAMP:
        return None
    if not isinstance(author_id, str) or author_id not in people:
        raise invalid("entry author is unknown")
    return author_id


def _entry_quote(raw: object) -> SnapshotQuote | None:
    if raw is None:
        return None
    if not isinstance(raw, CompactQuote):
        raise invalid("quote type is invalid")
    quote = SnapshotQuote(
        sender=_bounded_text(raw.sender, label="quote sender", maximum=80, required=True),
        text=_bounded_text(raw.text, label="quote text", maximum=240),
        media_label=_bounded_text(raw.media_label, label="quote media label", maximum=80),
        time=_bounded_text(raw.time, label="quote time", maximum=40),
    )
    if not quote.text and not quote.media_label:
        raise invalid("quote content is empty")
    return quote


def _validate_entry_content(kind: EntryKind, text: str, time: str, media_ref: str | None) -> None:
    if kind is EntryKind.TIMESTAMP and not (text or time):
        raise invalid("timestamp entry is empty")
    if kind is EntryKind.IMAGE and media_ref is None:
        raise invalid("image entry requires a media reference")
    if kind not in {EntryKind.TIMESTAMP, EntryKind.IMAGE} and not text and media_ref is None:
        raise invalid("entry content is empty")


def _entry(raw: CompactItem, people: set[str]) -> SnapshotEntry:
    if not isinstance(raw, CompactItem):
        raise invalid("entry type is invalid")
    kind = _enum(EntryKind, raw.k)
    author_id = _entry_author(kind, raw.by, people)
    text = _bounded_text(raw.text, label="entry text", maximum=MAX_ITEM_TEXT)
    time = _bounded_text(raw.time, label="entry time", maximum=40)
    media_ref = _opaque_ref(raw.media, label="scene.items[].media")
    _validate_entry_content(kind, text, time, media_ref)
    return SnapshotEntry(kind, author_id, text, time, media_ref, _entry_quote(raw.quote))


def _validate_capability(
    theme: SnapshotTheme, mode: SceneMode, entries: tuple[SnapshotEntry, ...], draft: str
) -> None:
    capability = THEME_CAPABILITIES[theme]
    if mode not in capability.modes:
        raise SocialSnapshotError(
            SocialSnapshotErrorCode.UNSUPPORTED_MODE, "mode is unavailable for this theme"
        )
    if any(entry.kind not in capability.entry_kinds for entry in entries):
        raise SocialSnapshotError(
            SocialSnapshotErrorCode.UNSUPPORTED_ENTRY,
            "entry kind is unavailable for this theme",
        )
    if not capability.supports_quotes and any(entry.quote is not None for entry in entries):
        raise SocialSnapshotError(
            SocialSnapshotErrorCode.UNSUPPORTED_ENTRY, "quotes are unavailable for this theme"
        )
    if draft and not capability.supports_draft:
        raise SocialSnapshotError(
            SocialSnapshotErrorCode.UNSUPPORTED_ENTRY, "draft is unavailable for this theme"
        )


def _validate_limits(
    request: SocialSnapshotRequest,
    entries: tuple[SnapshotEntry, ...],
    *,
    title: str,
    draft: str,
) -> None:
    text_count = len(title) + len(draft)
    text_count += sum(
        len(entry.text)
        + len(entry.time)
        + (len(entry.quote.text) + len(entry.quote.media_label) if entry.quote else 0)
        for entry in entries
    )
    refs = {person.avatar for person in request.people if person.avatar}
    media_refs = {entry.media_ref for entry in entries if entry.media_ref}
    if text_count > MAX_TOTAL_TEXT:
        raise SocialSnapshotError(SocialSnapshotErrorCode.LIMIT_EXCEEDED, "text limit exceeded")
    if len(refs | media_refs) > MAX_ASSET_REFS or len(media_refs) > MAX_MEDIA_REFS:
        raise SocialSnapshotError(SocialSnapshotErrorCode.LIMIT_EXCEEDED, "asset limit exceeded")
    canonical_size = len(
        json.dumps(asdict(request), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    if canonical_size > MAX_REQUEST_BYTES:
        raise SocialSnapshotError(
            SocialSnapshotErrorCode.LIMIT_EXCEEDED, "request byte limit exceeded"
        )


def _integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise invalid(f"{label} must be an integer")
    return value


def _bounded_integer(value: object, *, label: str, minimum: int, maximum: int) -> int:
    number = _integer(value, label=label)
    if not minimum <= number <= maximum:
        raise SocialSnapshotError(SocialSnapshotErrorCode.LIMIT_EXCEEDED, f"{label} exceeded")
    return number


def _canvas_height(raw: CompactUi, theme: SnapshotTheme) -> int | None:
    capability = THEME_CAPABILITIES[theme]
    height = raw.height
    if height is not None:
        height = _bounded_integer(
            height, label="canvas height", minimum=MIN_HEIGHT, maximum=MAX_HEIGHT
        )
    if height is None and not capability.supports_auto_height:
        height = 1920
    return height


def _ui(request: SocialSnapshotRequest, theme: SnapshotTheme) -> SnapshotUi:
    raw = request.ui
    if not isinstance(raw, CompactUi):
        raise invalid("ui type is invalid")
    battery_percent = _integer(raw.battery_percent, label="battery percent")
    if not 0 <= battery_percent <= 100:
        raise invalid("battery percent must be between 0 and 100")
    if not isinstance(raw.battery_charging, bool):
        raise invalid("battery charging must be a boolean")
    width = _bounded_integer(raw.width, label="canvas width", minimum=MIN_WIDTH, maximum=MAX_WIDTH)
    height = _canvas_height(raw, theme)
    segment_height = _bounded_integer(
        raw.segment_height, label="segment height", minimum=MIN_HEIGHT, maximum=MAX_HEIGHT
    )
    if width * (height or segment_height) > MAX_PART_PIXELS:
        raise SocialSnapshotError(SocialSnapshotErrorCode.LIMIT_EXCEEDED, "canvas pixels exceeded")
    return SnapshotUi(
        subtitle=_bounded_text(raw.subtitle, label="subtitle", maximum=100),
        clock=_bounded_text(raw.clock, label="clock", maximum=20, required=True),
        battery_percent=battery_percent,
        battery_charging=raw.battery_charging,
        width=width,
        height=height,
        segment_height=segment_height,
    )


def normalize_request(request: SocialSnapshotRequest) -> SocialSnapshotScene:
    if not isinstance(request, SocialSnapshotRequest):
        raise invalid("request type is invalid")
    theme = _enum(SnapshotTheme, request.theme)
    mode = _enum(SceneMode, request.mode)
    if not isinstance(request.people, tuple) or not isinstance(request.items, tuple):
        raise invalid("people and items must be immutable sequences")
    if not 1 <= len(request.people) <= MAX_PARTICIPANTS:
        raise SocialSnapshotError(
            SocialSnapshotErrorCode.LIMIT_EXCEEDED, "participant limit exceeded"
        )
    if not 1 <= len(request.items) <= MAX_ENTRIES:
        raise SocialSnapshotError(SocialSnapshotErrorCode.LIMIT_EXCEEDED, "entry limit exceeded")
    participants = tuple(_participant(person) for person in request.people)
    people = {participant.participant_id for participant in participants}
    if len(people) != len(participants):
        raise invalid("person ids must be unique")
    entries = tuple(_entry(item, people) for item in request.items)
    title = _bounded_text(request.title, label="title", maximum=MAX_TITLE, required=True)
    draft = _bounded_text(request.draft, label="draft", maximum=MAX_DRAFT)
    _validate_capability(theme, mode, entries, draft)
    ui = _ui(request, theme)
    _validate_limits(request, entries, title=title, draft=draft)
    canonical = json.dumps(
        asdict(request), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return SocialSnapshotScene(
        theme=theme,
        mode=mode,
        title=title,
        participants=participants,
        entries=entries,
        draft=draft,
        ui=ui,
        disclosure=DISCLOSURE,
        request_fingerprint=fingerprint,
    )
