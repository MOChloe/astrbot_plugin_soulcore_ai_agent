"""Private value objects shared by background publication modules."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ....storage.sqlite.codec import _load
from ..domain import (
    BackgroundAuthorKind,
    BackgroundInitializationStep,
)

ROLE_AUTHORS = frozenset(
    {
        BackgroundAuthorKind.KEYFRAME,
        BackgroundAuthorKind.ORDINARY,
    }
)
MAX_STORY_SOURCES = 12
INITIALIZATION_NEXT = {
    BackgroundInitializationStep.WORLD: BackgroundInitializationStep.LIFE_DIRECTION,
    BackgroundInitializationStep.LIFE_DIRECTION: BackgroundInitializationStep.STORY_SOURCE,
    BackgroundInitializationStep.STORY_SOURCE: BackgroundInitializationStep.ORDINARY_CURRENT,
    BackgroundInitializationStep.ORDINARY_CURRENT: BackgroundInitializationStep.READY,
    BackgroundInitializationStep.READY: BackgroundInitializationStep.READY,
}
INITIALIZATION_AUTHOR = {
    BackgroundInitializationStep.WORLD: BackgroundAuthorKind.WORLD,
    BackgroundInitializationStep.LIFE_DIRECTION: BackgroundAuthorKind.LIFE_DIRECTION,
    BackgroundInitializationStep.STORY_SOURCE: BackgroundAuthorKind.STORY_SOURCE,
    BackgroundInitializationStep.ORDINARY_CURRENT: BackgroundAuthorKind.ORDINARY,
}


def aware(value: datetime) -> datetime:
    return (value if value.tzinfo is not None else value.replace(tzinfo=UTC)).astimezone(UTC)


def json_mapping(value: object) -> dict[str, Any]:
    loaded = _load(value) if isinstance(value, str) else value
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def json_list(value: object) -> list[Any]:
    loaded = _load(value) if isinstance(value, str) else value
    return list(loaded) if isinstance(loaded, list) else []


@dataclass(frozen=True, slots=True)
class PublishContext:
    profile_id: str
    instance_id: str
    kind: BackgroundAuthorKind
    generation: int
    task_id: int
    published_at: datetime
    now: str
    next_due: str
    hard_due: str
    preserve_schedule: bool


@dataclass(frozen=True, slots=True)
class PublicationOutcome:
    publication: sqlite3.Row
    initialization_step: str
    timeline_event_ids: tuple[int, ...]
    story_source_refs: tuple[str, ...]
    foreground_message_cursor: int
    foreground_run_cursor: int
    next_due_at: str
    hard_due_at: str


@dataclass(frozen=True, slots=True)
class PublishedContent:
    publication_id: int
    state_version: int
    story_refs: tuple[str, ...]
    event_ids: tuple[int, ...]
    timeline_changed: bool
    view_changed: bool


@dataclass(frozen=True, slots=True)
class PublicationProgress:
    initialization_step: str
    message_cursor: int
    run_cursor: int


__all__ = [
    "INITIALIZATION_AUTHOR",
    "INITIALIZATION_NEXT",
    "MAX_STORY_SOURCES",
    "ROLE_AUTHORS",
    "PublicationOutcome",
    "PublicationProgress",
    "PublishContext",
    "PublishedContent",
    "aware",
    "json_list",
    "json_mapping",
]
