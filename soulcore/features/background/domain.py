"""Stable contracts for the five independent background authors.

The background system is a hidden, role-centric life simulator.  Its authors
operate at different creative scales, but an upper layer is reference material,
not a command that a lower layer must consume.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from ...contracts.delivery_visibility import DeliveryVisibility

DEFAULT_ORDINARY_MIN_MINUTES = 120
DEFAULT_ORDINARY_MAX_MINUTES = 360


class BackgroundAuthorKind(StrEnum):
    WORLD = "WORLD"
    LIFE_DIRECTION = "LIFE_DIRECTION"
    STORY_SOURCE = "STORY_SOURCE"
    KEYFRAME = "KEYFRAME"
    ORDINARY = "ORDINARY"


AUTHOR_ORDER = (
    BackgroundAuthorKind.WORLD,
    BackgroundAuthorKind.LIFE_DIRECTION,
    BackgroundAuthorKind.STORY_SOURCE,
    BackgroundAuthorKind.KEYFRAME,
    BackgroundAuthorKind.ORDINARY,
)

# These links only decide which independent author states are useful references.
# They never create a requirement to continue, adopt, or resolve an upper-layer
# idea.
REFERENCE_AUTHORS: dict[BackgroundAuthorKind, tuple[BackgroundAuthorKind, ...]] = {
    BackgroundAuthorKind.WORLD: (),
    BackgroundAuthorKind.LIFE_DIRECTION: (BackgroundAuthorKind.WORLD,),
    BackgroundAuthorKind.STORY_SOURCE: (
        BackgroundAuthorKind.WORLD,
        BackgroundAuthorKind.LIFE_DIRECTION,
    ),
    BackgroundAuthorKind.KEYFRAME: (
        BackgroundAuthorKind.WORLD,
        BackgroundAuthorKind.LIFE_DIRECTION,
        BackgroundAuthorKind.STORY_SOURCE,
    ),
    BackgroundAuthorKind.ORDINARY: (
        BackgroundAuthorKind.WORLD,
        BackgroundAuthorKind.LIFE_DIRECTION,
        BackgroundAuthorKind.STORY_SOURCE,
        BackgroundAuthorKind.KEYFRAME,
    ),
}


class BackgroundInitializationStep(StrEnum):
    WORLD = "WORLD"
    LIFE_DIRECTION = "LIFE_DIRECTION"
    STORY_SOURCE = "STORY_SOURCE"
    ORDINARY_CURRENT = "ORDINARY_CURRENT"
    READY = "READY"


class BackgroundTimelineSource(StrEnum):
    ORDINARY = "ORDINARY"
    KEYFRAME = "KEYFRAME"


@dataclass(frozen=True, slots=True)
class BackgroundInputVersions:
    """Publication fence captured before model work starts."""

    config_version: int
    continuity_version: int
    activity_epoch: int
    timeline_version: int
    view_version: int
    publication_version: int
    author_state_version: int
    # Load-time-only ownership fence for a role frame.  These values
    # deliberately stay out of ``as_dict`` because the durable task is created
    # before its model input is loaded; publication still receives this exact
    # in-memory snapshot through ``BackgroundAuthorInput``.
    frame_start_at: datetime | None = field(default=None, repr=False)
    frame_end_at: datetime | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_version": self.config_version,
            "continuity_version": self.continuity_version,
            "activity_epoch": self.activity_epoch,
            "timeline_version": self.timeline_version,
            "view_version": self.view_version,
            "publication_version": self.publication_version,
            "author_state_version": self.author_state_version,
        }


@dataclass(frozen=True, slots=True)
class RoleCurrentView:
    revision: int = 0
    as_of: datetime | str | None = None
    source: str = ""
    source_event_id: int | None = None
    source_publication_id: int | None = None
    narrative_time: str = ""
    location: str = ""
    doing: str = ""
    body_state: str = ""
    mood: str = ""
    intention: str = ""
    current_concern: str = ""
    # Dynamically attached from the timeline read model; never persisted as a
    # second copy of events and never contains hidden upper-layer material.
    recent_experiences: tuple[BackgroundTimelineEvent, ...] = ()

    @property
    def empty(self) -> bool:
        return not any(
            (
                self.narrative_time,
                self.location,
                self.doing,
                self.body_state,
                self.mood,
                self.intention,
                self.current_concern,
                self.recent_experiences,
            )
        )


@dataclass(frozen=True, slots=True)
class BackgroundFrameInterval:
    """One continuous settled interval owned by a role background frame."""

    start_at: datetime
    end_at: datetime

    def __post_init__(self) -> None:
        if self.end_at < self.start_at:
            raise ValueError("background frame interval cannot end before it starts")


@dataclass(frozen=True, slots=True)
class BackgroundAuthorState:
    author_kind: BackgroundAuthorKind
    state_version: int = 0
    content: dict[str, Any] = field(default_factory=dict)
    backend_id: str = ""
    last_success_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class BackgroundStorySource:
    story_source_id: int
    public_ref: str
    module_text: str
    shown_count: int = 0
    engagement_state: str = "PENDING"


@dataclass(frozen=True, slots=True)
class BackgroundTimelineEvent:
    timeline_event_id: int
    public_ref: str
    source: BackgroundTimelineSource
    content: str
    frame_start_at: datetime | None = None
    frame_end_at: datetime | None = None
    leftover_text: str = ""
    leftover_retired_at: str = ""


@dataclass(frozen=True, slots=True)
class BackgroundVisibleReferences:
    """Exact M/V targets present in the final model-visible author input."""

    story_sources: Mapping[str, BackgroundStorySource] = field(
        default_factory=lambda: MappingProxyType({})
    )
    timeline_events: Mapping[str, BackgroundTimelineEvent] = field(
        default_factory=lambda: MappingProxyType({})
    )


@dataclass(frozen=True, slots=True)
class ForegroundContinuityMessage:
    """One recent conversation item offered to the background writer.

    The runtime preserves speaker and order; interpretation is left to the
    writer just as it would be at an improvisational game table.
    """

    message_id: int
    direction: str
    role: str
    participant_id: str
    speaker_name: str
    plain_text: str
    internal_memo: str = ""
    components: tuple[dict[str, Any], ...] = ()
    delivery_visibility: DeliveryVisibility = DeliveryVisibility.NOT_VISIBLE
    occurred_at: datetime | None = None
    scene_narration_before: tuple[str, ...] = ()
    scene_narration_after: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ForegroundContinuityResult:
    command: str
    ok: bool
    result: str


@dataclass(frozen=True, slots=True)
class ForegroundContinuityRun:
    run_id: int
    source: str
    reason: str
    finished_at: datetime | None
    results: tuple[ForegroundContinuityResult, ...]


@dataclass(frozen=True, slots=True)
class BackgroundAuthorInput:
    profile_id: str
    instance_id: str
    author_kind: BackgroundAuthorKind
    generation: int
    initialization_state: str
    initialization_step: BackgroundInitializationStep
    config: dict[str, Any]
    author_state: BackgroundAuthorState
    reference_states: tuple[BackgroundAuthorState, ...]
    story_sources: tuple[BackgroundStorySource, ...]
    recent_timeline: tuple[BackgroundTimelineEvent, ...]
    foreground_messages: tuple[ForegroundContinuityMessage, ...]
    foreground_runs: tuple[ForegroundContinuityRun, ...]
    current_view: RoleCurrentView
    ordinary_frame_interval: BackgroundFrameInterval | None
    keyframe_frame_interval: BackgroundFrameInterval | None
    seed: dict[str, Any]
    lore: tuple[dict[str, Any], ...]
    boundaries: tuple[dict[str, Any], ...]
    versions: BackgroundInputVersions
    prompt_now: datetime
    timezone_name: str
    # A bootstrap owns one stable story-time anchor.  In the configured story
    # timezone, the opening Keyframe hands off at 16:00 on the prior calendar
    # day and the opening Ordinary frame owns the full interval to the anchor.
    # The prose before that seam is not represented as a machine-audited range.
    initialization_anchor_at: datetime | None = None
    opening_keyframe_completed: bool = False
    # Exact scanned frontiers used for transactional cursor advancement. They
    # remain internal and are never projected as model-visible identifiers.
    foreground_message_through: int = 0
    foreground_run_through: int = 0


@dataclass(frozen=True, slots=True)
class BackgroundStorySourceDraft:
    module_text: str


@dataclass(frozen=True, slots=True)
class BackgroundTimelineEventDraft:
    source: BackgroundTimelineSource
    content: str
    frame_start_at: datetime
    frame_end_at: datetime
    leftover_text: str = ""


@dataclass(frozen=True, slots=True)
class BackgroundDraft:
    content: dict[str, Any]
    story_sources: tuple[BackgroundStorySourceDraft, ...] = ()
    timeline_events: tuple[BackgroundTimelineEventDraft, ...] = ()
    current_view: dict[str, Any] = field(default_factory=dict)
    consumed_foreground_through_message_id: int = 0
    consumed_foreground_through_run_id: int = 0
    retired_timeline_event_ids: tuple[int, ...] = ()
    engaged_story_ids: tuple[int, ...] = ()
    concluded_story_ids: tuple[int, ...] = ()
    creator_output: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class BackgroundPublicationResult:
    publication_id: int
    public_ref: str
    author_kind: BackgroundAuthorKind
    generation: int
    next_due_at: datetime
    hard_due_at: datetime
    initialization_step: BackgroundInitializationStep = BackgroundInitializationStep.READY
    timeline_event_ids: tuple[int, ...] = ()
    story_source_refs: tuple[str, ...] = ()
    foreground_message_cursor: int = 0
    foreground_run_cursor: int = 0


class BackgroundDraftStale(RuntimeError):
    """A model draft was based on continuity that changed before publication."""


class BackgroundDisabled(RuntimeError):
    """The instance was disabled before an author could publish."""


__all__ = [
    "AUTHOR_ORDER",
    "DEFAULT_ORDINARY_MAX_MINUTES",
    "DEFAULT_ORDINARY_MIN_MINUTES",
    "REFERENCE_AUTHORS",
    "BackgroundAuthorInput",
    "BackgroundAuthorKind",
    "BackgroundAuthorState",
    "BackgroundDisabled",
    "BackgroundDraft",
    "BackgroundDraftStale",
    "BackgroundFrameInterval",
    "BackgroundInitializationStep",
    "BackgroundInputVersions",
    "BackgroundPublicationResult",
    "BackgroundStorySource",
    "BackgroundStorySourceDraft",
    "BackgroundTimelineEvent",
    "BackgroundVisibleReferences",
    "BackgroundTimelineEventDraft",
    "BackgroundTimelineSource",
    "ForegroundContinuityMessage",
    "ForegroundContinuityResult",
    "ForegroundContinuityRun",
    "RoleCurrentView",
]
