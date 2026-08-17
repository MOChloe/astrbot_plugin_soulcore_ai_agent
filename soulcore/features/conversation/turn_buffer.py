"""Durable input-turn buffering contracts.

The classifier consumes lightweight current-turn and recent-dialogue projections,
which deliberately cannot carry component payloads, URLs, asset identifiers, or host
paths.  A bounded reply-reference projection is retained separately for Main Core
reconstruction and is not passed to the buffering classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

TURN_BUFFER_RECENT_DIALOGUE_LIMIT = 4


class TurnBufferStatus(StrEnum):
    PENDING = "PENDING"
    CLASSIFYING = "CLASSIFYING"
    WAITING = "WAITING"
    CLAIMED = "CLAIMED"
    RESOLVED = "RESOLVED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class TurnBufferMessageProjection:
    message_id: int
    sender_id: str
    sender_name: str
    plain_text: str
    media_types: tuple[str, ...]
    occurred_at: datetime
    reply_reference: str = ""


@dataclass(frozen=True, slots=True)
class TurnBufferDialogueProjection:
    """One lightweight, visible dialogue line preceding the buffered turn."""

    message_id: int
    is_character: bool
    sender_id: str
    plain_text: str
    media_types: tuple[str, ...]
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class TurnBufferBatch:
    batch_id: str
    profile_id: str
    instance_id: str
    generation: int
    activity_epoch: int
    status: TurnBufferStatus
    message_ids: tuple[int, ...]
    requested_delay_seconds: int | None = None
    ai_elapsed_seconds: float | None = None
    remaining_delay_seconds: float | None = None
    due_at: datetime | None = None
    lease_owner: str | None = None
    lease_token: int = 0
    lease_until: datetime | None = None
    main_core_task_ref: str | None = None
    error_code: str = ""
    resolution_outcome: str = ""
    version: int = 1
    created_at: datetime | None = None
    updated_at: datetime | None = None
    resolved_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            TurnBufferStatus.RESOLVED,
            TurnBufferStatus.CANCELLED,
            TurnBufferStatus.FAILED,
        }


__all__ = [
    "TURN_BUFFER_RECENT_DIALOGUE_LIMIT",
    "TurnBufferBatch",
    "TurnBufferDialogueProjection",
    "TurnBufferMessageProjection",
    "TurnBufferStatus",
]
