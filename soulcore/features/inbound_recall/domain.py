"""Durable inbound-recall facts shared by the adapter and grace worker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class InboundRecallVisibility(StrEnum):
    NONE = "NONE"
    PREFIX = "PREFIX"
    FULL = "FULL"


@dataclass(frozen=True, slots=True)
class OneBotRecallNotice:
    notice_type: str
    platform_message_id: str
    sender_id: str
    operator_id: str
    received_at: datetime
    platform_occurred_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class InboundRecallHold:
    profile_id: str
    instance_id: str
    ledger_message_id: int
    platform_instance_id: str
    route_umo: str
    platform_message_id: str
    scope: str
    direct_address: bool
    received_at: datetime
    grace_until: datetime
    previous_activity_at: datetime | None
    status: str
    lease_token: int
    lease_until: datetime | None
    committed_full_at: datetime | None
    original_plain_text: str
    original_components_json: str


@dataclass(frozen=True, slots=True)
class InboundRecallTarget:
    receipt_id: str
    notice: OneBotRecallNotice
    hold: InboundRecallHold
    duplicate: bool = False


@dataclass(frozen=True, slots=True)
class InboundRecallDecision:
    visibility: InboundRecallVisibility
    probability_seen: float
    attention_sample: float
    read_sample: float
    read_fraction: float
    exposed_text: str
    media_kinds: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InboundRecallSettlement:
    receipt_id: str
    profile_id: str
    instance_id: str
    target_message_id: int
    recall_event_message_id: int
    event_text: str
    visibility: InboundRecallVisibility
    inserted: bool
    scope: str
    direct_address: bool
    route_umo: str


__all__ = [
    "InboundRecallDecision",
    "InboundRecallHold",
    "InboundRecallSettlement",
    "InboundRecallTarget",
    "InboundRecallVisibility",
    "OneBotRecallNotice",
]
