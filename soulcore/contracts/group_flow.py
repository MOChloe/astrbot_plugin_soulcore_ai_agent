"""Stable cross-domain contracts for durable group-chat flow control."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class GroupFlowStatus(StrEnum):
    COLLECTING = "COLLECTING"
    JUDGING = "JUDGING"
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING_FIRST_ATTEMPT = "WAITING_FIRST_ATTEMPT"
    RESOLVED = "RESOLVED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class GroupFlowPolicy:
    profile_id: str
    scope: str = "group"
    quiet_seconds: int = 30
    base_message_count: int = 2
    ordinary_min_reply_gap_seconds: int = 0
    judge_token_budget: int = 2048
    version: int = 1
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.profile_id.strip() or self.scope != "group":
            raise ValueError("group flow policy requires a group profile")
        if not 5 <= self.quiet_seconds <= 300:
            raise ValueError("quiet_seconds must be between 5 and 300")
        if not 1 <= self.base_message_count <= 50:
            raise ValueError("base_message_count must be between 1 and 50")
        if not 0 <= self.ordinary_min_reply_gap_seconds <= 86400:
            raise ValueError("ordinary_min_reply_gap_seconds must be between 0 and 86400")
        if not 512 <= self.judge_token_budget <= 8192:
            raise ValueError("judge_token_budget must be between 512 and 8192")
        if self.version < 1:
            raise ValueError("policy version must be positive")


@dataclass(frozen=True, slots=True)
class GroupFlowInboundMessage:
    message_id: int
    occurred_at: datetime
    sender_id: str = ""
    sender_name: str = ""
    plain_text: str = ""
    media_kinds: tuple[str, ...] = ()
    media_cluster_keys: tuple[str, ...] = ()
    direct_address: bool = False

    def __post_init__(self) -> None:
        if self.message_id < 1:
            raise ValueError("group flow messages must already exist in the ledger")


@dataclass(frozen=True, slots=True)
class GroupFlowSourceMessage:
    message_id: int
    occurred_at: datetime
    sender_id: str = ""
    sender_name: str = ""
    plain_text: str = ""
    media_kinds: tuple[str, ...] = ()
    media_cluster_keys: tuple[str, ...] = ()
    role: str = "user"
    direction: str = "INBOUND"
    delivery_status: str = ""


@dataclass(frozen=True, slots=True)
class GroupFlowWindow:
    window_id: str
    profile_id: str
    instance_id: str
    status: GroupFlowStatus
    message_ids: tuple[int, ...]
    first_message_id: int
    last_message_id: int
    message_count: int
    rate_ewma: float = 0.0
    repeat_ratio: float = 0.0
    judge_threshold: int = 2
    judge_through_message_id: int | None = None
    frozen_through_message_id: int | None = None
    next_judge_at: datetime | None = None
    quiet_due_at: datetime | None = None
    dynamic_due_at: datetime | None = None
    direct_due_at: datetime | None = None
    direct_address: bool = False
    judge_result: str = ""
    judge_error_code: str = ""
    lease_owner: str | None = None
    lease_token: int = 0
    lease_until: datetime | None = None
    main_core_task_ref: str | None = None
    first_attempt_started_at: datetime | None = None
    error_code: str = ""
    version: int = 1
    created_at: datetime | None = None
    updated_at: datetime | None = None
    resolved_at: datetime | None = None
    resolution_outcome: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            GroupFlowStatus.RESOLVED,
            GroupFlowStatus.FAILED,
            GroupFlowStatus.CANCELLED,
        }


@dataclass(frozen=True, slots=True)
class GroupRunFence:
    window_id: str
    frozen_through_message_id: int
    lease_token: int
    version: int
    main_core_task_ref: str

    def __post_init__(self) -> None:
        if not self.window_id.strip() or not self.main_core_task_ref.strip():
            raise ValueError("group run fence identifiers cannot be empty")
        if self.frozen_through_message_id < 1:
            raise ValueError("group run fence requires a frozen message boundary")
        if self.lease_token < 0 or self.version < 1:
            raise ValueError("group run fence tokens must be non-negative")

    def as_metadata(self) -> dict[str, str | int]:
        return {
            "window_id": self.window_id,
            "frozen_through_message_id": self.frozen_through_message_id,
            "lease_token": self.lease_token,
            "version": self.version,
            "main_core_task_ref": self.main_core_task_ref,
        }

    @classmethod
    def from_metadata(cls, value: object) -> GroupRunFence | None:
        if not isinstance(value, Mapping):
            return None
        try:
            return cls(
                window_id=str(value.get("window_id") or ""),
                frozen_through_message_id=int(value["frozen_through_message_id"]),
                lease_token=int(value["lease_token"]),
                version=int(value["version"]),
                main_core_task_ref=str(value.get("main_core_task_ref") or ""),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True, slots=True)
class GroupReplyRelocationCheck:
    profile_id: str
    instance_id: str
    fence: GroupRunFence
    delta_window_id: str
    delta_through_message_id: int
    check_token: int
    final_recheck: bool = False
    pending_first_text: str = ""

    def __post_init__(self) -> None:
        if not self.profile_id.strip() or not self.instance_id.strip():
            raise ValueError("group reply relocation requires an instance")
        if not self.delta_window_id.strip() or self.delta_through_message_id < 1:
            raise ValueError("group reply relocation requires a durable delta snapshot")
        if self.check_token < 1:
            raise ValueError("group reply relocation check token must be positive")


@dataclass(frozen=True, slots=True)
class GroupReplyRelocationResult:
    applied: bool
    previous_status: GroupFlowStatus | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class GroupFlowDiagnostic:
    window: GroupFlowWindow | None
    next_window: GroupFlowWindow | None = None
    algorithm: Mapping[str, object] = field(default_factory=dict)


__all__ = [
    "GroupFlowDiagnostic",
    "GroupFlowInboundMessage",
    "GroupFlowPolicy",
    "GroupReplyRelocationCheck",
    "GroupReplyRelocationResult",
    "GroupFlowSourceMessage",
    "GroupFlowStatus",
    "GroupFlowWindow",
    "GroupRunFence",
]
