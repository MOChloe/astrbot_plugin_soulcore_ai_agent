from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(UTC)


def stable_instance_id(
    route_umo: str,
    *,
    platform_id: str = "",
    message_type: str = "",
    target_id: str = "",
) -> str:
    """Build an opaque, versioned identity without normalizing platform IDs.

    Length-prefixing prevents delimiter ambiguity.  Unknown UMO layouts fall
    back to the complete opaque UMO, so two routes can never be merged merely
    because parsing failed.
    """

    raw = str(route_umo or "").strip()
    platform = str(platform_id or "").strip()
    message = str(message_type or "").strip()
    target = str(target_id or "").strip()
    parts = (
        ("parsed", platform, message, target)
        if all((platform, message, target))
        else ("opaque", raw)
    )
    canonical = "".join(f"{len(part)}:{part}" for part in parts)
    return "ci:v1:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class WakeSource(StrEnum):
    FOREGROUND_MESSAGE = "FOREGROUND_MESSAGE"
    PLUGIN_WAKE = "PLUGIN_WAKE"
    DEFERRED_MESSAGE = "DEFERRED_MESSAGE"
    TIMER = "TIMER"


class RunStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SUPERSEDED = "SUPERSEDED"


class WakeupStatus(StrEnum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class OutboxStatus(StrEnum):
    PENDING = "PENDING"
    SENDING = "SENDING"
    PLATFORM_ACCEPTED_UNCONFIRMED = "PLATFORM_ACCEPTED_UNCONFIRMED"
    PARTIALLY_ATTEMPTED = "PARTIALLY_ATTEMPTED"
    FAILED = "FAILED"
    UNKNOWN_AFTER_CRASH = "UNKNOWN_AFTER_CRASH"
    CANCELLED = "CANCELLED"


class OutboxInterruptPolicy(StrEnum):
    PRESERVE = "PRESERVE"
    CANCEL_ON_PLAYER_MESSAGE = "CANCEL_ON_PLAYER_MESSAGE"


class ExpressionBatchStatus(StrEnum):
    ACTIVE = "ACTIVE"
    SETTLED = "SETTLED"
    PARTIALLY_SETTLED = "PARTIALLY_SETTLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class RouteReadiness(StrEnum):
    READY = "READY"
    ROUTE_NOT_READY = "ROUTE_NOT_READY"


class InstanceInitializationState(StrEnum):
    UNINITIALIZED = "UNINITIALIZED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"


@dataclass(frozen=True, slots=True)
class InstanceInitializationDecision:
    state: InstanceInitializationState
    started: bool = False
    arrived_before_ready: bool = False

    @property
    def accepts_messages(self) -> bool:
        return self.state is InstanceInitializationState.READY and not self.arrived_before_ready


class MessageDirection(StrEnum):
    INBOUND = "INBOUND"
    OUTBOUND = "OUTBOUND"


class MessageRetractionStatus(StrEnum):
    PENDING = "PENDING"
    SENDING = "SENDING"
    RETRACTED = "RETRACTED"
    FAILED = "FAILED"
    UNKNOWN_AFTER_CRASH = "UNKNOWN_AFTER_CRASH"
    CANCELLED = "CANCELLED"


@dataclass(slots=True)
class RoleProfile:
    profile_id: str
    name: str = ""
    enabled: bool = False
    quick_setup_decided: bool = False
    thinking_complexity: str = "标准"
    background_life_enabled: bool = False
    background_life_version: int = 1
    turn_buffer_enabled: bool = True
    image_generation_enabled: bool = False
    file_artifacts_enabled: bool = False
    web_search_enabled: bool = False
    web_search_intensity: str = "STANDARD"
    proactive_enabled: bool = True
    extra_background: str = ""
    min_wakeup_minutes: int = 15
    max_wakeup_minutes: int = 55
    low_frequency_min_wakeup_minutes: int = 180
    low_frequency_max_wakeup_minutes: int = 480
    orphaned: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class ScopeConfig:
    """Configuration shared by new private or group conversation instances."""

    profile_id: str
    scope: str
    proactive_enabled: bool = True
    extra_background: str = ""
    world_texture_prompt: str = ""
    media_original_retention_days: int = 30
    min_wakeup_minutes: int = 15
    max_wakeup_minutes: int = 55
    low_frequency_min_wakeup_minutes: int = 180
    low_frequency_max_wakeup_minutes: int = 480
    max_context_tokens: int = 128000
    target_context_tokens: int = 64000
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class CharacterInstance:
    """One persistent character living in exactly one conversation route."""

    profile_id: str
    instance_id: str
    route_umo: str
    platform_id: str = ""
    message_type: str = ""
    target_id: str = ""
    scope: str = "private"
    session_kind: str = ""
    readiness: RouteReadiness = RouteReadiness.READY
    initialization_state: InstanceInitializationState = InstanceInitializationState.READY
    initialization_completed_at: datetime | None = None
    proactive_enabled: bool = True
    extra_background: str = ""
    min_wakeup_minutes: int = 15
    max_wakeup_minutes: int = 55
    low_frequency_min_wakeup_minutes: int = 180
    low_frequency_max_wakeup_minutes: int = 480
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class InstanceChatPolicy:
    """Administrator-owned settings for one conversation route."""

    profile_id: str
    instance_id: str
    soulcore_enabled: bool = True
    image_send_enabled: bool = True
    private_fallback_player_name: str = ""
    private_name_override_enabled: bool = False
    version: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class InterruptedExpression:
    """One expression the character formed before a later message interrupted it."""

    ordinal: int
    content: str = ""
    expression_kind: str = "TEXT"
    internal_memo: str = ""


@dataclass(slots=True)
class ConversationMessage:
    """One immutable entry in SoulCore's instance-owned message ledger."""

    message_id: int
    profile_id: str
    instance_id: str
    direction: MessageDirection
    role: str
    internal_memo: str = ""
    expression_batch_id: str | None = None
    expression_ordinal: int | None = None
    sender_id: str = ""
    sender_name: str = ""
    plain_text: str = ""
    identity_template: str = ""
    components: list[dict[str, Any]] = field(default_factory=list)
    delivery_status: str = "RECEIVED"
    idempotency_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime | None = None
    created_at: datetime | None = None
    knowledge_eligibility: str = "ELIGIBLE"
    knowledge_eligibility_reason: str = ""
    interrupted_expressions: list[InterruptedExpression] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PlatformMessageFragment:
    """One physical platform message bound to an immutable ledger record."""

    message_ref: str
    profile_id: str
    instance_id: str
    ledger_message_id: int
    fragment_ordinal: int
    platform_instance_id: str
    route_umo: str
    platform_message_id: str
    direction: MessageDirection
    content_kind: str
    platform_reference_id: str = ""
    content_projection: str = ""
    sender_id: str = ""
    native_reply_supported: bool = False
    member_mention_supported: bool = False
    self_retraction_supported: bool = False
    returns_platform_message_id: bool = True
    accepted_at: datetime | None = None
    retractable_until: datetime | None = None
    retraction_status: MessageRetractionStatus | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class MessageRetractionAction:
    """A durable, idempotent request to retract one physical message fragment."""

    action_id: int
    profile_id: str
    instance_id: str
    source_run_id: int
    expression_batch_id: str
    step_ordinal: int
    idempotency_key: str
    status: MessageRetractionStatus
    target_message_ref: str | None = None
    target_output_ordinal: int | None = None
    delay_after_previous_seconds: int = 0
    not_before_at: datetime | None = None
    attempted_at: datetime | None = None
    completed_at: datetime | None = None
    error_code: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class DialogueSummary:
    summary_id: int
    profile_id: str
    instance_id: str
    version: int
    strategy_id: str
    strategy_version: int
    covered_from_message_id: int
    covered_through_message_id: int
    structured: dict[str, Any] = field(default_factory=dict)
    rendered_text: str = ""
    token_count: int = 0
    created_at: datetime | None = None


@dataclass(slots=True)
class ContextBuildReport:
    profile_id: str
    instance_id: str
    model_id: str = ""
    token_count_mode: str = "ESTIMATED"
    hard_token_limit: int = 128000
    target_token_budget: int = 64000
    fill_budget: int = 44800
    total_tokens: int = 0
    report: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(slots=True)
class CoreState:
    profile_id: str
    instance_id: str
    state_epoch: int = 0
    activity_epoch: int = 0
    low_frequency_mode: bool = False
    low_frequency_reason: str = ""
    low_frequency_since: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class CoreWakeRequest:
    profile_id: str
    instance_id: str
    source: WakeSource
    reason: str = ""
    route_umo: str | None = None
    user_message: str | None = None
    timer_prompt: str = ""
    expected_state_epoch: int | None = None
    expected_activity_epoch: int | None = None
    wakeup_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    requested_at: datetime = field(default_factory=utc_now)
    # Internal scheduler continuity.  These fields are consumed before the
    # MainCore lease and are never projected into model-visible metadata.
    proactive_frame_source_ref: str = field(default="", repr=False)
    proactive_frame_planned_at: datetime | None = field(default=None, repr=False)


@dataclass(slots=True)
class CommittedCoreRunEvidence:
    """Non-display evidence from the one MainCore round that actually committed.

    Peripheral runtimes may consume this snapshot after the RolePlay turn has
    finished.  It deliberately excludes rejected rounds and tool transcripts.
    """

    working_text: str = field(default="", repr=False)
    decision_kind: str = ""
    output_status: str = ""
    state_epoch: int = 0
    activity_epoch: int = 0


@dataclass(slots=True)
class CoreRunResult:
    run_id: int
    status: RunStatus
    state_epoch: int | None = None
    reply: str | None = None
    memo: str | None = None
    expression_steps: list[dict[str, Any]] = field(default_factory=list)
    expression_batch_id: str | None = None
    media_asset_ids: list[str] = field(default_factory=list)
    superseded: bool = False
    error: str | None = None
    error_code: str | None = None
    retryable: bool | None = None
    diagnostic: Mapping[str, Any] = field(default_factory=dict)
    sticker_ref_ids: list[str] = field(default_factory=list)
    file_asset_ids: list[str] = field(default_factory=list)
    important_todo_ids: list[str] = field(default_factory=list)
    silent: bool = False
    had_output: bool = False
    silence_reason: str = ""
    committed_evidence: CommittedCoreRunEvidence | None = field(default=None, repr=False)


@dataclass(slots=True)
class ExpressionBatch:
    batch_id: str
    profile_id: str
    instance_id: str
    source_run_id: int
    activity_epoch: int
    route_umo: str
    status: ExpressionBatchStatus
    output_count: int
    retraction_count: int = 0
    step_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    settled_at: datetime | None = None


@dataclass(slots=True)
class Wakeup:
    wakeup_id: int
    profile_id: str
    instance_id: str
    source: WakeSource
    due_at: datetime
    reason: str
    conversation_ref: str | None
    idempotency_key: str | None
    payload: dict[str, Any]
    status: WakeupStatus
    attempts: int
    lease_until: datetime | None = None
    last_error: str | None = None
    generation: int = 0
    lease_token: int = 0
    version: int = 0
    intent_kind: str = ""
    linked_task_id: int | None = None


@dataclass(slots=True)
class OutboxItem:
    outbox_id: int
    profile_id: str
    instance_id: str
    umo: str
    payload: dict[str, Any]
    status: OutboxStatus
    idempotency_key: str
    attempts: int
    activity_epoch: int = 0
    expression_batch_id: str | None = None
    expression_ordinal: int | None = None
    expression_step_ordinal: int | None = None
    not_before_at: datetime | None = None
    interrupt_policy: OutboxInterruptPolicy = OutboxInterruptPolicy.PRESERVE
    depends_on_idempotency_key: str | None = None
    context_message_id: int | None = None
    last_error_code: str = ""
    last_error: str | None = None
    last_diagnostic_code: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
