"""Pure CAS- and idempotency-fenced Timer state transitions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime
from enum import StrEnum
from typing import TypeAlias

from .domain import (
    DeliveryAssociationRef,
    ExecutionEnvelopeRef,
    IdempotencyKey,
    TimerOccurrence,
    TimerOccurrenceStatus,
    TimerRule,
    TimerRuleStatus,
    require_aware,
)
from .errors import TimerErrorCode, fail


class RuleAction(StrEnum):
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    CANCEL = "CANCEL"


class OccurrenceAction(StrEnum):
    MARK_DUE = "MARK_DUE"
    CLAIM = "CLAIM"
    RELEASE_CLAIM = "RELEASE_CLAIM"
    START_PROVIDER = "START_PROVIDER"
    COMPLETE_NO_OP = "COMPLETE_NO_OP"
    HANDOFF_DELIVERY = "HANDOFF_DELIVERY"
    COMPLETE_DELIVERY = "COMPLETE_DELIVERY"
    FAIL = "FAIL"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    CANCEL = "CANCEL"
    SUPERSEDE_REQUEUE = "SUPERSEDE_REQUEUE"
    MARK_MISSED_COALESCED = "MARK_MISSED_COALESCED"
    ENTER_RECOVERY = "ENTER_RECOVERY"
    RECOVER_REQUEUE = "RECOVER_REQUEUE"
    RECOVER_RUNNING = "RECOVER_RUNNING"
    RECOVER_DELIVERY = "RECOVER_DELIVERY"
    RECOVER_COMPLETE = "RECOVER_COMPLETE"
    RECOVER_FAIL = "RECOVER_FAIL"


_SIMPLE_OCCURRENCE_TARGETS = {
    (TimerOccurrenceStatus.SCHEDULED, OccurrenceAction.MARK_DUE): TimerOccurrenceStatus.WAITING,
    (TimerOccurrenceStatus.WAITING, OccurrenceAction.CLAIM): TimerOccurrenceStatus.CLAIMED,
    (TimerOccurrenceStatus.CLAIMED, OccurrenceAction.RELEASE_CLAIM): TimerOccurrenceStatus.WAITING,
    (TimerOccurrenceStatus.RUNNING, OccurrenceAction.COMPLETE_NO_OP): (
        TimerOccurrenceStatus.COMPLETED
    ),
    (TimerOccurrenceStatus.WAITING_DELIVERY, OccurrenceAction.COMPLETE_DELIVERY): (
        TimerOccurrenceStatus.COMPLETED
    ),
    (TimerOccurrenceStatus.SCHEDULED, OccurrenceAction.MARK_MISSED_COALESCED): (
        TimerOccurrenceStatus.MISSED_COALESCED
    ),
    (TimerOccurrenceStatus.WAITING, OccurrenceAction.MARK_MISSED_COALESCED): (
        TimerOccurrenceStatus.MISSED_COALESCED
    ),
    (TimerOccurrenceStatus.RECOVERING, OccurrenceAction.RECOVER_RUNNING): (
        TimerOccurrenceStatus.RUNNING
    ),
    (TimerOccurrenceStatus.RECOVERING, OccurrenceAction.RECOVER_DELIVERY): (
        TimerOccurrenceStatus.WAITING_DELIVERY
    ),
    (TimerOccurrenceStatus.RECOVERING, OccurrenceAction.RECOVER_COMPLETE): (
        TimerOccurrenceStatus.COMPLETED
    ),
}

_STATUS_OCCURRENCE_TARGETS = {
    (TimerOccurrenceStatus.SCHEDULED, OccurrenceAction.PAUSE): TimerOccurrenceStatus.PAUSED,
    (TimerOccurrenceStatus.WAITING, OccurrenceAction.PAUSE): TimerOccurrenceStatus.PAUSED,
    (TimerOccurrenceStatus.CLAIMED, OccurrenceAction.PAUSE): TimerOccurrenceStatus.PAUSED,
    (TimerOccurrenceStatus.SCHEDULED, OccurrenceAction.CANCEL): TimerOccurrenceStatus.CANCELLED,
    (TimerOccurrenceStatus.WAITING, OccurrenceAction.CANCEL): TimerOccurrenceStatus.CANCELLED,
    (TimerOccurrenceStatus.CLAIMED, OccurrenceAction.CANCEL): TimerOccurrenceStatus.CANCELLED,
    (TimerOccurrenceStatus.RUNNING, OccurrenceAction.CANCEL): TimerOccurrenceStatus.CANCELLED,
    (TimerOccurrenceStatus.PAUSED, OccurrenceAction.CANCEL): TimerOccurrenceStatus.CANCELLED,
    (TimerOccurrenceStatus.RECOVERING, OccurrenceAction.CANCEL): TimerOccurrenceStatus.CANCELLED,
    (TimerOccurrenceStatus.CLAIMED, OccurrenceAction.FAIL): TimerOccurrenceStatus.FAILED,
    (TimerOccurrenceStatus.RUNNING, OccurrenceAction.FAIL): TimerOccurrenceStatus.FAILED,
    (TimerOccurrenceStatus.WAITING_DELIVERY, OccurrenceAction.FAIL): TimerOccurrenceStatus.FAILED,
    (TimerOccurrenceStatus.RECOVERING, OccurrenceAction.RECOVER_FAIL): TimerOccurrenceStatus.FAILED,
}

_TransitionValues: TypeAlias = tuple[
    TimerOccurrenceStatus,
    int,
    ExecutionEnvelopeRef | None,
    DeliveryAssociationRef | None,
    TimerOccurrenceStatus | None,
]


def _operation_fingerprint(action: StrEnum, **parts: str) -> str:
    payload = {"action": action.value, **parts}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_replay(
    *,
    current_version: int,
    expected_version: int,
    current_key: str,
    current_fingerprint: str,
    key: IdempotencyKey,
    fingerprint: str,
) -> bool:
    if current_key == key.value:
        if current_fingerprint != fingerprint:
            raise fail(TimerErrorCode.IDEMPOTENCY_CONFLICT)
        return True
    if current_version != expected_version:
        raise fail(TimerErrorCode.VERSION_CONFLICT)
    return False


def transition_rule(
    rule: TimerRule,
    action: RuleAction,
    *,
    expected_version: int,
    operation_key: IdempotencyKey,
) -> TimerRule:
    fingerprint = _operation_fingerprint(action)
    if _is_replay(
        current_version=rule.version,
        expected_version=expected_version,
        current_key=rule.last_operation_key,
        current_fingerprint=rule.last_operation_fingerprint,
        key=operation_key,
        fingerprint=fingerprint,
    ):
        return rule
    targets = {
        (TimerRuleStatus.ACTIVE, RuleAction.PAUSE): TimerRuleStatus.PAUSED,
        (TimerRuleStatus.ACTIVE, RuleAction.CANCEL): TimerRuleStatus.CANCELLED,
        (TimerRuleStatus.PAUSED, RuleAction.RESUME): TimerRuleStatus.ACTIVE,
        (TimerRuleStatus.PAUSED, RuleAction.CANCEL): TimerRuleStatus.CANCELLED,
    }
    target = targets.get((rule.status, action))
    if target is None:
        raise fail(TimerErrorCode.INVALID_STATE)
    return replace(
        rule,
        status=target,
        version=rule.version + 1,
        last_operation_key=operation_key.value,
        last_operation_fingerprint=fingerprint,
    )


def transition_occurrence(
    occurrence: TimerOccurrence,
    action: OccurrenceAction,
    *,
    expected_version: int,
    operation_key: IdempotencyKey,
    now: datetime,
    execution_ref: ExecutionEnvelopeRef | None = None,
    delivery_ref: DeliveryAssociationRef | None = None,
) -> TimerOccurrence:
    now = require_aware(now)
    fingerprint = _operation_fingerprint(
        action,
        execution_ref=execution_ref.value if execution_ref else "",
        delivery_ref=delivery_ref.value if delivery_ref else "",
    )
    if _is_replay(
        current_version=occurrence.version,
        expected_version=expected_version,
        current_key=occurrence.last_operation_key,
        current_fingerprint=occurrence.last_operation_fingerprint,
        key=operation_key,
        fingerprint=fingerprint,
    ):
        return occurrence

    target, generation, next_execution, next_delivery, recovery_from = _transition_values(
        occurrence,
        action,
        now=now,
        execution_ref=execution_ref,
        delivery_ref=delivery_ref,
    )
    if action not in {OccurrenceAction.START_PROVIDER, OccurrenceAction.HANDOFF_DELIVERY} and (
        execution_ref is not None or delivery_ref is not None
    ):
        raise fail(TimerErrorCode.INVALID_REFERENCE)
    if target is TimerOccurrenceStatus.WAITING_DELIVERY and next_delivery is None:
        raise fail(TimerErrorCode.INVALID_REFERENCE)
    if target is TimerOccurrenceStatus.RUNNING and next_execution is None:
        raise fail(TimerErrorCode.INVALID_REFERENCE)
    return replace(
        occurrence,
        status=target,
        version=occurrence.version + 1,
        generation=generation,
        execution_ref=next_execution,
        delivery_ref=next_delivery,
        recovery_from=recovery_from,
        last_operation_key=operation_key.value,
        last_operation_fingerprint=fingerprint,
    )


def _transition_values(
    occurrence: TimerOccurrence,
    action: OccurrenceAction,
    *,
    now: datetime,
    execution_ref: ExecutionEnvelopeRef | None,
    delivery_ref: DeliveryAssociationRef | None,
) -> _TransitionValues:
    target = _SIMPLE_OCCURRENCE_TARGETS.get((occurrence.status, action))
    if target is not None:
        return _target_values(occurrence, target)
    values = _reference_transition_values(
        occurrence,
        action,
        execution_ref=execution_ref,
        delivery_ref=delivery_ref,
    )
    if values is not None:
        return values
    values = _status_transition_values(occurrence, action, now=now)
    if values is not None:
        return values
    values = _recovery_transition_values(occurrence, action)
    if values is not None:
        return values
    raise fail(TimerErrorCode.INVALID_STATE)


def _target_values(occurrence: TimerOccurrence, target: TimerOccurrenceStatus) -> _TransitionValues:
    return (
        target,
        occurrence.generation,
        occurrence.execution_ref,
        occurrence.delivery_ref,
        None,
    )


def _reference_transition_values(
    occurrence: TimerOccurrence,
    action: OccurrenceAction,
    *,
    execution_ref: ExecutionEnvelopeRef | None,
    delivery_ref: DeliveryAssociationRef | None,
) -> _TransitionValues | None:
    if action is OccurrenceAction.START_PROVIDER:
        if occurrence.status is not TimerOccurrenceStatus.CLAIMED:
            return None
        if execution_ref is None or delivery_ref is not None:
            raise fail(TimerErrorCode.INVALID_REFERENCE)
        return (
            TimerOccurrenceStatus.RUNNING,
            occurrence.generation,
            execution_ref,
            occurrence.delivery_ref,
            None,
        )
    if action is OccurrenceAction.HANDOFF_DELIVERY:
        if occurrence.status is not TimerOccurrenceStatus.RUNNING:
            return None
        if delivery_ref is None or execution_ref is not None:
            raise fail(TimerErrorCode.INVALID_REFERENCE)
        return (
            TimerOccurrenceStatus.WAITING_DELIVERY,
            occurrence.generation,
            occurrence.execution_ref,
            delivery_ref,
            None,
        )
    return None


def _status_transition_values(
    occurrence: TimerOccurrence, action: OccurrenceAction, *, now: datetime
) -> _TransitionValues | None:
    target = _STATUS_OCCURRENCE_TARGETS.get((occurrence.status, action))
    if target is not None:
        return _target_values(occurrence, target)
    if action is OccurrenceAction.RESUME and occurrence.status is TimerOccurrenceStatus.PAUSED:
        target = (
            TimerOccurrenceStatus.WAITING
            if occurrence.original_due_at <= now
            else TimerOccurrenceStatus.SCHEDULED
        )
        return _target_values(occurrence, target)
    return None


def _recovery_transition_values(
    occurrence: TimerOccurrence, action: OccurrenceAction
) -> _TransitionValues | None:
    if (
        action is OccurrenceAction.SUPERSEDE_REQUEUE
        and occurrence.status is TimerOccurrenceStatus.RUNNING
    ):
        return TimerOccurrenceStatus.WAITING, occurrence.generation + 1, None, None, None
    if action is OccurrenceAction.ENTER_RECOVERY and occurrence.status in {
        TimerOccurrenceStatus.CLAIMED,
        TimerOccurrenceStatus.RUNNING,
        TimerOccurrenceStatus.WAITING_DELIVERY,
    }:
        return (
            TimerOccurrenceStatus.RECOVERING,
            occurrence.generation,
            occurrence.execution_ref,
            occurrence.delivery_ref,
            occurrence.status,
        )
    if (
        action is OccurrenceAction.RECOVER_REQUEUE
        and occurrence.status is TimerOccurrenceStatus.RECOVERING
    ):
        return TimerOccurrenceStatus.WAITING, occurrence.generation + 1, None, None, None
    return None


__all__ = [
    "OccurrenceAction",
    "RuleAction",
    "transition_occurrence",
    "transition_rule",
]
