"""Deterministic Timer downtime recovery and periodic missed coalescing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ..contracts import RollOccurrenceCommand
from ..domain import (
    IdempotencyKey,
    TimerOccurrence,
    TimerOccurrenceStatus,
    TimerRule,
    TimerRuleId,
    TimerRuleStatus,
    TimerScope,
    WeeklyTimerRule,
    YearlyTimerRule,
    require_aware,
)
from ..ports import (
    TimerOccurrenceMutationWriter,
    TimerOccurrenceRollReader,
    TimerOccurrenceRollWriter,
    TimerPageReader,
)
from ..repository import AdvanceOccurrenceCommand
from ..rules import plan_occurrence_roll
from ..transitions import OccurrenceAction


class TimerRecoveryRepository(
    TimerPageReader,
    TimerOccurrenceRollReader,
    TimerOccurrenceRollWriter,
    TimerOccurrenceMutationWriter,
    Protocol,
):
    pass


@dataclass(frozen=True, slots=True)
class TimerRecoveryResult:
    rules_checked: int = 0
    queued_missed: int = 0
    coalesced: int = 0
    future_materialized: int = 0


class TimerRuntimeRecovery:
    """Recover periodic rules without enumerating every missed wall-clock tick."""

    def __init__(self, repository: TimerRecoveryRepository) -> None:
        self._repository = repository

    async def reconcile_scope(
        self,
        scope: TimerScope,
        *,
        now: datetime,
    ) -> TimerRecoveryResult:
        now = require_aware(now)
        rules = await self._active_periodic_rules(scope)
        latest = await self._latest_occurrences(
            scope, tuple(rule.rule_id for rule in rules.values())
        )
        queued = coalesced = future = 0
        for rule_id, rule in rules.items():
            occurrence = latest.get(rule_id)
            if occurrence is None or occurrence.original_due_at > now:
                continue
            if occurrence.status in {
                TimerOccurrenceStatus.CLAIMED,
                TimerOccurrenceStatus.RUNNING,
                TimerOccurrenceStatus.WAITING_DELIVERY,
                TimerOccurrenceStatus.RECOVERING,
            }:
                continue
            plan = plan_occurrence_roll(
                rule.schedule,
                last_materialized_due_at=occurrence.original_due_at,
                recovered_at=now,
            )
            if occurrence.status in {
                TimerOccurrenceStatus.SCHEDULED,
                TimerOccurrenceStatus.WAITING,
            }:
                if plan.latest_missed_due_at is None:
                    if occurrence.status is TimerOccurrenceStatus.SCHEDULED:
                        await self._advance(occurrence, OccurrenceAction.MARK_DUE, now)
                        queued += 1
                else:
                    await self._advance(
                        occurrence,
                        OccurrenceAction.MARK_MISSED_COALESCED,
                        now,
                    )
                    coalesced += 1
            materialized = await self._repository.apply_roll(
                RollOccurrenceCommand(
                    scope=scope,
                    rule_id=rule.rule_id,
                    last_materialized_due_at=occurrence.original_due_at,
                    through=now,
                ),
                plan,
            )
            queued += sum(item.status is TimerOccurrenceStatus.WAITING for item in materialized)
            future += sum(item.status is TimerOccurrenceStatus.SCHEDULED for item in materialized)
            coalesced += plan.coalesced_count
        return TimerRecoveryResult(len(rules), queued, coalesced, future)

    async def _active_periodic_rules(self, scope: TimerScope) -> dict[str, TimerRule]:
        rules: dict[str, TimerRule] = {}
        cursor = 0
        while True:
            page = await self._repository.list_rules(
                scope,
                limit=64,
                after_created_sequence=cursor,
            )
            for rule in page.items:
                if rule.status is TimerRuleStatus.ACTIVE and isinstance(
                    rule.schedule, (WeeklyTimerRule, YearlyTimerRule)
                ):
                    rules[rule.rule_id.value] = rule
            if page.next_created_sequence is None:
                return rules
            cursor = page.next_created_sequence

    async def _latest_occurrences(
        self,
        scope: TimerScope,
        rule_ids: tuple[TimerRuleId, ...],
    ) -> dict[str, TimerOccurrence]:
        occurrences = await self._repository.latest_occurrences_for_rules(scope, rule_ids)
        return {item.rule_id.value: item for item in occurrences}

    async def _advance(
        self,
        occurrence: TimerOccurrence,
        action: OccurrenceAction,
        now: datetime,
    ) -> None:
        payload = (
            f"{action.value}:{occurrence.scope.profile_id}:{occurrence.scope.instance_id}:"
            f"{occurrence.occurrence_id.value}:{occurrence.version}:{occurrence.generation}"
        )
        await self._repository.advance_occurrence(
            AdvanceOccurrenceCommand(
                scope=occurrence.scope,
                occurrence_id=occurrence.occurrence_id,
                action=action,
                expected_version=occurrence.version,
                expected_generation=occurrence.generation,
                now=now,
                idempotency_key=IdempotencyKey(
                    f"timer-recovery:{hashlib.sha256(payload.encode()).hexdigest()}"
                ),
            )
        )


__all__ = ["TimerRecoveryResult", "TimerRuntimeRecovery"]
