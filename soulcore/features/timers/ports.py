"""Narrow Timer persistence and clock boundaries; no universal repository."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from .contracts import (
    RollOccurrenceCommand,
)
from .domain import (
    TimerOccurrence,
    TimerOccurrenceId,
    TimerRule,
    TimerRuleId,
    TimerScope,
)
from .repository import (
    AdvanceOccurrenceCommand,
    MutateClaimedOccurrenceCommand,
    OccurrenceMutationResult,
    OccurrencePage,
    RulePage,
)
from .rules import OccurrenceRollPlan


class TimerOccurrenceRollReader(Protocol):
    async def get_rule(self, scope: TimerScope, rule_id: TimerRuleId) -> TimerRule | None: ...

    async def get_occurrence(
        self, scope: TimerScope, occurrence_id: TimerOccurrenceId
    ) -> TimerOccurrence | None: ...


class TimerOccurrenceRollWriter(Protocol):
    async def apply_roll(
        self, command: RollOccurrenceCommand, plan: OccurrenceRollPlan
    ) -> tuple[TimerOccurrence, ...]: ...


class TimerPageReader(Protocol):
    async def list_rules(
        self, scope: TimerScope, *, limit: int, after_created_sequence: int = 0
    ) -> RulePage: ...

    async def list_occurrences(
        self,
        scope: TimerScope,
        *,
        limit: int,
        after: tuple[datetime, int, str] | None = None,
    ) -> OccurrencePage: ...

    async def list_due_scheduled_occurrences(
        self, scope: TimerScope, *, through: datetime, limit: int
    ) -> tuple[TimerOccurrence, ...]: ...

    async def latest_occurrences_for_rules(
        self, scope: TimerScope, rule_ids: tuple[TimerRuleId, ...]
    ) -> tuple[TimerOccurrence, ...]: ...

    async def first_waiting_occurrence(self, scope: TimerScope) -> TimerOccurrence | None: ...


class TimerOccurrenceMutationWriter(Protocol):
    async def advance_occurrence(self, command: AdvanceOccurrenceCommand) -> TimerOccurrence: ...

    async def mutate_claimed_occurrence(
        self, command: MutateClaimedOccurrenceCommand
    ) -> OccurrenceMutationResult: ...


__all__ = [
    "TimerOccurrenceMutationWriter",
    "TimerPageReader",
    "TimerOccurrenceRollReader",
    "TimerOccurrenceRollWriter",
]
