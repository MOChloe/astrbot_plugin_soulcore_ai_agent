"Idempotent Timer scanner that enqueues one durable head task per instance."

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from ....contracts.models import CharacterInstance, RoleProfile
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
from ..task_identity import timer_run_task_idempotency_key
from ..transitions import OccurrenceAction
from .executor import TimerRuntimeExecutor
from .tasks import TIMER_RUN_TASK_TYPE


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


class TimerRuntimeProfiles(Protocol):
    async def list_profiles(self, *, include_orphaned: bool = True) -> Sequence[RoleProfile]: ...

    async def list_character_instances(
        self,
        profile_id: str,
        scope: str | None = None,
    ) -> Sequence[CharacterInstance]: ...


class TimerTaskCreator(Protocol):
    async def create_ai_task(
        self,
        profile_id: str,
        task_type: str,
        **values: object,
    ) -> Mapping[str, object]: ...


class TimerLifecycleRecovery(Protocol):
    async def recover_if_due(self, *, now: datetime, force: bool = False) -> int: ...


class TimerRuntimeWorker:
    def __init__(
        self,
        *,
        profiles: TimerRuntimeProfiles,
        timers: TimerPageReader,
        tasks: TimerTaskCreator,
        executor: TimerRuntimeExecutor,
        recovery: TimerRuntimeRecovery,
        lifecycle: TimerLifecycleRecovery | None = None,
        poll_seconds: float = 5.0,
    ) -> None:
        self._profiles = profiles
        self._timers = timers
        self._tasks = tasks
        self._executor = executor
        self._recovery = recovery
        self._lifecycle = lifecycle
        self._poll_seconds = max(0.1, float(poll_seconds))
        self._loop_task: asyncio.Task[None] | None = None
        self._closed = False
        self.last_error = ""

    @property
    def running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    def start(self) -> None:
        if self.running:
            return
        self._closed = False
        self._loop_task = asyncio.create_task(
            self._loop(),
            name="soulcore-timer-runtime",
        )

    async def stop(self) -> None:
        self._closed = True
        loop, self._loop_task = self._loop_task, None
        if loop is None:
            return
        loop.cancel()
        with suppress(asyncio.CancelledError):
            await loop

    async def scan_once(self, *, now: datetime | None = None) -> int:
        current = require_aware(now or datetime.now(UTC))
        enqueued = 0
        errors: list[str] = []
        if self._lifecycle is not None:
            try:
                await self._lifecycle.recover_if_due(now=current)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                errors.append(f"lifecycle: {type(exc).__name__}: {exc}")
        for profile in await self._profiles.list_profiles(include_orphaned=False):
            if not profile.enabled:
                continue
            try:
                instances = await self._profiles.list_character_instances(profile.profile_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                errors.append(f"{profile.profile_id}: {type(exc).__name__}: {exc}")
                continue
            for instance in instances:
                scope = TimerScope(profile.profile_id, instance.instance_id)
                try:
                    await self._executor.reconcile_scope(scope, now=current)
                    await self._recovery.reconcile_scope(scope, now=current)
                    await self._executor.mark_due(scope, now=current)
                    head = await self._waiting_head(scope)
                    if head is None:
                        continue
                    await self._enqueue(head)
                    enqueued += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    errors.append(
                        f"{scope.profile_id}/{scope.instance_id}: {type(exc).__name__}: {exc}"
                    )
                    continue
        self.last_error = "; ".join(errors)
        return enqueued

    async def _loop(self) -> None:
        while not self._closed:
            try:
                await self.scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(self._poll_seconds)

    async def _waiting_head(self, scope: TimerScope) -> TimerOccurrence | None:
        return await self._timers.first_waiting_occurrence(scope)

    async def _enqueue(self, occurrence: TimerOccurrence) -> None:
        await self._tasks.create_ai_task(
            occurrence.scope.profile_id,
            TIMER_RUN_TASK_TYPE,
            instance_id=occurrence.scope.instance_id,
            task_class="BACKGROUND",
            capability="conversation.timer_run",
            due_at=occurrence.original_due_at,
            priority=0,
            mutex_key="main-core-runtime",
            idempotency_key=timer_run_task_idempotency_key(
                occurrence.scope.profile_id,
                occurrence.scope.instance_id,
                occurrence.occurrence_id.value,
                occurrence.generation,
            ),
            generation=occurrence.generation + 1,
            input_data={
                "profile_id": occurrence.scope.profile_id,
                "instance_id": occurrence.scope.instance_id,
                "occurrence_id": occurrence.occurrence_id.value,
                "stable_ref": occurrence.stable_ref.value,
                "generation": occurrence.generation,
                "original_due_at": occurrence.original_due_at.isoformat(),
            },
            recovery_policy="RESTART_SAFE",
            retry_policy={"delays_hours": [1 / 60, 5 / 60, 15 / 60, 1]},
            max_attempts=4,
            actor_type="SYSTEM",
            actor_id="timer-runtime",
        )


__all__ = ["TimerRecoveryResult", "TimerRuntimeRecovery", "TimerRuntimeWorker"]
