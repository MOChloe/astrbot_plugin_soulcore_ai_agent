"""Run-scoped Main Core adapter for Timer reads and deferred mutations."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from ...contracts.models import CoreWakeRequest, WakeSource
from . import main_core_views as timer_views
from .constants import (
    MAX_CREATE_ACTIONS_PER_RUN,
    MAX_MANAGE_ACTIONS_PER_RUN,
    MAX_SEMANTIC_CANDIDATES,
)
from .contracts import (
    CreateTimerCommand,
    CreateTimerOutcome,
    CreateTimerResult,
    ManageTimerAction,
    ManageTimerCommand,
    ManageTimerResult,
    PreparedTimerCreation,
    ReviseTimerCommand,
    ReviseTimerResult,
)
from .domain import (
    IdempotencyKey,
    OccurrenceStableRef,
    OpaqueTimerRef,
    SourceMessageRef,
    SourceRunRef,
    TimerOccurrence,
    TimerOccurrenceId,
    TimerOccurrenceStatus,
    TimerRule,
    TimerRuleId,
    TimerRuleRevision,
    TimerRuleStatus,
    TimerScope,
    normalize_prompt,
)
from .errors import TimerDomainError, TimerErrorCode, fail
from .natural_time import (
    ArrangementChangeKind,
    ArrangementChangeResolution,
    NaturalTimeStatus,
    interpret_arrangement_change,
    interpret_natural_time,
    natural_time_candidate_payload,
)
from .projection import TimerRefTarget, project_candidates
from .rules import (
    exact_timer_fingerprint,
    next_occurrence,
    parse_and_normalize_rule,
)


class TimerMainCoreReader(Protocol):
    """Persistence operations available to one Timer Main Core run."""

    async def find_exact(self, scope: TimerScope, fingerprint: str) -> Any | None: ...

    async def list_manageable(
        self,
        scope: TimerScope,
        source_run_ref: SourceRunRef,
        *,
        limit: int,
        query: str = "",
    ) -> tuple[Any, ...]: ...

    async def resolve_run_ref(
        self,
        scope: TimerScope,
        source_run_ref: SourceRunRef,
        opaque_ref: OpaqueTimerRef,
        target: TimerRefTarget,
    ) -> tuple[Any, Any | None, int] | None: ...

    async def get_rule(self, scope: TimerScope, rule_id: TimerRuleId) -> TimerRule | None: ...

    def create_or_reuse_in_transaction(
        self,
        conn: sqlite3.Connection,
        command: CreateTimerCommand,
        prepared: PreparedTimerCreation,
    ) -> CreateTimerResult: ...

    def apply_management_in_transaction(
        self,
        conn: sqlite3.Connection,
        command: ManageTimerCommand,
    ) -> ManageTimerResult: ...

    def apply_revision_in_transaction(
        self,
        conn: sqlite3.Connection,
        command: ReviseTimerCommand,
    ) -> ReviseTimerResult: ...


@dataclass(frozen=True, slots=True)
class PendingTimerCreation:
    rule: dict[str, object]
    prompt: str
    source_message_refs: tuple[SourceMessageRef, ...]
    idempotency_key: IdempotencyKey
    time_expression: str = ""
    timezone: str = ""


@dataclass(frozen=True, slots=True)
class PendingTimerManagement:
    opaque_ref: OpaqueTimerRef
    target: TimerRefTarget
    action: ManageTimerAction
    expected_version: int
    idempotency_key: IdempotencyKey


@dataclass(frozen=True, slots=True)
class PendingTimerRevision:
    opaque_ref: OpaqueTimerRef
    expected_version: int
    idempotency_key: IdempotencyKey
    rule: dict[str, object] | None = None
    prompt: str | None = None
    time_expression: str = ""
    timezone: str = ""


@dataclass(frozen=True, slots=True)
class CreatedTimerTimelineEvent:
    """One newly committed arrangement to append to normal private dialogue history."""

    idempotency_key: str
    schedule_summary: str
    action_template: str


class TimerMainCoreService:
    """Open isolated run contexts without granting the model persistence access."""

    def __init__(self, reader: TimerMainCoreReader) -> None:
        self._reader = reader

    def open_run(
        self,
        *,
        profile_id: str,
        instance_id: str,
        core_run_id: int,
        checked_at: datetime,
        source_message_refs: tuple[SourceMessageRef, ...] = (),
        timezone: str = "",
    ) -> TimerRunToolContext:
        return TimerRunToolContext(
            reader=self._reader,
            scope=TimerScope(profile_id, instance_id),
            source_run_ref=SourceRunRef(f"core-run:{core_run_id}"),
            checked_at=checked_at,
            source_message_refs=source_message_refs,
            timezone=timezone,
        )


def build_timer_wake_request(
    *,
    profile_id: str,
    instance_id: str,
    route_umo: str,
    prompt: str,
    admission_fence: Mapping[str, object],
    source_message_refs: tuple[SourceMessageRef, ...] = (),
    caused_by_run_ref: str = "",
    ai_task_id: int | None = None,
    requested_at: datetime | None = None,
) -> CoreWakeRequest:
    """Build TASK-0019's public, fence-carrying Timer Main Core request."""

    values: dict[str, Any] = {
        "profile_id": profile_id,
        "instance_id": instance_id,
        "source": WakeSource.TIMER,
        "reason": "Timer到期",
        "route_umo": route_umo,
        "timer_prompt": normalize_prompt(prompt),
        "metadata": {
            "timer_admission_fence": dict(admission_fence),
            "caused_by_run_ref": str(caused_by_run_ref or ""),
            "source_message_refs": tuple(item.value for item in source_message_refs),
        },
    }
    if ai_task_id is not None and int(ai_task_id) > 0:
        values["metadata"].update(ai_task_managed=True, ai_task_id=int(ai_task_id))
    if requested_at is not None:
        values["requested_at"] = requested_at
    return CoreWakeRequest(**values)


class TimerRunToolContext:
    """Collect mutations for one normal Run; persistence waits for final commit."""

    def __init__(
        self,
        *,
        reader: TimerMainCoreReader,
        scope: TimerScope,
        source_run_ref: SourceRunRef,
        checked_at: datetime,
        source_message_refs: tuple[SourceMessageRef, ...],
        timezone: str,
    ) -> None:
        self._reader = reader
        self.scope = scope
        self.source_run_ref = source_run_ref
        self.checked_at = checked_at
        self.source_message_refs = source_message_refs
        self.timezone = timer_views.validated_timezone(timezone)
        self.creations: list[PendingTimerCreation] = []
        self.managements: list[PendingTimerManagement] = []
        self.revisions: list[PendingTimerRevision] = []
        self._creation_fingerprints: dict[str, int] = {}
        self._management_fingerprints: dict[str, int] = {}
        self._revision_fingerprints: dict[str, int] = {}

    async def stage_natural_creation(
        self,
        *,
        time_expression: str,
        action_text: str,
    ) -> dict[str, Any]:
        if not self.timezone:
            raise fail(TimerErrorCode.INVALID_TIMEZONE)
        resolution = interpret_natural_time(
            time_expression,
            now=self.checked_at,
            timezone=self.timezone,
        )
        if resolution.status is NaturalTimeStatus.INVALID:
            return {"ok": False, "message": resolution.message}
        if resolution.status is NaturalTimeStatus.AMBIGUOUS:
            return {
                "status": "needs_clarification",
                "message": resolution.message,
                "candidates": [
                    natural_time_candidate_payload(item) for item in resolution.candidates
                ],
            }
        candidate = resolution.unique
        if candidate is None or candidate.rule is None:
            raise fail(TimerErrorCode.INVALID_RULE)
        result = await self.stage_creation(
            rule=candidate.rule,
            prompt=action_text,
            time_expression=str(time_expression),
            timezone=candidate.timezone,
        )
        result.update(
            time_expression=str(time_expression).strip(),
            timezone=candidate.timezone,
            schedule_summary=candidate.summary,
            action_summary=timer_views.short_quote(normalize_prompt(action_text)),
        )
        if result.get("status") == "pending_final_commit":
            result["message"] = "安排已经明确，将随本次行动最终提交后生效"
        return result

    async def stage_creation(
        self,
        *,
        rule: Mapping[str, object],
        prompt: str,
        time_expression: str = "",
        timezone: str = "",
    ) -> dict[str, Any]:
        normalized_prompt = normalize_prompt(prompt)
        raw_rule = dict(rule)
        checked_schedule = parse_and_normalize_rule(raw_rule, committed_at=self.checked_at)
        expression = str(time_expression or "").strip()
        timer_timezone = str(timezone or "").strip()
        local_fingerprint = _fingerprint(
            "create", raw_rule, normalized_prompt, expression, timer_timezone
        )
        prior = self._creation_fingerprints.get(local_fingerprint)
        if prior is not None:
            return timer_views.pending_creation_payload(self.creations[prior], idempotent=True)
        if len(self.creations) >= MAX_CREATE_ACTIONS_PER_RUN:
            raise fail(TimerErrorCode.LIMIT_EXCEEDED)

        exact = await self._reader.find_exact(
            self.scope,
            exact_timer_fingerprint(self.scope, checked_schedule, normalized_prompt),
        )
        if exact is not None:
            return {"status": "already_exists", "message": "相同安排已经存在"}

        intent = PendingTimerCreation(
            rule=raw_rule,
            prompt=normalized_prompt,
            source_message_refs=self.source_message_refs,
            idempotency_key=IdempotencyKey(
                f"main-core-timer-create:{self.source_run_ref.value}:{len(self.creations) + 1}"
            ),
            time_expression=expression,
            timezone=timer_timezone,
        )
        self._creation_fingerprints[local_fingerprint] = len(self.creations)
        self.creations.append(intent)
        return timer_views.pending_creation_payload(intent, idempotent=False)

    async def list_timers(self, *, limit: int, query: str = "") -> dict[str, Any]:
        if isinstance(limit, bool) or not 1 <= limit <= MAX_SEMANTIC_CANDIDATES:
            raise fail(TimerErrorCode.OUT_OF_RANGE)
        query = str(query or "").strip()
        if len(query) > 200:
            raise fail(TimerErrorCode.OUT_OF_RANGE)
        sources = await self._reader.list_manageable(
            self.scope,
            self.source_run_ref,
            limit=limit,
            query=query,
        )
        candidates = project_candidates(tuple(sources))
        return {
            "status": "ok",
            "timers": [timer_views.candidate_payload(item) for item in candidates],
        }

    async def list_arrangements(self, *, limit: int, query: str = "") -> dict[str, Any]:
        if isinstance(limit, bool) or not 1 <= limit <= MAX_SEMANTIC_CANDIDATES:
            raise fail(TimerErrorCode.OUT_OF_RANGE)
        query = str(query or "").strip()
        if len(query) > 200:
            raise fail(TimerErrorCode.OUT_OF_RANGE)
        range_kind = timer_views.arrangement_range_kind(query)
        sources = await self._reader.list_manageable(
            self.scope,
            self.source_run_ref,
            limit=MAX_SEMANTIC_CANDIDATES * 2,
            query="" if range_kind else query,
        )
        series = tuple(item for item in sources if item.target is TimerRefTarget.SERIES)
        if range_kind:
            series = tuple(
                item
                for item in series
                if timer_views.arrangement_in_range(
                    item.next_due_at,
                    range_kind=range_kind,
                    checked_at=self.checked_at,
                    timezone=self.timezone,
                )
            )
        series = tuple(
            sorted(
                series,
                key=lambda item: (
                    item.next_due_at is None,
                    item.next_due_at or self.checked_at,
                    item.opaque_ref.value,
                ),
            )[:limit]
        )
        return {
            "status": "ok",
            "arrangements": [
                timer_views.arrangement_payload(item, self.timezone) for item in series
            ],
        }

    async def stage_natural_adjustment(
        self,
        *,
        target: str,
        change: str,
    ) -> dict[str, Any]:
        if not self.timezone:
            raise fail(TimerErrorCode.INVALID_TIMEZONE)
        change_resolution = interpret_arrangement_change(
            change,
            now=self.checked_at,
            timezone=self.timezone,
        )
        if change_resolution.status is NaturalTimeStatus.INVALID:
            return {"ok": False, "message": change_resolution.message}
        if change_resolution.status is NaturalTimeStatus.AMBIGUOUS:
            return _change_clarification_payload(change_resolution)
        source_result = await self._resolve_arrangement_target(target)
        if isinstance(source_result, dict):
            return source_result
        result = self._stage_resolved_adjustment(source_result, change_resolution, change)
        result["message"] = "调整已经明确，将随本次行动最终提交后生效"
        return result

    def _stage_resolved_adjustment(
        self,
        source: Any,
        change_resolution: ArrangementChangeResolution,
        change: str,
    ) -> dict[str, Any]:
        kind = change_resolution.kind
        if kind in {
            ArrangementChangeKind.PAUSE,
            ArrangementChangeKind.RESUME,
            ArrangementChangeKind.CANCEL,
        }:
            result = self._stage_management_source(source, kind.value)
            result["change_summary"] = {
                ArrangementChangeKind.PAUSE: "暂停",
                ArrangementChangeKind.RESUME: "继续",
                ArrangementChangeKind.CANCEL: "取消",
            }[kind]
            return result
        if kind is ArrangementChangeKind.RESCHEDULE:
            return self._stage_reschedule(source, change_resolution, change)
        if kind is ArrangementChangeKind.REWRITE:
            result = self._stage_revision_source(source, prompt=change_resolution.action_text)
            result["change_summary"] = (
                f"改内容为 {timer_views.short_quote(change_resolution.action_text)}"
            )
            return result
        raise fail(TimerErrorCode.INVALID_RULE)

    def _stage_reschedule(
        self,
        source: Any,
        change_resolution: ArrangementChangeResolution,
        change: str,
    ) -> dict[str, Any]:
        time_resolution = change_resolution.time_resolution
        candidate = time_resolution.unique if time_resolution is not None else None
        if candidate is None or candidate.rule is None:
            raise fail(TimerErrorCode.INVALID_RULE)
        if (
            source.rule.schedule.kind.value in {"WEEKLY", "YEARLY"}
            and candidate.rule["kind"] in {"ABSOLUTE", "RELATIVE"}
            and not timer_views.explicit_entire_series(change)
        ):
            return _recurring_change_clarification(source, self.timezone)
        result = self._stage_revision_source(
            source,
            rule=candidate.rule,
            time_expression=candidate.time_expression,
            timezone=candidate.timezone,
        )
        result["change_summary"] = f"改时间为 {candidate.summary}"
        return result

    async def stage_management(
        self,
        *,
        timer_ref: str,
        target: str,
        action: str,
    ) -> dict[str, Any]:
        opaque_ref = OpaqueTimerRef(str(timer_ref or "").strip())
        try:
            parsed_target = TimerRefTarget(str(target or "").strip().upper())
            parsed_action = ManageTimerAction(str(action or "").strip().upper())
        except ValueError:
            raise fail(TimerErrorCode.INVALID_RULE) from None
        resolved = await self._reader.resolve_run_ref(
            self.scope,
            self.source_run_ref,
            opaque_ref,
            parsed_target,
        )
        if resolved is None:
            raise fail(TimerErrorCode.INVALID_REFERENCE)
        expected_version = int(resolved[2])
        fingerprint = _fingerprint(
            "manage", opaque_ref.value, parsed_target.value, parsed_action.value, expected_version
        )
        prior = self._management_fingerprints.get(fingerprint)
        if prior is not None:
            return timer_views.pending_management_payload(self.managements[prior], idempotent=True)
        if len(self.managements) >= MAX_MANAGE_ACTIONS_PER_RUN:
            raise fail(TimerErrorCode.LIMIT_EXCEEDED)
        intent = PendingTimerManagement(
            opaque_ref=opaque_ref,
            target=parsed_target,
            action=parsed_action,
            expected_version=expected_version,
            idempotency_key=IdempotencyKey(
                f"main-core-timer-manage:{self.source_run_ref.value}:{len(self.managements) + 1}"
            ),
        )
        self._management_fingerprints[fingerprint] = len(self.managements)
        self.managements.append(intent)
        return timer_views.pending_management_payload(intent, idempotent=False)

    async def _resolve_arrangement_target(
        self,
        target: str,
    ) -> Any:
        text = str(target or "").strip()
        if not text:
            return {"ok": False, "message": "需要说明要调整哪件安排"}
        try:
            opaque_ref = OpaqueTimerRef(text)
        except TimerDomainError:
            opaque_ref = None
        if opaque_ref is not None:
            resolved = await self._reader.resolve_run_ref(
                self.scope,
                self.source_run_ref,
                opaque_ref,
                TimerRefTarget.SERIES,
            )
            if resolved is not None:
                rule = await self._reader.get_rule(self.scope, resolved[0])
                if rule is not None:
                    return timer_views.source_from_resolved(opaque_ref, rule, self.checked_at)
        sources = await self._reader.list_manageable(
            self.scope,
            self.source_run_ref,
            limit=MAX_SEMANTIC_CANDIDATES * 2,
            query=text,
        )
        series = tuple(item for item in sources if item.target is TimerRefTarget.SERIES)
        if len(series) == 1:
            return series[0]
        if not series:
            return {
                "status": "needs_clarification",
                "message": "没有找到能唯一对应的安排；可以先看看近期安排",
                "candidates": [],
            }
        return {
            "status": "needs_clarification",
            "message": "这个描述对应多件安排；请选择其中一项短引用",
            "candidates": [
                timer_views.arrangement_payload(item, self.timezone)
                for item in series[:MAX_SEMANTIC_CANDIDATES]
            ],
        }

    def _stage_management_source(
        self,
        source: Any,
        action: str,
    ) -> dict[str, Any]:
        parsed_action = ManageTimerAction(action)
        fingerprint = _fingerprint(
            "manage",
            source.opaque_ref.value,
            TimerRefTarget.SERIES.value,
            parsed_action.value,
            source.rule.version,
        )
        prior = self._management_fingerprints.get(fingerprint)
        if prior is not None:
            return timer_views.pending_management_payload(self.managements[prior], idempotent=True)
        if len(self.managements) + len(self.revisions) >= MAX_MANAGE_ACTIONS_PER_RUN:
            raise fail(TimerErrorCode.LIMIT_EXCEEDED)
        intent = PendingTimerManagement(
            opaque_ref=source.opaque_ref,
            target=TimerRefTarget.SERIES,
            action=parsed_action,
            expected_version=source.rule.version,
            idempotency_key=IdempotencyKey(
                f"main-core-timer-manage:{self.source_run_ref.value}:"
                f"{len(self.managements) + len(self.revisions) + 1}"
            ),
        )
        self._management_fingerprints[fingerprint] = len(self.managements)
        self.managements.append(intent)
        return timer_views.pending_management_payload(intent, idempotent=False)

    def _stage_revision_source(
        self,
        source: Any,
        *,
        rule: Mapping[str, object] | None = None,
        prompt: str | None = None,
        time_expression: str = "",
        timezone: str = "",
    ) -> dict[str, Any]:
        raw_rule = dict(rule) if rule is not None else None
        normalized_prompt = normalize_prompt(prompt) if prompt is not None else None
        fingerprint = _fingerprint(
            "revise",
            source.opaque_ref.value,
            source.rule.version,
            raw_rule,
            normalized_prompt,
            time_expression,
            timezone,
        )
        prior = self._revision_fingerprints.get(fingerprint)
        if prior is not None:
            return timer_views.pending_revision_payload(self.revisions[prior], idempotent=True)
        if len(self.managements) + len(self.revisions) >= MAX_MANAGE_ACTIONS_PER_RUN:
            raise fail(TimerErrorCode.LIMIT_EXCEEDED)
        intent = PendingTimerRevision(
            opaque_ref=source.opaque_ref,
            expected_version=source.rule.version,
            idempotency_key=IdempotencyKey(
                f"main-core-timer-revise:{self.source_run_ref.value}:"
                f"{len(self.managements) + len(self.revisions) + 1}"
            ),
            rule=raw_rule,
            prompt=normalized_prompt,
            time_expression=str(time_expression or "").strip(),
            timezone=str(timezone or "").strip(),
        )
        self._revision_fingerprints[fingerprint] = len(self.revisions)
        self.revisions.append(intent)
        return timer_views.pending_revision_payload(intent, idempotent=False)

    def commit_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        committed_at: datetime,
        on_created: Callable[[CreatedTimerTimelineEvent], None] | None = None,
    ) -> tuple[CreateTimerResult | ManageTimerResult | ReviseTimerResult, ...]:
        """Apply staged intents through the repository's caller-owned transaction API."""

        if len(self.creations) > MAX_CREATE_ACTIONS_PER_RUN:
            raise fail(TimerErrorCode.LIMIT_EXCEEDED)
        if len(self.managements) > MAX_MANAGE_ACTIONS_PER_RUN:
            raise fail(TimerErrorCode.LIMIT_EXCEEDED)
        if len(self.managements) + len(self.revisions) > MAX_MANAGE_ACTIONS_PER_RUN:
            raise fail(TimerErrorCode.LIMIT_EXCEEDED)
        results: list[CreateTimerResult | ManageTimerResult | ReviseTimerResult] = []
        for ordinal, create_intent in enumerate(self.creations, start=1):
            command, prepared = self._prepare_creation(create_intent, ordinal, committed_at)
            result = self._reader.create_or_reuse_in_transaction(conn, command, prepared)
            results.append(result)
            if result.outcome is CreateTimerOutcome.CREATED and on_created is not None:
                on_created(
                    CreatedTimerTimelineEvent(
                        idempotency_key=f"{create_intent.idempotency_key.value}:timeline",
                        schedule_summary=timer_views.schedule_summary(
                            prepared.rule.schedule,
                            prepared.first_occurrence.original_due_at,
                            self.timezone,
                            prepared.rule.timezone,
                        ),
                        action_template=create_intent.prompt,
                    )
                )
        for manage_intent in self.managements:
            results.append(
                self._reader.apply_management_in_transaction(
                    conn,
                    ManageTimerCommand(
                        self.scope,
                        self.source_run_ref,
                        manage_intent.opaque_ref,
                        manage_intent.target,
                        manage_intent.action,
                        manage_intent.expected_version,
                        manage_intent.idempotency_key,
                    ),
                )
            )
        for revision in self.revisions:
            schedule = (
                parse_and_normalize_rule(revision.rule, committed_at=committed_at)
                if revision.rule is not None
                else None
            )
            results.append(
                self._reader.apply_revision_in_transaction(
                    conn,
                    ReviseTimerCommand(
                        self.scope,
                        self.source_run_ref,
                        revision.opaque_ref,
                        revision.expected_version,
                        revision.idempotency_key,
                        committed_at,
                        schedule=schedule,
                        prompt=revision.prompt,
                        time_expression=revision.time_expression,
                        timezone=revision.timezone,
                    ),
                )
            )
        return tuple(results)

    def _prepare_creation(
        self,
        intent: PendingTimerCreation,
        ordinal: int,
        committed_at: datetime,
    ) -> tuple[CreateTimerCommand, PreparedTimerCreation]:
        schedule = parse_and_normalize_rule(intent.rule, committed_at=committed_at)
        due_at = next_occurrence(schedule, after=committed_at)
        if due_at is None:
            raise fail(TimerErrorCode.INVALID_RULE)
        fingerprint = exact_timer_fingerprint(self.scope, schedule, intent.prompt)
        identity = _fingerprint(
            "commit",
            *self.scope.fingerprint_parts,
            self.source_run_ref.value,
            ordinal,
        )
        rule_id = TimerRuleId(f"rule:{identity[:48]}")
        occurrence_id = TimerOccurrenceId(f"occurrence:{identity[:48]}")
        rule = TimerRule(
            rule_id,
            self.scope,
            schedule,
            intent.prompt,
            fingerprint,
            TimerRuleStatus.ACTIVE,
            1,
            1,
            committed_at,
            self.source_run_ref,
            intent.source_message_refs,
            time_expression=intent.time_expression,
            timezone=intent.timezone,
            revisions=(
                (
                    TimerRuleRevision(
                        1,
                        committed_at,
                        schedule,
                        intent.prompt,
                        intent.time_expression,
                        intent.timezone,
                    ),
                )
                if intent.time_expression or intent.timezone
                else ()
            ),
        )
        occurrence = TimerOccurrence(
            occurrence_id,
            OccurrenceStableRef(f"stable:{identity[:48]}"),
            rule_id,
            self.scope,
            due_at,
            TimerOccurrenceStatus.SCHEDULED,
            1,
            0,
            1,
            committed_at,
        )
        command = CreateTimerCommand(
            self.scope,
            schedule,
            intent.prompt,
            fingerprint,
            self.source_run_ref,
            intent.idempotency_key,
            intent.source_message_refs,
        )
        return command, PreparedTimerCreation(rule, occurrence)


def _change_clarification_payload(
    resolution: ArrangementChangeResolution,
) -> dict[str, Any]:
    time_resolution = resolution.time_resolution
    candidates = (
        [natural_time_candidate_payload(item) for item in time_resolution.candidates]
        if time_resolution is not None
        else []
    )
    return {
        "status": "needs_clarification",
        "message": resolution.message,
        "candidates": candidates,
    }


def _recurring_change_clarification(source: Any, timezone: str) -> dict[str, Any]:
    payload = timer_views.arrangement_payload(source, timezone)
    return {
        "status": "needs_clarification",
        "message": "这是重复安排；请说明只改下一次，还是修改整个重复安排",
        "candidates": [
            {**payload, "summary": "只改下一次（请在调整中明确写“下一次”）"},
            {**payload, "summary": "修改整个重复安排（请明确写“整个安排”）"},
        ],
    }


def _fingerprint(kind: str, *values: object) -> str:
    payload = json.dumps([kind, *values], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "CreatedTimerTimelineEvent",
    "PendingTimerCreation",
    "PendingTimerManagement",
    "PendingTimerRevision",
    "TimerMainCoreReader",
    "TimerMainCoreService",
    "TimerRunToolContext",
    "build_timer_wake_request",
]
