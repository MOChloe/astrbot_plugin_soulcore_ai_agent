"""Atomic SQLite repository for Timer rules, occurrences, and occupancy."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import datetime

from ....storage.sqlite.codec import encode_datetime
from ....storage.sqlite.repository import SqliteRepository
from ..constants import (
    MAX_NONTERMINAL_OCCURRENCES_PER_INSTANCE,
    MAX_NONTERMINAL_RULES_PER_INSTANCE,
)
from ..contracts import (
    CreateTimerCommand,
    CreateTimerOutcome,
    CreateTimerResult,
    ManageTimerAction,
    ManageTimerCommand,
    ManageTimerOutcome,
    ManageTimerResult,
    PreparedTimerCreation,
    ReviseTimerCommand,
    ReviseTimerResult,
    RollOccurrenceCommand,
)
from ..domain import (
    IdempotencyKey,
    OccurrenceStableRef,
    OpaqueTimerRef,
    SourceRunRef,
    TimerOccurrence,
    TimerOccurrenceId,
    TimerOccurrenceStatus,
    TimerRule,
    TimerRuleId,
    TimerRuleRevision,
    TimerRuleStatus,
    TimerScope,
)
from ..errors import TimerDomainError, TimerErrorCode, fail
from ..projection import TimerProjectionSource, TimerRefTarget
from ..repository import (
    AdvanceOccurrenceCommand,
    MutateClaimedOccurrenceCommand,
    OccurrenceMutationResult,
    OccurrencePage,
    RulePage,
)
from ..rules import OccurrenceRollPlan, canonical_rule, exact_timer_fingerprint, next_occurrence
from ..transitions import OccurrenceAction, RuleAction, transition_occurrence, transition_rule
from .codec import (
    decode_create_result,
    decode_manage_result,
    decode_occurrence,
    decode_revise_result,
    encode_create_result,
    encode_manage_result,
    encode_revise_result,
    rule_columns,
)
from .lifecycle import TimerLifecycleSqliteMixin, TimerProjectionQueriesMixin
from .operations import TimerSqliteOperations, receipt_time


class SqliteTimerRepository(
    TimerLifecycleSqliteMixin,
    TimerSqliteOperations,
    TimerProjectionQueriesMixin,
    SqliteRepository,
):
    async def find_exact(self, scope: TimerScope, fingerprint: str) -> TimerRule | None:
        return await self.db.call(lambda conn: self._find_exact(conn, scope, fingerprint))

    async def get_rule(self, scope: TimerScope, rule_id: TimerRuleId) -> TimerRule | None:
        return await self.db.call(lambda conn: self._get_rule(conn, scope, rule_id))

    async def get_occurrence(
        self, scope: TimerScope, occurrence_id: TimerOccurrenceId
    ) -> TimerOccurrence | None:
        return await self.db.call(lambda conn: self._get_occurrence(conn, scope, occurrence_id))

    async def list_rules(
        self, scope: TimerScope, *, limit: int, after_created_sequence: int = 0
    ) -> RulePage:
        _page_limit(limit)
        return await self.db.call(
            lambda conn: self._list_rules(conn, scope, limit, after_created_sequence)
        )

    async def list_occurrences(
        self,
        scope: TimerScope,
        *,
        limit: int,
        after: tuple[datetime, int, str] | None = None,
    ) -> OccurrencePage:
        _page_limit(limit)
        return await self.db.call(lambda conn: self._list_occurrences(conn, scope, limit, after))

    async def list_due_scheduled_occurrences(
        self, scope: TimerScope, *, through: datetime, limit: int
    ) -> tuple[TimerOccurrence, ...]:
        _page_limit(limit)
        return await self.db.call(
            lambda conn: self._list_due_scheduled_occurrences(conn, scope, through, limit)
        )

    async def latest_occurrences_for_rules(
        self, scope: TimerScope, rule_ids: tuple[TimerRuleId, ...]
    ) -> tuple[TimerOccurrence, ...]:
        if len(rule_ids) > 256:
            raise fail(TimerErrorCode.OUT_OF_RANGE)
        return await self.db.call(
            lambda conn: self._latest_occurrences_for_rules(conn, scope, rule_ids)
        )

    async def first_waiting_occurrence(self, scope: TimerScope) -> TimerOccurrence | None:
        return await self.db.call(lambda conn: self._first_waiting_occurrence(conn, scope))

    async def list_manageable(
        self,
        scope: TimerScope,
        source_run_ref: SourceRunRef,
        *,
        limit: int,
        query: str = "",
    ) -> tuple[TimerProjectionSource, ...]:
        return await self._persisted_projection_sources(
            scope,
            source_run_ref,
            limit=limit,
            include_occurrences=True,
            query=query,
        )

    async def resolve_run_ref(
        self,
        scope: TimerScope,
        source_run_ref: SourceRunRef,
        opaque_ref: OpaqueTimerRef,
        target: TimerRefTarget,
    ) -> tuple[TimerRuleId, TimerOccurrenceId | None, int] | None:
        return await self.db.call(
            lambda conn: self._resolve_run_ref(conn, scope, source_run_ref, opaque_ref, target)
        )

    def create_or_reuse_in_transaction(
        self,
        conn: sqlite3.Connection,
        command: CreateTimerCommand,
        prepared: PreparedTimerCreation,
    ) -> CreateTimerResult:
        """Apply creation inside the caller-owned transaction."""

        return self._create(conn, command, prepared)

    async def advance_occurrence(self, command: AdvanceOccurrenceCommand) -> TimerOccurrence:
        result = await self.uow.run(lambda conn: self._advance_occurrence(conn, command))
        await self.db.publish_backup_after_commit()
        return result

    def apply_management_in_transaction(
        self,
        conn: sqlite3.Connection,
        command: ManageTimerCommand,
    ) -> ManageTimerResult:
        """Apply management inside the caller-owned transaction."""

        return self._manage(conn, command)

    def apply_revision_in_transaction(
        self,
        conn: sqlite3.Connection,
        command: ReviseTimerCommand,
    ) -> ReviseTimerResult:
        """Apply a schedule or action-text revision in the caller-owned transaction."""

        return self._revise(conn, command)

    async def apply_roll(
        self, command: RollOccurrenceCommand, plan: OccurrenceRollPlan
    ) -> tuple[TimerOccurrence, ...]:
        result = await self.uow.run(lambda conn: self._roll(conn, command, plan))
        await self.db.publish_backup_after_commit()
        return result

    async def mutate_claimed_occurrence(
        self, command: MutateClaimedOccurrenceCommand
    ) -> OccurrenceMutationResult:
        result = await self.uow.run(lambda conn: self._mutate_claimed(conn, command))
        await self.db.publish_backup_after_commit()
        return result

    async def _persisted_projection_sources(
        self,
        scope: TimerScope,
        source_run_ref: SourceRunRef,
        *,
        limit: int,
        include_occurrences: bool,
        query: str = "",
    ) -> tuple[TimerProjectionSource, ...]:
        _page_limit(limit)
        result = await self.uow.run(
            lambda conn: self._projection_sources(
                conn, scope, source_run_ref, limit, include_occurrences, query
            )
        )
        await self.db.publish_backup_after_commit()
        return result

    def _create(
        self,
        conn: sqlite3.Connection,
        command: CreateTimerCommand,
        prepared: PreparedTimerCreation,
    ) -> CreateTimerResult:
        self._require_parent(conn, command.scope)
        request_fingerprint = _fingerprint(
            "create",
            command.fingerprint,
            prepared.rule.rule_id.value,
            prepared.first_occurrence.occurrence_id.value,
            encode_datetime(prepared.first_occurrence.original_due_at) or "",
        )
        replay = self._receipt(
            conn, command.scope, command.idempotency_key, "CREATE", request_fingerprint
        )
        if replay is not None:
            return decode_create_result(replay)
        self._validate_prepared(command, prepared)
        existing = self._find_exact(conn, command.scope, command.fingerprint)
        if existing is not None:
            result = CreateTimerResult(
                CreateTimerOutcome.ALREADY_EXISTS,
                self._bind_ref(conn, command.scope, command.source_run_ref, existing, None),
            )
            self._save_create_receipt(
                conn, command, request_fingerprint, result, prepared.rule.created_at
            )
            return result
        rule_count, occurrence_count = self._count_nonterminal(conn, command.scope)
        if rule_count >= MAX_NONTERMINAL_RULES_PER_INSTANCE:
            raise fail(TimerErrorCode.LIMIT_EXCEEDED)
        if occurrence_count >= MAX_NONTERMINAL_OCCURRENCES_PER_INSTANCE:
            raise fail(TimerErrorCode.LIMIT_EXCEEDED)
        rule = replace(
            prepared.rule,
            created_sequence=self._next_sequence(conn, "timer_rules", command.scope),
        )
        occurrence = replace(
            prepared.first_occurrence,
            created_sequence=self._next_sequence(conn, "timer_occurrences", command.scope),
        )
        self._insert_rule(conn, rule)
        self._insert_occurrence(conn, occurrence)
        result = CreateTimerResult(
            CreateTimerOutcome.CREATED,
            self._bind_ref(conn, command.scope, command.source_run_ref, rule, None),
        )
        self._save_create_receipt(conn, command, request_fingerprint, result, rule.created_at)
        return result

    def _save_create_receipt(
        self,
        conn: sqlite3.Connection,
        command: CreateTimerCommand,
        request_fingerprint: str,
        result: CreateTimerResult,
        created_at: datetime,
    ) -> None:
        self._insert_receipt(
            conn,
            command.scope,
            command.idempotency_key,
            "CREATE",
            request_fingerprint,
            encode_create_result(result),
            created_at,
        )

    def _manage(self, conn: sqlite3.Connection, command: ManageTimerCommand) -> ManageTimerResult:
        self._require_parent(conn, command.scope)
        request_fingerprint = _fingerprint(
            "manage",
            command.source_run_ref.value,
            command.opaque_ref.value,
            command.target.value,
            command.action.value,
            str(command.expected_version),
        )
        replay = self._receipt(
            conn, command.scope, command.idempotency_key, "MANAGE", request_fingerprint
        )
        if replay is not None:
            return replace(decode_manage_result(replay), outcome=ManageTimerOutcome.REPLAYED)
        resolved = self._resolve_run_ref(
            conn,
            command.scope,
            command.source_run_ref,
            command.opaque_ref,
            command.target,
        )
        if resolved is None:
            raise fail(TimerErrorCode.INVALID_REFERENCE)
        rule_id, occurrence_id, _ = resolved
        result = (
            self._manage_series(conn, command, rule_id)
            if command.target is TimerRefTarget.SERIES
            else self._manage_occurrence(conn, command, occurrence_id)
        )
        self._insert_receipt(
            conn,
            command.scope,
            command.idempotency_key,
            "MANAGE",
            request_fingerprint,
            encode_manage_result(result),
            receipt_time(conn),
        )
        return result

    def _revise(self, conn: sqlite3.Connection, command: ReviseTimerCommand) -> ReviseTimerResult:
        self._require_parent(conn, command.scope)
        request_fingerprint = _fingerprint(
            "revise",
            command.source_run_ref.value,
            command.opaque_ref.value,
            str(command.expected_version),
            json.dumps(
                canonical_rule(command.schedule) if command.schedule is not None else None,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            command.prompt or "",
            command.time_expression,
            command.timezone,
        )
        replay = self._receipt(
            conn,
            command.scope,
            command.idempotency_key,
            "REVISE",
            request_fingerprint,
        )
        if replay is not None:
            return replace(decode_revise_result(replay), outcome=ManageTimerOutcome.REPLAYED)
        resolved = self._resolve_run_ref(
            conn,
            command.scope,
            command.source_run_ref,
            command.opaque_ref,
            TimerRefTarget.SERIES,
        )
        if resolved is None:
            raise fail(TimerErrorCode.INVALID_REFERENCE)
        rule = self._required_rule(conn, command.scope, resolved[0])
        if rule.version != command.expected_version:
            raise fail(TimerErrorCode.VERSION_CONFLICT)
        if rule.status in {TimerRuleStatus.CANCELLED, TimerRuleStatus.COMPLETED}:
            raise fail(TimerErrorCode.INVALID_STATE)

        schedule = command.schedule or rule.schedule
        prompt = command.prompt or rule.prompt
        time_expression = (
            command.time_expression if command.schedule is not None else rule.time_expression
        )
        timezone = command.timezone if command.schedule is not None else rule.timezone
        version = rule.version + 1
        revision = TimerRuleRevision(
            version,
            command.changed_at,
            schedule,
            prompt,
            time_expression,
            timezone,
        )
        updated = replace(
            rule,
            schedule=schedule,
            prompt=prompt,
            fingerprint=exact_timer_fingerprint(command.scope, schedule, prompt),
            version=version,
            last_operation_key=command.idempotency_key.value,
            last_operation_fingerprint=request_fingerprint,
            time_expression=time_expression,
            timezone=timezone,
            revisions=(*rule.revisions, revision),
        )
        duplicate = self._find_exact(conn, command.scope, updated.fingerprint)
        if duplicate is not None and duplicate.rule_id != rule.rule_id:
            raise fail(TimerErrorCode.INVALID_STATE)
        if command.schedule is not None:
            self._replace_future_occurrences(conn, rule, updated, command)
        self._update_rule(conn, rule, updated)
        result = ReviseTimerResult(
            ManageTimerOutcome.APPLIED,
            command.opaque_ref,
            updated.version,
        )
        self._insert_receipt(
            conn,
            command.scope,
            command.idempotency_key,
            "REVISE",
            request_fingerprint,
            encode_revise_result(result),
            command.changed_at,
        )
        return result

    def _replace_future_occurrences(
        self,
        conn: sqlite3.Connection,
        before_rule: TimerRule,
        after_rule: TimerRule,
        command: ReviseTimerCommand,
    ) -> None:
        rows = conn.execute(
            """SELECT * FROM timer_occurrences
            WHERE profile_id = ? AND instance_id = ? AND rule_id = ?
              AND status NOT IN ('COMPLETED', 'CANCELLED', 'FAILED', 'MISSED_COALESCED')
            ORDER BY original_due_at, created_sequence, stable_ref""",
            (*command.scope.fingerprint_parts, before_rule.rule_id.value),
        ).fetchall()
        occurrences = tuple(decode_occurrence(dict(row)) for row in rows)
        if any(
            item.status
            in {
                TimerOccurrenceStatus.CLAIMED,
                TimerOccurrenceStatus.RUNNING,
                TimerOccurrenceStatus.WAITING_DELIVERY,
                TimerOccurrenceStatus.RECOVERING,
            }
            for item in occurrences
        ):
            raise fail(TimerErrorCode.INVALID_STATE)
        for occurrence in occurrences:
            operation_key = IdempotencyKey(
                "revise-cancel-"
                + _fingerprint(
                    command.idempotency_key.value,
                    occurrence.occurrence_id.value,
                )[:48]
            )
            cancelled = transition_occurrence(
                occurrence,
                OccurrenceAction.CANCEL,
                expected_version=occurrence.version,
                operation_key=operation_key,
                now=command.changed_at,
            )
            self._update_occurrence(conn, occurrence, cancelled)

        due_at = next_occurrence(after_rule.schedule, after=command.changed_at)
        if due_at is None:
            raise fail(TimerErrorCode.INVALID_RULE)
        identity = _fingerprint(
            *command.scope.fingerprint_parts,
            after_rule.rule_id.value,
            str(after_rule.version),
            encode_datetime(due_at) or "",
        )
        occurrence = TimerOccurrence(
            TimerOccurrenceId(f"occurrence:{identity[:48]}"),
            stable_ref=OccurrenceStableRef(f"stable:{identity[:48]}"),
            rule_id=after_rule.rule_id,
            scope=command.scope,
            original_due_at=due_at,
            status=(
                TimerOccurrenceStatus.PAUSED
                if after_rule.status is TimerRuleStatus.PAUSED
                else TimerOccurrenceStatus.SCHEDULED
            ),
            version=1,
            generation=0,
            created_sequence=self._next_sequence(conn, "timer_occurrences", command.scope),
            created_at=command.changed_at,
        )
        self._insert_occurrence(conn, occurrence)

    def _manage_series(
        self,
        conn: sqlite3.Connection,
        command: ManageTimerCommand,
        rule_id: TimerRuleId,
    ) -> ManageTimerResult:
        rule = self._required_rule(conn, command.scope, rule_id)
        updated = transition_rule(
            rule,
            RuleAction(command.action.value),
            expected_version=command.expected_version,
            operation_key=command.idempotency_key,
        )
        self._update_rule(conn, rule, updated)
        rows = conn.execute(
            """SELECT * FROM timer_occurrences
            WHERE profile_id = ? AND instance_id = ? AND rule_id = ?
            ORDER BY original_due_at, created_sequence, stable_ref""",
            (*command.scope.fingerprint_parts, rule_id.value),
        ).fetchall()
        for row in rows:
            self._manage_series_occurrence(
                conn, decode_occurrence(dict(row)), command.action, command.idempotency_key
            )
        return ManageTimerResult(
            ManageTimerOutcome.APPLIED,
            command.opaque_ref,
            updated.status.value,
            updated.version,
        )

    def _manage_series_occurrence(
        self,
        conn: sqlite3.Connection,
        occurrence: TimerOccurrence,
        action: ManageTimerAction,
        operation_key: IdempotencyKey,
    ) -> None:
        transition = _series_occurrence_action(action, occurrence.status)
        if transition is None:
            return
        derived = IdempotencyKey(
            "series-" + _fingerprint(operation_key.value, occurrence.occurrence_id.value)[:48]
        )
        now = receipt_time(conn)
        updated = transition_occurrence(
            occurrence,
            transition,
            expected_version=occurrence.version,
            operation_key=derived,
            now=now,
        )
        self._update_occurrence(conn, occurrence, updated)
        if occurrence.status in {
            TimerOccurrenceStatus.CLAIMED,
            TimerOccurrenceStatus.RUNNING,
        }:
            self._release_active_for_resource(conn, occurrence.scope, occurrence.occurrence_id, now)

    def _manage_occurrence(
        self,
        conn: sqlite3.Connection,
        command: ManageTimerCommand,
        occurrence_id: TimerOccurrenceId | None,
    ) -> ManageTimerResult:
        if occurrence_id is None:
            raise fail(TimerErrorCode.INVALID_REFERENCE)
        occurrence = self._required_occurrence(conn, command.scope, occurrence_id)
        try:
            updated = transition_occurrence(
                occurrence,
                OccurrenceAction(command.action.value),
                expected_version=command.expected_version,
                operation_key=command.idempotency_key,
                now=receipt_time(conn),
            )
        except TimerDomainError as exc:
            if exc.code is not TimerErrorCode.INVALID_STATE:
                raise
            return ManageTimerResult(
                ManageTimerOutcome.TOO_LATE_OR_UNKNOWN,
                command.opaque_ref,
                occurrence.status.value,
                occurrence.version,
            )
        self._update_occurrence(conn, occurrence, updated)
        if occurrence.status in {
            TimerOccurrenceStatus.CLAIMED,
            TimerOccurrenceStatus.RUNNING,
        }:
            self._release_active_for_resource(
                conn, occurrence.scope, occurrence.occurrence_id, receipt_time(conn)
            )
        return ManageTimerResult(
            ManageTimerOutcome.APPLIED,
            command.opaque_ref,
            updated.status.value,
            updated.version,
        )

    def _bind_ref(
        self,
        conn: sqlite3.Connection,
        scope: TimerScope,
        source_run_ref: SourceRunRef,
        rule: TimerRule,
        occurrence: TimerOccurrence | None,
    ) -> OpaqueTimerRef:
        target = TimerRefTarget.OCCURRENCE if occurrence else TimerRefTarget.SERIES
        occurrence_id = occurrence.occurrence_id.value if occurrence else None
        row = conn.execute(
            """SELECT opaque_ref FROM timer_run_refs
            WHERE profile_id = ? AND instance_id = ? AND source_run_ref = ?
              AND target = ? AND rule_id = ? AND occurrence_id IS ?""",
            (
                *scope.fingerprint_parts,
                source_run_ref.value,
                target.value,
                rule.rule_id.value,
                occurrence_id,
            ),
        ).fetchone()
        if row is not None:
            return OpaqueTimerRef(str(row["opaque_ref"]))
        opaque = OpaqueTimerRef(
            _fingerprint(
                *scope.fingerprint_parts,
                source_run_ref.value,
                target.value,
                rule.rule_id.value,
                occurrence_id or "",
            )[:32]
        )
        conn.execute(
            """INSERT INTO timer_run_refs(
                profile_id, instance_id, source_run_ref, opaque_ref, target,
                rule_id, occurrence_id, target_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                *scope.fingerprint_parts,
                source_run_ref.value,
                opaque.value,
                target.value,
                rule.rule_id.value,
                occurrence_id,
                occurrence.version if occurrence else rule.version,
                encode_datetime(occurrence.created_at if occurrence else rule.created_at),
            ),
        )
        return opaque

    @staticmethod
    def _insert_rule(conn: sqlite3.Connection, rule: TimerRule) -> None:
        conn.execute(
            """INSERT INTO timer_rules(
                profile_id, instance_id, rule_id, schedule_kind, schedule_json,
                prompt, fingerprint, status, version, created_sequence, created_at,
                source_run_ref, source_message_refs_json, last_operation_key,
                last_operation_fingerprint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (*rule.scope.fingerprint_parts, rule.rule_id.value, *rule_columns(rule)),
        )

    @staticmethod
    def _update_rule(conn: sqlite3.Connection, before: TimerRule, after: TimerRule) -> None:
        columns = rule_columns(after)
        cursor = conn.execute(
            """UPDATE timer_rules SET schedule_kind = ?, schedule_json = ?, prompt = ?,
                fingerprint = ?, status = ?, version = ?, last_operation_key = ?,
                last_operation_fingerprint = ?
            WHERE profile_id = ? AND instance_id = ? AND rule_id = ? AND version = ?""",
            (
                columns[0],
                columns[1],
                columns[2],
                columns[3],
                columns[4],
                columns[5],
                columns[10],
                columns[11],
                *before.scope.fingerprint_parts,
                before.rule_id.value,
                before.version,
            ),
        )
        if cursor.rowcount != 1:
            raise fail(TimerErrorCode.VERSION_CONFLICT)

    @staticmethod
    def _validate_prepared(command: CreateTimerCommand, prepared: PreparedTimerCreation) -> None:
        rule = prepared.rule
        occurrence = prepared.first_occurrence
        if (
            rule.scope != command.scope
            or occurrence.scope != command.scope
            or rule.schedule != command.schedule
            or rule.prompt != command.prompt
            or rule.fingerprint != command.fingerprint
            or rule.source_run_ref != command.source_run_ref
            or rule.source_message_refs != command.source_message_refs
            or rule.status is not TimerRuleStatus.ACTIVE
            or occurrence.status is not TimerOccurrenceStatus.SCHEDULED
        ):
            raise fail(TimerErrorCode.INVALID_STATE)


def _series_occurrence_action(
    action: ManageTimerAction, status: TimerOccurrenceStatus
) -> OccurrenceAction | None:
    if action is ManageTimerAction.PAUSE and status in {
        TimerOccurrenceStatus.SCHEDULED,
        TimerOccurrenceStatus.WAITING,
        TimerOccurrenceStatus.CLAIMED,
    }:
        return OccurrenceAction.PAUSE
    if action is ManageTimerAction.RESUME and status is TimerOccurrenceStatus.PAUSED:
        return OccurrenceAction.RESUME
    if action is ManageTimerAction.CANCEL and status in {
        TimerOccurrenceStatus.SCHEDULED,
        TimerOccurrenceStatus.WAITING,
        TimerOccurrenceStatus.CLAIMED,
        TimerOccurrenceStatus.PAUSED,
    }:
        return OccurrenceAction.CANCEL
    return None


def _fingerprint(*parts: str) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _page_limit(limit: int) -> None:
    if isinstance(limit, bool) or not 1 <= limit <= 256:
        raise fail(TimerErrorCode.OUT_OF_RANGE)


__all__ = ["SqliteTimerRepository"]
