"""Transactional occurrence, lease, roll, and shared SQLite primitives."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import datetime

from ....storage.sqlite.codec import decode_datetime, encode_datetime
from ..constants import MAX_NONTERMINAL_OCCURRENCES_PER_INSTANCE
from ..contracts import RollOccurrenceCommand
from ..domain import (
    IdempotencyKey,
    OccurrenceStableRef,
    TimerOccurrence,
    TimerOccurrenceId,
    TimerOccurrenceStatus,
    TimerRuleId,
    TimerScope,
)
from ..errors import TimerErrorCode, fail
from ..repository import (
    AdvanceOccurrenceCommand,
    InstanceOccupancy,
    InstanceOccupancyStatus,
    MutateClaimedOccurrenceCommand,
    OccurrenceMutationResult,
)
from ..rules import OccurrenceRollPlan
from ..transitions import OccurrenceAction, transition_occurrence
from .codec import (
    decode_mutation_result,
    decode_occupancy,
    decode_occurrence,
    decode_occurrence_result,
    encode_mutation_result,
    encode_occurrence_result,
    occurrence_columns,
)
from .rule_completion import complete_one_shot_rule_for_occurrence

_TERMINAL_OCCURRENCES = tuple(status.value for status in TimerOccurrenceStatus if status.terminal)
_RELEASE_ACTIONS = {
    OccurrenceAction.RELEASE_CLAIM,
    OccurrenceAction.COMPLETE_NO_OP,
    OccurrenceAction.COMPLETE_DELIVERY,
    OccurrenceAction.FAIL,
    OccurrenceAction.PAUSE,
    OccurrenceAction.CANCEL,
    OccurrenceAction.SUPERSEDE_REQUEUE,
}
_CLAIMED_ACTIONS = _RELEASE_ACTIONS | {
    OccurrenceAction.START_PROVIDER,
    OccurrenceAction.HANDOFF_DELIVERY,
}


class TimerSqliteOperations:
    def _advance_occurrence(
        self, conn: sqlite3.Connection, command: AdvanceOccurrenceCommand
    ) -> TimerOccurrence:
        request_fingerprint = _fingerprint(
            "advance",
            command.occurrence_id.value,
            command.action.value,
            str(command.expected_version),
            str(command.expected_generation),
        )
        replay = self._receipt(
            conn, command.scope, command.idempotency_key, "ADVANCE", request_fingerprint
        )
        if replay is not None:
            return decode_occurrence_result(replay)
        current = self._required_occurrence(conn, command.scope, command.occurrence_id)
        self._check_occurrence_fence(current, command.expected_version, command.expected_generation)
        updated = transition_occurrence(
            current,
            command.action,
            expected_version=command.expected_version,
            operation_key=command.idempotency_key,
            now=command.now,
        )
        self._update_occurrence(conn, current, updated)
        self._insert_receipt(
            conn,
            command.scope,
            command.idempotency_key,
            "ADVANCE",
            request_fingerprint,
            encode_occurrence_result(updated),
            command.now,
        )
        return updated

    def _mutate_claimed(
        self, conn: sqlite3.Connection, command: MutateClaimedOccurrenceCommand
    ) -> OccurrenceMutationResult:
        if command.action not in _CLAIMED_ACTIONS:
            raise fail(TimerErrorCode.INVALID_STATE)
        request_fingerprint = _fingerprint(
            "mutation",
            command.occurrence_id.value,
            command.action.value,
            str(command.expected_version),
            str(command.expected_generation),
            command.occupancy_id,
            str(command.expected_occupancy_version),
            command.lease_owner,
            command.lease_token,
            command.execution_ref.value if command.execution_ref else "",
            command.delivery_ref.value if command.delivery_ref else "",
        )
        replay = self._receipt(
            conn, command.scope, command.idempotency_key, "MUTATE", request_fingerprint
        )
        if replay is not None:
            return decode_mutation_result(replay)
        occurrence = self._required_occurrence(conn, command.scope, command.occurrence_id)
        self._check_occurrence_fence(
            occurrence, command.expected_version, command.expected_generation
        )
        occupancy = self._required_active_occupancy(conn, command.scope, command.occupancy_id)
        self._check_occupancy_fence(
            occupancy,
            version=command.expected_occupancy_version,
            generation=command.expected_generation,
            owner=command.lease_owner,
            token=command.lease_token,
            now=command.now,
        )
        if occupancy.resource_ref != occurrence.occurrence_id.value:
            raise fail(TimerErrorCode.SCOPE_MISMATCH)
        updated_occurrence = transition_occurrence(
            occurrence,
            command.action,
            expected_version=command.expected_version,
            operation_key=command.idempotency_key,
            now=command.now,
            execution_ref=command.execution_ref,
            delivery_ref=command.delivery_ref,
        )
        updated_occupancy = self._advance_occupancy(
            occupancy, command.now, release=command.action in _RELEASE_ACTIONS
        )
        self._update_occurrence(conn, occurrence, updated_occurrence)
        if updated_occurrence.status is TimerOccurrenceStatus.COMPLETED:
            row = conn.execute(
                """SELECT * FROM timer_occurrences
                WHERE profile_id = ? AND instance_id = ? AND occurrence_id = ?""",
                (*occurrence.scope.fingerprint_parts, occurrence.occurrence_id.value),
            ).fetchone()
            if row is not None:
                complete_one_shot_rule_for_occurrence(conn, row)
        self._update_occupancy(conn, occupancy, updated_occupancy)
        result = OccurrenceMutationResult(updated_occurrence, updated_occupancy)
        self._insert_receipt(
            conn,
            command.scope,
            command.idempotency_key,
            "MUTATE",
            request_fingerprint,
            encode_mutation_result(result),
            command.now,
        )
        return result

    def _roll(
        self,
        conn: sqlite3.Connection,
        command: RollOccurrenceCommand,
        plan: OccurrenceRollPlan,
    ) -> tuple[TimerOccurrence, ...]:
        self._require_parent(conn, command.scope)
        if not self._rule_exists(conn, command.scope, command.rule_id):
            raise fail(TimerErrorCode.SCOPE_MISMATCH)
        key = (*command.scope.fingerprint_parts, command.rule_id.value)
        last_due = encode_datetime(command.last_materialized_due_at)
        through = encode_datetime(command.through)
        existing = conn.execute(
            """SELECT result_occurrence_ids_json FROM timer_occurrence_rolls
            WHERE profile_id = ? AND instance_id = ? AND rule_id = ?
                AND last_materialized_due_at = ? AND through_at = ?""",
            (*key, last_due, through),
        ).fetchone()
        if existing is not None:
            return self._load_occurrence_ids(
                conn, command.scope, json.loads(str(existing["result_occurrence_ids_json"]))
            )
        results = self._materialize_roll_plan(conn, command, plan)
        conn.execute(
            """INSERT INTO timer_occurrence_rolls(
                profile_id, instance_id, rule_id, last_materialized_due_at, through_at,
                latest_missed_due_at, coalesced_count, next_future_due_at,
                result_occurrence_ids_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                *key,
                last_due,
                through,
                encode_datetime(plan.latest_missed_due_at),
                plan.coalesced_count,
                encode_datetime(plan.next_future_due_at),
                json.dumps([item.occurrence_id.value for item in results]),
                through,
            ),
        )
        return results

    def _materialize_roll_plan(
        self,
        conn: sqlite3.Connection,
        command: RollOccurrenceCommand,
        plan: OccurrenceRollPlan,
    ) -> tuple[TimerOccurrence, ...]:
        due_states = (
            (plan.latest_missed_due_at, TimerOccurrenceStatus.WAITING),
            (plan.next_future_due_at, TimerOccurrenceStatus.SCHEDULED),
        )
        return tuple(
            self._materialize_roll_occurrence(
                conn, command.scope, command.rule_id, due_at, status, command.through
            )
            for due_at, status in due_states
            if due_at is not None
        )

    def _materialize_roll_occurrence(
        self,
        conn: sqlite3.Connection,
        scope: TimerScope,
        rule_id: TimerRuleId,
        due_at: datetime,
        status: TimerOccurrenceStatus,
        created_at: datetime,
    ) -> TimerOccurrence:
        row = conn.execute(
            """SELECT * FROM timer_occurrences
            WHERE profile_id = ? AND instance_id = ? AND rule_id = ? AND original_due_at = ?""",
            (*scope.fingerprint_parts, rule_id.value, encode_datetime(due_at)),
        ).fetchone()
        if row is not None:
            return decode_occurrence(dict(row))
        _, occurrence_count = self._count_nonterminal(conn, scope)
        if not status.terminal and occurrence_count >= MAX_NONTERMINAL_OCCURRENCES_PER_INSTANCE:
            raise fail(TimerErrorCode.LIMIT_EXCEEDED)
        digest = _fingerprint(
            *scope.fingerprint_parts, rule_id.value, encode_datetime(due_at) or ""
        )
        occurrence = TimerOccurrence(
            occurrence_id=TimerOccurrenceId(f"roll:{digest[:48]}"),
            stable_ref=OccurrenceStableRef(f"stable:{digest[16:64]}"),
            rule_id=rule_id,
            scope=scope,
            original_due_at=due_at,
            status=status,
            version=1,
            generation=0,
            created_sequence=self._next_sequence(conn, "timer_occurrences", scope),
            created_at=created_at,
        )
        self._insert_occurrence(conn, occurrence)
        return occurrence

    @staticmethod
    def _get_occurrence(
        conn: sqlite3.Connection, scope: TimerScope, occurrence_id: TimerOccurrenceId
    ) -> TimerOccurrence | None:
        row = conn.execute(
            """SELECT * FROM timer_occurrences
            WHERE profile_id = ? AND instance_id = ? AND occurrence_id = ?""",
            (*scope.fingerprint_parts, occurrence_id.value),
        ).fetchone()
        return decode_occurrence(dict(row)) if row is not None else None

    @staticmethod
    def _active_occupancy(conn: sqlite3.Connection, scope: TimerScope) -> InstanceOccupancy | None:
        row = conn.execute(
            """SELECT * FROM instance_main_core_occupancies
            WHERE profile_id = ? AND instance_id = ? AND status = 'ACTIVE'""",
            scope.fingerprint_parts,
        ).fetchone()
        return decode_occupancy(dict(row)) if row is not None else None

    def _required_occurrence(
        self, conn: sqlite3.Connection, scope: TimerScope, occurrence_id: TimerOccurrenceId
    ) -> TimerOccurrence:
        result = self._get_occurrence(conn, scope, occurrence_id)
        if result is None:
            raise fail(TimerErrorCode.SCOPE_MISMATCH)
        return result

    @staticmethod
    def _count_nonterminal(conn: sqlite3.Connection, scope: TimerScope) -> tuple[int, int]:
        placeholders = ",".join("?" for _ in _TERMINAL_OCCURRENCES)
        rules = conn.execute(
            f"""SELECT COUNT(*) FROM timer_rules rule
            WHERE rule.profile_id = ? AND rule.instance_id = ?
              AND rule.status NOT IN ('CANCELLED', 'COMPLETED')
              AND (
                rule.schedule_kind IN ('WEEKLY','YEARLY')
                OR EXISTS (
                    SELECT 1 FROM timer_occurrences occurrence
                    WHERE occurrence.profile_id = rule.profile_id
                      AND occurrence.instance_id = rule.instance_id
                      AND occurrence.rule_id = rule.rule_id
                      AND occurrence.status NOT IN ({placeholders})
                )
              )""",
            (*scope.fingerprint_parts, *_TERMINAL_OCCURRENCES),
        ).fetchone()
        occurrences = conn.execute(
            f"""SELECT COUNT(*) FROM timer_occurrences
            WHERE profile_id = ? AND instance_id = ? AND status NOT IN ({placeholders})""",
            (*scope.fingerprint_parts, *_TERMINAL_OCCURRENCES),
        ).fetchone()
        return int(rules[0]), int(occurrences[0])

    @staticmethod
    def _insert_occurrence(conn: sqlite3.Connection, occurrence: TimerOccurrence) -> None:
        conn.execute(
            """INSERT INTO timer_occurrences(
                profile_id, instance_id, occurrence_id, stable_ref, rule_id,
                original_due_at, status, version, generation, created_sequence,
                created_at, execution_ref, delivery_ref, recovery_from,
                last_operation_key, last_operation_fingerprint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                *occurrence.scope.fingerprint_parts,
                occurrence.occurrence_id.value,
                *occurrence_columns(occurrence),
            ),
        )

    @staticmethod
    def _update_occurrence(
        conn: sqlite3.Connection, before: TimerOccurrence, after: TimerOccurrence
    ) -> None:
        cursor = conn.execute(
            """UPDATE timer_occurrences SET status = ?, version = ?, generation = ?,
                execution_ref = ?, delivery_ref = ?, recovery_from = ?,
                last_operation_key = ?, last_operation_fingerprint = ?
            WHERE profile_id = ? AND instance_id = ? AND occurrence_id = ?
                AND version = ? AND generation = ?""",
            (
                after.status.value,
                after.version,
                after.generation,
                after.execution_ref.value if after.execution_ref else None,
                after.delivery_ref.value if after.delivery_ref else None,
                after.recovery_from.value if after.recovery_from else None,
                after.last_operation_key,
                after.last_operation_fingerprint,
                *before.scope.fingerprint_parts,
                before.occurrence_id.value,
                before.version,
                before.generation,
            ),
        )
        if cursor.rowcount != 1:
            raise fail(TimerErrorCode.VERSION_CONFLICT)

    @staticmethod
    def _update_occupancy(
        conn: sqlite3.Connection, before: InstanceOccupancy, after: InstanceOccupancy
    ) -> None:
        cursor = conn.execute(
            """UPDATE instance_main_core_occupancies SET status = ?, version = ?,
                lease_expires_at = ?, updated_at = ?, released_at = ?
            WHERE profile_id = ? AND instance_id = ? AND occupancy_id = ?
                AND status = ? AND version = ? AND generation = ?
                AND lease_owner = ? AND lease_token = ?""",
            (
                after.status.value,
                after.version,
                encode_datetime(after.lease_expires_at),
                encode_datetime(after.updated_at),
                encode_datetime(after.released_at),
                *before.scope.fingerprint_parts,
                before.occupancy_id,
                before.status.value,
                before.version,
                before.generation,
                before.lease_owner,
                before.lease_token,
            ),
        )
        if cursor.rowcount != 1:
            raise fail(TimerErrorCode.VERSION_CONFLICT)

    def _required_active_occupancy(
        self, conn: sqlite3.Connection, scope: TimerScope, occupancy_id: str
    ) -> InstanceOccupancy:
        row = conn.execute(
            """SELECT * FROM instance_main_core_occupancies
            WHERE profile_id = ? AND instance_id = ? AND occupancy_id = ?
                AND status = 'ACTIVE'""",
            (*scope.fingerprint_parts, occupancy_id),
        ).fetchone()
        if row is None:
            raise fail(TimerErrorCode.INVALID_STATE)
        return decode_occupancy(dict(row))

    @staticmethod
    def _advance_occupancy(
        occupancy: InstanceOccupancy, now: datetime, *, release: bool
    ) -> InstanceOccupancy:
        return replace(
            occupancy,
            status=(
                InstanceOccupancyStatus.RELEASED if release else InstanceOccupancyStatus.ACTIVE
            ),
            version=occupancy.version + 1,
            updated_at=now,
            released_at=now if release else None,
        )

    def _release_active_for_resource(
        self,
        conn: sqlite3.Connection,
        scope: TimerScope,
        occurrence_id: TimerOccurrenceId,
        now: datetime,
    ) -> None:
        current = self._active_occupancy(conn, scope)
        if current is None or current.resource_ref != occurrence_id.value:
            return
        self._update_occupancy(conn, current, self._advance_occupancy(current, now, release=True))

    @staticmethod
    def _check_occurrence_fence(occurrence: TimerOccurrence, version: int, generation: int) -> None:
        if occurrence.version != version or occurrence.generation != generation:
            raise fail(TimerErrorCode.VERSION_CONFLICT)

    @staticmethod
    def _check_occupancy_fence(
        occupancy: InstanceOccupancy,
        *,
        version: int,
        generation: int,
        owner: str,
        token: str,
        now: datetime,
    ) -> None:
        if occupancy.version != version or occupancy.generation != generation:
            raise fail(TimerErrorCode.VERSION_CONFLICT)
        if occupancy.lease_owner != owner or occupancy.lease_token != token:
            raise fail(TimerErrorCode.INVALID_REFERENCE)
        if occupancy.lease_expires_at <= now:
            raise fail(TimerErrorCode.INVALID_STATE)

    @staticmethod
    def _next_sequence(conn: sqlite3.Connection, table: str, scope: TimerScope) -> int:
        if table not in {"timer_rules", "timer_occurrences"}:
            raise ValueError("invalid Timer sequence table")
        row = conn.execute(
            f"""SELECT COALESCE(MAX(created_sequence), 0) + 1
            FROM {table} WHERE profile_id = ? AND instance_id = ?""",
            scope.fingerprint_parts,
        ).fetchone()
        return int(row[0])

    @staticmethod
    def _receipt(
        conn: sqlite3.Connection,
        scope: TimerScope,
        key: IdempotencyKey,
        operation_kind: str,
        request_fingerprint: str,
    ) -> str | None:
        row = conn.execute(
            """SELECT operation_kind, request_fingerprint, result_json
            FROM timer_operation_receipts
            WHERE profile_id = ? AND instance_id = ? AND idempotency_key = ?""",
            (*scope.fingerprint_parts, key.value),
        ).fetchone()
        if row is None:
            return None
        if (
            str(row["operation_kind"]) != operation_kind
            or str(row["request_fingerprint"]) != request_fingerprint
        ):
            raise fail(TimerErrorCode.IDEMPOTENCY_CONFLICT)
        return str(row["result_json"])

    @staticmethod
    def _insert_receipt(
        conn: sqlite3.Connection,
        scope: TimerScope,
        key: IdempotencyKey,
        operation_kind: str,
        request_fingerprint: str,
        result_json: str,
        created_at: datetime,
    ) -> None:
        conn.execute(
            """INSERT INTO timer_operation_receipts(
                profile_id, instance_id, idempotency_key, operation_kind,
                request_fingerprint, result_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                *scope.fingerprint_parts,
                key.value,
                operation_kind,
                request_fingerprint,
                result_json,
                encode_datetime(created_at),
            ),
        )

    @staticmethod
    def _load_occurrence_ids(
        conn: sqlite3.Connection, scope: TimerScope, values: object
    ) -> tuple[TimerOccurrence, ...]:
        if not isinstance(values, list):
            raise ValueError("invalid persisted Timer roll result")
        result: list[TimerOccurrence] = []
        for value in values:
            occurrence = TimerSqliteOperations._get_occurrence(
                conn, scope, TimerOccurrenceId(str(value))
            )
            if occurrence is None:
                raise ValueError("persisted Timer roll references a missing occurrence")
            result.append(occurrence)
        return tuple(result)

    @staticmethod
    def _rule_exists(conn: sqlite3.Connection, scope: TimerScope, rule_id: TimerRuleId) -> bool:
        return (
            conn.execute(
                """SELECT 1 FROM timer_rules
                WHERE profile_id = ? AND instance_id = ? AND rule_id = ?""",
                (*scope.fingerprint_parts, rule_id.value),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _require_parent(conn: sqlite3.Connection, scope: TimerScope) -> None:
        row = conn.execute(
            """SELECT 1 FROM character_instances
            WHERE profile_id = ? AND instance_id = ?""",
            scope.fingerprint_parts,
        ).fetchone()
        if row is None:
            raise fail(TimerErrorCode.SCOPE_MISMATCH)


def _fingerprint(*parts: str) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def receipt_time(conn: sqlite3.Connection) -> datetime:
    row = conn.execute("SELECT strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')").fetchone()
    value = decode_datetime(str(row[0]))
    if value is None:
        raise RuntimeError("SQLite did not return a transaction timestamp")
    return value


__all__ = ["TimerSqliteOperations", "receipt_time"]
