"""Durable candidates and CAS application for Timer lifecycle review."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ....storage.sqlite.codec import decode_datetime, encode_datetime
from ..domain import (
    OpaqueTimerRef,
    SourceRunRef,
    TimerOccurrence,
    TimerOccurrenceId,
    TimerRule,
    TimerRuleId,
    TimerScope,
    require_aware,
)
from ..errors import TimerErrorCode, fail
from ..lifecycle import (
    TimerLifecycleDecision,
    TimerLifecycleEvidence,
    TimerLifecycleReview,
    TimerLifecycleReviewStatus,
)
from ..projection import TimerProjectionSource, TimerRefTarget
from ..repository import OccurrencePage, RulePage
from .codec import decode_occurrence, decode_rule


class TimerProjectionQueriesMixin:
    """Read-side SQL shared by the concrete Timer repository."""

    if TYPE_CHECKING:

        @staticmethod
        def _require_parent(conn: sqlite3.Connection, scope: TimerScope) -> None: ...

        @staticmethod
        def _get_occurrence(
            conn: sqlite3.Connection,
            scope: TimerScope,
            occurrence_id: TimerOccurrenceId,
        ) -> TimerOccurrence | None: ...

        def _bind_ref(
            self,
            conn: sqlite3.Connection,
            scope: TimerScope,
            source_run_ref: SourceRunRef,
            rule: TimerRule,
            occurrence: TimerOccurrence | None,
        ) -> OpaqueTimerRef: ...

    def _projection_sources(
        self,
        conn: sqlite3.Connection,
        scope: TimerScope,
        source_run_ref: SourceRunRef,
        limit: int,
        include_occurrences: bool,
        query: str = "",
    ) -> tuple[TimerProjectionSource, ...]:
        self._require_parent(conn, scope)
        folded_query = str(query or "").strip().casefold()
        sql = """SELECT * FROM timer_rules
            WHERE profile_id = ? AND instance_id = ?
              AND status NOT IN ('CANCELLED', 'COMPLETED')
            ORDER BY created_sequence, rule_id"""
        params: tuple[object, ...] = scope.fingerprint_parts
        if not folded_query:
            sql += " LIMIT ?"
            params = (*params, limit)
        rows = conn.execute(sql, params).fetchall()
        result: list[TimerProjectionSource] = []
        for row in rows:
            rule = decode_rule(dict(row))
            if folded_query and all(
                folded_query not in value.casefold()
                for value in (rule.prompt, rule.time_expression)
            ):
                continue
            occurrence = self._next_manageable_occurrence(conn, scope, rule.rule_id)
            result.append(self._series_projection(conn, source_run_ref, rule, occurrence))
            if include_occurrences and occurrence is not None and len(result) < limit:
                result.append(self._occurrence_projection(conn, source_run_ref, rule, occurrence))
            if len(result) >= limit:
                break
        return tuple(result)

    def _series_projection(
        self,
        conn: sqlite3.Connection,
        source_run_ref: SourceRunRef,
        rule: TimerRule,
        occurrence: TimerOccurrence | None,
    ) -> TimerProjectionSource:
        return TimerProjectionSource(
            self._bind_ref(conn, rule.scope, source_run_ref, rule, None),
            TimerRefTarget.SERIES,
            rule,
            next_due_at=occurrence.original_due_at if occurrence else None,
        )

    def _occurrence_projection(
        self,
        conn: sqlite3.Connection,
        source_run_ref: SourceRunRef,
        rule: TimerRule,
        occurrence: TimerOccurrence,
    ) -> TimerProjectionSource:
        return TimerProjectionSource(
            self._bind_ref(conn, rule.scope, source_run_ref, rule, occurrence),
            TimerRefTarget.OCCURRENCE,
            rule,
            occurrence=occurrence,
        )

    def _next_manageable_occurrence(
        self, conn: sqlite3.Connection, scope: TimerScope, rule_id: TimerRuleId
    ) -> TimerOccurrence | None:
        row = conn.execute(
            """SELECT * FROM timer_occurrences
            WHERE profile_id = ? AND instance_id = ? AND rule_id = ?
              AND status NOT IN ('COMPLETED', 'CANCELLED', 'FAILED', 'MISSED_COALESCED')
            ORDER BY original_due_at, created_sequence, stable_ref LIMIT 1""",
            (*scope.fingerprint_parts, rule_id.value),
        ).fetchone()
        return decode_occurrence(dict(row)) if row is not None else None

    @staticmethod
    def _find_exact(
        conn: sqlite3.Connection, scope: TimerScope, fingerprint: str
    ) -> TimerRule | None:
        row = conn.execute(
            """SELECT * FROM timer_rules
            WHERE profile_id = ? AND instance_id = ? AND fingerprint = ?
              AND status NOT IN ('CANCELLED', 'COMPLETED')""",
            (*scope.fingerprint_parts, fingerprint),
        ).fetchone()
        return decode_rule(dict(row)) if row is not None else None

    @staticmethod
    def _get_rule(
        conn: sqlite3.Connection, scope: TimerScope, rule_id: TimerRuleId
    ) -> TimerRule | None:
        row = conn.execute(
            """SELECT * FROM timer_rules
            WHERE profile_id = ? AND instance_id = ? AND rule_id = ?""",
            (*scope.fingerprint_parts, rule_id.value),
        ).fetchone()
        return decode_rule(dict(row)) if row is not None else None

    def _required_rule(
        self, conn: sqlite3.Connection, scope: TimerScope, rule_id: TimerRuleId
    ) -> TimerRule:
        result = self._get_rule(conn, scope, rule_id)
        if result is None:
            raise fail(TimerErrorCode.SCOPE_MISMATCH)
        return result

    @staticmethod
    def _list_rules(
        conn: sqlite3.Connection,
        scope: TimerScope,
        limit: int,
        after_created_sequence: int,
    ) -> RulePage:
        rows = conn.execute(
            """SELECT * FROM timer_rules
            WHERE profile_id = ? AND instance_id = ? AND created_sequence > ?
              AND status NOT IN ('CANCELLED', 'COMPLETED')
            ORDER BY created_sequence, rule_id LIMIT ?""",
            (*scope.fingerprint_parts, after_created_sequence, limit + 1),
        ).fetchall()
        items = tuple(decode_rule(dict(row)) for row in rows[:limit])
        next_sequence = items[-1].created_sequence if len(rows) > limit and items else None
        return RulePage(items, next_sequence)

    @staticmethod
    def _list_occurrences(
        conn: sqlite3.Connection,
        scope: TimerScope,
        limit: int,
        after: tuple[datetime, int, str] | None,
    ) -> OccurrencePage:
        cursor = after or (datetime.min.replace(tzinfo=UTC), 0, "")
        due_at = encode_datetime(require_aware(cursor[0]))
        rows = conn.execute(
            """SELECT * FROM timer_occurrences
            WHERE profile_id = ? AND instance_id = ? AND (
                original_due_at > ? OR
                (original_due_at = ? AND created_sequence > ?) OR
                (original_due_at = ? AND created_sequence = ? AND stable_ref > ?)
            ) ORDER BY original_due_at, created_sequence, stable_ref LIMIT ?""",
            (
                *scope.fingerprint_parts,
                due_at,
                due_at,
                cursor[1],
                due_at,
                cursor[1],
                cursor[2],
                limit + 1,
            ),
        ).fetchall()
        items = tuple(decode_occurrence(dict(row)) for row in rows[:limit])
        next_cursor = None
        if len(rows) > limit and items:
            last = items[-1]
            next_cursor = (last.original_due_at, last.created_sequence, last.stable_ref.value)
        return OccurrencePage(items, next_cursor)

    @staticmethod
    def _list_due_scheduled_occurrences(
        conn: sqlite3.Connection,
        scope: TimerScope,
        through: datetime,
        limit: int,
    ) -> tuple[TimerOccurrence, ...]:
        rows = conn.execute(
            """SELECT * FROM timer_occurrences
            WHERE profile_id = ? AND instance_id = ? AND status = 'SCHEDULED'
              AND original_due_at <= ?
            ORDER BY original_due_at, created_sequence, stable_ref LIMIT ?""",
            (*scope.fingerprint_parts, encode_datetime(require_aware(through)), limit),
        ).fetchall()
        return tuple(decode_occurrence(dict(row)) for row in rows)

    @staticmethod
    def _first_waiting_occurrence(
        conn: sqlite3.Connection, scope: TimerScope
    ) -> TimerOccurrence | None:
        row = conn.execute(
            """SELECT * FROM timer_occurrences
            WHERE profile_id = ? AND instance_id = ? AND status = 'WAITING'
            ORDER BY original_due_at, created_sequence, stable_ref LIMIT 1""",
            scope.fingerprint_parts,
        ).fetchone()
        return decode_occurrence(dict(row)) if row is not None else None

    @staticmethod
    def _latest_occurrences_for_rules(
        conn: sqlite3.Connection,
        scope: TimerScope,
        rule_ids: tuple[TimerRuleId, ...],
    ) -> tuple[TimerOccurrence, ...]:
        if not rule_ids:
            return ()
        placeholders = ",".join("?" for _ in rule_ids)
        values = tuple(item.value for item in rule_ids)
        rows = conn.execute(
            f"""SELECT occurrence.* FROM timer_occurrences occurrence
            JOIN (
                SELECT rule_id, MAX(original_due_at) AS latest_due_at
                FROM timer_occurrences
                WHERE profile_id = ? AND instance_id = ?
                  AND rule_id IN ({placeholders})
                GROUP BY rule_id
            ) latest ON latest.rule_id = occurrence.rule_id
              AND latest.latest_due_at = occurrence.original_due_at
            WHERE occurrence.profile_id = ? AND occurrence.instance_id = ?
            ORDER BY occurrence.rule_id""",
            (*scope.fingerprint_parts, *values, *scope.fingerprint_parts),
        ).fetchall()
        return tuple(decode_occurrence(dict(row)) for row in rows)

    def _resolve_run_ref(
        self,
        conn: sqlite3.Connection,
        scope: TimerScope,
        source_run_ref: SourceRunRef,
        opaque_ref: OpaqueTimerRef,
        target: TimerRefTarget,
    ) -> tuple[TimerRuleId, TimerOccurrenceId | None, int] | None:
        row = conn.execute(
            """SELECT rule_id, occurrence_id, target_version FROM timer_run_refs
            WHERE profile_id = ? AND instance_id = ? AND source_run_ref = ?
                AND opaque_ref = ? AND target = ?""",
            (
                *scope.fingerprint_parts,
                source_run_ref.value,
                opaque_ref.value,
                target.value,
            ),
        ).fetchone()
        if row is None:
            return None
        rule_id = TimerRuleId(str(row["rule_id"]))
        if row["occurrence_id"] is None:
            rule = self._get_rule(conn, scope, rule_id)
            return (rule_id, None, int(row["target_version"])) if rule is not None else None
        occurrence_id = TimerOccurrenceId(str(row["occurrence_id"]))
        occurrence = self._get_occurrence(conn, scope, occurrence_id)
        return (
            (rule_id, occurrence_id, int(row["target_version"])) if occurrence is not None else None
        )


_RECURRING_KINDS = ("WEEKLY", "YEARLY")
_SUCCESS_OCCURRENCE_STATUSES = ("COMPLETED", "WAITING_DELIVERY")


class TimerLifecycleSqliteMixin:
    async def create_lifecycle_review(
        self,
        *,
        scope: TimerScope,
        occurrence_id: TimerOccurrenceId,
        occurrence_generation: int,
        main_core_run_id: int,
        expected_activity_epoch: int,
        evidence: TimerLifecycleEvidence,
        now: datetime,
    ) -> TimerLifecycleReview | None:
        stamp = require_aware(now)
        result = await self.uow.run(
            lambda conn: self._create_lifecycle_review(
                conn,
                scope=scope,
                occurrence_id=occurrence_id,
                occurrence_generation=occurrence_generation,
                main_core_run_id=main_core_run_id,
                expected_activity_epoch=expected_activity_epoch,
                evidence=evidence,
                now=stamp,
            )
        )
        await self.db.publish_backup_after_commit()
        return result

    async def get_lifecycle_review(
        self, scope: TimerScope, review_id: str
    ) -> TimerLifecycleReview | None:
        return await self.db.call(lambda conn: self._get_lifecycle_review(conn, scope, review_id))

    async def bind_lifecycle_review_task(
        self,
        scope: TimerScope,
        review_id: str,
        task_id: int,
        *,
        now: datetime,
    ) -> bool:
        stamp = encode_datetime(require_aware(now))

        def operation(conn: sqlite3.Connection) -> bool:
            changed = conn.execute(
                """UPDATE timer_lifecycle_reviews SET task_id = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND review_id = ?
                  AND status = 'PENDING' AND task_id IS NULL""",
                (task_id, stamp, *scope.fingerprint_parts, review_id),
            ).rowcount
            if changed == 1:
                return True
            row = conn.execute(
                """SELECT task_id FROM timer_lifecycle_reviews
                WHERE profile_id = ? AND instance_id = ? AND review_id = ?""",
                (*scope.fingerprint_parts, review_id),
            ).fetchone()
            return row is not None and int(row["task_id"] or 0) == int(task_id)

        result = bool(await self.uow.run(operation))
        await self.db.publish_backup_after_commit()
        return result

    async def list_lifecycle_reviews_needing_tasks(
        self, *, limit: int, now: datetime
    ) -> tuple[TimerLifecycleReview, ...]:
        bounded = max(1, min(64, int(limit)))
        stamp = encode_datetime(require_aware(now))

        def operation(conn: sqlite3.Connection) -> tuple[TimerLifecycleReview, ...]:
            conn.execute(
                """UPDATE timer_lifecycle_reviews AS review
                SET status = 'SKIPPED', error_code = 'OCCURRENCE_NOT_COMPLETED',
                    updated_at = ?, decided_at = ?
                WHERE status = 'PENDING' AND EXISTS (
                    SELECT 1 FROM timer_occurrences occurrence
                    WHERE occurrence.profile_id = review.profile_id
                      AND occurrence.instance_id = review.instance_id
                      AND occurrence.occurrence_id = review.occurrence_id
                      AND occurrence.status IN ('CANCELLED', 'FAILED', 'MISSED_COALESCED')
                )""",
                (stamp, stamp),
            )
            conn.execute(
                """UPDATE timer_lifecycle_reviews AS review
                SET status = 'STALE', error_code = 'CAS_FENCE_CHANGED',
                    updated_at = ?, decided_at = ?
                WHERE review.status = 'PENDING' AND (
                    NOT EXISTS (
                        SELECT 1 FROM timer_rules rule
                        WHERE rule.profile_id = review.profile_id
                          AND rule.instance_id = review.instance_id
                          AND rule.rule_id = review.rule_id
                          AND rule.status = 'ACTIVE'
                          AND rule.version = review.expected_rule_version
                    )
                    OR NOT EXISTS (
                        SELECT 1 FROM timer_occurrences occurrence
                        WHERE occurrence.profile_id = review.profile_id
                          AND occurrence.instance_id = review.instance_id
                          AND occurrence.occurrence_id = review.occurrence_id
                          AND occurrence.generation = review.occurrence_generation
                          AND occurrence.status IN ('WAITING_DELIVERY', 'COMPLETED')
                    )
                    OR NOT EXISTS (
                        SELECT 1 FROM instance_core_state state
                        WHERE state.profile_id = review.profile_id
                          AND state.instance_id = review.instance_id
                          AND state.activity_epoch = review.expected_activity_epoch
                    )
                    OR EXISTS (
                        SELECT 1 FROM instance_core_runs run
                        WHERE run.profile_id = review.profile_id
                          AND run.instance_id = review.instance_id
                          AND run.run_id > review.main_core_run_id
                          AND run.source = 'FOREGROUND_MESSAGE'
                    )
                    OR EXISTS (
                        SELECT 1 FROM timer_occurrences current_occurrence
                        JOIN timer_occurrences newer
                          ON newer.profile_id = current_occurrence.profile_id
                         AND newer.instance_id = current_occurrence.instance_id
                         AND newer.rule_id = current_occurrence.rule_id
                         AND newer.original_due_at > current_occurrence.original_due_at
                        WHERE current_occurrence.profile_id = review.profile_id
                          AND current_occurrence.instance_id = review.instance_id
                          AND current_occurrence.occurrence_id = review.occurrence_id
                          AND (
                            newer.execution_ref IS NOT NULL OR newer.status IN (
                              'CLAIMED', 'RUNNING', 'WAITING_DELIVERY', 'RECOVERING',
                              'COMPLETED', 'FAILED'
                            )
                          )
                    )
                )""",
                (stamp, stamp),
            )
            conn.execute(
                """UPDATE timer_lifecycle_reviews AS review
                SET task_id = NULL, error_code = 'BROKEN_TASK_RECOVERED',
                    updated_at = ?
                WHERE review.status = 'PENDING' AND review.task_id IS NOT NULL
                  AND EXISTS (
                    SELECT 1 FROM ai_tasks task WHERE task.task_id = review.task_id
                      AND task.status IN (
                        'DEFERRED', 'SUCCEEDED', 'FAILED', 'CANCELLED',
                        'RECOVERY_REQUIRED'
                      )
                  )""",
                (stamp,),
            )
            rows = conn.execute(
                """SELECT review.* FROM timer_lifecycle_reviews review
                JOIN timer_occurrences occurrence
                  ON occurrence.profile_id = review.profile_id
                 AND occurrence.instance_id = review.instance_id
                 AND occurrence.occurrence_id = review.occurrence_id
                LEFT JOIN ai_tasks task ON task.task_id = review.task_id
                WHERE review.status = 'PENDING'
                  AND occurrence.status = 'COMPLETED'
                  AND (review.task_id IS NULL OR task.task_id IS NULL)
                ORDER BY review.created_at, review.review_id LIMIT ?""",
                (bounded * 8,),
            ).fetchall()
            selected: list[TimerLifecycleReview] = []
            seen: set[tuple[str, str]] = set()
            for row in rows:
                scope_key = (str(row["profile_id"]), str(row["instance_id"]))
                if scope_key in seen:
                    continue
                seen.add(scope_key)
                selected.append(_decode_review(row))
                if len(selected) >= bounded:
                    break
            return tuple(selected)

        result = await self.uow.run(operation)
        await self.db.publish_backup_after_commit()
        return result

    async def apply_lifecycle_review_result(
        self,
        scope: TimerScope,
        review_id: str,
        *,
        decision: TimerLifecycleDecision | None,
        error_code: str = "",
        now: datetime,
    ) -> TimerLifecycleReviewStatus:
        stamp = require_aware(now)
        status = await self.uow.run(
            lambda conn: self._apply_lifecycle_review_result(
                conn,
                scope,
                review_id,
                decision=decision,
                error_code=error_code,
                now=stamp,
            )
        )
        await self.db.publish_backup_after_commit()
        return status

    async def recover_lifecycle_review_candidates(
        self, *, limit: int, now: datetime
    ) -> tuple[TimerLifecycleReview, ...]:
        bounded = max(1, min(16, int(limit)))
        stamp = require_aware(now)
        result = await self.uow.run(
            lambda conn: self._recover_lifecycle_review_candidates(conn, bounded, stamp)
        )
        await self.db.publish_backup_after_commit()
        return result

    async def timer_lifecycle_snapshot(self, scope: TimerScope) -> dict[str, Any]:
        def operation(conn: sqlite3.Connection) -> dict[str, Any]:
            active = conn.execute(
                """SELECT COUNT(*) FROM timer_rules WHERE profile_id = ? AND instance_id = ?
                AND status = 'ACTIVE' AND schedule_kind IN ('WEEKLY', 'YEARLY')""",
                scope.fingerprint_parts,
            ).fetchone()
            pending = conn.execute(
                """SELECT COUNT(*) FROM timer_lifecycle_reviews
                WHERE profile_id = ? AND instance_id = ? AND status = 'PENDING'""",
                scope.fingerprint_parts,
            ).fetchone()
            completed = conn.execute(
                """SELECT COUNT(*) FROM timer_lifecycle_reviews
                WHERE profile_id = ? AND instance_id = ? AND status = 'COMPLETED'""",
                scope.fingerprint_parts,
            ).fetchone()
            recent = conn.execute(
                """SELECT status, decision, error_code, updated_at
                FROM timer_lifecycle_reviews WHERE profile_id = ? AND instance_id = ?
                ORDER BY updated_at DESC, review_id DESC LIMIT 1""",
                scope.fingerprint_parts,
            ).fetchone()
            return {
                "active_recurring_count": int(active[0]),
                "pending_review_count": int(pending[0]),
                "auto_completed_count": int(completed[0]),
                "recent": (
                    {
                        "status": str(recent["status"]),
                        "decision": str(recent["decision"] or ""),
                        "error_code": str(recent["error_code"] or ""),
                        "updated_at": str(recent["updated_at"]),
                    }
                    if recent is not None
                    else None
                ),
            }

        return await self.db.call(operation)

    def _create_lifecycle_review(
        self,
        conn: sqlite3.Connection,
        *,
        scope: TimerScope,
        occurrence_id: TimerOccurrenceId,
        occurrence_generation: int,
        main_core_run_id: int,
        expected_activity_epoch: int,
        evidence: TimerLifecycleEvidence,
        now: datetime,
    ) -> TimerLifecycleReview | None:
        row = conn.execute(
            """SELECT occurrence.*, rule.prompt AS rule_prompt,
                rule.schedule_kind, rule.status AS rule_status, rule.version AS rule_version,
                rule.source_message_refs_json
            FROM timer_occurrences occurrence
            JOIN timer_rules rule ON rule.profile_id = occurrence.profile_id
              AND rule.instance_id = occurrence.instance_id
              AND rule.rule_id = occurrence.rule_id
            WHERE occurrence.profile_id = ? AND occurrence.instance_id = ?
              AND occurrence.occurrence_id = ?""",
            (*scope.fingerprint_parts, occurrence_id.value),
        ).fetchone()
        if row is None or str(row["schedule_kind"]) not in _RECURRING_KINDS:
            return None
        if str(row["rule_status"]) != "ACTIVE":
            return None
        if int(row["generation"]) != int(occurrence_generation):
            return None
        if str(row["status"]) not in _SUCCESS_OCCURRENCE_STATUSES:
            return None
        if str(row["execution_ref"] or "") != f"core-run:{int(main_core_run_id)}":
            return None
        run = conn.execute(
            """SELECT status FROM instance_core_runs
            WHERE profile_id = ? AND instance_id = ? AND run_id = ?""",
            (*scope.fingerprint_parts, int(main_core_run_id)),
        ).fetchone()
        if run is None or str(run["status"]) != "COMPLETED":
            return None
        enriched = TimerLifecycleEvidence(
            timer_description=str(evidence.timer_description or row["rule_prompt"]),
            source_messages=(
                evidence.source_messages
                or _source_message_texts(conn, scope, row["source_message_refs_json"])
            ),
            working_text=str(evidence.working_text or ""),
            decision_kind=str(evidence.decision_kind or ""),
            output_status=str(evidence.output_status or ""),
        )
        review_id = _review_id(scope, occurrence_id, occurrence_generation)
        stamp = encode_datetime(now)
        conn.execute(
            """INSERT INTO timer_lifecycle_reviews(
                profile_id, instance_id, review_id, rule_id, occurrence_id,
                occurrence_generation, main_core_run_id, expected_rule_version,
                expected_activity_epoch, evidence_json, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)
            ON CONFLICT(profile_id, instance_id, rule_id, occurrence_id,
                occurrence_generation) DO NOTHING""",
            (
                *scope.fingerprint_parts,
                review_id,
                str(row["rule_id"]),
                occurrence_id.value,
                int(occurrence_generation),
                int(main_core_run_id),
                int(row["rule_version"]),
                max(0, int(expected_activity_epoch)),
                _encode_evidence(enriched),
                stamp,
                stamp,
            ),
        )
        return self._get_lifecycle_review(conn, scope, review_id)

    @staticmethod
    def _get_lifecycle_review(
        conn: sqlite3.Connection, scope: TimerScope, review_id: str
    ) -> TimerLifecycleReview | None:
        row = conn.execute(
            """SELECT * FROM timer_lifecycle_reviews
            WHERE profile_id = ? AND instance_id = ? AND review_id = ?""",
            (*scope.fingerprint_parts, str(review_id)),
        ).fetchone()
        return _decode_review(row) if row is not None else None

    @staticmethod
    def _apply_lifecycle_review_result(
        conn: sqlite3.Connection,
        scope: TimerScope,
        review_id: str,
        *,
        decision: TimerLifecycleDecision | None,
        error_code: str,
        now: datetime,
    ) -> TimerLifecycleReviewStatus:
        review = conn.execute(
            """SELECT review.*, occurrence.original_due_at
            FROM timer_lifecycle_reviews review
            JOIN timer_occurrences occurrence
              ON occurrence.profile_id = review.profile_id
             AND occurrence.instance_id = review.instance_id
             AND occurrence.occurrence_id = review.occurrence_id
            WHERE review.profile_id = ? AND review.instance_id = ?
              AND review.review_id = ?""",
            (*scope.fingerprint_parts, review_id),
        ).fetchone()
        if review is None:
            raise KeyError((scope.profile_id, scope.instance_id, review_id))
        current = TimerLifecycleReviewStatus(str(review["status"]))
        if current is not TimerLifecycleReviewStatus.PENDING:
            return current
        stamp = encode_datetime(now)
        if decision is None or error_code:
            conn.execute(
                """UPDATE timer_lifecycle_reviews SET status = 'ERROR_KEEP',
                    error_code = ?, updated_at = ?, decided_at = ?
                WHERE profile_id = ? AND instance_id = ? AND review_id = ?
                  AND status = 'PENDING'""",
                (
                    str(error_code or "UNKNOWN")[:128],
                    stamp,
                    stamp,
                    *scope.fingerprint_parts,
                    review_id,
                ),
            )
            return TimerLifecycleReviewStatus.ERROR_KEEP
        if not _completion_fence_is_current(conn, review):
            conn.execute(
                """UPDATE timer_lifecycle_reviews SET status = 'STALE', decision = ?,
                    error_code = 'CAS_FENCE_CHANGED', updated_at = ?, decided_at = ?
                WHERE profile_id = ? AND instance_id = ? AND review_id = ?
                  AND status = 'PENDING'""",
                (decision.value, stamp, stamp, *scope.fingerprint_parts, review_id),
            )
            return TimerLifecycleReviewStatus.STALE
        if not decision.completes_rule:
            conn.execute(
                """UPDATE timer_lifecycle_reviews SET status = 'KEPT', decision = ?,
                    updated_at = ?, decided_at = ?, applied_at = ?
                WHERE profile_id = ? AND instance_id = ? AND review_id = ?
                  AND status = 'PENDING'""",
                (decision.value, stamp, stamp, stamp, *scope.fingerprint_parts, review_id),
            )
            return TimerLifecycleReviewStatus.KEPT
        changed = conn.execute(
            """UPDATE timer_rules SET status = 'COMPLETED', version = version + 1,
                last_operation_key = ?, last_operation_fingerprint = ''
            WHERE profile_id = ? AND instance_id = ? AND rule_id = ?
              AND status = 'ACTIVE' AND version = ?""",
            (
                f"system:lifecycle-completed:{review_id}",
                review["profile_id"],
                review["instance_id"],
                review["rule_id"],
                review["expected_rule_version"],
            ),
        ).rowcount
        if changed != 1:
            conn.execute(
                """UPDATE timer_lifecycle_reviews SET status = 'STALE', decision = ?,
                    error_code = 'RULE_CAS_LOST', updated_at = ?, decided_at = ?
                WHERE profile_id = ? AND instance_id = ? AND review_id = ?
                  AND status = 'PENDING'""",
                (decision.value, stamp, stamp, *scope.fingerprint_parts, review_id),
            )
            return TimerLifecycleReviewStatus.STALE
        conn.execute(
            """UPDATE timer_occurrences SET status = 'CANCELLED', version = version + 1,
                last_operation_key = ?, last_operation_fingerprint = ''
            WHERE profile_id = ? AND instance_id = ? AND rule_id = ?
              AND original_due_at > ? AND status IN ('SCHEDULED', 'WAITING', 'PAUSED')""",
            (
                f"system:lifecycle-completed:{review_id}",
                review["profile_id"],
                review["instance_id"],
                review["rule_id"],
                review["original_due_at"],
            ),
        )
        conn.execute(
            """UPDATE timer_lifecycle_reviews SET status = 'COMPLETED', decision = ?,
                updated_at = ?, decided_at = ?, applied_at = ?
            WHERE profile_id = ? AND instance_id = ? AND review_id = ?
              AND status = 'PENDING'""",
            (decision.value, stamp, stamp, stamp, *scope.fingerprint_parts, review_id),
        )
        return TimerLifecycleReviewStatus.COMPLETED

    def _recover_lifecycle_review_candidates(
        self,
        conn: sqlite3.Connection,
        limit: int,
        now: datetime,
    ) -> tuple[TimerLifecycleReview, ...]:
        rows = conn.execute(
            """SELECT occurrence.*, rule.prompt AS rule_prompt,
                rule.version AS rule_version, rule.source_message_refs_json,
                run.run_id AS main_core_run_id, run.workflow_id, run.request_json,
                state.activity_epoch
            FROM timer_occurrences occurrence
            JOIN timer_rules rule ON rule.profile_id = occurrence.profile_id
              AND rule.instance_id = occurrence.instance_id
              AND rule.rule_id = occurrence.rule_id
            JOIN instance_core_runs run ON run.profile_id = occurrence.profile_id
              AND run.instance_id = occurrence.instance_id
              AND run.run_id = CAST(substr(occurrence.execution_ref, 10) AS INTEGER)
            JOIN instance_core_state state ON state.profile_id = occurrence.profile_id
              AND state.instance_id = occurrence.instance_id
            LEFT JOIN timer_lifecycle_reviews review
              ON review.profile_id = occurrence.profile_id
             AND review.instance_id = occurrence.instance_id
             AND review.rule_id = occurrence.rule_id
             AND review.occurrence_id = occurrence.occurrence_id
             AND review.occurrence_generation = occurrence.generation
            WHERE rule.status = 'ACTIVE'
              AND rule.schedule_kind IN ('WEEKLY', 'YEARLY')
              AND occurrence.status IN ('COMPLETED', 'WAITING_DELIVERY')
              AND run.status = 'COMPLETED' AND review.review_id IS NULL
              AND NOT EXISTS (
                SELECT 1 FROM timer_occurrences newer
                WHERE newer.profile_id = occurrence.profile_id
                  AND newer.instance_id = occurrence.instance_id
                  AND newer.rule_id = occurrence.rule_id
                  AND newer.original_due_at > occurrence.original_due_at
                  AND newer.status IN ('COMPLETED', 'WAITING_DELIVERY')
              )
            ORDER BY occurrence.original_due_at DESC, occurrence.created_sequence DESC
            LIMIT ?""",
            (limit * 16,),
        ).fetchall()
        result: list[TimerLifecycleReview] = []
        seen_scopes: set[tuple[str, str]] = set()
        for row in rows:
            scope = TimerScope(str(row["profile_id"]), str(row["instance_id"]))
            if scope.fingerprint_parts in seen_scopes:
                continue
            seen_scopes.add(scope.fingerprint_parts)
            working_text = _terminal_working_text(conn, row["workflow_id"])
            review = self._create_lifecycle_review(
                conn,
                scope=scope,
                occurrence_id=TimerOccurrenceId(str(row["occurrence_id"])),
                occurrence_generation=int(row["generation"]),
                main_core_run_id=int(row["main_core_run_id"]),
                expected_activity_epoch=int(row["activity_epoch"]),
                evidence=TimerLifecycleEvidence(
                    timer_description=str(row["rule_prompt"]),
                    source_messages=_source_message_texts(
                        conn, scope, row["source_message_refs_json"]
                    ),
                    working_text=working_text,
                    decision_kind=_historical_decision_kind(conn, row["main_core_run_id"]),
                    output_status=(
                        "OUTPUT_COMMITTED" if row["delivery_ref"] else "SILENT_COMMITTED"
                    ),
                ),
                now=now,
            )
            if review is None:
                continue
            if not working_text:
                self._apply_lifecycle_review_result(
                    conn,
                    scope,
                    review.review_id,
                    decision=None,
                    error_code="MISSING_EVIDENCE",
                    now=now,
                )
                review = self._get_lifecycle_review(conn, scope, review.review_id)
                assert review is not None
            result.append(review)
            if len(result) >= limit:
                break
        return tuple(result)


def _completion_fence_is_current(conn: sqlite3.Connection, review: sqlite3.Row) -> bool:
    occurrence = conn.execute(
        """SELECT status, generation, original_due_at FROM timer_occurrences
        WHERE profile_id = ? AND instance_id = ? AND occurrence_id = ?""",
        (review["profile_id"], review["instance_id"], review["occurrence_id"]),
    ).fetchone()
    rule = conn.execute(
        """SELECT status, version FROM timer_rules
        WHERE profile_id = ? AND instance_id = ? AND rule_id = ?""",
        (review["profile_id"], review["instance_id"], review["rule_id"]),
    ).fetchone()
    state = conn.execute(
        """SELECT activity_epoch FROM instance_core_state
        WHERE profile_id = ? AND instance_id = ?""",
        (review["profile_id"], review["instance_id"]),
    ).fetchone()
    if occurrence is None or rule is None or state is None:
        return False
    if str(occurrence["status"]) != "COMPLETED":
        return False
    if int(occurrence["generation"]) != int(review["occurrence_generation"]):
        return False
    if str(rule["status"]) != "ACTIVE" or int(rule["version"]) != int(
        review["expected_rule_version"]
    ):
        return False
    if int(state["activity_epoch"]) != int(review["expected_activity_epoch"]):
        return False
    newer_foreground = conn.execute(
        """SELECT 1 FROM instance_core_runs WHERE profile_id = ? AND instance_id = ?
          AND run_id > ? AND source = 'FOREGROUND_MESSAGE' LIMIT 1""",
        (review["profile_id"], review["instance_id"], review["main_core_run_id"]),
    ).fetchone()
    if newer_foreground is not None:
        return False
    newer_started = conn.execute(
        """SELECT 1 FROM timer_occurrences WHERE profile_id = ? AND instance_id = ?
          AND rule_id = ? AND original_due_at > ? AND (
            execution_ref IS NOT NULL OR status IN (
              'CLAIMED', 'RUNNING', 'WAITING_DELIVERY', 'RECOVERING', 'COMPLETED', 'FAILED'
            )
          ) LIMIT 1""",
        (
            review["profile_id"],
            review["instance_id"],
            review["rule_id"],
            occurrence["original_due_at"],
        ),
    ).fetchone()
    return newer_started is None


def _source_message_texts(
    conn: sqlite3.Connection,
    scope: TimerScope,
    source_refs_json: object,
) -> tuple[str, ...]:
    try:
        values = json.loads(str(source_refs_json or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()
    identifiers: list[int] = []
    for raw in values if isinstance(values, list) else ():
        text = str(raw or "")
        if not text.startswith("ledger-message:"):
            continue
        try:
            identifier = int(text.removeprefix("ledger-message:"))
        except ValueError:
            continue
        if identifier > 0:
            identifiers.append(identifier)
    result: list[str] = []
    for identifier in identifiers[:8]:
        row = conn.execute(
            """SELECT plain_text FROM instance_messages
            WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
            (*scope.fingerprint_parts, identifier),
        ).fetchone()
        if row is not None and str(row["plain_text"] or "").strip():
            result.append(str(row["plain_text"]).strip()[:1000])
    return tuple(result)


def _terminal_working_text(conn: sqlite3.Connection, workflow_id: object) -> str:
    if workflow_id is None:
        return ""
    rows = conn.execute(
        """SELECT attempt.evaluation_json FROM ai_provider_attempts attempt
        JOIN ai_work_nodes node ON node.node_id = attempt.node_id
        WHERE node.workflow_id = ? AND attempt.status = 'SUCCEEDED'
        ORDER BY attempt.round_no DESC, attempt.attempt_no DESC, attempt.attempt_id DESC""",
        (int(workflow_id),),
    ).fetchall()
    for row in rows:
        try:
            value = json.loads(str(row["evaluation_json"] or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping) and bool(value.get("terminal")):
            return str(value.get("working_record") or "").strip()
    return ""


def _historical_decision_kind(conn: sqlite3.Connection, run_id: object) -> str:
    row = conn.execute(
        "SELECT decision_json FROM instance_core_runs WHERE run_id = ?",
        (int(run_id),),
    ).fetchone()
    if row is None:
        return ""
    try:
        decision = json.loads(str(row["decision_json"] or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    if not isinstance(decision, Mapping):
        return ""
    if decision.get("temporary_absence"):
        return "TEMPORARY_ABSENCE"
    if decision.get("no_op"):
        return "NO_REPLY"
    return "EXPRESSION"


def _review_id(scope: TimerScope, occurrence_id: TimerOccurrenceId, generation: int) -> str:
    payload = ":".join((*scope.fingerprint_parts, occurrence_id.value, str(generation)))
    return f"review:{hashlib.sha256(payload.encode()).hexdigest()[:48]}"


def _encode_evidence(value: TimerLifecycleEvidence) -> str:
    return json.dumps(
        {
            "timer_description": value.timer_description,
            "source_messages": list(value.source_messages),
            "working_text": value.working_text,
            "decision_kind": value.decision_kind,
            "output_status": value.output_status,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _decode_review(row: sqlite3.Row) -> TimerLifecycleReview:
    raw = json.loads(str(row["evidence_json"]))
    return TimerLifecycleReview(
        review_id=str(row["review_id"]),
        scope=TimerScope(str(row["profile_id"]), str(row["instance_id"])),
        rule_id=TimerRuleId(str(row["rule_id"])),
        occurrence_id=TimerOccurrenceId(str(row["occurrence_id"])),
        occurrence_generation=int(row["occurrence_generation"]),
        main_core_run_id=int(row["main_core_run_id"]),
        expected_rule_version=int(row["expected_rule_version"]),
        expected_activity_epoch=int(row["expected_activity_epoch"]),
        evidence=TimerLifecycleEvidence(
            timer_description=str(raw.get("timer_description") or ""),
            source_messages=tuple(str(item) for item in raw.get("source_messages", ()) or ()),
            working_text=str(raw.get("working_text") or ""),
            decision_kind=str(raw.get("decision_kind") or ""),
            output_status=str(raw.get("output_status") or ""),
        ),
        status=TimerLifecycleReviewStatus(str(row["status"])),
        decision=(
            TimerLifecycleDecision(str(row["decision"])) if str(row["decision"] or "") else None
        ),
        error_code=str(row["error_code"] or ""),
        task_id=int(row["task_id"]) if row["task_id"] is not None else None,
        created_at=decode_datetime(str(row["created_at"])),
        updated_at=decode_datetime(str(row["updated_at"])),
    )


__all__ = ["TimerLifecycleSqliteMixin"]
