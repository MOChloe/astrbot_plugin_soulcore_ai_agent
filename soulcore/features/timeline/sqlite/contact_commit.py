from __future__ import annotations

from dataclasses import dataclass

from ..contact_models import contact_day_bucket_transition
from .support import Any, Mapping, _dt, _dump, _load, datetime, sqlite3


@dataclass(frozen=True, slots=True)
class ContactCommitContext:
    profile_id: str
    instance_id: str
    expected_version: int
    lease_token: int
    next_check_at: datetime | None
    result: str
    reason: str
    success: bool
    attempted: bool
    answered: bool
    timeline_event_watermark: int | None
    evidence_watermarks: Mapping[str, Any] | None
    deferred_evidence: Mapping[str, Any] | None
    task_id: int | None
    cooldown_until: datetime | None
    expected_generation: int | None
    expected_state_epoch: int | None
    expected_activity_epoch: int | None
    attempt_ref: str | None
    now_text: str
    bucket: str


class ContactCommitTransaction:
    def __init__(self, context: ContactCommitContext) -> None:
        self.context = context

    def __call__(self, conn: sqlite3.Connection) -> bool:
        row = self._load_claim(conn)
        if row is None or not self._generation_matches(row):
            return False
        if not self._epochs_match(conn, row):
            return False
        frozen_evidence = _load(row["evidence_snapshot_json"]) or {}
        if not self._watermark_matches(row, frozen_evidence):
            return False
        deferred = self._deferred_evidence(row)
        self._store_ready_attempt(conn, row, deferred)
        self._consume_suppressed_evidence(conn, row, frozen_evidence)
        self._update_state(conn, row, deferred)
        return True

    def _load_claim(self, conn: sqlite3.Connection) -> sqlite3.Row | None:
        context = self.context
        return conn.execute(
            """SELECT * FROM instance_contact_state WHERE profile_id = ?
            AND instance_id = ? AND version = ? AND lease_token = ?
            AND lease_until IS NOT NULL""",
            (
                context.profile_id,
                context.instance_id,
                context.expected_version,
                context.lease_token,
            ),
        ).fetchone()

    def _generation_matches(self, row: sqlite3.Row) -> bool:
        expected = self.context.expected_generation
        return expected is None or int(row["generation"]) == int(expected)

    def _epochs_match(self, conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
        context = self.context
        frozen_activity = row["activity_epoch_snapshot"]
        required_activity = context.expected_activity_epoch
        if required_activity is None and frozen_activity is not None:
            required_activity = int(frozen_activity)
        required_state = context.expected_state_epoch
        if required_activity is None and required_state is None:
            return True
        actual = conn.execute(
            """SELECT state_epoch, activity_epoch FROM instance_core_state
            WHERE profile_id = ? AND instance_id = ?""",
            (context.profile_id, context.instance_id),
        ).fetchone()
        return bool(
            actual is not None
            and (required_state is None or int(actual["state_epoch"]) == int(required_state))
            and (
                required_activity is None or int(actual["activity_epoch"]) == int(required_activity)
            )
        )

    def _watermark_matches(self, row: sqlite3.Row, frozen_evidence: Mapping[str, Any]) -> bool:
        watermark = self.context.timeline_event_watermark
        if watermark is None:
            return True
        through = frozen_evidence.get(
            "timeline_event_through",
            row["timeline_event_watermark"],
        )
        return int(watermark) <= int(through)

    def _deferred_evidence(self, row: sqlite3.Row) -> dict[str, Any]:
        context = self.context
        if context.deferred_evidence is not None:
            return dict(context.deferred_evidence)
        return dict(_load(row["deferred_evidence_json"]) or {})

    def _store_ready_attempt(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        deferred: dict[str, Any],
    ) -> None:
        context = self.context
        if context.result.upper() != "READY":
            return
        attempt_ref = str(context.attempt_ref or "").strip()
        if not attempt_ref:
            raise ValueError("READY contact commit requires attempt_ref")
        conn.execute(
            """INSERT INTO contact_attempts(
                profile_id, instance_id, attempt_ref, generation, task_id,
                evidence_snapshot_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING""",
            (
                context.profile_id,
                context.instance_id,
                attempt_ref,
                int(row["generation"]),
                context.task_id,
                row["evidence_snapshot_json"],
                context.now_text,
            ),
        )
        deferred["attempt_ref"] = attempt_ref
        deferred["generation"] = int(row["generation"])

    def _update_state(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        deferred: dict[str, Any],
    ) -> None:
        context = self.context
        old_count = int(row["daily_success_count"])
        bucket, carry_daily_count = contact_day_bucket_transition(
            row["daily_bucket"], context.bucket
        )
        count = (old_count if carry_daily_count else 0) + int(context.success)
        unanswered = (
            0 if context.answered else int(row["consecutive_unanswered"]) + int(context.attempted)
        )
        watermark = self._result_watermark(row)
        evidence = (
            context.evidence_watermarks
            if context.evidence_watermarks is not None
            else _load(row["evidence_watermarks_json"]) or {}
        )
        conn.execute(
            """UPDATE instance_contact_state SET next_check_at = ?,
            last_attempt_at = CASE WHEN ? THEN ? ELSE last_attempt_at END,
            last_success_at = CASE WHEN ? THEN ? ELSE last_success_at END,
            daily_bucket = ?, daily_success_count = ?,
            consecutive_unanswered = ?, cooldown_until = ?,
            timeline_event_watermark = ?, evidence_watermarks_json = ?,
            deferred_evidence_json = ?, last_result = ?, last_reason = ?,
            last_committed_task_id = COALESCE(?, last_committed_task_id),
            lease_until = NULL, evidence_snapshot_json = '{}',
            version = version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND version = ?
              AND lease_token = ?""",
            (
                _dt(context.next_check_at),
                int(context.attempted),
                context.now_text,
                int(context.success),
                context.now_text,
                bucket,
                count,
                unanswered,
                _dt(context.cooldown_until),
                watermark,
                _dump(evidence),
                _dump(deferred),
                context.result,
                context.reason,
                context.task_id,
                context.now_text,
                context.profile_id,
                context.instance_id,
                context.expected_version,
                context.lease_token,
            ),
        )

    def _consume_suppressed_evidence(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        frozen_evidence: Mapping[str, Any],
    ) -> None:
        if self.context.result.upper() not in {
            "DISABLED_CONSUMED",
            "DAILY_LIMIT",
            "UNANSWERED_LIMIT",
        }:
            return
        attempt_ref = (
            f"contact:{self.context.profile_id}:{self.context.instance_id}:"
            f"{int(row['generation'])}:suppressed"
        )
        common = (
            self.context.profile_id,
            self.context.instance_id,
            attempt_ref,
            int(row["generation"]),
            self.context.now_text,
            self.context.now_text,
            self.context.result.lower(),
        )
        timeline_through = int(
            frozen_evidence.get("timeline_event_through", row["timeline_event_watermark"])
        )
        conn.execute(
            """INSERT INTO contact_evidence_reservations(
                reservation_id, profile_id, instance_id, attempt_ref,
                contact_generation, evidence_kind, evidence_ref, status,
                reserved_at, resolved_at, resolution_reason
            ) SELECT 'contact-evidence:' || lower(hex(randomblob(16))),
                ?, ?, ?, ?, 'ROLE_TIMELINE_EVENT', CAST(event.event_id AS TEXT),
                'STALE', ?, ?, ?
            FROM background_role_timeline_events event
            WHERE event.profile_id = ? AND event.instance_id = ?
              AND event.event_id > ? AND event.event_id <= ?
              AND NOT EXISTS (
                SELECT 1 FROM contact_evidence_reservations prior
                WHERE prior.profile_id = event.profile_id
                  AND prior.instance_id = event.instance_id
                  AND prior.evidence_kind = 'ROLE_TIMELINE_EVENT'
                  AND prior.evidence_ref = CAST(event.event_id AS TEXT)
                  AND prior.status IN ('RESERVED','CONSUMED','STALE')
              )""",
            common
            + (
                self.context.profile_id,
                self.context.instance_id,
                int(row["timeline_event_watermark"]),
                timeline_through,
            ),
        )
        action_through = int(frozen_evidence.get("action_event_through", 0) or 0)
        conn.execute(
            """INSERT INTO contact_evidence_reservations(
                reservation_id, profile_id, instance_id, attempt_ref,
                contact_generation, evidence_kind, evidence_ref, status,
                reserved_at, resolved_at, resolution_reason
            ) SELECT 'contact-evidence:' || lower(hex(randomblob(16))),
                ?, ?, ?, ?, 'ACTION_RESULT', event.intent_id || ':' || event.event_id,
                'STALE', ?, ?, ?
            FROM character_intent_events event
            JOIN character_intents intent ON intent.intent_id = event.intent_id
            WHERE event.profile_id = ? AND event.instance_id = ?
              AND intent.intent_kind = 'ACTION_INTENT'
              AND event.to_status IN ('COMPLETED','CANCELLED','EXPIRED')
              AND event.event_id <= ?
              AND NOT EXISTS (
                SELECT 1 FROM contact_evidence_reservations prior
                WHERE prior.profile_id = event.profile_id
                  AND prior.instance_id = event.instance_id
                  AND prior.evidence_kind = 'ACTION_RESULT'
                  AND prior.evidence_ref = event.intent_id || ':' || event.event_id
                  AND prior.status IN ('RESERVED','CONSUMED','STALE')
              )""",
            common
            + (
                self.context.profile_id,
                self.context.instance_id,
                action_through,
            ),
        )

    def _result_watermark(self, row: sqlite3.Row) -> int:
        if self.context.result.upper() == "READY":
            return int(row["timeline_event_watermark"])
        return max(
            int(row["timeline_event_watermark"]),
            int(self.context.timeline_event_watermark or 0),
        )
