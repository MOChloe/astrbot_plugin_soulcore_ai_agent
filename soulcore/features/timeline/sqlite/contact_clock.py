from __future__ import annotations

from ..contact_models import contact_day_bucket_transition
from .contact_answer import mark_latest_contact_attempt_answered_sql
from .contact_commit import ContactCommitContext, ContactCommitTransaction
from .support import (
    Any,
    Mapping,
    _contact_day_bucket,
    _dt,
    _dump,
    _now,
    datetime,
    sqlite3,
    timedelta,
)


def finalize_contact_attempt_sql(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    attempt_ref: str,
    generation: int,
    attempted: bool,
    success: bool,
    answered: bool,
    task_id: int | None,
    current: datetime,
    now_text: str,
) -> bool:
    """Finalize one contact attempt and its quota counters in the caller's transaction."""

    bucket = _contact_day_bucket(conn, profile_id, current)
    row = conn.execute(
        """SELECT * FROM contact_attempts WHERE profile_id = ?
        AND instance_id = ? AND attempt_ref = ? AND generation = ?""",
        (profile_id, instance_id, attempt_ref, int(generation)),
    ).fetchone()
    if row is None:
        raise KeyError((profile_id, instance_id, attempt_ref))
    if row["status"] == "FINALIZED":
        return False
    if task_id is not None and row["task_id"] not in (None, int(task_id)):
        return False
    cursor = conn.execute(
        """UPDATE contact_attempts SET status = 'FINALIZED', attempted = ?,
        success = ?, answered = ?, task_id = COALESCE(task_id, ?),
        finalized_at = ? WHERE profile_id = ? AND instance_id = ?
          AND attempt_ref = ? AND generation = ? AND status = 'READY'""",
        (
            int(attempted),
            int(success),
            int(answered),
            task_id,
            now_text,
            profile_id,
            instance_id,
            attempt_ref,
            int(generation),
        ),
    )
    if cursor.rowcount != 1:
        return False
    state = conn.execute(
        """SELECT daily_bucket, daily_success_count,
        consecutive_unanswered FROM instance_contact_state
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    if state is None:
        raise KeyError((profile_id, instance_id))
    effective_bucket, carry_daily_count = contact_day_bucket_transition(
        state["daily_bucket"], bucket
    )
    count = (int(state["daily_success_count"]) if carry_daily_count else 0) + int(success)
    unanswered_count = 0 if answered else int(state["consecutive_unanswered"]) + int(attempted)
    conn.execute(
        """UPDATE instance_contact_state SET
        last_attempt_at = CASE WHEN ? THEN ? ELSE last_attempt_at END,
        last_success_at = CASE WHEN ? THEN ? ELSE last_success_at END,
        daily_bucket = ?, daily_success_count = ?,
        consecutive_unanswered = ?,
        last_committed_task_id = COALESCE(?, last_committed_task_id),
        version = version + 1, updated_at = ?
        WHERE profile_id = ? AND instance_id = ?""",
        (
            int(attempted),
            now_text,
            int(success),
            now_text,
            effective_bucket,
            count,
            unanswered_count,
            task_id,
            now_text,
            profile_id,
            instance_id,
        ),
    )
    return True


class ContactClockRecords:
    async def claim_contact_clock(
        self,
        *,
        now: datetime | None = None,
        limit: int = 10,
        lease_seconds: int = 120,
        profile_id: str | None = None,
        instance_id: str | None = None,
    ) -> list[dict[str, Any]]:
        current = now or _now()
        now_text = _dt(current)
        lease_text = _dt(current + timedelta(seconds=max(1, lease_seconds)))

        def operation(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            conn.execute(
                """UPDATE instance_contact_state SET lease_until = NULL,
                lease_token = lease_token + 1, version = version + 1,
                generation = generation + 1,
                updated_at = ? WHERE lease_until IS NOT NULL AND lease_until <= ?""",
                (now_text, now_text),
            )
            sql = """SELECT cs.profile_id, cs.instance_id
                FROM instance_contact_state cs
                JOIN character_instances ci ON ci.profile_id = cs.profile_id
                    AND ci.instance_id = cs.instance_id
                JOIN role_profiles rp ON rp.profile_id = cs.profile_id
                WHERE cs.next_check_at IS NOT NULL AND cs.next_check_at <= ?
                  AND cs.lease_until IS NULL AND rp.enabled = 1
                  AND ci.initialization_state = 'READY'
                  AND ci.readiness = 'READY'"""
            params: list[Any] = [now_text]
            if profile_id is not None:
                sql += " AND cs.profile_id = ?"
                params.append(profile_id)
            if instance_id is not None:
                sql += " AND cs.instance_id = ?"
                params.append(instance_id)
            sql += " ORDER BY cs.next_check_at, cs.profile_id, cs.instance_id LIMIT ?"
            params.append(max(0, int(limit)))
            keys = [(r[0], r[1]) for r in conn.execute(sql, params)]
            claimed: list[sqlite3.Row] = []
            for owner, iid in keys:
                snapshot = conn.execute(
                    """SELECT cs.timeline_event_watermark, core.activity_epoch,
                    COALESCE(
                        MAX(event.event_id),
                        cs.timeline_event_watermark
                    ) AS through_event_id,
                    COALESCE((
                        SELECT MAX(action_event.event_id)
                        FROM character_intent_events action_event
                        JOIN character_intents action_intent
                          ON action_intent.intent_id = action_event.intent_id
                        WHERE action_event.profile_id = cs.profile_id
                          AND action_event.instance_id = cs.instance_id
                          AND action_intent.intent_kind = 'ACTION_INTENT'
                          AND action_event.to_status IN ('COMPLETED','CANCELLED','EXPIRED')
                    ), 0) AS through_action_event_id
                    FROM instance_contact_state cs
                    JOIN instance_core_state core ON core.profile_id = cs.profile_id
                        AND core.instance_id = cs.instance_id
                    LEFT JOIN background_role_timeline_events event
                      ON event.profile_id = cs.profile_id
                     AND event.instance_id = cs.instance_id
                    WHERE cs.profile_id = ? AND cs.instance_id = ?""",
                    (owner, iid),
                ).fetchone()
                evidence = {
                    "timeline_event_after": int(snapshot["timeline_event_watermark"]),
                    "timeline_event_through": int(snapshot["through_event_id"]),
                    "action_event_through": int(snapshot["through_action_event_id"]),
                }
                conn.execute(
                    """UPDATE instance_contact_state SET lease_until = ?,
                    lease_token = lease_token + 1, version = version + 1,
                    generation = generation + 1, last_check_at = ?,
                    activity_epoch_snapshot = ?, evidence_snapshot_json = ?,
                    updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND lease_until IS NULL""",
                    (
                        lease_text,
                        now_text,
                        int(snapshot["activity_epoch"]),
                        _dump(evidence),
                        now_text,
                        owner,
                        iid,
                    ),
                )
                row = conn.execute(
                    """SELECT cs.*, ci.route_umo, ci.platform_id,
                    ci.message_type, ci.target_id, ci.scope,
                    core.state_epoch, core.activity_epoch
                    FROM instance_contact_state cs
                    JOIN character_instances ci ON ci.profile_id = cs.profile_id
                        AND ci.instance_id = cs.instance_id
                    JOIN instance_core_state core ON core.profile_id = cs.profile_id
                        AND core.instance_id = cs.instance_id
                    WHERE cs.profile_id = ? AND cs.instance_id = ?
                      AND cs.lease_until = ?""",
                    (owner, iid, lease_text),
                ).fetchone()
                if row is not None:
                    claimed.append(row)
            return claimed

        rows = await self.uow.run(operation)
        return [
            self._record(
                row,
                json_columns=(
                    "evidence_watermarks_json",
                    "deferred_evidence_json",
                    "evidence_snapshot_json",
                ),
            )
            for row in rows
        ]

    async def list_role_timeline_events(
        self,
        profile_id: str,
        instance_id: str,
        *,
        after_event_id: int,
        through_event_id: int,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        """Return the frozen slice of new role-lived events for contact evaluation."""

        rows = await self.db.fetch_all(
            """SELECT event_id, public_ref, source, content, leftover_text,
                frame_start_at, frame_end_at, created_at
            FROM background_role_timeline_events AS event
            WHERE event.profile_id = ? AND event.instance_id = ?
              AND event.event_id > ? AND event.event_id <= ?
              AND NOT EXISTS (
                SELECT 1 FROM contact_evidence_reservations AS reservation
                WHERE reservation.profile_id = event.profile_id
                  AND reservation.instance_id = event.instance_id
                  AND reservation.evidence_kind = 'ROLE_TIMELINE_EVENT'
                  AND reservation.evidence_ref = CAST(event.event_id AS TEXT)
                  AND reservation.status IN ('RESERVED', 'CONSUMED', 'STALE')
              )
            ORDER BY event.event_id
            LIMIT ?""",
            (
                profile_id,
                instance_id,
                max(0, int(after_event_id)),
                max(0, int(through_event_id)),
                max(1, min(int(limit), 20)),
            ),
        )
        return [self._record(row, json_columns=()) for row in rows]

    async def commit_contact_clock(
        self,
        profile_id: str,
        instance_id: str,
        *,
        expected_version: int,
        lease_token: int,
        next_check_at: datetime | None,
        result: str,
        reason: str = "",
        success: bool = False,
        attempted: bool = False,
        answered: bool = False,
        timeline_event_watermark: int | None = None,
        evidence_watermarks: Mapping[str, Any] | None = None,
        deferred_evidence: Mapping[str, Any] | None = None,
        task_id: int | None = None,
        cooldown_until: datetime | None = None,
        now: datetime | None = None,
        expected_generation: int | None = None,
        expected_state_epoch: int | None = None,
        expected_activity_epoch: int | None = None,
        attempt_ref: str | None = None,
    ) -> dict[str, Any] | None:
        current = now or _now()

        def operation(conn: sqlite3.Connection) -> bool:
            context = ContactCommitContext(
                profile_id=profile_id,
                instance_id=instance_id,
                expected_version=expected_version,
                lease_token=lease_token,
                next_check_at=next_check_at,
                result=str(result),
                reason=str(reason),
                success=success,
                attempted=attempted,
                answered=answered,
                timeline_event_watermark=timeline_event_watermark,
                evidence_watermarks=evidence_watermarks,
                deferred_evidence=deferred_evidence,
                task_id=task_id,
                cooldown_until=cooldown_until,
                expected_generation=expected_generation,
                expected_state_epoch=expected_state_epoch,
                expected_activity_epoch=expected_activity_epoch,
                attempt_ref=attempt_ref,
                now_text=_dt(current),
                bucket=_contact_day_bucket(conn, profile_id, current),
            )
            return ContactCommitTransaction(context)(conn)

        if not await self.uow.run(operation):
            return None
        return await self.get_contact_state(profile_id, instance_id)

    async def finalize_contact_attempt(
        self,
        profile_id: str,
        instance_id: str,
        attempt_ref: str,
        *,
        generation: int,
        attempted: bool,
        success: bool,
        answered: bool,
        task_id: int | None = None,
        now: datetime | None = None,
    ) -> bool:
        reference = str(attempt_ref or "").strip()
        if not reference:
            raise ValueError("attempt_ref cannot be empty")
        current = now or _now()
        now_text = _dt(current)

        def operation(conn: sqlite3.Connection) -> bool:
            return finalize_contact_attempt_sql(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                attempt_ref=reference,
                generation=int(generation),
                attempted=attempted,
                success=success,
                answered=answered,
                task_id=task_id,
                current=current,
                now_text=now_text,
            )

        return await self.uow.run(operation)

    async def mark_latest_contact_attempt_answered(
        self,
        profile_id: str,
        instance_id: str,
        *,
        player_message_id: int,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Link a real player inbound to the latest attempted unanswered contact.

        A platform call can finish without an end-to-end delivery receipt.  A
        later player response is the first durable confirmation that the
        contact really arrived, so it promotes that attempt and releases its
        outbound ledger row for knowledge formation atomically.
        """

        current = now or _now()

        def operation(conn: sqlite3.Connection) -> sqlite3.Row | None:
            return mark_latest_contact_attempt_answered_sql(
                conn,
                profile_id,
                instance_id,
                player_message_id=int(player_message_id),
                now=current,
                refresh_knowledge_task=self._refresh_knowledge_task_sql,
            )

        row = await self.uow.run(operation)
        return self._record(row, json_columns=("evidence_snapshot_json",)) if row else None

    async def release_contact_clock(
        self,
        profile_id: str,
        instance_id: str,
        *,
        expected_version: int,
        lease_token: int,
        next_check_at: datetime | None = None,
        reason: str = "",
    ) -> bool:
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE instance_contact_state SET lease_until = NULL,
                next_check_at = COALESCE(?, next_check_at), last_result = 'RELEASED',
                last_reason = ?, version = version + 1, updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND version = ?
                  AND lease_token = ? AND lease_until IS NOT NULL""",
                (
                    _dt(next_check_at),
                    reason,
                    _dt(_now()),
                    profile_id,
                    instance_id,
                    expected_version,
                    lease_token,
                ),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1

    async def expedite_contact_clock(
        self,
        profile_id: str,
        instance_id: str,
        *,
        event_id: str | int,
        due_at: datetime,
        now: datetime | None = None,
    ) -> bool:
        event_key = str(event_id).strip()
        if not event_key:
            raise ValueError("event_id cannot be empty")
        now_text = _dt(now or _now())
        due_text = _dt(due_at)

        def operation(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                """INSERT INTO contact_expedite_events(
                    profile_id, instance_id, event_id, requested_at, requested_due_at
                ) VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING""",
                (profile_id, instance_id, event_key, now_text, due_text),
            )
            if cursor.rowcount != 1:
                return False
            updated = conn.execute(
                """UPDATE instance_contact_state SET next_check_at = CASE
                    WHEN next_check_at IS NULL OR next_check_at > ? THEN ?
                    ELSE next_check_at END,
                    generation = generation + 1, version = version + 1,
                    updated_at = ? WHERE profile_id = ? AND instance_id = ?""",
                (due_text, due_text, now_text, profile_id, instance_id),
            )
            if updated.rowcount != 1:
                raise KeyError((profile_id, instance_id))
            return True

        return await self.uow.run(operation)

    async def invalidate_contact_clock_for_foreground(
        self,
        profile_id: str,
        instance_id: str,
        *,
        activity_epoch: int,
        defer_until: datetime | None = None,
        reason: str = "foreground-activity",
    ) -> bool:
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> bool:
            current = conn.execute(
                """SELECT activity_epoch FROM instance_core_state
                WHERE profile_id = ? AND instance_id = ?""",
                (profile_id, instance_id),
            ).fetchone()
            if current is None or int(current[0]) != int(activity_epoch):
                return False
            cursor = conn.execute(
                """UPDATE instance_contact_state SET
                next_check_at = CASE WHEN ? IS NULL THEN next_check_at
                    WHEN next_check_at IS NULL OR next_check_at < ? THEN ?
                    ELSE next_check_at END,
                lease_until = NULL, lease_token = lease_token + 1,
                generation = generation + 1, version = version + 1,
                activity_epoch_snapshot = ?, evidence_snapshot_json = '{}',
                last_result = 'INVALIDATED', last_reason = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?""",
                (
                    _dt(defer_until),
                    _dt(defer_until),
                    _dt(defer_until),
                    int(activity_epoch),
                    reason,
                    now,
                    profile_id,
                    instance_id,
                ),
            )
            return cursor.rowcount == 1

        return await self.uow.run(operation)
