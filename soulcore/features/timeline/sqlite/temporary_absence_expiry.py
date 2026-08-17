from __future__ import annotations

from dataclasses import dataclass

from ..temporary_absence import (
    TemporaryAbsenceExpiryWake,
    temporary_absence_expiry_payload,
)
from .support import Any, _dt, _dump, _parse, datetime, sqlite3


@dataclass(frozen=True, slots=True)
class _ExpiryPreparation:
    profile_id: str
    instance_id: str
    wakeup_id: int
    wakeup_generation: int
    wakeup_lease_token: int
    wakeup_version: int
    wakeup_idempotency_key: str
    gate_generation: int
    activity_epoch: int
    now: datetime
    point: str


class TemporaryAbsenceExpiryRecords:
    """Durable natural-expiry ownership for explicit temporary absences."""

    @staticmethod
    def _clear_finished_temporary_absence_in_transaction(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        *,
        gate_generation: int,
        snapshot_generation: int,
        point: str,
    ) -> bool:
        changed = conn.execute(
            """UPDATE instance_state_gate_snapshots
            SET reason_code = '', expression_context = '',
                not_before_at = NULL, until_at = NULL, source_run_id = NULL,
                version = version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?
              AND action = 'OPEN' AND reason_code = 'TEMPORARY_ABSENCE'
              AND generation = ?
              AND NOT EXISTS (
                SELECT 1 FROM deferred_message_batches AS batch
                WHERE batch.profile_id = instance_state_gate_snapshots.profile_id
                  AND batch.instance_id = instance_state_gate_snapshots.instance_id
                  AND batch.gate_generation = ?
                  AND batch.status IN ('PENDING', 'CLAIMED')
              )""",
            (
                point,
                profile_id,
                instance_id,
                int(snapshot_generation),
                int(gate_generation),
            ),
        ).rowcount
        return changed == 1

    async def ensure_expired_temporary_absence_wakeups(
        self,
        *,
        now: datetime,
        limit: int = 100,
    ) -> int:
        point = _dt(now)

        def operation(conn: sqlite3.Connection) -> int:
            rows = conn.execute(
                """SELECT snapshot.profile_id, snapshot.instance_id,
                    snapshot.generation, snapshot.expression_context,
                    snapshot.not_before_at, snapshot.until_at,
                    snapshot.source_run_id, state.activity_epoch,
                    instance.route_umo,
                    EXISTS (
                        SELECT 1 FROM instance_messages AS message
                        WHERE message.profile_id = snapshot.profile_id
                          AND message.instance_id = snapshot.instance_id
                          AND message.direction = 'INBOUND'
                          AND message.delivery_status = 'RECEIVED'
                          AND message.occurred_at >= snapshot.until_at
                    ) AS post_expiry_inbound
                FROM instance_state_gate_snapshots AS snapshot
                JOIN instance_core_state AS state
                  ON state.profile_id = snapshot.profile_id
                 AND state.instance_id = snapshot.instance_id
                JOIN character_instances AS instance
                  ON instance.profile_id = snapshot.profile_id
                 AND instance.instance_id = snapshot.instance_id
                WHERE snapshot.action = 'DEFER'
                  AND snapshot.reason_code = 'TEMPORARY_ABSENCE'
                  AND snapshot.until_at IS NOT NULL AND snapshot.until_at <= ?
                ORDER BY snapshot.until_at, snapshot.profile_id, snapshot.instance_id LIMIT ?""",
                (point, max(1, int(limit))),
            ).fetchall()
            created = 0
            for row in rows:
                started_at = _parse(str(row["not_before_at"] or ""))
                planned_until = _parse(str(row["until_at"] or ""))
                if started_at is None or planned_until is None:
                    raise ValueError("temporary absence snapshot has invalid timestamps")
                activity_epoch = int(row["activity_epoch"])
                marker = TemporaryAbsenceExpiryWake(
                    gate_generation=int(row["generation"]),
                    activity_epoch=(
                        max(0, activity_epoch - 1)
                        if bool(row["post_expiry_inbound"])
                        else activity_epoch
                    ),
                    source_run_id=int(row["source_run_id"] or 0),
                )
                created += conn.execute(
                    """INSERT INTO instance_wakeups(
                        profile_id, instance_id, source, due_at, reason,
                        conversation_ref, idempotency_key, payload_json, status,
                        intent_kind, created_at, updated_at
                    ) VALUES (?, ?, 'PLUGIN_WAKE', ?, ?, ?, ?, ?, 'PENDING',
                        'PLUGIN_WAKE', ?, ?)
                    ON CONFLICT(profile_id, instance_id, idempotency_key)
                        WHERE idempotency_key IS NOT NULL DO UPDATE SET
                        due_at = excluded.due_at,
                        reason = excluded.reason,
                        conversation_ref = excluded.conversation_ref,
                        payload_json = excluded.payload_json,
                        status = 'PENDING', lease_until = NULL, last_error = NULL,
                        generation = instance_wakeups.generation + 1,
                        lease_token = instance_wakeups.lease_token + 1,
                        version = instance_wakeups.version + 1,
                        updated_at = excluded.updated_at
                    WHERE instance_wakeups.status IN ('FAILED', 'CANCELLED')""",
                    (
                        row["profile_id"],
                        row["instance_id"],
                        _dt(planned_until),
                        "此前主动暂离的时间自然结束",
                        row["route_umo"],
                        marker.idempotency_key,
                        _dump(
                            temporary_absence_expiry_payload(
                                reason=str(row["expression_context"] or ""),
                                started_at=started_at,
                                planned_until=planned_until,
                                gate_generation=marker.gate_generation,
                                activity_epoch=marker.activity_epoch,
                                source_run_id=marker.source_run_id,
                            )
                        ),
                        point,
                        point,
                    ),
                ).rowcount
            return created

        return int(await self.uow.run(operation))

    @staticmethod
    def _claimed_expiry_wakeup_exists(
        conn: sqlite3.Connection,
        context: _ExpiryPreparation,
    ) -> bool:
        row = conn.execute(
            """SELECT 1 FROM instance_wakeups
            WHERE profile_id = ? AND instance_id = ? AND wakeup_id = ?
              AND status = 'CLAIMED' AND generation = ?
              AND lease_token = ? AND version = ?
              AND source = 'PLUGIN_WAKE' AND idempotency_key = ?""",
            (
                context.profile_id,
                context.instance_id,
                context.wakeup_id,
                context.wakeup_generation,
                context.wakeup_lease_token,
                context.wakeup_version,
                context.wakeup_idempotency_key,
            ),
        ).fetchone()
        return row is not None

    @staticmethod
    def _temporary_absence_policy_enabled(
        conn: sqlite3.Connection,
        context: _ExpiryPreparation,
    ) -> bool:
        policy = conn.execute(
            """SELECT COALESCE(override.enabled, policy.enabled) AS enabled
            FROM character_instances AS instance
            JOIN scope_state_gate_policies AS policy
              ON policy.profile_id = instance.profile_id AND policy.scope = instance.scope
            LEFT JOIN instance_state_gate_overrides AS override
              ON override.profile_id = instance.profile_id
             AND override.instance_id = instance.instance_id
            WHERE instance.profile_id = ? AND instance.instance_id = ?""",
            (context.profile_id, context.instance_id),
        ).fetchone()
        return policy is not None and bool(policy["enabled"])

    @staticmethod
    def _clear_disabled_temporary_absence(
        conn: sqlite3.Connection,
        context: _ExpiryPreparation,
    ) -> None:
        conn.execute(
            """UPDATE instance_state_gate_snapshots SET action = 'OPEN',
            reason_code = '', expression_context = '', not_before_at = NULL,
            until_at = NULL, source_run_id = NULL,
            generation = CASE WHEN action = 'DEFER' THEN generation + 1 ELSE generation END,
            version = version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?
              AND reason_code = 'TEMPORARY_ABSENCE'
              AND ((action = 'DEFER' AND generation = ?)
                OR (action = 'OPEN' AND generation = ?))""",
            (
                context.point,
                context.profile_id,
                context.instance_id,
                context.gate_generation,
                context.gate_generation + 1,
            ),
        )

    @staticmethod
    def _open_due_temporary_absence(
        conn: sqlite3.Connection,
        snapshot: Any,
        context: _ExpiryPreparation,
    ) -> tuple[dict[str, Any] | None, int]:
        action = str(snapshot["action"] or "").upper()
        snapshot_generation = int(snapshot["generation"])
        if action == "OPEN":
            if snapshot_generation == context.gate_generation + 1:
                return None, snapshot_generation
            return {"outcome": "SUPERSEDED"}, snapshot_generation
        planned_until = _parse(str(snapshot["until_at"] or ""))
        if action != "DEFER" or snapshot_generation != context.gate_generation:
            return {"outcome": "SUPERSEDED"}, snapshot_generation
        if planned_until is None:
            return {"outcome": "SUPERSEDED"}, snapshot_generation
        if planned_until > context.now:
            return {"outcome": "NOT_DUE", "retry_at": _dt(planned_until)}, snapshot_generation
        changed = conn.execute(
            """UPDATE instance_state_gate_snapshots SET action = 'OPEN',
            generation = generation + 1, version = version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND version = ?
              AND action = 'DEFER' AND reason_code = 'TEMPORARY_ABSENCE'
              AND generation = ?""",
            (
                context.point,
                context.profile_id,
                context.instance_id,
                int(snapshot["version"]),
                context.gate_generation,
            ),
        ).rowcount
        outcome = None if changed == 1 else {"outcome": "SUPERSEDED"}
        return outcome, snapshot_generation + int(changed == 1)

    def _prepare_temporary_absence_expiry_in_transaction(
        self,
        conn: sqlite3.Connection,
        context: _ExpiryPreparation,
    ) -> dict[str, Any]:
        if not self._claimed_expiry_wakeup_exists(conn, context):
            return {"outcome": "LEASE_LOST"}
        snapshot = conn.execute(
            """SELECT * FROM instance_state_gate_snapshots
            WHERE profile_id = ? AND instance_id = ?""",
            (context.profile_id, context.instance_id),
        ).fetchone()
        if snapshot is None or str(snapshot["reason_code"] or "") != "TEMPORARY_ABSENCE":
            return {"outcome": "SUPERSEDED"}
        if not self._temporary_absence_policy_enabled(conn, context):
            self._clear_disabled_temporary_absence(conn, context)
            return {"outcome": "SUPERSEDED"}
        outcome, snapshot_generation = self._open_due_temporary_absence(conn, snapshot, context)
        if outcome is not None:
            return outcome
        pending = conn.execute(
            """SELECT 1 FROM deferred_message_batches
            WHERE profile_id = ? AND instance_id = ? AND gate_generation = ?
              AND status IN ('PENDING', 'CLAIMED') LIMIT 1""",
            (context.profile_id, context.instance_id, context.gate_generation),
        ).fetchone()
        if pending is not None:
            return {"outcome": "DEFERRED_MESSAGES"}
        state = conn.execute(
            """SELECT activity_epoch FROM instance_core_state
            WHERE profile_id = ? AND instance_id = ?""",
            (context.profile_id, context.instance_id),
        ).fetchone()
        if state is None:
            return {"outcome": "SUPERSEDED"}
        if int(state["activity_epoch"]) == context.activity_epoch:
            return {"outcome": "DISPATCH"}
        self._clear_finished_temporary_absence_in_transaction(
            conn,
            context.profile_id,
            context.instance_id,
            gate_generation=context.gate_generation,
            snapshot_generation=snapshot_generation,
            point=context.point,
        )
        return {"outcome": "FOREGROUND_ACTIVITY"}

    async def prepare_temporary_absence_expiry(
        self,
        profile_id: str,
        instance_id: str,
        *,
        wakeup_id: int,
        expected_wakeup_generation: int,
        wakeup_lease_token: int,
        expected_wakeup_version: int,
        expected_wakeup_idempotency_key: str,
        gate_generation: int,
        expected_activity_epoch: int,
        now: datetime,
    ) -> dict[str, Any]:
        context = _ExpiryPreparation(
            profile_id=profile_id,
            instance_id=instance_id,
            wakeup_id=int(wakeup_id),
            wakeup_generation=int(expected_wakeup_generation),
            wakeup_lease_token=int(wakeup_lease_token),
            wakeup_version=int(expected_wakeup_version),
            wakeup_idempotency_key=str(expected_wakeup_idempotency_key),
            gate_generation=int(gate_generation),
            activity_epoch=int(expected_activity_epoch),
            now=now,
            point=_dt(now),
        )
        return dict(
            await self.uow.run(
                lambda conn: self._prepare_temporary_absence_expiry_in_transaction(conn, context)
            )
        )

    async def finalize_temporary_absence_expiry(
        self,
        profile_id: str,
        instance_id: str,
        *,
        gate_generation: int,
        now: datetime,
    ) -> bool:
        point = _dt(now)
        return bool(
            await self.uow.run(
                lambda conn: self._clear_finished_temporary_absence_in_transaction(
                    conn,
                    profile_id,
                    instance_id,
                    gate_generation=int(gate_generation),
                    snapshot_generation=int(gate_generation) + 1,
                    point=point,
                )
            )
        )


__all__ = ["TemporaryAbsenceExpiryRecords"]
