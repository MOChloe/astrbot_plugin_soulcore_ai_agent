"""Atomic recovery-wake claim and new Main Core run creation."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any

from ....contracts.models import RunStatus, WakeSource
from ....storage.sqlite.codec import dump_json, encode_datetime
from ..work_checkpoint import WorkRecoveryAction, WorkScope
from ..work_file_runtime import RecoveryWakeStatus, WorkRecoveryRunStart
from ..work_recovery import restore_work_session_for_new_run


class WorkRecoveryRunCommands:
    uow: Any
    db: Any

    async def start_work_recovery_run(
        self,
        *,
        profile_id: str,
        instance_id: str,
        work_ref: str,
        checkpoint_version: int,
        wakeup_id: int,
        reason: str,
        request: dict[str, Any],
        now: datetime,
    ) -> WorkRecoveryRunStart | None:
        result = await self.uow.run(
            lambda conn: self._start_work_recovery_run_sql(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                work_ref=work_ref,
                checkpoint_version=checkpoint_version,
                wakeup_id=wakeup_id,
                reason=reason,
                request=request,
                now=now,
            )
        )
        await self.db.publish_backup_after_commit()
        return result

    @staticmethod
    def _start_work_recovery_run_sql(
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        instance_id: str,
        work_ref: str,
        checkpoint_version: int,
        wakeup_id: int,
        reason: str,
        request: dict[str, Any],
        now: datetime,
    ) -> WorkRecoveryRunStart | None:
        from .work_file_operations import WorkFileSqliteOperations

        scope = WorkScope(profile_id, instance_id)
        operations = WorkFileSqliteOperations()
        wake = operations.get_recovery_wake(conn, scope, work_ref, checkpoint_version)
        if wake is None or wake.wakeup_id != int(wakeup_id):
            return None
        state = conn.execute(
            """SELECT state_epoch, activity_epoch FROM instance_core_state
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        ).fetchone()
        checkpoint_row = conn.execute(
            """SELECT checkpoint_json, recovery_envelope_json
            FROM main_core_work_checkpoints
            WHERE profile_id = ? AND instance_id = ? AND work_ref = ?
              AND checkpoint_version = ? AND status = 'RECOVERY_READY'""",
            (profile_id, instance_id, work_ref, checkpoint_version),
        ).fetchone()
        if state is None or checkpoint_row is None:
            return None
        current_gate = conn.execute(
            """SELECT role.enabled AS role_enabled,
                      instance.readiness,
                      scope.max_context_tokens,
                      scope.target_context_tokens
            FROM character_instances instance
            JOIN role_profiles role ON role.profile_id = instance.profile_id
            JOIN scope_configs scope
              ON scope.profile_id = instance.profile_id AND scope.scope = instance.scope
            WHERE instance.profile_id = ? AND instance.instance_id = ?""",
            (profile_id, instance_id),
        ).fetchone()
        if not _current_gate_allows_recovery(current_gate):
            _terminalize_wake(
                conn,
                wake,
                RecoveryWakeStatus.SUPERSEDED,
                "current instance, role, route, or budget gate rejected recovery",
                now,
            )
            return None
        from .work_checkpoint_codec import ready_record

        ready = ready_record(
            checkpoint_row["checkpoint_json"], checkpoint_row["recovery_envelope_json"]
        )
        if int(state["activity_epoch"]) != ready.checkpoint.baseline.activity_generation:
            _terminalize_wake(
                conn,
                wake,
                RecoveryWakeStatus.SUPERSEDED,
                "new player activity superseded file work recovery",
                now,
            )
            return None
        controlled_refs = _current_controlled_refs(conn, scope, work_ref)
        restored = restore_work_session_for_new_run(
            ready.envelope,
            expected_scope=scope,
            expected_work_ref=work_ref,
            expected_checkpoint_version=checkpoint_version,
            expected_run_generation=ready.checkpoint.run_generation,
            current_controlled_resource_refs=controlled_refs,
            currently_allowed_actions=frozenset(
                {
                    WorkRecoveryAction.REASSESS_PLAN,
                    WorkRecoveryAction.UPDATE_WORK,
                    WorkRecoveryAction.COMPLETE_WORK,
                    WorkRecoveryAction.CANCEL_WORK,
                }
            ),
        )
        if not restored.accepted or restored.session is None:
            _terminalize_wake(
                conn,
                wake,
                RecoveryWakeStatus.SUPERSEDED,
                str(restored.rejection or "recovery revalidation failed"),
                now,
            )
            return None
        if wake.status is RecoveryWakeStatus.CLAIMED:
            return _replay_or_retry_claimed_run(
                conn,
                operations,
                wake,
                restored.session,
                controlled_refs,
                state,
                reason,
                request,
                now,
            )
        if wake.status is not RecoveryWakeStatus.READY:
            return None
        return _create_claimed_run(
            conn,
            operations,
            wake,
            restored.session,
            controlled_refs,
            state,
            reason,
            request,
            now,
        )


def _current_gate_allows_recovery(row: sqlite3.Row | None) -> bool:
    if row is None or not bool(row["role_enabled"]) or str(row["readiness"]) != "READY":
        return False
    maximum = int(row["max_context_tokens"])
    target = int(row["target_context_tokens"])
    return maximum > 0 and 0 < target <= maximum


def _replay_or_retry_claimed_run(
    conn: sqlite3.Connection,
    operations: Any,
    wake: Any,
    session: Any,
    controlled_refs: frozenset[str],
    state: sqlite3.Row,
    reason: str,
    request: dict[str, Any],
    now: datetime,
) -> WorkRecoveryRunStart | None:
    run_id = int(wake.claimed_run_id or 0)
    run = conn.execute(
        """SELECT status FROM instance_core_runs
        WHERE profile_id = ? AND instance_id = ? AND run_id = ?""",
        (wake.scope.profile_id, wake.scope.instance_id, run_id),
    ).fetchone()
    if run is None:
        return None
    if run["status"] in {RunStatus.FAILED.value, RunStatus.SUPERSEDED.value}:
        return _create_rebound_run(
            conn,
            operations,
            wake,
            session,
            controlled_refs,
            state,
            reason,
            request,
            now,
        )
    if run["status"] != RunStatus.RUNNING.value:
        return None
    claim = operations.claim_recovery_wake(
        conn,
        wake.scope,
        wake.work_ref,
        wake.checkpoint_version,
        wake.wakeup_id,
        run_id,
        now,
    )
    return WorkRecoveryRunStart(
        run_id=run_id,
        session=session,
        envelope=claim.envelope,
        controlled_resource_refs=controlled_refs,
        expected_state_epoch=int(state["state_epoch"]),
        expected_activity_epoch=int(state["activity_epoch"]),
        replayed=True,
    )


def _create_rebound_run(
    conn: sqlite3.Connection,
    operations: Any,
    wake: Any,
    session: Any,
    controlled_refs: frozenset[str],
    state: sqlite3.Row,
    reason: str,
    request: dict[str, Any],
    now: datetime,
) -> WorkRecoveryRunStart:
    previous_run_id = int(wake.claimed_run_id or 0)
    cursor = conn.execute(
        """INSERT INTO instance_core_runs(
        profile_id, instance_id, source, status, reason, request_json,
        expected_state_epoch, started_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            wake.scope.profile_id,
            wake.scope.instance_id,
            WakeSource.PLUGIN_WAKE.value,
            RunStatus.RUNNING.value,
            reason,
            dump_json(request),
            int(state["state_epoch"]),
            encode_datetime(now),
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("recovery retry run insert did not return a row id")
    run_id = int(cursor.lastrowid)
    rebound = conn.execute(
        """UPDATE main_core_work_recovery_wakes
        SET claimed_run_id = ?, claimed_at = ?, updated_at = ?
        WHERE profile_id = ? AND instance_id = ? AND work_ref = ?
        AND checkpoint_version = ? AND wakeup_id = ? AND status = 'CLAIMED'
        AND claimed_run_id = ?""",
        (
            run_id,
            encode_datetime(now),
            encode_datetime(now),
            wake.scope.profile_id,
            wake.scope.instance_id,
            wake.work_ref,
            wake.checkpoint_version,
            wake.wakeup_id,
            previous_run_id,
        ),
    )
    if rebound.rowcount != 1:
        raise RuntimeError("recovery retry wake rebind lost")
    claim = operations.claim_recovery_wake(
        conn,
        wake.scope,
        wake.work_ref,
        wake.checkpoint_version,
        wake.wakeup_id,
        run_id,
        now,
    )
    return WorkRecoveryRunStart(
        run_id=run_id,
        session=session,
        envelope=claim.envelope,
        controlled_resource_refs=controlled_refs,
        expected_state_epoch=int(state["state_epoch"]),
        expected_activity_epoch=int(state["activity_epoch"]),
    )


def _create_claimed_run(
    conn: sqlite3.Connection,
    operations: Any,
    wake: Any,
    session: Any,
    controlled_refs: frozenset[str],
    state: sqlite3.Row,
    reason: str,
    request: dict[str, Any],
    now: datetime,
) -> WorkRecoveryRunStart:
    cursor = conn.execute(
        """INSERT INTO instance_core_runs(
                profile_id, instance_id, source, status, reason, request_json,
                expected_state_epoch, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            wake.scope.profile_id,
            wake.scope.instance_id,
            WakeSource.PLUGIN_WAKE.value,
            RunStatus.RUNNING.value,
            reason,
            dump_json(request),
            int(state["state_epoch"]),
            encode_datetime(now),
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("recovery run insert did not return a row id")
    run_id = int(cursor.lastrowid)
    claim = operations.claim_recovery_wake(
        conn,
        wake.scope,
        wake.work_ref,
        wake.checkpoint_version,
        wake.wakeup_id,
        run_id,
        now,
    )
    return WorkRecoveryRunStart(
        run_id=run_id,
        session=session,
        envelope=claim.envelope,
        controlled_resource_refs=controlled_refs,
        expected_state_epoch=int(state["state_epoch"]),
        expected_activity_epoch=int(state["activity_epoch"]),
    )


def _current_controlled_refs(
    conn: sqlite3.Connection, scope: WorkScope, work_ref: str
) -> frozenset[str]:
    rows = conn.execute(
        """SELECT binding.request_ref, binding.status, binding.resource_ref,
                  binding.todo_id, asset.file_status, todo.status AS todo_status
        FROM main_core_work_file_bindings binding
        LEFT JOIN file_assets asset ON asset.asset_id = binding.resource_ref
        LEFT JOIN important_todos todo ON todo.todo_id = binding.todo_id
        WHERE binding.profile_id = ? AND binding.instance_id = ?
          AND binding.work_ref = ? ORDER BY binding.request_ref""",
        (scope.profile_id, scope.instance_id, work_ref),
    ).fetchall()
    refs: set[str] = set()
    for row in rows:
        refs.add(str(row["request_ref"]))
        if row["resource_ref"] and row["file_status"] == "AVAILABLE":
            refs.add(str(row["resource_ref"]))
        if row["todo_id"] and row["todo_status"] in {
            "PENDING",
            "SELECTED",
            "DELIVERY_PENDING",
            "DELIVERY_UNKNOWN",
        }:
            refs.add(str(row["todo_id"]))
    return frozenset(refs)


def _terminalize_wake(
    conn: sqlite3.Connection,
    wake: Any,
    status: RecoveryWakeStatus,
    reason: str,
    now: datetime,
) -> None:
    conn.execute(
        """UPDATE main_core_work_recovery_wakes
        SET status = ?, terminal_reason = ?, updated_at = ?
        WHERE profile_id = ? AND instance_id = ? AND work_ref = ?
          AND checkpoint_version = ? AND status = 'READY'""",
        (
            status.value,
            str(reason or "")[:400],
            encode_datetime(now),
            wake.scope.profile_id,
            wake.scope.instance_id,
            wake.work_ref,
            wake.checkpoint_version,
        ),
    )


__all__ = ["WorkRecoveryRunCommands"]
