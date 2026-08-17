"""SQLite primitives for file-backed Main Core work recovery."""

from __future__ import annotations

import sqlite3
from datetime import datetime

from ....storage.sqlite.codec import decode_datetime, dump_json, encode_datetime, load_json
from ..work_checkpoint import (
    ControlledWorkResult,
    MainCoreWorkCheckpoint,
    WorkCallbackEnvelope,
    WorkCallbackOutcome,
    WorkCheckpointStatus,
    WorkRecoveryBaseline,
    WorkRecoveryRuntime,
    WorkScope,
)
from ..work_checkpoint_repository import ApplyWorkCallbackCommand
from ..work_checkpoint_storage_errors import WorkCheckpointStorageErrorCode, storage_fail
from ..work_file_runtime import (
    FileWorkBindingRecord,
    FileWorkBindingSpec,
    FileWorkBindingStatus,
    RecoveryWakeClaim,
    RecoveryWakeRecord,
    RecoveryWakeStatus,
)
from .work_checkpoint_codec import ready_record
from .work_checkpoint_operations import WorkCheckpointSqliteOperations


class WorkFileSqliteOperations:
    """Explicit-connection operations that compose into larger transactions."""

    def bind_file_jobs(
        self,
        conn: sqlite3.Connection,
        checkpoint: MainCoreWorkCheckpoint,
        bindings: tuple[FileWorkBindingSpec, ...],
        now: datetime,
    ) -> tuple[FileWorkBindingRecord, ...]:
        if not bindings or len(bindings) > 3:
            raise storage_fail(WorkCheckpointStorageErrorCode.OUT_OF_RANGE)
        if len({item.job_id for item in bindings}) != len(bindings) or len(
            {item.request_ref for item in bindings}
        ) != len(bindings):
            raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_STATE)
        self._require_waiting_checkpoint(conn, checkpoint)
        callback_sequence = checkpoint.callback_sequence + 1
        for binding in bindings:
            job = conn.execute(
                """SELECT profile_id, instance_id, status FROM file_generation_jobs
                WHERE job_id = ?""",
                (binding.job_id,),
            ).fetchone()
            if job is None:
                raise storage_fail(WorkCheckpointStorageErrorCode.NOT_FOUND)
            if (job["profile_id"], job["instance_id"]) != (
                checkpoint.scope.profile_id,
                checkpoint.scope.instance_id,
            ):
                raise storage_fail(WorkCheckpointStorageErrorCode.SCOPE_MISMATCH)
            if str(job["status"]) not in {"QUEUED", "RUNNING"}:
                raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_STATE)
            existing = conn.execute(
                "SELECT * FROM main_core_work_file_bindings WHERE job_id = ?",
                (binding.job_id,),
            ).fetchone()
            if existing is not None:
                if not _same_binding(existing, checkpoint, binding, callback_sequence):
                    raise storage_fail(WorkCheckpointStorageErrorCode.IDEMPOTENCY_CONFLICT)
                continue
            conn.execute(
                """INSERT INTO main_core_work_file_bindings(
                    job_id, profile_id, instance_id, work_ref, request_ref, slot_id,
                    checkpoint_version, run_generation, callback_sequence,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)""",
                (
                    binding.job_id,
                    checkpoint.scope.profile_id,
                    checkpoint.scope.instance_id,
                    checkpoint.work_ref,
                    binding.request_ref,
                    binding.slot_id,
                    checkpoint.checkpoint_version,
                    checkpoint.run_generation,
                    callback_sequence,
                    encode_datetime(now),
                    encode_datetime(now),
                ),
            )
        return self.list_file_bindings(conn, checkpoint.scope, checkpoint.work_ref)

    def list_file_bindings(
        self, conn: sqlite3.Connection, scope: WorkScope, work_ref: str
    ) -> tuple[FileWorkBindingRecord, ...]:
        rows = conn.execute(
            """SELECT * FROM main_core_work_file_bindings
            WHERE profile_id = ? AND instance_id = ? AND work_ref = ?
            ORDER BY request_ref, job_id""",
            (scope.profile_id, scope.instance_id, work_ref),
        ).fetchall()
        return tuple(_binding_record(row) for row in rows)

    def complete_file_job(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        status: FileWorkBindingStatus | str,
        resource_ref: str,
        result_kind: str,
        result_summary: str,
        todo_id: str,
        now: datetime,
    ) -> bool:
        status = FileWorkBindingStatus(status)
        row = conn.execute(
            "SELECT * FROM main_core_work_file_bindings WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            return False
        current = _binding_record(row)
        if status is FileWorkBindingStatus.PENDING:
            raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_STATE)
        if current.status is not FileWorkBindingStatus.PENDING:
            expected_ref = resource_ref if status is FileWorkBindingStatus.SUCCEEDED else ""
            if (
                current.status is not status
                or current.resource_ref != expected_ref
                or current.result_kind != result_kind
                or current.todo_id != todo_id
            ):
                raise storage_fail(WorkCheckpointStorageErrorCode.IDEMPOTENCY_CONFLICT)
            self._finalize_ready_group(conn, current.scope, current.work_ref, now)
            return True
        updated = conn.execute(
            """UPDATE main_core_work_file_bindings
            SET status = ?, resource_ref = ?, result_kind = ?, result_summary = ?,
                todo_id = ?, completed_at = ?, updated_at = ?
            WHERE job_id = ? AND status = 'PENDING'""",
            (
                status.value,
                resource_ref if status is FileWorkBindingStatus.SUCCEEDED else "",
                result_kind,
                str(result_summary or "")[:1000],
                todo_id,
                encode_datetime(now),
                encode_datetime(now),
                job_id,
            ),
        )
        if updated.rowcount != 1:
            raise storage_fail(WorkCheckpointStorageErrorCode.VERSION_CONFLICT)
        self._finalize_ready_group(conn, current.scope, current.work_ref, now)
        return True

    def _finalize_ready_group(
        self,
        conn: sqlite3.Connection,
        scope: WorkScope,
        work_ref: str,
        now: datetime,
    ) -> None:
        bindings = self.list_file_bindings(conn, scope, work_ref)
        if not bindings or any(item.status is FileWorkBindingStatus.PENDING for item in bindings):
            return
        checkpoint_ops = WorkCheckpointSqliteOperations()
        checkpoint = checkpoint_ops.get_checkpoint(conn, scope, work_ref)
        if checkpoint is None:
            raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_PERSISTED_DATA)
        if checkpoint.status is WorkCheckpointStatus.RECOVERY_READY:
            self._ensure_recovery_wake(conn, scope, work_ref, checkpoint.checkpoint_version, now)
            return
        if checkpoint.status is not WorkCheckpointStatus.WAITING or checkpoint.lease is None:
            return
        callback_results = tuple(
            ControlledWorkResult(
                item.slot_id,
                item.resource_ref
                if item.status is FileWorkBindingStatus.SUCCEEDED
                else str(item.todo_id or ""),
                item.result_kind,
                item.callback_sequence,
            )
            for item in bindings
        )
        if any(not item.resource_ref for item in callback_results):
            raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_PERSISTED_DATA)
        outcome = _aggregate_outcome(bindings)
        runtime = WorkRecoveryRuntime(
            scope=scope,
            now=now,
            baseline=self._current_baseline(conn, scope),
            controlled_resource_refs=frozenset(item.resource_ref for item in callback_results),
            process_restarted=False,
        )
        callback = WorkCallbackEnvelope(
            scope=scope,
            work_ref=work_ref,
            checkpoint_version=bindings[0].checkpoint_version,
            run_generation=bindings[0].run_generation,
            callback_sequence=bindings[0].callback_sequence,
            lease_owner=checkpoint.lease.owner,
            lease_token=checkpoint.lease.token,
            idempotency_key=f"file-work-callback:{scope.instance_id}:{checkpoint.callback_sequence + 1}",
            outcome=outcome,
            result_summary=_aggregate_summary(bindings, outcome),
            results=callback_results,
        )
        result = checkpoint_ops.apply_callback(conn, ApplyWorkCallbackCommand(callback, runtime))
        if not result.decision.accepted:
            return
        ready = result.decision.checkpoint
        self._ensure_recovery_wake(conn, scope, work_ref, ready.checkpoint_version, now)

    def _ensure_recovery_wake(
        self,
        conn: sqlite3.Connection,
        scope: WorkScope,
        work_ref: str,
        checkpoint_version: int,
        now: datetime,
    ) -> None:
        existing = self._recovery_wake(conn, scope, work_ref, checkpoint_version)
        if existing is not None:
            return
        todo = conn.execute(
            """SELECT todo_id FROM main_core_work_file_bindings
            WHERE profile_id = ? AND instance_id = ? AND work_ref = ?
              AND todo_id IS NOT NULL ORDER BY request_ref LIMIT 1""",
            (scope.profile_id, scope.instance_id, work_ref),
        ).fetchone()
        if todo is None:
            raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_PERSISTED_DATA)
        key = f"important-todo:{todo['todo_id']}"
        payload = {"work_ref": work_ref, "work_version": checkpoint_version}
        conn.execute(
            """INSERT INTO instance_wakeups(
                profile_id, instance_id, source, due_at, reason,
                conversation_ref, idempotency_key, payload_json, status,
                intent_kind, created_at, updated_at
            ) SELECT ?, ?, 'PLUGIN_WAKE', ?, ?, ci.route_umo, ?, ?, 'PENDING',
                'PLUGIN_WAKE', ?, ? FROM character_instances ci
            WHERE ci.profile_id = ? AND ci.instance_id = ?
            ON CONFLICT(profile_id, instance_id, idempotency_key)
                WHERE idempotency_key IS NOT NULL DO NOTHING""",
            (
                scope.profile_id,
                scope.instance_id,
                encode_datetime(now),
                "受控文件结果已经就绪，需要重新判断后续表达。",
                key,
                dump_json(payload),
                encode_datetime(now),
                encode_datetime(now),
                scope.profile_id,
                scope.instance_id,
            ),
        )
        wake = conn.execute(
            """SELECT wakeup_id FROM instance_wakeups
            WHERE profile_id = ? AND instance_id = ? AND idempotency_key = ?""",
            (scope.profile_id, scope.instance_id, key),
        ).fetchone()
        if wake is None:
            raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_STATE)
        self.register_recovery_wake(
            conn, scope, work_ref, checkpoint_version, int(wake["wakeup_id"]), now
        )

    @staticmethod
    def _current_baseline(conn: sqlite3.Connection, scope: WorkScope) -> WorkRecoveryBaseline:
        row = conn.execute(
            """SELECT state.state_epoch, state.activity_epoch,
                      rp.file_artifacts_enabled
            FROM instance_core_state state
            JOIN character_instances ci
              ON ci.profile_id = state.profile_id AND ci.instance_id = state.instance_id
            JOIN role_profiles rp ON rp.profile_id = state.profile_id
            WHERE state.profile_id = ? AND state.instance_id = ?""",
            (scope.profile_id, scope.instance_id),
        ).fetchone()
        if row is None:
            raise storage_fail(WorkCheckpointStorageErrorCode.SCOPE_MISMATCH)
        permission_revision = int(bool(row["file_artifacts_enabled"]))
        return WorkRecoveryBaseline(
            int(row["activity_epoch"]),
            int(row["state_epoch"]),
            permission_revision,
            0,
        )

    def register_recovery_wake(
        self,
        conn: sqlite3.Connection,
        scope: WorkScope,
        work_ref: str,
        checkpoint_version: int,
        wakeup_id: int,
        now: datetime,
    ) -> RecoveryWakeRecord:
        row = conn.execute(
            """SELECT checkpoint_json, recovery_envelope_json, status,
                      checkpoint_version
            FROM main_core_work_checkpoints
            WHERE profile_id = ? AND instance_id = ? AND work_ref = ?""",
            (scope.profile_id, scope.instance_id, work_ref),
        ).fetchone()
        if row is None:
            raise storage_fail(WorkCheckpointStorageErrorCode.NOT_FOUND)
        if (
            row["status"] != WorkCheckpointStatus.RECOVERY_READY.value
            or int(row["checkpoint_version"]) != int(checkpoint_version)
            or row["recovery_envelope_json"] is None
        ):
            raise storage_fail(WorkCheckpointStorageErrorCode.VERSION_CONFLICT)
        wake = conn.execute(
            """SELECT profile_id, instance_id, source, payload_json
            FROM instance_wakeups WHERE wakeup_id = ?""",
            (wakeup_id,),
        ).fetchone()
        if wake is None:
            raise storage_fail(WorkCheckpointStorageErrorCode.NOT_FOUND)
        expected_payload = {"work_ref": work_ref, "work_version": int(checkpoint_version)}
        if (
            (wake["profile_id"], wake["instance_id"]) != (scope.profile_id, scope.instance_id)
            or wake["source"] != "PLUGIN_WAKE"
            or load_json(wake["payload_json"]) != expected_payload
        ):
            raise storage_fail(WorkCheckpointStorageErrorCode.SCOPE_MISMATCH)
        existing = self._recovery_wake(conn, scope, work_ref, checkpoint_version)
        if existing is not None:
            if existing.wakeup_id != int(wakeup_id):
                raise storage_fail(WorkCheckpointStorageErrorCode.IDEMPOTENCY_CONFLICT)
            return existing
        conn.execute(
            """INSERT INTO main_core_work_recovery_wakes(
                profile_id, instance_id, work_ref, checkpoint_version, wakeup_id,
                status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'READY', ?, ?)""",
            (
                scope.profile_id,
                scope.instance_id,
                work_ref,
                checkpoint_version,
                wakeup_id,
                encode_datetime(now),
                encode_datetime(now),
            ),
        )
        registered = self._recovery_wake(conn, scope, work_ref, checkpoint_version)
        assert registered is not None
        return registered

    def get_recovery_wake(
        self,
        conn: sqlite3.Connection,
        scope: WorkScope,
        work_ref: str,
        checkpoint_version: int,
    ) -> RecoveryWakeRecord | None:
        return self._recovery_wake(conn, scope, work_ref, checkpoint_version)

    def claim_recovery_wake(
        self,
        conn: sqlite3.Connection,
        scope: WorkScope,
        work_ref: str,
        checkpoint_version: int,
        wakeup_id: int,
        run_id: int,
        now: datetime,
    ) -> RecoveryWakeClaim:
        wake = self._recovery_wake(conn, scope, work_ref, checkpoint_version)
        if wake is None or wake.wakeup_id != int(wakeup_id):
            raise storage_fail(WorkCheckpointStorageErrorCode.NOT_FOUND)
        checkpoint_row = conn.execute(
            """SELECT checkpoint_json, recovery_envelope_json, status,
                      checkpoint_version, expires_at
            FROM main_core_work_checkpoints
            WHERE profile_id = ? AND instance_id = ? AND work_ref = ?""",
            (scope.profile_id, scope.instance_id, work_ref),
        ).fetchone()
        if checkpoint_row is None or checkpoint_row["recovery_envelope_json"] is None:
            raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_PERSISTED_DATA)
        envelope = ready_record(
            checkpoint_row["checkpoint_json"], checkpoint_row["recovery_envelope_json"]
        ).envelope
        if wake.status is RecoveryWakeStatus.CLAIMED:
            if wake.claimed_run_id != int(run_id):
                raise storage_fail(WorkCheckpointStorageErrorCode.VERSION_CONFLICT)
            return RecoveryWakeClaim(wake=wake, envelope=envelope, replayed=True)
        if wake.status is not RecoveryWakeStatus.READY:
            raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_STATE)
        expires_at = decode_datetime(checkpoint_row["expires_at"])
        if expires_at is None or now >= expires_at:
            conn.execute(
                """UPDATE main_core_work_recovery_wakes
                SET status = 'EXPIRED', updated_at = ?, terminal_reason = ?
                WHERE profile_id = ? AND instance_id = ? AND work_ref = ?
                  AND checkpoint_version = ? AND status = 'READY'""",
                (
                    encode_datetime(now),
                    "checkpoint lifetime elapsed before recovery claim",
                    scope.profile_id,
                    scope.instance_id,
                    work_ref,
                    checkpoint_version,
                ),
            )
            raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_STATE)
        run = conn.execute(
            """SELECT status FROM instance_core_runs
            WHERE profile_id = ? AND instance_id = ? AND run_id = ?""",
            (scope.profile_id, scope.instance_id, run_id),
        ).fetchone()
        if run is None:
            raise storage_fail(WorkCheckpointStorageErrorCode.NOT_FOUND)
        updated = conn.execute(
            """UPDATE main_core_work_recovery_wakes
            SET status = 'CLAIMED', claimed_run_id = ?, claimed_at = ?, updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND work_ref = ?
              AND checkpoint_version = ? AND wakeup_id = ? AND status = 'READY'""",
            (
                run_id,
                encode_datetime(now),
                encode_datetime(now),
                scope.profile_id,
                scope.instance_id,
                work_ref,
                checkpoint_version,
                wakeup_id,
            ),
        )
        if updated.rowcount != 1:
            raise storage_fail(WorkCheckpointStorageErrorCode.VERSION_CONFLICT)
        claimed = self._recovery_wake(conn, scope, work_ref, checkpoint_version)
        assert claimed is not None
        return RecoveryWakeClaim(wake=claimed, envelope=envelope)

    @staticmethod
    def _require_waiting_checkpoint(
        conn: sqlite3.Connection, checkpoint: MainCoreWorkCheckpoint
    ) -> None:
        row = conn.execute(
            """SELECT status, checkpoint_version, run_generation, callback_sequence
            FROM main_core_work_checkpoints
            WHERE profile_id = ? AND instance_id = ? AND work_ref = ?""",
            (
                checkpoint.scope.profile_id,
                checkpoint.scope.instance_id,
                checkpoint.work_ref,
            ),
        ).fetchone()
        if row is None:
            raise storage_fail(WorkCheckpointStorageErrorCode.NOT_FOUND)
        expected = (
            checkpoint.status.value,
            checkpoint.checkpoint_version,
            checkpoint.run_generation,
            checkpoint.callback_sequence,
        )
        actual = tuple(
            row[key]
            for key in (
                "status",
                "checkpoint_version",
                "run_generation",
                "callback_sequence",
            )
        )
        if checkpoint.status is not WorkCheckpointStatus.WAITING or actual != expected:
            raise storage_fail(WorkCheckpointStorageErrorCode.VERSION_CONFLICT)

    @staticmethod
    def _recovery_wake(
        conn: sqlite3.Connection,
        scope: WorkScope,
        work_ref: str,
        checkpoint_version: int,
    ) -> RecoveryWakeRecord | None:
        row = conn.execute(
            """SELECT * FROM main_core_work_recovery_wakes
            WHERE profile_id = ? AND instance_id = ? AND work_ref = ?
              AND checkpoint_version = ?""",
            (scope.profile_id, scope.instance_id, work_ref, checkpoint_version),
        ).fetchone()
        return _wake_record(row) if row is not None else None


def _same_binding(
    row: sqlite3.Row,
    checkpoint: MainCoreWorkCheckpoint,
    binding: FileWorkBindingSpec,
    callback_sequence: int,
) -> bool:
    return (
        row["profile_id"],
        row["instance_id"],
        row["work_ref"],
        row["request_ref"],
        row["slot_id"],
        row["checkpoint_version"],
        row["run_generation"],
        row["callback_sequence"],
    ) == (
        checkpoint.scope.profile_id,
        checkpoint.scope.instance_id,
        checkpoint.work_ref,
        binding.request_ref,
        binding.slot_id,
        checkpoint.checkpoint_version,
        checkpoint.run_generation,
        callback_sequence,
    )


def _binding_record(row: sqlite3.Row) -> FileWorkBindingRecord:
    created_at = decode_datetime(row["created_at"])
    updated_at = decode_datetime(row["updated_at"])
    assert created_at is not None and updated_at is not None
    return FileWorkBindingRecord(
        scope=WorkScope(row["profile_id"], row["instance_id"]),
        work_ref=row["work_ref"],
        job_id=row["job_id"],
        request_ref=row["request_ref"],
        slot_id=row["slot_id"],
        checkpoint_version=int(row["checkpoint_version"]),
        run_generation=int(row["run_generation"]),
        callback_sequence=int(row["callback_sequence"]),
        status=FileWorkBindingStatus(row["status"]),
        resource_ref=row["resource_ref"],
        result_kind=row["result_kind"],
        result_summary=row["result_summary"],
        todo_id=row["todo_id"],
        created_at=created_at,
        updated_at=updated_at,
        completed_at=decode_datetime(row["completed_at"]),
    )


def _wake_record(row: sqlite3.Row) -> RecoveryWakeRecord:
    created_at = decode_datetime(row["created_at"])
    updated_at = decode_datetime(row["updated_at"])
    assert created_at is not None and updated_at is not None
    return RecoveryWakeRecord(
        scope=WorkScope(row["profile_id"], row["instance_id"]),
        work_ref=row["work_ref"],
        checkpoint_version=int(row["checkpoint_version"]),
        wakeup_id=int(row["wakeup_id"]),
        status=RecoveryWakeStatus(row["status"]),
        claimed_run_id=(int(row["claimed_run_id"]) if row["claimed_run_id"] else None),
        created_at=created_at,
        updated_at=updated_at,
        claimed_at=decode_datetime(row["claimed_at"]),
        terminal_reason=row["terminal_reason"],
    )


def _aggregate_outcome(
    bindings: tuple[FileWorkBindingRecord, ...],
) -> WorkCallbackOutcome:
    if any(item.status is FileWorkBindingStatus.FAILED for item in bindings):
        return WorkCallbackOutcome.FAILED
    if any(item.status is FileWorkBindingStatus.CANCELLED for item in bindings):
        return WorkCallbackOutcome.CANCELLED
    return WorkCallbackOutcome.SUCCEEDED


def _aggregate_summary(
    bindings: tuple[FileWorkBindingRecord, ...], outcome: WorkCallbackOutcome
) -> str:
    return (
        f"FILE_ARTIFACT_GENERATION {outcome.value.lower()}; "
        f"有 {len(bindings)} 个受控结果需要重新判断。"
    )


__all__ = ["WorkFileSqliteOperations"]
