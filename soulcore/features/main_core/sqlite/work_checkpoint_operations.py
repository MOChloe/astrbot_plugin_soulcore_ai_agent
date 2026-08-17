"""Composable SQLite operations for durable Main Core work checkpoints."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime

from ....storage.sqlite.codec import encode_datetime
from ..work_checkpoint import (
    MainCoreWorkCheckpoint,
    WorkCheckpointError,
    WorkCheckpointStatus,
    WorkScope,
    cancel_checkpoint,
    expire_checkpoint,
    release_callback_lease,
    renew_callback_lease,
    supersede_checkpoint,
)
from ..work_checkpoint_repository import (
    ApplyWorkCallbackCommand,
    ClaimWorkCheckpointLeaseCommand,
    FreezeWorkCheckpointCommand,
    RecoveryReadyRecord,
    ReleaseWorkCheckpointLeaseCommand,
    RenewWorkCheckpointLeaseCommand,
    TransitionWorkCheckpointCommand,
    WorkCallbackStorageResult,
    WorkCheckpointEvent,
    WorkCheckpointEventKind,
    WorkCheckpointMutationResult,
    WorkCheckpointTerminalAction,
)
from ..work_checkpoint_storage_errors import (
    WorkCheckpointStorageErrorCode,
    storage_fail,
)
from ..work_recovery import MainCoreWorkRecoveryEnvelope, decide_work_recovery
from .work_checkpoint_codec import (
    decode_callback_result,
    decode_checkpoint,
    decode_event_row,
    decode_mutation_result,
    encode_callback_result,
    encode_checkpoint,
    encode_mutation_result,
    encode_recovery_envelope,
    ready_record,
)


class WorkCheckpointSqliteOperations:
    """Low-level primitives safe to call inside another shared-engine transaction."""

    def freeze_checkpoint(
        self, conn: sqlite3.Connection, command: FreezeWorkCheckpointCommand
    ) -> WorkCheckpointMutationResult:
        checkpoint = command.checkpoint
        self._require_parent(conn, checkpoint.scope)
        fingerprint = _fingerprint("freeze", encode_checkpoint(checkpoint))
        existing = self._get_row(conn, checkpoint.scope, checkpoint.work_ref)
        if existing is not None:
            replay = self._receipt(
                conn,
                checkpoint.scope,
                checkpoint.work_ref,
                command.idempotency_key,
                "FREEZE",
                fingerprint,
            )
            if replay is None:
                raise storage_fail(WorkCheckpointStorageErrorCode.VERSION_CONFLICT)
            return decode_mutation_result(replay)
        self._insert_checkpoint(conn, checkpoint, None, command.now)
        result = WorkCheckpointMutationResult(checkpoint)
        self._insert_event(
            conn,
            checkpoint,
            WorkCheckpointEventKind.CREATED,
            command.idempotency_key,
            fingerprint,
            "",
            command.now,
        )
        self._insert_receipt(
            conn,
            checkpoint.scope,
            checkpoint.work_ref,
            command.idempotency_key,
            "FREEZE",
            fingerprint,
            encode_mutation_result(result),
            command.now,
        )
        return result

    def apply_callback(
        self, conn: sqlite3.Connection, command: ApplyWorkCallbackCommand
    ) -> WorkCallbackStorageResult:
        callback = command.callback
        current, _ = self._required_state(conn, callback.scope, callback.work_ref)
        fingerprint = _callback_request_fingerprint(command)
        replay = self._receipt(
            conn,
            callback.scope,
            callback.work_ref,
            callback.idempotency_key,
            "CALLBACK",
            fingerprint,
        )
        if replay is not None:
            return decode_callback_result(replay)
        decision = decide_work_recovery(current, callback, command.runtime)
        if decision.checkpoint != current:
            self._update_checkpoint(
                conn,
                current,
                decision.checkpoint,
                decision.envelope,
                command.runtime.now,
            )
        result = WorkCallbackStorageResult(decision)
        event_kind = (
            WorkCheckpointEventKind.CALLBACK_ACCEPTED
            if decision.accepted
            else WorkCheckpointEventKind.CALLBACK_REJECTED
        )
        decision_code = "ACCEPTED" if decision.accepted else decision.rejection.value
        self._insert_event(
            conn,
            decision.checkpoint,
            event_kind,
            callback.idempotency_key,
            fingerprint,
            decision_code,
            command.runtime.now,
        )
        self._insert_receipt(
            conn,
            callback.scope,
            callback.work_ref,
            callback.idempotency_key,
            "CALLBACK",
            fingerprint,
            encode_callback_result(result),
            command.runtime.now,
        )
        return result

    def claim_lease(
        self, conn: sqlite3.Connection, command: ClaimWorkCheckpointLeaseCommand
    ) -> WorkCheckpointMutationResult:
        fingerprint = _fingerprint(
            "claim",
            str(command.expected_version),
            command.lease.owner,
            str(command.lease.token),
            encode_datetime(command.lease.expires_at) or "",
        )
        replay = self._mutation_replay(conn, command, "CLAIM_LEASE", fingerprint)
        if replay is not None:
            return replay
        current, _ = self._required_state(conn, command.scope, command.work_ref)
        if current.lease is None or current.lease.expires_at > command.now:
            raise storage_fail(WorkCheckpointStorageErrorCode.LEASE_CONFLICT)
        updated = self._domain_renew(current, command.expected_version, command.lease, command.now)
        return self._finish_mutation(
            conn,
            current,
            updated,
            command.idempotency_key,
            "CLAIM_LEASE",
            fingerprint,
            WorkCheckpointEventKind.LEASE_CLAIMED,
            command.now,
        )

    def renew_lease(
        self, conn: sqlite3.Connection, command: RenewWorkCheckpointLeaseCommand
    ) -> WorkCheckpointMutationResult:
        fingerprint = _fingerprint(
            "renew",
            str(command.expected_version),
            command.expected_lease_owner,
            str(command.expected_lease_token),
            command.lease.owner,
            str(command.lease.token),
            encode_datetime(command.lease.expires_at) or "",
        )
        replay = self._mutation_replay(conn, command, "RENEW_LEASE", fingerprint)
        if replay is not None:
            return replay
        current, _ = self._required_state(conn, command.scope, command.work_ref)
        self._require_active_lease(
            current,
            command.expected_lease_owner,
            command.expected_lease_token,
            command.now,
        )
        updated = self._domain_renew(current, command.expected_version, command.lease, command.now)
        return self._finish_mutation(
            conn,
            current,
            updated,
            command.idempotency_key,
            "RENEW_LEASE",
            fingerprint,
            WorkCheckpointEventKind.LEASE_RENEWED,
            command.now,
        )

    def release_lease(
        self, conn: sqlite3.Connection, command: ReleaseWorkCheckpointLeaseCommand
    ) -> WorkCheckpointMutationResult:
        fingerprint = _fingerprint(
            "release",
            str(command.expected_version),
            command.expected_lease_owner,
            str(command.expected_lease_token),
            encode_datetime(command.now) or "",
        )
        replay = self._mutation_replay(conn, command, "RELEASE_LEASE", fingerprint)
        if replay is not None:
            return replay
        current, _ = self._required_state(conn, command.scope, command.work_ref)
        try:
            updated = release_callback_lease(
                current,
                expected_version=command.expected_version,
                expected_owner=command.expected_lease_owner,
                expected_token=command.expected_lease_token,
                now=command.now,
            )
        except WorkCheckpointError as exc:
            raise _domain_failure(exc) from exc
        return self._finish_mutation(
            conn,
            current,
            updated,
            command.idempotency_key,
            "RELEASE_LEASE",
            fingerprint,
            WorkCheckpointEventKind.LEASE_RELEASED,
            command.now,
        )

    def transition_terminal(
        self, conn: sqlite3.Connection, command: TransitionWorkCheckpointCommand
    ) -> WorkCheckpointMutationResult:
        fingerprint = _fingerprint(
            "terminal",
            command.action.value,
            str(command.expected_version),
            command.reason,
            encode_datetime(command.now) or "",
        )
        replay = self._mutation_replay(conn, command, "TERMINAL", fingerprint)
        if replay is not None:
            return replay
        current, _ = self._required_state(conn, command.scope, command.work_ref)
        try:
            updated = _terminal_transition(current, command)
        except WorkCheckpointError as exc:
            raise _domain_failure(exc) from exc
        event_kind = WorkCheckpointEventKind(updated.status.value)
        return self._finish_mutation(
            conn,
            current,
            updated,
            command.idempotency_key,
            "TERMINAL",
            fingerprint,
            event_kind,
            command.now,
        )

    def get_checkpoint(
        self, conn: sqlite3.Connection, scope: WorkScope, work_ref: str
    ) -> MainCoreWorkCheckpoint | None:
        row = self._get_row(conn, scope, work_ref)
        return self._decode_state(row)[0] if row is not None else None

    def list_events(
        self, conn: sqlite3.Connection, scope: WorkScope, work_ref: str, limit: int
    ) -> tuple[WorkCheckpointEvent, ...]:
        rows = conn.execute(
            """SELECT * FROM main_core_work_checkpoint_events
            WHERE profile_id = ? AND instance_id = ? AND work_ref = ?
            ORDER BY event_sequence LIMIT ?""",
            (*_scope(scope), work_ref, limit),
        ).fetchall()
        return tuple(decode_event_row(dict(row)) for row in rows)

    def _mutation_replay(
        self,
        conn: sqlite3.Connection,
        command: object,
        operation_kind: str,
        fingerprint: str,
    ) -> WorkCheckpointMutationResult | None:
        scope = command.scope  # type: ignore[attr-defined]
        work_ref = command.work_ref  # type: ignore[attr-defined]
        key = command.idempotency_key  # type: ignore[attr-defined]
        replay = self._receipt(conn, scope, work_ref, key, operation_kind, fingerprint)
        return decode_mutation_result(replay) if replay is not None else None

    def _finish_mutation(
        self,
        conn: sqlite3.Connection,
        current: MainCoreWorkCheckpoint,
        updated: MainCoreWorkCheckpoint,
        key: str,
        operation_kind: str,
        fingerprint: str,
        event_kind: WorkCheckpointEventKind,
        now: datetime,
    ) -> WorkCheckpointMutationResult:
        self._update_checkpoint(conn, current, updated, None, now)
        result = WorkCheckpointMutationResult(updated)
        self._insert_event(conn, updated, event_kind, key, fingerprint, "", now)
        self._insert_receipt(
            conn,
            updated.scope,
            updated.work_ref,
            key,
            operation_kind,
            fingerprint,
            encode_mutation_result(result),
            now,
        )
        return result

    @staticmethod
    def _domain_renew(
        current: MainCoreWorkCheckpoint,
        expected_version: int,
        lease: object,
        now: datetime,
    ) -> MainCoreWorkCheckpoint:
        try:
            return renew_callback_lease(
                current,
                expected_version=expected_version,
                lease=lease,  # type: ignore[arg-type]
                now=now,
            )
        except WorkCheckpointError as exc:
            raise _domain_failure(exc) from exc

    @staticmethod
    def _require_active_lease(
        current: MainCoreWorkCheckpoint, owner: str, token: int, now: datetime
    ) -> None:
        lease = current.lease
        if lease is None or lease.owner != owner or lease.token != token or lease.expires_at <= now:
            raise storage_fail(WorkCheckpointStorageErrorCode.LEASE_CONFLICT)

    def _required_state(
        self, conn: sqlite3.Connection, scope: WorkScope, work_ref: str
    ) -> tuple[MainCoreWorkCheckpoint, RecoveryReadyRecord | None]:
        row = self._get_row(conn, scope, work_ref)
        if row is None:
            raise storage_fail(WorkCheckpointStorageErrorCode.NOT_FOUND)
        return self._decode_state(row)

    @staticmethod
    def _get_row(conn: sqlite3.Connection, scope: WorkScope, work_ref: str) -> sqlite3.Row | None:
        return conn.execute(
            """SELECT * FROM main_core_work_checkpoints
            WHERE profile_id = ? AND instance_id = ? AND work_ref = ?""",
            (*_scope(scope), work_ref),
        ).fetchone()

    @staticmethod
    def _decode_state(
        row: sqlite3.Row,
    ) -> tuple[MainCoreWorkCheckpoint, RecoveryReadyRecord | None]:
        checkpoint = decode_checkpoint(row["checkpoint_json"])
        _validate_row_mirror(row, checkpoint)
        if checkpoint.status is WorkCheckpointStatus.RECOVERY_READY:
            record = ready_record(row["checkpoint_json"], row["recovery_envelope_json"])
            return checkpoint, record
        if row["recovery_envelope_json"] is not None:
            raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_PERSISTED_DATA)
        return checkpoint, None

    @staticmethod
    def _insert_checkpoint(
        conn: sqlite3.Connection,
        checkpoint: MainCoreWorkCheckpoint,
        recovery_envelope: MainCoreWorkRecoveryEnvelope | None,
        now: datetime,
    ) -> None:
        conn.execute(
            """INSERT INTO main_core_work_checkpoints(
                profile_id, instance_id, work_ref, checkpoint_json,
                recovery_envelope_json, status, checkpoint_version, run_generation,
                callback_sequence, lease_owner, lease_token, lease_expires_at,
                created_at, expires_at, updated_at, terminal_reason,
                last_idempotency_key, last_callback_fingerprint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            _checkpoint_columns(checkpoint, recovery_envelope, now),
        )

    @staticmethod
    def _update_checkpoint(
        conn: sqlite3.Connection,
        before: MainCoreWorkCheckpoint,
        after: MainCoreWorkCheckpoint,
        recovery_envelope: MainCoreWorkRecoveryEnvelope | None,
        now: datetime,
    ) -> None:
        values = _checkpoint_columns(after, recovery_envelope, now)
        cursor = conn.execute(
            """UPDATE main_core_work_checkpoints SET
                checkpoint_json = ?, recovery_envelope_json = ?, status = ?,
                checkpoint_version = ?, run_generation = ?, callback_sequence = ?,
                lease_owner = ?, lease_token = ?, lease_expires_at = ?, created_at = ?,
                expires_at = ?, updated_at = ?, terminal_reason = ?,
                last_idempotency_key = ?, last_callback_fingerprint = ?
            WHERE profile_id = ? AND instance_id = ? AND work_ref = ?
                AND checkpoint_version = ? AND run_generation = ?
                AND callback_sequence = ? AND status = ?""",
            (
                *values[3:],
                *_scope(before.scope),
                before.work_ref,
                before.checkpoint_version,
                before.run_generation,
                before.callback_sequence,
                before.status.value,
            ),
        )
        if cursor.rowcount != 1:
            raise storage_fail(WorkCheckpointStorageErrorCode.VERSION_CONFLICT)

    @staticmethod
    def _insert_event(
        conn: sqlite3.Connection,
        checkpoint: MainCoreWorkCheckpoint,
        kind: WorkCheckpointEventKind,
        key: str,
        fingerprint: str,
        decision_code: str,
        now: datetime,
    ) -> None:
        row = conn.execute(
            """SELECT COALESCE(MAX(event_sequence), 0) + 1
            FROM main_core_work_checkpoint_events
            WHERE profile_id = ? AND instance_id = ? AND work_ref = ?""",
            (*_scope(checkpoint.scope), checkpoint.work_ref),
        ).fetchone()
        conn.execute(
            """INSERT INTO main_core_work_checkpoint_events(
                profile_id, instance_id, work_ref, event_sequence, event_kind,
                checkpoint_version, run_generation, callback_sequence,
                checkpoint_status, idempotency_key, request_fingerprint,
                decision_code, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                *_scope(checkpoint.scope),
                checkpoint.work_ref,
                int(row[0]),
                kind.value,
                checkpoint.checkpoint_version,
                checkpoint.run_generation,
                checkpoint.callback_sequence,
                checkpoint.status.value,
                key,
                fingerprint,
                decision_code,
                encode_datetime(now),
            ),
        )

    @staticmethod
    def _receipt(
        conn: sqlite3.Connection,
        scope: WorkScope,
        work_ref: str,
        key: str,
        operation_kind: str,
        fingerprint: str,
    ) -> str | None:
        row = conn.execute(
            """SELECT operation_kind, request_fingerprint, result_json
            FROM main_core_work_checkpoint_receipts
            WHERE profile_id = ? AND instance_id = ? AND work_ref = ?
                AND idempotency_key = ?""",
            (*_scope(scope), work_ref, key),
        ).fetchone()
        if row is None:
            return None
        if row["operation_kind"] != operation_kind or row["request_fingerprint"] != fingerprint:
            raise storage_fail(WorkCheckpointStorageErrorCode.IDEMPOTENCY_CONFLICT)
        return str(row["result_json"])

    @staticmethod
    def _insert_receipt(
        conn: sqlite3.Connection,
        scope: WorkScope,
        work_ref: str,
        key: str,
        operation_kind: str,
        fingerprint: str,
        result_json: str,
        now: datetime,
    ) -> None:
        conn.execute(
            """INSERT INTO main_core_work_checkpoint_receipts(
                profile_id, instance_id, work_ref, idempotency_key,
                operation_kind, request_fingerprint, result_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                *_scope(scope),
                work_ref,
                key,
                operation_kind,
                fingerprint,
                result_json,
                encode_datetime(now),
            ),
        )

    @staticmethod
    def _require_parent(conn: sqlite3.Connection, scope: WorkScope) -> None:
        row = conn.execute(
            """SELECT 1 FROM character_instances
            WHERE profile_id = ? AND instance_id = ?""",
            _scope(scope),
        ).fetchone()
        if row is None:
            raise storage_fail(WorkCheckpointStorageErrorCode.SCOPE_MISMATCH)


def _checkpoint_columns(
    checkpoint: MainCoreWorkCheckpoint,
    recovery_envelope: MainCoreWorkRecoveryEnvelope | None,
    now: datetime,
) -> tuple[object, ...]:
    lease = checkpoint.lease
    envelope_json = (
        encode_recovery_envelope(recovery_envelope) if recovery_envelope is not None else None
    )
    return (
        *_scope(checkpoint.scope),
        checkpoint.work_ref,
        encode_checkpoint(checkpoint),
        envelope_json,
        checkpoint.status.value,
        checkpoint.checkpoint_version,
        checkpoint.run_generation,
        checkpoint.callback_sequence,
        lease.owner if lease else None,
        lease.token if lease else None,
        encode_datetime(lease.expires_at) if lease else None,
        encode_datetime(checkpoint.created_at),
        encode_datetime(checkpoint.expires_at),
        encode_datetime(now),
        checkpoint.terminal_reason,
        checkpoint.last_idempotency_key,
        checkpoint.last_callback_fingerprint,
    )


def _validate_row_mirror(row: sqlite3.Row, checkpoint: MainCoreWorkCheckpoint) -> None:
    lease = checkpoint.lease
    expected = (
        checkpoint.scope.profile_id,
        checkpoint.scope.instance_id,
        checkpoint.work_ref,
        checkpoint.status.value,
        checkpoint.checkpoint_version,
        checkpoint.run_generation,
        checkpoint.callback_sequence,
        lease.owner if lease else None,
        lease.token if lease else None,
        encode_datetime(lease.expires_at) if lease else None,
        encode_datetime(checkpoint.created_at),
        encode_datetime(checkpoint.expires_at),
        checkpoint.terminal_reason,
        checkpoint.last_idempotency_key,
        checkpoint.last_callback_fingerprint,
    )
    actual = tuple(
        row[name]
        for name in (
            "profile_id",
            "instance_id",
            "work_ref",
            "status",
            "checkpoint_version",
            "run_generation",
            "callback_sequence",
            "lease_owner",
            "lease_token",
            "lease_expires_at",
            "created_at",
            "expires_at",
            "terminal_reason",
            "last_idempotency_key",
            "last_callback_fingerprint",
        )
    )
    if actual != expected:
        raise storage_fail(WorkCheckpointStorageErrorCode.INVALID_PERSISTED_DATA)


def _callback_request_fingerprint(command: ApplyWorkCallbackCommand) -> str:
    callback = command.callback
    runtime = command.runtime
    payload = {
        "scope": [callback.scope.profile_id, callback.scope.instance_id],
        "work_ref": callback.work_ref,
        "checkpoint_version": callback.checkpoint_version,
        "run_generation": callback.run_generation,
        "callback_sequence": callback.callback_sequence,
        "lease_owner": callback.lease_owner,
        "lease_token": callback.lease_token,
        "outcome": callback.outcome.value,
        "result_summary": callback.result_summary,
        "results": [
            [item.slot_id, item.resource_ref, item.result_kind, item.source_callback_sequence]
            for item in callback.results
        ],
        "runtime_scope": [runtime.scope.profile_id, runtime.scope.instance_id],
        "now": encode_datetime(runtime.now),
        "baseline": [
            runtime.baseline.activity_generation,
            runtime.baseline.role_state_revision,
            runtime.baseline.permission_revision,
            runtime.baseline.budget_revision,
        ],
        "controlled_refs": sorted(runtime.controlled_resource_refs),
        "process_restarted": runtime.process_restarted,
    }
    return _fingerprint(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _terminal_transition(
    current: MainCoreWorkCheckpoint, command: TransitionWorkCheckpointCommand
) -> MainCoreWorkCheckpoint:
    if command.action is WorkCheckpointTerminalAction.CANCEL:
        return cancel_checkpoint(
            current, expected_version=command.expected_version, reason=command.reason
        )
    if command.action is WorkCheckpointTerminalAction.EXPIRE:
        return expire_checkpoint(
            current,
            expected_version=command.expected_version,
            now=command.now,
            reason=command.reason,
        )
    return supersede_checkpoint(
        current, expected_version=command.expected_version, reason=command.reason
    )


def _domain_failure(exc: WorkCheckpointError) -> Exception:
    text = str(exc)
    if "stale checkpoint version" in text:
        return storage_fail(WorkCheckpointStorageErrorCode.VERSION_CONFLICT)
    if "lease" in text:
        return storage_fail(WorkCheckpointStorageErrorCode.LEASE_CONFLICT)
    return storage_fail(WorkCheckpointStorageErrorCode.INVALID_STATE)


def _scope(scope: WorkScope) -> tuple[str, str]:
    return scope.profile_id, scope.instance_id


def _fingerprint(*parts: str) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = ["WorkCheckpointSqliteOperations"]
