"""Atomic player-profile participant in the final Main Core SQLite transaction."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from ....features.player_profiles import (
    AddProfileEntry,
    PlayerProfileScope,
    ProfileConflictError,
    ProfileErrorCode,
    ProfileMessageEvidence,
    ProfileMutationResult,
    ReviseProfileEntry,
    WithdrawProfileEntry,
)
from ....features.player_profiles.service import decode_persisted_profile_evidence
from ....storage.sqlite.codec import decode_datetime
from ....storage.sqlite.player_profile_transactions import commit_profile_command

PRIVATE_TARGET_REF = "private-counterpart"


class PlayerProfileTransactionWriter:
    def __init__(self, context: Any) -> None:
        self.context = context

    def apply(self, conn: sqlite3.Connection) -> None:
        mutations = list(self.context.player_profile_mutations)
        if len(mutations) > 12:
            raise ValueError("too many player profile mutations in one Main Core result")
        now = decode_datetime(self.context.now)
        if now is None:
            raise ValueError("invalid Main Core commit timestamp")
        for mutation in mutations:
            command = self._validate_envelope(conn, mutation)
            self._commit_command(conn, command, now)

    def _validate_envelope(self, conn: sqlite3.Connection, mutation: Mapping[str, Any]) -> Any:
        command = mutation.get("command")
        if not isinstance(command, (AddProfileEntry, ReviseProfileEntry, WithdrawProfileEntry)):
            raise ValueError("invalid player profile command envelope")
        member_ref = str(mutation.get("member_ref") or "")
        sender_id = str(mutation.get("sender_id") or "")
        evidence_message_id = int(mutation.get("evidence_message_id") or 0)
        evidence_quote = _normalize_text(mutation.get("evidence_quote"))
        reuse_existing_evidence = bool(mutation.get("reuse_existing_evidence", False))
        instance_scope = str(self.context.instance.scope)
        self._validate_target(command.scope, instance_scope, member_ref, sender_id)
        if reuse_existing_evidence:
            self._validate_reused_existing_evidence(conn, command)
        else:
            self._validate_evidence(
                conn,
                command,
                sender_id=sender_id,
                evidence_message_id=evidence_message_id,
                evidence_quote=evidence_quote,
            )
        return command

    def _validate_target(
        self,
        scope: PlayerProfileScope,
        instance_scope: str,
        member_ref: str,
        sender_id: str,
    ) -> None:
        if (scope.profile_id, scope.instance_id) != (
            self.context.profile_id,
            self.context.instance_id,
        ):
            raise ProfileConflictError(
                ProfileErrorCode.SCOPE_MISMATCH,
                "player profile command escaped its Main Core instance",
            )
        if instance_scope == "group":
            expected = _member_ref(
                scope.profile_id,
                scope.instance_id,
                self.context.run_id,
                sender_id,
            )
            if not member_ref or member_ref != expected or scope.subject_key != sender_id:
                raise ProfileConflictError(
                    ProfileErrorCode.SCOPE_MISMATCH,
                    "group member reference is stale or belongs to another run",
                )
        elif member_ref or scope.subject_key != PRIVATE_TARGET_REF:
            raise ProfileConflictError(
                ProfileErrorCode.SCOPE_MISMATCH,
                "private player profile target must remain implicit",
            )

    @staticmethod
    def _validate_evidence(
        conn: sqlite3.Connection,
        command: Any,
        *,
        sender_id: str,
        evidence_message_id: int,
        evidence_quote: str,
    ) -> None:
        scope = command.scope
        row = conn.execute(
            """SELECT sender_id, direction, role, plain_text FROM instance_messages
            WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
            (scope.profile_id, scope.instance_id, evidence_message_id),
        ).fetchone()
        if (
            row is None
            or str(row["direction"]) != "INBOUND"
            or str(row["role"]) != "user"
            or not sender_id
            or str(row["sender_id"]) != sender_id
            or len(evidence_quote) < 2
            or evidence_quote not in _normalize_text(row["plain_text"])
        ):
            raise ProfileConflictError(
                ProfileErrorCode.SCOPE_MISMATCH,
                "player profile evidence is outside the run target scope",
            )
        evidence = (
            command.evidence
            if isinstance(command, WithdrawProfileEntry)
            else command.draft.evidence
        )
        expected_evidence_ref = f"ledger-message:{evidence_message_id}"
        if (
            len(evidence) != 1
            or not isinstance(evidence[0], ProfileMessageEvidence)
            or evidence[0].message_ref != expected_evidence_ref
        ):
            raise ProfileConflictError(
                ProfileErrorCode.SCOPE_MISMATCH,
                "player profile command evidence changed after command validation",
            )

    @staticmethod
    def _validate_reused_existing_evidence(conn: sqlite3.Connection, command: Any) -> None:
        if not isinstance(command, (ReviseProfileEntry, WithdrawProfileEntry)):
            raise ProfileConflictError(
                ProfileErrorCode.SCOPE_MISMATCH,
                "only existing impression changes may reuse controlled evidence",
            )
        scope = command.scope
        row = conn.execute(
            """SELECT revision.entry_version, revision.status, revision.evidence_json
            FROM player_profile_entries AS current
            JOIN player_profile_entry_revisions AS revision
              ON revision.profile_id = current.profile_id
             AND revision.instance_id = current.instance_id
             AND revision.subject_key = current.subject_key
             AND revision.entry_id = current.entry_id
             AND revision.entry_version = current.current_version
            WHERE current.profile_id = ? AND current.instance_id = ?
              AND current.subject_key = ? AND current.entry_id = ?""",
            (
                scope.profile_id,
                scope.instance_id,
                scope.subject_key,
                command.entry_id,
            ),
        ).fetchone()
        submitted = (
            command.evidence
            if isinstance(command, WithdrawProfileEntry)
            else command.draft.evidence
        )
        if (
            row is None
            or str(row["status"]) != "ACTIVE"
            or int(row["entry_version"]) != int(command.expected_entry_version)
            or not submitted
            or tuple(submitted) != decode_persisted_profile_evidence(row["evidence_json"], scope)
        ):
            raise ProfileConflictError(
                ProfileErrorCode.SCOPE_MISMATCH,
                "reused profile evidence no longer matches the active impression",
            )

    def _commit_command(
        self, conn: sqlite3.Connection, command: Any, now: datetime
    ) -> ProfileMutationResult:
        return commit_profile_command(conn, command, now)


def _member_ref(profile_id: str, instance_id: str, run_id: int, sender_id: str) -> str:
    source = f"{profile_id}\x1f{instance_id}\x1f{run_id}\x1f{sender_id}"
    return f"member_ref:v1:{hashlib.sha256(source.encode()).hexdigest()[:20]}"


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


__all__ = ["PlayerProfileTransactionWriter"]
