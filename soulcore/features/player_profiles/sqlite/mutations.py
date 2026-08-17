"""Shared atomic mutation persistence for Main Core and administrator actions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from ....storage.sqlite.codec import encode_datetime
from ..commands import (
    ProfileCommand,
    ProfileCommandReceipt,
    ProfileMutationKind,
    ProfileMutationOutcome,
    ProfileMutationResult,
    apply_profile_command,
    command_fingerprint,
    replay_profile_command,
)
from ..domain import (
    PlayerProfileEntry,
    PlayerProfileScope,
    PlayerProfileSnapshot,
    ProfileAdminEvidence,
    ProfileEvidence,
    ProfileMessageEvidence,
)
from ..errors import ProfileConflictError, ProfileErrorCode
from .codec import decode_entry


def commit_profile_command(
    conn: sqlite3.Connection,
    command: ProfileCommand,
    now: datetime,
) -> ProfileMutationResult:
    fingerprint = command_fingerprint(command)
    receipt = conn.execute(
        """SELECT command_fingerprint, result_json FROM player_profile_command_receipts
        WHERE profile_id = ? AND instance_id = ? AND subject_key = ?
          AND idempotency_key = ?""",
        (*command.scope.persistence_key, command.idempotency_key),
    ).fetchone()
    if receipt is not None:
        if str(receipt["command_fingerprint"]) != fingerprint:
            raise ProfileConflictError(
                ProfileErrorCode.IDEMPOTENCY_CONFLICT,
                "player profile idempotency key was reused for another command",
            )
        stored = _decode_result(
            receipt["result_json"],
            scope=command.scope,
            idempotency_key=command.idempotency_key,
            command_fingerprint=fingerprint,
        )
        return replay_profile_command(
            command,
            ProfileCommandReceipt(command.idempotency_key, fingerprint, stored),
        )
    snapshot = load_snapshot(conn, command.scope)
    result = apply_profile_command(snapshot, command, now=now)
    timestamp = encode_datetime(now)
    if snapshot.version == 0:
        conn.execute(
            """INSERT INTO player_profiles(
                profile_id, instance_id, subject_key, current_version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            (*command.scope.persistence_key, result.profile_version, timestamp, timestamp),
        )
    else:
        cursor = conn.execute(
            """UPDATE player_profiles SET current_version = ?, updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND subject_key = ?
              AND current_version = ?""",
            (
                result.profile_version,
                timestamp,
                *command.scope.persistence_key,
                result.previous_profile_version,
            ),
        )
        if cursor.rowcount != 1:
            raise ProfileConflictError(
                ProfileErrorCode.VERSION_CONFLICT,
                "player profile changed before the result committed",
            )
    insert_revision(conn, result.entry)
    conn.execute(
        """INSERT INTO player_profile_entries(
            profile_id, instance_id, subject_key, entry_id, current_entry_version
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(profile_id, instance_id, subject_key, entry_id) DO UPDATE SET
            current_entry_version = excluded.current_entry_version""",
        (*command.scope.persistence_key, result.entry.entry_id, result.entry.version),
    )
    conn.execute(
        """INSERT INTO player_profile_command_receipts(
            profile_id, instance_id, subject_key, idempotency_key,
            command_fingerprint, result_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            *command.scope.persistence_key,
            command.idempotency_key,
            result.command_fingerprint,
            _encode_result(result),
            timestamp,
        ),
    )
    return result


def load_snapshot(
    conn: sqlite3.Connection,
    scope: PlayerProfileScope,
) -> PlayerProfileSnapshot:
    aggregate = conn.execute(
        """SELECT current_version FROM player_profiles
        WHERE profile_id = ? AND instance_id = ? AND subject_key = ?""",
        scope.persistence_key,
    ).fetchone()
    if aggregate is None:
        return PlayerProfileSnapshot.empty(scope)
    rows = conn.execute(
        """SELECT revision.* FROM player_profile_entries AS current
        JOIN player_profile_entry_revisions AS revision
          ON revision.profile_id = current.profile_id
         AND revision.instance_id = current.instance_id
         AND revision.subject_key = current.subject_key
         AND revision.entry_id = current.entry_id
         AND revision.entry_version = current.current_entry_version
        WHERE current.profile_id = ? AND current.instance_id = ? AND current.subject_key = ?
        ORDER BY current.entry_id""",
        scope.persistence_key,
    ).fetchall()
    return PlayerProfileSnapshot(
        scope=scope,
        version=int(aggregate["current_version"]),
        entries=tuple(decode_entry(dict(row), scope) for row in rows),
    )


def encode_evidence(evidence: Sequence[ProfileEvidence]) -> str:
    payload: list[dict[str, str]] = []
    for item in evidence:
        if isinstance(item, ProfileMessageEvidence):
            payload.append(
                {
                    "kind": "MESSAGE",
                    "message_ref": item.message_ref,
                    "note": item.note,
                }
            )
        elif isinstance(item, ProfileAdminEvidence):
            payload.append(
                {
                    "kind": "ADMIN",
                    "actor": item.actor,
                    "reason": item.reason,
                }
            )
        else:
            raise ValueError("unsupported player profile evidence")
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def insert_revision(conn: sqlite3.Connection, entry: PlayerProfileEntry) -> None:
    conn.execute(
        """INSERT INTO player_profile_entry_revisions(
            profile_id, instance_id, subject_key, entry_id, entry_version,
            layer, category, text, source_type, evidence_json, confidence,
            sensitivity, status, confirmed_at, created_at, updated_at,
            withdrawal_evidence_json, withdrawn_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            *entry.scope.persistence_key,
            entry.entry_id,
            entry.version,
            entry.layer.value,
            entry.category.value,
            entry.text,
            entry.source_type.value,
            encode_evidence(entry.evidence),
            entry.confidence,
            entry.sensitivity.value,
            entry.status.value,
            encode_datetime(entry.confirmed_at),
            encode_datetime(entry.created_at),
            encode_datetime(entry.updated_at),
            encode_evidence(entry.withdrawal_evidence),
            encode_datetime(entry.withdrawn_at),
        ),
    )


def _entry_payload(entry: PlayerProfileEntry) -> dict[str, Any]:
    return {
        "entry_id": entry.entry_id,
        "entry_version": entry.version,
        "layer": entry.layer.value,
        "category": entry.category.value,
        "text": entry.text,
        "source_type": entry.source_type.value,
        "evidence": json.loads(encode_evidence(entry.evidence)),
        "confidence": entry.confidence,
        "sensitivity": entry.sensitivity.value,
        "status": entry.status.value,
        "confirmed_at": encode_datetime(entry.confirmed_at),
        "created_at": encode_datetime(entry.created_at),
        "updated_at": encode_datetime(entry.updated_at),
        "withdrawal_evidence": json.loads(encode_evidence(entry.withdrawal_evidence)),
        "withdrawn_at": encode_datetime(entry.withdrawn_at),
    }


def _encode_result(result: ProfileMutationResult) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "kind": result.kind.value,
            "previous_profile_version": result.previous_profile_version,
            "profile_version": result.profile_version,
            "entry": _entry_payload(result.entry),
            "prior_entry": _entry_payload(result.prior_entry) if result.prior_entry else None,
            "snapshot": {
                "version": result.snapshot.version,
                "entries": [_entry_payload(entry) for entry in result.snapshot.entries],
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _decode_result(
    raw: object,
    *,
    scope: PlayerProfileScope,
    idempotency_key: str,
    command_fingerprint: str,
) -> ProfileMutationResult:
    payload = json.loads(str(raw))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported player profile receipt")
    snapshot_payload = payload.get("snapshot")
    if not isinstance(snapshot_payload, dict) or not isinstance(
        snapshot_payload.get("entries"), list
    ):
        raise ValueError("invalid player profile receipt snapshot")
    prior = payload.get("prior_entry")
    snapshot = PlayerProfileSnapshot(
        scope=scope,
        version=int(snapshot_payload["version"]),
        entries=tuple(_entry_from_payload(item, scope) for item in snapshot_payload["entries"]),
    )
    return ProfileMutationResult(
        kind=ProfileMutationKind(str(payload["kind"])),
        outcome=ProfileMutationOutcome.APPLIED,
        scope=scope,
        idempotency_key=idempotency_key,
        command_fingerprint=command_fingerprint,
        previous_profile_version=int(payload["previous_profile_version"]),
        profile_version=int(payload["profile_version"]),
        entry=_entry_from_payload(payload["entry"], scope),
        prior_entry=_entry_from_payload(prior, scope) if prior is not None else None,
        snapshot=snapshot,
    )


def _entry_from_payload(raw: object, scope: PlayerProfileScope) -> PlayerProfileEntry:
    if not isinstance(raw, dict):
        raise ValueError("invalid player profile receipt entry")
    row = dict(raw)
    row["evidence_json"] = json.dumps(row.pop("evidence"), ensure_ascii=False)
    row["withdrawal_evidence_json"] = json.dumps(row.pop("withdrawal_evidence"), ensure_ascii=False)
    return decode_entry(row, scope)


__all__ = ["commit_profile_command", "encode_evidence", "insert_revision", "load_snapshot"]
