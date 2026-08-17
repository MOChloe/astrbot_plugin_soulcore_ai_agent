"""Pure command reducer for atomic and idempotent profile persistence adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import TypeAlias

from .domain import (
    PlayerProfileEntry,
    PlayerProfileScope,
    PlayerProfileSnapshot,
    ProfileAdminEvidence,
    ProfileEntryDraft,
    ProfileEntryStatus,
    ProfileEvidence,
    ProfileMessageEvidence,
)
from .errors import ProfileConflictError, ProfileErrorCode, ProfileValidationError
from .validation import validate_aware_datetime, validate_identifier


@dataclass(frozen=True, slots=True)
class AddProfileEntry:
    idempotency_key: str
    scope: PlayerProfileScope
    expected_profile_version: int
    entry_id: str
    draft: ProfileEntryDraft

    def __post_init__(self) -> None:
        _validate_scope_and_draft(self.scope, self.draft)
        object.__setattr__(
            self,
            "idempotency_key",
            validate_identifier("idempotency_key", self.idempotency_key),
        )
        object.__setattr__(self, "entry_id", validate_identifier("entry_id", self.entry_id))
        _validate_expected_version(
            "expected_profile_version", self.expected_profile_version, zero=True
        )


@dataclass(frozen=True, slots=True)
class ReviseProfileEntry:
    idempotency_key: str
    scope: PlayerProfileScope
    expected_profile_version: int
    entry_id: str
    expected_entry_version: int
    draft: ProfileEntryDraft

    def __post_init__(self) -> None:
        _validate_scope_and_draft(self.scope, self.draft)
        object.__setattr__(
            self,
            "idempotency_key",
            validate_identifier("idempotency_key", self.idempotency_key),
        )
        object.__setattr__(self, "entry_id", validate_identifier("entry_id", self.entry_id))
        _validate_expected_version(
            "expected_profile_version", self.expected_profile_version, zero=True
        )
        _validate_expected_version("expected_entry_version", self.expected_entry_version)


@dataclass(frozen=True, slots=True)
class WithdrawProfileEntry:
    idempotency_key: str
    scope: PlayerProfileScope
    expected_profile_version: int
    entry_id: str
    expected_entry_version: int
    evidence: tuple[ProfileEvidence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, PlayerProfileScope):
            raise ProfileValidationError(
                ProfileErrorCode.INVALID_VALUE,
                "profile command requires a stable player profile scope",
            )
        object.__setattr__(
            self,
            "idempotency_key",
            validate_identifier("idempotency_key", self.idempotency_key),
        )
        object.__setattr__(self, "entry_id", validate_identifier("entry_id", self.entry_id))
        _validate_expected_version(
            "expected_profile_version", self.expected_profile_version, zero=True
        )
        _validate_expected_version("expected_entry_version", self.expected_entry_version)
        if not self.evidence:
            raise ProfileValidationError(
                ProfileErrorCode.INVALID_VALUE,
                "withdrawing a profile entry requires controlled message evidence",
            )
        if not isinstance(self.evidence, tuple) or not all(
            isinstance(item, (ProfileMessageEvidence, ProfileAdminEvidence))
            for item in self.evidence
        ):
            raise ProfileValidationError(
                ProfileErrorCode.INVALID_VALUE,
                "withdrawal evidence must use controlled profile evidence",
            )


@dataclass(frozen=True, slots=True)
class RestoreProfileEntry:
    idempotency_key: str
    scope: PlayerProfileScope
    expected_profile_version: int
    entry_id: str
    expected_entry_version: int
    evidence: tuple[ProfileEvidence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, PlayerProfileScope):
            raise ProfileValidationError(
                ProfileErrorCode.INVALID_VALUE,
                "profile command requires a stable player profile scope",
            )
        object.__setattr__(
            self,
            "idempotency_key",
            validate_identifier("idempotency_key", self.idempotency_key),
        )
        object.__setattr__(self, "entry_id", validate_identifier("entry_id", self.entry_id))
        _validate_expected_version(
            "expected_profile_version", self.expected_profile_version, zero=True
        )
        _validate_expected_version("expected_entry_version", self.expected_entry_version)
        if (
            not self.evidence
            or not isinstance(self.evidence, tuple)
            or not all(
                isinstance(item, (ProfileMessageEvidence, ProfileAdminEvidence))
                for item in self.evidence
            )
        ):
            raise ProfileValidationError(
                ProfileErrorCode.INVALID_VALUE,
                "restoring a profile entry requires controlled profile evidence",
            )


ProfileCommand: TypeAlias = (
    AddProfileEntry | ReviseProfileEntry | WithdrawProfileEntry | RestoreProfileEntry
)


class ProfileMutationKind(StrEnum):
    ADD = "ADD"
    REVISE = "REVISE"
    WITHDRAW = "WITHDRAW"
    RESTORE = "RESTORE"


class ProfileMutationOutcome(StrEnum):
    APPLIED = "APPLIED"
    IDEMPOTENT_REPLAY = "IDEMPOTENT_REPLAY"


@dataclass(frozen=True, slots=True)
class ProfileMutationResult:
    kind: ProfileMutationKind
    outcome: ProfileMutationOutcome
    scope: PlayerProfileScope
    idempotency_key: str
    command_fingerprint: str
    previous_profile_version: int
    profile_version: int
    entry: PlayerProfileEntry
    prior_entry: PlayerProfileEntry | None
    snapshot: PlayerProfileSnapshot


@dataclass(frozen=True, slots=True)
class ProfileCommandReceipt:
    idempotency_key: str
    command_fingerprint: str
    result: ProfileMutationResult


def _validate_scope_and_draft(
    scope: PlayerProfileScope,
    draft: ProfileEntryDraft,
) -> None:
    if not isinstance(scope, PlayerProfileScope) or not isinstance(draft, ProfileEntryDraft):
        raise ProfileValidationError(
            ProfileErrorCode.INVALID_VALUE,
            "profile command requires a stable scope and validated entry draft",
        )


def _validate_expected_version(name: str, value: int, *, zero: bool = False) -> None:
    minimum = 0 if zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ProfileValidationError(
            ProfileErrorCode.INVALID_VALUE,
            f"{name} must be an integer greater than or equal to {minimum}",
        )


def _evidence_payload(evidence: tuple[ProfileEvidence, ...]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for item in evidence:
        row: dict[str, object] = {"scope": item.scope.persistence_key}
        if isinstance(item, ProfileMessageEvidence):
            row.update(
                {
                    "kind": "MESSAGE",
                    "message_ref": item.message_ref,
                    "note": item.note,
                }
            )
        elif isinstance(item, ProfileAdminEvidence):
            row.update({"kind": "ADMIN", "actor": item.actor, "reason": item.reason})
        else:
            raise ProfileValidationError(
                ProfileErrorCode.INVALID_VALUE,
                "unsupported player profile evidence",
            )
        payload.append(row)
    return payload


def _draft_payload(draft: ProfileEntryDraft) -> dict[str, object]:
    return {
        "layer": draft.layer.value,
        "category": draft.category.value,
        "text": draft.text,
        "source_type": draft.source_type.value,
        "evidence": _evidence_payload(draft.evidence),
        "confirmed_at": draft.confirmed_at.isoformat(),
        "confidence": draft.confidence,
        "sensitivity": draft.sensitivity.value,
        "input_modality": draft.input_modality.value,
    }


def command_fingerprint(command: ProfileCommand) -> str:
    payload: dict[str, object] = {
        "command": type(command).__name__,
        "scope": command.scope.persistence_key,
        "expected_profile_version": command.expected_profile_version,
        "entry_id": command.entry_id,
    }
    if isinstance(command, AddProfileEntry):
        payload["draft"] = _draft_payload(command.draft)
    elif isinstance(command, ReviseProfileEntry):
        payload["expected_entry_version"] = command.expected_entry_version
        payload["draft"] = _draft_payload(command.draft)
    else:
        payload["expected_entry_version"] = command.expected_entry_version
        payload["evidence"] = _evidence_payload(command.evidence)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_command_scope(snapshot: PlayerProfileSnapshot, command: ProfileCommand) -> None:
    if snapshot.scope != command.scope:
        raise ProfileConflictError(
            ProfileErrorCode.SCOPE_MISMATCH,
            "profile command targets a different profile subject scope",
        )
    if isinstance(command, (AddProfileEntry, ReviseProfileEntry)):
        evidence = command.draft.evidence
    else:
        evidence = command.evidence
    if any(item.scope != command.scope for item in evidence):
        raise ProfileConflictError(
            ProfileErrorCode.SCOPE_MISMATCH,
            "profile command evidence targets a different profile subject scope",
        )
    if snapshot.version != command.expected_profile_version:
        raise ProfileConflictError(
            ProfileErrorCode.VERSION_CONFLICT,
            "player profile changed after the command snapshot was created",
        )


def _replace_current_entry(
    snapshot: PlayerProfileSnapshot,
    replacement: PlayerProfileEntry,
) -> tuple[PlayerProfileEntry, ...]:
    return tuple(
        replacement if entry.entry_id == replacement.entry_id else entry
        for entry in snapshot.entries
    )


def apply_profile_command(
    snapshot: PlayerProfileSnapshot,
    command: ProfileCommand,
    *,
    now: datetime,
) -> ProfileMutationResult:
    """Apply one command; persistence commits its result and receipt atomically."""
    validate_aware_datetime("now", now)
    _validate_command_scope(snapshot, command)
    fingerprint = command_fingerprint(command)
    existing = snapshot.find_entry(command.entry_id)
    if isinstance(command, AddProfileEntry):
        if existing is not None:
            raise ProfileConflictError(
                ProfileErrorCode.ENTRY_ALREADY_EXISTS,
                "the player profile entry already exists",
            )
        current = PlayerProfileEntry(
            entry_id=command.entry_id,
            scope=command.scope,
            version=1,
            layer=command.draft.layer,
            category=command.draft.category,
            text=command.draft.text,
            source_type=command.draft.source_type,
            evidence=command.draft.evidence,
            confidence=command.draft.confidence,
            sensitivity=command.draft.sensitivity,
            status=ProfileEntryStatus.ACTIVE,
            confirmed_at=command.draft.confirmed_at,
            created_at=now,
            updated_at=now,
        )
        entries = (*snapshot.entries, current)
        prior = None
        kind = ProfileMutationKind.ADD
    else:
        if existing is None:
            raise ProfileConflictError(
                ProfileErrorCode.ENTRY_NOT_FOUND,
                "the player profile entry does not exist",
            )
        if existing.version != command.expected_entry_version:
            raise ProfileConflictError(
                ProfileErrorCode.VERSION_CONFLICT,
                "player profile entry changed after the command snapshot was created",
            )
        restoring = isinstance(command, RestoreProfileEntry)
        if restoring and existing.status is not ProfileEntryStatus.WITHDRAWN:
            raise ProfileConflictError(
                ProfileErrorCode.INVALID_VALUE,
                "only a withdrawn player profile entry can be restored",
            )
        if not restoring and existing.status is not ProfileEntryStatus.ACTIVE:
            raise ProfileConflictError(
                ProfileErrorCode.ENTRY_WITHDRAWN,
                "the player profile entry is not active",
            )
        prior = existing if restoring else replace(existing, status=ProfileEntryStatus.SUPERSEDED)
        if isinstance(command, ReviseProfileEntry):
            current = PlayerProfileEntry(
                entry_id=existing.entry_id,
                scope=existing.scope,
                version=existing.version + 1,
                layer=command.draft.layer,
                category=command.draft.category,
                text=command.draft.text,
                source_type=command.draft.source_type,
                evidence=command.draft.evidence,
                confidence=command.draft.confidence,
                sensitivity=command.draft.sensitivity,
                status=ProfileEntryStatus.ACTIVE,
                confirmed_at=command.draft.confirmed_at,
                created_at=existing.created_at,
                updated_at=now,
            )
            kind = ProfileMutationKind.REVISE
        elif isinstance(command, WithdrawProfileEntry):
            current = replace(
                existing,
                version=existing.version + 1,
                status=ProfileEntryStatus.WITHDRAWN,
                updated_at=now,
                withdrawal_evidence=command.evidence,
                withdrawn_at=now,
            )
            kind = ProfileMutationKind.WITHDRAW
        else:
            current = replace(
                existing,
                version=existing.version + 1,
                status=ProfileEntryStatus.ACTIVE,
                evidence=(*existing.evidence, *command.evidence),
                updated_at=now,
                withdrawal_evidence=(),
                withdrawn_at=None,
            )
            kind = ProfileMutationKind.RESTORE
        entries = _replace_current_entry(snapshot, current)
    updated = PlayerProfileSnapshot(
        scope=snapshot.scope,
        version=snapshot.version + 1,
        entries=entries,
    )
    return ProfileMutationResult(
        kind=kind,
        outcome=ProfileMutationOutcome.APPLIED,
        scope=command.scope,
        idempotency_key=command.idempotency_key,
        command_fingerprint=fingerprint,
        previous_profile_version=snapshot.version,
        profile_version=updated.version,
        entry=current,
        prior_entry=prior,
        snapshot=updated,
    )


def replay_profile_command(
    command: ProfileCommand,
    receipt: ProfileCommandReceipt,
) -> ProfileMutationResult:
    """Replay an already committed command without producing another mutation."""

    fingerprint = command_fingerprint(command)
    if (
        receipt.idempotency_key != command.idempotency_key
        or receipt.result.idempotency_key != command.idempotency_key
        or receipt.command_fingerprint != fingerprint
        or receipt.result.command_fingerprint != fingerprint
        or receipt.result.scope != command.scope
    ):
        raise ProfileConflictError(
            ProfileErrorCode.IDEMPOTENCY_CONFLICT,
            "idempotency key was already used for a different profile mutation",
        )
    return replace(receipt.result, outcome=ProfileMutationOutcome.IDEMPOTENT_REPLAY)


__all__ = [
    "AddProfileEntry",
    "ProfileCommand",
    "ProfileCommandReceipt",
    "ProfileMutationKind",
    "ProfileMutationOutcome",
    "ProfileMutationResult",
    "ReviseProfileEntry",
    "RestoreProfileEntry",
    "WithdrawProfileEntry",
    "apply_profile_command",
    "command_fingerprint",
    "replay_profile_command",
]
