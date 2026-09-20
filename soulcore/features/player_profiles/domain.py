"Pure domain types for versioned, instance-isolated player profiles."

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TypeAlias


class ProfileErrorCode(StrEnum):
    INVALID_VALUE = "INVALID_VALUE"
    SCOPE_NOT_FOUND = "SCOPE_NOT_FOUND"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    ENTRY_NOT_FOUND = "ENTRY_NOT_FOUND"
    ENTRY_ALREADY_EXISTS = "ENTRY_ALREADY_EXISTS"
    ENTRY_WITHDRAWN = "ENTRY_WITHDRAWN"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    HIGH_RISK_SECRET = "HIGH_RISK_SECRET"
    VISUAL_INPUT_FORBIDDEN = "VISUAL_INPUT_FORBIDDEN"


class PlayerProfileError(ValueError):
    """Base error whose public text is safe to return to Main Core."""

    def __init__(self, code: ProfileErrorCode, public_message: str) -> None:
        self.code = code
        self.public_message = public_message
        super().__init__(f"{code.value}: {public_message}")


class ProfileValidationError(PlayerProfileError):
    pass


class ProfileConflictError(PlayerProfileError):
    pass


MAX_IDENTIFIER_CHARS = 200
MAX_ENTRY_TEXT_CHARS = 1000
MAX_EVIDENCE_NOTE_CHARS = 160

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)(?:password|passwd|pwd|passcode|api[\s_-]*key|access[\s_-]*token|"
    r"refresh[\s_-]*token|client[\s_-]*secret|密码|口令|密钥|令牌)"
    r"\s*(?:is|是|为|[:=：])\s*\S{4,}"
)
_KNOWN_TOKEN = re.compile(
    r"(?i)(?:sk-[a-z0-9_-]{16,}|gh[pousr]_[a-z0-9]{20,}|"
    r"eyJ[a-z0-9_-]{8,}\.[a-z0-9_-]{8,}\.[a-z0-9_-]{8,})"
)
_CHINESE_ID = re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)")
_FINANCIAL_ACCOUNT = re.compile(r"(?<!\d)(?:\d[ -]?){15,18}\d(?!\d)")
_EXACT_ADDRESS = re.compile(
    r"(?:住址|家庭地址|详细地址|精确地址|home address|exact address)"
    r"\s*(?:是|为|is|[:=：])?\s*.{0,80}"
    r"(?:\d+(?:号|弄|栋|幢|单元|室)|(?:road|street|avenue|lane)\b)",
    re.IGNORECASE,
)


def validate_identifier(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ProfileValidationError(
            ProfileErrorCode.INVALID_VALUE,
            f"{name} must be a non-empty bounded identifier",
        )
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > MAX_IDENTIFIER_CHARS
        or _CONTROL_CHARACTERS.search(normalized)
    ):
        raise ProfileValidationError(
            ProfileErrorCode.INVALID_VALUE,
            f"{name} must be a non-empty bounded identifier",
        )
    return normalized


def validate_aware_datetime(name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ProfileValidationError(
            ProfileErrorCode.INVALID_VALUE,
            f"{name} must include a timezone",
        )


def contains_high_risk_secret(text: str) -> bool:
    """Return only a classification; callers must never log the rejected text."""

    candidate = str(text)
    if _KNOWN_TOKEN.search(candidate) or _CHINESE_ID.search(candidate):
        return True
    if _CREDENTIAL_ASSIGNMENT.search(candidate):
        return True
    if _FINANCIAL_ACCOUNT.search(candidate):
        return True
    return bool(_EXACT_ADDRESS.search(candidate))


def validate_profile_text(text: str) -> str:
    if not isinstance(text, str):
        raise ProfileValidationError(
            ProfileErrorCode.INVALID_VALUE,
            "profile text must be non-empty and within the bounded field limit",
        )
    normalized = " ".join(text.split())
    if not normalized or len(normalized) > MAX_ENTRY_TEXT_CHARS:
        raise ProfileValidationError(
            ProfileErrorCode.INVALID_VALUE,
            "profile text must be non-empty and within the bounded field limit",
        )
    if _CONTROL_CHARACTERS.search(normalized):
        raise ProfileValidationError(
            ProfileErrorCode.INVALID_VALUE,
            "profile text contains unsupported control characters",
        )
    if contains_high_risk_secret(normalized):
        raise ProfileValidationError(
            ProfileErrorCode.HIGH_RISK_SECRET,
            "high-risk secret material cannot be stored in a player profile",
        )
    return normalized


def validate_evidence_note(note: str) -> str:
    if not isinstance(note, str):
        raise ProfileValidationError(
            ProfileErrorCode.INVALID_VALUE,
            "evidence note exceeds its compact bounded field",
        )
    normalized = " ".join(note.split())
    if len(normalized) > MAX_EVIDENCE_NOTE_CHARS or _CONTROL_CHARACTERS.search(normalized):
        raise ProfileValidationError(
            ProfileErrorCode.INVALID_VALUE,
            "evidence note exceeds its compact bounded field",
        )
    if normalized and contains_high_risk_secret(normalized):
        raise ProfileValidationError(
            ProfileErrorCode.HIGH_RISK_SECRET,
            "high-risk secret material cannot be stored in profile evidence",
        )
    return normalized


class ProfileLayer(StrEnum):
    PLAYER_FACT = "PLAYER_FACT"
    AI_OBSERVATION = "AI_OBSERVATION"


class ProfileCategory(StrEnum):
    SELF_DESCRIPTION = "SELF_DESCRIPTION"
    LIKE = "LIKE"
    DISLIKE = "DISLIKE"
    INTEREST = "INTEREST"
    HABIT = "HABIT"
    COMMUNICATION_PREFERENCE = "COMMUNICATION_PREFERENCE"
    BOUNDARY = "BOUNDARY"
    AVOID_TOPIC = "AVOID_TOPIC"
    RELATIONSHIP_NAME = "RELATIONSHIP_NAME"
    ALIAS = "ALIAS"
    INSTANCE_ROLE = "INSTANCE_ROLE"
    LITERARY_IMPRESSION = "LITERARY_IMPRESSION"
    OTHER = "OTHER"


class ProfileSourceType(StrEnum):
    PLAYER_STATEMENT = "PLAYER_STATEMENT"
    STRONG_MESSAGE_EVIDENCE = "STRONG_MESSAGE_EVIDENCE"
    PLAYER_CORRECTION = "PLAYER_CORRECTION"
    AI_OBSERVATION = "AI_OBSERVATION"


class ProfileSensitivity(StrEnum):
    NORMAL = "NORMAL"
    PRIVATE = "PRIVATE"
    SENSITIVE = "SENSITIVE"


class ProfileEntryStatus(StrEnum):
    ACTIVE = "ACTIVE"
    WITHDRAWN = "WITHDRAWN"
    SUPERSEDED = "SUPERSEDED"


class ProfileInputModality(StrEnum):
    TEXT = "TEXT"


def _validate_entry_draft_types(
    *,
    input_modality: ProfileInputModality,
    layer: ProfileLayer,
    category: ProfileCategory,
    source_type: ProfileSourceType,
    sensitivity: ProfileSensitivity,
    evidence: tuple[ProfileEvidence, ...],
) -> None:
    checks = (
        (input_modality, ProfileInputModality, "profile input modality is not supported"),
        (layer, ProfileLayer, "profile layer is not supported"),
        (category, ProfileCategory, "profile category is not supported"),
        (source_type, ProfileSourceType, "profile source type is not supported"),
        (sensitivity, ProfileSensitivity, "profile sensitivity is not supported"),
    )
    if input_modality is not ProfileInputModality.TEXT:
        raise ProfileValidationError(
            ProfileErrorCode.VISUAL_INPUT_FORBIDDEN,
            "player profiles accept controlled text evidence only; visual input is forbidden",
        )
    for value, expected_type, message in checks[1:]:
        if not isinstance(value, expected_type):
            raise ProfileValidationError(ProfileErrorCode.INVALID_VALUE, message)
    if not isinstance(evidence, tuple) or not all(_is_profile_evidence(item) for item in evidence):
        raise ProfileValidationError(
            ProfileErrorCode.INVALID_VALUE,
            "profile evidence must use controlled message or administrator references",
        )


def _normalized_confidence(
    layer: ProfileLayer,
    source_type: ProfileSourceType,
    confidence: float | None,
) -> float | None:
    if layer is ProfileLayer.PLAYER_FACT:
        if source_type is ProfileSourceType.AI_OBSERVATION or confidence is not None:
            raise ProfileValidationError(
                ProfileErrorCode.INVALID_VALUE,
                "player facts cannot be stored as AI observations or carry AI confidence",
            )
        return None
    if source_type is not ProfileSourceType.AI_OBSERVATION:
        raise ProfileValidationError(
            ProfileErrorCode.INVALID_VALUE,
            "AI observations require the AI observation source marker",
        )
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ProfileValidationError(
            ProfileErrorCode.INVALID_VALUE,
            "AI observations require confidence between zero and one",
        )
    return float(confidence)


def _validate_withdrawal_state(
    status: ProfileEntryStatus,
    evidence: tuple[ProfileEvidence, ...],
    withdrawn_at: datetime | None,
) -> None:
    if status is ProfileEntryStatus.WITHDRAWN and withdrawn_at is None:
        raise ProfileValidationError(
            ProfileErrorCode.INVALID_VALUE,
            "withdrawn entries require a withdrawal timestamp",
        )
    if status is ProfileEntryStatus.WITHDRAWN and not evidence:
        raise ProfileValidationError(
            ProfileErrorCode.INVALID_VALUE,
            "withdrawn entries require controlled withdrawal evidence",
        )
    if status is not ProfileEntryStatus.WITHDRAWN and withdrawn_at is not None:
        raise ProfileValidationError(
            ProfileErrorCode.INVALID_VALUE,
            "only withdrawn entries may carry a withdrawal timestamp",
        )
    if status is not ProfileEntryStatus.WITHDRAWN and evidence:
        raise ProfileValidationError(
            ProfileErrorCode.INVALID_VALUE,
            "only withdrawn entries may carry withdrawal evidence",
        )


def _validate_evidence_scope(
    scope: PlayerProfileScope,
    evidence: tuple[ProfileEvidence, ...],
    message: str,
) -> None:
    if any(item.scope != scope for item in evidence):
        raise ProfileValidationError(ProfileErrorCode.SCOPE_MISMATCH, message)


@dataclass(frozen=True, slots=True)
class PlayerProfileScope:
    profile_id: str
    instance_id: str
    subject_key: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", validate_identifier("profile_id", self.profile_id))
        object.__setattr__(
            self,
            "instance_id",
            validate_identifier("instance_id", self.instance_id),
        )
        object.__setattr__(
            self,
            "subject_key",
            validate_identifier("subject_key", self.subject_key),
        )

    @property
    def persistence_key(self) -> tuple[str, str, str]:
        return (self.profile_id, self.instance_id, self.subject_key)


@dataclass(frozen=True, slots=True)
class ProfileMessageEvidence:
    scope: PlayerProfileScope
    message_ref: str
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.scope, PlayerProfileScope):
            raise ProfileValidationError(
                ProfileErrorCode.INVALID_VALUE,
                "profile evidence requires a stable player profile scope",
            )
        object.__setattr__(
            self,
            "message_ref",
            validate_identifier("message_ref", self.message_ref),
        )
        object.__setattr__(self, "note", validate_evidence_note(self.note))


@dataclass(frozen=True, slots=True)
class ProfileAdminEvidence:
    scope: PlayerProfileScope
    actor: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.scope, PlayerProfileScope):
            raise ProfileValidationError(
                ProfileErrorCode.INVALID_VALUE,
                "administrator evidence requires a stable player profile scope",
            )
        object.__setattr__(self, "actor", validate_identifier("actor", self.actor))
        object.__setattr__(self, "reason", validate_evidence_note(self.reason))
        if not self.reason:
            raise ProfileValidationError(
                ProfileErrorCode.INVALID_VALUE,
                "administrator evidence requires an operation reason",
            )


ProfileEvidence: TypeAlias = ProfileMessageEvidence | ProfileAdminEvidence


def _is_profile_evidence(value: object) -> bool:
    return isinstance(value, (ProfileMessageEvidence, ProfileAdminEvidence))


@dataclass(frozen=True, slots=True)
class ProfileEntryDraft:
    layer: ProfileLayer
    category: ProfileCategory
    text: str
    source_type: ProfileSourceType
    evidence: tuple[ProfileEvidence, ...]
    confirmed_at: datetime
    confidence: float | None = None
    sensitivity: ProfileSensitivity = ProfileSensitivity.NORMAL
    input_modality: ProfileInputModality = ProfileInputModality.TEXT

    def __post_init__(self) -> None:
        _validate_entry_draft_types(
            input_modality=self.input_modality,
            layer=self.layer,
            category=self.category,
            source_type=self.source_type,
            sensitivity=self.sensitivity,
            evidence=self.evidence,
        )
        object.__setattr__(self, "text", validate_profile_text(self.text))
        validate_aware_datetime("confirmed_at", self.confirmed_at)
        if not self.evidence:
            raise ProfileValidationError(
                ProfileErrorCode.INVALID_VALUE,
                "a profile entry requires controlled message evidence",
            )
        object.__setattr__(
            self,
            "confidence",
            _normalized_confidence(self.layer, self.source_type, self.confidence),
        )


@dataclass(frozen=True, slots=True)
class PlayerProfileEntry:
    entry_id: str
    scope: PlayerProfileScope
    version: int
    layer: ProfileLayer
    category: ProfileCategory
    text: str
    source_type: ProfileSourceType
    evidence: tuple[ProfileEvidence, ...]
    confidence: float | None
    sensitivity: ProfileSensitivity
    status: ProfileEntryStatus
    confirmed_at: datetime
    created_at: datetime
    updated_at: datetime
    withdrawal_evidence: tuple[ProfileEvidence, ...] = ()
    withdrawn_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_id", validate_identifier("entry_id", self.entry_id))
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ProfileValidationError(
                ProfileErrorCode.INVALID_VALUE,
                "entry version must be positive",
            )
        if not isinstance(self.status, ProfileEntryStatus):
            raise ProfileValidationError(
                ProfileErrorCode.INVALID_VALUE,
                "profile entry status is not supported",
            )
        if not isinstance(self.withdrawal_evidence, tuple) or not all(
            _is_profile_evidence(item) for item in self.withdrawal_evidence
        ):
            raise ProfileValidationError(
                ProfileErrorCode.INVALID_VALUE,
                "withdrawal evidence must use controlled message or administrator references",
            )
        object.__setattr__(self, "text", validate_profile_text(self.text))
        validate_aware_datetime("confirmed_at", self.confirmed_at)
        validate_aware_datetime("created_at", self.created_at)
        validate_aware_datetime("updated_at", self.updated_at)
        if self.withdrawn_at is not None:
            validate_aware_datetime("withdrawn_at", self.withdrawn_at)
        _validate_withdrawal_state(
            self.status,
            self.withdrawal_evidence,
            self.withdrawn_at,
        )
        ProfileEntryDraft(
            layer=self.layer,
            category=self.category,
            text=self.text,
            source_type=self.source_type,
            evidence=self.evidence,
            confidence=self.confidence,
            sensitivity=self.sensitivity,
            confirmed_at=self.confirmed_at,
        )
        _validate_evidence_scope(
            self.scope,
            self.evidence,
            "profile evidence belongs to a different profile subject scope",
        )
        _validate_evidence_scope(
            self.scope,
            self.withdrawal_evidence,
            "profile withdrawal evidence belongs to a different subject scope",
        )


@dataclass(frozen=True, slots=True)
class PlayerProfileSnapshot:
    scope: PlayerProfileScope
    version: int
    entries: tuple[PlayerProfileEntry, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scope, PlayerProfileScope):
            raise ProfileValidationError(
                ProfileErrorCode.INVALID_VALUE,
                "profile snapshot requires a stable player profile scope",
            )
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 0:
            raise ProfileValidationError(
                ProfileErrorCode.INVALID_VALUE,
                "profile version cannot be negative",
            )
        if not isinstance(self.entries, tuple) or not all(
            isinstance(entry, PlayerProfileEntry) for entry in self.entries
        ):
            raise ProfileValidationError(
                ProfileErrorCode.INVALID_VALUE,
                "profile snapshot entries must be an immutable current-entry tuple",
            )
        identifiers: set[str] = set()
        for entry in self.entries:
            if entry.scope != self.scope:
                raise ProfileValidationError(
                    ProfileErrorCode.SCOPE_MISMATCH,
                    "snapshot contains an entry from another profile subject scope",
                )
            if entry.status is ProfileEntryStatus.SUPERSEDED:
                raise ProfileValidationError(
                    ProfileErrorCode.INVALID_VALUE,
                    "historical superseded revisions cannot appear in the current snapshot",
                )
            if entry.entry_id in identifiers:
                raise ProfileValidationError(
                    ProfileErrorCode.INVALID_VALUE,
                    "snapshot contains duplicate current entry identifiers",
                )
            identifiers.add(entry.entry_id)

    @classmethod
    def empty(cls, scope: PlayerProfileScope) -> PlayerProfileSnapshot:
        return cls(scope=scope, version=0)

    @property
    def effective_entries(self) -> tuple[PlayerProfileEntry, ...]:
        return tuple(entry for entry in self.entries if entry.status is ProfileEntryStatus.ACTIVE)

    def find_entry(self, entry_id: str) -> PlayerProfileEntry | None:
        return next((entry for entry in self.entries if entry.entry_id == entry_id), None)


__all__ = [
    "MAX_ENTRY_TEXT_CHARS",
    "MAX_EVIDENCE_NOTE_CHARS",
    "MAX_IDENTIFIER_CHARS",
    "PlayerProfileEntry",
    "PlayerProfileError",
    "PlayerProfileScope",
    "PlayerProfileSnapshot",
    "ProfileAdminEvidence",
    "ProfileCategory",
    "ProfileConflictError",
    "ProfileEntryDraft",
    "ProfileEntryStatus",
    "ProfileErrorCode",
    "ProfileEvidence",
    "ProfileInputModality",
    "ProfileLayer",
    "ProfileMessageEvidence",
    "ProfileSensitivity",
    "ProfileSourceType",
    "ProfileValidationError",
    "contains_high_risk_secret",
    "validate_aware_datetime",
    "validate_evidence_note",
    "validate_identifier",
    "validate_profile_text",
]
