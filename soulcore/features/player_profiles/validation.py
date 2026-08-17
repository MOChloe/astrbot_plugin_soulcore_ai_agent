"""Deterministic validation without retaining or echoing rejected secrets."""

from __future__ import annotations

import re
from datetime import datetime

from .errors import ProfileErrorCode, ProfileValidationError

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


__all__ = [
    "MAX_ENTRY_TEXT_CHARS",
    "MAX_EVIDENCE_NOTE_CHARS",
    "MAX_IDENTIFIER_CHARS",
    "contains_high_risk_secret",
    "validate_aware_datetime",
    "validate_evidence_note",
    "validate_identifier",
    "validate_profile_text",
]
