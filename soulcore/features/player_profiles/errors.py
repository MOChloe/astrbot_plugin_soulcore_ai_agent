"""Typed, redaction-safe failures for the player-profile domain."""

from __future__ import annotations

from enum import StrEnum


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


__all__ = [
    "PlayerProfileError",
    "ProfileConflictError",
    "ProfileErrorCode",
    "ProfileValidationError",
]
