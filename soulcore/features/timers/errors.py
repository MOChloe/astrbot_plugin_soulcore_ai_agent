"""Safe, payload-free failures raised by the Timer domain."""

from __future__ import annotations

from enum import StrEnum


class TimerErrorCode(StrEnum):
    INVALID_SCOPE = "INVALID_SCOPE"
    INVALID_REFERENCE = "INVALID_REFERENCE"
    INVALID_PROMPT = "INVALID_PROMPT"
    INVALID_RULE = "INVALID_RULE"
    UNSUPPORTED_RULE = "UNSUPPORTED_RULE"
    INVALID_TIMEZONE = "INVALID_TIMEZONE"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    INVALID_STATE = "INVALID_STATE"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"


_SAFE_MESSAGES: dict[TimerErrorCode, str] = {
    TimerErrorCode.INVALID_SCOPE: "timer scope is invalid",
    TimerErrorCode.INVALID_REFERENCE: "timer reference is invalid",
    TimerErrorCode.INVALID_PROMPT: "timer prompt is invalid",
    TimerErrorCode.INVALID_RULE: "timer rule is invalid",
    TimerErrorCode.UNSUPPORTED_RULE: "timer rule type is not supported",
    TimerErrorCode.INVALID_TIMEZONE: "timer timezone is invalid",
    TimerErrorCode.OUT_OF_RANGE: "timer value is outside the allowed range",
    TimerErrorCode.INVALID_STATE: "timer state transition is not allowed",
    TimerErrorCode.VERSION_CONFLICT: "timer version has changed",
    TimerErrorCode.IDEMPOTENCY_CONFLICT: "timer operation key was reused",
    TimerErrorCode.SCOPE_MISMATCH: "timer does not belong to this scope",
    TimerErrorCode.LIMIT_EXCEEDED: "timer domain limit was exceeded",
}


class TimerDomainError(ValueError):
    """A controlled error which never includes caller payloads."""

    def __init__(self, code: TimerErrorCode) -> None:
        self.code = code
        super().__init__(_SAFE_MESSAGES[code])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code.value!r})"


def fail(code: TimerErrorCode) -> TimerDomainError:
    return TimerDomainError(code)


__all__ = ["TimerDomainError", "TimerErrorCode", "fail"]
