"""Internal-only failures for durable Main Core work checkpoints."""

from __future__ import annotations

from enum import StrEnum


class WorkCheckpointStorageErrorCode(StrEnum):
    NOT_FOUND = "NOT_FOUND"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    LEASE_CONFLICT = "LEASE_CONFLICT"
    INVALID_STATE = "INVALID_STATE"
    INVALID_PERSISTED_DATA = "INVALID_PERSISTED_DATA"
    OUT_OF_RANGE = "OUT_OF_RANGE"


class WorkCheckpointStorageError(RuntimeError):
    """A stable internal error that never embeds persisted or callback content."""

    def __init__(self, code: WorkCheckpointStorageErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


def storage_fail(code: WorkCheckpointStorageErrorCode) -> WorkCheckpointStorageError:
    return WorkCheckpointStorageError(code)


__all__ = [
    "WorkCheckpointStorageError",
    "WorkCheckpointStorageErrorCode",
    "storage_fail",
]
