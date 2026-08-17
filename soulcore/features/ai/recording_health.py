"""Fail-open health state for the AI work recorder itself."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


@dataclass(frozen=True, slots=True)
class AIRecordingHealthSnapshot:
    healthy: bool
    consecutive_failures: int
    total_failures: int
    failed_operation: str
    error_type: str
    error_message: str
    first_failure_at: str | None
    last_failure_at: str | None
    recovered_at: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "failed_operation": self.failed_operation,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "first_failure_at": self.first_failure_at,
            "last_failure_at": self.last_failure_at,
            "recovered_at": self.recovered_at,
        }


@dataclass(slots=True)
class _ActiveFailure:
    operation: str
    consecutive_failures: int
    error_type: str
    error_message: str
    first_failure_at: datetime
    last_failure_at: datetime


class AIRecordingHealth:
    def __init__(self) -> None:
        self._lock = Lock()
        self._total_failures = 0
        self._active_failures: dict[str, _ActiveFailure] = {}
        self._last_failure: _ActiveFailure | None = None
        self._recovered_at: datetime | None = None

    def record_failure(self, operation: str, exc: BaseException) -> None:
        now = datetime.now(UTC)
        operation_name = str(operation)
        message = str(exc).strip()[:500]
        with self._lock:
            self._total_failures += 1
            previous = self._active_failures.get(operation_name)
            failure = _ActiveFailure(
                operation=operation_name,
                consecutive_failures=(previous.consecutive_failures + 1 if previous else 1),
                error_type=type(exc).__name__,
                error_message=message,
                first_failure_at=previous.first_failure_at if previous else now,
                last_failure_at=now,
            )
            self._active_failures[operation_name] = failure
            self._last_failure = failure
            self._recovered_at = None
        logger.error(
            "AI work recording failed",
            extra={
                "ai_recording_operation": str(operation),
                "ai_recording_error_type": type(exc).__name__,
                "ai_recording_error": message,
            },
        )

    def record_success(self, operation: str) -> None:
        recovered = False
        now = datetime.now(UTC)
        operation_name = str(operation)
        with self._lock:
            if operation_name not in self._active_failures:
                return
            self._active_failures.pop(operation_name, None)
            if not self._active_failures:
                self._recovered_at = now
                recovered = True
            else:
                self._last_failure = max(
                    self._active_failures.values(), key=lambda item: item.last_failure_at
                )
        if recovered:
            logger.info(
                "AI work recording recovered",
                extra={"ai_recording_operation": operation_name},
            )

    def snapshot(self) -> AIRecordingHealthSnapshot:
        with self._lock:
            current = (
                max(self._active_failures.values(), key=lambda item: item.last_failure_at)
                if self._active_failures
                else self._last_failure
            )
            return AIRecordingHealthSnapshot(
                healthy=not self._active_failures,
                consecutive_failures=(current.consecutive_failures if self._active_failures else 0)
                if current
                else 0,
                total_failures=self._total_failures,
                failed_operation=current.operation if current else "",
                error_type=current.error_type if current else "",
                error_message=current.error_message if current else "",
                first_failure_at=_iso(current.first_failure_at if current else None),
                last_failure_at=_iso(current.last_failure_at if current else None),
                recovered_at=_iso(self._recovered_at),
            )

    def reset_for_tests(self) -> None:
        with self._lock:
            self._total_failures = 0
            self._active_failures.clear()
            self._last_failure = None
            self._recovered_at = None


ai_recording_health = AIRecordingHealth()


__all__ = ["AIRecordingHealth", "AIRecordingHealthSnapshot", "ai_recording_health"]
