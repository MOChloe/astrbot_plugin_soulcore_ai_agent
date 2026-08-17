"""SoulCore-owned runtime limits shared by bootstrap and platform adapters."""

from __future__ import annotations

import math

from .ai_models import DEFAULT_AI_OPERATION_TIMEOUT_SECONDS

SCHEDULER_POLL_SECONDS = 5
AI_OPERATION_TIMEOUT_SECONDS = int(DEFAULT_AI_OPERATION_TIMEOUT_SECONDS)
DEFAULT_RESPONSE_POLISH_TIMEOUT_SECONDS = 60
MIN_RESPONSE_POLISH_TIMEOUT_SECONDS = 10
MAX_RESPONSE_POLISH_TIMEOUT_SECONDS = 600
IMAGE_GENERATION_TIMEOUT_SECONDS = 600
FILE_ARTIFACT_PDF_TIMEOUT_SECONDS = 3000
AI_BACKGROUND_CONCURRENCY = 2
# A background task may span restarts, but one immutable input must eventually
# become terminal.  With the standard 5h/10h/20h/24h backoff this budget gives
# transient provider outages more than ten days to recover without allowing an
# unbounded RETRY_WAIT/audit history.
DURABLE_AI_MAX_ATTEMPTS = 12


def require_response_polish_timeout_seconds(value: object) -> int:
    """Validate the one profile-owned deadline used by the whole polish request."""

    if isinstance(value, bool):
        raise ValueError("response_polish_timeout_seconds must be an integer")
    if not isinstance(value, (int, float, str)):
        raise ValueError("response_polish_timeout_seconds must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("response_polish_timeout_seconds must be an integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError("response_polish_timeout_seconds must be an integer")
    timeout = int(numeric)
    if not MIN_RESPONSE_POLISH_TIMEOUT_SECONDS <= timeout <= MAX_RESPONSE_POLISH_TIMEOUT_SECONDS:
        raise ValueError(
            "response_polish_timeout_seconds must be between "
            f"{MIN_RESPONSE_POLISH_TIMEOUT_SECONDS} and "
            f"{MAX_RESPONSE_POLISH_TIMEOUT_SECONDS}"
        )
    return timeout


__all__ = [
    "AI_BACKGROUND_CONCURRENCY",
    "AI_OPERATION_TIMEOUT_SECONDS",
    "DEFAULT_RESPONSE_POLISH_TIMEOUT_SECONDS",
    "DURABLE_AI_MAX_ATTEMPTS",
    "FILE_ARTIFACT_PDF_TIMEOUT_SECONDS",
    "IMAGE_GENERATION_TIMEOUT_SECONDS",
    "MAX_RESPONSE_POLISH_TIMEOUT_SECONDS",
    "MIN_RESPONSE_POLISH_TIMEOUT_SECONDS",
    "SCHEDULER_POLL_SECONDS",
    "require_response_polish_timeout_seconds",
]
