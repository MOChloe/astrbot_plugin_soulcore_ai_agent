"""Sanitized AI failure classification and operator diagnostics."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from ...contracts.ai_models import (
    AIErrorCode,
    AIErrorInfo,
    AIInvocationError,
)

_DIAGNOSTIC_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"https?://\S+", re.IGNORECASE), "[redacted-url]"),
    (re.compile(r"(?:[A-Za-z]:\\|\\\\)[^\s]+"), "[redacted-path]"),
    (re.compile(r"/(?:home|Users|tmp|var|etc)/[^\s]+"), "[redacted-path]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), "[redacted-secret]"),
    (
        re.compile(
            r"(?:[\"']?(?:api[_ -]?key|access[_ -]?token|authorization|cookie|"
            r"password|passwd|secret|credential)[\"']?"
            r"\s*(?:[:=]|\s)\s*)[\"']?[^\"',;\s}\]]+",
            re.IGNORECASE,
        ),
        "[redacted-secret]",
    ),
    (
        re.compile(r"\b[A-Z][A-Z0-9_]{2,}\s*=\s*[^\s,;]+"),
        "[redacted-secret]",
    ),
)


def _redact_diagnostic_text(value: Any, *, limit: int = 240) -> str:
    """Return bounded operator-facing text without endpoints, paths or keys."""

    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or "")).strip()
    for pattern, replacement in _DIAGNOSTIC_SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text)[: max(1, int(limit))]


def _safe_diagnostic_identifier(value: Any, *, limit: int = 160) -> str:
    text = _redact_diagnostic_text(value, limit=limit)
    if not text:
        return ""
    if "[redacted-" in text:
        return "[redacted]"
    # Backend/model identifiers are configuration labels, never arbitrary
    # provider payloads.  Replace control/punctuation outside their normal
    # alphabet so a malicious label cannot turn the console into a data sink.
    return re.sub(r"[^0-9A-Za-z._:@/+\[\]\- ]", "?", text)[:limit]


def _safe_nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _exception_type_chain(exc: BaseException, *, limit: int = 6) -> list[str]:
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    output: list[str] = []
    while pending and len(output) < max(1, int(limit)):
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        output.append(re.sub(r"[^0-9A-Za-z_.]", "?", type(current).__name__)[:100] or "Exception")
        children: list[BaseException] = []
        for child in (
            getattr(current, "__cause__", None),
            getattr(current, "cause", None),
            getattr(current, "__context__", None),
        ):
            if isinstance(child, BaseException) and id(child) not in seen:
                children.append(child)
        for attribute in ("exceptions", "errors", "causes"):
            nested = getattr(current, attribute, None)
            values = nested.values() if isinstance(nested, Mapping) else nested
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                children.extend(
                    child
                    for child in values
                    if isinstance(child, BaseException) and id(child) not in seen
                )
        pending.extend(children)
    return output


def _exception_diagnostics(exc: BaseException) -> dict[str, Any]:
    chain = _exception_type_chain(exc)
    details: dict[str, Any] = {
        "exception_type": chain[0] if chain else "Exception",
        "cause_chain": chain,
    }
    if len(chain) > 1:
        details["cause_type"] = chain[1]
    if isinstance(exc, AIInvocationError):
        details["source_error_code"] = exc.info.code.value
        details["source_phase"] = _safe_diagnostic_identifier(exc.info.phase, limit=80)
    return details


def safe_ai_failure_details(exc: BaseException) -> dict[str, Any]:
    """Build an allow-listed failure view safe for DB metadata and the UI."""

    info = exc.info if isinstance(exc, AIInvocationError) else classify_generic_error(exc)
    raw = dict(info.details or {})
    exception = _exception_diagnostics(exc)
    chain = raw.get("cause_chain")
    if not isinstance(chain, (list, tuple)):
        chain = exception.get("cause_chain", ())
    result = _base_failure_details(info, raw, exception, chain)
    result["attempts"] = [
        _sanitized_attempt(value)
        for value in list(raw.get("attempts") or ())[:10]
        if isinstance(value, Mapping)
    ]
    result = {key: value for key, value in result.items() if value not in (None, "")}
    for key in _RECOVERY_BOOLEAN_KEYS:
        if raw.get(key) is True:
            result[key] = True
    _attach_rejection_details(result, exc)
    return result


def _attach_rejection_details(result: dict[str, Any], exc: BaseException) -> None:
    errors = getattr(exc, "errors", None)
    if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
        result["rejections"] = ["command_validation_rejected"] * min(len(errors), 3)
    rounds = getattr(exc, "rounds", None)
    if isinstance(rounds, Sequence) and not isinstance(rounds, (str, bytes)):
        result["rejection_rounds"] = [
            {
                "round": max(0, int(getattr(value, "number", 0) or 0)),
                "reason": "command_validation_rejected",
            }
            for value in list(rounds)[-3:]
        ]


_RECOVERY_BOOLEAN_KEYS = (
    "external_side_effect_unknown",
    "possible_duplicate_billing",
    "recovery_required",
)


def _base_failure_details(
    info: AIErrorInfo,
    raw: Mapping[str, Any],
    exception: Mapping[str, Any],
    chain: Sequence[Any],
) -> dict[str, Any]:
    return {
        "error_code": str(info.code.value)[:80],
        "message": _redact_diagnostic_text(info.safe_message, limit=500),
        "http_status": info.status_code,
        "provider_error_code": _safe_diagnostic_identifier(raw.get("api_code"), limit=160),
        "provider_response": _safe_provider_response(raw.get("provider_response")),
        "backend_id": _safe_diagnostic_identifier(info.backend_id or raw.get("backend_id")),
        "model_id": _safe_diagnostic_identifier(raw.get("model_id")),
        "context_error_kind": _safe_diagnostic_identifier(raw.get("context_error_kind"), limit=80),
        "configured_max_context_tokens": _safe_nonnegative_int(
            raw.get("configured_max_context_tokens")
        ),
        "required_context_tokens": _safe_nonnegative_int(raw.get("required_context_tokens")),
        "input_text_tokens": _safe_nonnegative_int(raw.get("input_text_tokens")),
        "input_image_tokens": _safe_nonnegative_int(raw.get("input_image_tokens")),
        "reserved_output_tokens": _safe_nonnegative_int(raw.get("reserved_output_tokens")),
        "round": _safe_nonnegative_int(raw.get("round")),
        "phase": _safe_diagnostic_identifier(info.phase or raw.get("phase"), limit=80),
        "exception_type": _safe_diagnostic_identifier(
            raw.get("exception_type") or exception.get("exception_type"), limit=100
        ),
        "cause_type": _safe_diagnostic_identifier(
            raw.get("cause_type") or exception.get("cause_type"), limit=100
        ),
        "cause_chain": [
            _safe_diagnostic_identifier(value, limit=100)
            for value in list(chain)[:6]
            if _safe_diagnostic_identifier(value, limit=100)
        ],
    }


def _safe_provider_response(value: Any) -> str:
    """Keep provider error metadata while removing echoed requests and credentials."""

    if value in (None, ""):
        return ""
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return _redact_diagnostic_text(value, limit=1000)
    if not isinstance(parsed, (Mapping, list, tuple)):
        return _redact_diagnostic_text(parsed, limit=1000)
    cleaned = _redact_provider_value(parsed)
    return _redact_diagnostic_text(
        json.dumps(cleaned, ensure_ascii=False, separators=(",", ":")), limit=2000
    )


_PROVIDER_PRIVATE_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "password",
        "passwd",
        "secret",
        "token",
        "credential",
        "prompt",
        "input",
        "messages",
        "content",
        "request",
        "request_body",
        "body",
        "data",
        "binary",
    }
)


def _redact_provider_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[redacted]"
                if str(key).strip().lower() in _PROVIDER_PRIVATE_KEYS
                else _redact_provider_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_provider_value(item) for item in value[:20]]
    return value


def _sanitized_attempt(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "attempt_no": max(0, int(value.get("attempt_no") or 0)),
        "backend_id": _safe_diagnostic_identifier(value.get("backend_id")),
        "model_id": _safe_diagnostic_identifier(value.get("model_id")),
        "error_code": _safe_diagnostic_identifier(value.get("error_code"), limit=80),
        "phase": _safe_diagnostic_identifier(value.get("phase"), limit=80),
        "exception_type": _safe_diagnostic_identifier(value.get("exception_type"), limit=100),
        "cause_type": _safe_diagnostic_identifier(value.get("cause_type"), limit=100),
        "http_status": value.get("http_status"),
        "provider_error_code": _safe_diagnostic_identifier(
            value.get("provider_error_code"), limit=160
        ),
        "decision": _safe_diagnostic_identifier(value.get("decision"), limit=100),
    }


def classify_generic_error(exc: BaseException, backend_id: str = "") -> AIErrorInfo:
    special = _special_error_info(exc, backend_id)
    if special is not None:
        return special
    name = type(exc).__name__
    code = _matched_error_code(name, str(exc).lower())
    return AIErrorInfo(
        code,
        f"AI invocation failed: {name}",
        retryable=code in _RETRYABLE_CODES,
        switch_backend=code in _SWITCHABLE_CODES,
        open_circuit=code in _CIRCUIT_OPEN_CODES,
        backend_id=backend_id,
        phase="adapter",
        details=_exception_diagnostics(exc),
    )


_ERROR_PATTERNS = (
    (AIErrorCode.AUTHENTICATION, ("401", "unauthorized", "invalid api key", "authentication")),
    (AIErrorCode.PERMISSION, ("403", "forbidden", "permission denied", "model access")),
    (
        AIErrorCode.QUOTA_EXHAUSTED,
        ("insufficient_quota", "quota exhausted", "billing", "欠费", "余额不足"),
    ),
    (AIErrorCode.RATE_LIMIT, ("429", "rate limit", "too many requests", "限流")),
    (AIErrorCode.TIMEOUT, ("timeout", "timed out")),
    (AIErrorCode.REMOTE_5XX, ("500", "502", "503", "504", "server error")),
    (AIErrorCode.SAFETY_REFUSAL, ("content policy", "safety refusal", "content filter")),
    (AIErrorCode.EMPTY_OUTPUT, ("empty", "no usable output")),
)
_CIRCUIT_OPEN_CODES = frozenset(
    {
        AIErrorCode.AUTHENTICATION,
        AIErrorCode.PERMISSION,
        AIErrorCode.QUOTA_EXHAUSTED,
        AIErrorCode.RATE_LIMIT,
    }
)
_RETRYABLE_CODES = frozenset(
    {AIErrorCode.TIMEOUT, AIErrorCode.REMOTE_5XX, AIErrorCode.EMPTY_OUTPUT}
)
_SWITCHABLE_CODES = _CIRCUIT_OPEN_CODES | _RETRYABLE_CODES


def _matched_error_code(name: str, lowered: str) -> AIErrorCode:
    if name == "AllInvocationError" or "allinvocationerror" in lowered:
        return AIErrorCode.REMOTE_5XX
    if name == "ContextBudgetExceeded" or "contextbudgetexceeded" in lowered:
        return AIErrorCode.CONTEXT_BUDGET
    if "context" in lowered and ("token" in lowered or "length" in lowered):
        return AIErrorCode.CONTEXT_BUDGET
    for code, patterns in _ERROR_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return code
    return AIErrorCode.INTERNAL


def _special_error_info(exc: BaseException, backend_id: str) -> AIErrorInfo | None:
    if isinstance(exc, AIInvocationError):
        return replace(
            exc.info,
            safe_message=_redact_diagnostic_text(exc.info.safe_message),
            backend_id=exc.info.backend_id or backend_id,
            details={**dict(exc.info.details), **_exception_diagnostics(exc)},
        )
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return AIErrorInfo(
            AIErrorCode.TIMEOUT,
            "AI backend timed out",
            retryable=True,
            switch_backend=True,
            backend_id=backend_id,
            phase="timeout",
            details=_exception_diagnostics(exc),
        )
    if isinstance(exc, (ConnectionError, OSError)):
        return AIErrorInfo(
            AIErrorCode.NETWORK,
            "AI backend connection failed",
            retryable=True,
            switch_backend=True,
            backend_id=backend_id,
            phase="transport",
            details=_exception_diagnostics(exc),
        )
    return None


__all__ = [
    "classify_generic_error",
    "safe_ai_failure_details",
]
