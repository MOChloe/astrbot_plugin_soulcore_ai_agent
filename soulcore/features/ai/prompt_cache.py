"""Provider-neutral prompt-cache negotiation helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol

from ...contracts.ai_models import (
    AIBackendDescriptor,
    AICompletion,
    AIErrorCode,
    AIErrorInfo,
    AIInvocationError,
    AIModelRequest,
    AIPromptCacheBreakpoint,
    AIPromptCachePolicy,
    AIPromptCacheSection,
    AIPromptCacheState,
    AIPromptCacheWireMode,
)
from .prompt_cache_quality import (
    QUALITY_REJECTION_KIND,
    prompt_cache_quality_predecessor,
)

_GPT_VERSION = re.compile(r"(?:^|[/_:.-])gpt-(\d+)(?:\.(\d+))?", re.IGNORECASE)
_CACHE_FIELDS = (
    "prompt_cache",
    "prompt cache",
    "cache_control",
    "cache control",
    "cache breakpoint",
)
_REJECTION_SIGNALS = (
    "unknown",
    "unrecognized",
    "unsupported",
    "not supported",
    "not permitted",
    "not allowed",
    "invalid",
    "validation",
    "extra field",
    "extra_forbidden",
)


@dataclass(frozen=True, slots=True)
class PromptCacheObservation:
    state: AIPromptCacheState
    cache_read_tokens: int
    cache_write_tokens: int
    cache_status: str
    evidence: Mapping[str, Any]


class PromptCacheRepositoryPort(Protocol):
    async def claim_ai_prompt_cache_capability(
        self, backend_id: str, **values: object
    ) -> Mapping[str, Any]: ...

    async def observe_ai_prompt_cache_capability(
        self, backend_id: str, **values: object
    ) -> object: ...

    async def reject_ai_prompt_cache_capability(
        self, backend_id: str, **values: object
    ) -> object: ...


def candidate_cache_mode(descriptor: AIBackendDescriptor) -> AIPromptCacheWireMode:
    protocol = str(descriptor.metadata.get("protocol") or "").strip().upper()
    base_url = str(descriptor.metadata.get("base_url") or "").strip().lower()
    model = str(descriptor.model or "").strip().lower()
    if protocol == "ANTHROPIC":
        return AIPromptCacheWireMode.ANTHROPIC_EPHEMERAL
    if protocol == "OPENAI":
        return (
            AIPromptCacheWireMode.OPENAI_EXPLICIT
            if supports_openai_explicit_cache(model)
            else AIPromptCacheWireMode.OPENAI_AUTO
        )
    if protocol != "OPENAI_COMPATIBLE":
        return AIPromptCacheWireMode.DISABLED
    if "api.anthropic.com" in base_url:
        return AIPromptCacheWireMode.DISABLED
    if supports_openai_explicit_cache(model):
        return AIPromptCacheWireMode.OPENAI_EXPLICIT
    # An OpenAI-compatible route owns its wire contract even when the selected
    # model happens to be Claude. Inferring Anthropic content blocks from the
    # model name changes string messages into a vendor-specific array that many
    # gateways accept but silently exclude from their native prefix cache.
    return AIPromptCacheWireMode.OPENAI_AUTO


def supports_openai_explicit_cache(model: str) -> bool:
    match = _GPT_VERSION.search(str(model or ""))
    if match is None:
        return False
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    return major > 5 or (major == 5 and minor >= 6)


def _requested_ttl(mode: AIPromptCacheWireMode) -> str:
    if mode is AIPromptCacheWireMode.OPENAI_EXPLICIT:
        return "30m"
    if mode is AIPromptCacheWireMode.ANTHROPIC_EPHEMERAL:
        return "1h"
    return ""


def split_prompt_text(
    text: str,
    policy: AIPromptCachePolicy,
    section: AIPromptCacheSection,
) -> tuple[tuple[str, AIPromptCacheBreakpoint | None], ...]:
    """Split one logical section without changing a single source character."""

    source = str(text or "")
    boundaries = sorted(
        (
            item
            for item in policy.breakpoints[:4]
            if item.section is section and 0 < item.section_end <= len(source)
        ),
        key=lambda item: item.section_end,
    )
    by_end: dict[int, AIPromptCacheBreakpoint] = {}
    for item in boundaries:
        by_end[item.section_end] = item
    segments: list[tuple[str, AIPromptCacheBreakpoint | None]] = []
    cursor = 0
    for end, boundary in by_end.items():
        if end <= cursor:
            continue
        segments.append((source[cursor:end], boundary))
        cursor = end
    if cursor < len(source):
        segments.append((source[cursor:], None))
    if not segments and source:
        segments.append((source, None))
    if "".join(segment for segment, _boundary in segments) != source:
        raise ValueError("prompt cache section split changed logical text")
    return tuple(segments)


def prompt_cache_policy_debug(
    policy: AIPromptCachePolicy,
    hint: Any | None = None,
) -> dict[str, Any]:
    return {
        "wire_mode": policy.wire_mode.value,
        "candidate_mode": policy.candidate_mode.value,
        "state": policy.state.value,
        "cache_family": policy.cache_key,
        "requested_ttl": policy.requested_ttl,
        "actual_ttl": policy.actual_ttl,
        "probing": policy.probing,
        "suppression_reason": policy.suppression_reason,
        "breakpoints": [_breakpoint_evidence(item) for item in policy.breakpoints],
        "rebase_reasons": list(getattr(hint, "rebase_reasons", ()) or ()),
    }


def _breakpoint_evidence(item: AIPromptCacheBreakpoint) -> dict[str, Any]:
    return {
        "boundary_id": item.boundary_id,
        "semantic_kind": item.semantic_kind.value,
        "section": item.section.value,
        "section_end": item.section_end,
        "document_end": item.document_end,
        "prefix_tokens": item.prefix_tokens,
        "prefix_hash": item.prefix_hash,
        "selection_slot": item.selection_slot,
        "selection_reason": item.selection_reason,
    }


def prompt_cache_config_fingerprint(
    *,
    protocol: str,
    base_url: str,
    model: str,
    credential_id: str,
    package_config: Mapping[str, Any] | None = None,
    model_config: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "protocol": str(protocol or "").strip().upper(),
        "base_url": str(base_url or "").strip().rstrip("/").lower(),
        "model": str(model or "").strip(),
        "credential_id": str(credential_id or "").strip(),
        "package_config": dict(package_config or {}),
        "model_config": dict(model_config or {}),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def prompt_cache_key(descriptor: AIBackendDescriptor, request: AIModelRequest) -> str:
    """Build a stable cache family key; exact-prefix hashes stay in breakpoints."""

    hint = request.prompt_cache_hint
    payload = {
        "backend_id": descriptor.backend_id,
        "model": str(request.model or descriptor.model),
        "purpose": request.work_purpose.value,
        "role": str(request.metadata.get("prompt_cache_role_id") or request.profile_id),
        "profile_id": request.profile_id,
        "instance_id": request.instance_id,
        "prompt_protocol_version": (
            hint.prompt_protocol_version if hint is not None else "soulcore-prompt-v2"
        ),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sc:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:48]


def prompt_cache_attempt_usage(
    policy: AIPromptCachePolicy,
    status: str,
) -> dict[str, Any]:
    return {
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cache_mode": policy.wire_mode.value,
        "cache_candidate_mode": policy.candidate_mode.value,
        "cache_status": str(status or ""),
        "cache_suppression_reason": policy.suppression_reason,
        "cache_requested_ttl": policy.requested_ttl,
        "cache_actual_ttl": policy.actual_ttl,
        "cache_family": policy.cache_key,
        "cache_breakpoint_count": len(policy.breakpoints),
    }


def prompt_cache_error_attempt_usage(
    policy: AIPromptCachePolicy,
    error: BaseException,
) -> dict[str, Any]:
    rejected = (
        isinstance(error, AIInvocationError)
        and error.info.code is AIErrorCode.PROMPT_CACHE_MARKER_UNSUPPORTED
    )
    return prompt_cache_attempt_usage(policy, "REJECTED" if rejected else "ERROR")


async def prepare_prompt_cache_request(
    repository: PromptCacheRepositoryPort,
    request: AIModelRequest,
    descriptor: AIBackendDescriptor,
) -> AIModelRequest:
    mode = _eligible_cache_mode(request, descriptor)
    if mode is AIPromptCacheWireMode.DISABLED:
        return _without_prompt_cache(request)
    fingerprint = str(descriptor.metadata.get("prompt_cache_config_fingerprint") or "")
    if not fingerprint:
        return _without_prompt_cache(request)
    record = await repository.claim_ai_prompt_cache_capability(
        descriptor.backend_id,
        model_id=descriptor.model,
        config_fingerprint=fingerprint,
        wire_mode=mode.value,
        probe_owner=request.invocation_id,
    )
    policy = _policy_from_cache_claim(request, descriptor, mode, record)
    return replace(request, prompt_cache_policy=policy)


def _eligible_cache_mode(
    request: AIModelRequest,
    descriptor: AIBackendDescriptor,
) -> AIPromptCacheWireMode:
    hint = request.prompt_cache_hint
    if hint is None or not hint.eligible or request.capability_request is not None:
        return AIPromptCacheWireMode.DISABLED
    return candidate_cache_mode(descriptor)


def _without_prompt_cache(request: AIModelRequest) -> AIModelRequest:
    return replace(request, prompt_cache_policy=AIPromptCachePolicy())


def _policy_from_cache_claim(
    request: AIModelRequest,
    descriptor: AIBackendDescriptor,
    mode: AIPromptCacheWireMode,
    record: Mapping[str, Any],
) -> AIPromptCachePolicy:
    enabled = bool(record.get("cache_enabled"))
    state = _cache_state(record.get("state"))
    cache_key = prompt_cache_key(descriptor, request)
    hint = request.prompt_cache_hint
    breakpoints = tuple(hint.selected[:4]) if hint is not None else ()
    requested_ttl = _requested_ttl(mode)
    predecessor = prompt_cache_quality_predecessor(record.get("evidence"), cache_key)
    started_at = datetime.now(UTC).isoformat()
    if not enabled:
        rejection = record.get("rejection")
        quality_suspended = (
            isinstance(rejection, Mapping)
            and str(rejection.get("kind") or "") == QUALITY_REJECTION_KIND
        )
        return AIPromptCachePolicy(
            candidate_mode=mode,
            state=state,
            cache_key=cache_key,
            requested_ttl=requested_ttl,
            actual_ttl=requested_ttl,
            breakpoints=breakpoints,
            suppression_reason=(
                "CACHE_QUALITY_PROBE_IN_FLIGHT"
                if quality_suspended and state is AIPromptCacheState.PROBING
                else "CACHE_QUALITY_SUSPENDED"
                if quality_suspended
                else "CACHE_CAPABILITY_UNAVAILABLE"
            ),
            quality_predecessor_id=predecessor,
            quality_started_at=started_at,
        )
    return AIPromptCachePolicy(
        wire_mode=mode,
        candidate_mode=mode,
        state=state,
        cache_key=cache_key,
        requested_ttl=requested_ttl,
        actual_ttl=requested_ttl,
        breakpoints=breakpoints,
        probing=state in {AIPromptCacheState.UNTESTED, AIPromptCacheState.PROBING},
        quality_predecessor_id=predecessor,
        quality_started_at=started_at,
    )


async def observe_prompt_cache_completion(
    repository: PromptCacheRepositoryPort,
    request: AIModelRequest,
    descriptor: AIBackendDescriptor,
    completion: AICompletion,
) -> AICompletion:
    normalized, observation = normalize_prompt_cache_usage(
        completion,
        request.prompt_cache_policy,
    )
    policy = request.prompt_cache_policy
    candidate_mode = (
        policy.candidate_mode
        if policy.candidate_mode is not AIPromptCacheWireMode.DISABLED
        else policy.wire_mode
    )
    if candidate_mode is AIPromptCacheWireMode.DISABLED:
        return normalized
    if (
        policy.wire_mode is AIPromptCacheWireMode.DISABLED
        and not observation.evidence.get("read_fields")
        and not observation.evidence.get("write_fields")
    ):
        return normalized
    fingerprint = str(descriptor.metadata.get("prompt_cache_config_fingerprint") or "")
    result = await repository.observe_ai_prompt_cache_capability(
        descriptor.backend_id,
        config_fingerprint=fingerprint,
        wire_mode=candidate_mode.value,
        state=observation.state.value,
        evidence=dict(observation.evidence),
        cache_read_tokens=observation.cache_read_tokens,
        cache_write_tokens=observation.cache_write_tokens,
        observation_id=request.invocation_id,
        predecessor_id=policy.quality_predecessor_id,
        request_started_at=policy.quality_started_at,
        ttl_seconds=_quality_ttl_seconds(policy, candidate_mode),
        cache_applied=policy.wire_mode is not AIPromptCacheWireMode.DISABLED,
    )
    if isinstance(result, Mapping) and str(result.get("cache_status") or ""):
        usage = dict(normalized.usage)
        usage["cache_status"] = str(result["cache_status"])
        normalized = replace(normalized, usage=usage)
    return normalized


async def reject_prompt_cache_policy(
    repository: PromptCacheRepositoryPort,
    request: AIModelRequest,
    descriptor: AIBackendDescriptor,
    error: AIErrorInfo,
) -> None:
    mode = request.prompt_cache_policy.wire_mode
    if mode is AIPromptCacheWireMode.DISABLED:
        return
    await repository.reject_ai_prompt_cache_capability(
        descriptor.backend_id,
        config_fingerprint=str(descriptor.metadata.get("prompt_cache_config_fingerprint") or ""),
        wire_mode=mode.value,
        reason={
            "error_code": error.code.value,
            "status_code": error.status_code,
            "api_code": str(error.details.get("api_code") or "")[:160],
            "safe_message": str(error.safe_message or "")[:300],
            "requested_ttl": request.prompt_cache_policy.requested_ttl,
            "actual_ttl": request.prompt_cache_policy.actual_ttl,
        },
    )


def is_explicit_cache_rejection(status_code: int, provider_response: Any) -> bool:
    if int(status_code) not in {400, 422}:
        return False
    text = _response_text(provider_response).lower()[:8000]
    return any(field in text for field in _CACHE_FIELDS) and any(
        signal in text for signal in _REJECTION_SIGNALS
    )


def normalize_prompt_cache_usage(
    completion: AICompletion,
    policy: AIPromptCachePolicy,
) -> tuple[AICompletion, PromptCacheObservation]:
    usage = dict(completion.usage)
    read_tokens, read_keys = _read_cache_tokens(usage)
    write_tokens, write_keys = _write_cache_tokens(usage)
    recognized = bool(read_keys or write_keys)
    if recognized:
        state = AIPromptCacheState.CONFIRMED
        status = "HIT" if read_tokens else "WRITE" if write_tokens else "CONFIRMED_NO_HIT"
    elif policy.wire_mode is not AIPromptCacheWireMode.DISABLED:
        state = AIPromptCacheState.ACCEPTED_UNVERIFIED
        status = "ACCEPTED_UNVERIFIED"
    else:
        state = AIPromptCacheState.UNTESTED
        status = "NOT_USED"
    usage.update(
        {
            "cache_read_tokens": read_tokens,
            "cache_write_tokens": write_tokens,
            "cache_mode": policy.wire_mode.value,
            "cache_candidate_mode": policy.candidate_mode.value,
            "cache_status": status,
            "cache_suppression_reason": policy.suppression_reason,
            "cache_requested_ttl": policy.requested_ttl,
            "cache_actual_ttl": policy.actual_ttl,
            "cache_family": policy.cache_key,
            "cache_breakpoint_count": len(policy.breakpoints),
        }
    )
    observation = PromptCacheObservation(
        state=state,
        cache_read_tokens=read_tokens,
        cache_write_tokens=write_tokens,
        cache_status=status,
        evidence={
            "read_fields": read_keys,
            "write_fields": write_keys,
            "cache_read_tokens": read_tokens,
            "cache_write_tokens": write_tokens,
            "requested_ttl": policy.requested_ttl,
            "effective_ttl": policy.actual_ttl,
            "cache_family": policy.cache_key,
            "breakpoints": [_breakpoint_evidence(item) for item in policy.breakpoints],
        },
    )
    return replace(completion, usage=usage), observation


def _quality_ttl_seconds(
    policy: AIPromptCachePolicy,
    candidate_mode: AIPromptCacheWireMode,
) -> int:
    ttl = str(policy.actual_ttl or policy.requested_ttl).strip().lower()
    if ttl.endswith("m") and ttl[:-1].isdigit():
        return max(1, int(ttl[:-1])) * 60
    if ttl.endswith("h") and ttl[:-1].isdigit():
        return max(1, int(ttl[:-1])) * 3600
    # Automatic OpenAI-compatible caches do not expose one portable TTL.
    # Five minutes is deliberately conservative for quality comparisons.
    return 300 if candidate_mode is AIPromptCacheWireMode.OPENAI_AUTO else 3600


def _read_cache_tokens(usage: Mapping[str, Any]) -> tuple[int, tuple[str, ...]]:
    values, keys = _cache_token_fields(
        usage,
        ("cache_read_input_tokens", "cache_read_tokens"),
    )
    details = usage.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        nested_values, nested_keys = _cache_token_fields(
            details,
            ("cached_tokens",),
            prefix="prompt_tokens_details.",
        )
        values.extend(nested_values)
        keys.extend(nested_keys)
    claude_usage = _claude_billing_usage(usage)
    if claude_usage is not None:
        nested_values, nested_keys = _cache_token_fields(
            claude_usage,
            ("cache_read_input_tokens", "cache_read_tokens"),
            prefix="billing_usage.claude_usage.",
        )
        values.extend(nested_values)
        keys.extend(nested_keys)
    return (max(values, default=0), tuple(keys))


def _write_cache_tokens(usage: Mapping[str, Any]) -> tuple[int, tuple[str, ...]]:
    values, keys = _cache_token_fields(
        usage,
        ("cache_creation_input_tokens", "cache_write_tokens"),
    )
    details = usage.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        nested_values, nested_keys = _cache_token_fields(
            details,
            ("cached_creation_tokens", "cache_write_tokens"),
            prefix="prompt_tokens_details.",
        )
        values.extend(nested_values)
        keys.extend(nested_keys)
    creation = usage.get("cache_creation")
    if isinstance(creation, Mapping):
        _append_cache_creation_tiers(
            values,
            keys,
            creation,
            fields=("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"),
            prefix="cache_creation.",
        )
    _append_cache_creation_tiers(
        values,
        keys,
        usage,
        fields=("claude_cache_creation_5_m_tokens", "claude_cache_creation_1_h_tokens"),
    )
    claude_usage = _claude_billing_usage(usage)
    if claude_usage is not None:
        nested_values, nested_keys = _cache_token_fields(
            claude_usage,
            ("cache_creation_input_tokens", "cache_write_tokens"),
            prefix="billing_usage.claude_usage.",
        )
        values.extend(nested_values)
        keys.extend(nested_keys)
        _append_cache_creation_tiers(
            values,
            keys,
            claude_usage,
            fields=(
                "claude_cache_creation_5_m_tokens",
                "claude_cache_creation_1_h_tokens",
            ),
            prefix="billing_usage.claude_usage.",
        )
    return (max(values, default=0), tuple(keys))


def _cache_token_fields(
    usage: Mapping[str, Any],
    fields: tuple[str, ...],
    *,
    prefix: str = "",
) -> tuple[list[int], list[str]]:
    present = tuple(field for field in fields if field in usage)
    return (
        [_token_value(usage.get(field)) for field in present],
        [f"{prefix}{field}" for field in present],
    )


def _append_cache_creation_tiers(
    values: list[int],
    keys: list[str],
    usage: Mapping[str, Any],
    *,
    fields: tuple[str, ...],
    prefix: str = "",
) -> None:
    tier_values, tier_keys = _cache_token_fields(usage, fields, prefix=prefix)
    if not tier_keys:
        return
    values.append(sum(tier_values))
    keys.extend(tier_keys)


def _claude_billing_usage(usage: Mapping[str, Any]) -> Mapping[str, Any] | None:
    billing = usage.get("billing_usage")
    if not isinstance(billing, Mapping):
        return None
    claude = billing.get("claude_usage")
    return claude if isinstance(claude, Mapping) else None


def _token_value(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _response_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value or "")


def _cache_state(value: Any) -> AIPromptCacheState:
    try:
        return AIPromptCacheState(str(value or AIPromptCacheState.UNTESTED.value))
    except ValueError:
        return AIPromptCacheState.UNTESTED


__all__ = [
    "PromptCacheObservation",
    "candidate_cache_mode",
    "is_explicit_cache_rejection",
    "normalize_prompt_cache_usage",
    "observe_prompt_cache_completion",
    "prepare_prompt_cache_request",
    "prompt_cache_config_fingerprint",
    "prompt_cache_attempt_usage",
    "prompt_cache_error_attempt_usage",
    "prompt_cache_key",
    "prompt_cache_policy_debug",
    "reject_prompt_cache_policy",
    "supports_openai_explicit_cache",
    "split_prompt_text",
]
