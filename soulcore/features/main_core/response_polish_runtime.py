"""Runtime gate and Provider call for the optional response-polish stage."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...contracts.ai_models import (
    AIExecutionMode,
    AIModelRequest,
    AIRetryPolicy,
    AIWorkPurpose,
)
from ...shared.event_log import record_event
from ..ai import record_structured_acceptance, record_structured_rejection
from ..character_context import (
    projected_dialogue_reference,
    projection_diagnostic,
    require_character_run,
)
from ..character_model import ProjectionPurpose
from .expression_timeline import has_voice_expression_metadata
from .response_polish import (
    RESPONSE_POLISH_CAPABILITY,
    RESPONSE_POLISH_TIMEOUT_SECONDS,
    ResponsePolishResult,
    _contains_file_expression,
    _context_limit,
    build_polish_prompt,
    validate_polished_expression_steps,
)
from .response_polish_context import internal_copy_reference


def polish_model_request(
    request: Any,
    *,
    run_id: int,
    prompt: Any,
    attempt_no: int = 1,
    timeout_seconds: float = RESPONSE_POLISH_TIMEOUT_SECONDS,
) -> AIModelRequest:
    attempt = max(1, int(attempt_no))
    attempt_suffix = "" if attempt == 1 else f":correction:{attempt}"
    timeout = float(timeout_seconds)
    return AIModelRequest(
        invocation_id=uuid.uuid4().hex,
        work_purpose=AIWorkPurpose.RESPONSE_POLISH,
        logical_stage_key=f"core-run:{run_id}:response-polish{attempt_suffix}",
        context_text=prompt.context_text,
        turn_text=prompt.turn_text,
        prompt_cache_hint=prompt.prompt_cache_hint,
        execution_mode=AIExecutionMode.FOREGROUND_SYNC,
        profile_id=str(request.profile_id),
        instance_id=str(request.instance_id or ""),
        owner_kind="MAIN_CORE_RESPONSE_POLISH",
        owner_id=str(run_id),
        idempotency_key=f"core-run:{run_id}:response-polish{attempt_suffix}",
        retry_policy=AIRetryPolicy(
            max_attempts=3,
            backend_timeout_seconds=timeout,
        ),
        metadata={
            "capability": "text.completion",
            "routing_capability": RESPONSE_POLISH_CAPABILITY,
            "operation_timeout_seconds": timeout,
            "prompt_document": prompt.debug_payload(),
        },
    )


class _ResponsePolishCapabilityUnavailable(RuntimeError):
    """The exact polishing assignment stopped being authorized for this result."""


class _ResponsePolishFeatureDisabled(RuntimeError):
    """The user disabled response polishing while this turn was in flight."""


class _PolishAttemptFailure(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class _PolishAttemptResult:
    visible_steps: tuple[dict[str, Any], ...]
    invocation: Any
    projection: Any
    dropped_dialogue_messages: int


def _feature_disabled_result(
    expressions: tuple[dict[str, Any], ...],
) -> ResponsePolishResult:
    return ResponsePolishResult(
        expressions,
        {"status": "SKIPPED", "reason": "FEATURE_DISABLED"},
    )


def _freeze_visible_topology(
    expressions: tuple[dict[str, Any], ...],
    requested: bool,
) -> bool:
    return (
        requested
        or any(str(item.get("memo") or "").strip() for item in expressions)
        or has_voice_expression_metadata(expressions)
    )


class ResponsePolishMixin:
    async def _response_polish_feature_enabled(self, request: Any) -> bool:
        return bool(
            await self.profiles.get_profile_response_polish_enabled(str(request.profile_id))
        )

    async def _response_polish_timeout_seconds(self, request: Any) -> float:
        return float(
            await self.profiles.get_profile_response_polish_timeout_seconds(str(request.profile_id))
        )

    async def _response_polish_capability_available(
        self,
        request: Any,
        *,
        preferred_backend_id: str = "",
    ) -> bool:
        return bool(
            await self.model_gateway.is_capability_available(
                RESPONSE_POLISH_CAPABILITY,
                str(request.profile_id),
                preferred_backend_id=str(preferred_backend_id or ""),
            )
        )

    async def _polish_expression_steps(
        self,
        request: Any,
        *,
        role: Any,
        state: Any,
        run_id: int,
        prepared: Any,
        original_steps: Sequence[Mapping[str, Any]],
        working_text: str = "",
        freeze_visible_topology: bool = False,
    ) -> ResponsePolishResult:
        expressions = tuple(dict(item) for item in original_steps)
        freeze_visible_topology = _freeze_visible_topology(
            expressions,
            freeze_visible_topology,
        )
        if not await self._response_polish_feature_enabled(request):
            return _feature_disabled_result(expressions)
        if not any(item.get("kind") == "TEXT" for item in expressions):
            return ResponsePolishResult(expressions, {"status": "SKIPPED", "reason": "NO_TEXT"})
        raw_audit = {"raw_expression_steps": [dict(item) for item in expressions]}
        if _contains_file_expression(expressions) and not (
            await self.files.get_profile_file_artifacts_enabled(request.profile_id)
        ):
            return await self._polish_fallback(
                request,
                run_id,
                expressions,
                raw_audit,
                "FILE_ARTIFACTS_DISABLED",
            )
        backend_hint = prepared.polish_backend_hint
        if backend_hint is None:
            return await self._polish_fallback(
                request, run_id, expressions, raw_audit, "MODEL_NOT_CONFIGURED"
            )
        timeout_seconds = await self._response_polish_timeout_seconds(request)
        raw_audit["timeout_seconds"] = timeout_seconds
        try:
            attempt = await self._run_polish_attempt_with_timeout(
                request,
                role=role,
                state=state,
                run_id=run_id,
                prepared=prepared,
                expressions=expressions,
                backend_hint=backend_hint,
                freeze_visible_topology=freeze_visible_topology,
                working_text=working_text,
                timeout_seconds=timeout_seconds,
            )
        except _ResponsePolishFeatureDisabled:
            return _feature_disabled_result(expressions)
        except _PolishAttemptFailure as exc:
            return await self._polish_fallback(
                request,
                run_id,
                expressions,
                raw_audit,
                exc.reason,
            )
        audit = {
            **raw_audit,
            "status": "SUCCEEDED",
            "backend_id": str(attempt.invocation.backend_id or ""),
            "dropped_dialogue_messages": attempt.dropped_dialogue_messages,
            "thinking_complexity": str(
                getattr(
                    getattr(prepared, "thinking_policy", None),
                    "complexity",
                    "",
                )
            ),
            "character_model": projection_diagnostic(attempt.projection),
        }
        await self._record_polish_event(request, run_id, "INFO", "AI 润色完成", audit)
        return ResponsePolishResult(attempt.visible_steps, audit)

    async def _run_polish_attempt_with_timeout(
        self,
        request: Any,
        *,
        role: Any,
        state: Any,
        run_id: int,
        prepared: Any,
        expressions: tuple[dict[str, Any], ...],
        backend_hint: Any,
        freeze_visible_topology: bool,
        working_text: str,
        timeout_seconds: float,
    ) -> _PolishAttemptResult:
        try:
            async with asyncio.timeout(timeout_seconds):
                return await self._run_polish_attempt(
                    request,
                    role=role,
                    state=state,
                    run_id=run_id,
                    prepared=prepared,
                    expressions=expressions,
                    backend_hint=backend_hint,
                    invoker=self.model_gateway.invoke_model,
                    freeze_visible_topology=freeze_visible_topology,
                    working_text=working_text,
                    timeout_seconds=timeout_seconds,
                )
        except (asyncio.CancelledError, _ResponsePolishFeatureDisabled):
            raise
        except TimeoutError as exc:
            raise _PolishAttemptFailure("TIMEOUT") from exc
        except Exception as exc:
            reason = f"{type(exc).__name__}:{str(exc)[:160]}"
            raise _PolishAttemptFailure(reason) from exc

    async def _run_polish_attempt(
        self,
        request: Any,
        *,
        role: Any,
        state: Any,
        run_id: int,
        prepared: Any,
        expressions: tuple[dict[str, Any], ...],
        backend_hint: Any,
        invoker: Any,
        freeze_visible_topology: bool,
        working_text: str,
        timeout_seconds: float,
        attempt_no: int = 1,
    ) -> _PolishAttemptResult:
        if not await self._response_polish_feature_enabled(request):
            raise _ResponsePolishFeatureDisabled
        if not await self._response_polish_capability_available(request):
            raise _ResponsePolishCapabilityUnavailable(
                "response_polish_capability_unavailable_before_invoke"
            )
        projection = await require_character_run(request.profile_id).project(
            ProjectionPurpose.RESPONSE_POLISH,
        )
        await self._record_character_projection(request, projection, phase="response_polish")
        dialogue_reference = projected_dialogue_reference(projection)
        prompt, _output_reserve_tokens, dropped = build_polish_prompt(
            request=request,
            persona=projection.rendered_text,
            prepared_context=prepared.prepared_context,
            expressions=expressions,
            model_id=str(backend_hint.model or ""),
            max_context_tokens=_context_limit(role, backend_hint),
            working_text=working_text,
            freeze_visible_topology=freeze_visible_topology,
            speaking_style=(projection.custom_prompts.main_core_styles.speaking_style),
            writing_correction=(projection.custom_prompts.response_polish.writing_correction),
        )
        invocation = await invoker(
            polish_model_request(
                request,
                run_id=run_id,
                prompt=prompt,
                attempt_no=attempt_no,
                timeout_seconds=timeout_seconds,
            )
        )
        if not await self._response_polish_feature_enabled(request):
            raise _ResponsePolishFeatureDisabled
        if not await self._response_polish_capability_available(
            request,
            preferred_backend_id=str(invocation.backend_id or ""),
        ):
            raise _ResponsePolishCapabilityUnavailable(
                "response_polish_capability_changed_during_invoke"
            )
        try:
            internal_reference = internal_copy_reference(prompt.document)
            polished = validate_polished_expression_steps(
                str(invocation.completion.text or ""),
                original_steps=expressions,
                collector=prepared.collector,
                dialogue_reference=dialogue_reference,
                internal_reference=internal_reference,
                reference_map=prompt.reference_map,
                identity_catalog=(
                    prepared.prepared_context.identity_catalog
                    if prepared.prepared_context is not None
                    else None
                ),
                identity_scope=(
                    str(prepared.prepared_context.identity_context.scope)
                    if prepared.prepared_context is not None
                    and prepared.prepared_context.identity_context is not None
                    else "profile"
                ),
                freeze_visible_topology=freeze_visible_topology,
            )
        except Exception as exc:
            await record_structured_rejection(
                model_gateway=self.model_gateway,
                completion=invocation,
                round_no=max(1, int(attempt_no)),
                error=str(exc),
                terminal=True,
            )
            raise
        if not await self._response_polish_feature_enabled(request):
            raise _ResponsePolishFeatureDisabled
        await record_structured_acceptance(
            model_gateway=self.model_gateway,
            completion=invocation,
            round_no=max(1, int(attempt_no)),
            value=[dict(item) for item in polished],
        )
        return _PolishAttemptResult(tuple(polished), invocation, projection, dropped)

    async def _polish_fallback(
        self,
        request: Any,
        run_id: int,
        expressions: tuple[dict[str, Any], ...],
        raw_audit: dict[str, Any],
        reason: str,
    ) -> ResponsePolishResult:
        audit = {**raw_audit, "status": "FALLBACK", "reason": str(reason)}
        await self._record_polish_event(request, run_id, "WARN", "AI 润色降级为主 Core 原文", audit)
        return ResponsePolishResult(expressions, audit)

    async def _record_polish_event(
        self,
        request: Any,
        run_id: int,
        level: str,
        message: str,
        audit: Mapping[str, Any],
    ) -> None:
        await record_event(
            self.event_log,
            profile_id=request.profile_id,
            instance_id=request.instance_id,
            level=level,
            category="response.polish",
            message=message,
            details={
                "run_id": run_id,
                "status": str(audit.get("status") or ""),
                "reason": str(audit.get("reason") or ""),
                "backend_id": str(audit.get("backend_id") or ""),
                "dropped_dialogue_messages": int(audit.get("dropped_dialogue_messages") or 0),
                "timeout_seconds": float(audit.get("timeout_seconds") or 0),
                "character_model": dict(audit.get("character_model") or {}),
            },
        )


__all__ = ["ResponsePolishMixin"]
