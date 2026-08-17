from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar

from ...contracts.ai_models import (
    AIBackendDescriptor,
    AICompletion,
    AIErrorCode,
    AIErrorInfo,
    AIInvocationError,
    AIInvocationResult,
    AIModelRequest,
    AIPromptCachePolicy,
    AIRetryPolicy,
    AITaskStatus,
)
from .context_budget import (
    configured_model_context_tokens,
    measure_model_request_context,
)
from .diagnostics import (
    _exception_diagnostics,
    _safe_diagnostic_identifier,
)
from .invocation_support import (
    REQUEST_OPERATION_TIMEOUT_SECONDS_KEY,
)
from .invocation_support import (
    InvocationState as _InvocationState,
)
from .model_parameters import resolve_model_generation_parameters
from .ports import AIRepositoryPort
from .prompt_cache import (
    normalize_prompt_cache_usage,
    observe_prompt_cache_completion,
    prepare_prompt_cache_request,
    prompt_cache_attempt_usage,
    prompt_cache_error_attempt_usage,
    reject_prompt_cache_policy,
)
from .proxy_context_isolation import apply_proxy_context_isolation
from .registry import BackendRegistration, CircuitBreaker
from .transport_tracking import bind_transport_send_hook

if TYPE_CHECKING:
    from .ports import AIRepositoryPort

T = TypeVar("T")


@dataclass(slots=True)
class PromptCacheAttemptOutcome:
    completion: AICompletion | None = None
    error: AIErrorInfo | None = None


class PromptCacheInvocationMixin:
    if TYPE_CHECKING:
        repository: AIRepositoryPort

        def _attempt_timeout(self, request: Any) -> float: ...

        async def _recording(
            self, operation: str, call: Callable[[], Awaitable[T]]
        ) -> T | None: ...

        async def _invoke_backend(
            self,
            request: Any,
            registration: BackendRegistration,
            *,
            attempt_no: int,
            timeout: float,
        ) -> AICompletion: ...

        async def _record_cancelled_attempt(
            self, state: Any, registration: BackendRegistration, started_at: datetime
        ) -> None: ...

        def _classified_error(
            self, registration: BackendRegistration, error: Exception
        ) -> AIErrorInfo: ...

    async def _perform_attempt(
        self,
        state: Any,
        registration: BackendRegistration,
        attempt_started: datetime,
    ) -> PromptCacheAttemptOutcome:
        timeout = self._attempt_timeout(state.request)
        isolated_request = apply_proxy_context_isolation(
            state.request,
            registration.descriptor,
        )
        physical_request = isolated_request
        try:
            prepared = await self._recording(
                "claim_ai_prompt_cache_capability",
                lambda: prepare_prompt_cache_request(
                    self.repository,
                    isolated_request,
                    registration.descriptor,
                ),
            )
            physical_request = prepared or replace(
                isolated_request, prompt_cache_policy=AIPromptCachePolicy()
            )
            state.attempts += 1
            completion = await self._invoke_backend(
                physical_request,
                registration,
                attempt_no=state.attempts,
                timeout=timeout,
            )
            return PromptCacheAttemptOutcome(completion=completion)
        except asyncio.CancelledError:
            await self._record_cancelled_attempt(state, registration, attempt_started)
            raise
        except Exception as exc:
            error = self._classified_error(registration, exc)
            if error.code is not AIErrorCode.PROMPT_CACHE_MARKER_UNSUPPORTED:
                return PromptCacheAttemptOutcome(error=error)
            await self._recording(
                "reject_ai_prompt_cache_capability",
                lambda: reject_prompt_cache_policy(
                    self.repository,
                    physical_request,
                    registration.descriptor,
                    error,
                ),
            )
            return await self._unmarked_fallback(
                state,
                registration,
                attempt_started,
                physical_request,
                timeout,
            )

    async def _unmarked_fallback(
        self,
        state: Any,
        registration: BackendRegistration,
        attempt_started: datetime,
        physical_request: Any,
        timeout: float,
    ) -> PromptCacheAttemptOutcome:
        try:
            state.attempts += 1
            fallback = replace(physical_request, prompt_cache_policy=AIPromptCachePolicy())
            completion = await self._invoke_backend(
                fallback,
                registration,
                attempt_no=state.attempts,
                timeout=timeout,
            )
            return PromptCacheAttemptOutcome(completion=completion)
        except asyncio.CancelledError:
            await self._record_cancelled_attempt(state, registration, attempt_started)
            raise
        except Exception as exc:
            return PromptCacheAttemptOutcome(error=self._classified_error(registration, exc))


async def filter_paused_backends(
    repository: AIRepositoryPort,
    request: AIModelRequest,
    registrations: Sequence[BackendRegistration],
) -> Sequence[BackendRegistration]:
    pauses = await repository.list_ai_manager_pauses()
    requested_capability = _requested_capability(request)
    _reject_manager_pause(pauses, requested_capability)
    paused_backend_ids = _paused_backend_ids(pauses)
    persisted = {
        str(item.get("backend_id") or ""): bool(item.get("enabled"))
        for item in await repository.list_ai_backends()
        if str(item.get("backend_id") or "")
    }
    return [
        item
        for item in registrations
        if item.descriptor.backend_id not in paused_backend_ids
        and (
            not item.descriptor.metadata.get("package_id")
            or persisted.get(item.descriptor.backend_id, False)
        )
    ]


def _requested_capability(request: AIModelRequest) -> str:
    return (
        str(
            request.metadata.get("routing_capability")
            or request.metadata.get("capability")
            or "text.completion"
        )
        .strip()
        .lower()
    )


def _reject_manager_pause(pauses: Sequence[Mapping[str, Any]], capability: str) -> None:
    if any(_matches_pause(item, "GLOBAL") for item in pauses):
        raise AIInvocationError(
            AIErrorInfo(
                AIErrorCode.CAPACITY_BUSY,
                "All AI invocations are paused by an administrator",
                retryable=True,
                phase="manager_pause",
            )
        )
    if any(_matches_pause(item, "CAPABILITY", capability) for item in pauses):
        raise AIInvocationError(
            AIErrorInfo(
                AIErrorCode.CAPACITY_BUSY,
                f"AI capability {capability} is paused by an administrator",
                retryable=True,
                phase="manager_pause",
            )
        )


def _paused_backend_ids(pauses: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(item.get("scope_key") or "") for item in pauses if _matches_pause(item, "BACKEND")}


def _matches_pause(item: Mapping[str, Any], scope: str, key: str = "") -> bool:
    if not bool(item.get("paused")):
        return False
    if str(item.get("pause_scope") or "").upper() != scope:
        return False
    return not key or str(item.get("scope_key") or "").strip().lower() == key


_COMPLETE_TEXT_FINISH_REASONS = {
    "",
    "stop",
    "end_turn",
    "stop_sequence",
    "tool_calls",
    "tool_use",
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AIInvocationMixin(PromptCacheInvocationMixin):
    if TYPE_CHECKING:
        repository: AIRepositoryPort
        circuits: CircuitBreaker
        _active_invocations: dict[str, asyncio.Task[Any]]
        _scope_by_invocation: dict[str, tuple[str, str]]
        _blocked_invocation_scopes: set[tuple[str, str]]

        def _global_retry_policy(
            self,
            policy: AIRetryPolicy,
            *,
            operation_timeout_seconds: Any = None,
        ) -> AIRetryPolicy: ...

        async def _record_task(
            self,
            request: AIModelRequest,
            status: AITaskStatus,
            *,
            error: AIErrorInfo | None = None,
        ) -> None: ...

        async def _record_attempt(
            self,
            request: AIModelRequest,
            attempt: int,
            backend_id: str,
            model_id: str,
            started_at: datetime,
            status: AITaskStatus,
            error: AIErrorInfo | None,
            *,
            completion: AICompletion | None = None,
        ) -> None: ...

        async def _record_health(
            self,
            health: Any,
            descriptor: AIBackendDescriptor,
            capability: str,
        ) -> None: ...

        async def _record_package_health(
            self, health: Any, descriptor: AIBackendDescriptor
        ) -> None: ...

        async def _start_prompt_exchange(
            self,
            request: AIModelRequest,
            *,
            attempt_no: int,
            round_no: int,
            backend_id: str,
            model_id: str,
            started_at: datetime,
        ) -> int | None: ...

        async def _enrich_prompt_exchange(
            self,
            exchange_id: int | None,
            request: AIModelRequest,
            *,
            provider_request: Any,
            backend_id: str,
            model_id: str,
            round_no: int,
        ) -> None: ...

        async def _mark_prompt_exchange_sent(self, exchange_id: int | None) -> None: ...

        async def _finish_prompt_exchange(
            self,
            exchange_id: int | None,
            *,
            status: str,
            completion: AICompletion | None = None,
            provider_envelope: Any = None,
            error: BaseException | None = None,
            sent: bool | None = None,
            cache_usage: Mapping[str, Any] | None = None,
        ) -> None: ...

    async def _invoke_model(
        self,
        request: AIModelRequest,
        registrations: Sequence[BackendRegistration],
    ) -> AIInvocationResult:
        request = self._prepare_invocation_request(request)
        if not request.invocation_id:
            raise ValueError("invocation_id is required")
        scope = (str(request.profile_id or "default"), str(request.instance_id or ""))
        if scope in self._blocked_invocation_scopes:
            raise asyncio.CancelledError("discarded_by_instance_reset")
        current_task = asyncio.current_task()
        if current_task is not None:
            self._active_invocations[request.invocation_id] = current_task
            self._scope_by_invocation[request.invocation_id] = scope
        try:
            await self._record_task(request, AITaskStatus.RUNNING)
            try:
                registrations = await self._filter_paused_backends(request, registrations)
                if not registrations:
                    self._raise_missing_backend()
            except AIInvocationError as exc:
                await self._record_task(request, AITaskStatus.FAILED, error=exc.info)
                raise
            state = _InvocationState(
                request=request,
                started_at=_utcnow(),
                capability=self._request_capability(request),
                circuit_enabled=not bool(request.metadata.get("disable_circuit_breaker")),
            )
            for registration in registrations:
                if state.stop_all:
                    break
                result = await self._try_registration(state, registration)
                if result is not None:
                    return result
            if state.last_error is None:
                state.last_error = AIErrorInfo(
                    AIErrorCode.CIRCUIT_OPEN,
                    "All matching AI backends currently have open circuits",
                    retryable=True,
                    phase="select_backend",
                )
            terminal_status = (
                AITaskStatus.RECOVERY_REQUIRED
                if bool(state.last_error.details.get("recovery_required"))
                else AITaskStatus.FAILED
            )
            await self._record_task(request, terminal_status, error=state.last_error)
            raise AIInvocationError(state.last_error)
        except asyncio.CancelledError:
            await self._record_task(request, AITaskStatus.CANCELLED)
            raise
        finally:
            if self._active_invocations.get(request.invocation_id) is current_task:
                self._active_invocations.pop(request.invocation_id, None)
            self._scope_by_invocation.pop(request.invocation_id, None)

    def _prepare_invocation_request(self, request: AIModelRequest) -> AIModelRequest:
        return replace(
            request,
            retry_policy=self._global_retry_policy(
                request.retry_policy,
                operation_timeout_seconds=request.metadata.get(
                    REQUEST_OPERATION_TIMEOUT_SECONDS_KEY
                ),
            ),
        )

    async def _filter_paused_backends(
        self,
        request: AIModelRequest,
        registrations: Sequence[BackendRegistration],
    ) -> Sequence[BackendRegistration]:
        return await filter_paused_backends(self.repository, request, registrations)

    @staticmethod
    def _raise_missing_backend() -> None:
        raise AIInvocationError(
            AIErrorInfo(
                AIErrorCode.BACKEND_NOT_FOUND,
                "No enabled AI backend matches this request",
                phase="select_backend",
            )
        )

    async def _try_registration(
        self, state: _InvocationState, registration: BackendRegistration
    ) -> AIInvocationResult | None:
        scopes = self._allowed_circuit_scopes(state, registration)
        if scopes is None:
            return None
        circuit_scope, package_scope = scopes
        try:
            policy = state.request.retry_policy.normalized()
            for backend_attempt in range(1, policy.max_attempts + 1):
                attempt_started = _utcnow()
                outcome = await self._perform_attempt(state, registration, attempt_started)
                if outcome.completion is not None:
                    return await self._finish_success(
                        state,
                        registration,
                        outcome.completion,
                        attempt_started,
                        circuit_scope,
                        package_scope,
                    )
                state.last_error = self._prepare_failure(state, registration, outcome.error)
                await self._finish_failure(
                    state,
                    registration,
                    attempt_started,
                    circuit_scope,
                    package_scope,
                )
                if self._must_leave_backend(state.last_error):
                    state.stop_all = not state.last_error.switch_backend
                    break
                if self._may_retry_same_backend(
                    state.last_error, backend_attempt, policy.max_attempts
                ):
                    continue
                state.stop_all = not state.last_error.switch_backend
                break
            return None
        finally:
            if state.circuit_enabled:
                self.circuits.release_probe(circuit_scope)
                if package_scope:
                    self.circuits.release_probe(package_scope)

    def _allowed_circuit_scopes(
        self, state: _InvocationState, registration: BackendRegistration
    ) -> tuple[str, str] | None:
        circuit_scope = self.circuit_scope(registration.descriptor, state.capability)
        package_scope = self.package_circuit_scope(registration.descriptor)
        if not state.circuit_enabled:
            return circuit_scope, package_scope
        if not self.circuits.allow(circuit_scope):
            return None
        if package_scope and not self.circuits.allow(package_scope):
            self.circuits.release_probe(circuit_scope)
            return None
        return circuit_scope, package_scope

    @staticmethod
    def _attempt_timeout(request: AIModelRequest) -> float:
        return request.retry_policy.normalized().backend_timeout_seconds

    async def _invoke_backend(
        self,
        request: AIModelRequest,
        registration: BackendRegistration,
        *,
        attempt_no: int,
        timeout: float,
    ) -> AICompletion:
        backend_request = replace(
            request,
            parameters=resolve_model_generation_parameters(
                request.parameters,
                registration.descriptor.metadata,
            ),
        )
        started_at = _utcnow()
        round_no = max(1, int(request.metadata.get("round") or 1))
        record_id = await self._start_prompt_exchange(
            backend_request,
            attempt_no=attempt_no,
            round_no=round_no,
            backend_id=registration.descriptor.backend_id,
            model_id=registration.descriptor.model,
            started_at=started_at,
        )
        sent = False

        async def mark_sent(provider_request: Any) -> None:
            nonlocal sent
            await self._enrich_prompt_exchange(
                record_id,
                backend_request,
                provider_request=provider_request,
                backend_id=registration.descriptor.backend_id,
                model_id=registration.descriptor.model,
                round_no=round_no,
            )
            await self._mark_prompt_exchange_sent(record_id)
            sent = True

        try:
            self._validate_request_budget(backend_request, registration.descriptor)
            with bind_transport_send_hook(mark_sent):
                async with asyncio.timeout(timeout):
                    response = await registration.adapter.complete(
                        backend_request, registration.descriptor
                    )
        except TimeoutError as exc:
            timeout_error = AIInvocationError(self._timeout_error(request, registration, timeout))
            await self._finish_prompt_exchange(
                record_id,
                status="FAILED",
                error=timeout_error,
                sent=sent,
                cache_usage=prompt_cache_attempt_usage(
                    backend_request.prompt_cache_policy, "ERROR"
                ),
            )
            raise timeout_error from exc
        except asyncio.CancelledError as exc:
            await self._finish_prompt_exchange(
                record_id,
                status="CANCELLED",
                error=exc,
                sent=sent,
                cache_usage=prompt_cache_attempt_usage(
                    backend_request.prompt_cache_policy, "CANCELLED"
                ),
            )
            raise
        except Exception as exc:
            await self._finish_prompt_exchange(
                record_id,
                status="FAILED",
                error=exc,
                sent=sent,
                cache_usage=prompt_cache_error_attempt_usage(
                    backend_request.prompt_cache_policy, exc
                ),
            )
            raise
        completion = await self._validated_backend_completion(
            backend_request,
            registration,
            response.completion,
            record_id=record_id,
        )
        await self._finish_prompt_exchange(
            record_id,
            status="SUCCEEDED",
            completion=completion,
            provider_envelope=response.provider_envelope,
            sent=True,
        )
        return completion

    async def _validated_backend_completion(
        self,
        request: AIModelRequest,
        registration: BackendRegistration,
        completion: AICompletion,
        *,
        record_id: int | None,
    ) -> AICompletion:
        completion = await self._observe_prompt_cache_response(
            request, registration.descriptor, completion
        )
        finish_reason = str(completion.finish_reason or "").strip().casefold()
        if (
            completion.capability_output is None
            and finish_reason not in _COMPLETE_TEXT_FINISH_REASONS
        ):
            error = AIInvocationError(
                AIErrorInfo(
                    AIErrorCode.OUTPUT_CONTRACT,
                    "AI backend returned an incomplete or filtered completion",
                    retryable=False,
                    switch_backend=True,
                    backend_id=registration.descriptor.backend_id,
                    phase="response",
                    details={"finish_reason": finish_reason[:80]},
                )
            )
            await self._finish_prompt_exchange(
                record_id,
                status="FAILED",
                error=error,
                sent=True,
                cache_usage=dict(completion.usage),
            )
            raise error
        if (
            not str(completion.text or "").strip()
            and not completion.agent_output_items
            and completion.capability_output is None
        ):
            error = AIInvocationError(
                AIErrorInfo(
                    AIErrorCode.EMPTY_OUTPUT,
                    "AI backend returned an empty completion",
                    retryable=True,
                    switch_backend=True,
                    backend_id=registration.descriptor.backend_id,
                    phase="response",
                )
            )
            await self._finish_prompt_exchange(
                record_id,
                status="FAILED",
                error=error,
                sent=True,
                cache_usage=dict(completion.usage),
            )
            raise error
        return completion

    async def _observe_prompt_cache_response(
        self,
        request: AIModelRequest,
        descriptor: AIBackendDescriptor,
        completion: AICompletion,
    ) -> AICompletion:
        observed = await self._recording(
            "observe_ai_prompt_cache_capability",
            lambda: observe_prompt_cache_completion(
                self.repository, request, descriptor, completion
            ),
        )
        return observed or normalize_prompt_cache_usage(completion, request.prompt_cache_policy)[0]

    @staticmethod
    def _validate_request_budget(
        request: AIModelRequest,
        descriptor: AIBackendDescriptor,
    ) -> None:
        maximum = configured_model_context_tokens(descriptor)
        if maximum is None or request.capability_request is not None:
            return
        requirement = measure_model_request_context(
            request,
            model_id=descriptor.model,
        )
        if requirement.total_tokens <= maximum:
            return
        raise AIInvocationError(
            AIErrorInfo(
                AIErrorCode.CONTEXT_BUDGET,
                "SoulCore request exceeds this model's configured context window",
                switch_backend=True,
                backend_id=descriptor.backend_id,
                phase="prepare",
                details={
                    "input_tokens": requirement.input_text_tokens + requirement.input_image_tokens,
                    "input_text_tokens": requirement.input_text_tokens,
                    "input_image_tokens": requirement.input_image_tokens,
                    "reserved_output_tokens": requirement.reserved_output_tokens,
                    "required_context_tokens": requirement.total_tokens,
                    "max_context_tokens": maximum,
                },
            )
        )

    async def _record_cancelled_attempt(
        self,
        state: _InvocationState,
        registration: BackendRegistration,
        attempt_started: datetime,
    ) -> None:
        error = AIErrorInfo(
            AIErrorCode.INTERNAL,
            "Invocation was cancelled",
            details={
                "exception_type": "CancelledError",
                "cause_chain": ["CancelledError"],
            },
        )
        await self._record_attempt(
            state.request,
            state.attempts,
            registration.descriptor.backend_id,
            registration.descriptor.model,
            attempt_started,
            AITaskStatus.CANCELLED,
            error,
        )
        await self._record_task(state.request, AITaskStatus.CANCELLED)

    @staticmethod
    def _timeout_error(
        request: AIModelRequest,
        registration: BackendRegistration,
        timeout: float,
    ) -> AIErrorInfo:
        message = f"AI backend exceeded the effective {timeout:g} second operation deadline"
        return AIErrorInfo(
            AIErrorCode.TIMEOUT,
            message,
            retryable=True,
            switch_backend=True,
            backend_id=registration.descriptor.backend_id,
            phase="timeout",
            details={
                "exception_type": "TimeoutError",
                "cause_chain": ["TimeoutError"],
            },
        )

    @staticmethod
    def _classified_error(
        registration: BackendRegistration,
        exc: BaseException,
    ) -> AIErrorInfo:
        error = (
            exc.info
            if isinstance(exc, AIInvocationError)
            else registration.adapter.classify_error(exc, registration.descriptor)
        )
        error = replace(
            error,
            details={**dict(error.details), **_exception_diagnostics(exc)},
        )
        return error

    def _prepare_failure(
        self,
        state: _InvocationState,
        registration: BackendRegistration,
        error: AIErrorInfo | None,
    ) -> AIErrorInfo:
        if error is None:
            error = AIErrorInfo(AIErrorCode.INTERNAL, "Unknown invocation failure")
        current = replace(error, backend_id=registration.descriptor.backend_id)
        current = self._apply_replay_policy(state.request, current)
        diagnostic = self._attempt_diagnostic(state, registration, current)
        state.failure_attempts.append(diagnostic)
        return replace(
            current,
            details={
                **dict(current.details),
                "model_id": diagnostic["model_id"],
                "attempt_no": state.attempts,
                "attempts": tuple(state.failure_attempts[-10:]),
            },
        )

    @staticmethod
    def _apply_replay_policy(request: AIModelRequest, error: AIErrorInfo) -> AIErrorInfo:
        accepted_non_idempotent_image_response = (
            str(request.metadata.get("capability") or "") == "image.generate"
            and bool(request.metadata.get("non_replayable_on_unknown_failure"))
            and bool(error.details.get("provider_response_accepted"))
            and error.phase == "response"
        )
        if accepted_non_idempotent_image_response:
            error = replace(
                error,
                retryable=False,
                switch_backend=False,
                details={
                    **dict(error.details),
                    "external_side_effect_unknown": True,
                    "recovery_required": True,
                },
            )
        unknown_codes = {
            AIErrorCode.TIMEOUT,
            AIErrorCode.NETWORK,
            AIErrorCode.REMOTE_5XX,
        }
        unknown = error.code in unknown_codes and bool(
            request.metadata.get("non_replayable_on_unknown_failure")
        )
        if unknown:
            allow_switch = bool(request.metadata.get("allow_unknown_effect_backend_switch"))
            marker = "possible_duplicate_billing" if allow_switch else "recovery_required"
            error = replace(
                error,
                retryable=False,
                switch_backend=allow_switch,
                details={
                    **dict(error.details),
                    "external_side_effect_unknown": True,
                    marker: True,
                },
            )
        if bool(request.metadata.get("non_replayable")):
            error = replace(
                error,
                retryable=False,
                switch_backend=False,
                details={**dict(error.details), "non_replayable": True},
            )
        return error

    @staticmethod
    def _attempt_diagnostic(
        state: _InvocationState,
        registration: BackendRegistration,
        error: AIErrorInfo,
    ) -> dict[str, Any]:
        return {
            "attempt_no": state.attempts,
            "backend_id": _safe_diagnostic_identifier(registration.descriptor.backend_id),
            "model_id": _safe_diagnostic_identifier(registration.descriptor.model),
            "error_code": error.code.value,
            "phase": _safe_diagnostic_identifier(error.phase, limit=80),
            "exception_type": _safe_diagnostic_identifier(
                error.details.get("exception_type"), limit=100
            ),
            "cause_type": _safe_diagnostic_identifier(error.details.get("cause_type"), limit=100),
            "http_status": error.status_code,
            "provider_error_code": _safe_diagnostic_identifier(
                error.details.get("api_code"), limit=160
            ),
            "decision": ("retry_or_switch" if error.retryable or error.switch_backend else "stop"),
        }

    async def _finish_success(
        self,
        state: _InvocationState,
        registration: BackendRegistration,
        completion: AICompletion,
        attempt_started: datetime,
        circuit_scope: str,
        package_scope: str,
    ) -> AIInvocationResult:
        if state.circuit_enabled:
            health = self.circuits.record_success(circuit_scope)
            await self._record_health(health, registration.descriptor, state.capability)
            if package_scope:
                package_health = self.circuits.record_success(package_scope)
                await self._record_package_health(package_health, registration.descriptor)
        await self._record_attempt(
            state.request,
            state.attempts,
            registration.descriptor.backend_id,
            registration.descriptor.model,
            attempt_started,
            AITaskStatus.SUCCEEDED,
            None,
            completion=completion,
        )
        await self._record_task(state.request, AITaskStatus.SUCCEEDED)
        return AIInvocationResult(
            invocation_id=state.request.invocation_id,
            completion=completion,
            backend_id=registration.descriptor.backend_id,
            attempts=state.attempts,
            started_at=state.started_at,
            finished_at=_utcnow(),
        )

    async def _finish_failure(
        self,
        state: _InvocationState,
        registration: BackendRegistration,
        attempt_started: datetime,
        circuit_scope: str,
        package_scope: str,
    ) -> None:
        error = state.last_error
        if error is None:
            return
        if state.circuit_enabled and self._affects_backend_health(error):
            health = self.circuits.record_failure(circuit_scope, error)
            await self._record_health(health, registration.descriptor, state.capability)
            package_failure = package_scope and error.code in {
                AIErrorCode.AUTHENTICATION,
                AIErrorCode.PERMISSION,
                AIErrorCode.QUOTA_EXHAUSTED,
            }
            if package_failure:
                health = self.circuits.record_failure(
                    package_scope, replace(error, open_circuit=True)
                )
                await self._record_package_health(health, registration.descriptor)
        status = (
            AITaskStatus.RECOVERY_REQUIRED
            if error.details.get("recovery_required")
            else AITaskStatus.FAILED
        )
        await self._record_attempt(
            state.request,
            state.attempts,
            registration.descriptor.backend_id,
            registration.descriptor.model,
            attempt_started,
            status,
            error,
        )

    @staticmethod
    def _affects_backend_health(error: AIErrorInfo) -> bool:
        return error.code not in {
            AIErrorCode.INVALID_REQUEST,
            AIErrorCode.CONTEXT_BUDGET,
            AIErrorCode.OUTPUT_CONTRACT,
            AIErrorCode.COMMAND_PROTOCOL,
            AIErrorCode.COMMAND_FAILED,
            AIErrorCode.PROMPT_CACHE_MARKER_UNSUPPORTED,
        }

    @staticmethod
    def _must_leave_backend(error: AIErrorInfo) -> bool:
        return error.code in {
            AIErrorCode.TIMEOUT,
            AIErrorCode.AUTHENTICATION,
            AIErrorCode.PERMISSION,
            AIErrorCode.QUOTA_EXHAUSTED,
            AIErrorCode.RATE_LIMIT,
        }

    @staticmethod
    def _may_retry_same_backend(
        error: AIErrorInfo, backend_attempt: int, max_attempts: int
    ) -> bool:
        return (
            error.retryable
            and error.code
            in {
                AIErrorCode.NETWORK,
                AIErrorCode.REMOTE_5XX,
                AIErrorCode.EMPTY_OUTPUT,
            }
            and backend_attempt < max_attempts
        )

    @staticmethod
    def _request_capability(request: AIModelRequest) -> str:
        return str(request.metadata.get("capability") or "text.completion")

    @staticmethod
    def circuit_scope(descriptor: AIBackendDescriptor, capability: str) -> str:
        return "|".join(
            (
                descriptor.adapter_id,
                descriptor.backend_id,
                descriptor.credential_id or "-",
                str(capability or "-"),
            )
        )

    @staticmethod
    def package_circuit_scope(descriptor: AIBackendDescriptor) -> str:
        package_id = str(descriptor.metadata.get("package_id") or "").strip()
        if not package_id:
            return ""
        return "|".join(("api_package", package_id, descriptor.credential_id or "-"))

    def force_package_probe(self, descriptor: AIBackendDescriptor) -> str:
        scope = self.package_circuit_scope(descriptor)
        if scope:
            self.circuits.force_half_open(scope)
        return scope


__all__ = ["AIInvocationMixin"]
