"""Unified AI/backend invocation policy for SoulCore's AI feature.

The manager owns one bounded logical invocation: backend selection, timeout,
retry, circuit state, command replay protection and sanitized telemetry.
Durable scheduling and business commits deliberately remain with task workers.
"""

from __future__ import annotations

import asyncio
import math
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ...contracts.ai_models import (
    DEFAULT_AI_OPERATION_TIMEOUT_SECONDS,
    AIBackendDescriptor,
    AICapabilityEffect,
    AICapabilityRequest,
    AICapabilityResult,
    AICompletion,
    AIErrorCode,
    AIErrorInfo,
    AIExecutionMode,
    AIInvocationError,
    AIInvocationResult,
    AIModelRequest,
    AIRetryPolicy,
    AIWorkPurpose,
)
from ...shared.prompt_document import compile_task_prompt
from .context_budget import configured_model_context_tokens
from .diagnostics import classify_generic_error, safe_ai_failure_details
from .invocation import AIInvocationMixin
from .invocation_support import (
    REQUEST_OPERATION_TIMEOUT_SECONDS_KEY,
    CapabilityBackendAdapter,
)
from .ports import AIRepositoryPort
from .registry import (
    BackendPool,
    BackendRegistration,
    CapabilityAdapterRegistry,
    CapabilityRegistration,
    CircuitBreaker,
    CircuitPolicy,
)
from .telemetry import AITelemetryMixin

_ROUTING_TECHNICAL_CAPABILITIES = {
    "conversation.turn_buffer": "text.completion",
    "conversation.group_interjection": "text.completion",
    "conversation.group_reply_relocation": "text.completion",
    "conversation.timer_lifecycle_review": "text.completion",
    "conversation.response_polish": "text.completion",
    "conversation.summary": "text.completion",
    "memory.reasoning": "text.completion",
    "sticker.collect": "text.completion",
    "sticker.check": "text.completion",
}

MAX_REQUEST_OPERATION_TIMEOUT_SECONDS = 24 * 60 * 60


def _technical_capability(capability: str) -> str:
    normalized = str(capability or "").strip().lower()
    return _ROUTING_TECHNICAL_CAPABILITIES.get(normalized, normalized)


class AIRoutingMixin:
    if TYPE_CHECKING:
        backends: BackendPool
        repository: AIRepositoryPort
        circuits: CircuitBreaker
        operation_timeout_seconds: float
        operation_timeout_seconds_by_capability: dict[str, float]
        _active_invocations: dict[str, asyncio.Task[Any]]
        last_persistence_error: str

        def circuit_scope(self, descriptor: AIBackendDescriptor, capability: str) -> str: ...

    def _global_retry_policy(
        self,
        policy: AIRetryPolicy,
        *,
        operation_timeout_seconds: Any = None,
    ) -> AIRetryPolicy:
        normalized = policy.normalized()
        timeout = self.operation_timeout_seconds
        if operation_timeout_seconds is not None:
            try:
                candidate = float(operation_timeout_seconds)
            except (TypeError, ValueError):
                candidate = timeout
            if math.isfinite(candidate):
                timeout = max(1.0, min(float(MAX_REQUEST_OPERATION_TIMEOUT_SECONDS), candidate))
        return replace(normalized, backend_timeout_seconds=timeout)

    def _capability_operation_timeout_seconds(
        self, capability: str, metadata: Mapping[str, Any]
    ) -> Any:
        explicit = metadata.get(REQUEST_OPERATION_TIMEOUT_SECONDS_KEY)
        if explicit is not None:
            return explicit
        return self.operation_timeout_seconds_by_capability.get(str(capability))

    def emergency_stop(self) -> int:
        current = asyncio.current_task()
        tasks = {
            task
            for task in self._active_invocations.values()
            if task is not current and not task.done()
        }
        for task in tasks:
            task.cancel()
        return len(tasks)

    async def resolve_backend_hint(
        self,
        *,
        preferred_backend_id: str = "",
        umo: str = "",
        capability: str = "text.completion",
        profile_id: str = "",
        minimum_context_tokens: int = 0,
        requires_vision: bool = False,
    ) -> AIBackendDescriptor | None:
        del umo
        requested = (str(preferred_backend_id),) if str(preferred_backend_id).strip() else ()
        registrations = [
            item
            for item in self.backends.candidates(requested)
            if self._registration_profile_matches(item, profile_id)
            and self._registration_supports(item, capability)
        ]
        registrations = await self._apply_persisted_candidate_order(
            capability, registrations, profile_id=profile_id
        )
        minimum = max(0, int(minimum_context_tokens or 0))
        if minimum:
            registrations = [
                item
                for item in registrations
                if (configured_model_context_tokens(item.descriptor) or 0) >= minimum
            ]
        if requires_vision:
            registrations = [
                item for item in registrations if self._registration_supports_vision(item)
            ]
        return registrations[0].descriptor if registrations else None

    async def _apply_persisted_candidate_order(
        self,
        capability: str,
        registrations: Sequence[Any],
        *,
        profile_id: str = "",
        strict_persistence: bool = False,
    ) -> list[Any]:
        values = list(registrations)
        # Capability pools are the business-owned order for one exact use.  The
        # API model priority is only a legacy/default order and must not flatten
        # independently configured chains back into one global sequence.
        ordered = await self._pool_candidate_order(
            capability,
            values,
            strict_persistence=strict_persistence,
        )
        if ordered is not None:
            return ordered
        ordered = await self._api_model_candidate_order(
            capability,
            values,
            profile_id=profile_id,
            strict_persistence=strict_persistence,
        )
        if ordered is not None:
            return ordered
        return sorted(
            values,
            key=lambda item: (
                int(item.descriptor.metadata.get("priority", item.descriptor.priority) or 1),
                item.descriptor.backend_id,
            ),
        )

    async def _api_model_candidate_order(
        self,
        capability: str,
        values: Sequence[Any],
        *,
        profile_id: str,
        strict_persistence: bool = False,
    ) -> list[Any] | None:
        if not str(profile_id or "").strip():
            return None
        try:
            rows = await self.repository.list_ai_api_models(profile_id=str(profile_id))
        except Exception as exc:
            self.last_persistence_error = f"list_ai_api_models: {type(exc).__name__}: {exc}"
            if strict_persistence:
                raise
            rows = []
        if not rows:
            return None
        by_id = {item.descriptor.backend_id: item for item in values}
        return self._ordered_model_rows(capability, rows, by_id)

    @staticmethod
    def _ordered_model_rows(
        capability: str, rows: Sequence[Mapping[str, Any]], by_id: dict[str, Any]
    ) -> list[Any]:
        wanted = str(capability or "").strip().lower()
        ordered: list[Any] = []
        for row in sorted(
            rows,
            key=lambda item: (int(item.get("priority") or 1), str(item.get("backend_id") or "")),
        ):
            declared = {str(item).strip().lower() for item in row.get("capabilities") or ()}
            if not bool(row.get("enabled", True)) or row.get("archived_at"):
                continue
            # A logical assignment is business authority, not an adapter
            # compatibility hint. A generic text model may transport a polish
            # request, but it may not receive one unless that exact use was
            # assigned to the model.
            if wanted not in declared:
                continue
            registration = by_id.pop(str(row.get("backend_id") or ""), None)
            if registration is not None:
                ordered.append(registration)
        return ordered

    async def _pool_candidate_order(
        self,
        capability: str,
        values: Sequence[Any],
        *,
        strict_persistence: bool = False,
    ) -> list[Any] | None:
        try:
            rows = await self.repository.list_ai_capability_pool(capability)
        except Exception as exc:
            self.last_persistence_error = f"list_ai_capability_pool: {type(exc).__name__}: {exc}"
            if strict_persistence:
                raise
            return None
        if not rows:
            return None
        by_id = {item.descriptor.backend_id: item for item in values}
        relevant = [row for row in rows if str(row.get("backend_id") or "") in by_id]
        if not relevant:
            return None
        return [
            by_id[str(row.get("backend_id") or "")]
            for row in sorted(
                (row for row in relevant if bool(row.get("enabled"))),
                key=lambda row: (
                    int(str(row.get("priority") or 1)),
                    str(row.get("backend_id") or ""),
                ),
            )
            if str(row.get("backend_id") or "") in by_id
        ]

    def force_probe(self, descriptor: AIBackendDescriptor, capability: str) -> str:
        scope = self.circuit_scope(descriptor, capability)
        self.circuits.force_half_open(scope)
        return scope

    @staticmethod
    def _registration_profile_matches(item: BackendRegistration, profile_id: str) -> bool:
        owner = str(item.descriptor.metadata.get("profile_id") or "")
        return not owner or owner == profile_id

    @staticmethod
    def _registration_supports(registration: BackendRegistration, capability: str) -> bool:
        assigned = str(capability or "").strip().lower()
        wanted = _technical_capability(assigned)
        adapter_capabilities = {
            _technical_capability(str(item)) for item in registration.adapter.capabilities
        }
        if adapter_capabilities and wanted not in adapter_capabilities:
            return False
        declared = {
            str(item).strip().lower()
            for item in registration.descriptor.metadata.get("capabilities", ())
            if str(item).strip()
        }
        return not declared or assigned in declared

    @staticmethod
    def _registration_supports_vision(registration: BackendRegistration) -> bool:
        value = registration.descriptor.metadata.get("supports_vision", True)
        return value if isinstance(value, bool) else True


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AIManager(AIRoutingMixin, AIInvocationMixin, AITelemetryMixin):
    """SoulCore-owned entry point for direct model invocations."""

    def __init__(
        self,
        backends: BackendPool | None = None,
        *,
        repository: AIRepositoryPort,
        circuit_policy: CircuitPolicy | None = None,
        credential_vault: Any | None = None,
        capability_registry: CapabilityAdapterRegistry | None = None,
        operation_timeout_seconds: float = DEFAULT_AI_OPERATION_TIMEOUT_SECONDS,
        operation_timeout_seconds_by_capability: Mapping[str, float] | None = None,
        activation_gate: Any | None = None,
    ) -> None:
        self.backends = backends or BackendPool()
        self.repository = repository
        self.circuits = CircuitBreaker(circuit_policy)
        self.credential_vault = credential_vault
        self.capabilities = capability_registry or CapabilityAdapterRegistry()
        self._workflow_by_invocation: dict[str, int] = {}
        self._node_by_invocation: dict[str, int] = {}
        self._workflow_by_stage_key: dict[tuple[str, str, str], int] = {}
        self._node_by_stage_key: dict[tuple[str, str, str], int] = {}
        self._purpose_by_node: dict[int, str] = {}
        self._owned_workflows: set[int] = set()
        self._auto_nodes: set[int] = set()
        self._managed_nodes: set[int] = set()
        self._attempt_by_invocation_round: dict[tuple[str, int], int] = {}
        self._active_invocations: dict[str, asyncio.Task[Any]] = {}
        self._scope_by_invocation: dict[str, tuple[str, str]] = {}
        self._blocked_invocation_scopes: set[tuple[str, str]] = set()
        self.runtime_backend_metadata: dict[str, Mapping[str, Any]] = {}
        self.runtime_profile_backend_metadata: dict[str, Mapping[str, Any]] = {}
        self.operation_timeout_seconds = max(1.0, float(operation_timeout_seconds))
        self.operation_timeout_seconds_by_capability: dict[str, float] = {}
        for capability, raw_timeout in dict(operation_timeout_seconds_by_capability or {}).items():
            try:
                timeout = float(raw_timeout)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(timeout):
                continue
            self.operation_timeout_seconds_by_capability[str(capability)] = max(
                1.0,
                min(float(MAX_REQUEST_OPERATION_TIMEOUT_SECONDS), timeout),
            )
        self.last_persistence_error = ""
        self.activation_gate = activation_gate

    @asynccontextmanager
    async def quiesce_instance(
        self,
        profile_id: str,
        instance_id: str,
    ) -> AsyncIterator[None]:
        """Cancel and reject model calls owned by one resetting conversation."""

        scope = (str(profile_id or "default"), str(instance_id or ""))
        self._blocked_invocation_scopes.add(scope)
        current = asyncio.current_task()
        targets = {
            task
            for invocation_id, task in tuple(self._active_invocations.items())
            if self._scope_by_invocation.get(invocation_id) == scope
            and task is not current
            and not task.done()
        }
        for task in targets:
            task.cancel("discarded_by_instance_reset")
        if targets:
            await asyncio.gather(*targets, return_exceptions=True)
        self.discard_instance_work_state(*scope)
        try:
            yield
        finally:
            self.discard_instance_work_state(*scope)
            self._blocked_invocation_scopes.discard(scope)

    async def invoke_model(self, request: AIModelRequest) -> AIInvocationResult:
        if self.activation_gate is not None:
            await self.activation_gate.wait()
        request = self._prepare_model_request(request)
        explicit_routing, routing_capability = self._routing_capabilities(request)
        technical_capability = _technical_capability(routing_capability)
        registrations = self._model_registrations(request, routing_capability)
        if not request.backend_ids:
            registrations = await self._apply_persisted_candidate_order(
                routing_capability,
                registrations,
                profile_id=str(request.profile_id or "default"),
            )
            registrations = self._prioritize_preferred_backend(
                registrations,
                str(request.metadata.get("preferred_backend_id") or ""),
            )
        technical_request = replace(
            request,
            metadata={
                **dict(request.metadata),
                "routing_capability": routing_capability,
                **({"capability": technical_capability} if explicit_routing else {}),
            },
        )
        return await self._invoke_model(technical_request, registrations)

    async def is_capability_available(
        self,
        capability: str,
        profile_id: str,
        preferred_backend_id: str = "",
    ) -> bool:
        """Return whether the exact logical model assignment is usable now."""

        normalized = str(capability or "").strip().lower()
        if not normalized:
            return False
        requested = (
            (str(preferred_backend_id).strip(),) if str(preferred_backend_id).strip() else ()
        )
        probe = AIModelRequest(
            invocation_id="capability-availability",
            work_purpose=AIWorkPurpose.MODEL_REQUEST,
            backend_ids=requested,
            profile_id=str(profile_id or "default"),
            metadata={
                "capability": _technical_capability(normalized),
                "routing_capability": normalized,
            },
        )
        registrations = self._model_registrations(
            probe,
            normalized,
        )
        if not requested:
            registrations = await self._apply_persisted_candidate_order(
                normalized,
                registrations,
                profile_id=str(profile_id or "default"),
            )
        try:
            available = await self._filter_paused_backends(probe, registrations)
        except AIInvocationError:
            return False
        return bool(available)

    async def has_capability_provider(self, capability: str, profile_id: str) -> bool:
        """Return whether one direct-capability route is usable without invoking it."""

        normalized = str(capability or "").strip().lower()
        if not normalized:
            return False
        probe = AICapabilityRequest(
            invocation_id="direct-capability-availability",
            capability=normalized,
            work_purpose=AIWorkPurpose.MODEL_REQUEST,
            effect=AICapabilityEffect.READ_ONLY,
            execution_mode=AIExecutionMode.FOREGROUND_SYNC,
            profile_id=str(profile_id or "default"),
        )
        candidates = await self._capability_candidates(probe)
        registrations = self._capability_registrations(probe, candidates)
        proxy = self._capability_proxy_request(probe)
        try:
            registrations = list(await self._filter_paused_backends(proxy, registrations))
        except AIInvocationError:
            return False
        capability_name = self._request_capability(proxy)
        return any(
            self._capability_circuits_available(item, capability_name) for item in registrations
        )

    def _capability_circuits_available(
        self,
        registration: BackendRegistration,
        capability: str,
    ) -> bool:
        descriptor = registration.descriptor
        if not self.circuits.can_attempt(self.circuit_scope(descriptor, capability)):
            return False
        package_scope = self.package_circuit_scope(descriptor)
        return not package_scope or self.circuits.can_attempt(package_scope)

    async def resolve_text_context_window(
        self,
        *,
        profile_id: str,
        capability: str = "text.completion",
        preferred_backend_id: str = "",
        backend_id: str = "",
        minimum_context_tokens: int = 0,
    ) -> int:
        """Return the first routed candidate window able to fit required context."""

        normalized_capability = str(capability or "text.completion").strip()
        exact_backend_id = str(backend_id or "").strip()
        preferred = str(preferred_backend_id or "").strip()
        probe = AIModelRequest(
            invocation_id="context-window-probe",
            work_purpose=AIWorkPurpose.MODEL_REQUEST,
            backend_ids=(exact_backend_id,) if exact_backend_id else (),
            profile_id=str(profile_id or "default"),
            metadata={
                "capability": normalized_capability,
                "routing_capability": normalized_capability,
            },
        )
        registrations = self._model_registrations(probe, normalized_capability)
        if not exact_backend_id:
            registrations = await self._apply_persisted_candidate_order(
                normalized_capability,
                registrations,
                profile_id=probe.profile_id,
            )
            registrations = self._prioritize_preferred_backend(registrations, preferred)
        registrations = list(await self._filter_paused_backends(probe, registrations))
        if not registrations:
            self._raise_missing_backend()
        minimum = max(0, int(minimum_context_tokens or 0))
        windows = [
            (item.descriptor.backend_id, configured_model_context_tokens(item.descriptor))
            for item in registrations
        ]
        selected = next(
            (
                int(window)
                for _backend_id, window in windows
                if window is not None and int(window) >= minimum
            ),
            None,
        )
        if selected is not None:
            return selected
        raise AIInvocationError(
            AIErrorInfo(
                AIErrorCode.CONTEXT_BUDGET,
                "No routed text model can fit the required context",
                retryable=False,
                switch_backend=False,
                phase="select_context_model",
                details={
                    "context_error_kind": "no_model_context_window_can_fit_required_prompt",
                    "required_context_tokens": minimum,
                    "candidate_windows": tuple(windows),
                },
            )
        )

    async def generate_text(
        self,
        *,
        task_definition: str,
        task_input: str,
        output_contract: str,
        execution_record: str = "",
        profile_id: str,
        instance_id: str = "",
        capability: str = "text.completion",
        backend_id: str = "",
        preferred_backend_id: str = "",
        minimum_context_tokens: int = 0,
        owner_kind: str = "task",
        owner_id: str = "",
        idempotency_key: str = "",
        operation_timeout_seconds: float | None = None,
        execution_mode: AIExecutionMode = AIExecutionMode.BACKGROUND_DURABLE,
        work_purpose: AIWorkPurpose,
        logical_stage_key: str = "",
        managed_work_stage: bool = True,
        round_no: int = 1,
    ) -> AICompletion:
        """Run one compact single-document SoulCore task through the model transport."""

        invocation_id = uuid.uuid4().hex
        exact_backend_id = str(backend_id or "").strip()
        preferred_backend_id = str(preferred_backend_id or "").strip()
        prompt = compile_task_prompt(
            task_definition=task_definition,
            task_input=task_input,
            output_contract=output_contract,
            execution_record=execution_record,
            model_id=exact_backend_id or preferred_backend_id,
        )
        metadata: dict[str, Any] = {
            "capability": capability,
            "routing_capability": capability,
            "prompt_document": prompt.debug_payload(),
            "round": max(1, int(round_no)),
        }
        if operation_timeout_seconds is not None:
            metadata[REQUEST_OPERATION_TIMEOUT_SECONDS_KEY] = float(operation_timeout_seconds)
        if preferred_backend_id and not exact_backend_id:
            metadata["preferred_backend_id"] = preferred_backend_id
        if int(minimum_context_tokens or 0) > 0:
            metadata["minimum_context_tokens"] = int(minimum_context_tokens)
        result = await self.invoke_model(
            AIModelRequest(
                invocation_id=invocation_id,
                work_purpose=work_purpose,
                logical_stage_key=logical_stage_key or idempotency_key or invocation_id,
                managed_work_stage=managed_work_stage,
                context_text=prompt.context_text,
                turn_text=prompt.turn_text,
                input_images=prompt.image_urls,
                prompt_cache_hint=prompt.prompt_cache_hint,
                backend_ids=(exact_backend_id,) if exact_backend_id else (),
                execution_mode=execution_mode,
                profile_id=profile_id,
                instance_id=instance_id,
                owner_kind=owner_kind,
                owner_id=owner_id,
                idempotency_key=idempotency_key or invocation_id,
                metadata=metadata,
            )
        )
        return replace(result.completion, invocation_id=invocation_id)

    def _prepare_model_request(self, request: AIModelRequest) -> AIModelRequest:
        return replace(
            request,
            retry_policy=self._global_retry_policy(
                request.retry_policy,
                operation_timeout_seconds=request.metadata.get(
                    REQUEST_OPERATION_TIMEOUT_SECONDS_KEY
                ),
            ),
        )

    @staticmethod
    def _routing_capabilities(request: AIModelRequest) -> tuple[str, str]:
        explicit = str(request.metadata.get("routing_capability") or "")
        return explicit, str(explicit or request.metadata.get("capability") or "text.completion")

    def _model_registrations(
        self,
        request: AIModelRequest,
        routing_capability: str,
    ) -> list[BackendRegistration]:
        profile_id = str(request.profile_id or "default")
        return [
            item
            for item in self.backends.candidates(request.backend_ids)
            if self._registration_matches_profile(item, profile_id)
            and self._registration_supports(item, routing_capability)
            and (not request.input_images or self._registration_supports_vision(item))
            and self._registration_meets_context_minimum(item, request)
        ]

    @staticmethod
    def _registration_meets_context_minimum(
        registration: BackendRegistration,
        request: AIModelRequest,
    ) -> bool:
        try:
            minimum = max(0, int(request.metadata.get("minimum_context_tokens") or 0))
        except (TypeError, ValueError):
            minimum = 0
        if not minimum:
            return True
        window = configured_model_context_tokens(registration.descriptor)
        return window is not None and window >= minimum

    @staticmethod
    def _prioritize_preferred_backend(
        registrations: Sequence[BackendRegistration], preferred_backend_id: str
    ) -> list[BackendRegistration]:
        """Keep the full eligible pool while trying the resolved hint first."""

        preferred = str(preferred_backend_id or "").strip()
        values = list(registrations)
        if not preferred:
            return values
        return sorted(
            values,
            key=lambda item: item.descriptor.backend_id != preferred,
        )

    @staticmethod
    def _registration_matches_profile(registration: Any, profile_id: str) -> bool:
        owner = str(registration.descriptor.metadata.get("profile_id") or "")
        return not owner or owner == profile_id

    async def invoke_capability(self, request: AICapabilityRequest) -> AICapabilityResult:
        if self.activation_gate is not None:
            await self.activation_gate.wait()
        request = self._prepare_capability_request(request)
        candidates = await self._capability_candidates(request)
        registrations = self._capability_registrations(request, candidates)
        proxy = self._capability_proxy_request(request)
        result = await self._invoke_model(proxy, registrations)
        return AICapabilityResult(
            invocation_id=request.invocation_id,
            capability=request.capability,
            output=result.completion.capability_output,
            backend_id=result.backend_id,
            attempts=result.attempts,
            started_at=result.started_at,
            finished_at=result.finished_at,
        )

    def _prepare_capability_request(self, request: AICapabilityRequest) -> AICapabilityRequest:
        capability_timeout = self._capability_operation_timeout_seconds(
            request.capability, request.metadata
        )
        return replace(
            request,
            retry_policy=self._global_retry_policy(
                request.retry_policy,
                operation_timeout_seconds=capability_timeout,
            ),
            metadata={
                **dict(request.metadata),
                **(
                    {REQUEST_OPERATION_TIMEOUT_SECONDS_KEY: capability_timeout}
                    if capability_timeout is not None
                    else {}
                ),
            },
        )

    async def _capability_candidates(
        self, request: AICapabilityRequest
    ) -> list[CapabilityRegistration]:
        candidates = self.capabilities.candidates(request.capability, request.backend_ids)
        candidates = [
            item
            for item in candidates
            if self._registration_matches_profile(item, str(request.profile_id or "default"))
        ]
        if not request.backend_ids:
            candidates = await self._apply_persisted_candidate_order(
                request.capability,
                candidates,
                profile_id=str(request.profile_id or "default"),
                strict_persistence=True,
            )
        candidates = self._prioritize_reference_candidates(request, candidates)
        maximum_backend_candidates = int(request.metadata.get("maximum_backend_candidates") or 0)
        if maximum_backend_candidates > 0:
            candidates = candidates[: max(1, maximum_backend_candidates)]
        return list(candidates)

    @staticmethod
    def _prioritize_reference_candidates(
        request: AICapabilityRequest,
        candidates: Sequence[CapabilityRegistration],
    ) -> list[CapabilityRegistration]:
        references = tuple(request.payload.get("references") or ())
        if request.capability != "image.generate" or not references:
            return list(candidates)
        raw: list[CapabilityRegistration] = []
        fallback: list[CapabilityRegistration] = []
        incompatible: list[CapabilityRegistration] = []
        count = int(request.payload.get("count") or 1)
        for candidate in candidates:
            features = candidate.adapter.image_features
            if features is not None and count > max(1, min(5, int(features.maximum_outputs))):
                incompatible.append(candidate)
                continue
            supports_raw = bool(
                features is not None
                and features.reference_image
                and (len(references) <= 1 or features.multiple_references)
            )
            (raw if supports_raw else fallback).append(candidate)
        return raw + fallback + incompatible

    def _capability_registrations(
        self,
        request: AICapabilityRequest,
        candidates: Sequence[CapabilityRegistration],
    ) -> list[BackendRegistration]:
        registrations: list[BackendRegistration] = []
        for candidate in candidates:
            adapted = self._adapt_capability_candidate(request, candidate)
            if adapted is None:
                continue
            candidate_request, reference_degraded = adapted
            registrations.append(
                BackendRegistration(
                    candidate.descriptor,
                    CapabilityBackendAdapter(
                        candidate,
                        request_override=candidate_request,
                        reference_degraded=reference_degraded,
                    ),
                )
            )
        return registrations

    @staticmethod
    def _adapt_capability_candidate(
        request: AICapabilityRequest,
        candidate: CapabilityRegistration,
    ) -> tuple[AICapabilityRequest, bool] | None:
        if request.capability != "image.generate":
            return request, False
        features = candidate.adapter.image_features
        references = tuple(request.payload.get("references") or ())
        require_raw = bool(request.metadata.get("require_raw_references"))
        if features is None:
            return None if references and require_raw else (request, False)
        count = int(request.payload.get("count") or 1)
        maximum = max(1, min(5, int(features.maximum_outputs)))
        if count > maximum:
            return None
        supports = features.reference_image and (
            len(references) <= 1 or features.multiple_references
        )
        if not references or supports:
            return request, False
        if require_raw:
            return None
        return (
            replace(
                request,
                payload={
                    **dict(request.payload),
                    "references": (),
                    "reference_mode": "text",
                },
            ),
            True,
        )

    @staticmethod
    def _capability_proxy_request(
        request: AICapabilityRequest,
    ) -> AIModelRequest:
        policy = request.retry_policy
        if (
            request.effect is AICapabilityEffect.NON_IDEMPOTENT_WRITE
            and not request.idempotency_key
        ):
            policy = replace(policy, max_attempts=1)
        return AIModelRequest(
            invocation_id=request.invocation_id,
            work_purpose=request.work_purpose,
            logical_stage_key=request.logical_stage_key,
            managed_work_stage=request.managed_work_stage,
            backend_ids=request.backend_ids,
            execution_mode=request.execution_mode,
            profile_id=request.profile_id,
            instance_id=request.instance_id,
            owner_kind=request.owner_kind,
            owner_id=request.owner_id,
            idempotency_key=request.idempotency_key,
            retry_policy=policy,
            metadata={
                **dict(request.metadata),
                "capability": request.capability,
                # Web providers are retried and replaced inside the current
                # WebResearch operation. A failure must never disable the
                # provider for later Main Core, Sub Core or admin requests.
                "disable_circuit_breaker": request.capability
                in {"web.search", "web.image_search", "web.read"},
                "non_replayable": (
                    request.effect is AICapabilityEffect.NON_IDEMPOTENT_WRITE
                    and not request.idempotency_key
                ),
                # A stable SoulCore idempotency key does not prove that an
                # external image/API provider honours it.  A timed-out write
                # may already have completed and must be reconciled instead of
                # being blindly replayed on another backend.
                "non_replayable_on_unknown_failure": (
                    request.effect is AICapabilityEffect.NON_IDEMPOTENT_WRITE
                ),
            },
            capability_request=request,
        )

    async def submit_task(
        self,
        *,
        profile_id: str,
        task_type: str,
        capability: str,
        payload: Mapping[str, Any],
        instance_id: str,
        backend_id: str | None = None,
        priority: int = 0,
        idempotency_key: str | None = None,
        recovery_policy: str = "RESTART_SAFE",
        max_attempts: int = 3,
    ) -> Any:
        callback = self._repository_method("create_ai_task")
        return await callback(
            profile_id,
            task_type,
            instance_id=instance_id,
            task_class="BACKGROUND",
            capability=capability,
            priority=priority,
            backend_id=backend_id,
            idempotency_key=idempotency_key,
            input_data=dict(payload),
            recovery_policy=recovery_policy,
            max_attempts=max_attempts,
        )

    def _repository_method(self, name: str) -> Any:
        callbacks = {
            "create_ai_task": self.repository.create_ai_task,
            "request_pause_ai_task": self.repository.request_pause_ai_task,
            "resume_ai_task": self.repository.resume_ai_task,
            "request_cancel_ai_task": self.repository.request_cancel_ai_task,
            "manual_retry_ai_task": self.repository.manual_retry_ai_task,
        }
        try:
            return callbacks[name]
        except KeyError as exc:
            raise RuntimeError(f"unsupported AI task repository method: {name}") from exc


__all__ = [
    "AIManager",
    "BackendPool",
    "BackendRegistration",
    "CapabilityAdapterRegistry",
    "CapabilityRegistration",
    "CircuitBreaker",
    "CircuitPolicy",
    "classify_generic_error",
    "safe_ai_failure_details",
]
