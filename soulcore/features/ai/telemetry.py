"""Fail-open causal recording for AI workflows, nodes, and provider attempts."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

from ...contracts.ai_models import (
    AIBackendDescriptor,
    AIBackendHealth,
    AIBackendState,
    AIErrorInfo,
    AIExecutionMode,
    AIModelRequest,
    AITaskStatus,
)
from .diagnostics import _redact_diagnostic_text
from .prompt_cache import prompt_cache_policy_debug
from .prompt_debug import (
    capability_payload_view,
    prompt_jsonable,
    redact_prompt_text,
)
from .recording_health import ai_recording_health
from .telemetry_provider import AIProviderTelemetryHelpers
from .work_taxonomy import (
    AIWorkNodeRole,
    normalize_work_purpose,
    work_purpose_spec,
)
from .workflow_context import (
    AIWorkContext,
    bind_ai_work_context,
    current_ai_work_context,
)


class AIWorkflowTelemetryHelpers:
    if TYPE_CHECKING:
        _workflow_by_invocation: dict[str, int]
        _node_by_invocation: dict[str, int]
        _workflow_by_stage_key: dict[tuple[str, str, str], int]
        _node_by_stage_key: dict[tuple[str, str, str], int]
        _purpose_by_node: dict[int, str]
        _auto_nodes: set[int]
        _managed_nodes: set[int]
        _active_invocations: dict[str, asyncio.Task[Any]]
        _scope_by_invocation: dict[str, tuple[str, str]]
        _owned_workflows: set[int]

        async def _start_owned_workflow(self, request: AIModelRequest) -> AIWorkContext | None: ...

    def _bind_current_business_stage(
        self,
        invocation_id: str,
        context: AIWorkContext | None,
        purpose: str,
    ) -> bool:
        # Reuse only a business stage with the same purpose. Tool-triggered
        # Web/Image/File work must remain a visible child business stage.
        reusable = (
            context is not None
            and context.workflow_id > 0
            and context.node_id is not None
            and context.node_role == AIWorkNodeRole.BUSINESS_STAGE.value
            and str(context.purpose or "") == purpose
        )
        if not reusable or context is None or context.node_id is None:
            return False
        self._bind_invocation(invocation_id, context.workflow_id, context.node_id, purpose)
        return True

    def _bind_cached_stage(
        self,
        invocation_id: str,
        context: AIWorkContext | None,
        stage_key: tuple[str, str, str],
        purpose: str,
        *,
        managed: bool,
    ) -> bool:
        if context is not None or stage_key not in self._node_by_stage_key:
            return False
        workflow_id = self._workflow_by_stage_key[stage_key]
        node_id = self._node_by_stage_key[stage_key]
        self._bind_invocation(invocation_id, workflow_id, node_id, purpose)
        if managed:
            self._managed_nodes.add(node_id)
        return True

    async def _resolve_workflow_context(
        self, context: AIWorkContext | None, request: AIModelRequest
    ) -> AIWorkContext | None:
        return await self._start_owned_workflow(request) if context is None else context

    def _bind_invocation(
        self, invocation_id: str, workflow_id: int, node_id: int, purpose: str = ""
    ) -> None:
        self._workflow_by_invocation[str(invocation_id)] = int(workflow_id)
        self._node_by_invocation[str(invocation_id)] = int(node_id)
        if purpose:
            self._purpose_by_node[int(node_id)] = str(purpose)

    @staticmethod
    def _terminal_node_status(status: AITaskStatus) -> str | None:
        return {
            AITaskStatus.SUCCEEDED: "SUCCEEDED",
            AITaskStatus.FAILED: "FAILED",
            AITaskStatus.CANCELLED: "CANCELLED",
            AITaskStatus.RECOVERY_REQUIRED: "FAILED",
        }.get(status)

    @staticmethod
    def _terminal_error(error: AIErrorInfo | None) -> tuple[str, str]:
        if error is None:
            return "", ""
        return error.code.value, _redact_diagnostic_text(error.safe_message)

    def _clear_finished_invocation(
        self, invocation_id: str, node_id: int | None, managed_success: bool
    ) -> None:
        self._active_invocations.pop(invocation_id, None)
        self._node_by_invocation.pop(invocation_id, None)
        self._workflow_by_invocation.pop(invocation_id, None)
        self._scope_by_invocation.pop(invocation_id, None)
        if not managed_success and node_id is not None:
            self._forget_node(node_id)

    def discard_instance_work_state(self, profile_id: str, instance_id: str) -> None:
        """Forget in-memory work bindings whose durable rows are being reset."""

        scope = (str(profile_id or "default"), str(instance_id or ""))
        for invocation_id, bound_scope in tuple(self._scope_by_invocation.items()):
            if bound_scope != scope:
                continue
            self._active_invocations.pop(invocation_id, None)
            self._node_by_invocation.pop(invocation_id, None)
            self._workflow_by_invocation.pop(invocation_id, None)
            self._scope_by_invocation.pop(invocation_id, None)
        targets = [
            (key, int(node_id), int(self._workflow_by_stage_key.get(key) or 0))
            for key, node_id in tuple(self._node_by_stage_key.items())
            if key[:2] == scope
        ]
        for _key, node_id, workflow_id in targets:
            self._forget_node(node_id)
            if workflow_id > 0:
                self._owned_workflows.discard(workflow_id)

    def _forget_node(self, node_id: int) -> None:
        self._auto_nodes.discard(int(node_id))
        self._managed_nodes.discard(int(node_id))
        self._purpose_by_node.pop(int(node_id), None)
        for key, value in tuple(self._node_by_stage_key.items()):
            if int(value) == int(node_id):
                self._node_by_stage_key.pop(key, None)
                self._workflow_by_stage_key.pop(key, None)


if TYPE_CHECKING:
    from .ports import AIRepositoryPort

T = TypeVar("T")


def _record_int(value: object) -> int:
    """Decode repository identifiers without treating arbitrary objects as integers."""

    if isinstance(value, (int, float, str, bytes, bytearray)):
        return int(value)
    raise TypeError(f"repository identifier is not numeric: {type(value).__name__}")


class AITelemetryMixin(AIProviderTelemetryHelpers, AIWorkflowTelemetryHelpers):
    if TYPE_CHECKING:
        repository: AIRepositoryPort
        _workflow_by_invocation: dict[str, int]
        _node_by_invocation: dict[str, int]
        _workflow_by_stage_key: dict[tuple[str, str, str], int]
        _node_by_stage_key: dict[tuple[str, str, str], int]
        _purpose_by_node: dict[int, str]
        _owned_workflows: set[int]
        _auto_nodes: set[int]
        _managed_nodes: set[int]
        _attempt_by_invocation_round: dict[tuple[str, int], int]
        _active_invocations: dict[str, asyncio.Task[Any]]
        last_persistence_error: str

        def _request_capability(self, request: AIModelRequest) -> str: ...

    async def _record_task(
        self,
        request: AIModelRequest,
        status: AITaskStatus,
        *,
        error: AIErrorInfo | None = None,
    ) -> None:
        if status is AITaskStatus.RUNNING:
            await self._start_workflow_node(request)
            return
        await self._finish_workflow_node(request, status, error)

    async def _start_workflow_node(self, request: AIModelRequest) -> None:
        invocation_id = str(request.invocation_id)
        if invocation_id in self._node_by_invocation:
            return
        purpose = normalize_work_purpose(request.work_purpose)
        stage_key = str(request.logical_stage_key or request.invocation_id)
        stage_cache_key = (
            str(request.profile_id or "default"),
            str(request.instance_id or ""),
            stage_key,
        )
        context = current_ai_work_context()
        if self._bind_current_business_stage(invocation_id, context, purpose.value):
            return
        if self._bind_cached_stage(
            invocation_id,
            context,
            stage_cache_key,
            purpose.value,
            managed=request.managed_work_stage,
        ):
            return
        context = await self._resolve_workflow_context(context, request)
        if context is None or context.workflow_id <= 0:
            return
        await self._create_auto_node(
            request,
            invocation_id,
            stage_key,
            stage_cache_key,
            purpose.value,
            context,
        )

    async def _create_auto_node(
        self,
        request: AIModelRequest,
        invocation_id: str,
        stage_key: str,
        stage_cache_key: tuple[str, str, str],
        purpose: str,
        context: AIWorkContext,
    ) -> None:
        result = await self._recording(
            "start_ai_work_node",
            lambda: self.repository.start_ai_work_node(
                {
                    "workflow_id": context.workflow_id,
                    "parent_node_id": context.node_id,
                    "node_role": AIWorkNodeRole.BUSINESS_STAGE.value,
                    "node_kind": work_purpose_spec(purpose).kind.value,
                    "purpose": purpose,
                    "node_key": stage_key,
                    "input": self._prepared_node_input(request),
                }
            ),
        )
        if not isinstance(result, Mapping):
            return
        node_id = _record_int(result["node_id"])
        self._bind_invocation(invocation_id, context.workflow_id, node_id, purpose)
        self._workflow_by_stage_key[stage_cache_key] = context.workflow_id
        self._node_by_stage_key[stage_cache_key] = node_id
        self._auto_nodes.add(node_id)
        if request.managed_work_stage:
            self._managed_nodes.add(node_id)

    async def _start_owned_workflow(self, request: AIModelRequest) -> AIWorkContext | None:
        purpose = normalize_work_purpose(request.work_purpose)
        instance_key = str(request.instance_id or "-")
        context = await self.start_ai_workflow(
            profile_id=str(request.profile_id or "default"),
            instance_id=str(request.instance_id or ""),
            workflow_kind=(
                "CONVERSATION"
                if request.execution_mode is AIExecutionMode.FOREGROUND_SYNC
                else "BACKGROUND"
            ),
            primary_purpose=purpose.value,
            trigger_kind="MODEL_REQUEST",
            trigger_ref=str(request.owner_id or ""),
            reason=work_purpose_spec(purpose).reason,
            idempotency_key=(
                f"model-stage:{instance_key}:{request.logical_stage_key}"
                if request.logical_stage_key
                else (
                    f"model-invocation:{instance_key}:"
                    f"{request.idempotency_key or request.invocation_id}"
                )
            ),
        )
        if context is not None:
            self._owned_workflows.add(context.workflow_id)
        return context

    def _prepared_node_input(self, request: AIModelRequest) -> dict[str, object]:
        capability_payload: Any = None
        if request.capability_request is not None:
            capability_payload = capability_payload_view(
                request.capability_request.capability,
                request.capability_request.payload,
            )
        return {
            "execution_mode": request.execution_mode.value,
            "prompt_chars": len(request.logical_document),
            "input_image_count": len(request.input_images),
            "capability": self._request_capability(request),
            "logical_prompt": redact_prompt_text(request.logical_document),
            "context_text": redact_prompt_text(request.context_text),
            "turn_text": redact_prompt_text(request.turn_text),
            "prompt_document": prompt_jsonable(request.metadata.get("prompt_document") or {}),
            "prompt_cache": prompt_jsonable(
                prompt_cache_policy_debug(
                    request.prompt_cache_policy,
                    request.prompt_cache_hint,
                )
            ),
            "capability_input": capability_payload,
        }

    async def _finish_workflow_node(
        self,
        request: AIModelRequest,
        status: AITaskStatus,
        error: AIErrorInfo | None,
    ) -> None:
        invocation_id = str(request.invocation_id)
        node_id = self._node_by_invocation.get(invocation_id)
        workflow_id = self._workflow_by_invocation.get(invocation_id)
        node_status = self._terminal_node_status(status)
        if node_status is None:
            return
        managed_success = (
            node_id is not None
            and node_id in self._managed_nodes
            and status is AITaskStatus.SUCCEEDED
        )
        if node_id is not None and node_id in self._auto_nodes and not managed_success:
            await self._finish_auto_node(node_id, node_status, error)
        if workflow_id in self._owned_workflows and not managed_success:
            await self._finish_owned_workflow(int(workflow_id), node_status, error)
        self._clear_finished_invocation(invocation_id, node_id, managed_success)

    async def _finish_auto_node(self, node_id: int, status: str, error: AIErrorInfo | None) -> None:
        error_code, error_message = self._terminal_error(error)
        await self.finish_ai_work_node(
            node_id,
            status=status,
            error_code=error_code,
            error_message=error_message,
        )

    async def _finish_owned_workflow(
        self, workflow_id: int, status: str, error: AIErrorInfo | None
    ) -> None:
        error_code, error_message = self._terminal_error(error)
        await self.finish_ai_workflow(
            workflow_id,
            status=status,
            final_error_code=error_code,
            final_message=error_message,
        )
        self._owned_workflows.discard(workflow_id)

    # ------------------------------------------------------------------
    # Public work-recording service methods

    async def start_ai_workflow(self, **record: Any) -> AIWorkContext | None:
        result = await self._recording(
            "create_ai_workflow", lambda: self.repository.create_ai_workflow(record)
        )
        if not isinstance(result, Mapping):
            return None
        return AIWorkContext(_record_int(result["workflow_id"]))

    @staticmethod
    @contextmanager
    def bind_ai_workflow(context: AIWorkContext | None):
        with bind_ai_work_context(context):
            yield context

    async def finish_ai_workflow(self, workflow_id: int, **values: Any) -> None:
        if int(workflow_id) <= 0:
            return
        await self._recording(
            "finish_ai_workflow",
            lambda: self.repository.finish_ai_workflow(int(workflow_id), **values),
        )
        for key, bound_workflow_id in tuple(self._workflow_by_stage_key.items()):
            if int(bound_workflow_id) != int(workflow_id):
                continue
            node_id = self._node_by_stage_key.get(key)
            if node_id is not None:
                self._forget_node(int(node_id))
        self._owned_workflows.discard(int(workflow_id))

    async def record_ai_work_event(self, **record: Any) -> None:
        if int(record.get("workflow_id") or 0) <= 0:
            return
        await self._recording(
            "record_ai_work_event", lambda: self.repository.record_ai_work_event(record)
        )

    async def start_ai_work_node(self, **record: Any) -> AIWorkContext | None:
        result = await self._recording(
            "start_ai_work_node", lambda: self.repository.start_ai_work_node(record)
        )
        if not isinstance(result, Mapping):
            return None
        return AIWorkContext(
            _record_int(result["workflow_id"]),
            _record_int(result["node_id"]),
            str(result.get("node_role") or ""),
            str(result.get("purpose") or ""),
        )

    async def project_model_visible_message_ids(
        self,
        run_id: int,
        node_id: int,
        message_ids: tuple[int, ...],
        *,
        summary_ids: tuple[int, ...] = (),
        summary_coverage: tuple[tuple[int, int, int], ...] = (),
    ) -> Any:
        return await self.repository.project_model_visible_message_ids(
            int(run_id),
            int(node_id),
            tuple(int(value) for value in message_ids),
            summary_ids=summary_ids,
            summary_coverage=summary_coverage,
        )

    async def finish_ai_work_node(self, node_id: int, **values: Any) -> None:
        if int(node_id) <= 0:
            return
        await self._recording(
            "finish_ai_work_node",
            lambda: self.repository.finish_ai_work_node(int(node_id), **values),
        )
        self._forget_node(int(node_id))

    async def ai_work_node_context(self, invocation_id: str) -> AIWorkContext | None:
        node_id = self._node_by_invocation.get(str(invocation_id))
        workflow_id = self._workflow_by_invocation.get(str(invocation_id))
        if node_id and workflow_id:
            return AIWorkContext(
                workflow_id,
                node_id,
                AIWorkNodeRole.BUSINESS_STAGE.value,
                self._purpose_by_node.get(int(node_id)),
            )
        result = await self._recording(
            "get_ai_work_node_by_model_invocation",
            lambda: self.repository.get_ai_work_node_by_model_invocation(str(invocation_id)),
        )
        if not isinstance(result, Mapping):
            return None
        return AIWorkContext(
            _record_int(result["workflow_id"]),
            _record_int(result["node_id"]),
            str(result.get("node_role") or ""),
            str(result.get("purpose") or ""),
        )

    @staticmethod
    def _workflow_feature(request: AIModelRequest) -> str:
        return normalize_work_purpose(request.work_purpose).value

    @staticmethod
    def _workflow_reason(request: AIModelRequest) -> str:
        return work_purpose_spec(request.work_purpose).reason

    @staticmethod
    def _step_kind(request: AIModelRequest) -> str:
        return work_purpose_spec(request.work_purpose).kind.value

    # ------------------------------------------------------------------
    # Existing backend/circuit health persistence

    async def _record_health(
        self,
        health: AIBackendHealth,
        descriptor: AIBackendDescriptor,
        capability: str,
    ) -> None:
        value = asdict(health)
        for key in ("last_success_at", "last_failure_at", "updated_at"):
            if isinstance(value.get(key), datetime):
                value[key] = value[key].isoformat()
        circuit = value.get("circuit")
        if isinstance(circuit, dict):
            for key in ("opened_until", "updated_at"):
                if isinstance(circuit.get(key), datetime):
                    circuit[key] = circuit[key].isoformat()
        value["status"] = "HEALTHY" if health.state is AIBackendState.HEALTHY else "FAILED"
        value["error"] = health.last_error_code
        await self._recording(
            "record_ai_backend_health",
            lambda: self.repository.record_ai_backend_health(
                {**value, "backend_id": descriptor.backend_id}
            ),
        )
        breaker = health.circuit
        await self._recording(
            "record_ai_circuit_health",
            lambda: self.repository.record_ai_circuit_health(
                {
                    "circuit_scope": health.backend_id,
                    "backend_id": descriptor.backend_id,
                    "adapter_id": descriptor.adapter_id,
                    "credential_id": descriptor.credential_id,
                    "capability": capability,
                    "state": health.state.value,
                    "failure_count": breaker.failure_count if breaker is not None else 0,
                    "opened_until": breaker.opened_until if breaker is not None else None,
                    "last_error_code": health.last_error_code,
                    "last_success_at": health.last_success_at,
                    "last_failure_at": health.last_failure_at,
                }
            ),
        )

    async def _record_package_health(
        self,
        health: AIBackendHealth,
        descriptor: AIBackendDescriptor,
    ) -> None:
        circuit = health.circuit
        await self._recording(
            "record_ai_circuit_health",
            lambda: self.repository.record_ai_circuit_health(
                {
                    "circuit_scope": health.backend_id,
                    "backend_id": descriptor.backend_id,
                    "adapter_id": "api_package",
                    "credential_id": descriptor.credential_id,
                    "capability": "api.package",
                    "state": health.state.value,
                    "failure_count": circuit.failure_count if circuit is not None else 0,
                    "opened_until": circuit.opened_until if circuit is not None else None,
                    "last_error_code": health.last_error_code,
                    "last_success_at": health.last_success_at,
                    "last_failure_at": health.last_failure_at,
                }
            ),
        )

    async def _recording(self, operation: str, call: Callable[[], Awaitable[T]]) -> T | None:
        try:
            value = await call()
        except Exception as exc:
            self.last_persistence_error = f"{operation}: {type(exc).__name__}: {exc}"
            ai_recording_health.record_failure(operation, exc)
            return None
        ai_recording_health.record_success(operation)
        return value


__all__ = ["AITelemetryMixin"]
