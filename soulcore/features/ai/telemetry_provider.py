"""Provider exchange and model-output validation telemetry helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ...contracts.ai_models import (
    AICompletion,
    AIErrorInfo,
    AIInvocationError,
    AIModelRequest,
    AITaskStatus,
)
from .diagnostics import _safe_diagnostic_identifier
from .prompt_cache import prompt_cache_policy_debug
from .prompt_debug import (
    capability_payload_view,
    prompt_jsonable,
    prompt_response_view,
    redact_prompt_text,
)
from .work_taxonomy import normalize_work_purpose

if TYPE_CHECKING:
    from .ports import AIRepositoryPort


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _record_int(value: object) -> int:
    if isinstance(value, (int, float, str, bytes, bytearray)):
        return int(value)
    raise TypeError(f"repository identifier is not numeric: {type(value).__name__}")


@runtime_checkable
class _ProviderHTTPError(Protocol):
    status_code: int
    api_code: str
    provider_response: object


def _prompt_error_transport(error: BaseException, *, sent: bool | None = None) -> dict[str, Any]:
    info = error.info if isinstance(error, AIInvocationError) else None
    details = dict(info.details) if info is not None else {}
    provider_error = error if isinstance(error, _ProviderHTTPError) else None
    status_code = (
        info.status_code
        if info is not None
        else provider_error.status_code
        if provider_error
        else None
    )
    provider_code = (
        details.get("api_code")
        if info is not None
        else provider_error.api_code
        if provider_error
        else ""
    )
    return prompt_jsonable(
        {
            "sent": (
                bool(sent)
                if sent is not None
                else bool(status_code is not None or provider_error is not None)
            ),
            "phase": info.phase if info is not None else "transport",
            "status_code": status_code,
            "provider_error_code": provider_code,
            "error_code": info.code.value if info is not None else type(error).__name__,
            "retryable": info.retryable if info is not None else None,
            "switch_backend": info.switch_backend if info is not None else None,
        }
    )


class AIProviderTelemetryHelpers:
    if TYPE_CHECKING:
        repository: AIRepositoryPort
        _node_by_invocation: dict[str, int]
        _attempt_by_invocation_round: dict[tuple[str, int], int]
        _managed_nodes: set[int]
        _owned_workflows: set[int]

        async def _recording(
            self,
            operation: str,
            call: Callable[[], Awaitable[Any]],
        ) -> Any: ...

        async def finish_ai_work_node(self, node_id: int, **values: Any) -> None: ...

        async def finish_ai_workflow(self, workflow_id: int, **values: Any) -> None: ...

        def _forget_node(self, node_id: int) -> None: ...

        async def record_ai_work_event(self, **record: Any) -> None: ...

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
    ) -> None:
        # Actual provider attempts are recorded by the two-phase protocol below.
        del request, attempt, backend_id, model_id, started_at, status, error, completion

    async def _start_prompt_exchange(
        self,
        request: AIModelRequest,
        *,
        attempt_no: int,
        round_no: int,
        backend_id: str,
        model_id: str,
        started_at: datetime,
    ) -> int | None:
        node_id = self._node_by_invocation.get(request.invocation_id)
        if node_id is None:
            return None
        result = await self._recording(
            "start_ai_provider_attempt",
            lambda: self.repository.start_ai_provider_attempt(
                {
                    "node_id": node_id,
                    "invocation_id": str(request.invocation_id),
                    "round_no": max(1, int(round_no)),
                    "attempt_no": max(1, int(attempt_no)),
                    "backend_id": _safe_diagnostic_identifier(backend_id),
                    "model_id": _safe_diagnostic_identifier(model_id),
                    "request": None,
                    "started_at": started_at,
                }
            ),
        )
        if not isinstance(result, Mapping):
            return None
        with suppress(TypeError, ValueError):
            attempt_id = _record_int(result.get("attempt_id") or 0)
            if attempt_id > 0:
                key = (str(request.invocation_id), max(1, int(round_no)))
                self._attempt_by_invocation_round[key] = attempt_id
                while len(self._attempt_by_invocation_round) > 4096:
                    self._attempt_by_invocation_round.pop(
                        next(iter(self._attempt_by_invocation_round))
                    )
                return attempt_id
            return None
        return None

    def _provider_request_view(
        self, request: AIModelRequest, provider_request: Any, round_no: int
    ) -> dict[str, Any]:
        capability_input: Any = None
        if request.capability_request is not None:
            capability_input = capability_payload_view(
                request.capability_request.capability,
                request.capability_request.payload,
            )
        return {
            "logical_prompt": redact_prompt_text(request.logical_document),
            "context_text": redact_prompt_text(request.context_text),
            "turn_text": redact_prompt_text(request.turn_text),
            "agent_history": prompt_jsonable(request.agent_history),
            "agent_tools": prompt_jsonable(request.agent_tools),
            "capability_input": capability_input,
            "interaction": {
                "purpose": normalize_work_purpose(request.work_purpose).value,
                "owner_kind": request.owner_kind,
                "execution_mode": request.execution_mode.value,
                "profile_id": request.profile_id,
                "instance_id": request.instance_id,
                "round": max(1, int(round_no)),
            },
            "prompt_document": prompt_jsonable(request.metadata.get("prompt_document") or {}),
            "prompt_cache": prompt_jsonable(
                prompt_cache_policy_debug(
                    request.prompt_cache_policy,
                    request.prompt_cache_hint,
                )
            ),
            "provider_envelope": prompt_jsonable(provider_request),
        }

    async def _enrich_prompt_exchange(
        self,
        attempt_id: int | None,
        request: AIModelRequest,
        *,
        provider_request: Any,
        backend_id: str,
        model_id: str,
        round_no: int,
    ) -> None:
        if attempt_id is None:
            return
        await self._recording(
            "enrich_ai_provider_attempt",
            lambda: self.repository.enrich_ai_provider_attempt(
                int(attempt_id),
                backend_id=_safe_diagnostic_identifier(backend_id),
                model_id=_safe_diagnostic_identifier(model_id),
                request=self._provider_request_view(request, provider_request, round_no),
                transport={"phase": "prepared", "sent": False},
            ),
        )

    async def _mark_prompt_exchange_sent(self, attempt_id: int | None) -> None:
        if attempt_id is None:
            return
        await self._recording(
            "mark_ai_provider_attempt_sent",
            lambda: self.repository.mark_ai_provider_attempt_sent(
                int(attempt_id),
                sent_at=_utcnow(),
                transport={"phase": "request", "sent": True},
            ),
        )

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
    ) -> None:
        if exchange_id is None:
            return
        error_info = error.info if isinstance(error, AIInvocationError) else None
        await self._recording(
            "finish_ai_provider_attempt",
            lambda: self.repository.finish_ai_provider_attempt(
                int(exchange_id),
                status=str(status).upper(),
                response=(
                    {
                        **prompt_response_view(completion),
                        "provider_envelope": (
                            prompt_jsonable(provider_envelope)
                            if provider_envelope is not None
                            else None
                        ),
                    }
                    if completion is not None
                    else {
                        "provider_envelope": prompt_jsonable(
                            error.provider_response
                            if isinstance(error, _ProviderHTTPError)
                            else None
                        )
                    }
                    if error is not None
                    else None
                ),
                error_code=(
                    error_info.code.value
                    if error_info is not None
                    else type(error).__name__
                    if error is not None
                    else ""
                ),
                error_message=(
                    redact_prompt_text(f"{type(error).__name__}: {error}") if error else ""
                ),
                sent=bool(sent),
                usage=(
                    dict(completion.usage) if completion is not None else dict(cache_usage or {})
                ),
                transport=(
                    {"phase": "response", "sent": True}
                    if completion is not None
                    else _prompt_error_transport(error, sent=sent)
                    if error is not None
                    else {"sent": bool(sent)}
                ),
                finished_at=_utcnow(),
            ),
        )

    async def annotate_model_exchange(
        self,
        invocation_id: str,
        *,
        round_no: int,
        processing: Mapping[str, Any],
    ) -> None:
        normalized_round = max(1, int(round_no))
        target = await self._annotate_provider_attempt(
            str(invocation_id), normalized_round, processing
        )
        if target is None:
            return
        workflow_id, node_id = target
        await self._record_validation_events(
            workflow_id=workflow_id,
            node_id=node_id,
            round_no=normalized_round,
            processing=processing,
        )
        accepted = self._processing_accepted(processing)
        terminal_rejection = bool(processing.get("terminal_rejection"))
        if node_id in self._managed_nodes and (accepted or terminal_rejection):
            await self._finish_managed_validation(
                workflow_id,
                node_id,
                processing,
                accepted=accepted,
            )

    async def _annotate_provider_attempt(
        self,
        invocation_id: str,
        round_no: int,
        processing: Mapping[str, Any],
    ) -> tuple[int, int] | None:
        attempt_id = self._attempt_by_invocation_round.get((invocation_id, round_no))
        result = await self._recording(
            "annotate_ai_provider_attempt",
            lambda: self.repository.annotate_ai_provider_attempt(
                {
                    "attempt_id": attempt_id,
                    "invocation_id": invocation_id,
                    "round_no": round_no,
                    "evaluation": prompt_jsonable(processing),
                }
            ),
        )
        if not isinstance(result, Mapping):
            return None
        workflow_id = int(result.get("workflow_id") or 0)
        node_id = int(result.get("node_id") or 0)
        return (workflow_id, node_id) if workflow_id > 0 and node_id > 0 else None

    @staticmethod
    def _processing_accepted(processing: Mapping[str, Any]) -> bool:
        return bool(processing.get("accepted")) or (
            str(processing.get("validation_status") or "").upper() == "ACCEPTED"
        )

    async def _finish_managed_validation(
        self,
        workflow_id: int,
        node_id: int,
        processing: Mapping[str, Any],
        *,
        accepted: bool,
    ) -> None:
        status = "SUCCEEDED" if accepted else "FAILED"
        error_code = "" if accepted else "model_output_rejected_three_times"
        error_message = (
            "" if accepted else str(processing.get("rejection") or "模型输出连续未通过校验")
        )
        await self.finish_ai_work_node(
            node_id,
            status=status,
            error_code=error_code,
            error_message=error_message,
            summary="模型输出已通过业务校验" if accepted else "模型输出未通过业务校验",
        )
        if workflow_id in self._owned_workflows:
            await self.finish_ai_workflow(
                workflow_id,
                status=status,
                final_error_code=error_code,
                final_message=error_message,
            )
            self._owned_workflows.discard(workflow_id)
        self._forget_node(node_id)

    async def _record_validation_events(
        self,
        *,
        workflow_id: int,
        node_id: int,
        round_no: int,
        processing: Mapping[str, Any],
    ) -> None:
        await self._record_validation_status_event(
            workflow_id=workflow_id,
            node_id=node_id,
            round_no=round_no,
            processing=processing,
        )
        await self._record_normalization_event(
            workflow_id=workflow_id,
            node_id=node_id,
            round_no=round_no,
            processing=processing,
        )
        await self._record_cleaned_data_event(
            workflow_id=workflow_id,
            node_id=node_id,
            round_no=round_no,
            processing=processing,
        )
        await self._record_parsed_commands_event(
            workflow_id=workflow_id,
            node_id=node_id,
            round_no=round_no,
            processing=processing,
        )

    async def _record_validation_status_event(
        self,
        *,
        workflow_id: int,
        node_id: int,
        round_no: int,
        processing: Mapping[str, Any],
    ) -> None:
        rejection = str(processing.get("rejection") or "").strip()
        if rejection:
            await self.record_ai_work_event(
                workflow_id=workflow_id,
                node_id=node_id,
                event_category="VALIDATION",
                severity="ERROR" if processing.get("terminal_rejection") else "WARNING",
                code=(
                    "model_output_rejected_three_times"
                    if processing.get("terminal_rejection")
                    else "model_output_rejected"
                ),
                summary=rejection,
                details={"round": round_no},
            )
            return
        if self._processing_accepted(processing):
            await self.record_ai_work_event(
                workflow_id=workflow_id,
                node_id=node_id,
                event_category="VALIDATION",
                severity="INFO",
                code="model_output_accepted",
                summary="模型输出已通过业务校验",
                details={"round": round_no},
            )

    async def _record_normalization_event(
        self,
        *,
        workflow_id: int,
        node_id: int,
        round_no: int,
        processing: Mapping[str, Any],
    ) -> None:
        normalizations = list(processing.get("normalizations") or ())
        if not normalizations:
            return
        local_only = not processing.get("repair_kind") and not processing.get("rejection")
        await self.record_ai_work_event(
            workflow_id=workflow_id,
            node_id=node_id,
            event_category="VALIDATION",
            severity="INFO",
            code="model_output_locally_normalized",
            summary=(
                "接收端已在本地兼容模型输出，未重新请求模型"
                if local_only
                else "接收端已完成确定性的模型输出兼容"
            ),
            details={"round": round_no, "normalizations": normalizations},
        )

    async def _record_cleaned_data_event(
        self,
        *,
        workflow_id: int,
        node_id: int,
        round_no: int,
        processing: Mapping[str, Any],
    ) -> None:
        cleaned_fields = list(processing.get("cleaned_fields") or ())
        dropped_commands = max(0, int(processing.get("dropped_command_count") or 0))
        if not cleaned_fields and not dropped_commands:
            return
        await self.record_ai_work_event(
            workflow_id=workflow_id,
            node_id=node_id,
            event_category="DATA_CLEANED",
            severity="WARNING",
            code="harmless_model_data_removed",
            summary="已排除不影响主流程的异常数据",
            details={
                "round": round_no,
                "fields": cleaned_fields,
                "dropped_commands": dropped_commands,
            },
        )

    async def _record_parsed_commands_event(
        self,
        *,
        workflow_id: int,
        node_id: int,
        round_no: int,
        processing: Mapping[str, Any],
    ) -> None:
        commands = list(processing.get("parsed_commands") or ())
        if not commands:
            return
        await self.record_ai_work_event(
            workflow_id=workflow_id,
            node_id=node_id,
            event_category="COMMAND",
            severity="INFO",
            code="commands_parsed",
            summary=f"识别出 {len(commands)} 条内部动作",
            details={"round": round_no, "commands": commands},
        )


__all__ = ["AIProviderTelemetryHelpers"]
