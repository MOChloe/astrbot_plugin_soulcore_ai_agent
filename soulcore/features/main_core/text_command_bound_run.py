"Per-round orchestration helpers for the Main Core text command loop."

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any

from ...contracts.ai_models import AIAgentTranscriptTurn, AICompletion, AIInvocationError
from ...contracts.models import CoreWakeRequest
from ...shared.model_image_preview import (
    STRICT_MODEL_IMAGE_PREVIEW_TOTAL_BYTES,
    rebudget_model_image_data_uris,
)
from ..ai import current_ai_work_context
from ..ai.service import MainCoreCommandRegistry, ParsedCommand, ParsedModelTurn, parse_model_turn
from ..identity import decode_model_parameter_map
from .agent_protocol import parse_agent_response, transcript_turn
from .command_context import CollectorScope
from .roleplay_prompt_contracts import ExecutionRound


def _normalizations(round_item: ExecutionRound, parsed: Any) -> list[dict[str, Any]]:
    del round_item, parsed
    return []


def _cleaned_fields(parsed: Any, registry: MainCoreCommandRegistry) -> list[dict[str, str]]:
    return [
        {"command": item.name, "parameter": label}
        for item in parsed.commands
        if (spec := registry.get(item.name)) is not None and not spec.terminal
        for label in item.parameters
        if label not in {parameter.label for parameter in spec.parameters}
    ]


def _parsed_command_views(
    parsed: Any,
    registry: MainCoreCommandRegistry,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in parsed.commands:
        parameters = dict(item.parameters)
        spec = registry.get(item.name)
        if spec is not None and spec.body_parameter:
            parameters[spec.body_parameter] = str(item.unlabeled_content or "").strip()
        result.append(
            {
                "ordinal": item.ordinal,
                "name": item.name,
                "parameters": parameters,
            }
        )
    return result


def _result_views(round_item: ExecutionRound) -> list[dict[str, Any]]:
    return [
        {
            "ordinal": item.ordinal,
            "command": item.command_name,
            "ok": item.ok,
            "summary": item.content,
        }
        for item in round_item.results
    ]


def processing_view(
    round_item: ExecutionRound | None,
    *,
    registry: MainCoreCommandRegistry,
    completed: bool,
) -> dict[str, Any]:
    if round_item is None:
        return {
            "captured": False,
            "summary": "本次记录未采集解析与执行结果。",
            "terminal": bool(completed),
            "accepted": False,
            "validation_status": "MISSING",
        }
    parsed = parse_model_turn(round_item.payload_text or round_item.raw_text)
    normalizations = _normalizations(round_item, parsed)
    cleaned_fields = _cleaned_fields(parsed, registry)
    return {
        "captured": True,
        "model_visible_text": round_item.raw_text,
        "model_visible_items": [
            {
                "kind": item.kind,
                "text": item.text,
                "name": item.name,
                "call_id": item.call_id,
                "raw_arguments": item.raw_arguments,
                "argument_error": item.argument_error,
            }
            for item in round_item.output_items
        ],
        "channel": round_item.channel,
        "payload_text": round_item.payload_text,
        "commands": list(round_item.calls),
        "parsed_commands": _parsed_command_views(parsed, registry),
        "results": _result_views(round_item),
        "rejection": round_item.rejection,
        "execution_result": round_item.result_text,
        "normalizations": normalizations,
        "cleaned_fields": cleaned_fields,
        "dropped_command_count": max(0, len(parsed.commands) - len(round_item.calls)),
        "terminal": bool(completed),
        "accepted": not bool(round_item.rejection),
        "validation_status": "ACCEPTED" if not round_item.rejection else "REJECTED",
        "terminal_rejection": False,
    }


@dataclass(slots=True)
class _BoundRunState:
    registry: MainCoreCommandRegistry
    rounds: list[ExecutionRound]
    rejection_errors: list[str]
    reference_map: dict[str, Any]
    prepared_context: Any
    source_message_ids: set[int]
    background_item_refs: list[str]
    seen_background_item_refs: set[str]
    identity_catalog: Any | None
    identity_context: Any | None
    identity_scope: str
    logical_model_step: int = 1
    last_text: str = ""
    agent_history: list[AIAgentTranscriptTurn] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _BoundRoundInvocation:
    invocation_id: str
    last_text: str
    note_manager: Any | None
    note_checkpoint: int
    prompt_plan_hash: str
    completion: AICompletion


class BoundRunLoopMixin:
    """Keep the public loop focused on its stop conditions."""

    def _initialize_bound_run(
        self,
        features: Any,
        contexts: Any,
        collector: Any,
    ) -> _BoundRunState:
        self._completed_nonterminal_results.clear()
        self._batch_collector = collector
        registry = MainCoreCommandRegistry.from_command_set(features.commands)
        rounds: list[ExecutionRound] = []
        rejection_errors: list[str] = []
        reference_map: dict[str, Any] = {}
        prepared_context = contexts.prepared_context
        source_message_ids: set[int] = set()
        background_item_refs: list[str] = []
        seen_background_item_refs: set[str] = set()
        identity_catalog, identity_context, identity_scope = self._identity_run_context(
            prepared_context
        )
        return _BoundRunState(
            registry=registry,
            rounds=rounds,
            rejection_errors=rejection_errors,
            reference_map=reference_map,
            prepared_context=prepared_context,
            source_message_ids=source_message_ids,
            background_item_refs=background_item_refs,
            seen_background_item_refs=seen_background_item_refs,
            identity_catalog=identity_catalog,
            identity_context=identity_context,
            identity_scope=identity_scope,
        )

    async def _invoke_bound_round(
        self,
        run_state: _BoundRunState,
        *,
        model_gateway: Any,
        request: CoreWakeRequest,
        route: Any,
        run_id: int,
        role: Any,
        state: Any,
        features: Any,
        contexts: Any,
        collector: Any,
        round_number: int,
        operation_timeout_seconds: float,
        runtime_gate: Any | None,
    ) -> _BoundRoundInvocation:
        await self._require_enabled(
            runtime_gate,
            request.profile_id,
            str(request.instance_id or ""),
        )
        note_manager = collector.request_context_manager
        note_checkpoint = note_manager.checkpoint() if note_manager is not None else 0
        prompt_plan_hash = hashlib.sha256(
            str(collector.current_plan or "").encode("utf-8")
        ).hexdigest()
        prompt, context_requirement = await self._compile_routed_round(
            model_gateway=model_gateway,
            request=request,
            run_id=run_id,
            route=route,
            role=role,
            state=state,
            features=features,
            contexts=contexts,
            collector=collector,
            registry=run_state.registry,
            rounds=run_state.rounds,
            agent_history=run_state.agent_history,
            round_number=round_number,
            operation_timeout_seconds=operation_timeout_seconds,
        )
        run_state.reference_map.update(prompt.reference_map)
        collector.model_reference_map = dict(run_state.reference_map)
        try:
            result, invocation_id, projected_message_ids = await self._invoke_round(
                model_gateway=model_gateway,
                request=request,
                route=route,
                run_id=run_id,
                round_number=round_number,
                logical_step_number=run_state.logical_model_step,
                prompt=prompt,
                context_requirement=context_requirement,
                operation_timeout_seconds=operation_timeout_seconds,
                agent_history=tuple(run_state.agent_history),
            )
        except AIInvocationError as exc:
            strict_prompt = self._strict_image_retry_prompt(prompt, run_state.rounds, exc)
            if strict_prompt is None:
                raise
            await self._record_strict_image_retry(
                model_gateway,
                run_id=run_id,
                round_number=round_number,
                before=tuple(prompt.image_urls),
                after=tuple(strict_prompt.image_urls),
            )
            result, invocation_id, projected_message_ids = await self._invoke_round(
                model_gateway=model_gateway,
                request=request,
                route=route,
                run_id=run_id,
                round_number=round_number,
                logical_step_number=run_state.logical_model_step,
                prompt=strict_prompt,
                context_requirement=context_requirement,
                operation_timeout_seconds=operation_timeout_seconds,
                agent_history=tuple(run_state.agent_history),
            )
            prompt = strict_prompt
        await self._record_successful_background_projection(
            request,
            run_id,
            invocation_id,
            round_number,
            prompt,
            (run_state.seen_background_item_refs, run_state.background_item_refs),
        )
        self._discard_consumed_result_images(run_state.rounds)
        run_state.source_message_ids.update(projected_message_ids)
        last_text = str(result.completion.text or "")
        run_state.last_text = last_text
        return _BoundRoundInvocation(
            invocation_id=invocation_id,
            last_text=last_text,
            note_manager=note_manager,
            note_checkpoint=note_checkpoint,
            prompt_plan_hash=prompt_plan_hash,
            completion=result.completion,
        )

    @staticmethod
    def _strict_image_retry_prompt(
        prompt: Any,
        rounds: list[ExecutionRound],
        error: AIInvocationError,
    ) -> Any | None:
        if not rounds or not _is_provider_payload_limit(error):
            return None
        result_images = tuple(
            image
            for result in rounds[-1].results
            if result.ok
            for image in result.model_input_images
        )
        if not result_images:
            return None
        strict_images = rebudget_model_image_data_uris(
            tuple(prompt.image_urls),
            target_values=result_images,
            maximum_total_bytes=STRICT_MODEL_IMAGE_PREVIEW_TOTAL_BYTES,
        )
        if strict_images == tuple(prompt.image_urls):
            return None
        return replace(prompt, image_urls=strict_images)

    @staticmethod
    async def _record_strict_image_retry(
        model_gateway: Any,
        *,
        run_id: int,
        round_number: int,
        before: tuple[str, ...],
        after: tuple[str, ...],
    ) -> None:
        trace = current_ai_work_context()
        if trace is None or trace.workflow_id <= 0:
            return
        await model_gateway.record_ai_work_event(
            workflow_id=trace.workflow_id,
            node_id=trace.node_id,
            event_category="ROUTING",
            severity="WARNING",
            code="main_core_model_image_payload_rebounded",
            summary="模型请求体超限，已缩小模型预览并重试当前轮次",
            details={
                "run_id": int(run_id),
                "round": max(1, int(round_number)),
                "before_data_uri_chars": sum(
                    len(value) for value in before if value.startswith("data:image/")
                ),
                "after_data_uri_chars": sum(
                    len(value) for value in after if value.startswith("data:image/")
                ),
            },
        )

    async def _handle_invoked_bound_round(
        self,
        run_state: _BoundRunState,
        invocation: _BoundRoundInvocation,
        *,
        model_gateway: Any,
        request: CoreWakeRequest,
        route: Any,
        run_id: int,
        event: Any,
        features: Any,
        collector: Any,
        round_number: int,
        max_parallel: int,
        runtime_gate: Any | None,
    ) -> bool:
        await self._require_enabled(
            runtime_gate,
            request.profile_id,
            str(request.instance_id or ""),
        )
        model_step = await self._ai_work_node_context(model_gateway, invocation.invocation_id)
        agent_response = parse_agent_response(invocation.completion)
        with CollectorScope(collector):
            if not agent_response.valid:
                self._record_rejection(
                    run_state.rounds,
                    run_state.rejection_errors,
                    round_number,
                    invocation.last_text,
                    agent_response.error,
                    raw_text=agent_response.raw_model_text,
                )
                completed = False
            else:
                completed = await self._handle_bound_model_turn(
                    model_step=model_step,
                    text=agent_response.text,
                    channel=agent_response.channel,
                    raw_model_text=agent_response.raw_model_text,
                    round_number=round_number,
                    registry=run_state.registry,
                    reference_map=run_state.reference_map,
                    rounds=run_state.rounds,
                    rejection_errors=run_state.rejection_errors,
                    command_set=features.commands,
                    event=event,
                    max_parallel=max_parallel,
                    model_gateway=model_gateway,
                    run_id=run_id,
                    identity_catalog=run_state.identity_catalog,
                    identity_context=run_state.identity_context,
                    identity_scope=run_state.identity_scope,
                    runtime_gate=runtime_gate,
                    profile_id=request.profile_id,
                    instance_id=str(request.instance_id or ""),
                    thinking_policy=getattr(route, "thinking_policy", None),
                    has_plan=bool(str(collector.current_plan or "").strip()),
                )
        current_round = self._annotated_round(run_state, invocation, round_number)
        if current_round is not None:
            current_round = self._finalize_agent_round(
                run_state,
                current_round,
                agent_response,
            )
        await self._annotate_model_exchange(
            model_gateway,
            invocation.invocation_id,
            round_no=round_number,
            processing=processing_view(
                current_round,
                registry=run_state.registry,
                completed=completed,
            ),
        )
        if not completed and current_round is not None and not current_round.rejection:
            run_state.logical_model_step += 1
        return completed

    @staticmethod
    def _finalize_agent_round(
        run_state: _BoundRunState,
        current_round: ExecutionRound,
        agent_response: Any,
    ) -> ExecutionRound:
        result_text = str(current_round.result_text or current_round.rejection or "")
        runtime_notes = "\n\n".join(
            str(note) for note in current_round.runtime_notes if str(note or "").strip()
        )
        if runtime_notes:
            result_text = (
                f"{result_text}\n\n行动补充\n{runtime_notes}" if result_text else runtime_notes
            )
        current_round = replace(
            current_round,
            working_text=agent_response.outside_text,
            channel=agent_response.channel,
            payload_text=agent_response.text,
            result_text=result_text,
            output_items=agent_response.output_items,
            transport_mode=agent_response.transport_mode,
        )
        run_state.rounds[-1] = current_round
        run_state.agent_history.append(
            transcript_turn(
                agent_response,
                result_text,
                source_round_number=current_round.number,
                contains_plan=any(call == "制定Plan" for call in current_round.calls),
                unresolved_failure=bool(current_round.rejection)
                or any(not result.ok for result in current_round.results),
                public_references=tuple(
                    dict.fromkeys(
                        reference
                        for result in current_round.results
                        for reference in result.public_references
                    )
                ),
                contains_image_material=any(
                    result.media_asset_ids or result.model_input_images
                    for result in current_round.results
                ),
            )
        )
        return current_round

    @staticmethod
    def _annotated_round(
        run_state: _BoundRunState,
        invocation: _BoundRoundInvocation,
        round_number: int,
    ) -> ExecutionRound | None:
        rounds = run_state.rounds
        current_round = rounds[-1] if rounds and rounds[-1].number == round_number else None
        if current_round is None:
            return None
        note_manager = invocation.note_manager
        current_round = replace(
            current_round,
            runtime_notes=(
                tuple(note_manager.since(invocation.note_checkpoint))
                if note_manager is not None
                else ()
            ),
            prompt_plan_hash=invocation.prompt_plan_hash,
            invocation_id=invocation.invocation_id,
        )
        rounds[-1] = current_round
        return current_round


def _is_provider_payload_limit(error: AIInvocationError) -> bool:
    info = error.info
    if info.status_code == 413:
        return True
    if info.status_code != 400:
        return False
    provider_response = info.details.get("provider_response")
    try:
        response_text = json.dumps(provider_response, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        response_text = str(provider_response or "")
    lowered = response_text[:8000].lower()
    return any(
        marker in lowered
        for marker in (
            "request exceeds local input limits",
            "raw_body_limit_bytes",
            "request body is too large",
            "request_too_large",
            "payload too large",
        )
    )


def parse_identity_turn(
    text: str,
    registry: MainCoreCommandRegistry,
    identity_catalog: Any | None,
    identity_scope: str,
    identity_context: Any | None,
) -> tuple[ParsedModelTurn, str]:
    parsed = parse_model_turn(text)
    if not parsed.valid:
        return parsed, "；".join(parsed.errors)
    if identity_catalog is None:
        return parsed, ""
    try:
        return (
            decode_identity_turn(
                parsed,
                registry,
                identity_catalog,
                identity_scope,
                identity_context,
            ),
            "",
        )
    except ValueError as exc:
        return parsed, str(exc)


def decode_identity_turn(
    parsed: ParsedModelTurn,
    registry: MainCoreCommandRegistry,
    identity_catalog: Any,
    identity_scope: str,
    identity_context: Any = None,
) -> ParsedModelTurn:
    commands = []
    for command in parsed.commands:
        parameters = decode_model_parameter_map(
            command.parameters,
            identity_catalog,
            scope=identity_scope,
            identity_context=identity_context,
            identity_modes=registry.identity_modes(command.name),
        )
        commands.append(
            ParsedCommand(
                name=command.name,
                parameters=parameters,
                ordinal=command.ordinal,
                raw_text=command.raw_text,
            )
        )
    return ParsedModelTurn(
        working_text=parsed.working_text,
        commands=tuple(commands),
        errors=parsed.errors,
        raw_text=parsed.raw_text,
    )


__all__ = ["BoundRunLoopMixin", "decode_identity_turn", "parse_identity_turn", "processing_view"]
