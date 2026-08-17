"""SoulCore-owned model/action loop for the RolePlay text protocol."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from ...contracts.ai_models import (
    AIAgentToolResult,
    AIAgentTranscriptTurn,
    AIErrorCode,
    AIErrorInfo,
    AIExecutionMode,
    AIInvocationError,
    AIModelRequest,
    AIRetryPolicy,
    AIWorkPurpose,
)
from ...contracts.models import CoreWakeRequest, WakeSource
from ...contracts.thinking import DEFAULT_THINKING_POLICY, ThinkingComplexity
from ...shared.prompt_document import (
    TrustedPromptMarkup,
    join_prompt_markup,
    prompt_markup_block,
)
from ..ai import current_ai_work_context
from ..ai.service import (
    CommandExecutionResult,
    CommandProtocolError,
    MainCoreCommandRegistry,
    ModelContextRequirement,
    available_prompt_tokens,
    configured_model_context_tokens,
    register_result_references,
)
from ..character_model import StoryStylePrompts
from .agent_context import compact_agent_history_if_needed
from .agent_protocol import CONTINUE_CHANNEL, FINAL_CHANNEL, main_core_agent_tools, protocol_error
from .identity_command_projection import parse_identity_turn
from .processing_view import processing_view
from .roleplay_prompt import (
    ExecutionRound,
    RolePlayPromptCompiler,
)
from .text_command_bound_run import BoundRunLoopMixin
from .text_command_ordered_execution import OrderedCommandExecutionMixin
from .text_command_runtime import (
    NO_NEW_MAIN_CORE_ACTION,
    MainCoreStepRejectedThreeTimes,
    TextCommandRuntimeMixin,
)

logger = logging.getLogger(__name__)


def _projection_request(
    message_ids: tuple[int, ...],
    summary_ids: tuple[int, ...],
    summary_coverage: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[tuple[int, int, int], ...]]:
    visible_ids = tuple(sorted({int(value) for value in message_ids if int(value) > 0}))
    visible_summary_ids = tuple(sorted({int(value) for value in summary_ids if int(value) > 0}))
    coverage = tuple(
        (int(value[0]), int(value[1]), int(value[2]))
        for value in summary_coverage
        if len(value) == 3
    )
    return visible_ids, visible_summary_ids, coverage


def _projected_visible_ids(
    projected: Any,
    fallback: tuple[int, ...],
) -> tuple[int, ...]:
    if not isinstance(projected, Mapping):
        return fallback
    return tuple(
        sorted(
            {
                int(value)
                for value in projected.get("model_visible_message_ids", ())
                if int(value) > 0
            }
        )
    )


class MainCoreWorkRecordingMixin:
    """Own the MainCore business-stage lifecycle around the model/action loop."""

    async def run(
        self,
        *,
        model_gateway: Any,
        request: CoreWakeRequest,
        run_id: int,
        event: Any,
        route: Any,
        role: Any,
        state: Any,
        features: Any,
        contexts: Any,
        collector: Any,
        max_steps: int,
        max_parallel: int,
        operation_timeout_seconds: float,
        runtime_gate: Any | None = None,
        final_submitter: Any | None = None,
    ) -> Any:
        stage = await self._start_main_core_stage(model_gateway, request, run_id)
        values = {
            "model_gateway": model_gateway,
            "request": request,
            "run_id": run_id,
            "event": event,
            "route": route,
            "role": role,
            "state": state,
            "features": features,
            "contexts": contexts,
            "collector": collector,
            "max_steps": max_steps,
            "max_parallel": max_parallel,
            "operation_timeout_seconds": operation_timeout_seconds,
            "runtime_gate": runtime_gate,
            "final_submitter": final_submitter,
        }
        try:
            result = await self._run_recorded_main_core(model_gateway, stage, values)
        except asyncio.CancelledError:
            await self._cancel_main_core_stage(model_gateway, stage)
            raise
        except Exception as exc:
            await self._fail_main_core_stage(model_gateway, stage, exc)
            raise
        await self._complete_main_core_stage(model_gateway, stage, result)
        return result

    @staticmethod
    async def _start_main_core_stage(
        model_gateway: Any, request: CoreWakeRequest, run_id: int
    ) -> Any:
        parent = current_ai_work_context()
        if parent is None or parent.workflow_id <= 0:
            return None
        try:
            return await model_gateway.start_ai_work_node(
                workflow_id=parent.workflow_id,
                parent_node_id=parent.node_id,
                node_role="BUSINESS_STAGE",
                node_kind="MODEL",
                purpose=AIWorkPurpose.MAIN_CORE.value,
                node_key=f"main-core-run:{run_id}",
                input={"run_id": run_id, "source": request.source.value},
            )
        except Exception:
            logger.exception("failed to start MainCore AI work stage")
            return None

    async def _run_recorded_main_core(
        self, model_gateway: Any, stage: Any, values: dict[str, Any]
    ) -> Any:
        if stage is None:
            return await self._run_bound(**values)
        with self._bind_ai_workflow(model_gateway, stage):
            return await self._run_bound(**values)

    async def _cancel_main_core_stage(self, model_gateway: Any, stage: Any) -> None:
        if stage is None or stage.node_id is None:
            return
        await asyncio.shield(
            self._finish_ai_work_node(model_gateway, stage.node_id, status="CANCELLED")
        )

    async def _fail_main_core_stage(self, model_gateway: Any, stage: Any, exc: Exception) -> None:
        if stage is None or stage.node_id is None:
            return
        await self._finish_ai_work_node(
            model_gateway,
            stage.node_id,
            status="FAILED",
            error_code=getattr(exc, "code", type(exc).__name__),
            error_message=str(exc),
        )

    async def _complete_main_core_stage(self, model_gateway: Any, stage: Any, result: Any) -> None:
        if stage is None or stage.node_id is None:
            return
        hard_limit = bool(result.hard_limit_reached)
        await self._finish_ai_work_node(
            model_gateway,
            stage.node_id,
            status="FAILED" if hard_limit else "SUCCEEDED",
            error_code="main_core_hard_step_limit_exceeded" if hard_limit else "",
            error_message="Main Core 已达到最大内部步骤数" if hard_limit else "",
            summary="Main Core 已完成多轮生成、校验与内部动作",
            result={
                "round_count": len(result.rounds),
                "hard_limit_reached": hard_limit,
            },
        )

    @staticmethod
    async def _ai_work_node_context(model_gateway: Any, invocation_id: str) -> Any | None:
        try:
            return await model_gateway.ai_work_node_context(invocation_id)
        except Exception:
            logger.exception("failed to read MainCore AI work node context")
            return None

    @staticmethod
    async def _project_model_visible_message_ids(
        model_gateway: Any,
        *,
        run_id: int,
        message_ids: tuple[int, ...],
        summary_ids: tuple[int, ...] = (),
        summary_coverage: tuple[tuple[int, int, int], ...] = (),
    ) -> tuple[int, ...]:
        visible_ids, visible_summary_ids, coverage = _projection_request(
            message_ids,
            summary_ids,
            summary_coverage,
        )
        if not visible_ids and not visible_summary_ids and not coverage:
            return ()
        trace = current_ai_work_context()
        if trace is None or trace.node_id is None:
            raise RuntimeError("model-visible message projection has no active AI work node")
        projected = await model_gateway.project_model_visible_message_ids(
            run_id,
            trace.node_id,
            visible_ids,
            summary_ids=visible_summary_ids,
            summary_coverage=coverage,
        )
        if not projected:
            raise RuntimeError("model-visible messages changed before provider invocation")
        return _projected_visible_ids(projected, visible_ids)

    @staticmethod
    def _bind_ai_workflow(model_gateway: Any, context: Any) -> Any:
        return model_gateway.bind_ai_workflow(context)

    @staticmethod
    async def _annotate_model_exchange(
        model_gateway: Any,
        invocation_id: str,
        *,
        round_no: int,
        processing: dict[str, Any],
    ) -> None:
        try:
            await model_gateway.annotate_model_exchange(
                invocation_id, round_no=round_no, processing=processing
            )
        except Exception:
            logger.exception("failed to annotate MainCore provider attempt")

    @staticmethod
    async def _finish_ai_work_node(model_gateway: Any, node_id: int, **values: Any) -> None:
        try:
            await model_gateway.finish_ai_work_node(node_id, **values)
        except Exception:
            logger.exception("failed to finish MainCore AI work stage")


class ModelTurnHandlingMixin:
    async def _handle_model_turn(
        self,
        *,
        text: str,
        channel: str,
        raw_model_text: str,
        round_number: int,
        registry: MainCoreCommandRegistry,
        reference_map: dict[str, Any],
        rounds: list[ExecutionRound],
        rejection_errors: list[str],
        command_set: Any,
        event: Any,
        max_parallel: int,
        model_gateway: Any,
        run_id: int,
        identity_catalog: Any | None,
        identity_scope: str,
        identity_context: Any | None = None,
        runtime_gate: Any | None = None,
        profile_id: str = "",
        instance_id: str = "",
        thinking_policy: Any | None = None,
        has_plan: bool = False,
    ) -> bool:
        parsed = self._parse_actionable_model_turn(
            text=text,
            round_number=round_number,
            registry=registry,
            identity_catalog=identity_catalog,
            identity_scope=identity_scope,
            identity_context=identity_context,
            rounds=rounds,
            rejection_errors=rejection_errors,
            raw_model_text=raw_model_text,
        )
        if parsed is None:
            return False
        channel_error = self._agent_channel_error(
            parsed,
            registry,
            channel=channel,
            thinking_policy=thinking_policy,
            has_plan=has_plan,
        )
        if channel_error:
            self._record_rejection(
                rounds,
                rejection_errors,
                round_number,
                parsed.working_text,
                channel_error,
                raw_text=raw_model_text,
            )
            return False
        execution = await self._validated_turn_execution_or_reject(
            parsed=parsed,
            registry=registry,
            reference_map=reference_map,
            rounds=rounds,
            rejection_errors=rejection_errors,
            round_number=round_number,
            raw_model_text=raw_model_text,
        )
        if execution is None:
            return False
        return await self._execute_validated_model_turn(
            parsed=parsed,
            commands=execution.execution_commands,
            registry=registry,
            validated_by_ordinal=execution.validated_by_ordinal,
            validation_results=execution.validation_results,
            calls=execution.calls,
            reference_map=reference_map,
            rounds=rounds,
            rejection_errors=rejection_errors,
            round_number=round_number,
            raw_text=raw_model_text,
            command_set=command_set,
            event=event,
            max_parallel=max_parallel,
            model_gateway=model_gateway,
            run_id=run_id,
            runtime_gate=runtime_gate,
            profile_id=profile_id,
            instance_id=instance_id,
        )

    async def _validated_turn_execution_or_reject(
        self,
        *,
        parsed: Any,
        registry: MainCoreCommandRegistry,
        reference_map: dict[str, Any],
        rounds: list[ExecutionRound],
        rejection_errors: list[str],
        round_number: int,
        raw_model_text: str,
    ) -> Any | None:
        execution = self._validate_model_turn_commands(parsed, registry, reference_map)
        if execution.batch_validation_failure:
            self._record_execution_rejection(
                execution,
                execution.batch_validation_failure,
                parsed,
                reference_map,
                rounds,
                rejection_errors,
                round_number,
                raw_model_text,
            )
            return None
        if not execution.execution_commands:
            self._record_execution_rejection(
                execution,
                NO_NEW_MAIN_CORE_ACTION,
                parsed,
                reference_map,
                rounds,
                rejection_errors,
                round_number,
                raw_model_text,
            )
            return None
        preflight_error = await self._terminal_preflight_error(
            execution.execution_commands,
            registry,
            execution.validated_by_ordinal,
            reference_map=reference_map,
        )
        if preflight_error:
            self._record_execution_rejection(
                execution,
                preflight_error,
                parsed,
                reference_map,
                rounds,
                rejection_errors,
                round_number,
                raw_model_text,
            )
            return None
        return execution

    def _record_execution_rejection(
        self,
        execution: Any,
        error: str,
        parsed: Any,
        reference_map: dict[str, Any],
        rounds: list[ExecutionRound],
        rejection_errors: list[str],
        round_number: int,
        raw_model_text: str,
    ) -> None:
        results = register_result_references(tuple(execution.validation_results), reference_map)
        self._record_rejection(
            rounds,
            rejection_errors,
            round_number,
            parsed.working_text,
            error,
            execution.calls,
            results,
            raw_text=raw_model_text,
        )

    @staticmethod
    def _agent_channel_error(
        parsed: Any,
        registry: MainCoreCommandRegistry,
        *,
        channel: str,
        thinking_policy: Any | None,
        has_plan: bool,
    ) -> str:
        channel_error = _channel_contents_error(parsed, registry, channel)
        if channel_error:
            return channel_error
        if (
            getattr(thinking_policy, "complexity", None) == ThinkingComplexity.EXTREME
            and not has_plan
            and not _is_plan_only(parsed, channel)
        ):
            return protocol_error(
                "极致模式的第一次有效调用必须选择“继续行动”，且 text 中只能有一条“制定Plan”。"
                "工具外草稿不能代替 Plan。"
            )
        return ""

    def _record_rejection(
        self,
        rounds: list[ExecutionRound],
        errors: list[str],
        round_number: int,
        working_text: str,
        error: str,
        calls: tuple[str, ...] = (),
        results: Any = (),
        raw_text: str = "",
    ) -> None:
        rounds.append(
            ExecutionRound(
                round_number,
                working_text,
                calls,
                tuple(results),
                rejection=error,
                raw_text=raw_text,
                result_text=error,
            )
        )
        self._reject(errors, error, rounds)


def _channel_contents_error(
    parsed: Any,
    registry: MainCoreCommandRegistry,
    channel: str,
) -> str:
    if str(parsed.working_text or "").strip():
        return protocol_error(
            "通道字符串中含有 XML 指令之外的文字；草稿只能写在原生工具调用之外，"
            "或纯文本外层通道标签之外。"
        )
    terminal, nonterminal = _command_kinds(parsed, registry)
    if channel == CONTINUE_CHANNEL and terminal:
        return protocol_error("XML 指令本身可以解析，但“继续行动”中包含了最终表达指令。")
    if channel == FINAL_CHANNEL and nonterminal:
        return protocol_error("XML 指令本身可以解析，但“最终表达”中包含了非终止行动指令。")
    if channel not in {CONTINUE_CHANNEL, FINAL_CHANNEL}:
        return protocol_error("没有选择可执行的 Agent 通道。")
    return ""


def _command_kinds(
    parsed: Any,
    registry: MainCoreCommandRegistry,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    commands = tuple(parsed.commands)
    terminal = tuple(
        item for item in commands if bool((spec := registry.get(item.name)) and spec.terminal)
    )
    nonterminal = tuple(
        item for item in commands if bool((spec := registry.get(item.name)) and not spec.terminal)
    )
    return terminal, nonterminal


def _is_plan_only(parsed: Any, channel: str) -> bool:
    commands = tuple(parsed.commands)
    return channel == CONTINUE_CHANNEL and len(commands) == 1 and commands[0].name == "制定Plan"


class BoundModelTurnMixin:
    async def _handle_bound_model_turn(
        self,
        *,
        model_step: Any,
        text: str,
        channel: str,
        raw_model_text: str,
        round_number: int,
        registry: MainCoreCommandRegistry,
        reference_map: dict[str, Any],
        rounds: list[ExecutionRound],
        rejection_errors: list[str],
        command_set: Any,
        event: Any,
        max_parallel: int,
        model_gateway: Any,
        run_id: int,
        identity_catalog: Any,
        identity_context: Any,
        identity_scope: Any,
        runtime_gate: Any,
        profile_id: str,
        instance_id: str,
        thinking_policy: Any | None,
        has_plan: bool,
    ) -> bool:
        arguments = {
            "text": text,
            "channel": channel,
            "raw_model_text": raw_model_text,
            "round_number": round_number,
            "registry": registry,
            "reference_map": reference_map,
            "rounds": rounds,
            "rejection_errors": rejection_errors,
            "command_set": command_set,
            "event": event,
            "max_parallel": max_parallel,
            "model_gateway": model_gateway,
            "run_id": run_id,
            "identity_catalog": identity_catalog,
            "identity_context": identity_context,
            "identity_scope": identity_scope,
            "runtime_gate": runtime_gate,
            "profile_id": profile_id,
            "instance_id": instance_id,
            "thinking_policy": thinking_policy,
            "has_plan": has_plan,
        }
        if model_step is None:
            return await self._handle_model_turn(**arguments)
        with self._bind_ai_workflow(model_gateway, model_step):
            return await self._handle_model_turn(**arguments)


class MainCoreTurnParsingMixin:
    def _parse_model_turn_or_record_rejection(
        self,
        *,
        text: str,
        round_number: int,
        registry: MainCoreCommandRegistry,
        identity_catalog: Any | None,
        identity_scope: str,
        identity_context: Any | None,
        rounds: list[ExecutionRound],
        rejection_errors: list[str],
        raw_model_text: str = "",
    ) -> Any | None:
        parsed, parse_error = parse_identity_turn(
            text,
            registry,
            identity_catalog,
            identity_scope,
            identity_context,
        )
        if parse_error:
            self._record_rejection(
                rounds,
                rejection_errors,
                round_number,
                parsed.working_text,
                parse_error,
                raw_text=raw_model_text or text,
            )
            return None
        return parsed


def validated_commands(
    commands: Any,
    registry: MainCoreCommandRegistry,
    reference_map: dict[str, Any],
) -> tuple[list[Any], list[CommandExecutionResult]]:
    validated: list[Any] = []
    failures: list[CommandExecutionResult] = []
    for command in commands:
        try:
            validated.append(registry.validate(command, reference_map))
        except CommandProtocolError as exc:
            failures.append(CommandExecutionResult(command.ordinal, command.name, False, str(exc)))
    return validated, failures


_NO_COMPLETE_COMMAND_REJECTION = (
    "所选 Agent 通道中没有任何完整 XML 指令；本轮没有执行任何行动，也没有发送内容。"
    "请保留原意，在下一次调用中重新提交完整 XML。"
)


@dataclass(slots=True)
class ValidatedModelTurn:
    execution_commands: tuple[Any, ...]
    validation_results: list[Any]
    calls: tuple[str, ...]
    validated_by_ordinal: dict[int, Any]
    batch_validation_failure: str = ""


class ModelTurnPreparationMixin:
    def _parse_actionable_model_turn(
        self,
        *,
        text: str,
        round_number: int,
        registry: MainCoreCommandRegistry,
        identity_catalog: Any | None,
        identity_scope: str,
        identity_context: Any | None,
        rounds: list[ExecutionRound],
        rejection_errors: list[str],
        raw_model_text: str = "",
    ) -> Any | None:
        parsed = self._parse_model_turn_or_record_rejection(
            text=text,
            round_number=round_number,
            registry=registry,
            identity_catalog=identity_catalog,
            identity_scope=identity_scope,
            identity_context=identity_context,
            rounds=rounds,
            rejection_errors=rejection_errors,
            raw_model_text=raw_model_text,
        )
        if parsed is None:
            return None
        if parsed.commands:
            return parsed
        self._record_rejection(
            rounds,
            rejection_errors,
            round_number,
            parsed.working_text,
            _NO_COMPLETE_COMMAND_REJECTION,
            raw_text=raw_model_text or text,
        )
        return None

    def _validate_model_turn_commands(
        self,
        parsed: Any,
        registry: MainCoreCommandRegistry,
        reference_map: dict[str, Any],
    ) -> ValidatedModelTurn:
        commands = tuple(parsed.commands)
        validated, results = validated_commands(commands, registry, dict(reference_map))
        batch_failure = ""
        if results:
            batch_failure = (
                "本批至少一条指令未通过校验，因此没有执行任何指令，也没有发送任何内容。\n"
                + "\n".join(f"- {item.content}" for item in results)
            )
            validated = []
        calls = tuple(item.name for item in parsed.commands)
        if not batch_failure:
            validated, commands, replay_results = self._without_replayed_nonterminal_commands(
                validated, commands
            )
            results.extend(replay_results)
        else:
            commands = ()
        return ValidatedModelTurn(
            execution_commands=commands,
            validation_results=results,
            calls=calls,
            validated_by_ordinal={item.parsed.ordinal: item for item in validated},
            batch_validation_failure=batch_failure,
        )


@dataclass(frozen=True, slots=True)
class MainCoreModelResult:
    text: str
    hard_limit_reached: bool = False
    rounds: tuple[ExecutionRound, ...] = ()
    source_message_ids: tuple[int, ...] = ()
    background_item_refs: tuple[str, ...] = ()
    final_result: Any | None = None


@dataclass(frozen=True, slots=True)
class MainCoreFinalSubmissionAttempt:
    result: Any | None = None
    error: str = ""


class MainCoreTextCommandLoop(
    BoundRunLoopMixin,
    MainCoreTurnParsingMixin,
    BoundModelTurnMixin,
    MainCoreWorkRecordingMixin,
    OrderedCommandExecutionMixin,
    ModelTurnPreparationMixin,
    ModelTurnHandlingMixin,
    TextCommandRuntimeMixin,
):
    def __init__(
        self,
        compiler: RolePlayPromptCompiler | None = None,
        *,
        story_exposure_committer: Any | None = None,
    ) -> None:
        self.compiler = compiler or RolePlayPromptCompiler()
        self.story_exposure_committer = story_exposure_committer
        self._completed_nonterminal_results: dict[str, Any] = {}

    async def _run_bound(
        self,
        *,
        model_gateway: Any,
        request: CoreWakeRequest,
        run_id: int,
        event: Any,
        route: Any,
        role: Any,
        state: Any,
        features: Any,
        contexts: Any,
        collector: Any,
        max_steps: int,
        max_parallel: int,
        operation_timeout_seconds: float,
        runtime_gate: Any | None = None,
        final_submitter: Any | None = None,
    ) -> MainCoreModelResult:
        run_state = self._initialize_bound_run(features, contexts, collector)
        for round_number in range(1, max(1, int(max_steps)) + 1):
            terminal_state = self._snapshot_terminal_collector_state(collector)
            invocation = await self._invoke_bound_round(
                run_state,
                model_gateway=model_gateway,
                request=request,
                run_id=run_id,
                route=route,
                role=role,
                state=state,
                features=features,
                contexts=contexts,
                collector=collector,
                round_number=round_number,
                operation_timeout_seconds=operation_timeout_seconds,
                runtime_gate=runtime_gate,
            )
            completed = await self._handle_invoked_bound_round(
                run_state,
                invocation,
                model_gateway=model_gateway,
                request=request,
                route=route,
                run_id=run_id,
                event=event,
                features=features,
                collector=collector,
                round_number=round_number,
                max_parallel=max_parallel,
                runtime_gate=runtime_gate,
            )
            if completed:
                candidate = self._bound_run_result(run_state)
                if final_submitter is None:
                    return candidate
                submission = await final_submitter(candidate)
                if not isinstance(submission, MainCoreFinalSubmissionAttempt):
                    raise TypeError("MainCore final submitter returned an invalid result")
                if submission.result is not None:
                    return replace(candidate, final_result=submission.result)
                failure = self._final_submission_failure(submission.error)
                self._restore_terminal_collector_state(collector, terminal_state)
                failed_round = self._mark_final_submission_failure(run_state, failure)
                if failed_round is not None:
                    await self._annotate_model_exchange(
                        model_gateway,
                        failed_round.invocation_id,
                        round_no=failed_round.number,
                        processing=processing_view(
                            failed_round,
                            registry=run_state.registry,
                            completed=False,
                        ),
                    )
                run_state.logical_model_step += 1
                self._reject(run_state.rejection_errors, failure, run_state.rounds)
        self._discard_consumed_result_images(run_state.rounds)
        return self._bound_run_result(run_state, hard_limit_reached=True)

    @staticmethod
    def _bound_run_result(
        run_state: Any,
        *,
        hard_limit_reached: bool = False,
    ) -> MainCoreModelResult:
        return MainCoreModelResult(
            run_state.last_text,
            hard_limit_reached=hard_limit_reached,
            rounds=tuple(run_state.rounds),
            source_message_ids=tuple(sorted(run_state.source_message_ids)),
            background_item_refs=tuple(run_state.background_item_refs),
        )

    @staticmethod
    def _snapshot_terminal_collector_state(collector: Any) -> dict[str, Any]:
        return {
            name: deepcopy(getattr(collector, name))
            for name in (
                "decision",
                "commit_calls",
                "selected_media_asset_ids",
                "selected_sticker_ref_ids",
                "selected_important_todo_refs",
            )
            if hasattr(collector, name)
        }

    @staticmethod
    def _restore_terminal_collector_state(
        collector: Any,
        values: dict[str, Any],
    ) -> None:
        for name, value in values.items():
            setattr(collector, name, deepcopy(value))

    @staticmethod
    def _final_submission_failure(error: str) -> str:
        reason = str(error or "最终事务没有提交。").strip()
        return f"最终表达提交失败\n\n{reason}\n本轮没有提交最终表达；请根据失败原因重新提交。"

    @staticmethod
    def _mark_final_submission_failure(
        run_state: Any,
        failure: str,
    ) -> ExecutionRound | None:
        if not run_state.rounds:
            return None
        current = run_state.rounds[-1]
        result_text = f"最终表达：失败\n{failure}"
        ordinal = max((item.ordinal for item in current.results), default=0) + 1
        failed = replace(
            current,
            results=(
                *current.results,
                CommandExecutionResult(ordinal, "最终表达", False, failure),
            ),
            rejection=failure,
            result_text=result_text,
        )
        run_state.rounds[-1] = failed
        if run_state.agent_history:
            turn = run_state.agent_history[-1]
            run_state.agent_history[-1] = replace(
                turn,
                result_text=result_text,
                tool_results=tuple(
                    AIAgentToolResult(item.name, item.call_id, result_text)
                    for item in turn.tool_results
                ),
                unresolved_failure=True,
            )
        return failed

    @staticmethod
    def _terminal_segment_fingerprint(decision: dict[str, Any]) -> str:
        """Identify an expression segment before response polishing can change it."""

        payload = {
            "expression_steps": list(decision.get("expression_steps") or ()),
            "no_op": bool(decision.get("no_op")),
            "temporary_absence": decision.get("temporary_absence"),
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    async def _record_successful_background_projection(
        self,
        request: CoreWakeRequest,
        run_id: int,
        invocation_id: str,
        round_number: int,
        prompt: Any,
        ref_state: tuple[set[str], list[str]],
    ) -> None:
        seen_refs, accumulated_refs = ref_state
        prompt_refs = tuple(getattr(prompt, "background_item_refs", ()) or ())
        await self._settle_successful_story_exposure(
            profile_id=request.profile_id,
            instance_id=str(request.instance_id or ""),
            run_id=run_id,
            invocation_id=invocation_id,
            round_number=round_number,
            story_refs=tuple(ref for ref in prompt_refs if str(ref).startswith("bgstory:")),
        )
        for item_ref in prompt_refs:
            normalized_ref = str(item_ref or "").strip()
            if normalized_ref and normalized_ref not in seen_refs:
                seen_refs.add(normalized_ref)
                accumulated_refs.append(normalized_ref)

    async def _settle_successful_story_exposure(
        self,
        *,
        profile_id: str,
        instance_id: str,
        run_id: int,
        invocation_id: str,
        round_number: int,
        story_refs: tuple[str, ...],
    ) -> None:
        if not story_refs:
            return
        if self.story_exposure_committer is None:
            raise RuntimeError("story exposure committer is required for visible story material")
        settlement = asyncio.create_task(
            self.story_exposure_committer(
                profile_id,
                instance_id,
                run_id,
                invocation_id=invocation_id,
                round_no=round_number,
                story_refs=story_refs,
            )
        )
        cancellation: asyncio.CancelledError | None = None
        while not settlement.done():
            try:
                await asyncio.shield(settlement)
            except asyncio.CancelledError as exc:
                cancellation = exc
        await settlement
        if cancellation is not None:
            raise cancellation

    def _compile_round(
        self,
        *,
        request: CoreWakeRequest,
        route: Any,
        role: Any,
        state: Any,
        features: Any,
        contexts: Any,
        collector: Any,
        registry: MainCoreCommandRegistry,
        model_id: str,
        rounds: list[ExecutionRound] | tuple[ExecutionRound, ...] = (),
    ) -> Any:
        input_images = self._round_input_images(
            rounds,
            tuple(request.metadata.get("image_urls") or ()),
        )
        return self.compiler.compile(
            persona=route.persona,
            main_core_mode_prompts=route.main_core_mode_prompts,
            main_core_style_prompts=route.main_core_style_prompts,
            story_style_prompts=getattr(route, "story_style_prompts", StoryStylePrompts()),
            role=role,
            state=state,
            prepared_context=contexts.prepared_context,
            current_input=(
                (request.user_message or "") if features.responsibility.has_current_message else ""
            ),
            occurred_at=request.requested_at,
            timezone_name=str(getattr(collector, "timezone_name", "") or "").strip(),
            registry=registry,
            current_plan=str(collector.current_plan or ""),
            thinking_policy=(getattr(route, "thinking_policy", None) or DEFAULT_THINKING_POLICY),
            model_id=model_id,
            runtime_notes=join_prompt_markup(
                tuple(
                    value
                    if isinstance(value, TrustedPromptMarkup)
                    else prompt_markup_block("本轮运行要求", value)
                    for value in (
                        features.model_runtime_note,
                        self._runtime_notes(collector, request),
                    )
                    if str(value or "").strip()
                )
            ),
            current_asset_ids=tuple(collector.current_document_media_asset_ids),
            file_references=features.important_todo_refs,
            image_urls=input_images,
            max_prompt_tokens=self._max_prompt_tokens(
                role,
                route,
                contexts,
                input_image_count=len(input_images),
            ),
            responsibility=features.responsibility,
            trigger_reminders=tuple(
                getattr(getattr(contexts, "trigger_evaluation", None), "contents", ()) or ()
            ),
            previous_context_message_ids=tuple(
                getattr(contexts, "previous_context_message_ids", ()) or ()
            ),
        )

    @staticmethod
    def _round_input_images(
        rounds: list[ExecutionRound],
        current_input_images: tuple[Any, ...],
    ) -> tuple[str, ...]:
        latest_result_images = (
            tuple(
                image
                for result in rounds[-1].results
                if result.ok
                for image in result.model_input_images
            )
            if rounds
            else ()
        )
        current = tuple(str(image) for image in current_input_images if str(image).strip())
        return tuple(dict.fromkeys((*latest_result_images, *current)))[:5]

    @staticmethod
    def _discard_consumed_result_images(rounds: list[ExecutionRound]) -> None:
        if not rounds or not any(result.model_input_images for result in rounds[-1].results):
            return
        latest = rounds[-1]
        rounds[-1] = replace(
            latest,
            results=tuple(
                replace(result, model_input_images=()) if result.model_input_images else result
                for result in latest.results
            ),
        )

    @staticmethod
    def _max_prompt_tokens(
        role: Any,
        route: Any,
        contexts: Any,
        *,
        input_image_count: int = 0,
    ) -> int:
        del route
        prepared = contexts.prepared_context
        limit = int(prepared.effective_max_tokens) if prepared is not None else 0
        if limit < 1:
            try:
                limit = int(role.max_context_tokens)
            except (TypeError, ValueError):
                limit = 128000
        return available_prompt_tokens(
            limit,
            input_image_count=input_image_count,
        )

    async def _compile_routed_round(
        self,
        *,
        model_gateway: Any,
        request: CoreWakeRequest,
        run_id: int,
        route: Any,
        role: Any,
        state: Any,
        features: Any,
        contexts: Any,
        collector: Any,
        registry: MainCoreCommandRegistry,
        rounds: list[ExecutionRound],
        agent_history: list[AIAgentTranscriptTurn] | None = None,
        round_number: int,
        operation_timeout_seconds: float = 300.0,
    ) -> tuple[Any, ModelContextRequirement]:
        active_history = agent_history if agent_history is not None else []
        visited_backend_ids = {
            str(route.preferred_backend_id or ""),
        }
        while True:
            prompt = self._compile_round(
                request=request,
                route=route,
                role=role,
                state=state,
                features=features,
                contexts=contexts,
                collector=collector,
                registry=registry,
                rounds=rounds,
                model_id=self._model_id(route),
            )
            current = route.backend_hint
            current_limit = configured_model_context_tokens(current)
            requirement = await compact_agent_history_if_needed(
                model_gateway=model_gateway,
                history=active_history,
                base_prompt_tokens=prompt.total_tokens,
                input_image_count=len(prompt.image_urls),
                context_limit=current_limit,
                model_id=self._model_id(route),
                backend_id=str(current.backend_id if current is not None else ""),
                request=request,
                run_id=run_id,
                round_number=round_number,
                current_plan=str(collector.current_plan or ""),
                reference_map={
                    **dict(getattr(collector, "model_reference_map", {}) or {}),
                    **dict(prompt.reference_map),
                },
                timeout_seconds=operation_timeout_seconds,
            )
            if current_limit is None or requirement.total_tokens <= current_limit:
                return prompt, requirement

            replacement = await model_gateway.resolve_backend_hint(
                preferred_backend_id="",
                umo=route.route_umo,
                capability="chat.completion",
                profile_id=request.profile_id,
                minimum_context_tokens=requirement.total_tokens,
                requires_vision=bool(prompt.image_urls),
            )
            replacement_id = str(replacement.backend_id if replacement is not None else "")
            if replacement is None or not replacement_id or replacement_id in visited_backend_ids:
                raise self._missing_context_model_error(
                    route=route,
                    configured_max=current_limit,
                    requirement=requirement,
                    round_number=round_number,
                )
            replacement_limit = configured_model_context_tokens(replacement)
            if replacement_limit is None or replacement_limit < requirement.total_tokens:
                raise RuntimeError("context model resolver returned an undersized backend")

            await self._record_context_model_switch(
                model_gateway=model_gateway,
                run_id=run_id,
                round_number=round_number,
                before=route.backend_hint,
                after=replacement,
                before_limit=current_limit,
                after_limit=replacement_limit,
                requirement=requirement,
                has_images=bool(prompt.image_urls),
            )
            visited_backend_ids.add(replacement_id)
            route.backend_hint = replacement
            route.preferred_backend_id = replacement_id

    @staticmethod
    def _missing_context_model_error(
        *,
        route: Any,
        configured_max: int,
        requirement: ModelContextRequirement,
        round_number: int,
    ) -> AIInvocationError:
        required = requirement.total_tokens
        message = (
            f"你设置的最大上下文为{configured_max}，当前上下文为{required}，"
            f"但是在模型里没有找到可以处理上下文字数为{required}的模型。"
        )
        descriptor = route.backend_hint
        return AIInvocationError(
            AIErrorInfo(
                AIErrorCode.CONTEXT_BUDGET,
                message,
                backend_id=str(descriptor.backend_id if descriptor is not None else ""),
                phase="select_context_model",
                details={
                    "context_error_kind": "no_suitable_model",
                    "configured_max_context_tokens": configured_max,
                    "required_context_tokens": required,
                    "input_text_tokens": requirement.input_text_tokens,
                    "input_image_tokens": requirement.input_image_tokens,
                    "reserved_output_tokens": requirement.reserved_output_tokens,
                    "model_id": str(descriptor.model if descriptor is not None else ""),
                    "round": max(1, int(round_number)),
                },
            )
        )

    @staticmethod
    async def _record_context_model_switch(
        *,
        model_gateway: Any,
        run_id: int,
        round_number: int,
        before: Any,
        after: Any,
        before_limit: int,
        after_limit: int,
        requirement: ModelContextRequirement,
        has_images: bool,
    ) -> None:
        trace = current_ai_work_context()
        if trace is None or trace.workflow_id <= 0:
            return
        await model_gateway.record_ai_work_event(
            workflow_id=trace.workflow_id,
            node_id=trace.node_id,
            event_category="ROUTING",
            severity="INFO",
            code="main_core_context_model_switch",
            summary="Main Core 已切换到可容纳当前上下文的模型",
            details={
                "run_id": int(run_id),
                "round": max(1, int(round_number)),
                "from_backend_id": str(before.backend_id if before is not None else ""),
                "to_backend_id": str(after.backend_id),
                "from_model": str(before.model if before is not None else ""),
                "to_model": str(after.model),
                "from_max_context_tokens": int(before_limit),
                "to_max_context_tokens": int(after_limit),
                "required_context_tokens": requirement.total_tokens,
                "has_images": bool(has_images),
            },
        )

    @staticmethod
    async def _invoke_round(
        *,
        model_gateway: Any,
        request: CoreWakeRequest,
        route: Any,
        run_id: int,
        round_number: int,
        logical_step_number: int,
        prompt: Any,
        context_requirement: ModelContextRequirement,
        operation_timeout_seconds: float,
        agent_history: tuple[AIAgentTranscriptTurn, ...],
    ) -> Any:
        invocation_id = uuid.uuid4().hex
        foreground = request.source in {
            WakeSource.FOREGROUND_MESSAGE,
            WakeSource.DEFERRED_MESSAGE,
        }
        projected_message_ids = await MainCoreTextCommandLoop._project_model_visible_message_ids(
            model_gateway,
            run_id=run_id,
            message_ids=tuple(getattr(prompt, "source_message_ids", ()) or ()),
            summary_ids=tuple(getattr(prompt, "source_summary_ids", ()) or ()),
            summary_coverage=tuple(getattr(prompt, "source_summary_coverage", ()) or ()),
        )
        result = await model_gateway.invoke_model(
            AIModelRequest(
                invocation_id=invocation_id,
                work_purpose=AIWorkPurpose.MAIN_CORE,
                logical_stage_key=f"main-core-run:{run_id}",
                context_text=prompt.context_text,
                turn_text=prompt.turn_text,
                input_images=prompt.image_urls,
                prompt_cache_hint=prompt.prompt_cache_hint,
                agent_tools=main_core_agent_tools(),
                agent_history=agent_history,
                backend_ids=(),
                execution_mode=(
                    AIExecutionMode.FOREGROUND_SYNC
                    if foreground
                    else AIExecutionMode.BACKGROUND_DURABLE
                ),
                profile_id=request.profile_id,
                instance_id=str(request.instance_id or ""),
                owner_kind="roleplay_run",
                owner_id=str(run_id),
                idempotency_key=f"roleplay-run:{run_id}:round:{round_number}",
                retry_policy=AIRetryPolicy(
                    max_attempts=3,
                    backend_timeout_seconds=float(operation_timeout_seconds),
                ),
                metadata={
                    "capability": "chat.completion",
                    "routing_capability": "chat.completion",
                    "preferred_backend_id": str(route.preferred_backend_id or ""),
                    "run_id": run_id,
                    "round": round_number,
                    "context_requirement": {
                        "input_text_tokens": context_requirement.input_text_tokens,
                        "input_image_tokens": context_requirement.input_image_tokens,
                        "reserved_output_tokens": context_requirement.reserved_output_tokens,
                        "total_tokens": context_requirement.total_tokens,
                    },
                    "logical_model_step": logical_step_number,
                    "prompt_document": prompt.debug_payload(),
                },
            )
        )
        return result, invocation_id, projected_message_ids


__all__ = [
    "MainCoreFinalSubmissionAttempt",
    "MainCoreModelResult",
    "MainCoreStepRejectedThreeTimes",
    "MainCoreTextCommandLoop",
]
