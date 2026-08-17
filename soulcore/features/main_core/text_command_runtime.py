"""Tracing and small runtime helpers for the Main Core text command loop."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ...contracts.models import CoreWakeRequest
from ...shared.prompt_document import (
    TrustedPromptMarkup,
    join_prompt_markup,
    prompt_markup_block,
)
from ..ai import current_ai_work_context
from ..ai.service import CommandExecutionResult
from .roleplay_prompt import ExecutionRound

NO_NEW_MAIN_CORE_ACTION = (
    "这一轮没有新的可执行动作：这些调用和数据已经在本次行动中处理过。"
    "请直接使用前面返回的完整结果继续推进，只提交参数不同的新调用；"
    "需要回复对方时，请形成尚未发送的新表达。"
)


class MainCoreStepRejectedThreeTimes(RuntimeError):
    code = "main_core_step_rejected_three_times"

    def __init__(self, errors: list[str], rounds: list[ExecutionRound]) -> None:
        self.errors = tuple(errors)
        self.rounds = tuple(rounds)
        super().__init__(f"{self.code}: " + " | ".join(errors[-3:]))


class TextCommandRuntimeMixin:
    @staticmethod
    def _identity_run_context(prepared_context: Any) -> tuple[Any, Any, str]:
        if prepared_context is None:
            return None, None, "profile"
        identity_context = prepared_context.identity_context
        scope = identity_context.scope if identity_context is not None else "profile"
        return prepared_context.identity_catalog, identity_context, str(scope)

    @staticmethod
    @asynccontextmanager
    async def _command_trace_scope(
        model_gateway: Any,
        item: Any,
        *,
        run_id: int,
        round_number: int,
        runtime_gate: Any | None,
        profile_id: str,
        instance_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        await TextCommandRuntimeMixin._require_enabled(runtime_gate, profile_id, instance_id)
        parent = current_ai_work_context()
        state: dict[str, Any] = {}
        if parent is None or parent.workflow_id <= 0:
            yield state
            return
        trace = await model_gateway.start_ai_work_node(
            workflow_id=parent.workflow_id,
            parent_node_id=parent.node_id,
            node_role="INTERNAL_ACTION",
            node_kind="COMMAND",
            purpose=f"COMMAND:{item.spec.name}",
            node_key=(f"main-core-command:{run_id}:{round_number}:{item.parsed.ordinal}"),
            input={
                "command": item.spec.name,
                "ordinal": item.parsed.ordinal,
                "parameters": dict(item.parsed.parameters),
            },
        )
        if trace is None or trace.node_id is None:
            yield state
            return
        try:
            with model_gateway.bind_ai_workflow(trace):
                yield state
        finally:
            result = state.get("result")
            if result is not None and not isinstance(result, CommandExecutionResult):
                raise TypeError("command execution scope received an invalid result")
            ok = bool(result.ok) if result is not None else False
            await model_gateway.finish_ai_work_node(
                int(trace.node_id),
                status="SUCCEEDED" if ok else "FAILED",
                error_code="" if ok else "COMMAND_FAILED",
                error_message=(
                    "" if ok else str(result.content if result is not None else "指令执行失败")
                ),
                result=(
                    {
                        "ordinal": result.ordinal,
                        "command": item.spec.name,
                        "ok": ok,
                        "summary": str(result.content),
                    }
                    if result is not None
                    else None
                ),
            )

    @staticmethod
    async def _require_enabled(
        runtime_gate: Any | None,
        profile_id: str,
        instance_id: str = "",
    ) -> None:
        if runtime_gate is not None:
            await runtime_gate.require_enabled(profile_id, instance_id)

    @staticmethod
    @asynccontextmanager
    async def _terminal_trace_scope(
        model_gateway: Any,
        commands: list[Any],
        *,
        run_id: int,
        round_number: int,
    ) -> AsyncIterator[dict[str, Any]]:
        parent = current_ai_work_context()
        state: dict[str, Any] = {}
        if parent is None or parent.workflow_id <= 0:
            yield state
            return
        trace = await model_gateway.start_ai_work_node(
            workflow_id=parent.workflow_id,
            parent_node_id=parent.node_id,
            node_role="INTERNAL_ACTION",
            node_kind="COMMAND",
            purpose="FINAL_EXPRESSION_CONFIRMATION",
            node_key=(
                f"main-core-terminal:{run_id}:{round_number}:"
                f"{commands[0].parsed.ordinal if commands else 0}"
            ),
            input={
                "commands": [item.spec.name for item in commands],
                "message_count": sum(item.spec.send_kind != "SILENT" for item in commands),
            },
        )
        if trace is None or trace.node_id is None:
            yield state
            return
        try:
            with model_gateway.bind_ai_workflow(trace):
                yield state
        finally:
            ok = bool(state.get("ok", False))
            result = str(state.get("result") or "终止批次没有完成")
            await model_gateway.finish_ai_work_node(
                int(trace.node_id),
                status="SUCCEEDED" if ok else "FAILED",
                error_code="" if ok else "FINAL_EXPRESSION_REJECTED",
                error_message="" if ok else result,
                result={"ok": ok, "summary": result},
            )
            await model_gateway.record_ai_work_event(
                workflow_id=parent.workflow_id,
                node_id=int(trace.node_id),
                event_category="VALIDATION",
                severity="INFO" if ok else "ERROR",
                code="final_expression_confirmed" if ok else "final_expression_rejected",
                summary="最终表达已经确认" if ok else result,
                details={},
            )

    @staticmethod
    def _model_id(route: Any) -> str:
        if route.backend_hint is None:
            return str(route.preferred_backend_id)
        return str(
            route.backend_hint.model or route.backend_hint.backend_id or route.preferred_backend_id
        )

    @staticmethod
    def _runtime_notes(collector: Any, request: CoreWakeRequest) -> TrustedPromptMarkup:
        del collector
        foreground = str(request.metadata.get("foreground_context_notes") or "")
        blocks: list[TrustedPromptMarkup] = []
        if foreground:
            blocks.append(prompt_markup_block("当前输入补充", foreground))
        return join_prompt_markup(blocks)

    @staticmethod
    def _reject(errors: list[str], error: str, rounds: list[ExecutionRound]) -> None:
        errors.append(str(error or "协议校验失败").strip())
        if len(errors) >= 3 and len(set(errors[-3:])) == 1:
            raise MainCoreStepRejectedThreeTimes(errors, rounds)

    @staticmethod
    async def _commit_terminal(
        command_set: Any,
        event: Any,
        decision: dict[str, Any],
    ) -> str:
        handler = command_set.terminal_handler
        if not callable(handler):
            raise RuntimeError("终止提交处理器未注册")
        value = handler(
            event,
            **decision,
        )
        if inspect.isawaitable(value):
            value = await value
        return str(value or "")


__all__ = [
    "MainCoreStepRejectedThreeTimes",
    "NO_NEW_MAIN_CORE_ACTION",
    "TextCommandRuntimeMixin",
]
