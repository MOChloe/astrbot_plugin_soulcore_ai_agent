from __future__ import annotations

import asyncio
import logging
from typing import Any

from ...contracts.ai_models import AIWorkPurpose
from ...contracts.models import CoreWakeRequest
from ..ai import current_ai_work_context

logger = logging.getLogger(__name__)


class MainCoreWorkRecordStageMixin:
    """Record polish and final-commit stages around their real operations."""

    async def _run_response_polish_stage(
        self,
        request: CoreWakeRequest,
        *,
        role: Any,
        run_id: int,
        state: Any,
        prepared: Any,
        raw_visible_steps: list[dict[str, Any]],
        addressed: bool,
        segment_index: int | None = None,
        working_text: str = "",
    ) -> Any:
        stage = await self._start_response_polish_stage(
            run_id=run_id,
            raw_visible_steps=raw_visible_steps,
            addressed=addressed,
            segment_index=segment_index,
        )
        try:
            polish = await self._execute_response_polish(
                request,
                role=role,
                run_id=run_id,
                state=state,
                prepared=prepared,
                raw_visible_steps=raw_visible_steps,
                addressed=addressed,
                stage=stage,
                working_text=working_text,
            )
        except asyncio.CancelledError:
            await self._cancel_recorded_stage(stage)
            raise
        except Exception as exc:
            await self._fail_recorded_stage(stage, exc)
            raise
        await self._finish_response_polish_stage(stage, polish)
        return polish

    async def _start_response_polish_stage(
        self,
        *,
        run_id: int,
        raw_visible_steps: list[dict[str, Any]],
        addressed: bool,
        segment_index: int | None = None,
    ) -> Any:
        parent = current_ai_work_context()
        if parent is None or parent.workflow_id <= 0:
            return None
        return await self.model_gateway.start_ai_work_node(
            workflow_id=parent.workflow_id,
            parent_node_id=parent.node_id,
            node_role="BUSINESS_STAGE",
            node_kind="MODEL",
            purpose=AIWorkPurpose.RESPONSE_POLISH.value,
            node_key=(
                f"main-core-run:{run_id}:response-polish"
                if segment_index is None
                else f"main-core-run:{run_id}:segment:{segment_index}:response-polish"
            ),
            input={"expression_steps": raw_visible_steps, "addressed": addressed},
        )

    async def _execute_response_polish(
        self,
        request: CoreWakeRequest,
        *,
        role: Any,
        run_id: int,
        state: Any,
        prepared: Any,
        raw_visible_steps: list[dict[str, Any]],
        addressed: bool,
        stage: Any,
        working_text: str = "",
    ) -> Any:
        values = {
            "role": role,
            "state": state,
            "run_id": run_id,
            "prepared": prepared,
            "original_steps": raw_visible_steps,
            "freeze_visible_topology": addressed,
            "working_text": working_text,
        }
        if stage is None:
            return await self._polish_expression_steps(request, **values)
        with self.model_gateway.bind_ai_workflow(stage):
            return await self._polish_expression_steps(request, **values)

    async def _finish_response_polish_stage(self, stage: Any, polish: Any) -> None:
        if stage is None or stage.node_id is None:
            return
        status = str(polish.audit.get("status") or "FAILED").upper()
        summaries = {
            "SUCCEEDED": "AI 润色成功",
            "SKIPPED": "AI 润色已跳过",
            "FALLBACK": "AI 润色降级并沿用 MainCore 原文",
        }
        summary = summaries.get(status, "AI 润色未完成")
        if status == "SKIPPED" and polish.audit.get("reason") == "FEATURE_DISABLED":
            summary = "回复润色未启用，已直接采用 MainCore 表达"
        await self.model_gateway.finish_ai_work_node(
            stage.node_id,
            status=status,
            warning_code=(
                str(polish.audit.get("reason") or "response_polish_fallback")
                if status == "FALLBACK"
                else ""
            ),
            warning_message=(
                "润色阶段未能生成可用结果，已沿用 MainCore 原文" if status == "FALLBACK" else ""
            ),
            summary=summary,
            result={
                "audit": dict(polish.audit),
                "expression_steps": [dict(item) for item in polish.visible_steps],
            },
        )

    async def _run_final_result_commit_stage(self, **commit_values: Any) -> bool:
        run_id = int(commit_values["run_id"])
        actions = list(commit_values["actions"])
        stage = await self._start_final_result_commit_stage(run_id, len(actions))
        try:
            committed = await self._commit_with_recorded_stage(stage, commit_values)
        except asyncio.CancelledError:
            await self._cancel_recorded_stage(stage)
            raise
        except Exception as exc:
            await self._fail_recorded_stage(stage, exc)
            raise
        try:
            await self._finish_final_result_commit_stage(stage, committed, len(actions))
        except Exception:
            logger.exception(
                "failed to finish MainCore final-commit work record after transaction result"
            )
        return committed

    async def _commit_with_recorded_stage(self, stage: Any, commit_values: dict[str, Any]) -> bool:
        if stage is None:
            return await self._commit_finalized(**commit_values)
        with self.model_gateway.bind_ai_workflow(stage):
            return await self._commit_finalized(**commit_values)

    async def _start_final_result_commit_stage(self, run_id: int, action_count: int) -> Any:
        parent = current_ai_work_context()
        if parent is None or parent.workflow_id <= 0:
            return None
        return await self.model_gateway.start_ai_work_node(
            workflow_id=parent.workflow_id,
            parent_node_id=parent.node_id,
            node_role="SYSTEM_STAGE",
            node_kind="SYSTEM",
            purpose="FINAL_RESULT_COMMIT",
            node_key=f"main-core-run:{run_id}:final-result-commit",
            input={"outbound_action_count": action_count},
        )

    async def _finish_final_result_commit_stage(
        self, stage: Any, committed: bool, action_count: int
    ) -> None:
        if stage is None or stage.node_id is None:
            return
        await self.model_gateway.finish_ai_work_node(
            stage.node_id,
            status="SUCCEEDED" if committed else "INTERRUPTED",
            error_code="" if committed else "RUN_STATE_CHANGED",
            error_message="" if committed else "最终事务因运行状态变化未提交",
            summary="最终结果事务已提交" if committed else "最终结果事务未提交",
            result={"committed": committed, "outbound_action_count": action_count},
        )
        if committed:
            await self._record_commit_event(stage, action_count)

    async def _record_commit_event(self, stage: Any, action_count: int) -> None:
        await self.model_gateway.record_ai_work_event(
            workflow_id=stage.workflow_id,
            node_id=stage.node_id,
            event_category="TRANSACTION",
            severity="INFO",
            code="OUTBOX_COMMITTED" if action_count else "FINAL_RESULT_COMMITTED",
            summary="待发送消息已在最终事务中写入" if action_count else "最终静默结果已提交",
            details={"outbound_action_count": action_count},
        )

    async def _cancel_recorded_stage(self, stage: Any) -> None:
        if stage is None or stage.node_id is None:
            return
        await asyncio.shield(
            self.model_gateway.finish_ai_work_node(stage.node_id, status="CANCELLED")
        )

    async def _fail_recorded_stage(self, stage: Any, exc: Exception) -> None:
        if stage is None or stage.node_id is None:
            return
        await self.model_gateway.finish_ai_work_node(
            stage.node_id,
            status="FAILED",
            error_code=type(exc).__name__,
            error_message=str(exc),
        )


__all__ = ["MainCoreWorkRecordStageMixin"]
