"""Run-local planning command for multi-round Main Core actions."""

from __future__ import annotations

from typing import Any

from ..ai.service import ModelVisibleCommandResult
from .command_catalog_support import command, parameter
from .command_context import _active
from .command_outcomes import command_outcome_handler


def set_run_plan(_event: Any, content: str) -> str | ModelVisibleCommandResult:
    """Replace the current Main Core run's plan without persisting it."""

    plan = str(content or "").strip()
    if not plan:
        return "error: Plan内容不能为空；原有Plan保持不变。"
    _active().current_plan = plan
    return ModelVisibleCommandResult("Plan 已保存；下一轮继续。")


def run_plan_command() -> object:
    return command(
        "制定Plan",
        "set_run_plan",
        "为本次行动确定最终要形成的对方可见表达、作品或数据，以及它的内容组成、表达方式、尚需完成的行动和完成标准。",
        command_outcome_handler("set_run_plan", set_run_plan),
        parameter(
            "内容",
            "content",
            required=True,
            prompt_hint=(
                "最终目标与完成标准；准备采用的内容、结构、语气或风格；尚需查证或执行的事项"
            ),
            identity_mode="literal",
        ),
        serial=True,
        usage_guidance=(
            "写清最终目标、完成标准、关键取舍和仍需取得的结果；未知结果不写成事实。"
            "再次使用会整体替换当前 Plan。"
        ),
    )


__all__ = ["run_plan_command", "set_run_plan"]
