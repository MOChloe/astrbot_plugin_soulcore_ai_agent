"""Strict Main Core Timer commands; mutation waits for the final transaction."""

from __future__ import annotations

from typing import Any

from ..timers.service import MAX_SEMANTIC_CANDIDATES, TimerDomainError
from .command_context import _active


async def remember_future(
    _event: Any,
    time_expression: str,
    action_text: str,
) -> Any:
    context = _timer_context()
    if context is None:
        return "error: 本轮无法使用安排功能"
    try:
        return await context.stage_natural_creation(
            time_expression=time_expression,
            action_text=action_text,
        )
    except TimerDomainError as exc:
        return f"error: 未来安排未记下（{_timer_error_text(exc)}）"
    except (TypeError, ValueError):
        return "error: 未来安排未记下（参数无效）"


async def list_arrangements(
    _event: Any,
    query: str = "",
) -> Any:
    context = _timer_context()
    if context is None:
        return "error: 本轮无法使用安排功能"
    try:
        return await context.list_arrangements(limit=MAX_SEMANTIC_CANDIDATES, query=query)
    except TimerDomainError as exc:
        return f"error: 安排没有查看成功（{_timer_error_text(exc)}）"
    except (TypeError, ValueError):
        return "error: 安排没有查看成功（参数无效）"


async def adjust_arrangement(
    _event: Any,
    target: str,
    change: str,
) -> Any:
    context = _timer_context()
    if context is None:
        return "error: 本轮无法使用安排功能"
    try:
        return await context.stage_natural_adjustment(target=target, change=change)
    except TimerDomainError as exc:
        return f"error: 安排没有调整（{_timer_error_text(exc)}）"
    except (TypeError, ValueError):
        return "error: 安排没有调整（参数无效）"


def _timer_context() -> Any | None:
    return _active().timer_command_context


def _timer_error_text(error: TimerDomainError) -> str:
    return {
        "INVALID_REFERENCE": "引用无效或已经过期",
        "INVALID_PROMPT": "到时候做的事无效",
        "INVALID_RULE": "安排的时间或内容无效",
        "UNSUPPORTED_RULE": "暂不支持这种安排",
        "INVALID_TIMEZONE": "时区无效",
        "OUT_OF_RANGE": "数值超出允许范围",
        "INVALID_STATE": "当前状态不允许这个操作",
        "VERSION_CONFLICT": "安排已经变化，请重新查看",
        "SCOPE_MISMATCH": "安排不属于当前会话",
        "LIMIT_EXCEEDED": "本轮安排操作已达到上限",
    }.get(error.code.value, "参数或当前状态不允许")


__all__ = [
    "adjust_arrangement",
    "list_arrangements",
    "remember_future",
]
