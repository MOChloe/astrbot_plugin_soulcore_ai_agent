"""Command outcome tracking and automatic work-state terminal validation."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

from ...shared.event_log import record_event
from .command_context import (
    _active,
    _begin_command_outcome_scope,
    _command_outcome_was_recorded,
    _end_command_outcome_scope,
    _record_command_outcome,
)
from .command_result_projection import project_command_result
from .terminal_decision import commit_main_core_response
from .work_continuity import validate_terminal_work


async def commit_main_core_response_with_work_validation(
    _event: Any,
    expression_steps: list[dict[str, Any]] | None = None,
    memo: str = "",
    no_op: bool = False,
    temporary_absence: dict[str, Any] | None = None,
) -> str | None:
    """Validate hidden work state before accepting the final decision."""

    collector = _active()
    visible, persistent_text = _commit_texts(
        expression_steps=expression_steps,
    )
    work_error = validate_terminal_work(
        collector,
        has_visible_output=visible,
        visible_text=persistent_text,
    )
    if work_error:
        return work_error
    return await commit_main_core_response(
        _event,
        expression_steps=expression_steps,
        memo=memo,
        no_op=no_op,
        temporary_absence=temporary_absence,
    )


def command_outcome_handler(name: str, handler: Callable[..., Any]) -> Callable[..., Any]:
    """Record a compact real outcome without changing the command contract."""

    @wraps(handler)
    async def tracked(*args: Any, **kwargs: Any) -> Any:
        collector = _active()
        token = _begin_command_outcome_scope()
        try:
            result = handler(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
        except asyncio.CancelledError:
            if not _command_outcome_was_recorded():
                _record_command_outcome(collector, name, ok=False, error="CANCELLED")
            raise
        except Exception as exc:
            if not _command_outcome_was_recorded():
                _record_command_outcome(
                    collector,
                    name,
                    ok=False,
                    error=f"EXCEPTION_{type(exc).__name__.upper()}",
                )
            logging.getLogger(__name__).exception("Main Core command execution failed")
            if collector.event_log is not None:
                await record_event(
                    collector.event_log,
                    profile_id=collector.profile_id,
                    instance_id=collector.instance_id,
                    level="ERROR",
                    category="main_core.command",
                    message="主 Core 指令执行异常",
                    details={
                        "command": name,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else "",
                    },
                )
            raise
        finally:
            recorded = _command_outcome_was_recorded()
            _end_command_outcome_scope(token)
        if not recorded:
            ok, error = _compact_result_status(result)
            _record_command_outcome(collector, name, ok=ok, error=error)
        return project_command_result(name, result)

    return tracked


def _compact_result_status(result: Any) -> tuple[bool, str]:
    payload = result
    if isinstance(result, str):
        text = result.strip()
        if text.lower().startswith("error:"):
            return False, "COMMAND_RETURNED_ERROR"
        return True, ""
    if isinstance(payload, dict) and payload.get("ok") is False:
        return False, str(payload.get("error") or "COMMAND_RETURNED_ERROR")[:120]
    return True, ""


def _commit_texts(
    *,
    expression_steps: list[dict[str, Any]] | None,
) -> tuple[bool, str]:
    values = list(expression_steps or [])
    text_values: list[str] = []
    has_visible = False
    for item in values:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "TEXT").strip().upper()
        if kind != "RETRACT":
            has_visible = True
        text_values.append(str(item.get("text") or ""))
    return has_visible, "\n".join(value for value in text_values if value)


__all__ = [
    "commit_main_core_response_with_work_validation",
    "command_outcome_handler",
]
