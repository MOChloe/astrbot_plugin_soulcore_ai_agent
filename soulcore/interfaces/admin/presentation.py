"""Stable JSON views used by the administrator HTTP and command interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def ai_task_view(task: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(task)
    status = str(value.get("status") or "")
    task_type = str(value.get("task_type") or "").upper()
    actions: list[str] = []
    if status in {"SCHEDULED", "READY", "RETRY_WAIT", "RUNNING"}:
        actions.extend(["pause", "cancel"])
    elif status == "PAUSED":
        actions.extend(["resume", "cancel"])
    elif status in {"PAUSE_REQUESTED", "CANCEL_REQUESTED"}:
        actions.append("cancel")
    elif status in {"FAILED", "CANCELLED"} and task_type != "BACKGROUND_AUTHOR":
        actions.append("retry")
    elif status == "RECOVERY_REQUIRED":
        actions.append("cancel")
    value["allowed_actions"] = actions
    value["manual_retry_allowed"] = "retry" in actions
    value["automatic_retry_scheduled"] = status == "RETRY_WAIT"
    return jsonable(value)


def ai_task_run_ids(task: Mapping[str, Any]) -> set[int]:
    result: set[int] = set()
    run_keys = {
        "run_id",
        "core_run_id",
        "main_core_run_id",
    }

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key) in run_keys:
                    run_id = positive_int(child)
                    if run_id is not None:
                        result.add(run_id)
                else:
                    visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    for key in ("input", "checkpoint", "progress", "result"):
        visit(task.get(key))
    task_input = task.get("input")
    if isinstance(task_input, Mapping) and task_input.get("execution_mode"):
        run_id = positive_int(task_input.get("owner_id"))
        if run_id is not None:
            result.add(run_id)
    return result


def core_run_detail_view(run: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "run_id",
        "source",
        "status",
        "reason",
        "expected_state_epoch",
        "committed_state_epoch",
        "started_at",
        "finished_at",
        "error",
    )
    return jsonable({key: run.get(key) for key in keys})


def outbox_detail_view(item: Any) -> dict[str, Any]:
    value = jsonable(item)
    if not isinstance(value, Mapping):
        return {"value": value}
    keys = (
        "outbox_id",
        "umo",
        "status",
        "idempotency_key",
        "attempts",
        "activity_epoch",
        "last_error",
        "created_at",
        "updated_at",
    )
    return {key: value.get(key) for key in keys}
