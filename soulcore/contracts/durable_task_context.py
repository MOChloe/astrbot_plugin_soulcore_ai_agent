"""Task-local identity shared by durable execution and its prerequisites."""

from contextvars import ContextVar

_current_task_id: ContextVar[int | None] = ContextVar(
    "soulcore_current_durable_ai_task_id",
    default=None,
)


def current_durable_ai_task_id() -> int | None:
    return _current_task_id.get()


__all__ = ["_current_task_id", "current_durable_ai_task_id"]
