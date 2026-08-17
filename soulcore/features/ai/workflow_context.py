"""Causal context shared by AI, web, media and durable work.

The context identifies one business workflow visible in advanced settings. It is
deliberately independent from durable task leases: a single workflow may span
many model calls and service steps, while a future timer creates a new workflow
and keeps only a causal link to the source.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AIWorkContext:
    workflow_id: int
    node_id: int | None = None
    node_role: str | None = None
    purpose: str | None = None


_CURRENT_AI_WORK_CONTEXT: ContextVar[AIWorkContext | None] = ContextVar(
    "soulcore_ai_work_trace",
    default=None,
)


def current_ai_work_context() -> AIWorkContext | None:
    return _CURRENT_AI_WORK_CONTEXT.get()


@contextmanager
def bind_ai_work_context(
    context: AIWorkContext | None,
) -> Iterator[AIWorkContext | None]:
    token = _CURRENT_AI_WORK_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_AI_WORK_CONTEXT.reset(token)


__all__ = [
    "AIWorkContext",
    "bind_ai_work_context",
    "current_ai_work_context",
]
