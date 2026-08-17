"""Stable durable-task identity for one Timer occurrence generation."""

from __future__ import annotations

import hashlib


def timer_run_task_idempotency_key(
    profile_id: str,
    instance_id: str,
    occurrence_id: str,
    generation: int,
) -> str:
    raw = f"{profile_id}:{instance_id}:{occurrence_id}:{int(generation)}"
    return f"timer-run:{hashlib.sha256(raw.encode()).hexdigest()}"


__all__ = ["timer_run_task_idempotency_key"]
