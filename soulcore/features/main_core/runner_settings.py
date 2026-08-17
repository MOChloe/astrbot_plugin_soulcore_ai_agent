from __future__ import annotations

from dataclasses import dataclass

from ...contracts.ai_models import DEFAULT_AI_OPERATION_TIMEOUT_SECONDS
from ...contracts.thinking import (
    DEFAULT_THINKING_POLICY,
    MainCoreThinkingPolicy,
)


@dataclass(slots=True)
class RunnerSettings:
    thinking_policy: MainCoreThinkingPolicy = DEFAULT_THINKING_POLICY
    command_parallel_calls: int = 8
    command_timeout_seconds: int = int(DEFAULT_AI_OPERATION_TIMEOUT_SECONDS)

    def current_thinking_policy(self) -> MainCoreThinkingPolicy:
        return self.thinking_policy


__all__ = ["RunnerSettings"]
