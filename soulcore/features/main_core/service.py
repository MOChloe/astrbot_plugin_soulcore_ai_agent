"""Stable composition entry points for Main Core orchestration."""

from ...contracts.thinking import (
    DEFAULT_THINKING_POLICY,
    MainCoreThinkingPolicy,
    ThinkingComplexity,
    require_thinking_policy,
    thinking_policy_from_value,
    thinking_policy_options,
)
from .command_catalog import (
    build_main_core_commands,
    build_restricted_response_commands,
)
from .runner import MainCoreRunner, RunnerSettings

__all__ = [
    "DEFAULT_THINKING_POLICY",
    "MainCoreRunner",
    "MainCoreThinkingPolicy",
    "RunnerSettings",
    "ThinkingComplexity",
    "build_main_core_commands",
    "build_restricted_response_commands",
    "require_thinking_policy",
    "thinking_policy_from_value",
    "thinking_policy_options",
]
