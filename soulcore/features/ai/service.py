"""Stable composition entry points for the AI feature."""

from .command_protocol import (
    CommandExecutionResult,
    CommandParameter,
    CommandProtocolError,
    CommandSpec,
    MainCoreCommandRegistry,
    ModelVisibleCommandResult,
    ParsedCommand,
    ParsedModelTurn,
    ValidatedCommand,
    execute_nonterminal_batch,
    parse_model_turn,
    register_result_references,
    terminal_decision,
)
from .command_string_values import (
    boolean_validator,
    integer_validator,
    parse_boolean,
    parse_integer,
    parse_string_list,
    reference_validator,
    resolve_reference,
    resolve_reference_list,
)
from .context_budget import (
    DEFAULT_RESERVED_OUTPUT_TOKENS,
    ModelContextRequirement,
    available_prompt_tokens,
    configured_model_context_tokens,
    estimate_model_context_requirement,
)
from .diagnostics import classify_generic_error, safe_ai_failure_details
from .invocation_support import REQUEST_OPERATION_TIMEOUT_SECONDS_KEY
from .local_commands import MainCoreCommandSet
from .manager import AIManager
from .ports import (
    AIAdminQueryRepositoryPort,
    AIConfigurationRepositoryPort,
    AIRepositoryPort,
    DurableTaskRepositoryPort,
)
from .registry import (
    BackendPool,
    BackendRegistration,
    CapabilityAdapterRegistry,
    CapabilityRegistration,
    CircuitBreaker,
    CircuitPolicy,
)

__all__ = [
    "AIManager",
    "AIRepositoryPort",
    "AIAdminQueryRepositoryPort",
    "AIConfigurationRepositoryPort",
    "BackendPool",
    "BackendRegistration",
    "CapabilityAdapterRegistry",
    "CapabilityRegistration",
    "CircuitBreaker",
    "CircuitPolicy",
    "CommandExecutionResult",
    "CommandParameter",
    "CommandSpec",
    "CommandProtocolError",
    "DurableTaskRepositoryPort",
    "DEFAULT_RESERVED_OUTPUT_TOKENS",
    "MainCoreCommandRegistry",
    "ModelVisibleCommandResult",
    "MainCoreCommandSet",
    "ModelContextRequirement",
    "ParsedCommand",
    "ParsedModelTurn",
    "REQUEST_OPERATION_TIMEOUT_SECONDS_KEY",
    "ValidatedCommand",
    "classify_generic_error",
    "available_prompt_tokens",
    "boolean_validator",
    "configured_model_context_tokens",
    "estimate_model_context_requirement",
    "execute_nonterminal_batch",
    "parse_model_turn",
    "integer_validator",
    "parse_boolean",
    "parse_integer",
    "parse_string_list",
    "reference_validator",
    "register_result_references",
    "resolve_reference",
    "resolve_reference_list",
    "safe_ai_failure_details",
    "terminal_decision",
]
