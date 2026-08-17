"""Public AI feature API."""

from .diagnostics import classify_generic_error, safe_ai_failure_details
from .local_commands import MainCoreCommandSet
from .manager import AIManager
from .registry import BackendPool, CapabilityAdapterRegistry, CircuitBreaker
from .structured_validation import (
    StructuredOutputRejectedThreeTimes,
    record_structured_acceptance,
    record_structured_rejection,
    run_structured_text_session,
)
from .workflow_context import AIWorkContext, bind_ai_work_context, current_ai_work_context

__all__ = [
    "AIManager",
    "AIWorkContext",
    "BackendPool",
    "CapabilityAdapterRegistry",
    "CircuitBreaker",
    "MainCoreCommandSet",
    "StructuredOutputRejectedThreeTimes",
    "bind_ai_work_context",
    "classify_generic_error",
    "current_ai_work_context",
    "record_structured_acceptance",
    "record_structured_rejection",
    "run_structured_text_session",
    "safe_ai_failure_details",
]
