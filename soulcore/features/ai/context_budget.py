"""Shared model-context accounting for routing and provider preflight."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...contracts.ai_models import AIBackendDescriptor, AIModelRequest
from ...shared.token_meter import ConservativeTokenMeter

DEFAULT_RESERVED_OUTPUT_TOKENS = 8192
IMAGE_INPUT_TOKENS = 1024


@dataclass(frozen=True, slots=True)
class ModelContextRequirement:
    input_text_tokens: int
    input_image_tokens: int
    reserved_output_tokens: int
    total_tokens: int


def configured_model_context_tokens(descriptor: AIBackendDescriptor | None) -> int | None:
    """Return one explicitly configured positive context window."""

    if descriptor is None:
        return None
    raw = descriptor.metadata.get("max_context_tokens")
    try:
        value = int(str(raw))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def reserved_output_tokens(
    parameters: Mapping[str, Any] | None,
    *,
    default: int = DEFAULT_RESERVED_OUTPUT_TOKENS,
) -> int:
    values = dict(parameters or {})
    raw = values.get("max_completion_tokens", values.get("max_tokens", default))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return max(1, int(default))


def estimate_model_context_requirement(
    *,
    input_text_tokens: int,
    input_image_count: int = 0,
    parameters: Mapping[str, Any] | None = None,
) -> ModelContextRequirement:
    text_tokens = max(0, int(input_text_tokens))
    image_tokens = max(0, int(input_image_count)) * IMAGE_INPUT_TOKENS
    output_tokens = reserved_output_tokens(parameters)
    return ModelContextRequirement(
        input_text_tokens=text_tokens,
        input_image_tokens=image_tokens,
        reserved_output_tokens=output_tokens,
        total_tokens=text_tokens + image_tokens + output_tokens,
    )


def measure_model_request_context(
    request: AIModelRequest,
    *,
    model_id: str = "",
) -> ModelContextRequirement:
    meter = ConservativeTokenMeter(model_id)
    return estimate_model_context_requirement(
        input_text_tokens=meter.count_text(request.logical_document),
        input_image_count=len(request.input_images),
        parameters=request.parameters,
    )


def available_prompt_tokens(
    max_context_tokens: int,
    *,
    input_image_count: int = 0,
    parameters: Mapping[str, Any] | None = None,
) -> int:
    overhead = estimate_model_context_requirement(
        input_text_tokens=0,
        input_image_count=input_image_count,
        parameters=parameters,
    )
    return int(max_context_tokens) - overhead.total_tokens


__all__ = [
    "DEFAULT_RESERVED_OUTPUT_TOKENS",
    "IMAGE_INPUT_TOKENS",
    "ModelContextRequirement",
    "available_prompt_tokens",
    "configured_model_context_tokens",
    "estimate_model_context_requirement",
    "measure_model_request_context",
    "reserved_output_tokens",
]
