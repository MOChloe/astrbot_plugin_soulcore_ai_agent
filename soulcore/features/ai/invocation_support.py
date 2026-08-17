from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any

from ...contracts.ai_models import (
    AIBackendDescriptor,
    AIBackendResponse,
    AICapabilityRequest,
    AICompletion,
    AIErrorInfo,
    AIImageGenerationOutput,
    AIModelRequest,
)
from .registry import CapabilityRegistration

REQUEST_OPERATION_TIMEOUT_SECONDS_KEY = "operation_timeout_seconds"


@dataclass(slots=True)
class InvocationState:
    request: AIModelRequest
    started_at: datetime
    capability: str
    circuit_enabled: bool
    attempts: int = 0
    last_error: AIErrorInfo | None = None
    failure_attempts: list[dict[str, Any]] = field(default_factory=list)
    stop_all: bool = False


class CapabilityBackendAdapter:
    def __init__(
        self,
        registration: CapabilityRegistration,
        *,
        request_override: AICapabilityRequest | None = None,
        reference_degraded: bool = False,
    ) -> None:
        self.registration = registration
        self.adapter_id = registration.descriptor.adapter_id
        self.capabilities: Sequence[str] = tuple(registration.adapter.capabilities)
        self.request_override = request_override
        self.reference_degraded = bool(reference_degraded)

    async def complete(
        self, request: AIModelRequest, backend: AIBackendDescriptor
    ) -> AIBackendResponse:
        capability_request = request.capability_request
        if not isinstance(capability_request, AICapabilityRequest):
            raise ValueError("capability request payload is missing")
        output = await self.registration.adapter.invoke(
            self.request_override or capability_request, backend
        )
        if self.reference_degraded and isinstance(output, AIImageGenerationOutput):
            output = replace(
                output,
                reference_mode="text",
                warnings=tuple(output.warnings) + ("reference_images_degraded_to_text",),
            )
        return AIBackendResponse(
            AICompletion(
                text="",
                finish_reason="capability",
                model=backend.model,
                capability_output=output,
            )
        )

    def classify_error(self, exc: BaseException, backend: AIBackendDescriptor) -> AIErrorInfo:
        return self.registration.adapter.classify_error(exc, backend)


__all__ = [
    "CapabilityBackendAdapter",
    "InvocationState",
    "REQUEST_OPERATION_TIMEOUT_SECONDS_KEY",
]
