"""Stable domain contracts for SoulCore AI and external capability calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

DEFAULT_AI_OPERATION_TIMEOUT_SECONDS = 300.0


def ai_utc_now() -> datetime:
    return datetime.now(UTC)


class AIExecutionMode(StrEnum):
    FOREGROUND_SYNC = "FOREGROUND_SYNC"
    BACKGROUND_DURABLE = "BACKGROUND_DURABLE"
    DEBUG_EPHEMERAL = "DEBUG_EPHEMERAL"


class AIErrorCode(StrEnum):
    INVALID_REQUEST = "INVALID_REQUEST"
    BACKEND_NOT_FOUND = "BACKEND_NOT_FOUND"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    AUTHENTICATION = "AUTHENTICATION"
    PERMISSION = "PERMISSION"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    RATE_LIMIT = "RATE_LIMIT"
    NETWORK = "NETWORK"
    REMOTE_5XX = "REMOTE_5XX"
    TIMEOUT = "TIMEOUT"
    EMPTY_OUTPUT = "EMPTY_OUTPUT"
    CONTEXT_BUDGET = "CONTEXT_BUDGET"
    OUTPUT_CONTRACT = "OUTPUT_CONTRACT"
    SAFETY_REFUSAL = "SAFETY_REFUSAL"
    COMMAND_TIMEOUT = "COMMAND_TIMEOUT"
    COMMAND_FAILED = "COMMAND_FAILED"
    COMMAND_PROTOCOL = "COMMAND_PROTOCOL"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    CAPACITY_BUSY = "CAPACITY_BUSY"
    ADAPTER_INCOMPATIBLE = "ADAPTER_INCOMPATIBLE"
    PROMPT_CACHE_MARKER_UNSUPPORTED = "PROMPT_CACHE_MARKER_UNSUPPORTED"
    INTERNAL = "INTERNAL"


class AIWorkPurpose(StrEnum):
    """Administrator-visible purpose of one causal AI business stage.

    This is deliberately independent from ``owner_kind`` and routing
    capability.  Ownership answers who requested the work; this enum answers
    what the work means to an administrator.
    """

    MAIN_CORE = "MAIN_CORE"
    RESPONSE_POLISH = "RESPONSE_POLISH"
    CONVERSATION_SUMMARY = "CONVERSATION_SUMMARY"
    MEMORY_REASONING = "MEMORY_REASONING"
    KNOWLEDGE_ORGANIZATION = "KNOWLEDGE_ORGANIZATION"
    CHARACTER_PROFILE_IMPORT = "CHARACTER_PROFILE_IMPORT"
    TURN_CLASSIFICATION = "TURN_CLASSIFICATION"
    GROUP_INTERJECTION = "GROUP_INTERJECTION"
    GROUP_REPLY_RELOCATION = "GROUP_REPLY_RELOCATION"
    BACKGROUND_WORLD = "BACKGROUND_WORLD"
    BACKGROUND_LIFE_DIRECTION = "BACKGROUND_LIFE_DIRECTION"
    BACKGROUND_STORY_SOURCE = "BACKGROUND_STORY_SOURCE"
    BACKGROUND_KEYFRAME = "BACKGROUND_KEYFRAME"
    BACKGROUND_ORDINARY = "BACKGROUND_ORDINARY"
    FILE_GENERATION = "FILE_GENERATION"
    IMAGE_GENERATION = "IMAGE_GENERATION"
    IMAGE_UNDERSTANDING = "IMAGE_UNDERSTANDING"
    AUDIO_TRANSCRIPTION = "AUDIO_TRANSCRIPTION"
    AUDIO_SPEECH_GENERATION = "AUDIO_SPEECH_GENERATION"
    WEB_SEARCH = "WEB_SEARCH"
    WEB_READ = "WEB_READ"
    WEB_IMAGE_SEARCH = "WEB_IMAGE_SEARCH"
    STICKER_COLLECTION = "STICKER_COLLECTION"
    STICKER_CHECK = "STICKER_CHECK"
    TIMER_RUN = "TIMER_RUN"
    TIMER_LIFECYCLE_REVIEW = "TIMER_LIFECYCLE_REVIEW"
    ADMIN_MODEL_TEST = "ADMIN_MODEL_TEST"
    ADMIN_WEB_TEST = "ADMIN_WEB_TEST"
    MODEL_REQUEST = "MODEL_REQUEST"


class AIBackendState(StrEnum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"
    DISABLED = "DISABLED"


class AITaskStatus(StrEnum):
    SCHEDULED = "SCHEDULED"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSE_REQUESTED = "PAUSE_REQUESTED"
    PAUSED = "PAUSED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    RETRY_WAIT = "RETRY_WAIT"
    DEFERRED = "DEFERRED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


class AICapabilityEffect(StrEnum):
    READ_ONLY = "READ_ONLY"
    IDEMPOTENT_WRITE = "IDEMPOTENT_WRITE"
    NON_IDEMPOTENT_WRITE = "NON_IDEMPOTENT_WRITE"


class AICapabilityName(StrEnum):
    """Capability identifiers owned by the AI manager.

    Keeping these names in the domain layer prevents UI, workers and adapters
    from slowly developing incompatible spellings.
    """

    VISION_DESCRIBE = "vision.describe"
    IMAGE_GENERATE = "image.generate"
    AUDIO_TRANSCRIBE = "audio.transcribe"
    AUDIO_SPEECH = "audio.speech"
    WEB_SEARCH = "web.search"
    WEB_READ = "web.read"


@dataclass(frozen=True, slots=True)
class AIImageContent:
    """A transport-neutral image reference.

    Exactly one of ``data`` and ``url`` is normally present. ``asset_id`` is
    optional because provider adapters must not know about media persistence.
    """

    mime_type: str = "image/png"
    data: bytes = field(default=b"", repr=False, compare=False)
    url: str = ""
    asset_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AIVisionDescription:
    """Pixel evidence plus an evidence-bound social impression.

    ``visible_facts`` is the authoritative picture description.  Conversation
    history wording is derived at the media boundary, never supplied by a
    vision provider. ``social_impression`` is a stable impression conveyed by
    an obvious sticker, reaction image, or meme; it never states the current
    sender's intent. ``transient_source_marker_present`` is consumed only by
    the current quality check and must never be persisted.
    """

    visible_facts: str
    ocr_text: str = ""
    subject_identity: str = ""
    sequence_observation: str = ""
    visual_style: str = ""
    sticker_type: str = ""
    social_impression: str = ""
    visible_text_state: str = ""
    safe: bool | None = None
    safety_reason: str = ""
    transient_source_marker_present: bool | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    model: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class AIImageGenerationOutput:
    images: tuple[AIImageContent, ...]
    model: str = ""
    revised_prompt: str = ""
    reference_mode: str = "none"
    warnings: tuple[str, ...] = ()
    provider_metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class AIImageBackendCapabilities:
    """Explicit feature declaration for one configured generation backend."""

    text_to_image: bool = True
    reference_image: bool = False
    multiple_references: bool = False
    supported_ratios: tuple[str, ...] = ()
    supported_sizes: tuple[str, ...] = ()
    maximum_outputs: int = 1
    output_format: str = "image"


@dataclass(frozen=True, slots=True)
class AIAudioContent:
    """Transport-neutral in-memory audio.

    Audio bytes deliberately remain available to the next runtime boundary,
    while their representation and equality never copy or expose the payload.
    Provider adapters may use ``filename`` only as a multipart filename; it is
    not a durable source reference.
    """

    data: bytes = field(default=b"", repr=False, compare=False)
    mime_type: str = "application/octet-stream"
    filename: str = field(default="", repr=False, compare=False)
    duration_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class AITranscriptionResult:
    text: str
    language: str = ""
    duration_seconds: float | None = None
    model: str = ""
    provider_metadata: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, slots=True)
class AISpeechResult:
    audio: AIAudioContent
    model: str = ""
    voice: str = ""
    provider_metadata: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, slots=True)
class AIErrorInfo:
    code: AIErrorCode
    safe_message: str
    retryable: bool = False
    switch_backend: bool = False
    open_circuit: bool = False
    retry_after_seconds: float | None = None
    backend_id: str = ""
    phase: str = "invoke"
    status_code: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


class AIInvocationError(RuntimeError):
    def __init__(self, info: AIErrorInfo, *, cause: BaseException | None = None) -> None:
        super().__init__(info.safe_message)
        self.info = info
        self.__cause__ = cause


@dataclass(frozen=True, slots=True)
class AIRetryPolicy:
    max_attempts: int = 3
    backend_timeout_seconds: float = DEFAULT_AI_OPERATION_TIMEOUT_SECONDS

    def normalized(self) -> AIRetryPolicy:
        return AIRetryPolicy(
            max_attempts=max(1, min(10, int(self.max_attempts))),
            backend_timeout_seconds=max(1.0, float(self.backend_timeout_seconds)),
        )


@dataclass(frozen=True, slots=True)
class AIBackendDescriptor:
    backend_id: str
    adapter_id: str
    model: str = ""
    priority: int = 100
    enabled: bool = True
    credential_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AIPromptCacheWireMode(StrEnum):
    DISABLED = "DISABLED"
    OPENAI_AUTO = "OPENAI_AUTO"
    OPENAI_EXPLICIT = "OPENAI_EXPLICIT"
    ANTHROPIC_EPHEMERAL = "ANTHROPIC_EPHEMERAL"


class AIPromptCacheState(StrEnum):
    UNTESTED = "UNTESTED"
    PROBING = "PROBING"
    ACCEPTED_UNVERIFIED = "ACCEPTED_UNVERIFIED"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


class AIPromptCacheSection(StrEnum):
    """Logical prompt section containing a transport cache breakpoint."""

    CONTEXT = "context"
    TURN = "turn"


class AIPromptCacheSemanticKind(StrEnum):
    """Stable meanings used to choose and diagnose prompt-cache boundaries."""

    PROTOCOL = "PROTOCOL"
    CONTEXT = "CONTEXT"
    PREVIOUS_DIALOGUE = "PREVIOUS_DIALOGUE"
    CURRENT_DIALOGUE = "CURRENT_DIALOGUE"
    PREVIOUS_RUN = "PREVIOUS_RUN"
    CURRENT_RUN = "CURRENT_RUN"


@dataclass(frozen=True, slots=True)
class AIPromptCacheBreakpoint:
    """One exact prefix boundary in a logical prompt document.

    ``section_end`` is local to ``context_text`` or ``turn_text`` so provider
    adapters can split their own wire blocks without reconstructing the
    prompt. ``document_end`` and the cumulative digest describe the same
    position in ``AIModelRequest.logical_document`` for diagnostics.
    """

    boundary_id: str
    section: AIPromptCacheSection
    semantic_kind: AIPromptCacheSemanticKind
    section_end: int
    document_end: int
    prefix_tokens: int
    prefix_hash: str
    selection_slot: int
    selection_reason: str = ""
    block_name: str = ""


@dataclass(frozen=True, slots=True)
class AIPromptCacheHint:
    """Transport-neutral cache manifest for one complete logical document."""

    prompt_protocol_version: str = "soulcore-prompt-v2"
    candidates: tuple[AIPromptCacheBreakpoint, ...] = ()
    selected: tuple[AIPromptCacheBreakpoint, ...] = ()
    rebase_reasons: tuple[str, ...] = ()

    @property
    def eligible(self) -> bool:
        return (
            bool(self.selected)
            and max((item.prefix_tokens for item in self.selected), default=0) >= 1024
        )


@dataclass(frozen=True, slots=True)
class AIPromptCachePolicy:
    """One physical request's negotiated cache wire behavior."""

    wire_mode: AIPromptCacheWireMode = AIPromptCacheWireMode.DISABLED
    candidate_mode: AIPromptCacheWireMode = AIPromptCacheWireMode.DISABLED
    state: AIPromptCacheState = AIPromptCacheState.UNTESTED
    cache_key: str = ""
    requested_ttl: str = ""
    actual_ttl: str = ""
    breakpoints: tuple[AIPromptCacheBreakpoint, ...] = ()
    probing: bool = False
    suppression_reason: str = ""
    quality_predecessor_id: str = ""
    quality_started_at: str = ""

    @property
    def carries_explicit_marker(self) -> bool:
        return self.wire_mode in {
            AIPromptCacheWireMode.OPENAI_EXPLICIT,
            AIPromptCacheWireMode.ANTHROPIC_EPHEMERAL,
        }


@dataclass(frozen=True, slots=True)
class AIAgentToolDefinition:
    """One SoulCore-owned string channel exposed through a Provider tool envelope."""

    name: str
    description: str


@dataclass(frozen=True, slots=True)
class AIAgentOutputItem:
    """One model-visible assistant output item retained for exact replay."""

    kind: str
    text: str = ""
    name: str = ""
    call_id: str = ""
    raw_arguments: str = ""
    argument_error: str = ""
    provider_item: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    provider_protocol: str = ""


@dataclass(frozen=True, slots=True)
class AIAgentToolResult:
    """The complete string returned for one Provider-native tool call."""

    name: str
    call_id: str
    text: str


@dataclass(frozen=True, slots=True)
class AIAgentTranscriptTurn:
    """One completed MainCore agent exchange in Provider-neutral form."""

    output_items: tuple[AIAgentOutputItem, ...]
    result_text: str
    tool_results: tuple[AIAgentToolResult, ...] = ()
    transport_mode: str = "text_envelope"
    source_round_number: int = 0
    contains_plan: bool = False
    unresolved_failure: bool = False
    public_references: tuple[str, ...] = ()
    contains_image_material: bool = False


@dataclass(frozen=True, slots=True)
class AIModelRequest:
    invocation_id: str
    work_purpose: AIWorkPurpose
    logical_stage_key: str = ""
    managed_work_stage: bool = False
    context_text: str = ""
    turn_text: str = ""
    input_images: tuple[str, ...] = ()
    model: str = ""
    backend_ids: tuple[str, ...] = ()
    execution_mode: AIExecutionMode = AIExecutionMode.FOREGROUND_SYNC
    profile_id: str = ""
    instance_id: str = ""
    owner_kind: str = "model"
    owner_id: str = ""
    idempotency_key: str = ""
    retry_policy: AIRetryPolicy = field(default_factory=AIRetryPolicy)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    prompt_cache_hint: AIPromptCacheHint | None = None
    prompt_cache_policy: AIPromptCachePolicy = field(default_factory=AIPromptCachePolicy)
    capability_request: AICapabilityRequest | None = field(default=None, repr=False, compare=False)
    agent_tools: tuple[AIAgentToolDefinition, ...] = ()
    agent_history: tuple[AIAgentTranscriptTurn, ...] = ()

    @property
    def logical_document(self) -> str:
        values = [value for value in (self.context_text, self.turn_text) if value]
        for turn in self.agent_history:
            for item in turn.output_items:
                if item.provider_item:
                    values.append(str(dict(item.provider_item)))
                else:
                    values.extend(
                        value for value in (item.text, item.name, item.raw_arguments) if value
                    )
            if turn.result_text:
                values.append(turn.result_text)
        return "\n\n".join(values)


@dataclass(frozen=True, slots=True)
class AICompletion:
    text: str
    finish_reason: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)
    model: str = ""
    invocation_id: str = ""
    capability_output: Any = field(default=None, repr=False, compare=False)
    agent_output_items: tuple[AIAgentOutputItem, ...] = ()
    agent_transport_mode: str = ""


@dataclass(frozen=True, slots=True)
class AIBackendResponse:
    """One adapter result with transport diagnostics kept outside the completion."""

    completion: AICompletion
    provider_envelope: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class AIInvocationResult:
    invocation_id: str
    completion: AICompletion
    backend_id: str
    attempts: int
    started_at: datetime
    finished_at: datetime
    warnings: tuple[str, ...] = ()


class AIBackendAdapter(Protocol):
    adapter_id: str
    capabilities: Sequence[str]

    async def complete(
        self,
        request: AIModelRequest,
        backend: AIBackendDescriptor,
    ) -> AIBackendResponse: ...

    def classify_error(
        self,
        exc: BaseException,
        backend: AIBackendDescriptor,
    ) -> AIErrorInfo: ...


@dataclass(frozen=True, slots=True)
class AICapabilityRequest:
    invocation_id: str
    capability: str
    work_purpose: AIWorkPurpose
    logical_stage_key: str = ""
    managed_work_stage: bool = False
    payload: Mapping[str, Any] = field(default_factory=dict)
    backend_ids: tuple[str, ...] = ()
    effect: AICapabilityEffect = AICapabilityEffect.READ_ONLY
    execution_mode: AIExecutionMode = AIExecutionMode.BACKGROUND_DURABLE
    profile_id: str = ""
    instance_id: str = ""
    owner_kind: str = "capability"
    owner_id: str = ""
    idempotency_key: str = ""
    retry_policy: AIRetryPolicy = field(default_factory=AIRetryPolicy)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AICapabilityResult:
    invocation_id: str
    capability: str
    output: Any
    backend_id: str
    attempts: int
    started_at: datetime
    finished_at: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AICapabilityAdapter(Protocol):
    adapter_id: str
    capabilities: Sequence[str]
    image_features: AIImageBackendCapabilities | None

    async def invoke(
        self,
        request: AICapabilityRequest,
        backend: AIBackendDescriptor,
    ) -> Any: ...

    def classify_error(
        self,
        exc: BaseException,
        backend: AIBackendDescriptor,
    ) -> AIErrorInfo: ...


@dataclass(slots=True)
class AIBackendCircuit:
    backend_id: str
    state: AIBackendState = AIBackendState.HEALTHY
    failure_count: int = 0
    opened_until: datetime | None = None
    last_error_code: str = ""
    updated_at: datetime = field(default_factory=ai_utc_now)


@dataclass(slots=True)
class AIBackendHealth:
    backend_id: str
    state: AIBackendState = AIBackendState.HEALTHY
    success_count: int = 0
    failure_count: int = 0
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    last_error_code: str = ""
    circuit: AIBackendCircuit | None = None
    updated_at: datetime = field(default_factory=ai_utc_now)


__all__ = [
    "AIAgentOutputItem",
    "AIAgentToolDefinition",
    "AIAgentToolResult",
    "AIAgentTranscriptTurn",
    "AIBackendAdapter",
    "AIBackendCircuit",
    "AIBackendDescriptor",
    "AIBackendHealth",
    "AIBackendResponse",
    "AIBackendState",
    "AICapabilityAdapter",
    "AICapabilityEffect",
    "AICapabilityName",
    "AICapabilityRequest",
    "AICapabilityResult",
    "AICompletion",
    "AIAudioContent",
    "AIErrorCode",
    "AIErrorInfo",
    "AIExecutionMode",
    "AIInvocationError",
    "AIInvocationResult",
    "AIImageContent",
    "AIImageBackendCapabilities",
    "AIImageGenerationOutput",
    "AIModelRequest",
    "AIPromptCacheBreakpoint",
    "AIPromptCacheHint",
    "AIPromptCachePolicy",
    "AIPromptCacheSection",
    "AIPromptCacheSemanticKind",
    "AIPromptCacheState",
    "AIPromptCacheWireMode",
    "AIWorkPurpose",
    "AISpeechResult",
    "AITranscriptionResult",
    "AIRetryPolicy",
    "AITaskStatus",
    "AIVisionDescription",
    "ai_utc_now",
]
