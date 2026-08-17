"""Native Anthropic Messages API adapters."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ...contracts.ai_models import (
    AIAgentOutputItem,
    AIBackendDescriptor,
    AIBackendResponse,
    AICapabilityName,
    AICapabilityRequest,
    AICompletion,
    AIErrorCode,
    AIErrorInfo,
    AIExecutionMode,
    AIInvocationError,
    AIModelRequest,
    AIPromptCacheSection,
    AIPromptCacheWireMode,
    AIVisionDescription,
)
from ...contracts.vision import VisionInspectionMode, VisionSequenceKind
from ...shared.http_security import require_secure_http_url
from .agent_transcript import (
    ANTHROPIC_MESSAGES_PROTOCOL,
    matching_provider_item,
    text_transport_assistant_output,
)
from .image_requests import payload_images, vision_prompt
from .image_responses import invalid, vision_description
from .model_parameters import (
    normalize_model_custom_request_parameters,
    resolve_model_generation_parameters,
)
from .openai_compatible import (
    HTTPJSONResponse,
    JSONTransport,
    OpenAIHTTPStatusError,
    OpenAITransportError,
    UrllibJSONTransport,
    _is_context_window_error,
    _retry_after,
)
from .prompt_cache import is_explicit_cache_rejection, split_prompt_text
from .transport_tracking import json_transport_request, mark_transport_send

CredentialResolver = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class AnthropicMessagesConfig:
    backend_id: str
    base_url: str
    credential_id: str
    default_model: str = ""
    api_version: str = "2023-06-01"
    extra_headers: Mapping[str, str] = field(default_factory=dict, repr=False)

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/messages"


class AnthropicMessagesAdapter:
    adapter_id = "anthropic_messages"
    capabilities = frozenset({"chat.completion", "text.completion"})

    def __init__(
        self,
        config: AnthropicMessagesConfig,
        credential_resolver: CredentialResolver,
        transport: JSONTransport | None = None,
    ) -> None:
        if not str(config.backend_id or "").strip():
            raise ValueError("backend_id is required")
        require_secure_http_url(config.base_url, "base_url")
        self.config = config
        self.credential_resolver = credential_resolver
        self.transport = transport or UrllibJSONTransport()
        self._agent_transport_by_fingerprint: dict[str, str] = {}

    async def complete(
        self,
        request: AIModelRequest,
        backend: AIBackendDescriptor,
    ) -> AIBackendResponse:
        model = self._model(request, backend)
        agent_transport = self._agent_transport(request, model, backend.metadata)
        payload = self._payload(
            request,
            model,
            backend.metadata,
            agent_transport=agent_transport,
        )
        headers = self._headers(self._credential(backend))
        timeout = request.retry_policy.normalized().backend_timeout_seconds
        await mark_transport_send(json_transport_request(self.config.endpoint, payload))
        try:
            response = await self.transport.post_json(
                self.config.endpoint,
                headers=headers,
                payload=payload,
                timeout_seconds=timeout,
            )
            self._ensure_response(response)
        except OpenAIHTTPStatusError as exc:
            if request.prompt_cache_policy.carries_explicit_marker and is_explicit_cache_rejection(
                exc.status_code, exc.provider_response
            ):
                raise self._cache_marker_error(exc, backend.backend_id) from exc
            if not (
                request.agent_tools
                and agent_transport == "native_text_field"
                and _is_anthropic_tool_transport_unsupported(exc)
            ):
                raise
            fingerprint = self._agent_config_fingerprint(model, backend.metadata)
            self._agent_transport_by_fingerprint[fingerprint] = "text_envelope"
            agent_transport = "text_envelope"
            payload = self._payload(
                request,
                model,
                backend.metadata,
                agent_transport=agent_transport,
            )
            await mark_transport_send(json_transport_request(self.config.endpoint, payload))
            response = await self.transport.post_json(
                self.config.endpoint,
                headers=headers,
                payload=payload,
                timeout_seconds=timeout,
            )
            self._ensure_response(response)
        return AIBackendResponse(
            self._parse(
                response.data,
                backend.backend_id,
                model,
                agent_transport=agent_transport,
            ),
            response.raw_text or response.data,
        )

    def _agent_transport(
        self,
        request: AIModelRequest,
        model: str,
        backend_metadata: Mapping[str, Any] | None,
    ) -> str:
        if not request.agent_tools:
            return ""
        fingerprint = self._agent_config_fingerprint(model, backend_metadata)
        return self._agent_transport_by_fingerprint.get(fingerprint, "native_text_field")

    def _agent_config_fingerprint(
        self,
        model: str,
        backend_metadata: Mapping[str, Any] | None,
    ) -> str:
        configured = str((backend_metadata or {}).get("agent_config_fingerprint") or "").strip()
        if configured:
            return configured
        payload = {
            "endpoint": self.config.endpoint,
            "model": model,
            "protocol": "ANTHROPIC",
            "custom": (backend_metadata or {}).get("custom_request_parameters") or {},
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _model(self, request: AIModelRequest, backend: AIBackendDescriptor) -> str:
        model = str(request.model or backend.model or self.config.default_model).strip()
        if model:
            return model
        raise invalid("No model is configured for this backend", backend)

    def _credential(self, backend: AIBackendDescriptor) -> str:
        try:
            credential = self.credential_resolver(self.config.credential_id)
        except Exception as exc:
            raise AIInvocationError(
                AIErrorInfo(
                    AIErrorCode.AUTHENTICATION,
                    "The backend credential is unavailable",
                    switch_backend=True,
                    open_circuit=True,
                    backend_id=backend.backend_id,
                    phase="prepare",
                ),
                cause=exc,
            ) from None
        if credential:
            return credential
        raise AIInvocationError(
            AIErrorInfo(
                AIErrorCode.AUTHENTICATION,
                "The backend credential is empty",
                switch_backend=True,
                open_circuit=True,
                backend_id=backend.backend_id,
                phase="prepare",
            )
        )

    def _headers(self, credential: str) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": credential,
            "anthropic-version": self.config.api_version,
            **{str(key): str(value) for key, value in self.config.extra_headers.items()},
        }

    def _payload(
        self,
        request: AIModelRequest,
        model: str,
        backend_metadata: Mapping[str, Any] | None = None,
        *,
        agent_transport: str = "",
    ) -> dict[str, Any]:
        system, messages = self._messages(request, agent_transport=agent_transport)
        if not messages:
            raise AIInvocationError(
                AIErrorInfo(
                    AIErrorCode.INVALID_REQUEST,
                    "The model request has no text or images",
                    backend_id=self.config.backend_id,
                    phase="prepare",
                )
            )
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": self._max_tokens(request.parameters),
            "messages": messages,
        }
        if system:
            payload["system"] = system
        payload.update(
            normalize_model_custom_request_parameters(
                (backend_metadata or {}).get("custom_request_parameters")
            )
        )
        self._generation_parameters(payload, request.parameters)
        if request.agent_tools and agent_transport == "native_text_field":
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                }
                for tool in request.agent_tools
            ]
            payload["tool_choice"] = {"type": "any"}
        return payload

    def _messages(
        self,
        request: AIModelRequest,
        *,
        agent_transport: str = "",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        system = self._system_blocks(request)
        content: list[dict[str, Any]] = []
        if request.turn_text:
            content.extend(
                self._cache_aware_text_blocks(
                    request,
                    request.turn_text,
                    AIPromptCacheSection.TURN,
                )
            )
        content.extend(_anthropic_image_block(url) for url in request.input_images if url)
        if not content and request.context_text:
            content = system
            system = []
        messages = [{"role": "user", "content": content}] if content else []
        for turn in request.agent_history:
            messages.extend(
                _anthropic_agent_history_messages(turn, agent_transport=agent_transport)
            )
        return system, messages

    @staticmethod
    def _system_blocks(request: AIModelRequest) -> list[dict[str, Any]]:
        if not request.context_text:
            return []
        return AnthropicMessagesAdapter._cache_aware_text_blocks(
            request,
            request.context_text,
            AIPromptCacheSection.CONTEXT,
        )

    @staticmethod
    def _cache_aware_text_blocks(
        request: AIModelRequest,
        text: str,
        section: AIPromptCacheSection,
    ) -> list[dict[str, Any]]:
        if (
            request.prompt_cache_policy.wire_mode is not AIPromptCacheWireMode.ANTHROPIC_EPHEMERAL
            or not request.prompt_cache_policy.breakpoints
        ):
            return [{"type": "text", "text": text}]
        ttl = request.prompt_cache_policy.actual_ttl or "1h"
        blocks: list[dict[str, Any]] = []
        for segment, boundary in split_prompt_text(text, request.prompt_cache_policy, section):
            block: dict[str, Any] = {"type": "text", "text": segment}
            if boundary is not None:
                block["cache_control"] = {"type": "ephemeral", "ttl": ttl}
            blocks.append(block)
        return blocks

    @staticmethod
    def _max_tokens(parameters: Mapping[str, Any]) -> int:
        raw = parameters.get("max_tokens")
        if raw is None:
            raise AIInvocationError(
                AIErrorInfo(
                    AIErrorCode.INVALID_REQUEST,
                    "Anthropic 模型必须在模型设置中明确配置 max_tokens",
                    phase="prepare",
                )
            )
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 0
        if value < 1:
            raise AIInvocationError(
                AIErrorInfo(
                    AIErrorCode.INVALID_REQUEST,
                    "Anthropic 模型的 max_tokens 必须是正整数",
                    phase="prepare",
                )
            )
        return value

    @staticmethod
    def _generation_parameters(payload: dict[str, Any], parameters: Mapping[str, Any]) -> None:
        for key in ("temperature", "top_p", "top_k"):
            if key in parameters:
                payload[key] = parameters[key]
        stop = parameters.get("stop")
        if isinstance(stop, str) and stop:
            payload["stop_sequences"] = [stop]
        elif isinstance(stop, Sequence) and not isinstance(stop, (str, bytes)):
            payload["stop_sequences"] = [str(item) for item in stop if str(item)]
        effort = str(parameters.get("reasoning_effort") or "").strip().lower()
        if effort == "minimal":
            raise AIInvocationError(
                AIErrorInfo(
                    AIErrorCode.INVALID_REQUEST,
                    "Anthropic effort does not support minimal",
                    phase="prepare",
                )
            )
        if effort:
            if effort not in {"low", "medium", "high", "xhigh", "max"}:
                raise AIInvocationError(
                    AIErrorInfo(
                        AIErrorCode.INVALID_REQUEST,
                        "Anthropic effort value is unsupported",
                        phase="prepare",
                    )
                )
            payload["output_config"] = {"effort": effort}

    @staticmethod
    def _ensure_response(response: HTTPJSONResponse) -> None:
        if 200 <= response.status_code < 300:
            return
        error = response.data.get("error")
        api_code = (
            str(error.get("type") or error.get("code") or "") if isinstance(error, Mapping) else ""
        )
        raise OpenAIHTTPStatusError(
            response.status_code,
            api_code=api_code,
            retry_after_seconds=_retry_after(response.headers),
            provider_response=response.raw_text or response.data,
        )

    @staticmethod
    def _cache_marker_error(exc: OpenAIHTTPStatusError, backend_id: str) -> AIInvocationError:
        return AIInvocationError(
            AIErrorInfo(
                AIErrorCode.PROMPT_CACHE_MARKER_UNSUPPORTED,
                "Backend rejected the negotiated prompt-cache fields",
                backend_id=backend_id,
                phase="transport",
                status_code=exc.status_code,
                details={
                    "api_code": exc.api_code,
                    "provider_response": exc.provider_response,
                },
            ),
            cause=exc,
        )

    def classify_error(
        self,
        exc: BaseException,
        backend: AIBackendDescriptor,
    ) -> AIErrorInfo:
        if isinstance(exc, AIInvocationError):
            return exc.info
        if isinstance(exc, OpenAIHTTPStatusError):
            status = exc.status_code
            code = exc.api_code.lower()
            if status in {400, 413} and _is_context_window_error(code, exc.provider_response):
                error_code = AIErrorCode.CONTEXT_BUDGET
            elif status == 401:
                error_code = AIErrorCode.AUTHENTICATION
            elif status == 403:
                error_code = AIErrorCode.PERMISSION
            elif status == 429:
                error_code = AIErrorCode.RATE_LIMIT
            elif status in {408, 504}:
                error_code = AIErrorCode.TIMEOUT
            elif status >= 500:
                error_code = AIErrorCode.REMOTE_5XX
            else:
                error_code = AIErrorCode.INVALID_REQUEST
            switch = error_code not in {AIErrorCode.INVALID_REQUEST}
            return AIErrorInfo(
                error_code,
                f"Backend returned HTTP {status}",
                retryable=error_code
                in {AIErrorCode.RATE_LIMIT, AIErrorCode.TIMEOUT, AIErrorCode.REMOTE_5XX},
                switch_backend=switch,
                open_circuit=error_code
                in {AIErrorCode.AUTHENTICATION, AIErrorCode.PERMISSION, AIErrorCode.RATE_LIMIT},
                retry_after_seconds=exc.retry_after_seconds,
                backend_id=backend.backend_id,
                phase="transport",
                status_code=status,
                details={"api_code": exc.api_code, "provider_response": exc.provider_response},
            )
        if isinstance(exc, (OpenAITransportError, OSError)):
            return AIErrorInfo(
                AIErrorCode.NETWORK,
                "Could not reach the backend",
                retryable=True,
                switch_backend=True,
                backend_id=backend.backend_id,
                phase="transport",
            )
        return AIErrorInfo(
            AIErrorCode.INTERNAL,
            f"Anthropic adapter failed: {type(exc).__name__}",
            backend_id=backend.backend_id,
            phase="adapter",
        )

    @staticmethod
    def _parse(
        data: Mapping[str, Any],
        backend_id: str,
        model: str,
        *,
        agent_transport: str = "",
    ) -> AICompletion:
        blocks = data.get("content")
        text = "".join(
            str(block.get("text") or "")
            for block in blocks or ()
            if isinstance(block, Mapping) and block.get("type") == "text"
        )
        output_items = _anthropic_agent_output_items(blocks) if agent_transport else ()
        if not text.strip() and not any(item.kind == "tool_call" for item in output_items):
            raise AIInvocationError(
                AIErrorInfo(
                    AIErrorCode.EMPTY_OUTPUT,
                    "Backend returned an empty completion",
                    retryable=True,
                    switch_backend=True,
                    backend_id=backend_id,
                    phase="response",
                )
            )
        usage = data.get("usage")
        return AICompletion(
            text=text,
            finish_reason=str(data.get("stop_reason") or ""),
            usage=dict(usage) if isinstance(usage, Mapping) else {},
            model=str(data.get("model") or model),
            agent_output_items=output_items,
            agent_transport_mode=agent_transport,
        )


def _anthropic_agent_history_messages(turn: Any, *, agent_transport: str) -> list[dict[str, Any]]:
    if turn.transport_mode == "state_snapshot":
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (f"<较早行动压缩状态>\n{turn.result_text}\n</较早行动压缩状态>"),
                    }
                ],
            }
        ]
    if agent_transport == "text_envelope" or not turn.tool_results:
        assistant_text = text_transport_assistant_output(turn)
        messages: list[dict[str, Any]] = []
        if assistant_text:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
            )
        if turn.result_text:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<行动结果>\n{turn.result_text}\n</行动结果>",
                        }
                    ],
                }
            )
        return messages

    assistant_content: list[dict[str, Any]] = []
    for item in turn.output_items:
        provider_item = matching_provider_item(item, ANTHROPIC_MESSAGES_PROTOCOL)
        if provider_item is not None:
            assistant_content.append(provider_item)
            continue
        if item.kind == "text":
            assistant_content.append({"type": "text", "text": item.text})
        elif item.kind == "tool_call":
            assistant_content.append(
                {
                    "type": "tool_use",
                    "id": item.call_id,
                    "name": item.name,
                    "input": {"text": item.text},
                }
            )
    return [
        {"role": "assistant", "content": assistant_content},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": result.call_id,
                    "content": result.text,
                }
                for result in turn.tool_results
            ],
        },
    ]


def _anthropic_agent_output_items(blocks: Any) -> tuple[AIAgentOutputItem, ...]:
    items: list[AIAgentOutputItem] = []
    for block in blocks if isinstance(blocks, list) else ():
        if not isinstance(block, Mapping):
            continue
        kind = str(block.get("type") or "")
        if kind == "text":
            items.append(
                AIAgentOutputItem(
                    "text",
                    text=str(block.get("text") or ""),
                    provider_item=dict(block),
                    provider_protocol=ANTHROPIC_MESSAGES_PROTOCOL,
                )
            )
            continue
        if kind != "tool_use":
            items.append(
                AIAgentOutputItem(
                    "provider_item",
                    provider_item=dict(block),
                    provider_protocol=ANTHROPIC_MESSAGES_PROTOCOL,
                )
            )
            continue
        value = block.get("input")
        raw_arguments = json.dumps(value, ensure_ascii=False, default=str)
        text_value = ""
        argument_error = ""
        if (
            not isinstance(value, Mapping)
            or set(value) != {"text"}
            or not isinstance(value.get("text"), str)
        ):
            argument_error = "工具参数必须且只能包含字符串 text"
        else:
            text_value = value["text"]
        call_id = str(block.get("id") or "")
        if not call_id:
            argument_error = argument_error or "工具调用缺少 id"
        items.append(
            AIAgentOutputItem(
                "tool_call",
                text=text_value,
                name=str(block.get("name") or ""),
                call_id=call_id,
                raw_arguments=raw_arguments,
                argument_error=argument_error,
                provider_item=dict(block),
                provider_protocol=ANTHROPIC_MESSAGES_PROTOCOL,
            )
        )
    return tuple(items)


def _is_anthropic_tool_transport_unsupported(exc: OpenAIHTTPStatusError) -> bool:
    if exc.status_code not in {400, 404, 405, 422}:
        return False
    try:
        text = json.dumps(exc.provider_response, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(exc.provider_response or "")
    normalized = text.casefold()
    return any(marker in normalized for marker in ("tools", "tool_choice", "tool_use")) and any(
        marker in normalized
        for marker in (
            "unsupported",
            "not supported",
            "unknown field",
            "extra inputs are not permitted",
            "does not support",
            "不支持",
        )
    )


class AnthropicVisionDescribeAdapter:
    adapter_id = "anthropic_vision_describe"
    capabilities = (AICapabilityName.VISION_DESCRIBE.value,)
    image_features = None

    def __init__(self, messages_adapter: AnthropicMessagesAdapter) -> None:
        self.messages_adapter = messages_adapter

    async def invoke(
        self, request: AICapabilityRequest, backend: AIBackendDescriptor
    ) -> AIVisionDescription:
        images = payload_images(request.payload.get("images", ()))
        if not images:
            raise invalid("vision.describe requires at least one image", backend)
        try:
            sequence_kind = VisionSequenceKind(
                str(request.payload.get("sequence_kind") or VisionSequenceKind.SINGLE_IMAGE.value)
            )
            inspection_mode = VisionInspectionMode(
                str(request.payload.get("inspection_mode") or VisionInspectionMode.OBJECTIVE.value)
            )
        except ValueError:
            raise invalid(
                "vision.describe received an unsupported controlled mode", backend
            ) from None
        model_request = AIModelRequest(
            invocation_id=request.invocation_id,
            work_purpose=request.work_purpose,
            logical_stage_key=request.logical_stage_key,
            turn_text=vision_prompt(sequence_kind, inspection_mode),
            input_images=tuple(_image_url(image) for image in images),
            model=backend.model,
            execution_mode=AIExecutionMode.DEBUG_EPHEMERAL,
            retry_policy=request.retry_policy,
            parameters=resolve_model_generation_parameters({}, backend.metadata),
        )
        response = await self.messages_adapter.complete(model_request, backend)
        completion = response.completion
        proxy = HTTPJSONResponse(
            200,
            {
                "model": completion.model,
                "choices": [{"message": {"content": completion.text}}],
            },
        )
        return vision_description(
            proxy,
            completion.model,
            backend,
            require_source_marker=inspection_mode is VisionInspectionMode.STICKER_QUALITY,
            require_social_impression=inspection_mode is VisionInspectionMode.STICKER_QUALITY,
        )

    def classify_error(self, exc: BaseException, backend: AIBackendDescriptor) -> AIErrorInfo:
        return self.messages_adapter.classify_error(exc, backend)


def _anthropic_image_block(url: str) -> dict[str, Any]:
    if str(url).startswith("data:") and ";base64," in str(url):
        header, data = str(url).split(",", 1)
        media_type = header[5:].split(";", 1)[0] or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": str(url)}}


def _image_url(image: Any) -> str:
    if image.url:
        return str(image.url)
    return (
        f"data:{image.mime_type};base64,{base64.b64encode(image.data).decode('ascii')}"
        if image.data
        else ""
    )


__all__ = [
    "AnthropicMessagesAdapter",
    "AnthropicMessagesConfig",
    "AnthropicVisionDescribeAdapter",
]
