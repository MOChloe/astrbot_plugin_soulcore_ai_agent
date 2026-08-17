"""SoulCore-owned OpenAI-compatible HTTP transport."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ...contracts.ai_models import (
    AIBackendDescriptor,
    AIBackendResponse,
    AICompletion,
    AIErrorCode,
    AIErrorInfo,
    AIInvocationError,
    AIModelRequest,
    AIPromptCacheSection,
    AIPromptCacheWireMode,
)
from ...shared.http_security import require_secure_http_url
from .model_parameters import normalize_model_custom_request_parameters
from .openai_http_transport import (
    HTTPJSONResponse,
    JSONTransport,
    OpenAIHTTPStatusError,
    OpenAITransportError,
    UrllibJSONTransport,
    _retry_after,
)
from .openai_wire import (
    _completion_text,
    _empty_output,
    _first_choice,
    _is_context_window_error,
    _next_agent_transport,
    _openai_agent_history_messages,
    _openai_agent_output_items,
    _openai_responses_input,
    _openai_responses_output_items,
    _openai_responses_tool,
)
from .prompt_cache import is_explicit_cache_rejection, split_prompt_text
from .transport_tracking import json_transport_request, mark_transport_send

CredentialResolver = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class OpenAICompatibleConfig:
    backend_id: str
    base_url: str
    credential_id: str
    default_model: str = ""
    organization: str = ""
    project: str = ""
    extra_headers: Mapping[str, str] = field(default_factory=dict, repr=False)

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    @property
    def responses_endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/responses"


@dataclass(slots=True)
class _PromptCacheMessageState:
    context_replaced: bool = False
    turn_replaced: bool = False


class OpenAICompatibleAdapter:
    adapter_id = "openai_compatible"
    capabilities = frozenset({"chat.completion", "text.completion"})
    _PARAMETER_ALLOWLIST = {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_completion_tokens",
        "max_output_tokens",
        "seed",
        "reasoning_effort",
        "frequency_penalty",
        "presence_penalty",
        "stop",
    }

    def __init__(
        self,
        config: OpenAICompatibleConfig,
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
        credential = self._credential(backend)
        agent_transport = self._agent_transport(request, model, backend.metadata)
        headers = self._headers(credential)
        timeout_seconds = request.retry_policy.normalized().backend_timeout_seconds
        while True:
            endpoint = self._endpoint(
                request,
                backend.metadata,
                agent_transport=agent_transport,
            )
            payload = self._payload(
                request,
                model,
                backend.metadata,
                agent_transport=agent_transport,
            )
            await mark_transport_send(json_transport_request(endpoint, payload))
            try:
                response = await self.transport.post_json(
                    endpoint,
                    headers=headers,
                    payload=payload,
                    timeout_seconds=timeout_seconds,
                )
                self._ensure_response(response)
                break
            except OpenAIHTTPStatusError as exc:
                if (
                    request.prompt_cache_policy.carries_explicit_marker
                    and is_explicit_cache_rejection(exc.status_code, exc.provider_response)
                ):
                    raise self._cache_marker_error(exc, backend.backend_id) from exc
                next_transport = _next_agent_transport(agent_transport, exc)
                if not request.agent_tools or not next_transport:
                    raise
                fingerprint = self._agent_config_fingerprint(model, backend.metadata)
                self._agent_transport_by_fingerprint[fingerprint] = next_transport
                agent_transport = next_transport
        completion = (
            self._parse_responses(
                response.data,
                backend.backend_id,
                model,
                agent_transport=agent_transport,
            )
            if self._uses_responses(
                request,
                backend.metadata,
                agent_transport=agent_transport,
            )
            else self._parse(
                response.data,
                backend.backend_id,
                model,
                agent_transport=agent_transport,
            )
        )
        return AIBackendResponse(completion, response.raw_text or response.data)

    def _agent_transport(
        self,
        request: AIModelRequest,
        model: str,
        backend_metadata: Mapping[str, Any] | None,
    ) -> str:
        if not request.agent_tools:
            return ""
        fingerprint = self._agent_config_fingerprint(model, backend_metadata)
        default = (
            "native_freeform" if self._is_official_openai(backend_metadata) else "native_text_field"
        )
        return self._agent_transport_by_fingerprint.get(fingerprint, default)

    def _endpoint(
        self,
        request: AIModelRequest,
        backend_metadata: Mapping[str, Any] | None,
        *,
        agent_transport: str = "",
    ) -> str:
        return (
            self.config.responses_endpoint
            if self._uses_responses(
                request,
                backend_metadata,
                agent_transport=agent_transport,
            )
            else self.config.endpoint
        )

    @staticmethod
    def _is_official_openai(backend_metadata: Mapping[str, Any] | None) -> bool:
        return str((backend_metadata or {}).get("package_kind") or "").strip().lower() == "openai"

    @classmethod
    def _uses_responses(
        cls,
        request: AIModelRequest,
        backend_metadata: Mapping[str, Any] | None,
        *,
        agent_transport: str = "",
    ) -> bool:
        return (
            bool(request.agent_tools)
            and agent_transport == "native_freeform"
            and cls._is_official_openai(backend_metadata)
        )

    def _agent_config_fingerprint(
        self,
        model: str,
        backend_metadata: Mapping[str, Any] | None,
    ) -> str:
        configured = str((backend_metadata or {}).get("agent_config_fingerprint") or "").strip()
        if configured:
            return configured
        payload = {
            "endpoint": (
                self.config.responses_endpoint
                if self._is_official_openai(backend_metadata)
                else self.config.endpoint
            ),
            "model": model,
            "protocol": str((backend_metadata or {}).get("protocol") or "OPENAI_COMPATIBLE"),
            "custom": (backend_metadata or {}).get("custom_request_parameters") or {},
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _model(self, request: AIModelRequest, backend: AIBackendDescriptor) -> str:
        model = str(request.model or backend.model or self.config.default_model).strip()
        if model:
            return model
        raise AIInvocationError(
            AIErrorInfo(
                AIErrorCode.INVALID_REQUEST,
                "No model is configured for this backend",
                backend_id=backend.backend_id,
                phase="prepare",
            )
        )

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
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {credential}",
            **{str(key): str(value) for key, value in self.config.extra_headers.items()},
        }
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization
        if self.config.project:
            headers["OpenAI-Project"] = self.config.project
        return headers

    def _payload(
        self,
        request: AIModelRequest,
        model: str,
        backend_metadata: Mapping[str, Any] | None = None,
        *,
        agent_transport: str = "",
    ) -> dict[str, Any]:
        if self._uses_responses(
            request,
            backend_metadata,
            agent_transport=agent_transport,
        ):
            return self._responses_payload(
                request,
                model,
                backend_metadata,
                agent_transport=agent_transport,
            )
        messages = self._chat_messages(request, agent_transport=agent_transport)
        if not messages:
            raise AIInvocationError(
                AIErrorInfo(
                    AIErrorCode.INVALID_REQUEST,
                    "The model request has no text or images",
                    retryable=False,
                    switch_backend=False,
                    backend_id=self.config.backend_id,
                    phase="prepare",
                )
            )
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._cache_aware_messages(request, messages),
        }
        payload.update(
            normalize_model_custom_request_parameters(
                (backend_metadata or {}).get("custom_request_parameters")
            )
        )
        for key, value in request.parameters.items():
            if key in self._PARAMETER_ALLOWLIST:
                payload[key] = value
        if request.agent_tools and agent_transport == "native_text_field":
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                    },
                }
                for tool in request.agent_tools
            ]
            payload["tool_choice"] = "required"
        if request.prompt_cache_policy.wire_mode is AIPromptCacheWireMode.OPENAI_EXPLICIT:
            payload["prompt_cache_key"] = request.prompt_cache_policy.cache_key
            payload["prompt_cache_options"] = {
                "mode": "explicit",
                "ttl": request.prompt_cache_policy.actual_ttl or "30m",
            }
        return payload

    def _responses_payload(
        self,
        request: AIModelRequest,
        model: str,
        backend_metadata: Mapping[str, Any] | None,
        *,
        agent_transport: str,
    ) -> dict[str, Any]:
        input_items = _openai_responses_input(request, agent_transport=agent_transport)
        if not input_items:
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
            "input": input_items,
        }
        if request.context_text:
            payload["instructions"] = request.context_text
        parameters = normalize_model_custom_request_parameters(
            (backend_metadata or {}).get("custom_request_parameters")
        )
        for key, value in request.parameters.items():
            if key in self._PARAMETER_ALLOWLIST:
                parameters[key] = value
        output_limit = next(
            (
                parameters.pop(key)
                for key in ("max_output_tokens", "max_completion_tokens", "max_tokens")
                if key in parameters
            ),
            None,
        )
        if output_limit is not None:
            payload["max_output_tokens"] = output_limit
        effort = parameters.pop("reasoning_effort", None)
        if effort is not None:
            payload["reasoning"] = {"effort": effort}
        payload.update(parameters)
        if agent_transport in {"native_freeform", "native_text_field"}:
            payload["tools"] = [
                _openai_responses_tool(tool, freeform=agent_transport == "native_freeform")
                for tool in request.agent_tools
            ]
            payload["tool_choice"] = "required"
            payload["parallel_tool_calls"] = False
        if request.prompt_cache_policy.cache_key:
            payload["prompt_cache_key"] = request.prompt_cache_policy.cache_key
        return payload

    @staticmethod
    def _cache_aware_messages(
        request: AIModelRequest,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        mode = request.prompt_cache_policy.wire_mode
        if (
            not request.prompt_cache_policy.breakpoints
            or mode is not AIPromptCacheWireMode.OPENAI_EXPLICIT
        ):
            return messages
        state = _PromptCacheMessageState()
        return [
            OpenAICompatibleAdapter._cache_aware_message(request, message, state)
            for message in messages
        ]

    @staticmethod
    def _cache_aware_message(
        request: AIModelRequest,
        message: dict[str, Any],
        state: _PromptCacheMessageState,
    ) -> dict[str, Any]:
        content = message.get("content")
        section = OpenAICompatibleAdapter._matching_cache_section(request, content, state)
        if section is not None:
            state.context_replaced |= section is AIPromptCacheSection.CONTEXT
            state.turn_replaced |= section is AIPromptCacheSection.TURN
            text = (
                request.context_text
                if section is AIPromptCacheSection.CONTEXT
                else request.turn_text
            )
            return {
                **message,
                "content": OpenAICompatibleAdapter._marked_text_blocks(request, text, section),
            }
        if state.turn_replaced or not request.turn_text or not isinstance(content, list):
            return message
        replaced, state.turn_replaced = OpenAICompatibleAdapter._replace_turn_text_block(
            request, content
        )
        return {**message, "content": replaced}

    @staticmethod
    def _matching_cache_section(
        request: AIModelRequest,
        content: Any,
        state: _PromptCacheMessageState,
    ) -> AIPromptCacheSection | None:
        if not state.context_replaced and request.context_text and content == request.context_text:
            return AIPromptCacheSection.CONTEXT
        if not state.turn_replaced and request.turn_text and content == request.turn_text:
            return AIPromptCacheSection.TURN
        return None

    @staticmethod
    def _replace_turn_text_block(
        request: AIModelRequest,
        content: list[Any],
    ) -> tuple[list[dict[str, Any]], bool]:
        replaced: list[dict[str, Any]] = []
        matched = False
        for block in content:
            if OpenAICompatibleAdapter._is_unmatched_turn_text_block(request, block, matched):
                replaced.extend(
                    OpenAICompatibleAdapter._marked_text_blocks(
                        request,
                        request.turn_text,
                        AIPromptCacheSection.TURN,
                    )
                )
                matched = True
            else:
                replaced.append(dict(block))
        return replaced, matched

    @staticmethod
    def _is_unmatched_turn_text_block(
        request: AIModelRequest,
        block: Any,
        matched: bool,
    ) -> bool:
        return (
            not matched
            and isinstance(block, Mapping)
            and block.get("type") == "text"
            and block.get("text") == request.turn_text
        )

    @staticmethod
    def _marked_text_blocks(
        request: AIModelRequest,
        text: str,
        section: AIPromptCacheSection,
    ) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for segment, boundary in split_prompt_text(text, request.prompt_cache_policy, section):
            block: dict[str, Any] = {"type": "text", "text": segment}
            if boundary is not None:
                block["prompt_cache_breakpoint"] = {"mode": "explicit"}
            blocks.append(block)
        return blocks

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

    @staticmethod
    def _chat_messages(
        request: AIModelRequest,
        *,
        agent_transport: str = "",
    ) -> list[dict[str, Any]]:
        if request.context_text and not request.turn_text and not request.input_images:
            return [{"role": "user", "content": request.context_text}]
        messages: list[dict[str, Any]] = []
        if request.context_text:
            messages.append({"role": "system", "content": request.context_text})
        if request.turn_text and not request.input_images:
            messages.append({"role": "user", "content": request.turn_text})
        elif request.turn_text or request.input_images:
            content: list[dict[str, Any]] = []
            if request.turn_text:
                content.append({"type": "text", "text": request.turn_text})
            content.extend(
                {"type": "image_url", "image_url": {"url": url}} for url in request.input_images
            )
            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )
        for turn in request.agent_history:
            messages.extend(_openai_agent_history_messages(turn, agent_transport=agent_transport))
        return messages

    @staticmethod
    def _ensure_response(response: HTTPJSONResponse) -> None:
        if 200 <= response.status_code < 300:
            return
        error = response.data.get("error")
        api_code = (
            str(error.get("code") or error.get("type") or "") if isinstance(error, Mapping) else ""
        )
        raise OpenAIHTTPStatusError(
            response.status_code,
            api_code=api_code,
            retry_after_seconds=_retry_after(response.headers),
            provider_response=response.raw_text or response.data,
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
            if status == 400 and _is_context_window_error(code, exc.provider_response):
                error_code = AIErrorCode.CONTEXT_BUDGET
            elif status == 401:
                error_code = AIErrorCode.AUTHENTICATION
            elif status == 403:
                error_code = AIErrorCode.PERMISSION
            elif status in {402} or "quota" in code or "billing" in code:
                error_code = AIErrorCode.QUOTA_EXHAUSTED
            elif status == 429:
                error_code = AIErrorCode.RATE_LIMIT
            elif status in {408, 504}:
                error_code = AIErrorCode.TIMEOUT
            elif status >= 500:
                error_code = AIErrorCode.REMOTE_5XX
            else:
                error_code = AIErrorCode.INVALID_REQUEST
            switch = error_code in {
                AIErrorCode.CONTEXT_BUDGET,
                AIErrorCode.AUTHENTICATION,
                AIErrorCode.PERMISSION,
                AIErrorCode.QUOTA_EXHAUSTED,
                AIErrorCode.RATE_LIMIT,
                AIErrorCode.TIMEOUT,
                AIErrorCode.REMOTE_5XX,
            }
            immediate_circuit = error_code in {
                AIErrorCode.AUTHENTICATION,
                AIErrorCode.PERMISSION,
                AIErrorCode.QUOTA_EXHAUSTED,
                AIErrorCode.RATE_LIMIT,
            }
            return AIErrorInfo(
                error_code,
                f"Backend returned HTTP {status}",
                retryable=error_code
                in {
                    AIErrorCode.RATE_LIMIT,
                    AIErrorCode.TIMEOUT,
                    AIErrorCode.REMOTE_5XX,
                },
                switch_backend=switch,
                open_circuit=immediate_circuit,
                retry_after_seconds=exc.retry_after_seconds,
                backend_id=backend.backend_id,
                phase="transport",
                status_code=status,
                details={
                    "api_code": exc.api_code,
                    "provider_response": exc.provider_response,
                },
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
            f"OpenAI-compatible adapter failed: {type(exc).__name__}",
            backend_id=backend.backend_id,
            phase="adapter",
        )

    @staticmethod
    def _parse_responses(
        data: Mapping[str, Any],
        backend_id: str,
        fallback_model: str,
        *,
        agent_transport: str,
    ) -> AICompletion:
        raw_output = data.get("output")
        output = raw_output if isinstance(raw_output, list) else []
        items = _openai_responses_output_items(output)
        text = "".join(item.text for item in items if item.kind == "text")
        if not text.strip() and not any(item.kind == "tool_call" for item in items):
            raise _empty_output("Backend returned an empty response", backend_id)
        raw_usage = data.get("usage")
        usage: Mapping[str, Any] = raw_usage if isinstance(raw_usage, Mapping) else {}
        return AICompletion(
            text=text,
            finish_reason=str(data.get("status") or ""),
            usage=dict(usage),
            model=str(data.get("model") or fallback_model),
            agent_output_items=items,
            agent_transport_mode=agent_transport,
        )

    @staticmethod
    def _parse(
        data: Mapping[str, Any],
        backend_id: str,
        fallback_model: str,
        *,
        agent_transport: str = "",
    ) -> AICompletion:
        choice = _first_choice(data, backend_id)
        message = choice.get("message") if isinstance(choice, Mapping) else None
        if not isinstance(message, Mapping):
            message = {}
        content = message.get("content")
        text = _completion_text(content)
        output_items = _openai_agent_output_items(message) if agent_transport else ()
        if not text.strip() and not any(item.kind == "tool_call" for item in output_items):
            raise _empty_output("Backend returned an empty completion", backend_id)
        raw_usage = data.get("usage")
        usage: Mapping[str, Any] = raw_usage if isinstance(raw_usage, Mapping) else {}
        return AICompletion(
            text=text,
            finish_reason=str(choice.get("finish_reason") or ""),
            usage=dict(usage),
            model=str(data.get("model") or fallback_model),
            agent_output_items=output_items,
            agent_transport_mode=agent_transport,
        )


__all__ = [
    "HTTPJSONResponse",
    "JSONTransport",
    "OpenAICompatibleAdapter",
    "OpenAICompatibleConfig",
    "OpenAIHTTPStatusError",
    "OpenAITransportError",
    "UrllibJSONTransport",
]
