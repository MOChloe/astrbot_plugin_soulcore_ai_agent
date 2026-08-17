"""SoulCore-owned vision-description and image-generation capability adapters.

The adapters only translate trusted administrator configuration into provider
requests.  They do not persist files, select conversation targets, or expose a
generic model-controlled HTTP client.
"""

from __future__ import annotations

import asyncio
import base64
import json
import secrets
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import quote

from ...contracts.ai_models import (
    AIBackendDescriptor,
    AICapabilityName,
    AICapabilityRequest,
    AIErrorCode,
    AIErrorInfo,
    AIImageBackendCapabilities,
    AIImageGenerationOutput,
    AIInvocationError,
    AIVisionDescription,
)
from ...contracts.vision import VisionInspectionMode, VisionSequenceKind
from ...features.ai.openai_compatible import (
    HTTPJSONResponse,
    JSONTransport,
    OpenAIHTTPStatusError,
    OpenAITransportError,
    UrllibJSONTransport,
)
from ...shared.http_security import HTTPResponseTooLargeError, open_same_origin, read_limited
from .image_requests import (
    generation_input as _generation_input,
)
from .image_requests import (
    model as _model,
)
from .image_requests import (
    openai_generation_fields as _openai_generation_fields,
)
from .image_requests import (
    openai_image_parts as _openai_image_parts,
)
from .image_requests import (
    optional_image_config as _optional_image_config,
)
from .image_requests import (
    payload_images as _payload_images,
)
from .image_requests import (
    validate_features as _validate_features,
)
from .image_requests import (
    vision_prompt as _vision_prompt,
)
from .image_responses import (
    bearer_headers as _bearer_headers,
)
from .image_responses import (
    custom_output as _custom_output,
)
from .image_responses import (
    ensure_success as _ensure_success,
)
from .image_responses import (
    extension as _extension,
)
from .image_responses import (
    extract_chat_images as _extract_chat_images,
)
from .image_responses import (
    gemini_output as _gemini_output,
)
from .image_responses import (
    invalid as _invalid,
)
from .image_responses import (
    output_error as _output_error,
)
from .image_responses import (
    parse_openai_images as _parse_openai_images,
)
from .image_responses import (
    render_template as _render_template,
)
from .image_responses import (
    require_http_url as _require_http_url,
)
from .image_responses import (
    vision_description as _vision_description,
)
from .model_parameters import (
    normalize_model_custom_request_parameters,
    resolve_model_generation_parameters,
)
from .transport_tracking import (
    json_transport_request,
    mark_transport_send,
    multipart_transport_request,
)

CredentialResolver = Callable[[str], str]


class MultipartTransport(Protocol):
    async def post_multipart(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        fields: Mapping[str, str],
        files: Sequence[tuple[str, str, str, bytes]],
        timeout_seconds: float,
    ) -> HTTPJSONResponse: ...


class UrllibImageTransport(UrllibJSONTransport):
    """Standard-library JSON plus multipart transport for Images edits."""

    def __init__(self, *, max_response_bytes: int = 128 * 1024 * 1024) -> None:
        super().__init__(max_response_bytes=max_response_bytes)

    async def post_multipart(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        fields: Mapping[str, str],
        files: Sequence[tuple[str, str, str, bytes]],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        return await asyncio.to_thread(
            self._post_multipart,
            url,
            dict(headers),
            dict(fields),
            tuple(files),
            float(timeout_seconds),
            self.max_response_bytes,
        )

    @staticmethod
    def _post_multipart(
        url: str,
        headers: dict[str, str],
        fields: dict[str, str],
        files: tuple[tuple[str, str, str, bytes], ...],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> HTTPJSONResponse:
        boundary = "soulcore-" + secrets.token_hex(16)
        body = bytearray()
        for name, value in fields.items():
            body.extend(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
            )
        for name, filename, mime_type, data in files:
            safe_name = filename.replace('"', "")
            body.extend(
                (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                    f'filename="{safe_name}"\r\nContent-Type: {mime_type}\r\n\r\n'
                ).encode()
            )
            body.extend(data)
            body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("ascii"))
        request = urllib.request.Request(
            url,
            data=bytes(body),
            headers={**headers, "Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with open_same_origin(request, timeout=timeout_seconds) as response:
                raw = read_limited(response, max_response_bytes)
                parsed = json.loads(raw.decode("utf-8")) if raw else {}
                if not isinstance(parsed, Mapping):
                    raise OpenAITransportError("API response is not a JSON object")
                return HTTPJSONResponse(
                    int(getattr(response, "status", 200)),
                    dict(parsed),
                    {str(key): str(value) for key, value in response.headers.items()},
                )
        except urllib.error.HTTPError as exc:
            raw = exc.read(64 * 1024)
            api_code = ""
            try:
                parsed = json.loads(raw.decode("utf-8")) if raw else {}
                error = parsed.get("error") if isinstance(parsed, Mapping) else None
                if isinstance(error, Mapping):
                    api_code = str(error.get("code") or error.get("type") or "")
            except (UnicodeDecodeError, ValueError):
                pass
            raise OpenAIHTTPStatusError(exc.code, api_code=api_code) from None
        except HTTPResponseTooLargeError:
            raise OpenAITransportError("provider_response_too_large") from None
        except (urllib.error.URLError, OSError) as exc:
            raise OpenAITransportError(type(exc).__name__) from None


@dataclass(frozen=True, slots=True)
class OpenAIVisionConfig:
    base_url: str
    credential_id: str
    default_model: str = ""
    extra_headers: Mapping[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class OpenAIImagesConfig:
    base_url: str
    credential_id: str
    default_model: str = ""
    extra_headers: Mapping[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class OpenAIChatImageConfig:
    base_url: str
    credential_id: str
    default_model: str = ""
    extra_headers: Mapping[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class GeminiImageConfig:
    base_url: str = "https://generativelanguage.googleapis.com"
    credential_id: str = ""
    default_model: str = ""
    extra_headers: Mapping[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class CustomHTTPImageConfig:
    endpoint: str
    credential_id: str = ""
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "
    request_template: Mapping[str, Any] = field(default_factory=lambda: {"prompt": "{{prompt}}"})
    image_paths: tuple[str, ...] = ("images",)
    mime_type_path: str = "mime_type"
    model_path: str = "model"
    features: AIImageBackendCapabilities = field(default_factory=AIImageBackendCapabilities)
    extra_headers: Mapping[str, str] = field(default_factory=dict, repr=False)


class _CapabilityHTTPMixin:
    credential_resolver: CredentialResolver

    def _credential(self, credential_id: str, backend_id: str) -> str:
        try:
            value = self.credential_resolver(credential_id)
        except Exception as exc:
            raise AIInvocationError(
                AIErrorInfo(
                    AIErrorCode.AUTHENTICATION,
                    "The image backend credential is unavailable",
                    switch_backend=True,
                    open_circuit=True,
                    backend_id=backend_id,
                    phase="prepare",
                ),
                cause=exc,
            ) from None
        if not value:
            raise AIInvocationError(
                AIErrorInfo(
                    AIErrorCode.AUTHENTICATION,
                    "The image backend credential is empty",
                    switch_backend=True,
                    open_circuit=True,
                    backend_id=backend_id,
                    phase="prepare",
                )
            )
        return value

    def classify_error(self, exc: BaseException, backend: AIBackendDescriptor) -> AIErrorInfo:
        if isinstance(exc, AIInvocationError):
            return exc.info
        if isinstance(exc, OpenAIHTTPStatusError):
            status = exc.status_code
            api_code = exc.api_code.lower()
            if status == 401:
                code = AIErrorCode.AUTHENTICATION
            elif status == 403:
                code = AIErrorCode.PERMISSION
            elif status == 402 or "quota" in api_code or "billing" in api_code:
                code = AIErrorCode.QUOTA_EXHAUSTED
            elif status == 429:
                code = AIErrorCode.RATE_LIMIT
            elif status in {408, 504}:
                code = AIErrorCode.TIMEOUT
            elif status >= 500:
                code = AIErrorCode.REMOTE_5XX
            else:
                code = AIErrorCode.INVALID_REQUEST
            switch = code in {
                AIErrorCode.AUTHENTICATION,
                AIErrorCode.PERMISSION,
                AIErrorCode.QUOTA_EXHAUSTED,
                AIErrorCode.RATE_LIMIT,
                AIErrorCode.TIMEOUT,
                AIErrorCode.REMOTE_5XX,
            }
            return AIErrorInfo(
                code,
                f"Image backend returned HTTP {status}",
                retryable=code
                in {
                    AIErrorCode.RATE_LIMIT,
                    AIErrorCode.TIMEOUT,
                    AIErrorCode.REMOTE_5XX,
                },
                switch_backend=switch,
                open_circuit=code
                in {
                    AIErrorCode.AUTHENTICATION,
                    AIErrorCode.PERMISSION,
                    AIErrorCode.QUOTA_EXHAUSTED,
                    AIErrorCode.RATE_LIMIT,
                },
                retry_after_seconds=exc.retry_after_seconds,
                backend_id=backend.backend_id,
                phase="transport",
                status_code=status,
            )
        if isinstance(exc, (OpenAITransportError, OSError)):
            return AIErrorInfo(
                AIErrorCode.NETWORK,
                "Could not reach the image backend",
                retryable=True,
                switch_backend=True,
                backend_id=backend.backend_id,
                phase="transport",
            )
        return AIErrorInfo(
            AIErrorCode.INTERNAL,
            f"Image capability adapter failed: {type(exc).__name__}",
            backend_id=backend.backend_id,
            phase="adapter",
        )


class OpenAIVisionDescribeAdapter(_CapabilityHTTPMixin):
    adapter_id = "openai_vision_describe"
    capabilities = (AICapabilityName.VISION_DESCRIBE.value,)
    image_features = None

    def __init__(
        self,
        config: OpenAIVisionConfig,
        credential_resolver: CredentialResolver,
        transport: JSONTransport | None = None,
    ) -> None:
        _require_http_url(config.base_url, "base_url")
        self.config = config
        self.credential_resolver = credential_resolver
        self.transport = transport or UrllibImageTransport()

    async def invoke(
        self, request: AICapabilityRequest, backend: AIBackendDescriptor
    ) -> AIVisionDescription:
        model = _model(request, backend, self.config.default_model)
        images = _payload_images(request.payload.get("images", ()))
        if not images:
            raise _invalid("vision.describe requires at least one image", backend)
        try:
            sequence_kind = VisionSequenceKind(
                str(request.payload.get("sequence_kind") or VisionSequenceKind.SINGLE_IMAGE.value)
            )
            inspection_mode = VisionInspectionMode(
                str(request.payload.get("inspection_mode") or VisionInspectionMode.OBJECTIVE.value)
            )
        except ValueError:
            raise _invalid(
                "vision.describe received an unsupported controlled mode", backend
            ) from None
        prompt = _vision_prompt(sequence_kind, inspection_mode)
        content: list[Mapping[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(_openai_image_parts(images))
        headers = _bearer_headers(
            self._credential(self.config.credential_id, backend.backend_id),
            self.config.extra_headers,
        )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
        }
        payload.update(
            normalize_model_custom_request_parameters(
                backend.metadata.get("custom_request_parameters")
            )
        )
        payload.update(resolve_model_generation_parameters({}, backend.metadata))
        timeout_seconds = request.retry_policy.normalized().backend_timeout_seconds
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        await mark_transport_send(json_transport_request(endpoint, payload))
        response = await self.transport.post_json(
            endpoint,
            headers=headers,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        _ensure_success(response)
        return _vision_description(
            response,
            model,
            backend,
            require_source_marker=inspection_mode is VisionInspectionMode.STICKER_QUALITY,
            require_social_impression=inspection_mode is VisionInspectionMode.STICKER_QUALITY,
        )


class OpenAIImagesCapabilityAdapter(_CapabilityHTTPMixin):
    adapter_id = "openai_images"
    capabilities = (AICapabilityName.IMAGE_GENERATE.value,)
    image_features = AIImageBackendCapabilities(
        reference_image=True,
        multiple_references=True,
        maximum_outputs=5,
        output_format="url_or_base64",
    )

    def __init__(
        self,
        config: OpenAIImagesConfig,
        credential_resolver: CredentialResolver,
        transport: JSONTransport | MultipartTransport | None = None,
    ) -> None:
        _require_http_url(config.base_url, "base_url")
        self.config = config
        self.credential_resolver = credential_resolver
        self.transport = transport or UrllibImageTransport()

    async def invoke(
        self, request: AICapabilityRequest, backend: AIBackendDescriptor
    ) -> AIImageGenerationOutput:
        prompt, count, references = _generation_input(request, backend)
        _validate_features(self.image_features, request.payload, count, references, backend)
        model = _model(request, backend, self.config.default_model)
        headers = _bearer_headers(
            self._credential(self.config.credential_id, backend.backend_id),
            self.config.extra_headers,
        )
        common = _openai_generation_fields(request.payload, model, prompt, count)
        timeout_seconds = request.retry_policy.normalized().backend_timeout_seconds
        if references:
            multipart = getattr(self.transport, "post_multipart", None)
            if not callable(multipart):
                raise _invalid("Configured transport does not support image edits", backend)
            files = [
                (
                    "image[]" if len(references) > 1 else "image",
                    f"reference-{index}.{_extension(image.mime_type)}",
                    image.mime_type,
                    image.data,
                )
                for index, image in enumerate(references)
                if image.data
            ]
            if len(files) != len(references):
                raise _invalid("OpenAI image edits require local reference bytes", backend)
            edit_headers = {
                key: value for key, value in headers.items() if key.lower() != "content-type"
            }
            fields = {key: str(value) for key, value in common.items()}
            endpoint = self.config.base_url.rstrip("/") + "/images/edits"
            await mark_transport_send(multipart_transport_request(endpoint, fields, files))
            response = await multipart(
                endpoint,
                headers=edit_headers,
                fields=fields,
                files=files,
                timeout_seconds=timeout_seconds,
            )
            reference_mode = "raw"
        else:
            post_json = getattr(self.transport, "post_json", None)
            if not callable(post_json):
                raise _invalid("OpenAI image transport does not support JSON requests", backend)
            endpoint = self.config.base_url.rstrip("/") + "/images/generations"
            await mark_transport_send(json_transport_request(endpoint, common))
            response = await post_json(
                endpoint,
                headers=headers,
                payload=common,
                timeout_seconds=timeout_seconds,
            )
            reference_mode = "none"
        _ensure_success(response)
        return _parse_openai_images(response.data, model, reference_mode, backend)


class OpenAIChatImageCapabilityAdapter(_CapabilityHTTPMixin):
    adapter_id = "openai_chat_image"
    capabilities = (AICapabilityName.IMAGE_GENERATE.value,)
    image_features = AIImageBackendCapabilities(
        reference_image=True,
        multiple_references=True,
        maximum_outputs=5,
        output_format="multimodal_chat",
    )

    def __init__(
        self,
        config: OpenAIChatImageConfig,
        credential_resolver: CredentialResolver,
        transport: JSONTransport | None = None,
    ) -> None:
        _require_http_url(config.base_url, "base_url")
        self.config = config
        self.credential_resolver = credential_resolver
        self.transport = transport or UrllibJSONTransport()

    async def invoke(
        self, request: AICapabilityRequest, backend: AIBackendDescriptor
    ) -> AIImageGenerationOutput:
        prompt, count, references = _generation_input(request, backend)
        _validate_features(self.image_features, request.payload, count, references, backend)
        model = _model(request, backend, self.config.default_model)
        content: list[Mapping[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(_openai_image_parts(references))
        headers = _bearer_headers(
            self._credential(self.config.credential_id, backend.backend_id),
            self.config.extra_headers,
        )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "modalities": ["text", "image"],
            "n": count,
            **_optional_image_config(request.payload),
        }
        timeout_seconds = request.retry_policy.normalized().backend_timeout_seconds
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        await mark_transport_send(json_transport_request(endpoint, payload))
        response = await self.transport.post_json(
            endpoint,
            headers=headers,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        _ensure_success(response)
        images = _extract_chat_images(response.data)
        if not images:
            raise _output_error("Chat image backend returned no images", backend)
        return AIImageGenerationOutput(
            images=tuple(images[:count]),
            model=str(response.data.get("model") or model),
            reference_mode="raw" if references else "none",
        )


class GeminiImageCapabilityAdapter(_CapabilityHTTPMixin):
    adapter_id = "gemini_image"
    capabilities = (AICapabilityName.IMAGE_GENERATE.value,)
    image_features = AIImageBackendCapabilities(
        reference_image=True,
        multiple_references=True,
        maximum_outputs=1,
        output_format="inline_data",
    )

    def __init__(
        self,
        config: GeminiImageConfig,
        credential_resolver: CredentialResolver,
        transport: JSONTransport | None = None,
    ) -> None:
        _require_http_url(config.base_url, "base_url")
        self.config = config
        self.credential_resolver = credential_resolver
        self.transport = transport or UrllibJSONTransport()

    async def invoke(
        self, request: AICapabilityRequest, backend: AIBackendDescriptor
    ) -> AIImageGenerationOutput:
        prompt, count, references = _generation_input(request, backend)
        _validate_features(self.image_features, request.payload, count, references, backend)
        model = _model(request, backend, self.config.default_model)
        parts: list[Mapping[str, Any]] = [{"text": prompt}]
        parts.extend(
            {
                "inlineData": {
                    "mimeType": image.mime_type,
                    "data": base64.b64encode(image.data).decode("ascii"),
                }
            }
            for image in references
            if image.data
        )
        if len(parts) - 1 != len(references):
            raise _invalid("Gemini references require local image bytes", backend)
        image_config: dict[str, Any] = {}
        if str(request.payload.get("aspect_ratio") or "auto") != "auto":
            image_config["aspectRatio"] = str(request.payload["aspect_ratio"])
        if str(request.payload.get("size") or "auto") != "auto":
            image_config["imageSize"] = str(request.payload["size"])
        endpoint = (
            self.config.base_url.rstrip("/")
            + "/v1beta/models/"
            + quote(model, safe="-._")
            + ":generateContent"
        )
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-goog-api-key": self._credential(self.config.credential_id, backend.backend_id),
            **dict(self.config.extra_headers),
        }
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                **({"imageConfig": image_config} if image_config else {}),
            },
        }
        timeout_seconds = request.retry_policy.normalized().backend_timeout_seconds
        await mark_transport_send(json_transport_request(endpoint, payload))
        response = await self.transport.post_json(
            endpoint,
            headers=headers,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        _ensure_success(response)
        return _gemini_output(
            response,
            count=count,
            model=model,
            has_references=bool(references),
            backend=backend,
        )


class CustomHTTPImageCapabilityAdapter(_CapabilityHTTPMixin):
    adapter_id = "custom_http_image"
    capabilities = (AICapabilityName.IMAGE_GENERATE.value,)

    def __init__(
        self,
        config: CustomHTTPImageConfig,
        credential_resolver: CredentialResolver,
        transport: JSONTransport | None = None,
    ) -> None:
        _require_http_url(config.endpoint, "endpoint")
        self.config = config
        self.credential_resolver = credential_resolver
        self.transport = transport or UrllibJSONTransport()
        self.image_features = config.features

    async def invoke(
        self, request: AICapabilityRequest, backend: AIBackendDescriptor
    ) -> AIImageGenerationOutput:
        prompt, count, references = _generation_input(request, backend)
        _validate_features(self.image_features, request.payload, count, references, backend)
        values = {
            "prompt": prompt,
            "count": count,
            "aspect_ratio": str(request.payload.get("aspect_ratio") or "auto"),
            "size": str(request.payload.get("size") or "auto"),
            "model": _model(request, backend, ""),
            "references": [
                {
                    "mime_type": image.mime_type,
                    "data": base64.b64encode(image.data).decode("ascii"),
                    "url": image.url,
                }
                for image in references
            ],
        }
        payload = _render_template(self.config.request_template, values)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **dict(self.config.extra_headers),
        }
        if self.config.credential_id:
            headers[self.config.auth_header] = self.config.auth_prefix + self._credential(
                self.config.credential_id, backend.backend_id
            )
        timeout_seconds = request.retry_policy.normalized().backend_timeout_seconds
        await mark_transport_send(json_transport_request(self.config.endpoint, payload))
        response = await self.transport.post_json(
            self.config.endpoint,
            headers=headers,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        _ensure_success(response)
        return _custom_output(
            response,
            image_paths=self.config.image_paths,
            mime_type_path=self.config.mime_type_path,
            model_path=self.config.model_path,
            fallback_model=str(values["model"]),
            count=count,
            has_references=bool(references),
            backend=backend,
        )


__all__ = [
    "CustomHTTPImageCapabilityAdapter",
    "CustomHTTPImageConfig",
    "GeminiImageCapabilityAdapter",
    "GeminiImageConfig",
    "MultipartTransport",
    "OpenAIChatImageCapabilityAdapter",
    "OpenAIChatImageConfig",
    "OpenAIImagesCapabilityAdapter",
    "OpenAIImagesConfig",
    "OpenAIVisionConfig",
    "OpenAIVisionDescribeAdapter",
    "UrllibImageTransport",
]
