"""Provider response parsing for image and vision adapters."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from ...contracts.ai_models import (
    AIBackendDescriptor,
    AIErrorCode,
    AIErrorInfo,
    AIImageContent,
    AIImageGenerationOutput,
    AIInvocationError,
    AIVisionDescription,
)
from ...contracts.vision import VisionTextState
from ...features.ai.openai_compatible import HTTPJSONResponse, OpenAIHTTPStatusError
from ...shared.http_security import require_secure_http_url
from .command_protocol import parse_model_turn


def vision_description(
    response: HTTPJSONResponse,
    model: str,
    backend: AIBackendDescriptor,
    *,
    require_source_marker: bool = False,
    require_social_impression: bool = False,
) -> AIVisionDescription:
    raw = _vision_response_content(response.data)
    data, source_marker_present = vision_command(
        raw,
        backend,
        require_source_marker=require_source_marker,
        require_social_impression=require_social_impression,
    )
    visible, ocr_text, text_state, safe = _validated_vision_fields(data, backend)
    return AIVisionDescription(
        visible_facts=visible,
        ocr_text=ocr_text,
        subject_identity=_text(data, "subject_identity"),
        sequence_observation=_text(data, "sequence_observation"),
        visual_style=_text(data, "visual_style"),
        sticker_type=_text(data, "sticker_type"),
        social_impression=_text(data, "social_impression"),
        visible_text_state=text_state,
        safe=safe,
        safety_reason=_text(data, "safety_assessment"),
        transient_source_marker_present=source_marker_present,
        model=str(response.data.get("model") or model),
        raw=data,
    )


def _vision_response_content(data: Mapping[str, Any]) -> str:
    choices = data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message", {}) if isinstance(choice, Mapping) else {}
    raw = message.get("content", "") if isinstance(message, Mapping) else ""
    if isinstance(raw, list):
        raw = "".join(str(item.get("text") or "") for item in raw if isinstance(item, Mapping))
    return str(raw or "")


def _validated_vision_fields(
    data: Mapping[str, Any], backend: AIBackendDescriptor
) -> tuple[str, str, str, bool]:
    visible = _text(data, "visible_facts")
    if not visible:
        raise output_error("Vision backend omitted the objective picture description", backend)
    ocr_text = _text(data, "ocr_text")
    text_state = _text(data, "visible_text_state").upper()
    if text_state == VisionTextState.TRANSCRIBED.value and not ocr_text:
        raise output_error("Vision backend declared transcribed text without text", backend)
    if text_state == VisionTextState.NO_TEXT.value and ocr_text:
        raise output_error("Vision backend returned text while declaring no text", backend)
    safe = data.get("safe")
    if not isinstance(safe, bool):
        raise output_error("Vision backend omitted the safety judgment", backend)
    return visible, ocr_text, text_state, safe


def _text(data: Mapping[str, Any], key: str, fallback: str = "") -> str:
    return str(data.get(key) or fallback).strip()


def gemini_output(
    response: HTTPJSONResponse,
    *,
    count: int,
    model: str,
    has_references: bool,
    backend: AIBackendDescriptor,
) -> AIImageGenerationOutput:
    images: list[AIImageContent] = []
    candidates = response.data.get("candidates")
    for candidate in candidates if isinstance(candidates, list) else ():
        content = candidate.get("content", {}) if isinstance(candidate, Mapping) else {}
        parts = content.get("parts", ()) if isinstance(content, Mapping) else ()
        for part in parts:
            image = _gemini_part(part)
            if image is not None:
                images.append(image)
    if not images:
        raise output_error("Gemini returned no image parts", backend)
    warnings = () if len(images) >= count else ("provider_returned_fewer_images",)
    return AIImageGenerationOutput(
        tuple(images[:count]),
        model=model,
        reference_mode="raw" if has_references else "none",
        warnings=warnings,
    )


def custom_output(
    response: HTTPJSONResponse,
    *,
    image_paths: tuple[str, ...],
    mime_type_path: str,
    model_path: str,
    fallback_model: str,
    count: int,
    has_references: bool,
    backend: AIBackendDescriptor,
) -> AIImageGenerationOutput:
    mime_values = _mime_values(response.data, mime_type_path)
    images: list[AIImageContent] = []
    for path in image_paths:
        for item in path_get_all(response.data, path):
            mime = _mime_for_index(mime_values, len(images))
            images.extend(images_from_unknown(item, default_mime=mime))
    if not images:
        raise output_error("Custom image backend returned no configured image data", backend)
    model = str(path_get(response.data, model_path) or fallback_model)
    return AIImageGenerationOutput(
        tuple(images[:count]),
        model=model,
        reference_mode="raw" if has_references else "none",
    )


def _has_first_mapping(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and isinstance(value[0], Mapping)


def _gemini_part(part: Any) -> AIImageContent | None:
    if not isinstance(part, Mapping):
        return None
    inline = part.get("inlineData") or part.get("inline_data")
    if not isinstance(inline, Mapping):
        return None
    mime_type = inline.get("mimeType") or inline.get("mime_type")
    return image_from_base64(inline.get("data"), mime_type)


def _mime_values(data: Mapping[str, Any], path: str) -> list[str]:
    return [str(item or "").strip() for item in path_get_all(data, path) if str(item or "").strip()]


def _mime_for_index(values: list[str], index: int) -> str:
    if index < len(values):
        return values[index]
    return values[0] if values else "image/png"


def parse_openai_images(
    data: Mapping[str, Any], model: str, reference_mode: str, backend: AIBackendDescriptor
) -> AIImageGenerationOutput:
    images: list[AIImageContent] = []
    revised = ""
    for item in data.get("data", ()) if isinstance(data.get("data"), list) else ():
        if not isinstance(item, Mapping):
            continue
        revised = revised or str(item.get("revised_prompt") or "")
        image = image_from_base64(item.get("b64_json"), item.get("mime_type"))
        if image:
            images.append(image)
        elif str(item.get("url") or "").startswith(("http://", "https://")):
            images.append(
                AIImageContent(str(item.get("mime_type") or "image/png"), url=str(item["url"]))
            )
    if not images:
        raise output_error("OpenAI Images returned no images", backend)
    return AIImageGenerationOutput(
        tuple(images),
        model=str(data.get("model") or model),
        revised_prompt=revised,
        reference_mode=reference_mode,
    )


def extract_chat_images(data: Mapping[str, Any]) -> list[AIImageContent]:
    output: list[AIImageContent] = []
    choices = data.get("choices")
    for choice in choices if isinstance(choices, list) else ():
        message = choice.get("message", {}) if isinstance(choice, Mapping) else {}
        if not isinstance(message, Mapping):
            continue
        for item in message.get("images", ()) if isinstance(message.get("images"), list) else ():
            output.extend(images_from_unknown(item))
        content = message.get("content")
        if isinstance(content, str):
            output.extend(images_from_text(content))
        else:
            output.extend(images_from_response_value(content))

    # Newer compatible services often return Responses-style output even when
    # their endpoint remains /chat/completions.  Vendor gateways also use
    # top-level images/data/result containers.
    for key in ("output", "images", "data", "result", "results", "artifacts"):
        if key in data:
            output.extend(images_from_response_value(data.get(key), key_hint=key))

    deduplicated: list[AIImageContent] = []
    seen: set[str] = set()
    for image in output:
        identity = image.url or (
            f"{image.mime_type}:{hashlib.sha256(image.data).hexdigest()}" if image.data else ""
        )
        if identity and identity not in seen:
            seen.add(identity)
            deduplicated.append(image)
    return deduplicated


def images_from_response_value(value: Any, *, key_hint: str = "") -> list[AIImageContent]:
    if isinstance(value, str):
        return _images_from_string_value(value, key_hint)
    if isinstance(value, list):
        return [
            image for item in value for image in images_from_response_value(item, key_hint=key_hint)
        ]
    if not isinstance(value, Mapping):
        return []
    return _images_from_mapping_value(value, key_hint)


def _images_from_string_value(value: str, key_hint: str) -> list[AIImageContent]:
    direct_keys = {
        "image",
        "image_url",
        "output_image",
        "generated_image",
        "b64_json",
        "base64",
        "inline_data",
        "inlineData",
        "result",
    }
    if key_hint in direct_keys:
        parsed = images_from_unknown(value)
        if parsed:
            return parsed
    return images_from_text(value)


def _images_from_mapping_value(
    value: Mapping[str, Any],
    key_hint: str,
) -> list[AIImageContent]:

    output: list[AIImageContent] = []
    part_type = str(value.get("type") or "").strip().lower()
    image_types = {
        "image",
        "image_url",
        "output_image",
        "generated_image",
        "input_image",
        "inline_data",
        "inlinedata",
        "image_generation_call",
    }
    image_keys = {
        "image",
        "images",
        "image_url",
        "output_image",
        "generated_image",
        "b64_json",
        "base64",
        "inline_data",
        "inlineData",
        "source",
    }
    if part_type in image_types or key_hint in image_keys:
        output.extend(images_from_unknown(_direct_image_value(value)))
    nested_keys = image_keys | {
        "content",
        "parts",
        "output",
        "data",
        "result",
        "results",
        "artifacts",
    }
    for key, nested in value.items():
        if key in nested_keys:
            output.extend(images_from_response_value(nested, key_hint=key))
    return output


def _direct_image_value(value: Mapping[str, Any]) -> Any:
    return (
        value.get("image_url")
        or value.get("output_image")
        or value.get("image")
        or value.get("source")
        or value.get("result")
        or value.get("inlineData")
        or value.get("inline_data")
        or value
    )


def images_from_text(value: str) -> list[AIImageContent]:
    text = str(value or "").strip()
    if not text:
        return []
    output: list[AIImageContent] = []

    # A few gateways JSON-encode the multimodal content inside message.content.
    if text[:1] in {"{", "["}:
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError):
            decoded = None
        if isinstance(decoded, (Mapping, list)):
            output.extend(images_from_response_value(decoded))

    for match in re.finditer(r"data:image/[^;,\s]+;base64,[A-Za-z0-9+/=\r\n]+", text):
        image = image_from_data_url(match.group(0).replace("\r", "").replace("\n", ""))
        if image:
            output.append(image)

    markdown_urls = {
        match.group(1).strip() for match in re.finditer(r"!\[[^\]]*\]\((https?://[^)\s]+)\)", text)
    }
    raw_urls = {
        match.group(0).rstrip('.,;:!?)"]}') for match in re.finditer(r"https?://[^\s<>\"']+", text)
    }
    for url in sorted(markdown_urls | raw_urls):
        if url:
            output.append(AIImageContent(url=url))
    return output


def images_from_unknown(value: Any, *, default_mime: str = "image/png") -> list[AIImageContent]:
    if isinstance(value, str):
        return _images_from_unknown_string(value, default_mime)
    if not isinstance(value, Mapping):
        return []
    nested, mime_type = _mapping_image_value(value, default_mime)
    if isinstance(nested, str) and nested.startswith("data:"):
        image = image_from_data_url(nested)
        return [image] if image else []
    if isinstance(nested, str) and nested.startswith(("http://", "https://")):
        return [AIImageContent(mime_type, url=nested)]
    image = image_from_base64(nested, mime_type)
    return [image] if image else []


def _images_from_unknown_string(value: str, default_mime: str) -> list[AIImageContent]:
    image = image_from_data_url(value)
    if image:
        return [image]
    if value.startswith(("http://", "https://")):
        return [AIImageContent(url=value)]
    image = image_from_base64(value, default_mime)
    return [image] if image else []


def _mapping_image_value(
    value: Mapping[str, Any],
    default_mime: str,
) -> tuple[Any, str]:
    nested = next(
        (
            value.get(key)
            for key in ("url", "data", "b64_json", "base64", "image_base64", "bytes", "result")
            if value.get(key) is not None
        ),
        None,
    )
    source = value.get("source")
    source_mapping = source if isinstance(source, Mapping) else {}
    if nested is None:
        nested = (
            source_mapping.get("data") or source_mapping.get("url") or source_mapping.get("base64")
        )
    image_url = value.get("image_url")
    if nested is None and isinstance(image_url, Mapping):
        nested = image_url.get("url")
    mime = value.get("mime_type") or value.get("mimeType")
    mime = mime or source_mapping.get("media_type") or default_mime
    return nested, str(mime)


def image_from_base64(value: Any, mime_type: Any) -> AIImageContent | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        return None
    return AIImageContent(str(mime_type or "image/png"), data=data)


def image_from_data_url(value: str) -> AIImageContent | None:
    match = re.fullmatch(r"data:([^;,]+);base64,(.+)", value, re.DOTALL)
    return image_from_base64(match.group(2), match.group(1)) if match else None


def bearer_headers(credential: str, extra: Mapping[str, str]) -> Mapping[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {credential}",
        **dict(extra),
    }


def ensure_success(response: HTTPJSONResponse) -> None:
    if 200 <= response.status_code < 300:
        return
    error = response.data.get("error")
    code = str(error.get("code") or error.get("type") or "") if isinstance(error, Mapping) else ""
    raise OpenAIHTTPStatusError(response.status_code, api_code=code)


def vision_command(
    value: str,
    backend: AIBackendDescriptor,
    *,
    require_source_marker: bool = False,
    require_social_impression: bool = False,
) -> tuple[Mapping[str, Any], bool | None]:
    fields = _vision_fields(value, backend)
    data = _normalized_vision_data(fields, backend)
    source_marker_present = (
        _required_vision_boolean(
            fields,
            "存在非内容来源标记",
            "Vision backend returned an invalid source-marker value",
            backend,
        )
        if require_source_marker
        else None
    )
    if require_social_impression and not data["social_impression"]:
        raise output_error("Vision backend omitted the social impression", backend)
    return data, source_marker_present


def _vision_fields(value: str, backend: AIBackendDescriptor) -> Mapping[str, Any]:
    parsed = parse_model_turn(value)
    if parsed.errors or parsed.working_text or len(parsed.commands) != 1:
        raise output_error("Vision backend returned an invalid text-command contract", backend)
    command = parsed.commands[0]
    if command.name != "图像观察":
        raise output_error("Vision backend omitted the 图像观察 block", backend)
    return command.parameters


def _normalized_vision_data(
    fields: Mapping[str, Any],
    backend: AIBackendDescriptor,
) -> dict[str, Any]:
    mapping = {
        "visible_facts": "画面内容",
        "subject_identity": "主体身份",
        "sequence_observation": "动作变化",
        "visual_style": "画面媒介",
        "sticker_type": "表情包特征",
        "social_impression": "交流观感",
        "ocr_text": "正文文字",
        "visible_text_state": "文字状态",
        "safety_assessment": "安全说明",
    }
    data = {key: str(fields.get(label) or "").strip() for key, label in mapping.items()}
    data["social_impression"] = data["social_impression"][:80]
    data["visible_text_state"] = {
        "无正文文字": VisionTextState.NO_TEXT.value,
        "正文已转录": VisionTextState.TRANSCRIBED.value,
        "正文不能完整辨认": VisionTextState.UNCLEAR_TEXT.value,
    }.get(data["visible_text_state"], "")
    if not data["visible_text_state"]:
        raise output_error("Vision backend returned an invalid text-state value", backend)
    data["safe"] = _required_vision_boolean(
        fields,
        "安全",
        "Vision backend returned an invalid safety value",
        backend,
    )
    if not data["safe"] and not data["safety_assessment"]:
        raise output_error("Vision backend omitted the safety explanation", backend)
    return data


def _required_vision_boolean(
    fields: Mapping[str, Any],
    label: str,
    error_message: str,
    backend: AIBackendDescriptor,
) -> bool:
    value = str(fields.get(label) or "").strip().lower()
    if value not in {"是", "否"}:
        raise output_error(error_message, backend)
    return value == "是"


def render_template(value: Any, values: Mapping[str, Any]) -> Any:
    if isinstance(value, Mapping):
        return {str(key): render_template(item, values) for key, item in value.items()}
    if isinstance(value, list):
        return [render_template(item, values) for item in value]
    if isinstance(value, str):
        match = re.fullmatch(r"\{\{([a-z_]+)\}\}", value)
        if match:
            return values.get(match.group(1), "")
        output = value
        for key, replacement in values.items():
            if isinstance(replacement, (str, int, float)):
                output = output.replace("{{" + key + "}}", str(replacement))
        return output
    return value


def path_get(data: Any, path: str) -> Any:
    values = path_get_all(data, path)
    return values[0] if values else None


def path_get_all(data: Any, path: str) -> list[Any]:
    """Resolve dot paths with list indexes and ``*`` expansion."""

    parts = [part for part in str(path or "").split(".") if part]

    def descend(current: Any, index: int) -> list[Any]:
        if index >= len(parts):
            return list(current) if isinstance(current, list) else [current]
        part = parts[index]
        if part == "*":
            values = (
                list(current.values())
                if isinstance(current, Mapping)
                else list(current)
                if isinstance(current, list)
                else []
            )
            result: list[Any] = []
            for value in values:
                result.extend(descend(value, index + 1))
            return result
        if isinstance(current, Mapping) and part in current:
            return descend(current[part], index + 1)
        if isinstance(current, list) and part.isdigit():
            position = int(part)
            if 0 <= position < len(current):
                return descend(current[position], index + 1)
        return []

    return [value for value in descend(data, 0) if value is not None]


def extension(mime_type: str) -> str:
    return {"image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}.get(
        mime_type.lower(), "png"
    )


def require_http_url(value: str, name: str) -> None:
    require_secure_http_url(value, name)


def invalid(message: str, backend: AIBackendDescriptor) -> AIInvocationError:
    return AIInvocationError(
        AIErrorInfo(
            AIErrorCode.INVALID_REQUEST, message, backend_id=backend.backend_id, phase="prepare"
        )
    )


def output_error(message: str, backend: AIBackendDescriptor) -> AIInvocationError:
    return AIInvocationError(
        AIErrorInfo(
            AIErrorCode.OUTPUT_CONTRACT,
            message,
            retryable=True,
            switch_backend=True,
            backend_id=backend.backend_id,
            phase="response",
            details={"provider_response_accepted": True},
        )
    )
