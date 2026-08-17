"""Cost-aware administrator probes for configured AI backends and models."""

from __future__ import annotations

import base64
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from ....contracts.ai_models import (
    AIAudioContent,
    AICapabilityEffect,
    AICapabilityRequest,
    AIExecutionMode,
    AIImageContent,
    AIImageGenerationOutput,
    AIModelRequest,
    AIRetryPolicy,
    AISpeechResult,
    AITranscriptionResult,
    AIVisionDescription,
    AIWorkPurpose,
)
from ....contracts.vision import VisionInspectionMode, VisionSequenceKind
from ....features.ai.service import AIManager
from ....shared.prompt_document import compile_task_prompt
from .ai_support import build_audio_probe_sample, build_vision_probe_challenge


def _image_probe_summary(output: Any, _challenge: str) -> dict[str, Any]:
    if not isinstance(output, AIImageGenerationOutput):
        raise TypeError("image probe returned an invalid output")
    preview = ""
    if output.images:
        image = output.images[0]
        if image.data and len(image.data) <= 5 * 1024 * 1024:
            mime_type = str(image.mime_type or "image/png").lower()
            if mime_type in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
                preview = f"data:{mime_type};base64,{base64.b64encode(image.data).decode('ascii')}"
    return {"image_count": len(output.images), "preview_data_url": preview}


def _transcription_probe_summary(output: Any, _challenge: str) -> dict[str, Any]:
    if not isinstance(output, AITranscriptionResult):
        raise TypeError("audio transcription probe returned an invalid output")
    return {
        "transcription_received": True,
        "text_length": len(str(output.text or "")),
        "language": str(output.language or ""),
    }


def _speech_probe_summary(output: Any, _challenge: str) -> dict[str, Any]:
    if not isinstance(output, AISpeechResult):
        raise TypeError("speech probe returned an invalid output")
    if not output.audio.data:
        raise RuntimeError("语音服务返回了空音频")
    duration_seconds = output.audio.duration_seconds
    return {
        "audio_received": True,
        "audio_format": str(output.audio.mime_type or ""),
        "byte_length": len(output.audio.data),
        "duration_ms": (None if duration_seconds is None else int(round(duration_seconds * 1000))),
    }


def _vision_probe_summary(output: Any, challenge: str) -> dict[str, Any]:
    if not isinstance(output, AIVisionDescription):
        raise TypeError("vision probe returned an invalid output")
    evidence = " ".join(str(value or "") for value in (output.visible_facts, output.ocr_text))
    if challenge not in re.sub(r"[^0-9A-Z]", "", evidence.upper()):
        raise RuntimeError("视觉模型未能读取测试图校验码；图片可能未送达模型")
    return {"description_received": True, "image_challenge_verified": True}


_CAPABILITY_SUMMARIZERS = {
    "image.generate": _image_probe_summary,
    "audio.transcribe": _transcription_probe_summary,
    "audio.speech": _speech_probe_summary,
}


class AIProbeRepositoryPort(Protocol):
    async def get_ai_backend(self, backend_id: str) -> Mapping[str, Any] | None: ...
    async def get_ai_api_model(self, backend_id: str) -> Mapping[str, Any] | None: ...
    async def get_ai_api_package(
        self, package_id: str, **values: object
    ) -> Mapping[str, Any] | None: ...
    async def list_ai_api_models(
        self, *values: object, **named: object
    ) -> Sequence[Mapping[str, Any]]: ...
    async def record_ai_backend_success(self, backend_id: str) -> object: ...
    async def record_ai_backend_failure(
        self, backend_id: str, *values: object, **named: object
    ) -> object: ...


class AIProbeController:
    def __init__(self, repository: AIProbeRepositoryPort, ai_manager: AIManager) -> None:
        self.repository = repository
        self.ai_manager = ai_manager

    async def probe_backend(self, backend_id: str, profile_id: str) -> dict[str, Any]:
        backend = await self.repository.get_ai_backend(backend_id)
        if backend is None:
            raise ValueError("unknown AI backend")
        try:
            text = await self._invoke_backend(backend_id, profile_id, backend)
            return {
                "ok": True,
                "backend_id": backend_id,
                "response_received": bool(str(text or "").strip()),
                "billing_note": "该健康探针可能产生少量模型费用。",
            }
        except Exception as exc:
            await self._record_backend_failure(backend_id, backend, exc)
            return {"ok": False, "backend_id": backend_id, "error": f"{type(exc).__name__}: {exc}"}

    async def _invoke_backend(
        self, backend_id: str, profile_id: str, backend: Mapping[str, Any]
    ) -> str:
        del backend
        registration = self.ai_manager.backends.get(backend_id)
        if registration is None:
            raise RuntimeError("SoulCore 直连模型尚未成功加载")
        self.ai_manager.force_probe(registration.descriptor, "chat.completion")
        self.ai_manager.force_package_probe(registration.descriptor)
        result = await self.ai_manager.invoke_model(
            self._text_request(
                backend_id,
                profile_id,
                "BACKEND_PROBE",
                "probe",
                "chat.completion",
            )
        )
        return str(result.completion.text or "")

    async def _record_backend_failure(
        self, backend_id: str, backend: Mapping[str, Any], exc: Exception
    ) -> None:
        del backend_id, backend, exc

    async def probe_package(
        self, package_id: str, profile_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        package = await self.repository.get_ai_api_package(package_id, profile_id=profile_id)
        if package is None:
            raise ValueError("找不到该API配置")
        models = await self._probe_models(package_id, profile_id, payload)
        return await self.probe_model(str(models[0]["backend_id"]), profile_id, payload)

    async def _probe_models(
        self, package_id: str, profile_id: str, payload: Mapping[str, Any]
    ) -> list[Mapping[str, Any]]:
        models = [
            model
            for model in await self.repository.list_ai_api_models(package_id, profile_id=profile_id)
            if bool(model.get("enabled", True))
        ]
        requested = str(payload.get("capability") or "").strip().lower()
        if requested:
            models = [
                model for model in models if requested in set(model.get("capabilities") or ())
            ]
        if not models:
            raise ValueError("该API配置下没有匹配所选用途的模型")
        models.sort(key=self._probe_sort_key)
        return models

    async def probe_model(
        self, backend_id: str, profile_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        model, package = await self._enabled_model(backend_id, profile_id)
        capability = self._probe_capability(model, payload)
        probe_capability = self._runtime_capability(capability)
        try:
            summary = await self._invoke_model_probe(
                backend_id, profile_id, capability, probe_capability, package, payload
            )
            return {
                "ok": True,
                "package_id": str(model["package_id"]),
                "model_id": backend_id,
                "capability": capability,
                **summary,
                "billing_note": "该检测会产生一次真实模型调用和少量费用。",
            }
        except ValueError:
            raise
        except Exception as exc:
            return {
                "ok": False,
                "package_id": str(model["package_id"]),
                "model_id": backend_id,
                "capability": capability,
                "error": f"{type(exc).__name__}: {exc}",
            }

    async def _enabled_model(
        self, backend_id: str, profile_id: str
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        model = await self.repository.get_ai_api_model(backend_id)
        if model is None:
            raise ValueError("找不到该模型")
        package = await self.repository.get_ai_api_package(
            str(model["package_id"]), profile_id=profile_id
        )
        if package is None or not bool(package.get("enabled", True)):
            raise ValueError("所属API配置当前未启用")
        if not bool(model.get("enabled", True)):
            raise ValueError("该模型当前未启用")
        return model, package

    async def _invoke_model_probe(
        self,
        backend_id: str,
        profile_id: str,
        capability: str,
        probe_capability: str,
        package: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if probe_capability in {
            "chat.completion",
            "conversation.turn_buffer",
            "conversation.group_interjection",
            "conversation.group_reply_relocation",
            "conversation.timer_lifecycle_review",
            "conversation.response_polish",
            "conversation.summary",
            "memory.reasoning",
            "text.completion",
        }:
            return await self._invoke_text_model(backend_id, profile_id, probe_capability)
        descriptor = self._capability_descriptor(backend_id, capability)
        self.ai_manager.force_probe(descriptor, capability)
        self.ai_manager.force_package_probe(descriptor)
        request, challenge = self._capability_request(backend_id, profile_id, capability, payload)
        result = await self.ai_manager.invoke_capability(request)
        return self._capability_summary(capability, result, challenge)

    async def _invoke_text_model(
        self, backend_id: str, profile_id: str, capability: str
    ) -> dict[str, Any]:
        registration = self.ai_manager.backends.get(backend_id)
        if registration is None:
            raise RuntimeError("该文字模型尚未成功加载")
        self.ai_manager.force_probe(registration.descriptor, capability)
        self.ai_manager.force_package_probe(registration.descriptor)
        result = await self.ai_manager.invoke_model(
            self._text_request(
                backend_id,
                profile_id,
                "API_MODEL_PROBE",
                "api-model-probe",
                capability,
            )
        )
        return {"response_received": bool(str(result.completion.text or "").strip())}

    def _capability_descriptor(self, backend_id: str, capability: str) -> Any:
        candidates = self.ai_manager.capabilities.candidates(capability, (backend_id,))
        if not candidates:
            raise RuntimeError("该模型用途尚未成功加载")
        return candidates[0].descriptor

    @staticmethod
    def _capability_request(
        backend_id: str, profile_id: str, capability: str, payload: Mapping[str, Any]
    ) -> tuple[AICapabilityRequest, str]:
        challenge = ""
        if capability == "image.generate":
            if payload.get("confirm_cost") is not True:
                raise ValueError("真实生图检测会产生费用，请先确认后再执行")
            request_payload = {
                "prompt": "A simple solid green circle on a plain white background.",
                "count": 1,
                "aspect_ratio": "1:1",
                "size": "auto",
                "references": (),
            }
            effect = AICapabilityEffect.NON_IDEMPOTENT_WRITE
        elif capability == "audio.transcribe":
            request_payload = {
                "audio": AIAudioContent(
                    data=build_audio_probe_sample(),
                    mime_type="audio/wav",
                    filename="soulcore-audio-probe.wav",
                    duration_seconds=0.4,
                )
            }
            effect = AICapabilityEffect.READ_ONLY
        elif capability == "audio.speech":
            if payload.get("confirm_cost") is not True:
                raise ValueError("真实语音合成检测会产生费用，请先确认后再执行")
            request_payload = {"text": "你好，这是 SoulCore 语音连接测试。"}
            effect = AICapabilityEffect.NON_IDEMPOTENT_WRITE
        elif capability == "vision.describe":
            challenge, image = build_vision_probe_challenge()
            request_payload = {
                "images": (AIImageContent("image/png", data=image),),
                "sequence_kind": VisionSequenceKind.SINGLE_IMAGE.value,
                "inspection_mode": VisionInspectionMode.OCR_DIAGNOSTIC.value,
            }
            effect = AICapabilityEffect.READ_ONLY
        else:
            raise ValueError("该模型用途暂不支持连接检测")
        request = AICapabilityRequest(
            invocation_id=f"api-model-probe-{uuid.uuid4().hex}",
            capability=capability,
            work_purpose=AIWorkPurpose.ADMIN_MODEL_TEST,
            logical_stage_key=f"api-model-probe:{backend_id}:{uuid.uuid4().hex}",
            payload=request_payload,
            backend_ids=(backend_id,),
            effect=effect,
            execution_mode=AIExecutionMode.DEBUG_EPHEMERAL,
            profile_id=profile_id,
            owner_kind="API_MODEL_PROBE",
            idempotency_key=f"api-model-probe:{uuid.uuid4().hex}",
            retry_policy=AIRetryPolicy(max_attempts=1),
        )
        return request, challenge

    @staticmethod
    def _capability_summary(capability: str, result: Any, challenge: str) -> dict[str, Any]:
        summarize = _CAPABILITY_SUMMARIZERS.get(capability, _vision_probe_summary)
        return summarize(result.output, challenge)

    @staticmethod
    def _text_request(
        backend_id: str,
        profile_id: str,
        owner_kind: str,
        prefix: str,
        capability: str,
    ) -> AIModelRequest:
        # The probe has no dynamic turn data.  Keep one compact logical document
        # instead of inventing a duplicate task_input merely to force two chat
        # messages; the OpenAI-compatible boundary deliberately wraps a
        # context-only document as one user message.
        compiled = compile_task_prompt(
            task_definition="执行最小文字模型健康检查，不使用任何对话数据。",
            task_input="",
            output_contract="只返回 OK。",
            model_id=backend_id,
        )
        return AIModelRequest(
            invocation_id=f"{prefix}-{uuid.uuid4().hex}",
            work_purpose=AIWorkPurpose.ADMIN_MODEL_TEST,
            logical_stage_key=f"{prefix}:{backend_id}:{uuid.uuid4().hex}",
            backend_ids=(backend_id,),
            context_text=compiled.context_text,
            turn_text=compiled.turn_text,
            execution_mode=AIExecutionMode.DEBUG_EPHEMERAL,
            profile_id=profile_id,
            owner_kind=owner_kind,
            retry_policy=AIRetryPolicy(max_attempts=1),
            parameters={},
            metadata={
                "capability": capability,
                "routing_capability": capability,
                "prompt_document": compiled.debug_payload(),
            },
        )

    @staticmethod
    def _probe_capability(model: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
        capabilities = [str(item) for item in model.get("capabilities") or ()]
        requested = str(payload.get("capability") or "").strip().lower()
        if requested in capabilities:
            return requested
        order = (
            "chat.completion",
            "conversation.turn_buffer",
            "conversation.group_interjection",
            "conversation.group_reply_relocation",
            "conversation.timer_lifecycle_review",
            "conversation.response_polish",
            "conversation.summary",
            "memory.reasoning",
            "text.completion",
            "sticker.collect",
            "sticker.check",
            "vision.describe",
            "audio.transcribe",
            "audio.speech",
            "image.generate",
        )
        selected = next((item for item in order if item in capabilities), "")
        if not selected:
            raise ValueError("该模型没有当前可检测的用途")
        return selected

    @staticmethod
    def _runtime_capability(capability: str) -> str:
        if capability in {
            "conversation.turn_buffer",
            "conversation.group_interjection",
            "conversation.group_reply_relocation",
            "conversation.timer_lifecycle_review",
            "conversation.response_polish",
            "conversation.summary",
            "memory.reasoning",
        }:
            return "text.completion"
        aliases = {
            "sticker.collect",
            "sticker.check",
        }
        return "chat.completion" if capability in aliases else capability

    @staticmethod
    def _probe_sort_key(model: Mapping[str, Any]) -> tuple[int, int]:
        capabilities = set(model.get("capabilities") or ())
        text_rank = (
            0
            if {
                "chat.completion",
                "conversation.turn_buffer",
                "conversation.group_interjection",
                "conversation.group_reply_relocation",
                "conversation.timer_lifecycle_review",
                "conversation.response_polish",
                "memory.reasoning",
                "text.completion",
            }.intersection(capabilities)
            else 1
        )
        return text_rank, int(model.get("priority") or 1)


__all__ = ["AIProbeController", "AIProbeRepositoryPort"]
