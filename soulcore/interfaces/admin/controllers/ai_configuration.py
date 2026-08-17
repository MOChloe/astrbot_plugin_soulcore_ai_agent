"""SoulCore direct API package, model, and backend administration."""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from ....features.ai.model_parameters import (
    DEFAULT_MODEL_MAX_CONTEXT_TOKENS,
    MINIMUM_MODEL_MAX_CONTEXT_TOKENS,
    TEXT_GENERATION_CAPABILITIES,
    normalize_model_custom_request_parameters,
    normalize_model_generation_parameters,
)
from ....features.ai.ports import AIConfigurationRepositoryPort
from ....features.ai.proxy_context_isolation import (
    PROXY_CONTEXT_ISOLATION_CONFIG_KEY,
    proxy_context_isolation_enabled,
)
from ....features.profiles.credentials import CredentialVault
from ....shared.http_security import require_secure_http_url
from ..presentation import jsonable
from . import ai_audio_configuration
from .ai_support import AI_CAPABILITIES


def prompt_cache_view(record: Mapping[str, Any]) -> dict[str, Any]:
    state = str(record.get("state") or "UNTESTED").upper()
    read_tokens = max(0, int(record.get("cache_read_tokens") or 0))
    write_tokens = max(0, int(record.get("cache_write_tokens") or 0))
    quality = _prompt_cache_quality(record)
    quality_status = str(quality.get("status") or "OBSERVING").upper()
    quality_reason = str(quality.get("reason") or "")
    anomaly_count = max(0, int(quality.get("anomaly_count") or 0))
    recent_read = max(0, int(quality.get("last_read_tokens") or 0))
    recent_write = max(0, int(quality.get("last_write_tokens") or 0))
    return {
        "state": state,
        "label": _prompt_cache_label(state, quality_status, read_tokens, write_tokens),
        "wire_mode": str(record.get("wire_mode") or ""),
        "cache_read_tokens": read_tokens,
        "cache_write_tokens": write_tokens,
        "recent_cache_read_tokens": recent_read,
        "recent_cache_write_tokens": recent_write,
        "quality_status": quality_status,
        "quality_reason": quality_reason,
        "anomaly_count": anomaly_count,
        "next_probe_at": jsonable(record.get("next_probe_at")),
        "can_reprobe": _prompt_cache_can_reprobe(record, state, quality_status),
    }


def _prompt_cache_quality(record: Mapping[str, Any]) -> dict[str, Any]:
    evidence = record.get("evidence")
    if not isinstance(evidence, Mapping):
        return {}
    return dict(evidence.get("quality") or {})


def _prompt_cache_can_reprobe(record: Mapping[str, Any], state: str, quality_status: str) -> bool:
    rejection = record.get("rejection")
    rejection_kind = str(rejection.get("kind") or "") if isinstance(rejection, Mapping) else ""
    return (
        state == "REJECTED"
        and rejection_kind == "CACHE_QUALITY"
        and quality_status != "REPROBE_READY"
    )


def _prompt_cache_label(
    state: str, quality_status: str, read_tokens: int, write_tokens: int
) -> str:
    quality_labels = {
        "SUSPENDED": "缓存已暂停",
        "AUTO_WARNING": "服务商自动缓存异常",
        "UNCONTROLLABLE_WARNING": "服务商自动缓存异常",
        "ANOMALY": "缓存质量异常",
        "REPROBE_READY": "等待下一次真实请求复探",
        "REPROBING": "正在随真实请求复探",
    }
    if quality_status in quality_labels:
        return quality_labels[quality_status]
    if state == "CONFIRMED":
        if read_tokens > 0:
            return "已确认命中"
        if write_tokens > 0:
            return "已确认写入"
        return "已确认支持，尚未命中"
    return {
        "UNTESTED": "等待合适真实请求",
        "PROBING": "正在随真实请求验证",
        "ACCEPTED_UNVERIFIED": "已接受，尚无缓存用量证据",
        "REJECTED": "缓存标记被拒绝",
    }.get(state, "等待合适真实请求")


ReloadBackends = Callable[[], Awaitable[None]]
MODEL_CAPABILITIES = {
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
    "image.generate",
    "audio.transcribe",
    "audio.speech",
}
_AUDIO_CAPABILITIES = ai_audio_configuration.AUDIO_CAPABILITIES
_AUDIO_SPEECH_PROTOCOLS = ai_audio_configuration.AUDIO_SPEECH_PROTOCOLS
_LOCAL_NO_CREDENTIAL_PROTOCOLS = ai_audio_configuration.LOCAL_NO_CREDENTIAL_PROTOCOLS
_PROXY_CONTEXT_ISOLATION_PROTOCOLS = frozenset({"OPENAI", "OPENAI_COMPATIBLE", "ANTHROPIC"})
_SUPPORTED_PROTOCOLS = frozenset(
    {
        "OPENAI",
        "OPENAI_COMPATIBLE",
        "ANTHROPIC",
        "GEMINI",
        "CUSTOM_HTTP_IMAGE",
        *_AUDIO_SPEECH_PROTOCOLS,
    }
)
assert set(AI_CAPABILITIES) >= MODEL_CAPABILITIES


def _normalize_model_capabilities(raw: Any, *, reject_unknown: bool) -> list[str]:
    del reject_unknown
    if isinstance(raw, Mapping):
        raw = [name for name, enabled in raw.items() if enabled]
    elif isinstance(raw, str):
        raw = [raw]
    values = {name for item in raw or [] if (name := str(item).strip().lower())}
    if values - MODEL_CAPABILITIES:
        raise ValueError("请选择至少一项当前支持的模型用途")
    return sorted(values)


class AIConfigurationController:
    def __init__(
        self,
        repository: AIConfigurationRepositoryPort,
        credential_vault: CredentialVault,
        reload_backends: ReloadBackends,
        runtime_context: Any | None = None,
    ) -> None:
        self.repository = repository
        self.credential_vault = credential_vault
        self.reload_backends = reload_backends
        self.runtime_context = runtime_context

    async def snapshot(self, profile_id: str) -> dict[str, Any]:
        packages = await self.repository.list_ai_api_packages(profile_id)
        models = await self.repository.list_ai_api_models(profile_id=profile_id)
        backends = {
            str(row.get("backend_id") or ""): row
            for row in await self.repository.list_ai_backends()
        }
        cache_capabilities = {
            str(row.get("backend_id") or ""): row
            for row in await self.repository.list_ai_prompt_cache_capabilities()
        }
        result = self._package_views(packages, models, backends, cache_capabilities)
        pool_orders = await self._pool_orders(self._model_backend_ids(result))
        return {
            "api_packages": result,
            "effective_orders": self._effective_orders(result, pool_orders),
            "protocol_options": [
                "OPENAI",
                "OPENAI_COMPATIBLE",
                "ANTHROPIC",
                "GEMINI",
                "CUSTOM_HTTP_IMAGE",
                "MINIMAX_TTS",
                "MIMO_TTS",
                "GPT_SOVITS_V2",
                "GSVI_TTS",
            ],
            "capability_options": sorted(MODEL_CAPABILITIES),
            "native_stt_conflict": self._native_stt_conflict_view(),
        }

    def _native_stt_conflict_view(self) -> dict[str, Any]:
        context = self.runtime_context
        provider_manager = getattr(context, "provider_manager", None)
        settings = getattr(provider_manager, "provider_stt_settings", None)
        enabled_value = settings.get("enable") if isinstance(settings, Mapping) else None
        if not isinstance(enabled_value, bool):
            return {
                "state": "UNKNOWN",
                "enabled": None,
                "conflict": False,
                "checked": False,
            }
        enabled = enabled_value
        return {
            "state": "ENABLED" if enabled else "DISABLED",
            "enabled": enabled,
            "conflict": enabled,
            "checked": True,
        }

    def _package_views(
        self,
        packages: list[Mapping[str, Any]],
        models: list[Mapping[str, Any]],
        backends: Mapping[str, Mapping[str, Any]],
        cache_capabilities: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for model in models:
            grouped.setdefault(str(model["package_id"]), []).append(
                self._model_view(
                    model,
                    backends.get(str(model.get("backend_id") or ""), {}),
                    cache_capabilities.get(str(model.get("backend_id") or ""), {}),
                )
            )
        return [
            self._package_view(package, grouped.get(str(package["package_id"]), []))
            for package in packages
        ]

    @staticmethod
    def _model_backend_ids(packages: list[dict[str, Any]]) -> set[str]:
        return {
            str(model.get("backend_id") or "")
            for package in packages
            for model in package.get("models", [])
        }

    async def _pool_orders(self, model_ids: set[str]) -> dict[str, list[str]]:
        pool_orders: dict[str, list[str]] = {}
        for capability in sorted(MODEL_CAPABILITIES):
            rows = await self.repository.list_ai_capability_pool(capability)
            relevant = [row for row in rows if str(row.get("backend_id") or "") in model_ids]
            if relevant:
                pool_orders[capability] = [
                    str(row.get("backend_id") or "") for row in relevant if bool(row.get("enabled"))
                ]
        return pool_orders

    @staticmethod
    def _model_view(
        model: Mapping[str, Any],
        backend: Mapping[str, Any],
        cache_capability: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = dict(model.get("config") or {})
        capabilities = _normalize_model_capabilities(
            model.get("capabilities"), reject_unknown=False
        )
        audio_only = AIConfigurationController._audio_only(capabilities)
        text_capable = bool(set(capabilities).intersection(TEXT_GENERATION_CAPABILITIES))
        health = str(backend.get("health_status") or "").upper()
        status = AIConfigurationController._model_status(model, health)
        details = AIConfigurationController._model_capability_view(
            config,
            capabilities,
            cache_capability or {},
            audio_only=audio_only,
            text_capable=text_capable,
        )
        return {
            **jsonable(model),
            "capabilities": capabilities,
            **details,
            "audio_settings": AIConfigurationController._audio_settings_view(config, capabilities),
            "status": status,
        }

    @staticmethod
    def _model_status(model: Mapping[str, Any], health: str) -> str:
        if not model.get("enabled", True):
            return "DISABLED"
        return health if health and health != "UNKNOWN" else "CONFIGURED"

    @staticmethod
    def _model_capability_view(
        config: Mapping[str, Any],
        capabilities: list[str],
        cache_capability: Mapping[str, Any],
        *,
        audio_only: bool,
        text_capable: bool,
    ) -> dict[str, Any]:
        if audio_only:
            return {
                "max_context_tokens": None,
                "image_generation_mode": None,
                "supports_vision": None,
                "generation_parameters": {},
                "custom_request_parameters": {},
                "prompt_cache": None,
            }
        return {
            "max_context_tokens": (
                int(config.get("max_context_tokens") or DEFAULT_MODEL_MAX_CONTEXT_TOKENS)
                if text_capable
                else None
            ),
            "image_generation_mode": (
                "CHAT_RESPONSE"
                if str(config.get("image_generation_mode") or "").upper() == "CHAT_RESPONSE"
                else "IMAGES_API"
            ),
            "supports_vision": AIConfigurationController._supports_vision(config),
            "generation_parameters": normalize_model_generation_parameters(
                config.get("generation_parameters")
            ),
            "custom_request_parameters": normalize_model_custom_request_parameters(
                config.get("custom_request_parameters")
            ),
            "prompt_cache": prompt_cache_view(cache_capability),
        }

    def _package_view(
        self, package: Mapping[str, Any], models: list[dict[str, Any]]
    ) -> dict[str, Any]:
        credential = self._credential(str(package.get("credential_id") or ""))
        statuses = {str(model.get("status") or "CONFIGURED").upper() for model in models}
        enabled = sum(1 for model in models if model.get("enabled"))
        status = self._package_status(bool(package.get("enabled", True)), enabled, statuses)
        return {
            **jsonable(package),
            PROXY_CONTEXT_ISOLATION_CONFIG_KEY: proxy_context_isolation_enabled(
                package.get("config")
            ),
            "credential": credential,
            "credential_required": str(package.get("protocol") or "").upper()
            not in _LOCAL_NO_CREDENTIAL_PROTOCOLS,
            "editable": True,
            "models": models,
            "model_count": len(models),
            "enabled_model_count": enabled,
            "status": status,
        }

    def _credential(self, credential_id: str) -> dict[str, Any]:
        if not credential_id:
            return {"configured": False, "last4": "", "source": ""}
        info = self.credential_vault.describe(credential_id)
        return {"configured": info.configured, "last4": info.last4, "source": info.source}

    @staticmethod
    def _package_status(enabled: bool, count: int, statuses: set[str]) -> str:
        if not enabled:
            return "PAUSED"
        if count == 0:
            return "UNCONFIGURED"
        if statuses.intersection({"UNHEALTHY", "FAILED", "OPEN"}):
            return "UNHEALTHY"
        return "HEALTHY" if "HEALTHY" in statuses else "CONFIGURED"

    @staticmethod
    def _effective_orders(
        packages: list[dict[str, Any]],
        pool_orders: Mapping[str, list[str]] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        result = {}
        persisted_orders = pool_orders or {}
        for capability in sorted(MODEL_CAPABILITIES):
            selected = [
                {
                    **model,
                    "package_id": package["package_id"],
                    "package_display_name": package["display_name"],
                }
                for package in packages
                if package.get("enabled", True)
                for model in package.get("models", [])
                if model.get("enabled", True) and capability in set(model.get("capabilities") or ())
            ]
            if capability in persisted_orders:
                by_id = {str(model.get("backend_id") or ""): model for model in selected}
                selected = [
                    by_id[backend_id]
                    for backend_id in persisted_orders[capability]
                    if backend_id in by_id
                ]
            else:
                selected.sort(
                    key=lambda model: (
                        int(model.get("priority") or 1),
                        str(model.get("backend_id") or ""),
                    )
                )
            result[capability] = selected
        return result

    async def save_package(self, payload: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
        action = str(payload.get("action") or "save").strip().lower()
        package_id = str(payload.get("package_id") or "").strip()
        if not package_id and action == "save":
            package_id = f"api-package:{uuid.uuid4().hex}"
        self._validate_id(package_id, "API配置标识无效")
        existing = await self.repository.get_ai_api_package(package_id, profile_id=profile_id)
        version = self._expected_version(payload)
        if action in {"archive", "enable", "disable"}:
            result = await self._mutate_package(action, package_id, existing, version)
            await self.reload_backends()
            return {"ok": True, "package": jsonable(result)}
        if action != "save":
            raise ValueError("不支持的API配置操作")
        values = self._package_values(payload, existing)
        result = await self.repository.upsert_ai_api_package(
            package_id,
            profile_id=profile_id,
            credential_id=str((existing or {}).get("credential_id") or f"api-package-{package_id}"),
            expected_version=version,
            **values,
        )
        if existing is not None and self._package_cache_inputs_changed(existing, values):
            await self.repository.reset_ai_api_package_health(package_id)
        await self.reload_backends()
        return {"ok": True, "package_id": package_id, "package": jsonable(result)}

    async def _mutate_package(
        self, action: str, package_id: str, existing: Any, version: int | None
    ) -> Any:
        if existing is None:
            raise ValueError("找不到该API配置")
        if action == "archive":
            return await self.repository.archive_ai_api_package(
                package_id, expected_version=version
            )
        return await self.repository.set_ai_api_package_enabled(
            package_id, action == "enable", expected_version=version
        )

    @staticmethod
    def _package_values(payload: Mapping[str, Any], existing: Any) -> dict[str, Any]:
        protocol = (
            str(payload.get("protocol") or (existing or {}).get("protocol") or "OPENAI_COMPATIBLE")
            .strip()
            .upper()
        )
        if protocol not in _SUPPORTED_PROTOCOLS:
            raise ValueError("暂不支持这种API服务类型")
        defaults = {
            "OPENAI": "https://api.openai.com/v1",
            "ANTHROPIC": "https://api.anthropic.com/v1",
            "GEMINI": "https://generativelanguage.googleapis.com",
        }
        base_url = str(
            payload.get("base_url")
            or (existing or {}).get("base_url")
            or defaults.get(protocol, "")
        ).strip()
        require_secure_http_url(base_url, "API地址")
        config = AIConfigurationController._json_object(
            payload.get("config", (existing or {}).get("config", {})), "API高级设置必须是JSON对象"
        )
        isolation = payload.get(
            PROXY_CONTEXT_ISOLATION_CONFIG_KEY,
            config.get(PROXY_CONTEXT_ISOLATION_CONFIG_KEY, False),
        )
        if not isinstance(isolation, bool):
            raise ValueError("代理上下文隔离开关必须是布尔值")
        if protocol not in _PROXY_CONTEXT_ISOLATION_PROTOCOLS:
            isolation = False
        config[PROXY_CONTEXT_ISOLATION_CONFIG_KEY] = isolation
        return {
            "protocol": protocol,
            "display_name": str(
                payload.get("display_name")
                or (existing or {}).get("display_name")
                or "未命名API配置"
            ).strip(),
            "base_url": base_url,
            "enabled": bool(payload.get("enabled", (existing or {}).get("enabled", True))),
            "config": config,
        }

    async def save_credential(self, payload: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
        package_id = str(payload.get("package_id") or "").strip()
        package = await self.repository.get_ai_api_package(package_id, profile_id=profile_id)
        if package is None:
            raise ValueError("找不到该API配置")
        secret, env_name = (
            str(payload.get("secret") or ""),
            str(payload.get("env_name") or "").strip(),
        )
        if bool(secret) == bool(env_name):
            raise ValueError("请只填写密钥或环境变量名称中的一项")
        credential_id = str(package.get("credential_id") or "").strip()
        if not credential_id:
            raise RuntimeError("该API配置缺少密钥存储标识")
        info = (
            self.credential_vault.set_secret(credential_id, secret)
            if secret
            else self.credential_vault.set_env_reference(credential_id, env_name)
        )
        await self.repository.reset_ai_api_package_health(package_id)
        await self.reload_backends()
        return {
            "ok": True,
            "package_id": package_id,
            "credential_configured": info.configured,
            "credential_last4": info.last4,
            "credential_source": info.source,
        }

    async def save_model(self, payload: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
        package_id = str(payload.get("package_id") or "").strip()
        package = await self.repository.get_ai_api_package(package_id, profile_id=profile_id)
        if package is None:
            raise ValueError("找不到所属API配置")
        action = str(payload.get("action") or "save").strip().lower()
        backend_id = self._backend_id(payload)
        if not backend_id and action == "save":
            backend_id = f"api-model:{uuid.uuid4().hex}"
        self._validate_id(backend_id, "模型标识无效")
        existing = await self.repository.get_ai_api_model(backend_id)
        version = self._expected_version(payload)
        if action == "retry_prompt_cache":
            if existing is None or str(existing.get("package_id") or "") != package_id:
                raise ValueError("该模型不属于当前角色的API配置")
            if not await self.repository.request_ai_prompt_cache_reprobe(backend_id):
                raise ValueError("该模型当前没有可提前复探的缓存质量暂停")
            return {"ok": True, "backend_id": backend_id, "reprobe_armed": True}
        if action in {"archive", "enable", "disable"}:
            result = await self._mutate_model(action, backend_id, version)
            await self.reload_backends()
            return {"ok": True, "model": jsonable(result)}
        if action != "save":
            raise ValueError("不支持的模型操作")
        values = self._model_values(payload, package, existing)
        result = await self.repository.upsert_ai_api_model(
            package_id, backend_id, expected_version=version, **values
        )
        await self.repository.invalidate_ai_prompt_cache_capabilities([backend_id])
        await self.reload_backends()
        return {"ok": True, "backend_id": backend_id, "model": jsonable(result)}

    @staticmethod
    def _package_cache_inputs_changed(
        existing: Mapping[str, Any], values: Mapping[str, Any]
    ) -> bool:
        return any(
            existing.get(key) != values.get(key) for key in ("protocol", "base_url", "config")
        )

    async def _mutate_model(self, action: str, backend_id: str, version: int | None) -> Any:
        if action == "archive":
            return await self.repository.archive_ai_api_model(backend_id, expected_version=version)
        return await self.repository.set_ai_api_model_enabled(
            backend_id, action == "enable", expected_version=version
        )

    @staticmethod
    def _model_values(
        payload: Mapping[str, Any], package: Mapping[str, Any], existing: Any
    ) -> dict[str, Any]:
        current = existing or {}
        model_key = AIConfigurationController._model_key(payload, current)
        if not model_key:
            raise ValueError("请填写模型名称")
        capabilities = AIConfigurationController._capabilities(payload, existing)
        AIConfigurationController._validate_model_capabilities(
            str(package.get("protocol") or "").upper(), capabilities
        )
        config = AIConfigurationController._model_configuration(
            payload, package, capabilities, current
        )
        priority = AIConfigurationController._model_priority(payload, current)
        return {
            "model_key": model_key,
            "display_name": str(
                payload.get("display_name") or current.get("display_name") or model_key
            ).strip(),
            "capabilities": capabilities,
            "priority": max(1, min(999, priority)),
            "enabled": bool(payload.get("enabled", current.get("enabled", True))),
            "config": config,
        }

    @staticmethod
    def _model_configuration(
        payload: Mapping[str, Any],
        package: Mapping[str, Any],
        capabilities: list[str],
        current: Mapping[str, Any],
    ) -> dict[str, Any]:
        config = AIConfigurationController._json_object(
            payload.get("config", current.get("config", {})), "模型高级设置必须是JSON对象"
        )
        config["soulcore_configured"] = True
        audio_only = AIConfigurationController._audio_only(capabilities)
        text_capable = bool(set(capabilities).intersection(TEXT_GENERATION_CAPABILITIES))
        if audio_only:
            for key in (
                "supports_vision",
                "generation_parameters",
                "custom_request_parameters",
                "max_context_tokens",
                "image_generation_mode",
            ):
                config.pop(key, None)
        else:
            config["supports_vision"] = AIConfigurationController._model_supports_vision(
                payload, config, capabilities
            )
            config["generation_parameters"] = AIConfigurationController._generation_parameters(
                payload, package, config, capabilities
            )
            config["custom_request_parameters"] = (
                AIConfigurationController._custom_request_parameters(
                    payload,
                    package,
                    config,
                    capabilities,
                )
            )
            if text_capable:
                config["max_context_tokens"] = AIConfigurationController._model_context_limit(
                    payload, config
                )
            else:
                config.pop("max_context_tokens", None)
            config["image_generation_mode"] = AIConfigurationController._image_generation_mode(
                payload, config
            )
        AIConfigurationController._apply_audio_settings(
            payload,
            config,
            capabilities,
            str(package.get("protocol") or "").upper(),
        )
        return config

    _audio_only = staticmethod(ai_audio_configuration.audio_only)
    _audio_settings_view = staticmethod(ai_audio_configuration.audio_settings_view)
    _apply_audio_settings = staticmethod(ai_audio_configuration.apply_audio_settings)

    @staticmethod
    def _model_supports_vision(
        payload: Mapping[str, Any], config: Mapping[str, Any], capabilities: list[str]
    ) -> bool:
        if "supports_vision" in payload:
            supports_vision = payload["supports_vision"]
        elif "supports_vision" in config:
            supports_vision = config["supports_vision"]
        else:
            raise ValueError("模型必须明确声明是否支持图片输入")
        if not isinstance(supports_vision, bool):
            raise ValueError("模型视觉能力设置无效")
        if "vision.describe" in capabilities and not supports_vision:
            raise ValueError("承担图片识别任务的模型必须支持图片输入")
        return supports_vision

    @staticmethod
    def _generation_parameters(
        payload: Mapping[str, Any],
        package: Mapping[str, Any],
        config: Mapping[str, Any],
        capabilities: list[str],
    ) -> dict[str, Any]:
        parameters = normalize_model_generation_parameters(
            payload.get(
                "generation_parameters",
                config.get("generation_parameters"),
            )
        )
        if parameters and not set(capabilities).intersection(TEXT_GENERATION_CAPABILITIES):
            raise ValueError("只有承担文字任务的模型才能配置生成参数")
        if parameters and str(package.get("protocol") or "").upper() not in {
            "OPENAI",
            "OPENAI_COMPATIBLE",
            "ANTHROPIC",
        }:
            raise ValueError("当前 API 服务类型不支持这些文字生成参数")
        if str(package.get("protocol") or "").upper() == "ANTHROPIC":
            if "max_completion_tokens" in parameters:
                raise ValueError("Anthropic 原生协议只接受 max_tokens")
            if str(parameters.get("reasoning_effort") or "").lower() == "minimal":
                raise ValueError("Anthropic 原生协议不支持 minimal effort")
        return parameters

    @staticmethod
    def _custom_request_parameters(
        payload: Mapping[str, Any],
        package: Mapping[str, Any],
        config: Mapping[str, Any],
        capabilities: list[str],
    ) -> dict[str, Any]:
        parameters = normalize_model_custom_request_parameters(
            payload.get(
                "custom_request_parameters",
                config.get("custom_request_parameters"),
            )
        )
        if parameters and not set(capabilities).intersection(TEXT_GENERATION_CAPABILITIES):
            raise ValueError("只有承担文字任务的模型才能配置高级请求参数")
        if parameters and str(package.get("protocol") or "").upper() not in {
            "OPENAI",
            "OPENAI_COMPATIBLE",
            "ANTHROPIC",
        }:
            raise ValueError("当前 API 服务类型不支持高级文字请求参数")
        overlap = set(parameters).intersection(
            normalize_model_generation_parameters(config.get("generation_parameters"))
        )
        if overlap:
            raise ValueError(
                "以下参数已在常用生成参数中配置，请勿重复：" + "、".join(sorted(overlap))
            )
        return parameters

    @staticmethod
    def _model_context_limit(payload: Mapping[str, Any], config: Mapping[str, Any]) -> int:
        raw_limit = payload.get("max_context_tokens")
        if raw_limit in (None, ""):
            raw_limit = config.get("max_context_tokens")
        if raw_limit in (None, ""):
            raw_limit = DEFAULT_MODEL_MAX_CONTEXT_TOKENS
        raw_limit_text = str(raw_limit).strip()
        if not raw_limit_text.isascii() or not raw_limit_text.isdigit():
            raise ValueError("模型最大 Token 必须是正整数")
        max_context_tokens = int(raw_limit_text)
        if max_context_tokens < MINIMUM_MODEL_MAX_CONTEXT_TOKENS:
            raise ValueError("文字与视觉模型的最大 Token 不能低于 128000")
        return min(max_context_tokens, 10_000_000)

    @staticmethod
    def _image_generation_mode(payload: Mapping[str, Any], config: Mapping[str, Any]) -> str:
        mode = (
            str(
                payload.get("image_generation_mode")
                or config.get("image_generation_mode")
                or "IMAGES_API"
            )
            .strip()
            .upper()
        )
        if mode not in {"IMAGES_API", "CHAT_RESPONSE"}:
            raise ValueError("图片生成方式无效")
        return mode

    @staticmethod
    def _model_key(payload: Mapping[str, Any], current: Mapping[str, Any]) -> str:
        return str(payload.get("model_key") or current.get("model_key") or "").strip()

    @staticmethod
    def _model_priority(payload: Mapping[str, Any], current: Mapping[str, Any]) -> int:
        return int(payload.get("priority") or current.get("priority") or 1)

    @staticmethod
    def _capabilities(payload: Mapping[str, Any], existing: Any) -> list[str]:
        raw = payload.get("capabilities")
        raw = (existing or {}).get("capabilities", []) if raw is None else raw
        values = _normalize_model_capabilities(raw, reject_unknown=True)
        if not values:
            raise ValueError("请至少选择一项当前支持的模型用途")
        return values

    @staticmethod
    def _validate_model_capabilities(protocol: str, capabilities: list[str]) -> None:
        selected = set(capabilities)
        if protocol in {"GEMINI", "CUSTOM_HTTP_IMAGE"} and selected != {"image.generate"}:
            raise ValueError("该服务类型当前只支持图片生成")
        if protocol == "ANTHROPIC" and selected.intersection(
            {"image.generate", *_AUDIO_CAPABILITIES}
        ):
            raise ValueError("Anthropic 原生协议不提供图片生成或音频功能")
        if protocol in _AUDIO_SPEECH_PROTOCOLS and selected != {"audio.speech"}:
            raise ValueError("该语音服务当前只支持语音合成")
        if protocol not in _SUPPORTED_PROTOCOLS:
            raise ValueError("该API配置当前不能添加自定义模型")

    @staticmethod
    def _json_object(value: Any, message: str) -> dict[str, Any]:
        if isinstance(value, str):
            value = json.loads(value or "{}")
        if not isinstance(value, Mapping):
            raise ValueError(message)
        return dict(value)

    @staticmethod
    def _supports_vision(config: Mapping[str, Any]) -> bool:
        value = config["supports_vision"]
        if not isinstance(value, bool):
            raise ValueError("supports_vision must be a boolean")
        return value

    @staticmethod
    def _backend_id(payload: Mapping[str, Any]) -> str:
        return str(payload.get("backend_id") or "").strip()

    @staticmethod
    def _expected_version(payload: Mapping[str, Any]) -> int | None:
        value = payload.get("expected_version")
        return int(value) if value not in (None, "") else None

    @staticmethod
    def _validate_id(value: str, message: str) -> None:
        if not value or any(char in value for char in "\0\r\n"):
            raise ValueError(message)
