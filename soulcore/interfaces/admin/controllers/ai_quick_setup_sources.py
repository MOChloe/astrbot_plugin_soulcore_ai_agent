"""Candidate discovery and source ownership for guided AI setup."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from ....features.ai.model_parameters import (
    DEFAULT_MODEL_MAX_CONTEXT_TOKENS,
    MINIMUM_MODEL_MAX_CONTEXT_TOKENS,
)
from .ai_quick_setup_shared import (
    IMAGE_PROTOCOLS,
    TEXT_PROTOCOLS,
    ConfigurationPort,
    RepositoryPort,
)

_OPENAI_ADAPTERS = frozenset(
    {
        "openai_chat_completion",
        "longcat_chat_completion",
        "xiaomi_chat_completion",
        "zhipu_chat_completion",
        "groq_chat_completion",
        "xai_chat_completion",
        "aihubmix_chat_completion",
        "openrouter_chat_completion",
    }
)
_ANTHROPIC_ADAPTERS = frozenset(
    {
        "anthropic_chat_completion",
        "kimi_code_chat_completion",
        "minimax_token_plan",
        "xiaomi_token_plan",
    }
)


class AIQuickSetupSourceMixin:
    repository: RepositoryPort
    configuration: ConfigurationPort
    runtime_context: Any | None

    async def _resolve_candidate(
        self,
        profile_id: str,
        slot: str,
        payload: Mapping[str, Any],
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        selection = payload.get("selection")
        if not isinstance(selection, Mapping):
            raise ValueError("请选择一个模型")
        kind = str(selection.get("kind") or "").strip().lower()
        if kind == "existing":
            return await self._existing_candidate(profile_id, slot, selection)
        if kind == "astrbot":
            return await self._astrbot_candidate(profile_id, slot, selection, payload, snapshot)
        if kind == "manual":
            return await self._manual_candidate(profile_id, slot, selection, snapshot)
        raise ValueError("模型来源无效")

    async def _existing_candidate(
        self, profile_id: str, slot: str, selection: Mapping[str, Any]
    ) -> dict[str, Any]:
        backend_id = str(selection.get("backend_id") or "").strip()
        model = await self.repository.get_ai_api_model(backend_id)
        if model is None:
            raise ValueError("选择的模型已经不存在")
        package = await self.repository.get_ai_api_package(
            str(model["package_id"]), profile_id=profile_id
        )
        if package is None:
            raise ValueError("选择的模型不属于当前角色")
        self._require_protocol_for_slot(slot, str(package.get("protocol") or ""))
        config = dict(model.get("config") or {})
        max_context_tokens = int(
            config.get("max_context_tokens") or DEFAULT_MODEL_MAX_CONTEXT_TOKENS
        )
        if slot != "image" and max_context_tokens < MINIMUM_MODEL_MAX_CONTEXT_TOKENS:
            raise ValueError("这个模型的上下文窗口低于 SoulCore 要求的 128K")
        return {
            "backend_id": backend_id,
            "package_id": str(package["package_id"]),
            "model_key": str(model["model_key"]),
            "display_name": str(model.get("display_name") or model["model_key"]),
            "supports_vision": bool(config.get("supports_vision")),
            "max_context_tokens": max_context_tokens,
            "image_generation_mode": str(config.get("image_generation_mode") or "IMAGES_API"),
            "package": package,
            "redaction_values": self._package_redaction_values(package),
        }

    async def _astrbot_candidate(
        self,
        profile_id: str,
        slot: str,
        selection: Mapping[str, Any],
        payload: Mapping[str, Any],
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        source_ref = str(selection.get("source_ref") or "").strip()
        source = next(
            (
                item
                for item in self._astrbot_sources(include_private=True)
                if item["source_ref"] == source_ref
            ),
            None,
        )
        if source is None:
            raise ValueError("这个 AstrBot 模型当前不能无损导入")
        if not source.get("compatible"):
            reason = str(source.get("unavailable_reason") or "").strip()
            raise ValueError(reason or "这个 AstrBot 模型当前不能无损导入")
        self._require_source_usage(slot, source)
        secret = await self._astrbot_secret(str(source["provider_id"]))
        if not secret:
            raise ValueError("AstrBot 模型当前没有可复制的有效密钥")
        protocol = str(source["protocol"])
        base_url = str(source["base_url"])
        model_key = str(source["model_key"])
        matched = self._semantic_match(snapshot, protocol, base_url, model_key)
        package_id = (
            str(matched["package_id"])
            if matched
            else self._stable_id("quick-astrbot", profile_id, str(source["provider_id"]))
        )
        package_before = await self.repository.get_ai_api_package(package_id, profile_id=profile_id)
        credential_before = self._credential_state(package_before)
        rollback = {
            "package_id": package_id,
            "package_before": package_before,
            "credential_before": credential_before,
        }
        try:
            package = await self._save_package_and_secret(
                profile_id,
                package_id,
                protocol,
                str(source["display_name"]),
                base_url,
                secret,
            )
        except Exception:
            await self._restore_package_and_credential(profile_id, rollback)
            raise
        backend_id = (
            str(matched["backend_id"])
            if matched
            else self._stable_id("quick-model", package_id, model_key)
        )
        return {
            "backend_id": backend_id,
            "package_id": package_id,
            "model_key": model_key,
            "display_name": str(source["display_name"]),
            "supports_vision": bool(source.get("supports_vision")),
            "max_context_tokens": int(
                source.get("max_context_tokens") or DEFAULT_MODEL_MAX_CONTEXT_TOKENS
            ),
            "image_generation_mode": str(
                payload.get("image_generation_mode")
                or source.get("image_generation_mode")
                or "IMAGES_API"
            ),
            "package": package,
            "package_before": package_before,
            "credential_before": credential_before,
            "redaction_values": [secret],
        }

    async def _manual_candidate(
        self,
        profile_id: str,
        slot: str,
        selection: Mapping[str, Any],
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        values = self._manual_candidate_values(slot, selection)
        protocol = values["protocol"]
        base_url = values["base_url"]
        model_key = values["model_key"]
        secret = values["secret"]
        matched = self._semantic_match(snapshot, protocol, base_url, model_key)
        package_id = (
            str(matched["package_id"])
            if matched
            else self._stable_id("quick-manual", profile_id, protocol, base_url.casefold())
        )
        package_before = await self.repository.get_ai_api_package(package_id, profile_id=profile_id)
        credential_before = self._credential_state(package_before)
        rollback = {
            "package_id": package_id,
            "package_before": package_before,
            "credential_before": credential_before,
        }
        try:
            package = await self._save_package_and_secret(
                profile_id,
                package_id,
                protocol,
                values["display_name"],
                base_url,
                secret,
            )
        except Exception:
            await self._restore_package_and_credential(profile_id, rollback)
            raise
        backend_id = (
            str(matched["backend_id"])
            if matched
            else self._stable_id("quick-model", package_id, model_key)
        )
        candidate = {
            **values,
            "backend_id": backend_id,
            "package_id": package_id,
            "package": package,
            "package_before": package_before,
            "credential_before": credential_before,
            "redaction_values": [secret],
        }
        candidate.pop("protocol", None)
        candidate.pop("base_url", None)
        candidate.pop("secret", None)
        generation_parameters = candidate.pop("generation_parameters")
        if slot != "image":
            candidate["generation_parameters"] = generation_parameters
        return candidate

    def _manual_candidate_values(self, slot: str, selection: Mapping[str, Any]) -> dict[str, Any]:
        protocol = str(selection.get("protocol") or "OPENAI_COMPATIBLE").strip().upper()
        self._require_protocol_for_slot(slot, protocol)
        base_url = str(selection.get("base_url") or "").strip().rstrip("/")
        model_key = str(selection.get("model_key") or "").strip()
        secret = str(selection.get("secret") or "")
        if not base_url or not model_key or not secret:
            raise ValueError("请填写 API 地址、模型名称和密钥")
        reasoning_effort = str(selection.get("reasoning_effort") or "").strip().lower()
        return {
            "protocol": protocol,
            "base_url": base_url,
            "model_key": model_key,
            "secret": secret,
            "display_name": str(selection.get("display_name") or model_key).strip(),
            "supports_vision": bool(selection.get("supports_vision")),
            "max_context_tokens": _manual_context_limit(slot, selection),
            "image_generation_mode": str(selection.get("image_generation_mode") or "IMAGES_API"),
            "generation_parameters": (
                {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
            ),
        }

    async def _save_package_and_secret(
        self,
        profile_id: str,
        package_id: str,
        protocol: str,
        display_name: str,
        base_url: str,
        secret: str,
    ) -> Mapping[str, Any]:
        existing = await self.repository.get_ai_api_package(package_id, profile_id=profile_id)
        saved = await self.configuration.save_package(
            {
                "package_id": package_id,
                "expected_version": (existing or {}).get("version"),
                "protocol": protocol,
                "display_name": display_name,
                "base_url": base_url,
                "enabled": True,
            },
            profile_id,
        )
        await self.configuration.save_credential(
            {"package_id": package_id, "secret": secret}, profile_id
        )
        return dict(saved.get("package") or {})

    async def _restore_package_and_credential(
        self, profile_id: str, candidate: Mapping[str, Any]
    ) -> None:
        if "package_before" not in candidate and "credential_before" not in candidate:
            return
        before = candidate.get("package_before")
        package_id = str(candidate.get("package_id") or "")
        current = await self.repository.get_ai_api_package(package_id, profile_id=profile_id)
        await self._restore_quick_setup_package(profile_id, package_id, current, before)
        self._restore_quick_setup_credential(candidate, current, before)

    async def _restore_quick_setup_package(
        self,
        profile_id: str,
        package_id: str,
        current: Mapping[str, Any] | None,
        before: Any,
    ) -> None:
        if current is None:
            return
        if isinstance(before, Mapping):
            payload = {
                "package_id": package_id,
                "expected_version": current.get("version"),
                "protocol": str(before.get("protocol") or "OPENAI_COMPATIBLE"),
                "display_name": str(before.get("display_name") or "API 配置"),
                "base_url": str(before.get("base_url") or ""),
                "enabled": bool(before.get("enabled", True)),
                "config": dict(before.get("config") or {}),
            }
        else:
            payload = {
                "action": "disable",
                "package_id": package_id,
                "expected_version": current.get("version"),
            }
        await self.configuration.save_package(payload, profile_id)

    def _restore_quick_setup_credential(
        self,
        candidate: Mapping[str, Any],
        current: Mapping[str, Any] | None,
        before: Any,
    ) -> None:
        state = candidate.get("credential_before")
        if not isinstance(state, Mapping):
            return
        vault = getattr(self.configuration, "credential_vault", None)
        if vault is None:
            return
        credential_id = str(
            (before or {}).get("credential_id")
            if isinstance(before, Mapping)
            else (current or {}).get("credential_id") or ""
        )
        if not credential_id:
            return
        source = str(state.get("source") or "missing")
        if source == "file":
            vault.set_secret(credential_id, str(state["secret"]))
        elif source == "env":
            vault.set_env_reference(credential_id, str(state["reference"]))
        else:
            vault.delete(credential_id)

    def _astrbot_sources(self, *, include_private: bool = False) -> list[dict[str, Any]]:
        manager = getattr(self.runtime_context, "provider_manager", None)
        configs = getattr(manager, "providers_config", None)
        merge = getattr(manager, "get_merged_provider_config", None)
        if not isinstance(configs, list) or not callable(merge):
            return []
        result = []
        for raw in configs:
            if not isinstance(raw, Mapping):
                continue
            config = dict(merge(dict(raw)))
            if not bool(config.get("enable", True)):
                continue
            source = self._astrbot_source(config, include_private=include_private)
            if source is not None:
                result.append(source)
        return result

    def _credential_state(self, package: Mapping[str, Any] | None) -> dict[str, str]:
        if package is None:
            return {"source": "missing"}
        vault = getattr(self.configuration, "credential_vault", None)
        credential_id = str(package.get("credential_id") or "")
        if vault is None or not credential_id:
            return {"source": "missing"}
        info = vault.describe(credential_id)
        if info.source == "env":
            return {"source": "env", "reference": str(info.reference)}
        if info.configured:
            return {"source": "file", "secret": str(vault.resolve(credential_id))}
        return {"source": "missing"}

    def _package_redaction_values(self, package: Mapping[str, Any]) -> list[str]:
        vault = getattr(self.configuration, "credential_vault", None)
        credential_id = str(package.get("credential_id") or "")
        if vault is None or not credential_id:
            return []
        try:
            secret = str(vault.resolve(credential_id) or "")
        except Exception:
            return []
        return [secret] if secret else []

    def _astrbot_source(
        self, config: Mapping[str, Any], *, include_private: bool
    ) -> dict[str, Any] | None:
        if str(config.get("provider_type") or "") != "chat_completion":
            return None
        provider_id = str(config.get("id") or "").strip()
        model_key = str(config.get("model") or "").strip()
        base_url = str(config.get("api_base") or "").strip().rstrip("/")
        protocol = _astrbot_protocol(str(config.get("type") or "").strip(), base_url)
        max_context_tokens = _astrbot_context_limit(config)
        unavailable = _astrbot_unavailable_reason(
            config,
            provider_id=provider_id,
            model_key=model_key,
            base_url=base_url,
            protocol=protocol,
            max_context_tokens=max_context_tokens,
        )
        modalities = [str(item).lower() for item in config.get("modalities") or ()]
        supports_vision = protocol in TEXT_PROTOCOLS and (not modalities or "image" in modalities)
        source = {
            "source_ref": self._stable_id("astrbot-source", provider_id),
            "display_name": str(config.get("id") or model_key),
            "model_key": model_key,
            "protocol": protocol or "UNSUPPORTED",
            "endpoint_label": self._endpoint_label(base_url),
            "supports_vision": supports_vision,
            "max_context_tokens": max_context_tokens,
            "reasoning": bool(config.get("reasoning")),
            "usages": _astrbot_usages(protocol, supports_vision),
            "compatible": not unavailable,
            "unavailable_reason": unavailable,
        }
        if include_private:
            source.update(
                {
                    "provider_id": provider_id,
                    "base_url": base_url,
                    "image_generation_mode": "IMAGES_API",
                }
            )
        return source

    async def _astrbot_secret(self, provider_id: str) -> str:
        manager = getattr(self.runtime_context, "provider_manager", None)
        resolver = getattr(manager, "get_provider_by_id", None)
        if not callable(resolver):
            return ""
        provider = await resolver(provider_id)
        current_key = getattr(provider, "get_current_key", None)
        return str(current_key() or "") if callable(current_key) else ""

    @staticmethod
    def _semantic_match(
        snapshot: Mapping[str, Any], protocol: str, base_url: str, model_key: str
    ) -> dict[str, Any] | None:
        wanted_url = base_url.rstrip("/").casefold()
        for package in snapshot.get("api_packages") or ():
            if str(package.get("protocol") or "").upper() != protocol.upper():
                continue
            if str(package.get("base_url") or "").rstrip("/").casefold() != wanted_url:
                continue
            for model in package.get("models") or ():
                if str(model.get("model_key") or "") == model_key:
                    return {
                        "package_id": str(package.get("package_id") or ""),
                        "backend_id": str(model.get("backend_id") or ""),
                    }
        return None

    @staticmethod
    def _require_protocol_for_slot(slot: str, protocol: str) -> None:
        normalized = protocol.strip().upper()
        allowed = IMAGE_PROTOCOLS if slot == "image" else TEXT_PROTOCOLS
        if normalized not in allowed:
            raise ValueError("这个模型协议不适合当前步骤")

    @staticmethod
    def _require_source_usage(slot: str, source: Mapping[str, Any]) -> None:
        wanted = "image" if slot == "image" else "vision" if slot == "vision" else "text"
        if wanted not in set(source.get("usages") or ()):
            raise ValueError("这个 AstrBot 模型不适合当前步骤")

    @staticmethod
    def _stable_id(prefix: str, *parts: str) -> str:
        digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:32]
        return f"{prefix}:{digest}"

    @staticmethod
    def _endpoint_label(base_url: str) -> str:
        parsed = urlsplit(base_url)
        hostname = str(parsed.hostname or "")
        if not hostname:
            return "自定义地址"
        try:
            port = parsed.port
        except ValueError:
            port = None
        return f"{hostname}:{port}" if port is not None else hostname


def _manual_context_limit(slot: str, selection: Mapping[str, Any]) -> int:
    if slot == "image":
        return DEFAULT_MODEL_MAX_CONTEXT_TOKENS
    raw = selection.get("max_context_tokens")
    raw = DEFAULT_MODEL_MAX_CONTEXT_TOKENS if raw in (None, "") else raw
    text = str(raw).strip()
    if not text.isascii() or not text.isdigit():
        raise ValueError("模型最大 Token 必须是正整数")
    value = int(text)
    if value < MINIMUM_MODEL_MAX_CONTEXT_TOKENS:
        raise ValueError("文字与视觉模型的最大 Token 不能低于 128000")
    return min(value, 10_000_000)


def _astrbot_protocol(adapter: str, base_url: str) -> str:
    if adapter == "openai_chat_completion" and base_url == "https://api.openai.com/v1":
        return "OPENAI"
    if adapter in _OPENAI_ADAPTERS:
        return "OPENAI_COMPATIBLE"
    if adapter in _ANTHROPIC_ADAPTERS:
        return "ANTHROPIC"
    return "GEMINI" if adapter == "googlegenai_chat_completion" else ""


def _astrbot_context_limit(config: Mapping[str, Any]) -> int:
    try:
        return max(
            1,
            min(
                10_000_000,
                int(config.get("max_context_tokens") or DEFAULT_MODEL_MAX_CONTEXT_TOKENS),
            ),
        )
    except (TypeError, ValueError):
        return DEFAULT_MODEL_MAX_CONTEXT_TOKENS


def _astrbot_unavailable_reason(
    config: Mapping[str, Any],
    *,
    provider_id: str,
    model_key: str,
    base_url: str,
    protocol: str,
    max_context_tokens: int,
) -> str:
    if not provider_id or not model_key or not base_url:
        return "缺少模型名称或 API 地址"
    if not protocol:
        return "该 AstrBot 适配器需要专用协议"
    if config.get("proxy") or config.get("custom_headers"):
        return "当前配置依赖代理或自定义请求头"
    if protocol in TEXT_PROTOCOLS and max_context_tokens < MINIMUM_MODEL_MAX_CONTEXT_TOKENS:
        return "模型上下文窗口低于 SoulCore 要求的 128K"
    return ""


def _astrbot_usages(protocol: str, supports_vision: bool) -> list[str]:
    usages = ["image"] if protocol == "GEMINI" else ["text"] if protocol else []
    if supports_vision:
        usages.append("vision")
    if protocol in {"OPENAI", "OPENAI_COMPATIBLE"}:
        usages.append("image")
    return usages


__all__ = ["AIQuickSetupSourceMixin"]
