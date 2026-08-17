"""Load persisted AI package configuration into runtime adapter registries."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from ..contracts.ai_models import (
    AIBackendDescriptor,
    AIImageBackendCapabilities,
)
from ..features.ai.anthropic_messages import (
    AnthropicMessagesAdapter,
    AnthropicMessagesConfig,
    AnthropicVisionDescribeAdapter,
)
from ..features.ai.image_capabilities import (
    CustomHTTPImageCapabilityAdapter,
    CustomHTTPImageConfig,
    GeminiImageCapabilityAdapter,
    GeminiImageConfig,
    OpenAIChatImageCapabilityAdapter,
    OpenAIChatImageConfig,
    OpenAIImagesCapabilityAdapter,
    OpenAIImagesConfig,
    OpenAIVisionConfig,
    OpenAIVisionDescribeAdapter,
)
from ..features.ai.model_parameters import (
    DEFAULT_MODEL_MAX_CONTEXT_TOKENS,
    MINIMUM_MODEL_MAX_CONTEXT_TOKENS,
    TEXT_GENERATION_CAPABILITIES,
    normalize_model_custom_request_parameters,
    normalize_model_generation_parameters,
)
from ..features.ai.openai_compatible import (
    OpenAICompatibleAdapter,
    OpenAICompatibleConfig,
)
from ..features.ai.prompt_cache import prompt_cache_config_fingerprint
from ..features.ai.proxy_context_isolation import (
    PROXY_CONTEXT_ISOLATION_CONFIG_KEY,
    proxy_context_isolation_enabled,
)
from ..features.ai.service import BackendPool, CapabilityAdapterRegistry, CircuitBreaker
from ..features.web.domain import WebSearchProviderRecord
from ..features.web.providers import build_web_search_adapter
from .audio_runtime import (
    AUDIO_CAPABILITY_NAMES,
    AudioRuntimeAdapterFactory,
    AudioRuntimeModel,
)


class AIRuntimeRepositoryPort(Protocol):
    async def get_ai_backend(self, backend_id: str) -> Mapping[str, Any] | None: ...
    async def list_ai_api_models(self, **values: object) -> Sequence[Mapping[str, Any]]: ...
    async def list_ai_api_packages(self, **values: object) -> Sequence[Mapping[str, Any]]: ...
    async def list_ai_backends(self) -> Sequence[Mapping[str, Any]]: ...
    async def list_ai_circuit_states(self) -> Sequence[Mapping[str, Any]]: ...
    async def upsert_ai_api_model(self, *values: object, **named: object) -> object: ...
    async def upsert_ai_api_package(self, *values: object, **named: object) -> object: ...
    async def upsert_ai_backend(self, *values: object, **named: object) -> object: ...


class ProfilesRuntimeRepositoryPort(Protocol):
    async def list_profiles(self, **values: object) -> Sequence[Any]: ...


class WebRuntimeRepositoryPort(Protocol):
    async def list_web_search_providers(
        self, profile_id: str
    ) -> Sequence[WebSearchProviderRecord]: ...


@dataclass(frozen=True, slots=True)
class _ModelRegistration:
    backend_id: str
    model: Mapping[str, Any]
    package: Mapping[str, Any]
    metadata: Mapping[str, Any]
    base_url: str
    model_key: str
    credential_id: str
    capabilities: set[str]
    package_kind: str
    priority: int
    descriptor_metadata: Mapping[str, Any]


class AIRuntimeLoader:
    def __init__(
        self,
        *,
        ai_repository: AIRuntimeRepositoryPort,
        profiles_repository: ProfilesRuntimeRepositoryPort,
        web_repository: WebRuntimeRepositoryPort,
        manager: Any,
        credential_vault: Any,
    ) -> None:
        self.repository = ai_repository
        self.profiles = profiles_repository
        self.web = web_repository
        self.manager = manager
        self.credential_vault = credential_vault
        self._reload_lock = asyncio.Lock()

    async def reload(self) -> None:
        async with self._reload_lock:
            # Build a complete snapshot off to the side. Configuration writes
            # commit before this reload, so publishing a partial registry would
            # allow an already-disabled backend to survive in only one runtime
            # pool when a later read or adapter construction fails.
            circuits = CircuitBreaker(self.manager.circuits.policy)
            capabilities = CapabilityAdapterRegistry()
            backends = BackendPool()
            metadata = await self._register_packages(backends, capabilities)
            await self._register_web_providers(capabilities)
            await self._restore_circuit_scopes(circuits)

            # There must be no await between these assignments. One event-loop
            # turn observes either the previous complete snapshot or this one.
            self.manager.circuits = circuits
            self.manager.capabilities = capabilities
            self.manager.backends = backends
            self.manager.runtime_backend_metadata = metadata
            self.manager.runtime_profile_backend_metadata = {}

    async def _register_packages(
        self,
        pool: BackendPool,
        capability_registry: CapabilityAdapterRegistry,
    ) -> dict[str, Mapping[str, Any]]:
        packages = {
            str(item.get("package_id") or ""): item
            for item in await self.repository.list_ai_api_packages()
        }
        metadata: dict[str, Mapping[str, Any]] = {}
        for model in await self.repository.list_ai_api_models():
            backend_id = str(model.get("backend_id") or "").strip()
            package = packages.get(str(model.get("package_id") or ""))
            if not backend_id or package is None:
                continue
            item_metadata = self._runtime_metadata(model, package)
            metadata[backend_id] = item_metadata
            if self._model_enabled(model, package, item_metadata):
                self._register_model(
                    pool,
                    capability_registry,
                    backend_id,
                    model,
                    package,
                    item_metadata,
                )
        return metadata

    @staticmethod
    def _supports_text_backend(capabilities: set[str], kind: str) -> bool:
        return bool(capabilities.intersection(TEXT_GENERATION_CAPABILITIES)) and kind in {
            "openai",
            "openai_compatible",
            "anthropic",
        }

    def _register_model(
        self,
        pool: BackendPool,
        capability_registry: CapabilityAdapterRegistry,
        backend_id: str,
        model: Mapping[str, Any],
        package: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> None:
        item = self._model_registration(backend_id, model, package, metadata)
        if item is None:
            return
        self._register_text_backend(pool, item)
        self._register_vision_backend(capability_registry, item)
        self._register_image_backend(capability_registry, item)
        self._register_audio_backend(capability_registry, item)

    @staticmethod
    def _model_registration(
        backend_id: str,
        model: Mapping[str, Any],
        package: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> _ModelRegistration | None:
        base_url = str(package.get("base_url") or "").strip()
        model_key = str(model.get("model_key") or "").strip()
        credential_id = str(package.get("credential_id") or "").strip()
        capabilities = {
            str(item).strip().lower()
            for item in model.get("capabilities") or ()
            if str(item).strip()
        }
        kind, priority = str(metadata["package_kind"]), int(metadata["priority"])
        credential_optional = kind in {"gpt_sovits_v2", "gsvi_tts"}
        if not base_url or not model_key or (not credential_id and not credential_optional):
            return None
        descriptor_metadata = {**metadata, "base_url": base_url, "model": model_key}
        return _ModelRegistration(
            backend_id=backend_id,
            model=model,
            package=package,
            metadata=metadata,
            base_url=base_url,
            model_key=model_key,
            credential_id=credential_id,
            capabilities=capabilities,
            package_kind=kind,
            priority=priority,
            descriptor_metadata=descriptor_metadata,
        )

    def _register_text_backend(self, pool: BackendPool, item: _ModelRegistration) -> None:
        if not self._supports_text_backend(item.capabilities, item.package_kind):
            return
        adapter = self._text_adapter(
            item.package_kind,
            item.backend_id,
            item.base_url,
            item.credential_id,
            item.model_key,
        )
        pool.register(
            AIBackendDescriptor(
                item.backend_id,
                adapter.adapter_id,
                model=item.model_key,
                priority=item.priority,
                credential_id=item.credential_id,
                metadata=item.descriptor_metadata,
            ),
            adapter,
        )

    def _register_vision_backend(
        self,
        registry: CapabilityAdapterRegistry,
        item: _ModelRegistration,
    ) -> None:
        supported_kinds = {"openai", "openai_compatible", "anthropic"}
        if (
            "vision.describe" not in item.capabilities
            or not bool(item.metadata["supports_vision"])
            or item.package_kind not in supported_kinds
        ):
            return
        adapter = self._vision_adapter(
            item.package_kind,
            item.backend_id,
            item.base_url,
            item.credential_id,
            item.model_key,
        )
        self._register_capability_adapter(
            registry,
            item,
            adapter,
            capabilities=("vision.describe",),
            modalities=("image",),
        )

    def _register_image_backend(
        self,
        registry: CapabilityAdapterRegistry,
        item: _ModelRegistration,
    ) -> None:
        if "image.generate" not in item.capabilities:
            return
        adapter = self._image_adapter(
            item.package_kind,
            item.model,
            item.package,
            item.base_url,
            item.credential_id,
            item.model_key,
        )
        if adapter is None:
            return
        self._register_capability_adapter(
            registry,
            item,
            adapter,
            capabilities=("image.generate",),
        )

    def _register_audio_backend(
        self,
        registry: CapabilityAdapterRegistry,
        item: _ModelRegistration,
    ) -> None:
        audio_capabilities = item.capabilities.intersection(AUDIO_CAPABILITY_NAMES)
        if not audio_capabilities:
            return
        adapter = AudioRuntimeAdapterFactory(self.credential_vault.resolve).build(
            AudioRuntimeModel.from_records(
                package_kind=item.package_kind,
                model=item.model,
                package=item.package,
                base_url=item.base_url,
                credential_id=item.credential_id,
                model_key=item.model_key,
                capabilities=audio_capabilities,
            )
        )
        if adapter is None:
            return
        self._register_capability_adapter(
            registry,
            item,
            adapter,
            capabilities=tuple(sorted(audio_capabilities)),
            modalities=("audio",),
        )

    @staticmethod
    def _register_capability_adapter(
        registry: CapabilityAdapterRegistry,
        item: _ModelRegistration,
        adapter: Any,
        *,
        capabilities: tuple[str, ...],
        modalities: tuple[str, ...] = (),
    ) -> None:
        metadata = {**item.descriptor_metadata, "capabilities": list(capabilities)}
        if modalities:
            metadata["modalities"] = list(modalities)
        registry.register(
            AIBackendDescriptor(
                item.backend_id,
                adapter.adapter_id,
                model=item.model_key,
                priority=item.priority,
                credential_id=item.credential_id,
                metadata=metadata,
            ),
            adapter,
        )

    def _text_adapter(
        self,
        kind: str,
        backend_id: str,
        base_url: str,
        credential_id: str,
        model_key: str,
    ) -> Any:
        if kind == "anthropic":
            return AnthropicMessagesAdapter(
                AnthropicMessagesConfig(backend_id, base_url, credential_id, model_key),
                self.credential_vault.resolve,
            )
        return OpenAICompatibleAdapter(
            OpenAICompatibleConfig(backend_id, base_url, credential_id, model_key),
            self.credential_vault.resolve,
        )

    def _vision_adapter(
        self,
        kind: str,
        backend_id: str,
        base_url: str,
        credential_id: str,
        model_key: str,
    ) -> Any:
        if kind == "anthropic":
            messages_adapter = self._text_adapter(
                kind, backend_id, base_url, credential_id, model_key
            )
            return AnthropicVisionDescribeAdapter(messages_adapter)
        return OpenAIVisionDescribeAdapter(
            OpenAIVisionConfig(base_url, credential_id, model_key),
            self.credential_vault.resolve,
        )

    async def _register_web_providers(self, capability_registry: CapabilityAdapterRegistry) -> None:
        kind_map = {
            "TAVILY": "tavily",
            "BOCHA": "bocha",
            "BRAVE": "brave",
            "FIRECRAWL": "firecrawl",
            "BAIDU_AI": "baidu_ai_search",
            "EXA": "exa",
        }
        profiles = await self.profiles.list_profiles(include_orphaned=False)
        for profile in profiles:
            for row in await self.web.list_web_search_providers(str(profile.profile_id)):
                provider_kind = kind_map.get(str(row.provider_kind).upper())
                credential_id = str(row.credential_id or "").strip()
                if (
                    not row.enabled
                    or row.archived_at is not None
                    or not provider_kind
                    or not credential_id
                ):
                    continue
                adapter = build_web_search_adapter(
                    provider_kind, credential_id, self.credential_vault.resolve
                )
                capabilities = ["web.search"]
                if "web.image_search" in adapter.capabilities:
                    capabilities.append("web.image_search")
                if row.read_enabled and "web.read" in adapter.capabilities:
                    capabilities.append("web.read")
                capability_registry.register(
                    AIBackendDescriptor(
                        backend_id=str(row.backend_id),
                        adapter_id=adapter.adapter_id,
                        model=provider_kind,
                        priority=max(1, int(row.priority)),
                        credential_id=credential_id,
                        metadata={
                            "profile_id": str(row.profile_id),
                            "web_provider_id": str(row.provider_id),
                            "provider_kind": provider_kind,
                            "capabilities": capabilities,
                        },
                    ),
                    adapter,
                )

    async def _restore_circuit_scopes(self, circuits: CircuitBreaker) -> None:
        for circuit in await self.repository.list_ai_circuit_states():
            circuits.restore(
                str(circuit.get("circuit_scope") or ""),
                state=str(circuit.get("state") or "HEALTHY"),
                failure_count=int(circuit.get("failure_count") or 0),
                opened_until=self._parse_datetime(circuit.get("opened_until")),
                last_error_code=str(circuit.get("last_error_code") or ""),
            )

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    def _image_adapter(
        self,
        package_kind: str,
        model: Mapping[str, Any],
        package: Mapping[str, Any],
        base_url: str,
        credential_id: str,
        model_key: str,
    ) -> Any | None:
        effective = {**dict(package.get("config") or {}), **dict(model.get("config") or {})}
        resolver = self.credential_vault.resolve
        if package_kind == "gemini":
            return GeminiImageCapabilityAdapter(
                GeminiImageConfig(base_url, credential_id, model_key), resolver
            )
        if package_kind == "custom":
            return CustomHTTPImageCapabilityAdapter(
                self._custom_config(base_url, credential_id, effective), resolver
            )
        if package_kind not in {"openai", "openai_compatible"}:
            return None
        mode = str(effective.get("image_generation_mode") or "IMAGES_API").lower()
        if mode == "chat_response":
            return OpenAIChatImageCapabilityAdapter(
                OpenAIChatImageConfig(base_url, credential_id, model_key), resolver
            )
        return OpenAIImagesCapabilityAdapter(
            OpenAIImagesConfig(base_url, credential_id, model_key), resolver
        )

    @staticmethod
    def _custom_config(
        base_url: str, credential_id: str, config: Mapping[str, Any]
    ) -> CustomHTTPImageConfig:
        template = config.get("request_template")
        return CustomHTTPImageConfig(
            endpoint=str(config.get("endpoint") or base_url),
            credential_id=credential_id,
            auth_header=str(config.get("auth_header") or "Authorization"),
            auth_prefix=str(config.get("auth_prefix") or "Bearer "),
            request_template=(
                template if isinstance(template, Mapping) else {"prompt": "{{prompt}}"}
            ),
            image_paths=tuple(config.get("image_paths") or ("images",)),
            mime_type_path=str(config.get("mime_type_path") or "mime_type"),
            model_path=str(config.get("model_path") or "model"),
            features=AIImageBackendCapabilities(
                reference_image=bool(config.get("reference_image", False)),
                multiple_references=bool(config.get("multiple_references", False)),
                maximum_outputs=max(1, min(5, int(config.get("maximum_outputs") or 1))),
            ),
        )

    @staticmethod
    def _runtime_metadata(model: Mapping[str, Any], package: Mapping[str, Any]) -> dict[str, Any]:
        protocol = str(package.get("protocol") or "").upper()
        package_config = dict(package.get("config") or {})
        kinds = {
            "OPENAI": "openai",
            "OPENAI_COMPATIBLE": "openai_compatible",
            "ANTHROPIC": "anthropic",
            "GEMINI": "gemini",
            "CUSTOM_HTTP_IMAGE": "custom",
            "MINIMAX_TTS": "minimax_tts",
            "MIMO_TTS": "mimo_tts",
            "GPT_SOVITS_V2": "gpt_sovits_v2",
            "GSVI_TTS": "gsvi_tts",
        }
        capabilities = AIRuntimeLoader._runtime_text_capabilities(model.get("capabilities") or [])
        effective_config = {
            **package_config,
            **dict(model.get("config") or {}),
        }
        model_config = dict(model.get("config") or {})
        try:
            max_context_tokens = max(
                1,
                min(
                    10_000_000,
                    int(
                        effective_config.get("max_context_tokens")
                        or DEFAULT_MODEL_MAX_CONTEXT_TOKENS
                    ),
                ),
            )
        except (TypeError, ValueError):
            max_context_tokens = DEFAULT_MODEL_MAX_CONTEXT_TOKENS
        return {
            "package_id": str(package.get("package_id") or ""),
            "backend_id": str(model.get("backend_id") or ""),
            "priority": max(1, int(model.get("priority") or 1)),
            "capabilities": sorted(capabilities),
            "package_kind": kinds[protocol],
            "profile_id": str(package.get("profile_id") or "default"),
            "max_context_tokens": max_context_tokens,
            "supports_vision": AIRuntimeLoader._supports_vision(effective_config),
            "generation_parameters": normalize_model_generation_parameters(
                model_config.get("generation_parameters")
            ),
            "custom_request_parameters": normalize_model_custom_request_parameters(
                model_config.get("custom_request_parameters")
            ),
            "protocol": protocol,
            PROXY_CONTEXT_ISOLATION_CONFIG_KEY: proxy_context_isolation_enabled(package_config),
            "prompt_cache_config_fingerprint": AIRuntimeLoader._cache_config_fingerprint(
                model, package, protocol, model_config
            ),
        }

    @staticmethod
    def _cache_config_fingerprint(
        model: Mapping[str, Any],
        package: Mapping[str, Any],
        protocol: str,
        model_config: Mapping[str, Any],
    ) -> str:
        return prompt_cache_config_fingerprint(
            protocol=protocol,
            base_url=str(package.get("base_url") or ""),
            model=str(model.get("model_key") or ""),
            credential_id=str(package.get("credential_id") or ""),
            package_config=dict(package.get("config") or {}),
            model_config=model_config,
        )

    @staticmethod
    def _supports_vision(config: Mapping[str, Any]) -> bool:
        value = config["supports_vision"]
        if not isinstance(value, bool):
            raise ValueError("supports_vision must be a boolean")
        return value

    @staticmethod
    def _runtime_text_capabilities(raw_capabilities: Any) -> set[str]:
        capabilities = {str(item).strip().lower() for item in raw_capabilities if str(item).strip()}
        text_routes = {
            "sticker.collect",
            "sticker.check",
            "conversation.summary",
            "memory.reasoning",
            "conversation.turn_buffer",
            "conversation.group_interjection",
            "conversation.group_reply_relocation",
            "conversation.timer_lifecycle_review",
            "conversation.response_polish",
        }
        if capabilities.intersection(text_routes):
            capabilities.add("text.completion")
        return capabilities

    @staticmethod
    def _model_enabled(
        model: Mapping[str, Any],
        package: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> bool:
        capabilities = {str(item) for item in model.get("capabilities") or ()}
        context_ready = (
            not capabilities.intersection(TEXT_GENERATION_CAPABILITIES)
            or int(metadata.get("max_context_tokens") or 0) >= MINIMUM_MODEL_MAX_CONTEXT_TOKENS
        )
        return (
            bool(package.get("enabled", True))
            and not package.get("archived_at")
            and bool(model.get("enabled", True))
            and not model.get("archived_at")
            and context_ready
        )


__all__ = ["AIRuntimeLoader"]
