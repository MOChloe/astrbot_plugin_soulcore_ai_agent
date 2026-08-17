"""Strict AstrBot Embedding/Rerank provider selection for Recall."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .ports import RecallRepository


@dataclass(frozen=True, slots=True)
class RecallProviderSelection:
    embedding_provider_id: str = ""
    rerank_provider_id: str = ""
    embedding_source: str = "global"
    rerank_source: str = "global"
    version: int = 0


class RecallProviderConfigurationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class AstrBotRecallProviderRegistry:
    """Resolve only explicitly selected AstrBot-native provider types.

    An absent or invalid provider never causes selection of another provider.
    Text recall remains available and the readiness view explains the downgrade.
    """

    def __init__(self, context: Any, config: Any) -> None:
        self.context = context
        self.config = config if config is not None else {}

    async def selection(
        self, repository: RecallRepository, profile_id: str
    ) -> RecallProviderSelection:
        settings = await repository.get_role_settings(profile_id)
        global_embedding = self._config_value("recall_embedding_provider_id")
        global_rerank = self._config_value("recall_rerank_provider_id")
        embedding_override = settings.get("embedding_provider_id")
        rerank_override = settings.get("rerank_provider_id")
        return RecallProviderSelection(
            embedding_provider_id=str(
                global_embedding if embedding_override is None else embedding_override
            ).strip(),
            rerank_provider_id=str(
                global_rerank if rerank_override is None else rerank_override
            ).strip(),
            embedding_source="global" if embedding_override is None else "role_override",
            rerank_source="global" if rerank_override is None else "role_override",
            version=int(settings.get("version") or 0),
        )

    def embedding(self, provider_id: str) -> Any | None:
        provider_id = str(provider_id or "").strip()
        if not provider_id:
            return None
        provider = self.context.get_provider_by_id(provider_id)
        try:
            from astrbot.core.provider.provider import EmbeddingProvider
        except ImportError as exc:  # pragma: no cover - AstrBot always supplies this at runtime
            raise RecallProviderConfigurationError(
                "embedding_interface_unavailable", "AstrBot EmbeddingProvider 接口不可用"
            ) from exc
        if provider is None:
            raise RecallProviderConfigurationError(
                "embedding_provider_missing", "指定的 Embedding Provider 不存在"
            )
        if not isinstance(provider, EmbeddingProvider):
            raise RecallProviderConfigurationError(
                "embedding_wrong_interface", "指定 Provider 不是 Embedding Provider"
            )
        return provider

    def rerank(self, provider_id: str) -> Any | None:
        provider_id = str(provider_id or "").strip()
        if not provider_id:
            return None
        provider = self.context.get_provider_by_id(provider_id)
        try:
            from astrbot.core.provider.provider import RerankProvider
        except ImportError as exc:  # pragma: no cover
            raise RecallProviderConfigurationError(
                "rerank_interface_unavailable", "AstrBot RerankProvider 接口不可用"
            ) from exc
        if provider is None:
            raise RecallProviderConfigurationError(
                "rerank_provider_missing", "指定的 Rerank Provider 不存在"
            )
        if not isinstance(provider, RerankProvider):
            raise RecallProviderConfigurationError(
                "rerank_wrong_interface", "指定 Provider 不是 Rerank Provider"
            )
        return provider

    @staticmethod
    def embedding_fingerprint(provider: Any) -> tuple[str, int]:
        dimension = int(provider.get_dim())
        if dimension < 1:
            raise RecallProviderConfigurationError(
                "embedding_dimension", "Embedding Provider 返回了无效维度"
            )
        meta = provider.meta()
        payload = {
            "id": str(getattr(meta, "id", "") or ""),
            "type": str(getattr(meta, "type", "") or ""),
            "model": str(getattr(meta, "model", "") or provider.get_model() or ""),
            "dimension": dimension,
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        return digest, dimension

    def provider_options(self) -> dict[str, list[dict[str, str]]]:
        embeddings = tuple(self.context.get_all_embedding_providers() or ())
        reranks = tuple(
            getattr(getattr(self.context, "provider_manager", None), "rerank_provider_insts", ())
            or ()
        )
        return {
            "embedding": [self._provider_view(item) for item in embeddings],
            "rerank": [self._provider_view(item) for item in reranks],
        }

    @staticmethod
    def _provider_view(provider: Any) -> dict[str, str]:
        meta = provider.meta()
        return {
            "id": str(getattr(meta, "id", "") or ""),
            "model": str(getattr(meta, "model", "") or ""),
            "type": str(getattr(meta, "type", "") or ""),
        }

    def _config_value(self, key: str) -> str:
        getter = getattr(self.config, "get", None)
        if callable(getter):
            return str(getter(key, "") or "").strip()
        try:
            return str(self.config[key] or "").strip()
        except (KeyError, TypeError):
            return ""


__all__ = [
    "AstrBotRecallProviderRegistry",
    "RecallProviderConfigurationError",
    "RecallProviderSelection",
]
