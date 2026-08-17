"""Authoritative runtime policy for sticker visibility, acquisition and delivery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..profiles.ports import ProfilesRepositoryPort
from .domain import StickerConfig, StickerSourceKind
from .ports import StickerRepositoryPort


class StickerRuntimeDisabled(RuntimeError):
    """Raised when a current sticker operation is no longer authorized."""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason or "sticker_runtime_disabled")
        super().__init__(self.reason)


@dataclass(frozen=True, slots=True)
class StickerRuntimePolicy:
    config: StickerConfig
    web_search_enabled: bool
    image_generation_enabled: bool

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def player_collection_enabled(self) -> bool:
        return bool(self.enabled and self.config.player_collection_enabled)

    @property
    def web_collection_enabled(self) -> bool:
        return bool(self.enabled and self.config.web_collection_enabled and self.web_search_enabled)

    @property
    def generation_enabled(self) -> bool:
        return bool(
            self.enabled and self.config.generation_enabled and self.image_generation_enabled
        )

    @property
    def collection_enabled(self) -> bool:
        return bool(self.web_collection_enabled or self.generation_enabled)

    def require_enabled(self) -> None:
        if not self.enabled:
            raise StickerRuntimeDisabled("sticker_system_disabled")

    def require_collection(self) -> None:
        self.require_enabled()
        if not self.collection_enabled:
            raise StickerRuntimeDisabled("sticker_collection_sources_disabled")

    def require_source(self, source_kind: StickerSourceKind | str) -> None:
        self.require_enabled()
        source = StickerSourceKind(str(source_kind).upper())
        if source is StickerSourceKind.PLAYER and not self.player_collection_enabled:
            raise StickerRuntimeDisabled("sticker_player_collection_disabled")
        if source is StickerSourceKind.WEB and not self.web_collection_enabled:
            raise StickerRuntimeDisabled("sticker_web_collection_disabled")
        if source is StickerSourceKind.GENERATED and not self.generation_enabled:
            raise StickerRuntimeDisabled("sticker_generation_disabled")


async def load_sticker_runtime_policy(
    repository: StickerRepositoryPort,
    profiles: ProfilesRepositoryPort,
    profile_id: str,
    *,
    instance_id: str = "",
    scope: str = "",
    config: StickerConfig | None = None,
) -> StickerRuntimePolicy:
    resolved_scope = str(scope or "").strip()
    if not resolved_scope:
        instance = await profiles.get_character_instance(profile_id, str(instance_id))
        if instance is None:
            raise ValueError("sticker instance unavailable")
        resolved_scope = str(instance.scope)
    resolved_config = config or await repository.get_sticker_config(profile_id, resolved_scope)
    profile = await profiles.get_profile(profile_id)
    if profile is None:
        raise ValueError("sticker profile unavailable")
    return StickerRuntimePolicy(
        config=resolved_config,
        web_search_enabled=bool(profile.web_search_enabled),
        image_generation_enabled=bool(
            await profiles.get_profile_image_generation_enabled(profile_id)
        ),
    )


async def require_sticker_runtime_enabled(
    repository: StickerRepositoryPort,
    profiles: ProfilesRepositoryPort,
    profile_id: str,
    instance_id: str,
) -> StickerRuntimePolicy:
    policy = await load_sticker_runtime_policy(
        repository,
        profiles,
        profile_id,
        instance_id=instance_id,
    )
    policy.require_enabled()
    return policy


def sticker_source_kind(value: Any) -> StickerSourceKind:
    return value if isinstance(value, StickerSourceKind) else StickerSourceKind(str(value).upper())


__all__ = [
    "StickerRuntimeDisabled",
    "StickerRuntimePolicy",
    "load_sticker_runtime_policy",
    "require_sticker_runtime_enabled",
    "sticker_source_kind",
]
