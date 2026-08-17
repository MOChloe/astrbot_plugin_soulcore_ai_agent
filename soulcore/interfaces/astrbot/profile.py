"""AstrBot multi-configuration profile resolver."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ConfigProfile:
    id: str
    name: str
    path: str | None = None


class ProfileResolver:
    """Resolve a UMO to the stable AstrBot configuration profile id."""

    def __init__(self, context: Any) -> None:
        self.context = context

    @property
    def manager(self) -> Any:
        return self.context.astrbot_config_mgr

    async def resolve_umo(self, umo: str) -> ConfigProfile:
        return _coerce_profile(self.manager.get_conf_info(umo))

    async def resolve_event(self, event: Any) -> ConfigProfile:
        from .umo import physical_event_route

        return await self.resolve_umo(physical_event_route(event).raw)

    async def list_profiles(self) -> list[ConfigProfile]:
        profiles = [_coerce_profile(item) for item in self.manager.get_conf_list()]
        unique = {item.id: item for item in profiles}
        return list(unique.values())


def _coerce_profile(value: Any) -> ConfigProfile:
    if isinstance(value, ConfigProfile):
        return value
    if isinstance(value, Mapping):
        profile_id = str(value.get("id") or "").strip()
        if not profile_id:
            raise RuntimeError("AstrBot configuration profile has no id")
        return ConfigProfile(
            profile_id,
            str(value.get("name") or profile_id),
            str(value["path"]) if value.get("path") is not None else None,
        )
    profile_id = str(value.id).strip()
    if not profile_id:
        raise RuntimeError("AstrBot configuration profile has no id")
    return ConfigProfile(
        profile_id,
        str(getattr(value, "name", None) or profile_id),
        str(getattr(value, "path", "")) or None,
    )
