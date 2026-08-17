"""Profile and conversation runtime gates shared by every SoulCore execution path."""

from __future__ import annotations

from dataclasses import dataclass

from .ports import ProfilesRepositoryPort


class ProfileRuntimeDisabled(RuntimeError):
    """Raised when new SoulCore work is attempted for a disabled runtime scope."""

    def __init__(self, profile_id: str, instance_id: str = "") -> None:
        self.profile_id = str(profile_id or "")
        self.instance_id = str(instance_id or "")
        scope = f"{self.profile_id}/{self.instance_id}" if self.instance_id else self.profile_id
        super().__init__(f"SoulCore runtime is disabled: {scope}")


@dataclass(frozen=True, slots=True)
class ProfileRuntimeDecision:
    profile_id: str
    instance_id: str
    enabled: bool
    reason: str


class ProfileRuntimeGate:
    """Read-through gate with no cache across profile and conversation switches."""

    def __init__(self, repository: ProfilesRepositoryPort) -> None:
        self.repository = repository

    async def decision(
        self,
        profile_id: str,
        instance_id: str = "",
    ) -> ProfileRuntimeDecision:
        normalized = str(profile_id or "").strip()
        normalized_instance = str(instance_id or "").strip()
        if not normalized:
            return ProfileRuntimeDecision(
                normalized, normalized_instance, False, "missing_profile_id"
            )
        try:
            enabled = bool(await self.repository.get_profile_soulcore_enabled(normalized))
        except KeyError:
            return ProfileRuntimeDecision(
                normalized, normalized_instance, False, "profile_not_found"
            )
        if not enabled:
            return ProfileRuntimeDecision(
                normalized, normalized_instance, False, "profile_disabled"
            )
        if normalized_instance:
            policy = await self.repository.get_instance_chat_policy(
                normalized,
                normalized_instance,
            )
            if not policy.soulcore_enabled:
                return ProfileRuntimeDecision(
                    normalized,
                    normalized_instance,
                    False,
                    "instance_disabled",
                )
        return ProfileRuntimeDecision(normalized, normalized_instance, True, "runtime_enabled")

    async def is_enabled(self, profile_id: str, instance_id: str = "") -> bool:
        return (await self.decision(profile_id, instance_id)).enabled

    async def require_enabled(self, profile_id: str, instance_id: str = "") -> None:
        if not await self.is_enabled(profile_id, instance_id):
            raise ProfileRuntimeDisabled(profile_id, instance_id)

    async def image_send_enabled(self, profile_id: str, instance_id: str) -> bool:
        if not await self.is_enabled(profile_id, instance_id):
            return False
        policy = await self.repository.get_instance_chat_policy(profile_id, instance_id)
        return bool(policy.image_send_enabled)


__all__ = [
    "ProfileRuntimeDecision",
    "ProfileRuntimeDisabled",
    "ProfileRuntimeGate",
]
