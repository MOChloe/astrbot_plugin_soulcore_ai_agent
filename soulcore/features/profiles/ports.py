"""Persistence operations owned by profile, scope, and instance management."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ...contracts.models import (
    CharacterInstance,
    CoreState,
    InstanceChatPolicy,
    InstanceInitializationDecision,
    RoleProfile,
    ScopeConfig,
)


@dataclass(frozen=True, slots=True)
class ScopeConfigurationUpdate:
    """One administrator save spanning every scope-owned policy row."""

    profile_id: str
    scope: str
    role: Mapping[str, object]
    contact: Mapping[str, object]
    timezone: str | None
    delivery: Mapping[str, object]
    state_gate: Mapping[str, object]
    group_flow: Mapping[str, object] | None
    expected_scope_version: int
    expected_contact_version: int
    expected_timezone_version: int
    expected_delivery_version: int
    expected_state_gate_version: int
    expected_group_flow_version: int | None


class ScopeConfigurationConflict(ValueError):
    """A resource changed after the administrator loaded the settings page."""


class ScopeConfigurationTransactionPort(Protocol):
    async def save_scope_configuration(self, update: ScopeConfigurationUpdate) -> None: ...


class InstanceChatPolicyRepositoryPort(Protocol):
    async def get_instance_chat_policy(
        self,
        profile_id: str,
        instance_id: str,
    ) -> InstanceChatPolicy: ...

    async def upsert_instance_chat_policy(
        self,
        profile_id: str,
        instance_id: str,
        *,
        soulcore_enabled: bool,
        image_send_enabled: bool,
        expected_version: int,
        private_fallback_player_name: str = "",
        private_name_override_enabled: bool = False,
    ) -> InstanceChatPolicy | None: ...


class ProfilesRepositoryPort(InstanceChatPolicyRepositoryPort, Protocol):
    async def ensure_character_instance(
        self,
        profile_id: str,
        umo: str,
        platform_id: str = "",
        message_type: str = "",
        target_id: str = "",
        session_kind: str = "",
        ready: bool = True,
    ) -> CharacterInstance: ...

    async def ensure_profile(self, profile_id: str, name: str = "") -> RoleProfile: ...

    async def get_profile(self, profile_id: str) -> RoleProfile | None: ...

    async def list_profiles(self, *, include_orphaned: bool = True) -> list[RoleProfile]: ...

    async def get_console_preference(self, key: str) -> str: ...

    async def get_console_preferences(self, keys: Sequence[str]) -> dict[str, str]: ...

    async def set_console_preference(self, key: str, value: str) -> None: ...

    async def get_scope_config(self, profile_id: str, scope: str) -> ScopeConfig | None: ...

    async def list_scope_configs(self, profile_id: str) -> list[ScopeConfig]: ...

    async def get_scope_config_version(self, profile_id: str, scope: str) -> int: ...

    async def update_scope_config(
        self,
        profile_id: str,
        scope: str,
        patch: dict[str, object],
    ) -> ScopeConfig: ...

    async def set_profile_thinking_complexity(
        self,
        profile_id: str,
        complexity: str,
    ) -> RoleProfile: ...

    async def get_profile_soulcore_enabled(self, profile_id: str) -> bool: ...

    async def set_profile_soulcore_enabled(
        self,
        profile_id: str,
        enabled: bool,
    ) -> RoleProfile: ...

    async def set_profile_quick_setup_decided(
        self,
        profile_id: str,
        decided: bool,
    ) -> RoleProfile: ...

    async def finish_profile_quick_setup(
        self,
        profile_id: str,
        *,
        thinking_complexity: str,
    ) -> RoleProfile: ...

    async def get_profile_turn_buffer_enabled(self, profile_id: str) -> bool: ...

    async def set_profile_turn_buffer_enabled(
        self,
        profile_id: str,
        enabled: bool,
    ) -> RoleProfile: ...

    async def get_profile_image_generation_enabled(self, profile_id: str) -> bool: ...

    async def set_profile_image_generation_enabled(
        self,
        profile_id: str,
        enabled: bool,
    ) -> RoleProfile: ...

    async def get_profile_response_polish_enabled(self, profile_id: str) -> bool: ...

    async def set_profile_response_polish_enabled(
        self,
        profile_id: str,
        enabled: bool,
    ) -> RoleProfile: ...

    async def get_profile_response_polish_timeout_seconds(self, profile_id: str) -> int: ...

    async def set_profile_response_polish_timeout_seconds(
        self,
        profile_id: str,
        timeout_seconds: int,
    ) -> RoleProfile: ...

    async def get_profile_passive_no_reply_notice_enabled(self, profile_id: str) -> bool: ...

    async def set_profile_passive_no_reply_notice_enabled(
        self,
        profile_id: str,
        enabled: bool,
    ) -> RoleProfile: ...

    async def get_instance_state(self, profile_id: str, instance_id: str) -> CoreState: ...

    async def begin_instance_initialization(
        self,
        profile_id: str,
        instance_id: str,
        due_at: datetime,
        conversation_ref: str | None = None,
    ) -> InstanceInitializationDecision: ...

    async def mark_instance_ready(
        self,
        profile_id: str,
        instance_id: str,
        ready: bool,
    ) -> bool: ...

    async def get_character_instance(
        self,
        profile_id: str,
        instance_id: str,
    ) -> CharacterInstance | None: ...

    async def list_character_instances(
        self,
        profile_id: str,
        scope: str | None = None,
    ) -> list[CharacterInstance]: ...

    async def upsert_participant_identity(
        self, profile_id: str, instance_id: str, **values: object
    ) -> None: ...

    async def upsert_participant_identities(
        self, profile_id: str, instance_id: str, **values: object
    ) -> None: ...

    async def list_participant_identities(
        self, profile_id: str, instance_id: str
    ) -> list[dict[str, object]]: ...
