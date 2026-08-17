"""Profile and role-instance administrator controller."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from astrbot.api.star import Context

    from ....features.group_flow.ports import GroupFlowRepository
    from ....features.turn_buffer.worker import TurnBufferWorker

from ....features.files.ports import FileRepositoryPort
from ....features.profiles.ports import ProfilesRepositoryPort
from ....features.timeline.ports import TimelineRepositoryPort
from ....shared.event_log import EventLogPort, record_event
from ...astrbot.context_message import (
    instance_identity_labels,
    live_instance_display_names,
)
from ...astrbot.profile import ProfileResolver
from ..presentation import jsonable


class ProfilesAdminController:
    def __init__(
        self,
        profiles_repository: ProfilesRepositoryPort,
        timeline_repository: TimelineRepositoryPort,
        file_repository: FileRepositoryPort,
        event_log: EventLogPort,
        context: Context,
        profile_resolver: ProfileResolver,
        group_flow_repository: GroupFlowRepository | None = None,
        turn_buffer_worker: TurnBufferWorker | None = None,
    ) -> None:
        self.profiles_repository = profiles_repository
        self.timeline_repository = timeline_repository
        self.file_repository = file_repository
        self.event_log = event_log
        self.context = context
        self.profile_resolver = profile_resolver
        self.group_flow_repository = group_flow_repository
        self.turn_buffer_worker = turn_buffer_worker

    async def sync_profiles(self) -> list[Any]:
        assert self.profiles_repository is not None
        profiles = await self.profile_resolver.list_profiles()
        return await self.profiles_repository.sync_profiles(
            [{"id": item.id, "name": item.name} for item in profiles]
        )

    async def require_known_profile(self, profile_id: str) -> None:
        known = {item.id for item in await self.profile_resolver.list_profiles()}
        if profile_id not in known:
            raise ValueError(f"unknown AstrBot profile_id: {profile_id}")

    async def main_config_snapshot(self, profile_id: str) -> dict[str, Any]:
        """Return the one profile-wide SoulCore master switch."""

        assert self.profiles_repository is not None
        profile = await self.profiles_repository.get_profile(profile_id)
        if profile is None:
            profile = await self.profiles_repository.ensure_profile(profile_id)
        enabled = await self.profiles_repository.get_profile_soulcore_enabled(profile_id)
        turn_buffer_enabled = await self.profiles_repository.get_profile_turn_buffer_enabled(
            profile_id
        )
        image_generation_enabled = (
            await self.profiles_repository.get_profile_image_generation_enabled(profile_id)
        )
        response_polish_enabled = (
            await self.profiles_repository.get_profile_response_polish_enabled(profile_id)
        )
        response_polish_timeout_seconds = (
            await self.profiles_repository.get_profile_response_polish_timeout_seconds(profile_id)
        )
        passive_no_reply_notice_enabled = (
            await self.profiles_repository.get_profile_passive_no_reply_notice_enabled(profile_id)
        )
        file_artifacts_enabled = await self.file_repository.get_profile_file_artifacts_enabled(
            profile_id
        )
        return {
            "profile_id": profile_id,
            "enabled": enabled,
            "turn_buffer_enabled": turn_buffer_enabled,
            "image_generation_enabled": image_generation_enabled,
            "response_polish_enabled": response_polish_enabled,
            "response_polish_timeout_seconds": response_polish_timeout_seconds,
            "passive_no_reply_notice_enabled": passive_no_reply_notice_enabled,
            "file_artifacts_enabled": file_artifacts_enabled,
            "status": "ENABLED" if enabled else "DISABLED",
            "disabled_message": (
                "" if enabled else "SoulCore不会接管该配置档案的消息，已有配置和数据会保留。"
            ),
            "updated_at": jsonable(profile.updated_at),
        }

    async def save_main_config(
        self,
        profile_id: str,
        *,
        enabled: bool | None = None,
        turn_buffer_enabled: bool | None = None,
        image_generation_enabled: bool | None = None,
        response_polish_enabled: bool | None = None,
        response_polish_timeout_seconds: int | None = None,
        passive_no_reply_notice_enabled: bool | None = None,
        file_artifacts_enabled: bool | None = None,
    ) -> dict[str, Any]:
        """Patch profile-wide feature switches for only one AstrBot profile."""

        assert self.profiles_repository is not None
        changed: dict[str, bool | int] = {}
        if enabled is not None:
            await self.profiles_repository.set_profile_soulcore_enabled(profile_id, bool(enabled))
            changed["enabled"] = bool(enabled)
        if turn_buffer_enabled is not None:
            await self.profiles_repository.set_profile_turn_buffer_enabled(
                profile_id, bool(turn_buffer_enabled)
            )
            changed["turn_buffer_enabled"] = bool(turn_buffer_enabled)
        if image_generation_enabled is not None:
            await self.profiles_repository.set_profile_image_generation_enabled(
                profile_id, bool(image_generation_enabled)
            )
            changed["image_generation_enabled"] = bool(image_generation_enabled)
        if response_polish_enabled is not None:
            await self.profiles_repository.set_profile_response_polish_enabled(
                profile_id, bool(response_polish_enabled)
            )
            changed["response_polish_enabled"] = bool(response_polish_enabled)
        if response_polish_timeout_seconds is not None:
            await self.profiles_repository.set_profile_response_polish_timeout_seconds(
                profile_id, response_polish_timeout_seconds
            )
            changed["response_polish_timeout_seconds"] = response_polish_timeout_seconds
        if passive_no_reply_notice_enabled is not None:
            await self.profiles_repository.set_profile_passive_no_reply_notice_enabled(
                profile_id, bool(passive_no_reply_notice_enabled)
            )
            changed["passive_no_reply_notice_enabled"] = bool(passive_no_reply_notice_enabled)
        if file_artifacts_enabled is not None:
            await self.file_repository.set_profile_file_artifacts_enabled(
                profile_id, bool(file_artifacts_enabled)
            )
            changed["file_artifacts_enabled"] = bool(file_artifacts_enabled)
        if self.turn_buffer_worker is not None and (
            "enabled" in changed or "turn_buffer_enabled" in changed
        ):
            await self.turn_buffer_worker.reconcile_profile_switches(profile_id)
        snapshot = await self.main_config_snapshot(profile_id)
        await record_event(
            self.event_log,
            profile_id=profile_id,
            level="INFO",
            category="configuration",
            message="SoulCore主配置已更新",
            details=changed,
        )
        return {"ok": True, "main_config": snapshot}

    async def mark_quick_setup_decided(self, profile_id: str) -> Any:
        """Remember that this role will be configured without the guided flow."""

        profile = await self.profiles_repository.set_profile_quick_setup_decided(profile_id, True)
        await record_event(
            self.event_log,
            profile_id=profile_id,
            level="INFO",
            category="configuration",
            message="已选择自行配置 SoulCore",
            details={"quick_setup_decided": True},
        )
        return profile

    async def finish_quick_setup(
        self,
        profile_id: str,
        *,
        thinking_complexity: str,
    ) -> Any:
        """Atomically mark the guide complete and turn on the profile runtime."""

        profile = await self.profiles_repository.finish_profile_quick_setup(
            profile_id,
            thinking_complexity=thinking_complexity,
        )
        if self.turn_buffer_worker is not None:
            try:
                await self.turn_buffer_worker.reconcile_profile_switches(profile_id)
            except Exception as exc:
                await record_event(
                    self.event_log,
                    profile_id=profile_id,
                    level="WARNING",
                    category="configuration",
                    message="快速设置完成后缓冲任务未能立即刷新",
                    details={"error_type": type(exc).__name__},
                )
        await record_event(
            self.event_log,
            profile_id=profile_id,
            level="INFO",
            category="configuration",
            message="SoulCore 快速设置已完成并启用",
            details={
                "enabled": True,
                "quick_setup_decided": True,
                "thinking_complexity": thinking_complexity,
            },
        )
        return profile

    async def profile_snapshot(self, profile_id: str) -> dict[str, Any]:
        assert self.profiles_repository is not None
        role = await self.profiles_repository.get_profile(profile_id)
        return {"profile": jsonable(role)}

    async def scope_config_snapshot(self, profile_id: str, scope: str) -> dict[str, Any]:
        """Return one private/group template without touching live instances."""

        assert self.profiles_repository is not None
        config = await self.profiles_repository.get_scope_config(profile_id, scope)
        if config is None:
            raise ValueError(f"unknown scope configuration for profile {profile_id}: {scope}")
        scope_config_version = await self.profiles_repository.get_scope_config_version(
            profile_id, scope
        )
        contact = await self.timeline_repository.get_contact_policy(profile_id, scope)
        delivery = await self.timeline_repository.get_delivery_policy(profile_id, scope)
        state_gate = await self.timeline_repository.get_state_gate_policy(profile_id, scope)
        timezone_state = await self.timeline_repository.get_profile_timezone(profile_id)
        group_flow = (
            await self.group_flow_repository.get_group_flow_policy(profile_id, scope)
            if scope == "group" and self.group_flow_repository is not None
            else None
        )
        contact_value = jsonable(contact) if contact is not None else {}
        delivery_value = jsonable(delivery) if delivery is not None else {}
        gate_value = jsonable(state_gate) if state_gate is not None else {}
        return {
            "scope_config": self._scope_config_view(
                config,
                contact_value,
                delivery_value,
                gate_value,
                timezone_state,
                jsonable(group_flow) if group_flow is not None else {},
                scope_config_version,
            )
        }

    @classmethod
    def _scope_config_view(
        cls,
        config: Any,
        contact: dict[str, Any],
        delivery: dict[str, Any],
        gate: dict[str, Any],
        timezone_state: dict[str, Any],
        group_flow: dict[str, Any],
        scope_config_version: int,
    ) -> dict[str, Any]:
        config_value = jsonable(config)
        if not isinstance(config_value, dict):
            raise TypeError("scope configuration must be an object or mapping")
        excluded = {"profile_id", "scope", "version", "created_at", "updated_at"}
        contact_fields = {key: item for key, item in contact.items() if key not in excluded}
        for name in ("proactive_enabled", "quiet_enabled"):
            if name in contact_fields and contact_fields[name] is not None:
                contact_fields[name] = bool(contact_fields[name])
        group_flow_fields = (
            {
                "group_flow_policy_version": int(cls._value(group_flow, "version", 1)),
                "quiet_seconds": int(cls._value(group_flow, "quiet_seconds", 30)),
                "base_message_count": int(cls._value(group_flow, "base_message_count", 2)),
                "ordinary_min_reply_gap_seconds": int(
                    cls._value(group_flow, "ordinary_min_reply_gap_seconds", 0)
                ),
                "judge_token_budget": int(cls._value(group_flow, "judge_token_budget", 2048)),
            }
            if str(cls._value(config_value, "scope", "")) == "group"
            else {}
        )
        return {
            **config_value,
            **contact_fields,
            **group_flow_fields,
            "scope_config_version": scope_config_version,
            "contact_policy_version": int(cls._value(contact, "version", 1)),
            "delivery_policy_version": int(cls._value(delivery, "version", 1)),
            "state_gate_policy_version": int(cls._value(gate, "version", 1)),
            "state_message_gate_enabled": bool(gate.get("enabled", False)),
            "state_message_silent_enabled": bool(gate.get("silent_enabled", False)),
            "state_gate_max_hours": int(cls._value(gate, "max_gate_hours", 24)),
            "timezone_version": int(cls._value(timezone_state, "version", 1)),
            "timezone": str(cls._value(timezone_state, "timezone", "")),
            "group_send_qpm_limit": int(cls._value(delivery, "group_send_qpm_limit", 20)),
            "send_qpm_limit": int(cls._value(delivery, "send_qpm_limit", 20)),
            "max_context_tokens": int(cls._value(config_value, "max_context_tokens", 128000)),
            "target_context_tokens": int(cls._value(config_value, "target_context_tokens", 64000)),
        }

    @staticmethod
    def _value(values: dict[str, Any], key: str, default: Any) -> Any:
        value = values.get(key)
        return default if value in (None, "") else value

    async def role_instances_snapshot(self, profile_id: str) -> dict[str, Any]:
        """Expose stable conversation instances, grouped for the settings page."""

        assert self.profiles_repository is not None
        raw_items = await self.profiles_repository.list_character_instances(profile_id)
        serialized = self._serialized_instances(raw_items)
        directory_items = [self._directory_entry(value) for value in serialized]
        live_names = await live_instance_display_names(self.context, directory_items)
        private: list[dict[str, Any]] = []
        groups: list[dict[str, Any]] = []
        for value in serialized:
            item = self._instance_item(value, profile_id, live_names)
            if item is None:
                continue
            (groups if item["scope"] == "group" else private).append(item)
        private.sort(key=lambda item: (item["display_name"], item["instance_id"]))
        groups.sort(key=lambda item: (item["display_name"], item["instance_id"]))
        return {
            "profile_id": profile_id,
            "sections": {"private": private, "group": groups},
            "instances": [*private, *groups],
        }

    @staticmethod
    def _serialized_instances(raw_items: list[Any]) -> list[dict[str, Any]]:
        result = []
        for raw in raw_items:
            value = jsonable(raw)
            if isinstance(value, dict):
                result.append(value)
        return result

    @classmethod
    def _directory_entry(cls, value: dict[str, Any]) -> dict[str, Any]:
        return value

    @classmethod
    def _instance_item(
        cls,
        value: dict[str, Any],
        profile_id: str,
        live_names: dict[tuple[str, str, str], str],
    ) -> dict[str, Any] | None:
        directory = cls._directory_entry(value)
        route_umo = cls._first_text(directory.get("route_umo"))
        kind = str(directory.get("scope") or "").lower()
        scope = "group" if kind in {"group", "guild"} else "private"
        instance_id = cls._first_text(directory.get("instance_id"))
        if not instance_id or not route_umo:
            return None
        target_id = cls._first_text(directory.get("target_id"), instance_id)
        platform_id = cls._first_text(directory.get("platform_id"))
        identity = instance_identity_labels(
            scope=scope,
            target_id=target_id,
            instance_id=instance_id,
            display_name=live_names.get((platform_id, scope, target_id), ""),
        )
        return {
            **value,
            "instance_id": instance_id,
            "profile_id": profile_id,
            "scope": scope,
            **identity,
        }

    @staticmethod
    def _first_text(*values: Any) -> str:
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
        return ""

    async def require_role_instance(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        assert self.profiles_repository is not None
        raw = await self.profiles_repository.get_character_instance(profile_id, instance_id)
        if raw is not None:
            value = jsonable(raw)
            snapshot = await self.role_instances_snapshot(profile_id)
            for item in snapshot["instances"]:
                if item["instance_id"] == instance_id:
                    return {**item, **(value if isinstance(value, dict) else {})}
        snapshot = await self.role_instances_snapshot(profile_id)
        for item in snapshot["instances"]:
            if item["instance_id"] == instance_id:
                return item
        raise ValueError(f"unknown role instance_id for profile {profile_id}: {instance_id}")

    async def character_instance_snapshot(
        self, profile_id: str, instance_id: str | None
    ) -> dict[str, Any]:
        snapshot = await self.profile_snapshot(profile_id)
        if not instance_id:
            return snapshot
        assert self.profiles_repository is not None
        instance = await self.require_role_instance(profile_id, instance_id)
        state = await self.profiles_repository.get_instance_state(profile_id, instance_id)
        state_value = jsonable(state)
        profile = dict(snapshot.get("profile") or {})
        for key in (
            "enabled",
            "proactive_enabled",
            "extra_background",
            "min_wakeup_minutes",
            "max_wakeup_minutes",
            "low_frequency_min_wakeup_minutes",
            "low_frequency_max_wakeup_minutes",
        ):
            if key in instance:
                profile[key] = instance[key]
        return {"profile": profile, "state": state_value}
