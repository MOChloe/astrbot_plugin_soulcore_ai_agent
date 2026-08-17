"""Role scope and contact-policy administrator operations."""

from __future__ import annotations

import math
import re
import zoneinfo
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ....contracts.message_reference import normalize_private_fallback_player_name
from ....contracts.thinking import MainCoreThinkingPolicy, thinking_policy_from_value
from ....features.profiles.ports import (
    ProfilesRepositoryPort,
    ScopeConfigurationTransactionPort,
    ScopeConfigurationUpdate,
)
from ....features.timeline.ports import TimelineRepositoryPort
from ....shared.event_log import record_event
from ..presentation import jsonable
from .profiles import ProfilesAdminController


def _required_matching_version(
    payload: Mapping[str, Any], field: str, current: Any, conflict_message: str
) -> int:
    if field not in payload:
        raise ValueError(f"missing version field: {field}")
    expected = int(payload.get(field) or 0)
    if expected != int((current or {}).get("version") or 0):
        raise ValueError(conflict_message)
    return expected


class InstanceOverrideActionsMixin:
    """Keep the profile-settings controller focused on validation and snapshots."""

    timeline_repository: Any

    if TYPE_CHECKING:

        def _text(
            self,
            payload: Mapping[str, Any],
            name: str,
            *,
            default: str = "",
        ) -> str: ...

        async def _delete_instance_overrides(self, *args: Any) -> None: ...

        async def _save_instance_overrides(self, *args: Any) -> None: ...

    async def _instance_override_resources(
        self, profile_id: str, instance_id: str
    ) -> tuple[Any, Any, Any]:
        current = await self.timeline_repository.get_instance_contact_override(
            profile_id, instance_id
        )
        current_gate = await self.timeline_repository.get_instance_state_gate_override(
            profile_id, instance_id
        )
        current_delivery = await self.timeline_repository.get_instance_delivery_override(
            profile_id, instance_id
        )
        return current, current_gate, current_delivery

    @staticmethod
    def _instance_override_versions(
        payload: Mapping[str, Any], current: Any, current_gate: Any, current_delivery: Any
    ) -> tuple[int, int, int]:
        required = (
            "expected_version",
            "expected_state_gate_version",
            "expected_delivery_version",
        )
        missing = tuple(field for field in required if field not in payload)
        if missing:
            raise ValueError("missing version fields: " + ", ".join(missing))
        expected = _required_matching_version(
            payload,
            "expected_version",
            current,
            "instance Contact override changed; reload before saving",
        )
        expected_gate = _required_matching_version(
            payload,
            "expected_state_gate_version",
            current_gate,
            "instance state gate override changed; reload before saving",
        )
        expected_delivery = _required_matching_version(
            payload,
            "expected_delivery_version",
            current_delivery,
            "instance delivery override changed; reload before saving",
        )
        return expected, expected_gate, expected_delivery

    async def _apply_instance_override_action(
        self,
        profile_id: str,
        instance_id: str,
        payload: Mapping[str, Any],
        current: Any,
        current_gate: Any,
        current_delivery: Any,
        expected: int,
        expected_gate: int,
        expected_delivery: int,
    ) -> None:
        action = self._text(payload, "action", default="save").lower()
        if action == "delete":
            await self._delete_instance_overrides(
                profile_id,
                instance_id,
                current,
                current_gate,
                current_delivery,
                expected,
                expected_gate,
                expected_delivery,
            )
            return
        if action == "save":
            await self._save_instance_overrides(
                profile_id,
                instance_id,
                payload,
                current_delivery,
                expected,
                expected_gate,
                expected_delivery,
            )
            return
        raise ValueError("action must be save or delete")


def optional_qpm(supplied: Mapping[str, Any], name: str) -> int | None:
    value = supplied.get(name)
    if value in (None, ""):
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be positive or null")
    return parsed


def _bounded_integer(
    payload: Mapping[str, Any], name: str, default: int, minimum: int, maximum: int
) -> int:
    raw = payload.get(name, default)
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        value = int(raw)
        numeric = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(numeric) or numeric != value:
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def validate_group_flow_patch(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "quiet_seconds": _bounded_integer(payload, "quiet_seconds", 30, 5, 300),
        "base_message_count": _bounded_integer(payload, "base_message_count", 2, 1, 50),
        "ordinary_min_reply_gap_seconds": _bounded_integer(
            payload, "ordinary_min_reply_gap_seconds", 0, 0, 86400
        ),
        "judge_token_budget": _bounded_integer(payload, "judge_token_budget", 2048, 512, 8192),
    }


_INSTANCE_CONTACT_FIELDS = {
    "proactive_enabled",
    "check_min_minutes",
    "check_max_minutes",
    "quiet_enabled",
    "quiet_start",
    "quiet_end",
    "min_success_gap_minutes",
    "daily_limit_mode",
    "daily_success_limit",
    "unanswered_limit_mode",
    "max_consecutive_unanswered",
    "failure_mode",
    "retry_delay_minutes",
    "retry_max_attempts",
}


def _validate_effective_contact_interval(
    patch: Mapping[str, Any], effective: Mapping[str, Any]
) -> None:
    minimum = patch["check_min_minutes"]
    maximum = patch["check_max_minutes"]
    resolved_min = int(minimum if minimum is not None else effective["check_min_minutes"])
    resolved_max = int(maximum if maximum is not None else effective["check_max_minutes"])
    if resolved_max < resolved_min:
        raise ValueError("effective ContactClock interval must satisfy min <= max")


def _optional_failure_mode(supplied: Mapping[str, Any]) -> str | None:
    value = supplied.get("failure_mode")
    mode = None if value in (None, "") else str(value).strip().upper()
    if mode not in {None, "SKIP", "RETRY_BACKOFF"}:
        raise ValueError("failure_mode must inherit, SKIP or RETRY_BACKOFF")
    return mode


def validate_instance_contact_override(
    payload: Mapping[str, Any], effective: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the complete sparse editor payload without accepting other config."""

    supplied = payload.get("override")
    if not isinstance(supplied, Mapping):
        raise ValueError("override must be a JSON object")
    unknown = set(supplied) - _INSTANCE_CONTACT_FIELDS
    if unknown:
        raise ValueError(f"unsupported instance Contact fields: {sorted(unknown)}")

    def optional_bool(name: str) -> bool | None:
        value = supplied.get(name)
        if value is None:
            return None
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be boolean or null")
        return value

    def optional_positive(name: str) -> int | None:
        value = supplied.get(name)
        if value in (None, ""):
            return None
        parsed = int(value)
        if parsed < 1:
            raise ValueError(f"{name} must be a positive integer or null")
        return parsed

    def optional_time(name: str) -> str | None:
        value = supplied.get(name)
        if value in (None, ""):
            return None
        result = str(value).strip()
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", result):
            raise ValueError(f"{name} must use HH:MM or null")
        return result

    def limit_pair(mode_name: str, value_name: str) -> tuple[str, int | None]:
        mode = str(supplied.get(mode_name) or "INHERIT").strip().upper()
        if mode not in {"INHERIT", "LIMITED", "UNLIMITED"}:
            raise ValueError(f"{mode_name} must be INHERIT, LIMITED or UNLIMITED")
        value = optional_positive(value_name) if mode == "LIMITED" else None
        if mode == "LIMITED" and value is None:
            raise ValueError(f"{value_name} is required in LIMITED mode")
        return mode, value

    daily_mode, daily_limit = limit_pair("daily_limit_mode", "daily_success_limit")
    unanswered_mode, unanswered_limit = limit_pair(
        "unanswered_limit_mode", "max_consecutive_unanswered"
    )
    patch = {
        "proactive_enabled": optional_bool("proactive_enabled"),
        "check_min_minutes": optional_positive("check_min_minutes"),
        "check_max_minutes": optional_positive("check_max_minutes"),
        "quiet_enabled": optional_bool("quiet_enabled"),
        "quiet_start": optional_time("quiet_start"),
        "quiet_end": optional_time("quiet_end"),
        "min_success_gap_minutes": optional_positive("min_success_gap_minutes"),
        "daily_limit_mode": daily_mode,
        "daily_success_limit": daily_limit,
        "unanswered_limit_mode": unanswered_mode,
        "max_consecutive_unanswered": unanswered_limit,
        "failure_mode": _optional_failure_mode(supplied),
        "retry_delay_minutes": optional_positive("retry_delay_minutes"),
        "retry_max_attempts": optional_positive("retry_max_attempts"),
    }
    _validate_effective_contact_interval(patch, effective)
    return patch


if TYPE_CHECKING:
    from astrbot.api.star import Context

    from ....features.group_flow.ports import GroupFlowRepository


def _policy_view(value: Any, *boolean_fields: str) -> dict[str, Any]:
    """Serialize one SQLite policy without leaking integer booleans to editors."""

    rendered = jsonable(value) if value is not None else {}
    if not isinstance(rendered, dict):
        raise TypeError("policy view must be an object")
    for field in boolean_fields:
        if field in rendered and rendered[field] is not None:
            rendered[field] = bool(rendered[field])
    return rendered


class ProfileSettingsController(InstanceOverrideActionsMixin):
    validate_group_flow_patch = staticmethod(validate_group_flow_patch)

    def __init__(
        self,
        profiles_repository: ProfilesRepositoryPort,
        timeline_repository: TimelineRepositoryPort,
        context: Context,
        profiles: ProfilesAdminController,
        scope_configuration: ScopeConfigurationTransactionPort,
        group_flow_repository: GroupFlowRepository | None = None,
    ) -> None:
        self.profiles_repository = profiles_repository
        self.timeline_repository = timeline_repository
        self.context = context
        self.profiles = profiles
        self.scope_configuration = scope_configuration
        self.group_flow_repository = group_flow_repository

    def validate_role_patch(
        self,
        payload: Mapping[str, Any],
        *,
        scope: str = "private",
        thinking_policy: MainCoreThinkingPolicy | None = None,
    ) -> dict[str, Any]:
        for field in ("proactive_enabled",):
            if field in payload and not isinstance(payload[field], bool):
                raise ValueError(f"{field} must be a boolean")
        minimum = int(payload.get("min_wakeup_minutes", 15))
        maximum = int(payload.get("max_wakeup_minutes", 55))
        self._validate_interval(minimum, maximum, "wake interval")
        low_minimum = int(payload.get("low_frequency_min_wakeup_minutes", 180))
        low_maximum = int(payload.get("low_frequency_max_wakeup_minutes", 480))
        self._validate_interval(low_minimum, low_maximum, "low-frequency wake interval")
        selected_policy = thinking_policy or thinking_policy_from_value(None)
        max_context_tokens = int(
            payload.get("max_context_tokens", selected_policy.max_context_tokens)
        )
        target_context_tokens = int(
            payload.get("target_context_tokens", selected_policy.target_context_tokens)
        )
        media_original_retention_days = int(payload.get("media_original_retention_days", 30))
        if max_context_tokens < 128000:
            raise ValueError("max_context_tokens must be >= 128000")
        if target_context_tokens < 20000:
            raise ValueError("target_context_tokens must be >= 20000")
        if not 0 <= media_original_retention_days <= 3650:
            raise ValueError("media_original_retention_days must be between 0 and 3650")
        target_context_tokens = min(target_context_tokens, max_context_tokens)
        return {
            "proactive_enabled": bool(payload.get("proactive_enabled", True)),
            "extra_background": self._text(payload, "extra_background"),
            "world_texture_prompt": self._text(payload, "world_texture_prompt"),
            "media_original_retention_days": media_original_retention_days,
            "min_wakeup_minutes": minimum,
            "max_wakeup_minutes": maximum,
            "low_frequency_min_wakeup_minutes": low_minimum,
            "low_frequency_max_wakeup_minutes": low_maximum,
            "max_context_tokens": max_context_tokens,
            "target_context_tokens": target_context_tokens,
        }

    def validate_contact_patch(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        minimum = int(self._first(payload, "check_min_minutes", default=180))
        maximum = int(self._first(payload, "check_max_minutes", default=480))
        self._validate_interval(minimum, maximum, "ContactClock interval")
        quiet_start = self._text(payload, "quiet_start", default="23:00")
        quiet_end = self._text(payload, "quiet_end", default="08:00")
        for name, value in (("quiet_start", quiet_start), ("quiet_end", quiet_end)):
            if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
                raise ValueError(f"{name} must use HH:MM")
        timezone_name = self._text(payload, "timezone")
        if timezone_name:
            try:
                zoneinfo.ZoneInfo(timezone_name)
            except Exception as exc:
                raise ValueError("timezone must be a valid IANA timezone") from exc

        def optional_positive(name: str, default: int) -> int | None:
            if name not in payload:
                return default
            value = payload.get(name)
            if value in (None, ""):
                return None
            parsed = int(value)
            if parsed < 1:
                raise ValueError(f"{name} must be null or a positive integer")
            return parsed

        min_gap = int(self._first(payload, "min_success_gap_minutes", default=180))
        retry_delay = int(self._first(payload, "retry_delay_minutes", default=15))
        retry_attempts = int(self._first(payload, "retry_max_attempts", default=3))
        if min(min_gap, retry_delay, retry_attempts) < 1:
            raise ValueError("contact gap and retry values must be positive integers")
        failure_mode = self._text(payload, "failure_mode", default="SKIP").upper()
        if failure_mode not in {"SKIP", "RETRY_BACKOFF"}:
            raise ValueError("failure_mode must be SKIP or RETRY_BACKOFF")
        return {
            "proactive_enabled": bool(payload.get("proactive_enabled", True)),
            "check_min_minutes": minimum,
            "check_max_minutes": maximum,
            "quiet_enabled": bool(payload.get("quiet_enabled", True)),
            "quiet_start": quiet_start,
            "quiet_end": quiet_end,
            "timezone": timezone_name or None,
            "min_success_gap_minutes": min_gap,
            "daily_success_limit": optional_positive("daily_success_limit", 3),
            "max_consecutive_unanswered": optional_positive("max_consecutive_unanswered", 1),
            "failure_mode": failure_mode,
            "retry_delay_minutes": retry_delay,
            "retry_max_attempts": retry_attempts,
        }

    @staticmethod
    def _validate_interval(minimum: int, maximum: int, label: str) -> None:
        if minimum < 1:
            raise ValueError(f"{label} must satisfy 1 <= min <= max")
        if maximum < minimum:
            raise ValueError(f"{label} must satisfy 1 <= min <= max")

    @staticmethod
    def _first(payload: Mapping[str, Any], *names: str, default: Any) -> Any:
        for name in names:
            value = payload.get(name)
            if value not in (None, ""):
                return value
        return default

    @classmethod
    def _text(cls, payload: Mapping[str, Any], name: str, *, default: str = "") -> str:
        return str(cls._first(payload, name, default=default)).strip()

    @classmethod
    def _optional_text(cls, payload: Mapping[str, Any], *names: str) -> str | None:
        value = str(cls._first(payload, *names, default="")).strip()
        return value or None

    async def save_scope_configuration(
        self, profile_id: str, scope: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Atomically save the scope template and all scope-owned policies."""

        profile = await self.profiles_repository.get_profile(profile_id)
        if profile is None:
            raise ValueError(f"unknown profile: {profile_id}")
        role_patch = self.validate_role_patch(
            payload,
            scope=scope,
            thinking_policy=thinking_policy_from_value(profile.thinking_complexity),
        )
        contact_patch = self.validate_contact_patch(payload)
        group_limit = int(self._first(payload, "group_send_qpm_limit", default=20))
        if group_limit < 1:
            raise ValueError("group_send_qpm_limit must be a positive integer")
        send_limit = int(self._first(payload, "send_qpm_limit", default=20))
        if send_limit < 1:
            raise ValueError("send_qpm_limit must be a positive integer")
        group_flow_patch = self.validate_group_flow_patch(payload) if scope == "group" else None
        current_config = await self.profiles_repository.get_scope_config(profile_id, scope)
        if current_config is None:
            raise ValueError(f"unknown scope configuration for profile {profile_id}: {scope}")
        expected_scope_version = self._required_version(
            payload,
            "scope_config_version",
            "scope configuration",
        )
        current_scope_version = await self.profiles_repository.get_scope_config_version(
            profile_id, scope
        )
        if expected_scope_version != current_scope_version:
            raise ValueError("scope configuration changed; reload before saving")
        current = await self._scope_resources(profile_id, scope)
        expected = self._expected_versions(payload, current)
        self._validate_versions(current, expected)
        max_gate_hours = int(self._first(payload, "state_gate_max_hours", default=24))
        if not 1 <= max_gate_hours <= 24:
            raise ValueError("state_gate_max_hours must be between 1 and 24")
        await self.scope_configuration.save_scope_configuration(
            ScopeConfigurationUpdate(
                profile_id=profile_id,
                scope=scope,
                role=role_patch,
                contact={key: value for key, value in contact_patch.items() if key != "timezone"},
                timezone=contact_patch["timezone"],
                delivery={
                    "group_send_qpm_limit": group_limit,
                    "send_qpm_limit": send_limit,
                },
                state_gate={
                    "enabled": bool(payload.get("state_message_gate_enabled", False)),
                    "silent_enabled": bool(payload.get("state_message_silent_enabled", False)),
                    "max_gate_hours": max_gate_hours,
                },
                group_flow=group_flow_patch,
                expected_scope_version=expected_scope_version,
                expected_contact_version=expected["contact"],
                expected_timezone_version=expected["timezone"],
                expected_delivery_version=expected["delivery"],
                expected_state_gate_version=expected["gate"],
                expected_group_flow_version=expected.get("group_flow"),
            )
        )
        config = await self.profiles_repository.get_scope_config(profile_id, scope)
        assert config is not None
        snapshot = await self.profiles.scope_config_snapshot(profile_id, scope)
        await record_event(
            self.timeline_repository,
            profile_id=profile_id,
            level="INFO",
            category="configuration",
            message="生活、联系与投递配置已更新",
            details={"scope": scope},
        )
        return {"ok": True, **snapshot, "role_config": jsonable(config)}

    async def _scope_resources(self, profile_id: str, scope: str) -> dict[str, Mapping[str, Any]]:
        resources = {
            "contact": await self.timeline_repository.get_contact_policy(profile_id, scope),
            "delivery": await self.timeline_repository.get_delivery_policy(profile_id, scope),
            "gate": await self.timeline_repository.get_state_gate_policy(profile_id, scope),
            "timezone": await self.timeline_repository.get_profile_timezone(profile_id),
        }
        if scope == "group" and self.group_flow_repository is not None:
            resources["group_flow"] = jsonable(
                await self.group_flow_repository.get_group_flow_policy(profile_id, scope)
            )
        if any(value is None for value in resources.values()):
            raise RuntimeError("batch-one policy storage is unavailable")
        return resources  # type: ignore[return-value]

    @classmethod
    def _expected_versions(
        cls,
        payload: Mapping[str, Any],
        current: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, int]:
        keys = {
            "contact": "contact_policy_version",
            "delivery": "delivery_policy_version",
            "timezone": "timezone_version",
            "gate": "state_gate_policy_version",
        }
        if "group_flow" in current:
            keys["group_flow"] = "group_flow_policy_version"
        return {
            name: cls._required_version(payload, payload_key, name.replace("_", " "))
            for name, payload_key in keys.items()
        }

    @staticmethod
    def _required_version(
        payload: Mapping[str, Any],
        key: str,
        label: str,
    ) -> int:
        value = payload.get(key)
        if value in (None, "") or isinstance(value, bool):
            raise ValueError(f"{label} revision is required; reload before saving")
        version = int(value)
        if version < 1:
            raise ValueError(f"{label} revision is invalid; reload before saving")
        return version

    @staticmethod
    def _validate_versions(
        current: Mapping[str, Mapping[str, Any]], expected: Mapping[str, int]
    ) -> None:
        errors = {
            "contact": "contact policy changed; reload before saving",
            "delivery": "delivery policy changed; reload before saving",
            "timezone": "profile timezone changed; reload before saving",
            "gate": "state gate policy changed; reload before saving",
            "group_flow": "group flow policy changed; reload before saving",
        }
        for name, message in errors.items():
            if name not in current:
                continue
            actual = int(current[name].get("version") or 0)
            if expected[name] != actual:
                raise ValueError(message)

    async def instance_contact_override_snapshot(
        self, profile_id: str, instance_id: str
    ) -> dict[str, Any]:
        """Return only the sparse Contact exception and its resolved policy."""

        assert self.timeline_repository is not None
        override = await self.timeline_repository.get_instance_contact_override(
            profile_id, instance_id
        )
        effective = await self.timeline_repository.resolve_contact_policy(profile_id, instance_id)
        gate_override = await self.timeline_repository.get_instance_state_gate_override(
            profile_id, instance_id
        )
        effective_gate = await self.timeline_repository.resolve_state_gate_policy(
            profile_id, instance_id
        )
        delivery_override = await self.timeline_repository.get_instance_delivery_override(
            profile_id, instance_id
        )
        effective_delivery = await self.timeline_repository.resolve_expression_pacing_policy(
            profile_id, instance_id
        )
        chat_policy = await self.profiles_repository.get_instance_chat_policy(
            profile_id, instance_id
        )
        return {
            "profile_id": profile_id,
            "instance_id": instance_id,
            "scope": str(effective.get("scope") or ""),
            "chat_policy": _policy_view(
                chat_policy,
                "soulcore_enabled",
                "image_send_enabled",
                "private_name_override_enabled",
            ),
            "chat_policy_version": int(chat_policy.version),
            "override": _policy_view(override, "proactive_enabled", "quiet_enabled"),
            "override_version": int((override or {}).get("version") or 0),
            "effective": _policy_view(effective, "proactive_enabled", "quiet_enabled"),
            "state_gate_override": _policy_view(gate_override, "enabled", "silent_enabled"),
            "state_gate_override_version": int((gate_override or {}).get("version") or 0),
            "effective_state_gate": _policy_view(effective_gate, "enabled", "silent_enabled"),
            "delivery_override": jsonable(delivery_override) if delivery_override else {},
            "delivery_override_version": int((delivery_override or {}).get("version") or 0),
            "effective_delivery": jsonable(effective_delivery),
        }

    @staticmethod
    def validate_instance_chat_policy(payload: Mapping[str, Any], *, scope: str) -> dict[str, Any]:
        """Validate administrator-owned settings for one chat route."""

        supplied = payload.get("policy")
        if not isinstance(supplied, Mapping):
            raise ValueError("policy must be a JSON object")
        boolean_fields = {"soulcore_enabled", "image_send_enabled"}
        allowed = boolean_fields | {
            "private_fallback_player_name",
            "private_name_override_enabled",
        }
        unknown = set(supplied) - allowed
        if unknown:
            raise ValueError(f"unsupported instance chat fields: {sorted(unknown)}")
        missing = boolean_fields - set(supplied)
        if missing:
            raise ValueError(f"instance chat policy requires: {sorted(missing)}")
        for field in boolean_fields:
            if not isinstance(supplied[field], bool):
                raise ValueError(f"{field} must be a boolean")
        raw_override_enabled = supplied.get("private_name_override_enabled", False)
        if not isinstance(raw_override_enabled, bool):
            raise ValueError("private_name_override_enabled must be a boolean")
        fallback_name = normalize_private_fallback_player_name(
            supplied.get("private_fallback_player_name")
        )
        override_enabled = bool(raw_override_enabled)
        if scope != "private" and (fallback_name or override_enabled):
            raise ValueError("private chat names are only available for private chats")
        if override_enabled and not fallback_name:
            raise ValueError("private name override requires a configured private chat name")
        return {
            **{field: bool(supplied[field]) for field in boolean_fields},
            "private_fallback_player_name": fallback_name,
            "private_name_override_enabled": override_enabled,
        }

    async def save_instance_chat_policy(
        self,
        profile_id: str,
        instance_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """CAS-save permissions and the private display-name preference for one chat."""

        instance = await self.profiles_repository.get_character_instance(profile_id, instance_id)
        if instance is None:
            raise KeyError((profile_id, instance_id))
        policy = self.validate_instance_chat_policy(payload, scope=str(instance.scope))
        raw_version = payload.get("expected_version")
        if raw_version in (None, "") or isinstance(raw_version, bool):
            raise ValueError("instance chat policy revision is required; reload before saving")
        expected_version = int(raw_version)
        if expected_version < 0:
            raise ValueError("instance chat policy revision is invalid; reload before saving")
        saved = await self.profiles_repository.upsert_instance_chat_policy(
            profile_id,
            instance_id,
            soulcore_enabled=policy["soulcore_enabled"],
            image_send_enabled=policy["image_send_enabled"],
            expected_version=expected_version,
            private_fallback_player_name=policy["private_fallback_player_name"],
            private_name_override_enabled=policy["private_name_override_enabled"],
        )
        if saved is None:
            raise ValueError("instance chat policy changed; reload before saving")
        await record_event(
            self.timeline_repository,
            profile_id=profile_id,
            instance_id=instance_id,
            level="INFO",
            category="configuration",
            message="当前聊天设置已更新",
            details={
                "soulcore_enabled": saved.soulcore_enabled,
                "image_send_enabled": saved.image_send_enabled,
                "private_name_override_enabled": saved.private_name_override_enabled,
            },
        )
        snapshot = await self.instance_contact_override_snapshot(profile_id, instance_id)
        return {"ok": True, **snapshot}

    @staticmethod
    def validate_instance_contact_override(
        payload: Mapping[str, Any], effective: Mapping[str, Any]
    ) -> dict[str, Any]:
        return validate_instance_contact_override(payload, effective)

    async def save_instance_contact_override(
        self, profile_id: str, instance_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """CAS-save or remove one instance's Contact-only exception."""

        current, current_gate, current_delivery = await self._instance_override_resources(
            profile_id, instance_id
        )
        expected, expected_gate, expected_delivery = self._instance_override_versions(
            payload, current, current_gate, current_delivery
        )
        await self._apply_instance_override_action(
            profile_id,
            instance_id,
            payload,
            current,
            current_gate,
            current_delivery,
            expected,
            expected_gate,
            expected_delivery,
        )
        snapshot = await self.instance_contact_override_snapshot(profile_id, instance_id)
        return {"ok": True, **snapshot}

    async def _delete_instance_overrides(
        self,
        profile_id: str,
        instance_id: str,
        current: Any,
        current_gate: Any,
        current_delivery: Any,
        expected: int,
        expected_gate: int,
        expected_delivery: int,
    ) -> None:
        if current is not None:
            deleted = await self.timeline_repository.delete_instance_contact_override(
                profile_id, instance_id, expected_version=expected
            )
            if not deleted:
                raise ValueError("instance Contact override changed; reload before saving")
        if current_gate is not None:
            deleted = await self.timeline_repository.delete_instance_state_gate_override(
                profile_id, instance_id, expected_version=expected_gate
            )
            if not deleted:
                raise ValueError("instance state gate override changed; reload before saving")
        if current_delivery is not None:
            deleted = await self.timeline_repository.delete_instance_delivery_override(
                profile_id, instance_id, expected_version=expected_delivery
            )
            if not deleted:
                raise ValueError("instance delivery override changed; reload before saving")

    async def _save_instance_overrides(
        self,
        profile_id: str,
        instance_id: str,
        payload: Mapping[str, Any],
        current_delivery: Any,
        expected: int,
        expected_gate: int,
        expected_delivery: int,
    ) -> None:
        inherited = await self._inherited_contact_policy(profile_id, instance_id)
        patch = self.validate_instance_contact_override(payload, inherited)
        saved = await self.timeline_repository.upsert_instance_contact_override(
            profile_id, instance_id, patch, expected_version=expected
        )
        if saved is None:
            raise ValueError("instance Contact override changed; reload before saving")
        gate_patch = self._state_gate_patch(payload.get("state_gate_override"))
        saved_gate = await self.timeline_repository.upsert_instance_state_gate_override(
            profile_id, instance_id, gate_patch, expected_version=expected_gate
        )
        if saved_gate is None:
            raise ValueError("instance state gate override changed; reload before saving")
        if "delivery_override" not in payload:
            return
        raw_delivery = payload.get("delivery_override")
        if not isinstance(raw_delivery, Mapping):
            raise ValueError("delivery_override must be a JSON object")
        unknown_delivery = set(raw_delivery) - {
            "send_qpm_limit",
        }
        if unknown_delivery:
            raise ValueError(f"unsupported instance delivery fields: {sorted(unknown_delivery)}")
        send_limit = optional_qpm(raw_delivery, "send_qpm_limit")
        if send_limit is None:
            if current_delivery is not None:
                deleted = await self.timeline_repository.delete_instance_delivery_override(
                    profile_id, instance_id, expected_version=expected_delivery
                )
                if not deleted:
                    raise ValueError("instance delivery override changed; reload before saving")
        else:
            saved_delivery = await self.timeline_repository.upsert_instance_delivery_override(
                profile_id,
                instance_id,
                send_qpm_limit=send_limit,
                expected_version=expected_delivery,
            )
            if saved_delivery is None:
                raise ValueError("instance delivery override changed; reload before saving")

    async def _inherited_contact_policy(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        effective = await self.timeline_repository.resolve_contact_policy(profile_id, instance_id)
        scope = str(effective.get("scope") or "")
        inherited = await self.timeline_repository.get_contact_policy(profile_id, scope)
        if inherited is None:
            raise ValueError("base Contact policy is unavailable")
        result = dict(inherited)
        platform_id = str(effective.get("platform_instance_id") or "")
        if not platform_id:
            return result
        platform = await self.timeline_repository.get_platform_contact_policy(
            profile_id, scope, platform_id
        )
        if platform is None:
            return result
        ordinary = {
            "proactive_enabled",
            "check_min_minutes",
            "check_max_minutes",
            "quiet_enabled",
            "quiet_start",
            "quiet_end",
            "min_success_gap_minutes",
            "failure_mode",
            "retry_delay_minutes",
            "retry_max_attempts",
        }
        for name in ordinary:
            if platform.get(name) is not None:
                result[name] = platform[name]
        for mode_name, value_name in (
            ("daily_limit_mode", "daily_success_limit"),
            ("unanswered_limit_mode", "max_consecutive_unanswered"),
        ):
            if platform.get(mode_name) not in (None, "INHERIT"):
                result[mode_name] = platform[mode_name]
                result[value_name] = platform.get(value_name)
        return result

    @staticmethod
    def _state_gate_patch(raw: Any) -> dict[str, Any]:
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError("state_gate_override must be a JSON object")
        unknown = set(raw) - {"enabled", "silent_enabled", "max_gate_hours"}
        if unknown:
            raise ValueError(f"unsupported state gate override fields: {sorted(unknown)}")
        patch: dict[str, Any] = {}
        for name in ("enabled", "silent_enabled"):
            value = raw.get(name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean or null")
            patch[name] = value
        hours = raw.get("max_gate_hours")
        if hours in (None, ""):
            patch["max_gate_hours"] = None
        else:
            parsed = int(hours)
            if not 1 <= parsed <= 24:
                raise ValueError("max_gate_hours must be between 1 and 24 or null")
            patch["max_gate_hours"] = parsed
        return patch

    async def platform_contact_policy_snapshot(
        self, profile_id: str, instance: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Return the selected instance's platform-connection policy."""

        assert self.timeline_repository is not None
        instance_id = str(instance.get("instance_id") or "")
        scope = str(instance.get("scope") or "")
        platform_id = str(instance.get("platform_id") or "").strip()
        if not platform_id:
            raise ValueError("selected instance has no platform connection id")
        policy = await self.timeline_repository.get_platform_contact_policy(
            profile_id, scope, platform_id
        )
        effective = await self.timeline_repository.resolve_contact_policy(
            profile_id, instance_id, platform_instance_id=platform_id
        )
        return {
            "profile_id": profile_id,
            "instance_id": instance_id,
            "scope": scope,
            "platform_instance_id": platform_id,
            "policy": _policy_view(policy, "proactive_enabled", "quiet_enabled"),
            "policy_version": int((policy or {}).get("version") or 0),
            "effective": _policy_view(effective, "proactive_enabled", "quiet_enabled"),
        }

    async def save_platform_contact_policy(
        self,
        profile_id: str,
        instance: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """CAS-save one platform connection's Contact and delivery override."""

        assert self.timeline_repository is not None
        scope = str(instance.get("scope") or "")
        platform_id = str(instance.get("platform_id") or "").strip()
        if not platform_id:
            raise ValueError("selected instance has no platform connection id")
        current = await self.timeline_repository.get_platform_contact_policy(
            profile_id, scope, platform_id
        )
        expected_version = int(payload.get("expected_version") or 0)
        if expected_version != int((current or {}).get("version") or 0):
            raise ValueError("platform policy changed; reload before saving")
        action = str(payload.get("action") or "save").strip().lower()
        if action == "delete":
            await self._delete_platform_contact_policy(
                profile_id, scope, platform_id, expected_version, current
            )
        elif action == "save":
            await self._save_platform_contact_policy(
                profile_id, scope, platform_id, expected_version, payload
            )
        else:
            raise ValueError("action must be save or delete")
        snapshot = await self.platform_contact_policy_snapshot(profile_id, instance)
        return {"ok": True, **snapshot}

    async def _delete_platform_contact_policy(
        self,
        profile_id: str,
        scope: str,
        platform_id: str,
        expected_version: int,
        current: Mapping[str, Any] | None,
    ) -> None:
        if current is None:
            return
        assert self.timeline_repository is not None
        deleted = await self.timeline_repository.delete_platform_contact_policy(
            profile_id, scope, platform_id, expected_version=expected_version
        )
        if not deleted:
            raise ValueError("platform policy changed; reload before saving")

    async def _save_platform_contact_policy(
        self,
        profile_id: str,
        scope: str,
        platform_id: str,
        expected_version: int,
        payload: Mapping[str, Any],
    ) -> None:
        assert self.timeline_repository is not None
        supplied = payload.get("policy")
        if not isinstance(supplied, Mapping):
            raise ValueError("policy must be a JSON object")
        contact_fields = self._platform_contact_fields()
        allowed = contact_fields | {
            "template_id",
            "group_send_qpm_limit",
            "account_send_qpm_limit",
            "send_qpm_limit",
        }
        unknown = set(supplied) - allowed
        if unknown:
            raise ValueError(f"unsupported platform policy fields: {sorted(unknown)}")
        base = await self.timeline_repository.get_contact_policy(profile_id, scope)
        if base is None:
            raise ValueError("base Contact policy is unavailable")
        patch = self._platform_policy_patch(supplied, contact_fields, base)
        saved = await self.timeline_repository.upsert_platform_contact_policy(
            profile_id, scope, platform_id, patch, expected_version=expected_version
        )
        if saved is None:
            raise ValueError("platform policy changed; reload before saving")

    def _platform_policy_patch(
        self,
        supplied: Mapping[str, Any],
        contact_fields: set[str],
        base: Mapping[str, Any],
    ) -> dict[str, Any]:
        contact_patch = self.validate_instance_contact_override(
            {"override": {name: supplied.get(name) for name in contact_fields}}, base
        )
        return {
            **contact_patch,
            "template_id": str(supplied.get("template_id") or "").strip() or None,
            "group_send_qpm_limit": optional_qpm(supplied, "group_send_qpm_limit"),
            "account_send_qpm_limit": optional_qpm(supplied, "account_send_qpm_limit"),
            "send_qpm_limit": optional_qpm(supplied, "send_qpm_limit"),
        }

    @staticmethod
    def _platform_contact_fields() -> set[str]:
        return {
            "proactive_enabled",
            "check_min_minutes",
            "check_max_minutes",
            "quiet_enabled",
            "quiet_start",
            "quiet_end",
            "min_success_gap_minutes",
            "daily_limit_mode",
            "daily_success_limit",
            "unanswered_limit_mode",
            "max_consecutive_unanswered",
            "failure_mode",
            "retry_delay_minutes",
            "retry_max_attempts",
        }
