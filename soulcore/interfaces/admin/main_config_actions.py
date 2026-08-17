"""Validation adapter for the profile-wide runtime switches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...contracts.runtime_limits import require_response_polish_timeout_seconds

_SUPPORTED_SWITCHES = {
    "enabled",
    "turn_buffer_enabled",
    "image_generation_enabled",
    "response_polish_enabled",
    "passive_no_reply_notice_enabled",
    "file_artifacts_enabled",
}


def _optional_switch(payload: Mapping[str, Any], field: str) -> bool | None:
    return bool(payload[field]) if field in payload else None


async def handle_main_config_action(
    profiles: Any,
    profile_id: str,
    action: str,
    payload: Mapping[str, Any],
) -> Any:
    if action == "get_main_config":
        return await profiles.main_config_snapshot(profile_id)
    supplied_switches = _SUPPORTED_SWITCHES.intersection(payload)
    timeout_supplied = "response_polish_timeout_seconds" in payload
    if not supplied_switches and not timeout_supplied:
        raise ValueError(
            "enabled, turn_buffer_enabled, image_generation_enabled, "
            "response_polish_enabled, passive_no_reply_notice_enabled or "
            "file_artifacts_enabled, or response_polish_timeout_seconds must be supplied"
        )
    for field in supplied_switches:
        if not isinstance(payload.get(field), bool):
            raise ValueError(f"{field} must be a boolean")
    changes: dict[str, Any] = {}
    if timeout_supplied:
        changes["response_polish_timeout_seconds"] = require_response_polish_timeout_seconds(
            payload.get("response_polish_timeout_seconds")
        )
    return await profiles.save_main_config(
        profile_id,
        enabled=_optional_switch(payload, "enabled"),
        turn_buffer_enabled=_optional_switch(payload, "turn_buffer_enabled"),
        image_generation_enabled=_optional_switch(payload, "image_generation_enabled"),
        response_polish_enabled=_optional_switch(payload, "response_polish_enabled"),
        passive_no_reply_notice_enabled=_optional_switch(
            payload, "passive_no_reply_notice_enabled"
        ),
        file_artifacts_enabled=_optional_switch(payload, "file_artifacts_enabled"),
        **changes,
    )


__all__ = ["handle_main_config_action"]
