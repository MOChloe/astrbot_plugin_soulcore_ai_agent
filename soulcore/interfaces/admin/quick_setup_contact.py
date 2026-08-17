"""Natural-language guided contact presets over the existing role defaults."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ...features.profiles.ports import ProfilesRepositoryPort
from ...features.timeline.ports import TimelineRepositoryPort

CONTACT_PRESETS: dict[str, dict[str, int]] = {
    "occasional": {
        "check_min_minutes": 240,
        "check_max_minutes": 480,
        "min_success_gap_minutes": 360,
        "daily_success_limit": 2,
    },
    "natural": {
        "check_min_minutes": 120,
        "check_max_minutes": 240,
        "min_success_gap_minutes": 180,
        "daily_success_limit": 4,
    },
    "frequent": {
        "check_min_minutes": 60,
        "check_max_minutes": 120,
        "min_success_gap_minutes": 90,
        "daily_success_limit": 8,
    },
}
QUICK_SETUP_MAX_CONSECUTIVE_UNANSWERED = 2

_CONTACT_VIEW_FIELDS = (
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
)
_TIME_PATTERN = re.compile(r"(?:[01]\d|2[0-3]):(?:00|15|30|45)")


async def quick_setup_contact_snapshot(
    timeline: TimelineRepositoryPort,
    profiles: ProfilesRepositoryPort,
    profile_id: str,
) -> dict[str, Any]:
    private = await timeline.get_contact_policy(profile_id, "private")
    group = await timeline.get_contact_policy(profile_id, "group")
    timezone = await timeline.get_profile_timezone(profile_id)
    private_scope = await profiles.get_scope_config(profile_id, "private")
    group_scope = await profiles.get_scope_config(profile_id, "group")
    if private is None or group is None or private_scope is None or group_scope is None:
        raise ValueError("暂时没有读到角色的联系设置，请重新连接")

    private_view = _contact_view(
        private,
        await profiles.get_scope_config_version(profile_id, "private"),
    )
    group_view = _contact_view(
        group,
        await profiles.get_scope_config_version(profile_id, "group"),
    )
    scope_mismatch = bool(private_scope.proactive_enabled) != bool(
        private_view["proactive_enabled"]
    ) or bool(group_scope.proactive_enabled) != bool(group_view["proactive_enabled"])
    mixed = scope_mismatch or any(
        private_view.get(name) != group_view.get(name) for name in _CONTACT_VIEW_FIELDS
    )
    return {
        "private": private_view,
        "group": group_view,
        "timezone": str(timezone.get("timezone") or ""),
        "timezone_version": int(timezone.get("version") or 0),
        "mixed": mixed,
        "mode": "advanced" if mixed else _preset_mode(private_view),
    }


async def configure_quick_setup_contact(
    timeline: TimelineRepositoryPort,
    profiles: ProfilesRepositoryPort,
    profile_id: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    mode = str(payload.get("mode") or "").strip().lower()
    if mode not in {"off", *CONTACT_PRESETS}:
        raise ValueError("请选择角色主动联系你的频率")
    expected = payload.get("expected_versions")
    if not isinstance(expected, Mapping):
        raise ValueError("联系设置版本缺失，请重新载入后再保存")

    timezone: str | None = None
    contact_patch: dict[str, Any] = {}
    if mode != "off":
        quiet = payload.get("quiet")
        if not isinstance(quiet, Mapping) or type(quiet.get("enabled")) is not bool:
            raise ValueError("请选择是否设置安静时间")
        quiet_enabled = bool(quiet["enabled"])
        quiet_start = _quiet_time(quiet.get("start"), "开始时间")
        quiet_end = _quiet_time(quiet.get("end"), "结束时间")
        if quiet_enabled and quiet_start == quiet_end:
            raise ValueError("安静时间的开始和结束不能相同")
        timezone = _timezone(payload.get("timezone"))
        preset = CONTACT_PRESETS[mode]
        contact_patch = {
            **preset,
            "quiet_enabled": quiet_enabled,
            "quiet_start": quiet_start,
            "quiet_end": quiet_end,
            "daily_limit_mode": "LIMITED",
            "unanswered_limit_mode": "LIMITED",
            "max_consecutive_unanswered": QUICK_SETUP_MAX_CONSECUTIVE_UNANSWERED,
        }

    result = await timeline.configure_quick_setup_contact(
        profile_id,
        proactive_enabled=mode != "off",
        contact_patch=contact_patch,
        timezone=timezone,
        expected_versions=expected,
    )
    if result == "conflict":
        raise ValueError("角色联系设置刚刚在别处发生了变化，请重新确认后再保存")
    snapshot = await quick_setup_contact_snapshot(timeline, profiles, profile_id)
    return {
        "ok": True,
        "applied": result == "applied",
        "contact": snapshot,
    }


def validate_quick_setup_contact(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("请先完成角色的联系习惯设置")
    if bool(value.get("mixed")) or str(value.get("mode") or "") == "advanced":
        raise ValueError("请先在快速引导中确认并统一角色的联系习惯")
    if str(value.get("mode") or "") not in {"off", *CONTACT_PRESETS}:
        raise ValueError("请先完成角色的联系习惯设置")


def _contact_view(value: Mapping[str, Any], scope_version: int) -> dict[str, Any]:
    return {
        **{name: value.get(name) for name in _CONTACT_VIEW_FIELDS},
        "proactive_enabled": bool(value.get("proactive_enabled")),
        "quiet_enabled": bool(value.get("quiet_enabled")),
        "version": int(value.get("version") or 0),
        "scope_version": scope_version,
    }


def _preset_mode(value: Mapping[str, Any]) -> str:
    if not bool(value.get("proactive_enabled")):
        return "off"
    if str(value.get("daily_limit_mode") or "").upper() != "LIMITED":
        return "advanced"
    if str(value.get("unanswered_limit_mode") or "").upper() != "LIMITED":
        return "advanced"
    unanswered_limit = int(value.get("max_consecutive_unanswered") or 0)
    if not 1 <= unanswered_limit <= QUICK_SETUP_MAX_CONSECUTIVE_UNANSWERED:
        return "advanced"
    for name, preset in CONTACT_PRESETS.items():
        if all(int(value.get(field) or 0) == expected for field, expected in preset.items()):
            return name
    return "advanced"


def _quiet_time(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not _TIME_PATTERN.fullmatch(text):
        raise ValueError(f"{label}必须按 15 分钟选择")
    return text


def _timezone(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("暂时没有读到你所在的时区，请重新打开页面")
    try:
        ZoneInfo(text)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError("浏览器提供的时区无法识别，请重新打开页面") from exc
    return text


__all__ = [
    "CONTACT_PRESETS",
    "QUICK_SETUP_MAX_CONSECUTIVE_UNANSWERED",
    "configure_quick_setup_contact",
    "quick_setup_contact_snapshot",
    "validate_quick_setup_contact",
]
