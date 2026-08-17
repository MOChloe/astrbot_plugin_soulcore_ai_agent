"""Cross-feature contract for one conversation instance's first initialization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import InstanceInitializationState
from .system_notice import soulcore_system_notice

INSTANCE_INITIALIZATION_METADATA_KEY = "instance_initialization"
INSTANCE_INITIALIZATION_STARTED_NOTICE_KIND = "INSTANCE_INITIALIZATION_STARTED"
INSTANCE_INITIALIZATION_PROGRESS_NOTICE_KIND = "INSTANCE_INITIALIZATION_PROGRESS"
INSTANCE_INITIALIZATION_READY_NOTICE_KIND = "INSTANCE_INITIALIZATION_READY"
SYSTEM_NOTICE_KIND_KEY = "system_notice_kind"
INSTANCE_INITIALIZATION_TOTAL_STEPS = 5


def instance_initialization_progress_notice(completed_steps: int) -> str:
    completed = int(completed_steps)
    if not 1 <= completed <= INSTANCE_INITIALIZATION_TOTAL_STEPS:
        raise ValueError(
            f"initialization progress must be between 1 and {INSTANCE_INITIALIZATION_TOTAL_STEPS}"
        )
    percentage = completed * 100 // INSTANCE_INITIALIZATION_TOTAL_STEPS
    if completed == INSTANCE_INITIALIZATION_TOTAL_STEPS:
        return soulcore_system_notice(f"初始化完成：{percentage}%。可以开始聊天了。")
    return soulcore_system_notice(f"初始化进度：{percentage}%。")


INSTANCE_INITIALIZATION_STARTED_NOTICE = soulcore_system_notice(
    "正在初始化：0%。完成后会通知你。若由私聊触发，触发初始化的消息会在完成后继续处理；"
    "群聊触发消息及初始化期间的新消息不会被处理，请稍候。"
)
INSTANCE_INITIALIZATION_READY_NOTICE = instance_initialization_progress_notice(5)


def initialization_metadata(
    state: InstanceInitializationState,
    *,
    started_at: str = "",
    completed_at: str = "",
) -> dict[str, str]:
    value = {"state": state.value}
    if started_at:
        value["started_at"] = str(started_at)
    if completed_at:
        value["completed_at"] = str(completed_at)
    return value


def initialization_state_from_payload(
    payload: Mapping[str, Any] | None,
) -> InstanceInitializationState | None:
    if not isinstance(payload, Mapping):
        return None
    marker = payload.get(INSTANCE_INITIALIZATION_METADATA_KEY)
    if not isinstance(marker, Mapping):
        return None
    try:
        return InstanceInitializationState(str(marker.get("state") or ""))
    except ValueError:
        return None


def is_initialization_request(payload: Mapping[str, Any] | None) -> bool:
    return initialization_state_from_payload(payload) is InstanceInitializationState.INITIALIZING


__all__ = [
    "INSTANCE_INITIALIZATION_METADATA_KEY",
    "INSTANCE_INITIALIZATION_PROGRESS_NOTICE_KIND",
    "INSTANCE_INITIALIZATION_READY_NOTICE",
    "INSTANCE_INITIALIZATION_READY_NOTICE_KIND",
    "INSTANCE_INITIALIZATION_STARTED_NOTICE",
    "INSTANCE_INITIALIZATION_STARTED_NOTICE_KIND",
    "INSTANCE_INITIALIZATION_TOTAL_STEPS",
    "SYSTEM_NOTICE_KIND_KEY",
    "initialization_metadata",
    "instance_initialization_progress_notice",
    "initialization_state_from_payload",
    "is_initialization_request",
]
