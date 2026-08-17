"""Ephemeral user-facing feedback for passive turns that produce no reply."""

from __future__ import annotations

from typing import Any

from ...contracts.models import CoreRunResult, RunStatus
from ...contracts.system_notice import soulcore_system_notice
from ...features.timeline.state_gate import StateGateDecision, StateGateDisposition
from ...features.timeline.temporary_absence import TEMPORARY_ABSENCE_REASON_CODE

NO_REPLY_NOTICE = soulcore_system_notice("对方选择不回复。")
TEMPORARY_ABSENCE_NOTICE = soulcore_system_notice("对方选择暂时离开，当前不回复。")
TEMPORARY_ABSENCE_DEFERRED_NOTICE = soulcore_system_notice(
    "对方暂时离开，这条消息会在暂离结束后再处理。"
)
DEFERRED_REPLY_NOTICE = soulcore_system_notice("对方当前无法回复，这条消息会稍后处理。")


def main_core_no_reply_notice(result: CoreRunResult) -> str:
    """Describe only an explicit, completed no-output terminal decision."""

    if result.status is not RunStatus.COMPLETED or not result.silent or result.had_output:
        return ""
    if str(result.silence_reason or "").upper() == "TEMPORARY_ABSENCE":
        return TEMPORARY_ABSENCE_NOTICE
    return NO_REPLY_NOTICE


def state_gate_no_reply_notice(decision: StateGateDecision) -> str:
    """Explain a state gate that stops a user-triggered turn before Main Core."""

    disposition = decision.disposition
    reason_code = str(decision.reason_code or "")
    if disposition is StateGateDisposition.DEFER:
        return (
            TEMPORARY_ABSENCE_DEFERRED_NOTICE
            if reason_code == TEMPORARY_ABSENCE_REASON_CODE
            else DEFERRED_REPLY_NOTICE
        )
    if disposition is StateGateDisposition.SILENT:
        return (
            TEMPORARY_ABSENCE_NOTICE
            if reason_code == TEMPORARY_ABSENCE_REASON_CODE
            else NO_REPLY_NOTICE
        )
    return ""


async def send_ephemeral_passive_notice(
    *,
    profiles: Any,
    delivery: Any,
    event: Any,
    captured: Any,
    profile_id: str,
    instance_id: str,
    configured_group_limit: int,
    text: str,
) -> bool:
    """Send a notice directly to the platform without creating a conversation row."""

    if not text or not await profiles.get_profile_passive_no_reply_notice_enabled(profile_id):
        return False
    await delivery.send(
        captured,
        event.plain_result(text),
        profile_id=profile_id,
        instance_id=instance_id,
        configured_group_limit=configured_group_limit,
        proactive=False,
    )
    return True


__all__ = [
    "DEFERRED_REPLY_NOTICE",
    "NO_REPLY_NOTICE",
    "TEMPORARY_ABSENCE_DEFERRED_NOTICE",
    "TEMPORARY_ABSENCE_NOTICE",
    "main_core_no_reply_notice",
    "send_ephemeral_passive_notice",
    "state_gate_no_reply_notice",
]
