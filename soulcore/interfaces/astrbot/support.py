"""AstrBot-facing constants and small runtime helpers."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent
    from astrbot.api.star import Context

from ...contracts.models import CoreRunResult, OutboxStatus
from ...contracts.runtime_limits import (
    AI_OPERATION_TIMEOUT_SECONDS,
    FILE_ARTIFACT_PDF_TIMEOUT_SECONDS,
)
from ...contracts.system_notice import soulcore_system_notice
from .umo import CapturedUMO, RouteKind

# AstrBot uses this value as the persistent-data directory key.  The public
# plugin rename must not move existing installations onto an empty database.
PERSISTENT_DATA_NAMESPACE = "astrbot_plugin_soulcore"


def has_trusted_astrbot_command_marker(event: AstrMessageEvent) -> bool:
    """Accept only AstrBot's parsed-command marker, never command-like message text."""

    return bool(event.get_extra("handlers_parsed_params", {}))


def foreground_ai_error_message(result: CoreRunResult) -> str:
    """Project an internal failure into a short, actionable player message.

    Diagnostic fields deliberately never participate in this projection. Raw provider
    responses, status codes, model names and request parameters belong only in logs and
    administrator diagnostics.
    """

    code = str(result.error_code or "INTERNAL").upper()
    if code in {"AUTHENTICATION", "PERMISSION"}:
        return soulcore_system_notice(
            "SoulCore 现在没有权限使用负责生成回复的服务，所以这次没有回复。"
            "请联系管理员检查服务设置。"
        )
    if code == "QUOTA_EXHAUSTED":
        return soulcore_system_notice(
            "负责生成回复的服务已经没有可用额度，所以这次没有回复。请联系管理员处理后再试。"
        )
    if code == "BACKEND_NOT_FOUND":
        return soulcore_system_notice(
            "当前还没有设置好用于生成回复的服务，所以暂时无法聊天。请联系管理员完成设置。"
        )
    if code == "CIRCUIT_OPEN":
        return soulcore_system_notice(
            "负责生成回复的服务连续多次失败，SoulCore 正在等待它恢复。"
            "这次没有回复，请过一会儿再试。"
        )
    if code in {"RATE_LIMIT", "CAPACITY_BUSY"}:
        return soulcore_system_notice(
            "现在请求回复的人太多，服务暂时忙不过来。这次没有回复，请稍等一会儿再试。"
        )
    if code in {"NETWORK", "REMOTE_5XX"}:
        return soulcore_system_notice(
            "暂时联系不上负责生成回复的服务，所以这次没有回复。请稍后再试。"
        )
    if code == "TIMEOUT":
        return soulcore_system_notice(
            "这次生成回复花的时间太久，SoulCore 已经停止等待，所以没有回复。请稍后再试。"
        )
    if code == "CONTEXT_BUDGET":
        return soulcore_system_notice(
            "这条消息、附件和前面的对话加起来太长，当前服务无法一次处理完。"
            "请减少附件，或把内容分成几条较短的消息再发。"
        )
    if code == "SAFETY_REFUSAL":
        return soulcore_system_notice(
            "这条消息中有当前服务无法处理的内容，所以这次没有回复。请换一种更简短、直接的说法再试。"
        )
    if code == "EMPTY_OUTPUT":
        return soulcore_system_notice("这次没有生成任何可以发送的内容。请重新发送一次。")
    incompatible_service_errors = {
        "INVALID_REQUEST",
        "UNSUPPORTED_CAPABILITY",
        "ADAPTER_INCOMPATIBLE",
        "PROMPT_CACHE_MARKER_UNSUPPORTED",
    }
    if code in incompatible_service_errors:
        return soulcore_system_notice(
            "当前服务无法按 SoulCore 需要的方式处理这条消息，所以这次没有回复。"
            "请联系管理员调整或更换服务。"
        )
    unusable_output_errors = {
        "OUTPUT_CONTRACT",
        "MAIN_CORE_STEP_REJECTED_THREE_TIMES",
    }
    if code in unusable_output_errors:
        return soulcore_system_notice(
            "这次生成的内容无法正常发送。请重新发送一次；如果一直失败，请联系管理员。"
        )
    processing_step_errors = {
        "COMMAND_TIMEOUT",
        "COMMAND_FAILED",
        "COMMAND_PROTOCOL",
    }
    if code in processing_step_errors:
        return soulcore_system_notice(
            "处理这条消息时，有一步没有完成，所以这次没有回复。"
            "请重新发送一次；如果一直失败，请联系管理员。"
        )
    return soulcore_system_notice(
        "这次处理意外中断，所以没有生成回复。请重新发送一次；如果一直失败，请联系管理员。"
    )


def operation_timeout_seconds(config: Mapping[str, Any]) -> int:
    del config
    return AI_OPERATION_TIMEOUT_SECONDS


def file_artifact_operation_timeout_seconds(file_format: str) -> int:
    if str(file_format or "").strip().upper() == "PDF":
        return FILE_ARTIFACT_PDF_TIMEOUT_SECONDS
    return AI_OPERATION_TIMEOUT_SECONDS


class AstrBotSyntheticEventFactory:
    """Build a minimal event anchored to a previously captured route."""

    def __init__(self, context: Context) -> None:
        self.context = context

    def create(self, *, umo: str, metadata: Mapping[str, Any]) -> AstrMessageEvent:
        captured = CapturedUMO.parse(umo)
        if not captured.is_valid:
            raise ValueError("cannot build a synthetic event from an invalid UMO")
        platform = self.context.get_platform_inst(captured.platform_id)
        if platform is None:
            raise RuntimeError(f"platform is not running: {captured.platform_id}")

        from astrbot.api.event import AstrMessageEvent
        from astrbot.core.platform.astrbot_message import (
            AstrBotMessage,
            Group,
            MessageMember,
        )
        from astrbot.core.platform.message_type import MessageType

        message = AstrBotMessage()
        if captured.kind in (RouteKind.GROUP, RouteKind.GUILD):
            message.type = MessageType.GROUP_MESSAGE
            message.group = Group(group_id=captured.target_id)
        else:
            message.type = MessageType.FRIEND_MESSAGE
        message.session_id = captured.target_id
        message.message = []
        message.message_str = ""
        message.raw_message = None
        message.message_id = f"soulcore-{uuid.uuid4().hex}"
        message.self_id = "soulcore"
        message.sender = MessageMember(user_id=captured.target_id)
        event = AstrMessageEvent(
            message_str="",
            message_obj=message,
            platform_meta=platform.meta(),
            session_id=captured.target_id,
        )
        event.set_extra("soulcore_synthetic", True)
        event.set_extra("soulcore_metadata", dict(metadata))
        return event


__all__ = [
    "AstrBotSyntheticEventFactory",
    "OutboxStatus",
    "PERSISTENT_DATA_NAMESPACE",
    "file_artifact_operation_timeout_seconds",
    "foreground_ai_error_message",
    "has_trusted_astrbot_command_marker",
    "operation_timeout_seconds",
]
