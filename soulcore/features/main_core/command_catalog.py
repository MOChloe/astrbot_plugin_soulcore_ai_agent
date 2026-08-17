"""Single production catalog for SoulCore Main Core text commands."""

from __future__ import annotations

from ..ai.service import CommandParameter, CommandSpec, MainCoreCommandSet
from .command_catalog_conversation import conversation_commands
from .command_catalog_media import media_commands
from .command_catalog_timers import timer_commands
from .command_outcomes import commit_main_core_response_with_work_validation
from .run_plan_commands import run_plan_command
from .terminal_decision import commit_main_core_response


def scene_narration_command() -> object:
    return CommandSpec(
        name="旁白",
        internal_name="__scene_narration",
        description=(
            "想让同一批消息之间承载动作、场景或时间转折时使用；"
            "它进入角色自己的对话时间线，但对方看不到。"
        ),
        parameters=(
            CommandParameter(
                "内容",
                "content",
                required=True,
                prompt_hint=("第三人称动作或场景描写，例如“她刚说完，桌边的水杯忽然掉在地上”"),
                identity_mode="template",
            ),
        ),
        terminal=True,
        send_kind="NARRATION",
        usage_guidance=(
            "必须与至少一条发文字、发图片、发表情或发文件同批出现，不能单独结束行动。"
            "把它写在两条发送指令之间，就表示这段变化发生在两条消息之间；"
            "不要用它替代要让对方看到的消息。"
            "不要用旁白想象或补写对方没有明确提供的动作、想法、感受或现场。"
        ),
        body_parameter="内容",
    )


def build_main_core_commands(
    *,
    scope: str = "",
    include_visual: bool = True,
    include_web: bool = False,
    include_web_images: bool = True,
    include_stickers: bool = False,
    include_files: bool = False,
    include_file_delivery: bool | None = None,
    include_image_delivery: bool = True,
    include_current_image_inspection: bool = False,
    include_profile_query: bool = True,
    include_temporary_absence: bool = True,
) -> MainCoreCommandSet:
    commands = [
        run_plan_command(),
        scene_narration_command(),
        *conversation_commands(include_profile_query=include_profile_query, scope=scope),
        *timer_commands(),
    ]
    commands.extend(
        media_commands(
            include_visual=include_visual,
            include_web=include_web,
            include_web_images=include_web_images,
            include_stickers=include_stickers,
            include_files=include_files,
            include_current_image_inspection=include_current_image_inspection,
        )
    )
    return MainCoreCommandSet(
        commands,
        terminal_handler=commit_main_core_response_with_work_validation,
        disabled_terminal_send_kinds=tuple(
            kind
            for kind, disabled in (
                (
                    "FILE",
                    not (include_files if include_file_delivery is None else include_file_delivery),
                ),
                ("IMAGE", not include_image_delivery),
                ("STICKER", not include_stickers),
                ("ABSENCE", not include_temporary_absence),
            )
            if disabled
        ),
    )


def build_restricted_response_commands() -> MainCoreCommandSet:
    return MainCoreCommandSet(
        terminal_handler=commit_main_core_response,
        disabled_terminal_send_kinds=("FILE", "IMAGE", "STICKER", "ABSENCE"),
    )


__all__ = [
    "build_main_core_commands",
    "build_restricted_response_commands",
]
