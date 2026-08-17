"""Build the protocol and character portions of the MainCore prefix."""

from __future__ import annotations

from ...shared.prompt_document import PromptBlock, xml_text
from ..character_model import MainCoreStylePrompts, StoryStylePrompts
from .roleplay_prompt_contracts import (
    BACKGROUND_LIFE_AND_COMMUNICATION,
    COMMUNICATION_METHOD,
    CONTEXT_MATERIAL_USE,
    GROUP_CHAT_PARTICIPATION,
    PRIVATE_CHAT_PARTICIPATION,
    ROLEPLAY_CORE,
    BoundedPromptState,
)


def main_core_custom_prompt_blocks(
    main_core_style_prompts: MainCoreStylePrompts,
    story_style_prompts: StoryStylePrompts,
    *,
    include_relationship: bool = True,
    include_sticker: bool = True,
) -> list[PromptBlock]:
    """Return the configurable MainCore blocks that survive final trimming."""

    return [
        PromptBlock(
            "你与对方",
            (
                xml_text(main_core_style_prompts.relationship_context)
                if include_relationship
                else ""
            ),
        ),
        PromptBlock(
            "说话风格",
            xml_text(main_core_style_prompts.speaking_style),
        ),
        PromptBlock(
            "表情包风格",
            xml_text(main_core_style_prompts.sticker_style) if include_sticker else "",
        ),
        PromptBlock(
            "相处倾向",
            xml_text(main_core_style_prompts.thinking_style),
        ),
        PromptBlock(
            "聊天重心",
            xml_text(main_core_style_prompts.content_style),
        ),
        PromptBlock(
            "聊天尺度",
            xml_text(main_core_style_prompts.conversation_content),
        ),
        PromptBlock(
            "故事介入倾向",
            xml_text(story_style_prompts.involvement),
        ),
        PromptBlock(
            "故事姿态",
            xml_text(story_style_prompts.stance),
        ),
    ]


def role_protocol_prompt_blocks(state: BoundedPromptState) -> list[PromptBlock]:
    """Return the invariant RolePlay and chat-scope rules."""

    return [
        PromptBlock("角色定义", ROLEPLAY_CORE),
        PromptBlock("交流与表达方法", COMMUNICATION_METHOD),
        PromptBlock(
            "私聊交流方式",
            PRIVATE_CHAT_PARTICIPATION if state.identity_scope == "private" else "",
        ),
        PromptBlock(
            "群聊参与方式",
            GROUP_CHAT_PARTICIPATION if state.identity_scope == "group" else "",
        ),
    ]


def background_life_and_communication_prompt_block(enabled: bool) -> PromptBlock:
    """Return the fixed interpretation frame that travels with background life."""

    return PromptBlock(
        "你的生活与交流",
        BACKGROUND_LIFE_AND_COMMUNICATION if enabled else "",
    )


def role_identity_prompt_blocks(state: BoundedPromptState) -> list[PromptBlock]:
    """Return persona, style and world material after the action protocol."""

    return [
        PromptBlock("角色信息", xml_text(state.persona)),
        PromptBlock(
            "你与对方",
            xml_text(state.main_core_style_prompts.relationship_context),
        ),
        PromptBlock("背景与历史", CONTEXT_MATERIAL_USE),
        *main_core_custom_prompt_blocks(
            state.main_core_style_prompts,
            state.story_style_prompts,
            include_relationship=False,
        ),
        PromptBlock("世界观", xml_text(state.world)),
        background_life_and_communication_prompt_block(state.background_enabled),
    ]


__all__ = [
    "background_life_and_communication_prompt_block",
    "main_core_custom_prompt_blocks",
    "role_identity_prompt_blocks",
    "role_protocol_prompt_blocks",
]
