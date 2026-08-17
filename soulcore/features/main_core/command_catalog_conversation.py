"""Natural conversation, person-impression, history, and world commands."""

from __future__ import annotations

from ..ai.service import reference_validator
from .command_catalog_support import command, parameter
from .command_outcomes import command_outcome_handler
from .conversation_history_commands import browse_chat_history
from .knowledge_commands import recall_context
from .player_profile_commands import (
    forget_player_profile,
    recall_player_profile,
    remember_player_profile,
    revise_player_profile,
)


def conversation_commands(*, include_profile_query: bool = True, scope: str = "") -> list[object]:
    normalized_scope = str(scope or "").strip()
    if normalized_scope not in {"", "private", "group"}:
        raise ValueError("scope must be private, group, or empty")
    return [
        *_player_profile_commands(
            include_profile_query=include_profile_query,
            scope=normalized_scope,
        ),
        *_history_commands(),
    ]


def _player_profile_commands(*, include_profile_query: bool, scope: str) -> list[object]:
    remember_person = (
        ()
        if scope == "private"
        else (
            parameter(
                "人物",
                "member_ref",
                required=scope == "group",
                validator=reference_validator("人物"),
                prompt_hint=(
                    "当前可见的人物短引用"
                    if scope == "group"
                    else "群聊时填写当前可见的人物短引用；私聊删除"
                ),
            ),
        )
    )
    remember_guidance = {
        "private": (
            "只记录当前对方本人，不把虚构人物、举例中的人物或第三者写进对方印象。"
            "当前输入已经直接说明时省略[[依据]]；否则填写当前消息短引用。"
        ),
        "group": (
            "只记录[[人物]]指向的现实聊天成员，不把虚构人物、举例中的人物或第三者写进"
            "对方印象。[[人物]]使用当前可见人物短引用。当前输入本身已经直接说明时删除"
            "[[依据]]；否则用当前可见消息短引用。"
        ),
    }.get(
        scope,
        (
            "只记录现实聊天对象本人，不把虚构人物、举例中的人物或第三者写进对方印象。"
            "私聊删除[[人物]]整行；群聊用当前可见人物短引用。当前输入本身已经直接说明时"
            "删除[[依据]]；否则用当前可见消息短引用。"
        ),
    )
    commands = [
        command(
            "记住",
            "remember_player_profile",
            "对方刚刚显露了一件以后相处时真正有用、而且相对稳定的事，想把它记住时使用。",
            command_outcome_handler("remember_player_profile", remember_player_profile),
            parameter(
                "内容",
                "content",
                required=True,
                prompt_hint="以后想记得的那件事",
                identity_mode="template",
            ),
            *remember_person,
            parameter(
                "依据",
                "evidence_ref",
                prompt_hint="支持这件事的当前可见消息短引用；当前输入本身已经直接说明时可删除",
                validator=reference_validator("依据"),
            ),
            usage_guidance=remember_guidance,
            serial=True,
        ),
        command(
            "改印象",
            "revise_player_profile",
            "发现已有的一条印象不准确、过时或需要换成更合适的说法时使用。",
            command_outcome_handler("revise_player_profile", revise_player_profile),
            parameter(
                "原来的印象",
                "original_impression",
                required=True,
                prompt_hint="当前可见的印象短引用，或足以唯一识别它的自然描述",
                identity_mode="literal",
            ),
            parameter(
                "改成",
                "new_impression",
                required=True,
                prompt_hint="现在更合适的认识",
                identity_mode="template",
            ),
            parameter(
                "依据",
                "evidence_ref",
                prompt_hint="支持这次变化的当前可见消息短引用",
                validator=reference_validator("依据"),
            ),
            usage_guidance=(
                "优先使用当前可见的印象短引用；手边没有引用时可以自然描述。自然描述只有唯一"
                "匹配时才执行，多条合理候选会先返回少量候选。[[依据]]只在需要指出当前可见"
                "支持消息时填写。"
            ),
            serial=True,
        ),
        command(
            "忘掉",
            "forget_player_profile",
            "确定某条旧印象已经不成立，或不再希望把它作为以后相处的依据时使用。",
            command_outcome_handler("forget_player_profile", forget_player_profile),
            parameter(
                "印象",
                "impression",
                required=True,
                prompt_hint="当前可见的印象短引用，或足以唯一识别它的自然描述",
                identity_mode="literal",
            ),
            usage_guidance=(
                "只撤回自己形成的这条认识，不删除聊天原文。优先使用当前可见短引用；自然描述"
                "只有唯一匹配时才执行，多条合理候选会先返回少量候选。"
            ),
            serial=True,
        ),
    ]
    if include_profile_query:
        commands.append(_player_profile_recall_command(scope))
    return commands


def _player_profile_recall_command(scope: str) -> object:
    recall_person = (
        ()
        if scope == "private"
        else (
            parameter(
                "人物",
                "member_ref",
                required=scope == "group",
                validator=reference_validator("人物"),
                prompt_hint=(
                    "当前可见的人物短引用"
                    if scope == "group"
                    else "群聊时填写当前可见的人物短引用；私聊默认是正在交谈的人"
                ),
            ),
        )
    )
    recall_guidance = {
        "private": "默认回顾当前对方；结果会同时带回需要精确改动时可用的印象短引用。",
        "group": (
            "使用当前可见人物短引用选择要回顾的人；结果以自然认识为主，需要精确改动的"
            "条目会同时带可用短引用。"
        ),
    }.get(
        scope,
        (
            "私聊删除[[人物]]整行，默认是正在交谈的人；群聊使用当前可见人物短引用。"
            "结果以自然认识为主，需要精确改动的条目会同时带可用短引用。"
        ),
    )
    return command(
        "想想对某人的印象",
        "recall_player_profile",
        "想回顾自己对一个现实聊天对象已经形成的认识时使用。",
        command_outcome_handler("recall_player_profile", recall_player_profile),
        *recall_person,
        usage_guidance=recall_guidance,
    )


def _history_commands() -> list[object]:
    return [
        command(
            "回想",
            "recall_context",
            "眼前能看到的内容不足，需要按含义回想已知事实、历史事件或它们发生过的变化时使用。",
            command_outcome_handler("recall_context", recall_context),
            parameter(
                "想知道什么",
                "need",
                required=True,
                prompt_hint="自然描述想确认的事实、事件、关系或变化",
            ),
            usage_guidance=(
                "只用自然语言说明想确认的内容；涉及人物或时间范围时也写进同一句话，不添加额外参数。"
                "查事实、事件、关系或它们最早、后来、之前等时序变化，以及准备断言“没有这回事”时，"
                "都先用“回想”；“翻聊天记录”不能代替按含义回想。只有对方明确需要逐句原话或相邻"
                "消息顺序时才改用“翻聊天记录”。第一次结果不足时可再用自然语言补充条件一次。"
            ),
        ),
        command(
            "翻聊天记录",
            "browse_chat_history",
            "想亲自往前或往后翻某段聊天、确认原话和先后顺序时使用。",
            command_outcome_handler("browse_chat_history", browse_chat_history),
            parameter(
                "位置",
                "position",
                prompt_hint="例如“刚认识那阵子”“昨晚”“说到密室前后”或“接着刚才”",
            ),
            parameter(
                "方向",
                "direction",
                choices=("往更早", "往更新"),
                prompt_hint="默认往更早",
            ),
            usage_guidance=(
                "只用自然语言填写想看的位置与方向。它只用于逐句原话或相邻消息顺序，不用于"
                "按含义判断事实、事件或变化；问题出现“最早、后来、之前”等词并不等于要翻记录。"
            ),
            serial=True,
        ),
    ]


__all__ = ["conversation_commands"]
