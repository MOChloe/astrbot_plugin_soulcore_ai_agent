"""Natural arrangement commands for Main Core."""

from __future__ import annotations

from .command_catalog_support import command, parameter
from .command_outcomes import command_outcome_handler
from .timer_commands import adjust_arrangement, list_arrangements, remember_future


def timer_commands() -> list[object]:
    return [
        command(
            "记下未来的事",
            "remember_future",
            "让你在未来一个明确的时间重新回来，继续看一件事或做一件事。",
            command_outcome_handler("remember_future", remember_future),
            parameter(
                "时间",
                "time_expression",
                required=True,
                prompt_hint="什么时候回来，例如“十分钟后”“明天 21:00”“每周五 21:00”",
                identity_mode="literal",
            ),
            parameter(
                "到时候做什么",
                "action_text",
                required=True,
                prompt_hint="回来时想继续看、继续想或实际去做的事",
                identity_mode="template",
            ),
            usage_guidance=(
                "当你已经和对方约好某个时间再联系、回来继续、到时提醒或做一件事时，就使用这个动作"
                "把约定真正留下，不要只在发出的消息里口头答应。没有约定时，只要你自己确实想在某个"
                "时间主动回来，也可以使用。"
                "到了时间，你会重新面对那时的聊天和处境，再自己决定实际说什么、做什么。"
                "现在只需要写下什么时候回来，以及回来时想接着处理什么。"
            ),
            serial=True,
        ),
        command(
            "看看我的安排",
            "list_arrangements",
            "看看接下来答应过、计划过或暂停着的事情。",
            command_outcome_handler("list_arrangements", list_arrangements),
            parameter(
                "想看哪一段",
                "query",
                prompt_hint="例如“最近”“下周”“跟密室有关的”；删除时查看近期安排",
            ),
            usage_guidance=(
                "结果使用自然时间和行动短摘要，并为每项提供 TM 短引用。"
                "短引用只用于本次行动中精确调整，不是安排内容。"
            ),
        ),
        command(
            "调整安排",
            "adjust_arrangement",
            "取消、暂停、继续、改时间或改内容；不要求先专门查看安排。",
            command_outcome_handler("adjust_arrangement", adjust_arrangement),
            parameter(
                "哪件事",
                "target",
                required=True,
                prompt_hint="当前可见的 TM 短引用，或足以唯一识别它的自然描述",
                identity_mode="literal",
            ),
            parameter(
                "怎么改",
                "change",
                required=True,
                prompt_hint=("例如“取消”“先暂停”“继续”“改到明晚九点”“到时改成发张照片”"),
                identity_mode="template",
            ),
            usage_guidance=(
                "短引用直接定位；自然描述只有唯一匹配时才暂存修改，多项相似时只返回"
                "少量候选且不改动。重复安排改成单次时间时，要明确只改下一次还是整个安排；"
                "不能把待澄清或待最终提交说成已经调整成功。"
            ),
            serial=True,
        ),
    ]


__all__ = ["timer_commands"]
