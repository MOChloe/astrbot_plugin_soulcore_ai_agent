"Single production catalog for SoulCore Main Core text commands."

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from ..ai.service import (
    CommandParameter,
    CommandSpec,
    MainCoreCommandSet,
    ModelVisibleCommandResult,
    reference_validator,
)
from ..recall import RecallMode, RecallRequest
from ..timers.service import MAX_SEMANTIC_CANDIDATES, TimerDomainError
from .command_catalog_media import media_commands
from .command_catalog_support import (
    command,
    command_outcome_handler,
    commit_main_core_response_with_work_validation,
    parameter,
)
from .command_context import _active
from .conversation_history_commands import browse_chat_history
from .player_profile_commands import (
    forget_player_profile,
    recall_player_profile,
    remember_player_profile,
    revise_player_profile,
)
from .terminal_decision import commit_main_core_response


async def recall_context(_event: Any, need: str) -> Any:
    """Recall facts, events and changes without exposing retrieval controls."""

    collector = _active()
    if collector.recall_query_calls >= 2:
        return "error: 本次行动最多显式回想两次"
    value = str(need or "").strip()
    if not value:
        return "error: 想回想的内容不能为空"
    if not collector.profile_id or not collector.instance_id:
        return "error: 当前交流无法回想已保存资料"
    service = collector.recall_service
    if service is None:
        return "error: 回想暂时不可用"
    collector.recall_query_calls += 1
    current_time = collector.player_profile_confirmed_at
    if not isinstance(current_time, datetime):
        current_time = datetime.now(UTC)
    try:
        bundle = await asyncio.wait_for(
            service.recall(
                RecallRequest(
                    profile_id=collector.profile_id,
                    instance_id=collector.instance_id,
                    need=value,
                    mode=RecallMode.EXPLICIT,
                    current_time=current_time,
                    recent_visible_context=tuple(collector.recent_visible_context[-6:]),
                    visible_source_fingerprints=frozenset(collector.visible_history_fingerprints),
                    excluded_document_keys=frozenset(
                        collector.visible_recall_document_keys | collector.recalled_document_keys
                    ),
                    token_budget=1200,
                )
            ),
            timeout=10.0,
        )
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        return ModelVisibleCommandResult("回想超时；请缩小想确认的内容后再试。")
    except Exception:
        return ModelVisibleCommandResult("回想暂时无法完成可靠核对。")
    collector.recalled_document_keys.update(bundle.document_keys)
    return ModelVisibleCommandResult(service.render(bundle, token_budget=1200))


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


async def remember_future(
    _event: Any,
    time_expression: str,
    action_text: str,
) -> Any:
    context = _timer_context()
    if context is None:
        return "error: 本轮无法使用安排功能"
    try:
        return await context.stage_natural_creation(
            time_expression=time_expression,
            action_text=action_text,
        )
    except TimerDomainError as exc:
        return f"error: 未来安排未记下（{_timer_error_text(exc)}）"
    except (TypeError, ValueError):
        return "error: 未来安排未记下（参数无效）"


async def list_arrangements(
    _event: Any,
    query: str = "",
) -> Any:
    context = _timer_context()
    if context is None:
        return "error: 本轮无法使用安排功能"
    try:
        return await context.list_arrangements(limit=MAX_SEMANTIC_CANDIDATES, query=query)
    except TimerDomainError as exc:
        return f"error: 安排没有查看成功（{_timer_error_text(exc)}）"
    except (TypeError, ValueError):
        return "error: 安排没有查看成功（参数无效）"


async def adjust_arrangement(
    _event: Any,
    target: str,
    change: str,
) -> Any:
    context = _timer_context()
    if context is None:
        return "error: 本轮无法使用安排功能"
    try:
        return await context.stage_natural_adjustment(target=target, change=change)
    except TimerDomainError as exc:
        return f"error: 安排没有调整（{_timer_error_text(exc)}）"
    except (TypeError, ValueError):
        return "error: 安排没有调整（参数无效）"


def _timer_context() -> Any | None:
    return _active().timer_command_context


def _timer_error_text(error: TimerDomainError) -> str:
    return {
        "INVALID_REFERENCE": "引用无效或已经过期",
        "INVALID_PROMPT": "到时候做的事无效",
        "INVALID_RULE": "安排的时间或内容无效",
        "UNSUPPORTED_RULE": "暂不支持这种安排",
        "INVALID_TIMEZONE": "时区无效",
        "OUT_OF_RANGE": "数值超出允许范围",
        "INVALID_STATE": "当前状态不允许这个操作",
        "VERSION_CONFLICT": "安排已经变化，请重新查看",
        "SCOPE_MISMATCH": "安排不属于当前会话",
        "LIMIT_EXCEEDED": "本轮安排操作已达到上限",
    }.get(error.code.value, "参数或当前状态不允许")


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


def set_run_plan(_event: Any, content: str) -> str | ModelVisibleCommandResult:
    """Replace the current Main Core run's plan without persisting it."""

    plan = str(content or "").strip()
    if not plan:
        return "error: Plan内容不能为空；原有Plan保持不变。"
    _active().current_plan = plan
    return ModelVisibleCommandResult("Plan 已保存；下一轮继续。")


def run_plan_command() -> object:
    return command(
        "制定Plan",
        "set_run_plan",
        "为本次行动确定最终要形成的对方可见表达、作品或数据，以及它的内容组成、表达方式、尚需完成的行动和完成标准。",
        command_outcome_handler("set_run_plan", set_run_plan),
        parameter(
            "内容",
            "content",
            required=True,
            prompt_hint=(
                "最终目标与完成标准；准备采用的内容、结构、语气或风格；尚需查证或执行的事项"
            ),
            identity_mode="literal",
        ),
        serial=True,
        usage_guidance=(
            "写清最终目标、完成标准、关键取舍和仍需取得的结果；未知结果不写成事实。"
            "再次使用会整体替换当前 Plan。"
        ),
    )


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
    "adjust_arrangement",
    "build_main_core_commands",
    "build_restricted_response_commands",
    "conversation_commands",
    "list_arrangements",
    "recall_context",
    "remember_future",
    "run_plan_command",
    "set_run_plan",
    "timer_commands",
]
