"""Main Core terminal response validation."""

from __future__ import annotations

import inspect
from typing import Any

from .command_context import _active, validate_command_availability_claims
from .expression_timeline import normalize_expression_steps


def validate_expression_handles(collector: Any, steps: list[dict[str, Any]]) -> str:
    message_refs = dict(collector.message_ref_allowlist)
    member_refs = dict(collector.member_ref_allowlist)
    for index, step in enumerate(steps, start=1):
        error = _step_error(collector, step, index, message_refs, member_refs)
        if error:
            return error
    return ""


def _step_error(
    collector: Any,
    step: dict[str, Any],
    index: int,
    message_refs: dict[str, Any],
    member_refs: dict[str, Any],
) -> str:
    if step["kind"] == "RETRACT":
        return _message_error(step, index, message_refs, retract=True)
    error = _message_error(step, index, message_refs, retract=False)
    return error or _member_error(collector, step, index, member_refs)


def _message_error(step: dict[str, Any], index: int, refs: dict[str, Any], *, retract: bool) -> str:
    field = "target_message_ref" if retract else "reply_to_message_ref"
    ref = str(step.get(field) or "")
    permission = "retract_allowed" if retract else "reply_allowed"
    if not ref or (refs.get(ref) and bool(refs[ref].get(permission))):
        return ""
    action = "撤回" if retract else "回复"
    return f"error: 第 {index} 条内容的{action}目标在本轮不可用"


def _member_error(collector: Any, step: dict[str, Any], index: int, refs: dict[str, Any]) -> str:
    for member_ref in step.get("mention_member_refs") or ():
        member = refs.get(str(member_ref))
        if not member:
            return f"error: 第 {index} 条内容的提及对象在当前群聊中不可用"
        same_profile = str(member.get("profile_id") or "") == str(collector.profile_id)
        same_instance = str(member.get("instance_id") or "") == str(collector.instance_id)
        if not same_profile or not same_instance:
            return f"error: 第 {index} 条内容的提及对象不属于当前交流"
    return ""


async def commit_main_core_response(
    _event: Any,
    expression_steps: list[dict[str, Any]] | None = None,
    memo: str = "",
    no_op: bool = False,
    temporary_absence: dict[str, Any] | None = None,
) -> str | None:
    """Validate and collect the final expression decision."""
    collector = _active()
    expression_steps, memo, no_op = _normalize_terminal_values(
        expression_steps,
        memo,
        no_op,
        temporary_absence,
    )
    visible, error = await _validated_response_segment(
        collector,
        expression_steps,
        memo,
        no_op,
        temporary_absence=temporary_absence,
    )
    if error:
        return error
    _commit_main_core_decision(
        collector,
        normalized_steps=visible["expression_steps"],
        no_op=no_op,
        temporary_absence=visible.get("temporary_absence"),
    )
    return None


async def validate_main_core_response_segment(
    expression_steps: list[dict[str, Any]] | None = None,
    memo: str = "",
    no_op: bool = False,
    temporary_absence: dict[str, Any] | None = None,
) -> str:
    """Preflight a visible segment before any action in the model batch executes."""

    expression_steps, memo, no_op = _normalize_terminal_values(
        expression_steps,
        memo,
        no_op,
        temporary_absence,
    )
    _visible, error = await _validated_response_segment(
        _active(),
        expression_steps,
        memo,
        no_op,
        temporary_absence=temporary_absence,
    )
    return error


async def _validated_response_segment(
    collector: Any,
    expression_steps: list[dict[str, Any]] | None,
    memo: str,
    no_op: bool,
    temporary_absence: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    visible, error = await _validate_visible_commit(
        collector,
        expression_steps,
        memo,
        no_op,
        temporary_absence=temporary_absence,
    )
    if error:
        return {}, error
    return (
        visible,
        validate_command_availability_claims(collector, visible["reply"]) or "",
    )


async def _validate_visible_commit(
    collector: Any,
    expression_steps: list[dict[str, Any]] | None,
    memo: str,
    no_op: bool,
    temporary_absence: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    normalized_steps, error = normalize_expression_steps(expression_steps)
    if error:
        return {}, error
    error = _legacy_memo_error(memo)
    if error:
        return {}, error
    error = validate_expression_handles(collector, normalized_steps)
    if error:
        return {}, error
    resolved_absence, error = _validate_temporary_absence(collector, temporary_absence)
    if error:
        return {}, error
    normalized = [item for item in normalized_steps if item["kind"] != "RETRACT"]
    error = _delivery_budget_error(collector, normalized)
    if error:
        return {}, error
    text = "\n".join(item["text"] for item in normalized if item["kind"] == "TEXT").strip()
    refs = _expression_refs(normalized)
    files, error = _validate_file_selection(collector, refs["FILE"], text)
    if error:
        return {}, error
    stickers, error = await _validate_sticker_selection(collector, refs["STICKER"])
    if error:
        return {}, error
    media, error = await _validate_media_selection(collector, refs["IMAGE"])
    if error:
        return {}, error
    error = _required_reply_error(
        normalized,
        no_op,
        has_retraction=any(item["kind"] == "RETRACT" for item in normalized_steps),
        has_temporary_absence=resolved_absence is not None,
    )
    return {
        "expression_steps": normalized_steps,
        "visible_steps": normalized,
        "reply": text,
        "files": files,
        "stickers": stickers,
        "media": media,
        "temporary_absence": resolved_absence,
    }, error


def _validate_temporary_absence(
    collector: Any,
    value: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str]:
    if value is None:
        return None, ""
    context = collector.temporary_absence_command_context
    if context is None:
        return None, "error: 当前会话没有启用暂离"
    try:
        resolved = context.resolve(
            reason=str(value.get("reason") or ""),
            time_expression=str(value.get("time_expression") or ""),
        )
    except (TypeError, ValueError) as exc:
        return None, f"error: {exc}"
    return resolved, ""


def _legacy_memo_error(memo: str) -> str:
    if not str(memo or "").strip():
        return ""
    return "error: 留话必须写在对应的发消息指令内"


def _normalize_terminal_values(
    expression_steps: list[dict[str, Any]] | None,
    memo: str,
    no_op: bool,
    temporary_absence: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], str, bool]:
    steps = list(expression_steps or [])
    if not no_op:
        return steps, memo, False
    if steps or temporary_absence is not None:
        # An explicit expression or absence carries more information than the
        # redundant silent flag, so keep the action and discard the flag.
        return steps, "", False
    # A silent command has nowhere to persist a legacy batch-level memo.
    return steps, "", True


def _delivery_budget_error(
    collector: Any,
    normalized_steps: list[dict[str, Any]],
) -> str:
    budget = collector.delivery_output_budget
    if budget is None or len(normalized_steps) <= budget:
        return ""
    return (
        "error: 本轮当前投递额度最多允许"
        f"{budget}条可见表达，"
        f"但提交了{len(normalized_steps)}条。请合并表达后重新提交。"
    )


def _expression_refs(visible_steps: list[dict[str, Any]]) -> dict[str, list[str]]:
    result = {"IMAGE": [], "STICKER": [], "FILE": []}
    for item in visible_steps:
        if item["kind"] in result:
            result[item["kind"]].append(item["asset_ref_id"])
    return result


def _unique_strings(values: list[str] | None) -> list[str]:
    return list(
        dict.fromkeys(str(item or "").strip() for item in (values or []) if str(item or "").strip())
    )


def _validate_file_selection(
    collector: Any, values: list[str] | None, reply: str
) -> tuple[list[str], str]:
    del reply
    refs = _unique_strings(values)
    if len(refs) > 3:
        return refs, "error: 一轮最多发送三个已完成文件"
    unknown = [item for item in refs if item not in collector.important_todo_refs]
    if unknown:
        return refs, ("error: 发文件只能使用本轮可见的文件短引用：" + "、".join(unknown))
    return refs, ""


async def _validate_sticker_selection(
    collector: Any, values: list[str] | None
) -> tuple[list[str], str]:
    # Sticker repetition is expressive: preserve every timeline occurrence in
    # order.  Validation below still binds each ref to this exact Core run.
    refs = [str(item or "").strip() for item in (values or []) if str(item or "").strip()]
    if not refs:
        return refs, ""
    if collector.sticker_command_context is None:
        return refs, "error: 本轮无法发表情"
    try:
        await collector.sticker_command_context.validate_selection(refs)
    except Exception:
        return refs, "error: 当前表情短引用无法发送"
    return refs, ""


async def _validate_media_selection(
    collector: Any, values: list[str] | None
) -> tuple[list[str], str]:
    selected = _unique_strings(values)
    if len(selected) > 5:
        return selected, "error: 一轮最多发送五张图片"
    permitted = {
        *collector.generated_media_asset_ids,
        *collector.inspected_search_media_asset_ids,
        *collector.current_image_asset_ids,
    }
    unknown = [item for item in selected if item not in permitted]
    if unknown:
        return selected, (
            "error: 发图片只能选择本轮生成或查看过的图片短引用：" + "、".join(unknown)
        )
    if not selected:
        return selected, ""
    generated_or_searched = [
        item for item in selected if item not in set(collector.current_image_asset_ids)
    ]
    if generated_or_searched:
        validation = collector.visual_service.validate_selection(
            profile_id=collector.profile_id,
            instance_id=collector.instance_id,
            run_id=collector.core_run_id,
            asset_ids=generated_or_searched,
        )
        if inspect.isawaitable(validation):
            validation = await validation
        if validation not in (None, True, ""):
            return selected, f"error: 发图片未通过（{validation}）"
    return selected, ""


def _required_reply_error(
    visible_steps: list[dict[str, Any]],
    no_op: bool,
    *,
    has_retraction: bool = False,
    has_temporary_absence: bool = False,
) -> str:
    missing = not visible_steps
    if missing and not has_retraction and not no_op and not has_temporary_absence:
        return "error: 没有表达内容时必须使用不说了"
    return ""


def _commit_main_core_decision(
    collector: Any,
    *,
    normalized_steps: list[dict[str, Any]],
    no_op: bool,
    temporary_absence: dict[str, Any] | None = None,
) -> None:
    collector.commit_calls += 1
    visible = [step for step in normalized_steps if step["kind"] != "RETRACT"]
    refs = _expression_refs(visible)
    collector.selected_media_asset_ids = refs["IMAGE"]
    collector.selected_sticker_ref_ids = refs["STICKER"]
    collector.selected_important_todo_refs = refs["FILE"]
    collector.decision = {
        "expression_steps": normalized_steps,
        "image_generation_failures": list(collector.image_generation_failures),
        "no_op": bool(no_op),
    }
    if temporary_absence is not None:
        collector.decision["temporary_absence"] = dict(temporary_absence)


__all__ = ["commit_main_core_response", "validate_main_core_response_segment"]
