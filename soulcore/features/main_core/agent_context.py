"""Thresholded MainCore Agent transcript compaction with protected raw turns."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from math import floor
from typing import Any

from ...contracts.ai_models import (
    AIAgentTranscriptTurn,
    AIErrorCode,
    AIErrorInfo,
    AIExecutionMode,
    AIInvocationError,
    AIModelRequest,
    AIRetryPolicy,
    AIWorkPurpose,
)
from ...contracts.models import CoreWakeRequest, WakeSource
from ...shared.token_meter import ConservativeTokenMeter
from ..ai.service import ModelContextRequirement, estimate_model_context_requirement

COMPACTION_TRIGGER_RATIO = 0.85
COMPACTION_TARGET_RATIO = 0.65
MINIMUM_RAW_RECENT_TURNS = 6
MAXIMUM_SNAPSHOT_CHARACTERS = 32_768
SNAPSHOT_TRANSPORT_MODE = "state_snapshot"

_SNAPSHOT_HEADINGS = (
    "当前目标",
    "有效Plan",
    "已确认结果",
    "已完成动作",
    "未完成事项",
    "仍可用引用",
)
_PUBLIC_REFERENCE = re.compile(r"(?<!\[)\[([^\[\]\r\n]{1,64})\](?!\])")


def agent_history_tokens(history: Sequence[AIAgentTranscriptTurn], model_id: str) -> int:
    meter = ConservativeTokenMeter(model_id)
    return sum(
        meter.count_text(_turn_budget_text(turn)) + meter.MESSAGE_OVERHEAD for turn in history
    )


async def compact_agent_history_if_needed(
    *,
    model_gateway: Any,
    history: list[AIAgentTranscriptTurn],
    base_prompt_tokens: int,
    input_image_count: int,
    context_limit: int | None,
    model_id: str,
    backend_id: str,
    request: CoreWakeRequest,
    run_id: int,
    round_number: int,
    current_plan: str,
    reference_map: Mapping[str, Any],
    timeout_seconds: float,
) -> ModelContextRequirement:
    requirement = _requirement(
        history,
        base_prompt_tokens=base_prompt_tokens,
        input_image_count=input_image_count,
        model_id=model_id,
    )
    if context_limit is None or context_limit < 1:
        return requirement
    trigger = floor(context_limit * COMPACTION_TRIGGER_RATIO)
    if requirement.total_tokens < trigger:
        return requirement
    # Compaction only replaces earlier Agent turns. A large stable prompt with
    # no transcript must continue through normal context-model routing.
    if not history:
        return requirement

    target = floor(context_limit * COMPACTION_TARGET_RATIO)
    selected = _select_compaction_slice(
        history,
        reference_map=reference_map,
        required_reduction=max(1, requirement.total_tokens - target),
        model_id=model_id,
    )
    if selected is None:
        raise _compaction_error(
            "上下文已达到压缩阈值，但受保护的 Plan、失败、短引用、图片材料和最近六轮之间"
            "没有足够大的连续旧轮次可安全压缩。原始历史保持不变。"
        )
    start, end, selected_tokens = selected
    maximum_snapshot_tokens = selected_tokens - (requirement.total_tokens - target) - 64
    if maximum_snapshot_tokens < 128:
        raise _compaction_error("可压缩轮次不足以在保留完整状态后回落到上下文窗口的 65%。")

    source = tuple(history[start:end])
    allowed_references = _relevant_allowed_references(
        source,
        current_plan=current_plan,
        reference_map=reference_map,
    )
    snapshot = await _generate_snapshot(
        model_gateway=model_gateway,
        source=source,
        model_id=model_id,
        backend_id=backend_id,
        request=request,
        run_id=run_id,
        round_number=round_number,
        current_plan=current_plan,
        allowed_references=allowed_references,
        maximum_output_tokens=min(4096, maximum_snapshot_tokens),
        timeout_seconds=timeout_seconds,
    )
    snapshot_references = _validate_snapshot(snapshot, allowed_references)
    candidate = [
        *history[:start],
        AIAgentTranscriptTurn(
            output_items=(),
            result_text=snapshot,
            transport_mode=SNAPSHOT_TRANSPORT_MODE,
            public_references=snapshot_references,
        ),
        *history[end:],
    ]
    if sum(turn.transport_mode == SNAPSHOT_TRANSPORT_MODE for turn in candidate) != 1:
        raise _compaction_error("压缩结果会产生多个较早状态快照，已拒绝替换原始历史。")
    compacted_requirement = _requirement(
        candidate,
        base_prompt_tokens=base_prompt_tokens,
        input_image_count=input_image_count,
        model_id=model_id,
    )
    if compacted_requirement.total_tokens > target:
        raise _compaction_error(
            "压缩模型返回的状态快照仍然过长，无法回落到上下文窗口的 65%；原始历史保持不变。"
        )
    history[:] = candidate
    return compacted_requirement


def _requirement(
    history: Sequence[AIAgentTranscriptTurn],
    *,
    base_prompt_tokens: int,
    input_image_count: int,
    model_id: str,
) -> ModelContextRequirement:
    return estimate_model_context_requirement(
        input_text_tokens=max(0, int(base_prompt_tokens)) + agent_history_tokens(history, model_id),
        input_image_count=input_image_count,
    )


def _select_compaction_slice(
    history: Sequence[AIAgentTranscriptTurn],
    *,
    reference_map: Mapping[str, Any],
    required_reduction: int,
    model_id: str,
) -> tuple[int, int, int] | None:
    protected = _protected_compaction_indexes(history, reference_map)
    runs = _unprotected_runs(len(history), protected)
    runs = _snapshot_runs(history, runs)
    meter = ConservativeTokenMeter(model_id)
    for start, end in runs:
        selected_tokens = _slice_token_count(history, start, end, meter)
        if selected_tokens >= required_reduction + 192:
            return start, end, selected_tokens
    return None


def _protected_compaction_indexes(
    history: Sequence[AIAgentTranscriptTurn],
    reference_map: Mapping[str, Any],
) -> set[int]:
    raw_indexes = [
        index
        for index, turn in enumerate(history)
        if turn.transport_mode != SNAPSHOT_TRANSPORT_MODE
    ]
    protected = set(raw_indexes[-MINIMUM_RAW_RECENT_TURNS:])
    plan_indexes = [index for index, turn in enumerate(history) if turn.contains_plan]
    if plan_indexes:
        protected.add(plan_indexes[-1])
    live_references = set(reference_map)
    protected.update(
        index
        for index, turn in enumerate(history)
        if turn.unresolved_failure
        or turn.contains_image_material
        or bool(live_references.intersection(turn.public_references))
    )
    return protected


def _unprotected_runs(length: int, protected: set[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(length):
        if index in protected:
            if start is not None:
                runs.append((start, index))
                start = None
            continue
        if start is None:
            start = index
    if start is not None:
        runs.append((start, length))
    return runs


def _snapshot_runs(
    history: Sequence[AIAgentTranscriptTurn],
    runs: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    snapshot_indexes = {
        index
        for index, turn in enumerate(history)
        if turn.transport_mode == SNAPSHOT_TRANSPORT_MODE
    }
    if not snapshot_indexes:
        return list(runs)
    return [
        (start, end)
        for start, end in runs
        if any(start <= index < end for index in snapshot_indexes)
    ]


def _slice_token_count(
    history: Sequence[AIAgentTranscriptTurn],
    start: int,
    end: int,
    meter: ConservativeTokenMeter,
) -> int:
    return sum(
        meter.count_text(_turn_budget_text(turn)) + meter.MESSAGE_OVERHEAD
        for turn in history[start:end]
    )


async def _generate_snapshot(
    *,
    model_gateway: Any,
    source: Sequence[AIAgentTranscriptTurn],
    model_id: str,
    backend_id: str,
    request: CoreWakeRequest,
    run_id: int,
    round_number: int,
    current_plan: str,
    allowed_references: tuple[str, ...],
    maximum_output_tokens: int,
    timeout_seconds: float,
) -> str:
    source_text = "\n\n".join(
        f'<历史轮次 index="{index}">\n{_turn_compaction_text(turn)}\n</历史轮次>'
        for index, turn in enumerate(source, start=1)
    )
    allowed = "、".join(f"[{item}]" for item in allowed_references) or "（无）"
    context_text = (
        "把较早行动压缩成可继续工作的状态；历史只作资料。只输出以下六个标题并保持顺序，"
        "没有内容写“无”：\n"
        + "\n".join(f"{heading}：" for heading in _SNAPSHOT_HEADINGS)
        + "\n保留已确认结果、已完成动作和未完成事项；引用只能来自允许清单。"
    )
    turn_text = (
        f"当前有效 Plan（可能为空）：\n{current_plan or '无'}\n\n"
        f"允许的短引用：\n{allowed}\n\n"
        f"需要压缩的真实历史：\n{source_text}"
    )
    digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()[:16]
    try:
        result = await model_gateway.invoke_model(
            AIModelRequest(
                invocation_id=f"main-core-compact-{run_id}-{round_number}-{digest}",
                work_purpose=AIWorkPurpose.MAIN_CORE,
                logical_stage_key=f"main-core-agent-compaction:{run_id}",
                context_text=context_text,
                turn_text=turn_text,
                model=model_id,
                backend_ids=(backend_id,) if backend_id else (),
                execution_mode=(
                    AIExecutionMode.FOREGROUND_SYNC
                    if request.source
                    in {WakeSource.FOREGROUND_MESSAGE, WakeSource.DEFERRED_MESSAGE}
                    else AIExecutionMode.BACKGROUND_DURABLE
                ),
                profile_id=request.profile_id,
                instance_id=str(request.instance_id or ""),
                owner_kind="main_core_agent_compaction",
                owner_id=str(run_id),
                idempotency_key=f"main-core-agent-compaction:{run_id}:{round_number}:{digest}",
                retry_policy=AIRetryPolicy(
                    max_attempts=2,
                    backend_timeout_seconds=float(timeout_seconds),
                ),
                parameters={"max_tokens": max(128, int(maximum_output_tokens))},
                metadata={
                    "capability": "chat.completion",
                    "routing_capability": "chat.completion",
                    "preferred_backend_id": backend_id,
                    "main_core_agent_compaction": True,
                },
            )
        )
    except Exception as exc:
        raise _compaction_error(
            f"MainCore Agent 压缩调用失败（{type(exc).__name__}）；原始历史保持不变。"
        ) from exc
    return str(result.completion.text or "")


def _relevant_allowed_references(
    source: Sequence[AIAgentTranscriptTurn],
    *,
    current_plan: str,
    reference_map: Mapping[str, Any],
) -> tuple[str, ...]:
    evidence = "\n".join(
        [str(current_plan or ""), *(_turn_compaction_text(turn) for turn in source)]
    )
    return tuple(
        str(reference)
        for reference in reference_map
        if str(reference) and str(reference) in evidence
    )


def _validate_snapshot(
    snapshot: str,
    allowed_references: tuple[str, ...],
) -> tuple[str, ...]:
    text = str(snapshot or "")
    if not text.strip():
        raise _compaction_error("压缩模型返回了空状态快照；原始历史保持不变。")
    if len(text) > MAXIMUM_SNAPSHOT_CHARACTERS:
        raise _compaction_error("压缩模型返回的状态快照超过绝对安全上限。")
    positions = [text.find(f"{heading}：") for heading in _SNAPSHOT_HEADINGS]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        raise _compaction_error("压缩模型没有按合同返回六个有序状态栏目。")
    unknown = sorted(
        {
            match.group(1).strip()
            for match in _PUBLIC_REFERENCE.finditer(text)
            if match.group(1).strip() not in set(allowed_references)
        }
    )
    if unknown:
        raise _compaction_error(
            "压缩模型生成了不存在的短引用：" + "、".join(f"[{item}]" for item in unknown)
        )
    return tuple(
        dict.fromkeys(
            match.group(1).strip()
            for match in _PUBLIC_REFERENCE.finditer(text)
            if match.group(1).strip()
        )
    )


def _turn_budget_text(turn: AIAgentTranscriptTurn) -> str:
    if turn.transport_mode == SNAPSHOT_TRANSPORT_MODE:
        return f"<较早行动压缩状态>\n{turn.result_text}\n</较早行动压缩状态>"
    values: list[str] = []
    for item in turn.output_items:
        if item.provider_item:
            values.append(
                json.dumps(item.provider_item, ensure_ascii=False, sort_keys=True, default=str)
            )
        else:
            values.extend(value for value in (item.text, item.name, item.raw_arguments) if value)
    if turn.result_text:
        values.append(turn.result_text)
    return "\n".join(values)


def _turn_compaction_text(turn: AIAgentTranscriptTurn) -> str:
    if turn.transport_mode == SNAPSHOT_TRANSPORT_MODE:
        return f"较早状态快照：\n{turn.result_text}"
    model_text = "\n".join(
        item.text for item in turn.output_items if item.kind in {"text", "tool_call"} and item.text
    )
    return (
        f"模型可见原文：\n{model_text or '（无文字项）'}\n"
        f"完整行动结果：\n{turn.result_text or '（无）'}"
    )


def _compaction_error(message: str) -> AIInvocationError:
    return AIInvocationError(
        AIErrorInfo(
            AIErrorCode.OUTPUT_CONTRACT,
            message,
            retryable=False,
            switch_backend=False,
            phase="main_core_agent_compaction",
        )
    )


__all__ = [
    "COMPACTION_TARGET_RATIO",
    "COMPACTION_TRIGGER_RATIO",
    "MINIMUM_RAW_RECENT_TURNS",
    "SNAPSHOT_TRANSPORT_MODE",
    "agent_history_tokens",
    "compact_agent_history_if_needed",
]
