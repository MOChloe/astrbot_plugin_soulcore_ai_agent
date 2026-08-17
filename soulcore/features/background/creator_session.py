"""Bounded validation session for one background creator stage."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...contracts.ai_models import AIWorkPurpose
from ...shared.prompt_document import TrustedPromptMarkup, project_prompt_text
from ..ai import record_structured_rejection
from ..ai.service import DEFAULT_RESERVED_OUTPUT_TOKENS
from ..identity import (
    IdentityCatalog,
    IdentityRenderContext,
    project_identity_text_for_model,
)
from .creator_output import (
    REPAIR_SNAPSHOT,
    CreatorOutputResult,
    parse_snapshot_repair_output,
    recover_creator_output,
)
from .domain import BackgroundAuthorKind, BackgroundDraft
from .draft import draft_from_creator, normalize_optional_references
from .identity_projection import (
    background_identity_material,
    decode_visible_identity_data,
    finalize_identity_directory,
    participant_reference_map,
    provisional_identity_catalog_text,
    visible_identity_catalog,
)
from .output_contract import BackgroundOutputError
from .ports import (
    BackgroundIdentityPort,
    BackgroundModelGatewayPort,
    BackgroundTaskControl,
)
from .prompt_budget import (
    BackgroundPromptBudget,
    BackgroundPromptBudgetExceeded,
    FrozenBackgroundProjection,
    fit_stage_markup,
    required_stage_context_tokens,
    stage_context_tokens,
    visible_background_references,
)
from .prompts import build_creator_prompt, build_snapshot_repair_prompt, required_block_fragments


async def fit_routed_stage_input(
    *,
    model_gateway: BackgroundModelGatewayPort,
    profile_id: str,
    author_kind: BackgroundAuthorKind,
    stage: str,
    task_definition: str,
    task_input: TrustedPromptMarkup,
    output_contract: str,
    output_reserve_tokens: int,
    preferred_backend_id: str,
    finalize_task_input: Callable[[TrustedPromptMarkup], TrustedPromptMarkup] | None = None,
) -> tuple[TrustedPromptMarkup, int]:
    """Select a model that fits required blocks, then fill only optional space."""

    safety_margin_tokens = BackgroundPromptBudget().safety_margin_tokens
    required_fragments = required_block_fragments(author_kind, stage)
    required_context_tokens = required_stage_context_tokens(
        task_definition=task_definition,
        task_input=task_input,
        output_contract=output_contract,
        output_reserve_tokens=output_reserve_tokens,
        safety_margin_tokens=safety_margin_tokens,
        required_name_fragments=required_fragments,
    )
    window = await model_gateway.resolve_text_context_window(
        profile_id=profile_id,
        capability="text.completion",
        preferred_backend_id=preferred_backend_id,
        minimum_context_tokens=required_context_tokens,
    )
    budget = BackgroundPromptBudget(
        max_context_tokens=window,
        safety_margin_tokens=safety_margin_tokens,
    )
    fitted = fit_stage_markup(
        author_kind=author_kind,
        stage=stage,
        candidates=(("frozen_projection", task_input),),
        limit=budget.stage_input_limit(
            task_definition=task_definition,
            output_contract=output_contract,
            output_reserve_tokens=output_reserve_tokens,
        ),
        required_name_fragments=required_fragments,
    )
    if finalize_task_input is not None:
        fitted = finalize_task_input(fitted)
    final_context_tokens = stage_context_tokens(
        task_definition=task_definition,
        task_input=fitted,
        output_contract=output_contract,
        output_reserve_tokens=output_reserve_tokens,
        safety_margin_tokens=safety_margin_tokens,
    )
    if final_context_tokens > window:
        raise BackgroundPromptBudgetExceeded(
            stage=stage,
            block_name="final_prompt",
            block_tokens=final_context_tokens,
            limit=window,
        )
    return fitted, final_context_tokens


_WORK_PURPOSE = {
    BackgroundAuthorKind.WORLD: AIWorkPurpose.BACKGROUND_WORLD,
    BackgroundAuthorKind.LIFE_DIRECTION: AIWorkPurpose.BACKGROUND_LIFE_DIRECTION,
    BackgroundAuthorKind.STORY_SOURCE: AIWorkPurpose.BACKGROUND_STORY_SOURCE,
    BackgroundAuthorKind.KEYFRAME: AIWorkPurpose.BACKGROUND_KEYFRAME,
    BackgroundAuthorKind.ORDINARY: AIWorkPurpose.BACKGROUND_ORDINARY,
}


@dataclass(frozen=True, slots=True)
class CreatorSessionSpec:
    profile_id: str
    instance_id: str
    author_kind: BackgroundAuthorKind
    task_id: int
    logical_stage_key: str
    operation_timeout_seconds: int
    opening_keyframe: bool
    authoritative_time: str


@dataclass(frozen=True, slots=True)
class CreationResult:
    creator: dict[str, Any]
    draft: BackgroundDraft
    completion: Any
    round_no: int
    normalizations: tuple[str, ...] = ()
    repair_kind: str = ""


@dataclass(frozen=True, slots=True)
class _PreparedSession:
    spec: CreatorSessionSpec
    frozen: FrozenBackgroundProjection
    preferred_backend_id: str
    deadline: float
    definition: str
    task_input: TrustedPromptMarkup
    contract: str
    minimum_context_tokens: int
    provisional_catalog: str
    identity_context: IdentityRenderContext
    identity_catalog: IdentityCatalog
    visible_catalog: IdentityCatalog
    visible_references: Any


@dataclass(frozen=True, slots=True)
class _RoundPrompt:
    definition: str
    task_input: TrustedPromptMarkup
    contract: str
    minimum_context_tokens: int


@dataclass(slots=True)
class _ValidationState:
    mode: str = "FULL"
    preserved: CreatorOutputResult | None = None
    previous_error: str = ""


class _RoundRejected(BackgroundOutputError):
    def __init__(
        self,
        error: str,
        *,
        mode: str,
        preserved: CreatorOutputResult | None,
        normalizations: tuple[str, ...] = (),
    ) -> None:
        super().__init__(error)
        self.mode = mode
        self.preserved = preserved
        self.normalizations = normalizations


async def run_creator_session(
    spec: CreatorSessionSpec,
    *,
    frozen: FrozenBackgroundProjection,
    preferred_backend_id: str,
    character_projection: str,
    model_gateway: BackgroundModelGatewayPort,
    identity: BackgroundIdentityPort,
    control: BackgroundTaskControl | None,
) -> CreationResult:
    """Run at most three validation rounds under one timeout budget."""

    deadline = time.monotonic() + float(spec.operation_timeout_seconds)
    prepared = await _prepare_session(
        spec,
        frozen=frozen,
        preferred_backend_id=preferred_backend_id,
        character_projection=character_projection,
        model_gateway=model_gateway,
        identity=identity,
        deadline=deadline,
    )
    state = _ValidationState()
    for round_no in range(1, 4):
        if round_no > 1 and control is not None:
            await control.check_control()
        prompt = await _round_prompt(prepared, state=state, model_gateway=model_gateway)
        completion = await _generate_round(
            prepared,
            state=state,
            prompt=prompt,
            model_gateway=model_gateway,
            round_no=round_no,
        )
        try:
            return _accept_round(
                prepared,
                state=state,
                completion=completion,
                round_no=round_no,
            )
        except _RoundRejected as exc:
            state.mode = exc.mode
            state.preserved = exc.preserved
            rejection_normalizations = exc.normalizations
            error = str(exc).strip() or type(exc).__name__
        except BackgroundOutputError as exc:
            rejection_normalizations = ()
            error = str(exc).strip() or type(exc).__name__
        state.previous_error = error
        terminal = round_no == 3
        await _record_rejection(
            model_gateway,
            completion=completion,
            round_no=round_no,
            error=error,
            terminal=terminal,
            state=state,
            normalizations=rejection_normalizations,
        )
        if terminal:
            raise BackgroundOutputError(error)
    raise AssertionError("background creator validation loop did not terminate")


async def _prepare_session(
    spec: CreatorSessionSpec,
    *,
    frozen: FrozenBackgroundProjection,
    preferred_backend_id: str,
    character_projection: str,
    model_gateway: BackgroundModelGatewayPort,
    identity: BackgroundIdentityPort,
    deadline: float,
) -> _PreparedSession:
    identity_material = background_identity_material(
        frozen,
        additional_trusted_values=(character_projection,),
    )
    identity_context, identity_catalog = await identity.catalog(
        spec.profile_id,
        spec.instance_id,
        participant_ids=identity_material.participant_ids,
    )
    provisional_catalog = provisional_identity_catalog_text(identity_material, identity_catalog)
    definition, task_input, contract = build_creator_prompt(
        spec.author_kind,
        character_projection=character_projection,
        source=frozen,
        identity_catalog_text=provisional_catalog,
        participant_references=participant_reference_map(
            identity_context,
            identity_catalog,
            identity_material.participant_ids,
        ),
    )
    definition, task_input, contract = _project_identity(
        definition,
        task_input,
        contract,
        identity_context=identity_context,
        identity_catalog=identity_catalog,
    )
    task_input, minimum_context_tokens = await fit_routed_stage_input(
        model_gateway=model_gateway,
        profile_id=spec.profile_id,
        author_kind=spec.author_kind,
        stage="creator",
        task_definition=definition,
        task_input=task_input,
        output_contract=contract,
        output_reserve_tokens=DEFAULT_RESERVED_OUTPUT_TOKENS,
        preferred_backend_id=preferred_backend_id,
        finalize_task_input=lambda value: finalize_identity_directory(value, identity_catalog),
    )
    return _PreparedSession(
        spec=spec,
        frozen=frozen,
        preferred_backend_id=preferred_backend_id,
        deadline=deadline,
        definition=definition,
        task_input=task_input,
        contract=contract,
        minimum_context_tokens=minimum_context_tokens,
        provisional_catalog=provisional_catalog,
        identity_context=identity_context,
        identity_catalog=identity_catalog,
        visible_catalog=visible_identity_catalog(
            identity_catalog,
            (definition, str(task_input), contract),
        ),
        visible_references=visible_background_references(frozen, task_input),
    )


async def _round_prompt(
    prepared: _PreparedSession,
    *,
    state: _ValidationState,
    model_gateway: BackgroundModelGatewayPort,
) -> _RoundPrompt:
    if state.mode != REPAIR_SNAPSHOT:
        return _RoundPrompt(
            prepared.definition,
            prepared.task_input,
            prepared.contract,
            prepared.minimum_context_tokens,
        )
    if state.preserved is None or state.preserved.value is None:
        raise AssertionError("snapshot repair lost its preserved creator output")
    definition, task_input, contract = build_snapshot_repair_prompt(
        experience_text=state.preserved.experience_text,
        partial_snapshot_text=state.preserved.partial_snapshot_text,
        previous_view=prepared.frozen.source.current_view,
        authoritative_time=prepared.spec.authoritative_time,
        opening_keyframe=prepared.spec.opening_keyframe,
        identity_catalog_text=prepared.provisional_catalog,
    )
    definition, task_input, contract = _project_identity(
        definition,
        task_input,
        contract,
        identity_context=prepared.identity_context,
        identity_catalog=prepared.identity_catalog,
    )
    task_input, minimum_context_tokens = await fit_routed_stage_input(
        model_gateway=model_gateway,
        profile_id=prepared.spec.profile_id,
        author_kind=prepared.spec.author_kind,
        stage="snapshot_repair",
        task_definition=definition,
        task_input=task_input,
        output_contract=contract,
        output_reserve_tokens=DEFAULT_RESERVED_OUTPUT_TOKENS,
        preferred_backend_id=prepared.preferred_backend_id,
        finalize_task_input=lambda value: finalize_identity_directory(
            value,
            prepared.identity_catalog,
        ),
    )
    return _RoundPrompt(definition, task_input, contract, minimum_context_tokens)


async def _generate_round(
    prepared: _PreparedSession,
    *,
    state: _ValidationState,
    prompt: _RoundPrompt,
    model_gateway: BackgroundModelGatewayPort,
    round_no: int,
) -> Any:
    spec = prepared.spec
    remaining = _remaining_timeout(prepared.deadline)
    async with asyncio.timeout(remaining):
        return await model_gateway.generate_text(
            task_definition=prompt.definition,
            task_input=prompt.task_input,
            output_contract=prompt.contract,
            execution_record=_validation_feedback(state.previous_error, mode=state.mode),
            profile_id=spec.profile_id,
            instance_id=spec.instance_id,
            capability="text.completion",
            preferred_backend_id=prepared.preferred_backend_id,
            minimum_context_tokens=prompt.minimum_context_tokens,
            owner_kind="background_author",
            owner_id=str(spec.task_id),
            idempotency_key=f"{spec.logical_stage_key}:round:{round_no}",
            work_purpose=_WORK_PURPOSE[spec.author_kind],
            logical_stage_key=spec.logical_stage_key,
            round_no=round_no,
            operation_timeout_seconds=remaining,
        )


def _accept_round(
    prepared: _PreparedSession,
    *,
    state: _ValidationState,
    completion: Any,
    round_no: int,
) -> CreationResult:
    if _completion_was_truncated(completion):
        error = "上一份输出在结尾处被截断，不能沿用半截正文；请重新输出更短的完整结果"
        raise _RoundRejected(
            error,
            mode=state.mode if state.mode == REPAIR_SNAPSHOT else "FULL",
            preserved=state.preserved if state.mode == REPAIR_SNAPSHOT else None,
        )
    if state.mode == REPAIR_SNAPSHOT:
        creator, normalizations = _merge_snapshot_repair(prepared, state, completion)
        repair_kind = REPAIR_SNAPSHOT
    else:
        creator, normalizations = _recover_full_output(prepared, completion)
        repair_kind = "FULL_RETRY" if round_no > 1 else ""
    creator, reference_normalizations = normalize_optional_references(
        prepared.spec.author_kind,
        creator,
        visible_references=prepared.visible_references,
    )
    normalizations = unique((*normalizations, *reference_normalizations))
    decoded = decode_visible_identity_data(
        creator,
        prepared.visible_catalog,
        scope=str(prepared.identity_context.scope),
    )
    draft = draft_from_creator(
        prepared.spec.author_kind,
        decoded,
        source=prepared.frozen.source,
        visible_references=prepared.visible_references,
    )
    return CreationResult(
        creator=decoded,
        draft=draft,
        completion=completion,
        round_no=round_no,
        normalizations=normalizations,
        repair_kind=repair_kind,
    )


def _merge_snapshot_repair(
    prepared: _PreparedSession,
    state: _ValidationState,
    completion: Any,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    preserved = state.preserved
    if preserved is None or preserved.value is None:
        raise AssertionError("snapshot repair lost its preserved creator output")
    repaired = parse_snapshot_repair_output(
        completion.text,
        opening_keyframe=prepared.spec.opening_keyframe,
        authoritative_time=prepared.spec.authoritative_time,
    )
    creator = dict(preserved.value)
    creator["current_view"] = repaired.current_view
    return creator, unique((*preserved.normalizations, *repaired.normalizations))


def _recover_full_output(
    prepared: _PreparedSession,
    completion: Any,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    recovered = recover_creator_output(
        completion.text,
        author_kind=prepared.spec.author_kind,
        opening_keyframe=prepared.spec.opening_keyframe,
        authoritative_time=prepared.spec.authoritative_time,
    )
    if recovered.repair_kind == REPAIR_SNAPSHOT:
        raise _RoundRejected(
            recovered.error,
            mode=REPAIR_SNAPSHOT,
            preserved=recovered,
            normalizations=recovered.normalizations,
        )
    if not recovered.accepted or recovered.value is None:
        raise _RoundRejected(
            recovered.error,
            mode="FULL",
            preserved=None,
            normalizations=recovered.normalizations,
        )
    return recovered.value, recovered.normalizations


async def _record_rejection(
    model_gateway: BackgroundModelGatewayPort,
    *,
    completion: Any,
    round_no: int,
    error: str,
    terminal: bool,
    state: _ValidationState,
    normalizations: tuple[str, ...],
) -> None:
    processing: dict[str, object] = {"repair_kind": state.mode}
    if normalizations:
        processing["normalizations"] = [{"action": action} for action in normalizations]
    if state.preserved is not None and state.mode == REPAIR_SNAPSHOT:
        processing["preserved_sections"] = ["经历", "可选动作"]
    await record_structured_rejection(
        model_gateway=model_gateway,
        completion=completion,
        round_no=round_no,
        error=error,
        terminal=terminal,
        extra_processing=processing,
    )


def _project_identity(
    definition: str,
    task_input: TrustedPromptMarkup,
    contract: str,
    *,
    identity_context: IdentityRenderContext,
    identity_catalog: IdentityCatalog,
) -> tuple[str, TrustedPromptMarkup, str]:
    scope = str(identity_context.scope)
    projected_definition = project_identity_text_for_model(
        definition,
        identity_catalog,
        scope=scope,
    )
    projected_input = TrustedPromptMarkup(
        project_prompt_text(
            task_input,
            lambda value: project_identity_text_for_model(
                value,
                identity_catalog,
                scope=scope,
            ),
        )
    )
    projected_contract = project_identity_text_for_model(
        contract,
        identity_catalog,
        scope=scope,
    )
    return projected_definition, projected_input, projected_contract


def _validation_feedback(error: str, *, mode: str) -> str:
    detail = _model_visible_validation_error(error)
    if not detail:
        return ""
    if mode == REPAIR_SNAPSHOT:
        return f"上一份角色状态还不能使用。不要续写或重写经历，只重新写出完整状态：\n{detail}"
    return (
        "上一份内容还不能使用。不要接着它续写，也不要解释；请重新写出一份更短、"
        f"完整的结果：\n{detail}"
    )


def _model_visible_validation_error(error: str) -> str:
    detail = str(error or "").strip()
    if not detail:
        return ""
    exact = {
        "background creator returned no usable blocks": "没有写出可识别的内容区块",
        "background creator returned empty output": "没有写出任何内容",
        "background creator returned an unknown outer wrapper": "使用了本任务不接受的外层区块",
        "background creator contains ambiguous text outside its blocks": (
            "规定区块之外还有无法判断归属的文字"
        ),
        "生活帧没有可保留的经历正文": "没有写出完整可用的经历正文",
    }
    if detail in exact:
        return exact[detail]
    if detail.startswith("background creator returned incompatible outer tag:"):
        return "使用了本任务不接受的区块"
    if detail.startswith("background creator has mismatched tags:"):
        return "开始标签与结束标签不一致"
    if detail.startswith("background creator closing tag "):
        return "结束标签没有对应的开始标签"
    if detail.startswith("creator can only return "):
        return "写出了本任务不接受的区块"
    if detail.startswith("creator must return "):
        return "缺少本任务要求的区块，或同一区块数量不对"
    if detail.startswith("留下变化已解决 ordinal"):
        if "no active leftover" in detail:
            return "“留下变化已解决”所填编号没有仍待解决的变化"
        return "“留下变化已解决”只能填写眼前资料中实际存在的 V 编号"
    if detail.startswith("介入模组/模组已了结 ordinals contain an empty item"):
        return "“介入模组/模组已了结”中不能出现空编号"
    if detail.startswith("介入模组/模组已了结 ordinals must reference at most"):
        return "“介入模组/模组已了结”最多填写三个 M 编号"
    if detail.startswith("介入模组/模组已了结 ordinal"):
        return "“介入模组/模组已了结”只能填写眼前资料中实际存在的 M 编号"
    return _model_visible_field_or_fallback(detail)


def _model_visible_field_or_fallback(detail: str) -> str:
    required_fields = {
        "world change": "世界变化",
        "life direction": "人生方向",
        "story module": "故事模组",
        "life event content": "经历正文",
        "current role narrative time": "角色现在的时间",
        "current role location": "角色现在的地点",
        "current role activity": "角色正在做的事",
        "current role body state": "角色的身体状态",
        "current role mood": "角色的心情",
        "current role intention": "角色当前的打算",
        "current role concern": "角色当前的牵挂",
    }
    if detail.endswith(" is required"):
        field = detail.removesuffix(" is required")
        return f"缺少{required_fields.get(field, '必填内容')}"
    if re.search(r"[A-Za-z_]{3,}", detail):
        return "区块、字段或编号不符合本次写作要求"
    return detail


def _remaining_timeout(deadline: float) -> float:
    remaining = float(deadline) - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("background creator validation session timed out")
    return max(1.0, remaining)


def _completion_was_truncated(completion: Any) -> bool:
    reason = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(getattr(completion, "finish_reason", "") or "").strip().casefold(),
    ).strip("_")
    return reason in {
        "length",
        "max_tokens",
        "max_output_tokens",
        "token_limit",
        "output_token_limit",
    }


def unique(values: Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))


__all__ = [
    "CreationResult",
    "CreatorSessionSpec",
    "run_creator_session",
]
