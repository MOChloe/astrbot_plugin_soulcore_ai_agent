"""Final, command-free role-voice polishing for Main Core expression batches."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Any

from ...contracts.ai_models import (
    AIBackendDescriptor,
    AIPromptCacheSemanticKind,
)
from ...contracts.models import CoreWakeRequest, ScopeConfig
from ...contracts.runtime_limits import DEFAULT_RESPONSE_POLISH_TIMEOUT_SECONDS
from ...shared.prompt_document import (
    PromptBlock,
    PromptCacheBoundary,
    compile_prompt_document,
    xml_text,
)
from ...shared.token_meter import ConservativeTokenMeter
from ..ai.service import (
    CommandProtocolError,
    MainCoreCommandRegistry,
    ParsedCommand,
    ParsedModelTurn,
    parse_model_turn,
    terminal_decision,
)
from ..identity import (
    decode_model_parameter_map,
    encode_identity_template_for_model,
    project_identity_text_for_model,
)
from .command_context import validate_command_availability_claims
from .expression_timeline import (
    has_voice_expression_metadata,
    normalize_expression_steps,
    restore_expression_voice_metadata,
    restore_unbound_voice_parse_audit,
)
from .response_polish_context import (
    RESPONSE_POLISH_INTERNAL_CONTEXT_TOKENS,
    RESPONSE_POLISH_LOCAL_CONTEXT_TOKENS,
)
from .response_polish_context import project_internal_materials as _project_internal_materials
from .response_polish_context import (
    project_polish_conversation as _project_polish_conversation,
)
from .roleplay_prompt import RolePlayPromptCompiler


def project_polish_styles(
    speaking_style: str,
    writing_correction: str,
    identity_context: Any | None,
    identity_catalog: Any | None,
) -> tuple[str, str]:
    speaking_style_text = str(speaking_style or "").strip()
    writing_correction_text = str(writing_correction or "").strip()
    if identity_context is not None and identity_catalog is not None:
        scope = str(identity_context.scope)
        if speaking_style_text:
            speaking_style_text = project_identity_text_for_model(
                speaking_style_text,
                identity_catalog,
                scope=scope,
            )
        if writing_correction_text:
            writing_correction_text = project_identity_text_for_model(
                writing_correction_text,
                identity_catalog,
                scope=scope,
            )
    return speaking_style_text, writing_correction_text


def project_polish_persona(
    persona: str,
    identity_context: Any | None,
    identity_catalog: Any | None,
) -> str:
    persona_text = str(persona or "").strip()
    if identity_context is not None and identity_catalog is not None:
        persona_text = project_identity_text_for_model(
            persona_text,
            identity_catalog,
            scope=str(identity_context.scope),
        )
    return persona_text


RESPONSE_POLISH_CAPABILITY = "conversation.response_polish"
RESPONSE_POLISH_TIMEOUT_SECONDS = float(DEFAULT_RESPONSE_POLISH_TIMEOUT_SECONDS)


class ResponsePolishContractError(ValueError):
    """The polishing request or response could not satisfy its safe contract."""


@dataclass(frozen=True, slots=True)
class ResponsePolishResult:
    visible_steps: tuple[dict[str, Any], ...]
    audit: dict[str, Any]


def _task_definition(*, freeze_visible_topology: bool) -> str:
    structure_contract = (
        """消息的数量、类型和顺序保持原样。你可以调整每条消息的文字、语气、发送延迟和可打断
状态，但不增减消息，不改变它们之间的回复、提及或撤回关系。"""
        if freeze_visible_topology
        else """纯重复的文字气泡可以去掉；连续的文字气泡可以合并或重新拆分。图片、表情、文件
的相对顺序固定——它们仍留在原来相邻的文字段之间，不跨过文字段或其他资源。"""
    )
    return f"""《原始表达》已经确定了本轮想传达的事情和行动结果，但它只是尚未通过语言验收的草稿。你的首要职责是逐句质检并救回不合格的表达，最终交付这个人在当前即时通讯里真正会发出的消息；不能因为原稿语法完整、格式正确或大致看得懂，就默认它已经可以发送。

先结合角色信息、近期对话和当前输入，判断每条文字此刻真正想做什么，再按字面逐句读，并把整批连起来读。合格的文字必须意思清楚，搭配、指代、比喻与上下句关系正常，确实接在眼前的话题上；其中假定的动作、现场、亲密程度和关系姿态也必须在当前交流中成立。只是在摆出温柔、深情、强势、聪明或会照顾人的样子，却没有从当前人物、关系和现场里长出来的通用台词，不算合格。

任何一句不满足这些条件，就保留它真正承载的意图与事实，丢掉原句从头写；必要时重组整批、补上自然衔接、删掉没有实际内容的外壳，或者改成更朴素直接的话。不要沿着坏句只换近义词，也不要把一句修辞机械当成已经发生的事实继续往下接。只有原稿每一句都通过语言验收时，才可以原样沿用。不要输出质检说明。

改写不能翻转已经确定的决定和硬事实：同意还是拒绝、确定还是犹豫、承诺还是不承诺、对对方的核心期待与边界、关键人物与对象、事情成了还是没成。也不能让对方少知道《原始表达》已经明确给出的新人物、行动、打算、问题、原因、条件、时间或状态变化；同批重复的意思与不承载信息的口头外壳可以删除。单独发送的图片、表情包和文件资源项原样保留，不可新增、删除或替换；哪条消息回复谁、提及谁、撤回哪条，这些关系不动。文字中的 emoji、颜文字和其他文字表情仍属于可重写文字。

《原始表达》是本轮对外内容的唯一事实来源。《原稿形成时的相关考虑》和《发送后连续性留话》只帮助理解原稿已经做出的取舍，不得把只在其中出现的事实、计划、解释、人物、时间或事件新增到消息中；留话不属于改写内容，不要输出、复述或修改。角色信息、说话风格、临场对话以及可选的《表达修正参考》用于判断怎样表达，也不能成为新决定、新事实或新行动的来源。

{structure_contract}"""


def _polish_registry(*, include_files: bool = True) -> MainCoreCommandRegistry:
    registry = MainCoreCommandRegistry.terminal_only()
    return MainCoreCommandRegistry(
        tuple(
            replace(
                spec,
                parameters=tuple(
                    parameter
                    for parameter in spec.parameters
                    if parameter.label not in {"回复", "提及", "语音"}
                    and parameter.internal_name != "memo"
                ),
                usage_guidance="",
            )
            for spec in registry.specs
            if spec.send_kind
            in (
                {"TEXT", "IMAGE", "STICKER", "FILE"}
                if include_files
                else {"TEXT", "IMAGE", "STICKER"}
            )
        )
    )


def _polish_output_contract(*, include_files: bool, freeze_visible_topology: bool) -> str:
    resource_lines = [
        "- `<发图片>`：必填 `[[图片]]`，只能填写原始表达中已有的图片短引用。",
        "- `<发表情>`：必填 `[[表情]]`，只能填写原始表达中已有的表情短引用。",
    ]
    if include_files:
        resource_lines.append("- `<发文件>`：必填 `[[文件]]`，只能填写原始表达中已有的文件短引用。")
    rhythm_contract = (
        "可以根据改写后各条文字调整延迟，但不得借此改变消息数量、类型或顺序。"
        if freeze_visible_topology
        else "默认保留原始表达的发送节奏；合并或拆分连续文字气泡时，才按新文字段相应调整延迟。"
    )
    return "\n".join(
        (
            "发送指令必须使用下列真实格式；标签各自独占一行，参数写成 `[[参数名]]: 内容`。",
            "每个指令以同名结束标签收尾。一个指令对应一条实际发送的消息，可以按发送顺序连续输出多个指令。",
            "- `<发文字>`：必填 `[[内容]]`，内容不能为空。",
            *resource_lines,
            "每个发送指令可填写 `[[延迟]]` 与 `[[可被打断]]`。延迟是相对上一项的 0 至 120 整数秒，"
            "全部延迟合计不超过 300 秒；可被打断只填写“是”或“否”。不需要的可选参数删除整行。",
            f"{rhythm_contract}不要猜测未提供的设备或输入速度。",
            "只输出上面列出的发送指令。",
        )
    )


def _contains_file_expression(expressions: Sequence[Mapping[str, Any]]) -> bool:
    return any(str(item.get("kind") or "").strip().upper() == "FILE" for item in expressions)


def _bind_short_reference(
    internal: str,
    prefix: str,
    *,
    reference_map: dict[str, Any],
    internal_to_public: dict[str, str],
) -> str:
    value = str(internal or "").strip()
    if not value:
        return ""
    existing = internal_to_public.get(value)
    if existing:
        return existing
    index = 1
    while f"{prefix}{index}" in reference_map:
        index += 1
    public = f"{prefix}{index}"
    reference_map[public] = value
    internal_to_public[value] = public
    return public


def _render_original_commands(
    expressions: Sequence[Mapping[str, Any]],
    *,
    reference_map: dict[str, Any],
    internal_to_public: dict[str, str],
    identity_catalog: Any | None = None,
) -> str:
    names = {
        "TEXT": ("发文字", "内容", ""),
        "IMAGE": ("发图片", "图片", "I"),
        "STICKER": ("发表情", "表情", "S"),
        "FILE": ("发文件", "文件", "F"),
    }
    rendered: list[str] = []
    for item in expressions:
        block = _render_original_command(
            item,
            names=names,
            reference_map=reference_map,
            internal_to_public=internal_to_public,
            identity_catalog=identity_catalog,
        )
        if block:
            rendered.append(block)
    return "\n\n".join(rendered)


def _render_original_command(
    item: Mapping[str, Any],
    *,
    names: Mapping[str, tuple[str, str, str]],
    reference_map: dict[str, Any],
    internal_to_public: dict[str, str],
    identity_catalog: Any | None = None,
) -> str:
    kind = str(item.get("kind") or "").strip().upper()
    descriptor = names.get(kind)
    if descriptor is None:
        return ""
    name, content_label, prefix = descriptor
    content = (
        xml_text(
            encode_identity_template_for_model(str(item.get("text") or ""), identity_catalog)
            if identity_catalog is not None
            else item.get("text")
        )
        if kind == "TEXT"
        else _bind_short_reference(
            str(item.get("asset_ref_id") or ""),
            prefix,
            reference_map=reference_map,
            internal_to_public=internal_to_public,
        )
    )
    lines = [f"<{name}>", f"[[{content_label}]]: {content}"]
    lines.extend(
        (
            f"[[延迟]]: {int(item.get('delay_after_previous_seconds') or 0)}",
            "[[可被打断]]: " + ("是" if bool(item.get("can_be_interrupted", True)) else "否"),
            f"</{name}>",
        )
    )
    return "\n".join(lines)


def _compile_polish_document(
    *,
    persona: str,
    speaking_style: str,
    writing_correction: str,
    dialogue: Sequence[str],
    current: str,
    internal_working_text: str,
    internal_memos: str,
    original_commands: str,
    registry: MainCoreCommandRegistry,
    reference_map: Mapping[str, Any],
    model_id: str,
    identity_catalog_text: str,
    freeze_visible_topology: bool,
    trim_reasons: Sequence[str] = (),
) -> Any:
    include_files = any(spec.send_kind == "FILE" for spec in registry.specs)
    dialogue_text = "\n".join(dialogue)
    context_blocks = [
        PromptBlock(
            "任务定义",
            _task_definition(freeze_visible_topology=freeze_visible_topology),
        ),
        PromptBlock(
            "发送格式",
            _polish_output_contract(
                include_files=include_files,
                freeze_visible_topology=freeze_visible_topology,
            ),
            cache_boundaries=(
                PromptCacheBoundary(
                    "polish-protocol",
                    AIPromptCacheSemanticKind.PROTOCOL,
                    1,
                    selection_reason="润色任务协议末端",
                ),
            ),
        ),
        PromptBlock("说话风格", xml_text(speaking_style)),
        PromptBlock(
            "角色信息",
            xml_text(persona),
            cache_boundaries=(
                PromptCacheBoundary(
                    "polish-character",
                    AIPromptCacheSemanticKind.CONTEXT,
                    2,
                    selection_reason="润色角色信息末端",
                ),
            ),
        ),
        PromptBlock(
            "近期对话",
            dialogue_text,
            cache_boundaries=(
                PromptCacheBoundary(
                    "polish-dialogue",
                    AIPromptCacheSemanticKind.CURRENT_DIALOGUE,
                    3,
                    content_end=len(dialogue_text),
                    selection_reason="润色局部对话末端",
                ),
            )
            if dialogue_text
            else (),
        ),
    ]
    turn_blocks = [
        PromptBlock("当前输入", current),
        PromptBlock("身份引用", xml_text(identity_catalog_text)),
    ]
    if internal_working_text:
        turn_blocks.append(PromptBlock("原稿形成时的相关考虑", xml_text(internal_working_text)))
    if internal_memos:
        turn_blocks.append(PromptBlock("发送后连续性留话", xml_text(internal_memos)))
    if writing_correction:
        turn_blocks.append(PromptBlock("表达修正参考", xml_text(writing_correction)))
    turn_blocks.append(PromptBlock("原始表达", original_commands))
    return compile_prompt_document(
        context_blocks,
        turn_blocks,
        model_id=model_id,
        reference_map=reference_map,
        trim_reasons=trim_reasons,
    )


def _compile_polish_with_model_window(
    *,
    persona: str,
    speaking_style: str,
    writing_correction: str,
    dialogue: list[str],
    current: str,
    internal_working_text: str,
    internal_memos: str,
    original_commands: str,
    registry: MainCoreCommandRegistry,
    reference_map: Mapping[str, Any],
    model_id: str,
    identity_catalog: Any | None,
    freeze_visible_topology: bool,
    input_limit: int,
    dropped: int,
    trim_reasons: tuple[str, ...],
) -> tuple[Any, int]:
    compiled = _compile_polish_document(
        persona=persona,
        speaking_style=speaking_style,
        writing_correction=writing_correction,
        dialogue=dialogue,
        current=current,
        internal_working_text=internal_working_text,
        internal_memos=internal_memos,
        original_commands=original_commands,
        registry=registry,
        reference_map=reference_map,
        model_id=model_id,
        identity_catalog_text=identity_catalog.prompt_text() if identity_catalog else "",
        freeze_visible_topology=freeze_visible_topology,
        trim_reasons=trim_reasons,
    )
    if (internal_working_text or internal_memos) and compiled.total_tokens > input_limit:
        internal_working_text = ""
        internal_memos = ""
        trim_reasons = tuple(
            dict.fromkeys((*trim_reasons, "response_polish_model_window_internal_context"))
        )
        compiled = _compile_polish_document(
            persona=persona,
            speaking_style=speaking_style,
            writing_correction=writing_correction,
            dialogue=dialogue,
            current=current,
            internal_working_text=internal_working_text,
            internal_memos=internal_memos,
            original_commands=original_commands,
            registry=registry,
            reference_map=reference_map,
            model_id=model_id,
            identity_catalog_text=identity_catalog.prompt_text() if identity_catalog else "",
            freeze_visible_topology=freeze_visible_topology,
            trim_reasons=trim_reasons,
        )
    while dialogue and compiled.total_tokens > input_limit:
        dialogue.pop(0)
        dropped += 1
        trim_reasons = tuple(dict.fromkeys((*trim_reasons, "response_polish_model_window")))
        compiled = _compile_polish_document(
            persona=persona,
            speaking_style=speaking_style,
            writing_correction=writing_correction,
            dialogue=dialogue,
            current=current,
            internal_working_text=internal_working_text,
            internal_memos=internal_memos,
            original_commands=original_commands,
            registry=registry,
            reference_map=reference_map,
            model_id=model_id,
            identity_catalog_text=identity_catalog.prompt_text() if identity_catalog else "",
            freeze_visible_topology=freeze_visible_topology,
            trim_reasons=trim_reasons,
        )
    if input_limit < 1 or compiled.total_tokens > input_limit:
        raise ResponsePolishContractError("protected_polish_context_exceeds_model_window")
    return compiled, dropped


def build_polish_prompt(
    *,
    request: CoreWakeRequest,
    persona: str,
    prepared_context: Any,
    expressions: Sequence[Mapping[str, Any]],
    model_id: str,
    max_context_tokens: int,
    working_text: str = "",
    freeze_visible_topology: bool = False,
    speaking_style: str = "",
    writing_correction: str = "",
) -> tuple[Any, int, int]:
    """Build one bounded XML document with no internal Provider-message shape."""

    conversation = RolePlayPromptCompiler().project_conversation(
        prepared_context,
        fallback_input=str(request.user_message or ""),
        occurred_at=request.requested_at,
    )
    reference_map = dict(conversation.public_to_internal)
    internal_to_public = dict(conversation.internal_to_public)
    identity_context, identity_catalog = _polish_identities(prepared_context)
    speaking_style_text, writing_correction_text = project_polish_styles(
        speaking_style,
        writing_correction,
        identity_context,
        identity_catalog,
    )
    original_commands = _render_original_commands(
        expressions,
        reference_map=reference_map,
        internal_to_public=internal_to_public,
        identity_catalog=identity_catalog,
    )
    persona_text = project_polish_persona(persona, identity_context, identity_catalog)
    meter = ConservativeTokenMeter(model_id)
    internal_working_text, internal_memos, internal_trimmed = _project_internal_materials(
        working_text,
        expressions,
        identity_context,
        identity_catalog,
        meter=meter,
    )
    dialogue, current, dropped, current_trimmed = _project_polish_conversation(
        conversation,
        request,
        identity_context,
        identity_catalog,
        meter=meter,
    )
    registry = _polish_registry(include_files=_contains_file_expression(expressions))
    output_tokens = max(512, min(8192, meter.count_text(original_commands) * 2 + 256))
    input_limit = int(max_context_tokens) - output_tokens
    trim_reasons = tuple(
        reason
        for reason, active in (
            ("response_polish_local_dialogue_budget", dropped > 0),
            ("response_polish_current_input_budget", current_trimmed),
            ("response_polish_internal_context_budget", internal_trimmed),
        )
        if active
    )
    compiled, dropped = _compile_polish_with_model_window(
        persona=persona_text,
        speaking_style=speaking_style_text,
        writing_correction=writing_correction_text,
        dialogue=dialogue,
        current=current,
        internal_working_text=internal_working_text,
        internal_memos=internal_memos,
        original_commands=original_commands,
        registry=registry,
        reference_map=reference_map,
        model_id=model_id,
        identity_catalog=identity_catalog,
        freeze_visible_topology=freeze_visible_topology,
        input_limit=input_limit,
        dropped=dropped,
        trim_reasons=trim_reasons,
    )
    return compiled, output_tokens, dropped


def _polish_identities(prepared_context: Any) -> tuple[Any | None, Any | None]:
    if prepared_context is None:
        return None, None
    return prepared_context.identity_context, prepared_context.identity_catalog


def _non_text_multiset(expressions: Sequence[Mapping[str, Any]]) -> Counter[tuple[str, str]]:
    return Counter(
        (
            str(item.get("kind") or "").strip().upper(),
            str(item.get("asset_ref_id") or "").strip(),
        )
        for item in expressions
        if str(item.get("kind") or "").strip().upper() != "TEXT"
    )


def _resource_text_topology(
    expressions: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str], ...]:
    """Keep each resource in the same text run boundary without freezing text bubbles."""

    topology: list[tuple[str, str]] = []
    inside_text_run = False
    for item in expressions:
        kind = str(item.get("kind") or "").strip().upper()
        if kind == "TEXT":
            if not inside_text_run:
                topology.append(("TEXT", ""))
            inside_text_run = True
            continue
        inside_text_run = False
        topology.append((kind, str(item.get("asset_ref_id") or "").strip()))
    return tuple(topology)


def validate_polished_expression_steps(
    text: str,
    *,
    original_steps: Sequence[Mapping[str, Any]],
    collector: Any,
    reference_map: Mapping[str, Any],
    dialogue_reference: str = "",
    internal_reference: str = "",
    identity_catalog: Any | None = None,
    identity_scope: str = "profile",
    freeze_visible_topology: bool = False,
) -> list[dict[str, Any]]:
    freeze_visible_topology = freeze_visible_topology or has_voice_expression_metadata(
        original_steps
    )
    decision = _polish_decision(
        text,
        reference_map,
        identity_catalog=identity_catalog,
        identity_scope=identity_scope,
    )
    normalized, error = normalize_expression_steps(list(decision["expression_steps"]))
    if error:
        raise ResponsePolishContractError(error.removeprefix("error: ").strip())
    visible_text = _validate_polished_semantics(
        normalized,
        original_steps,
        collector,
        freeze_visible_topology=freeze_visible_topology,
    )
    _reject_reference_copy(visible_text, dialogue_reference)
    original_visible_text = "\n".join(
        str(item.get("text") or "")
        for item in original_steps
        if str(item.get("kind") or "").strip().upper() == "TEXT"
    )
    _reject_reference_copy(
        visible_text,
        internal_reference,
        reason_prefix="internal_reference",
        allowed_text=original_visible_text,
    )
    return _restore_expression_memos(normalized, original_steps)


def _polish_decision(
    text: str,
    reference_map: Mapping[str, Any],
    *,
    identity_catalog: Any | None = None,
    identity_scope: str = "profile",
) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ResponsePolishContractError("empty_output")
    registry = _polish_registry()
    parsed = parse_model_turn(raw)
    if parsed.errors:
        raise ResponsePolishContractError("invalid_command_protocol: " + "；".join(parsed.errors))
    if not parsed.commands:
        raise ResponsePolishContractError("no_send_commands")
    if identity_catalog is not None:
        parsed = ParsedModelTurn(
            working_text=parsed.working_text,
            commands=tuple(
                ParsedCommand(
                    name=command.name,
                    parameters=decode_model_parameter_map(
                        command.parameters,
                        identity_catalog,
                        scope=identity_scope,
                    ),
                    ordinal=command.ordinal,
                    raw_text=command.raw_text,
                )
                for command in parsed.commands
            ),
            errors=parsed.errors,
            raw_text=parsed.raw_text,
        )
    try:
        validated = []
        for command in parsed.commands:
            validated.append(registry.validate(command, reference_map))
        return terminal_decision(validated, reference_map)
    except CommandProtocolError as exc:
        raise ResponsePolishContractError(f"invalid_send_command: {exc}") from exc


def _validate_polished_semantics(
    normalized: Sequence[Mapping[str, Any]],
    original_steps: Sequence[Mapping[str, Any]],
    collector: Any,
    *,
    freeze_visible_topology: bool,
) -> str:
    visible_text = "\n".join(
        str(item.get("text") or "") for item in normalized if item.get("kind") == "TEXT"
    ).strip()
    original_had_text = any(
        str(item.get("kind") or "").strip().upper() == "TEXT" for item in original_steps
    )
    if original_had_text and not visible_text:
        raise ResponsePolishContractError("polished_batch_lost_all_text")
    if freeze_visible_topology and _visible_kind_sequence(normalized) != _visible_kind_sequence(
        original_steps
    ):
        raise ResponsePolishContractError("frozen_visible_timeline_changed")
    if _non_text_multiset(normalized) != _non_text_multiset(original_steps):
        raise ResponsePolishContractError("non_text_expressions_changed")
    if _resource_text_topology(normalized) != _resource_text_topology(original_steps):
        raise ResponsePolishContractError("non_text_expression_order_or_placement_changed")
    if validate_command_availability_claims(collector, visible_text):
        raise ResponsePolishContractError("invented_command_failure_claim")
    return visible_text


def _visible_kind_sequence(
    expressions: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    return tuple(str(item.get("kind") or "").strip().upper() for item in expressions)


def _restore_expression_memos(
    normalized: Sequence[Mapping[str, Any]],
    original_steps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    has_memos = any(str(item.get("memo") or "").strip() for item in original_steps)
    if not has_memos and not has_voice_expression_metadata(original_steps):
        return restore_unbound_voice_parse_audit(original_steps, normalized)
    if _visible_kind_sequence(normalized) != _visible_kind_sequence(original_steps):
        raise ResponsePolishContractError("frozen_visible_timeline_changed")
    restored: list[dict[str, Any]] = []
    for replacement, original in zip(normalized, original_steps, strict=True):
        step = restore_expression_voice_metadata(original, replacement)
        step.pop("memo", None)
        memo = str(original.get("memo") or "").strip()
        if memo:
            step["memo"] = memo
        restored.append(step)
    return restored


def _copy_text(value: str) -> str:
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


def _reject_reference_copy(
    visible_text: str,
    reference_text: str,
    *,
    reason_prefix: str = "dialogue_reference",
    allowed_text: str = "",
) -> None:
    candidate = _copy_text(visible_text)
    allowed = _copy_text(allowed_text)
    if not candidate:
        return
    for raw_example in str(reference_text or "").splitlines():
        example = _copy_text(raw_example)
        if len(example) < 12:
            continue
        matcher = SequenceMatcher(None, example, candidate, autojunk=False)
        match = matcher.find_longest_match()
        longest = match.size
        contiguous_limit = min(24, max(12, len(example) // 2))
        matched_text = candidate[match.b : match.b + match.size]
        copied_only_from_allowed = bool(matched_text and matched_text in allowed)
        if longest >= contiguous_limit and not copied_only_from_allowed:
            raise ResponsePolishContractError(f"{reason_prefix}_contiguous_copy")
        candidate_matches_allowed = bool(
            allowed
            and (
                candidate in allowed
                or SequenceMatcher(None, allowed, candidate, autojunk=False).ratio() >= 0.82
            )
        )
        if len(example) >= 16 and matcher.ratio() >= 0.82 and not candidate_matches_allowed:
            raise ResponsePolishContractError(f"{reason_prefix}_high_similarity")


def _context_limit(role: ScopeConfig, backend_hint: AIBackendDescriptor) -> int:
    metadata = dict(backend_hint.metadata)
    raw = metadata.get("max_context_tokens") or role.max_context_tokens
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 128000


__all__ = [
    "RESPONSE_POLISH_CAPABILITY",
    "RESPONSE_POLISH_INTERNAL_CONTEXT_TOKENS",
    "RESPONSE_POLISH_LOCAL_CONTEXT_TOKENS",
    "RESPONSE_POLISH_TIMEOUT_SECONDS",
    "ResponsePolishContractError",
    "ResponsePolishResult",
    "build_polish_prompt",
    "validate_polished_expression_steps",
]
