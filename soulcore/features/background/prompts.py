"""Compile one free-writing request for each background layer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...shared.prompt_document import (
    TrustedPromptMarkup,
    join_prompt_markup,
    prompt_field_lines,
    prompt_markup_block,
)
from ..ai.service import DEFAULT_RESERVED_OUTPUT_TOKENS
from .domain import (
    BackgroundAuthorInput,
    BackgroundAuthorKind,
    BackgroundAuthorState,
    BackgroundInitializationStep,
)
from .identity_projection import IDENTITY_BLOCK_NAME
from .prompt_budget import (
    BackgroundPromptBudget,
    FrozenBackgroundProjection,
    fit_stage_markup,
    freeze_background_projection,
)
from .prompt_definitions import (
    BOUNDARY_FIELDS,
    LAYER_IDENTITY,
    LORE_FIELDS,
    NARRATIVE_CRAFT,
    OPENING_DEFINITION_NOTES,
    SEED_FIELDS,
    SNAPSHOT_REPAIR_DEFINITION,
    creator_definition_suffix,
    creator_publication_contract,
    snapshot_repair_contract,
)
from .prompt_rendering import (
    RECENT_FOREGROUND_BLOCK_NAME,
    ROLE_CURRENT_STATE_BLOCK_NAME,
    ROLE_LOCATION_BLOCK_NAME,
    ROLE_PROFILE_BLOCK_NAME,
    foreground_handoff_block,
    frame_window_block,
    life_direction_block,
    location_block,
    mapping_block,
    prompt_time_block,
    records_block,
    story_source_blocks,
    timeline_block,
    trusted_identity_text,
    view_block,
    world_changes_block,
)

_CREATIVE_POSTURE_BLOCK_NAME = "创作姿态"
_SNAPSHOT_EXPERIENCE_TAIL_CHARS = 6000


def build_creator_prompt(
    author_kind: BackgroundAuthorKind,
    *,
    character_projection: str,
    source: BackgroundAuthorInput | FrozenBackgroundProjection,
    identity_catalog_text: str,
    participant_references: Mapping[str, str],
    budget: BackgroundPromptBudget | None = None,
) -> tuple[str, TrustedPromptMarkup, str]:
    frozen = _projection(author_kind, source)
    opening = _is_opening_pass(author_kind, frozen.source)
    definition = LAYER_IDENTITY[author_kind]
    if author_kind in {
        BackgroundAuthorKind.STORY_SOURCE,
        BackgroundAuthorKind.KEYFRAME,
        BackgroundAuthorKind.ORDINARY,
    }:
        definition = f"{definition}\n\n{NARRATIVE_CRAFT}"
    if opening:
        definition = f"{definition}\n\n{OPENING_DEFINITION_NOTES[author_kind]}"
    definition = f"{definition}\n\n{creator_definition_suffix(author_kind)}"
    contract = creator_publication_contract(
        author_kind,
        opening_keyframe=(opening and author_kind is BackgroundAuthorKind.KEYFRAME),
    )
    task_input = _source_markup(
        author_kind,
        frozen,
        character_projection=character_projection,
        identity_catalog_text=identity_catalog_text,
        participant_references=participant_references,
    )
    return (
        definition,
        _fit(author_kind, definition, contract, task_input, budget),
        contract,
    )


def build_snapshot_repair_prompt(
    *,
    experience_text: str,
    partial_snapshot_text: str,
    previous_view: object,
    authoritative_time: str,
    opening_keyframe: bool,
    identity_catalog_text: str = "",
) -> tuple[str, TrustedPromptMarkup, str]:
    """Build the narrow second-round request that may only replace the snapshot."""

    experience = str(experience_text or "").strip()
    if not experience:
        raise ValueError("snapshot repair requires preserved experience text")
    tail = experience[-_SNAPSHOT_EXPERIENCE_TAIL_CHARS:]
    if len(tail) < len(experience):
        tail = "（较早的经历正文已省略；以下是结束部分）\n" + tail
    blocks = [
        prompt_markup_block(
            "经历结束时间",
            prompt_field_lines((("时间", authoritative_time),), omit_empty=False),
        ),
    ]
    if str(identity_catalog_text or "").strip():
        blocks.append(prompt_markup_block(IDENTITY_BLOCK_NAME, identity_catalog_text))
    blocks.append(prompt_markup_block("已经确定的经历末段", tail))
    partial = str(partial_snapshot_text or "").strip()
    if partial:
        blocks.append(prompt_markup_block("经历中已有的角色状态线索", partial))
    previous_fields = _previous_view_fields(previous_view)
    if previous_fields:
        blocks.append(
            prompt_markup_block(
                "此前的角色状态",
                prompt_field_lines(previous_fields),
            )
        )
    return (
        SNAPSHOT_REPAIR_DEFINITION,
        join_prompt_markup(blocks),
        snapshot_repair_contract(opening_keyframe=opening_keyframe),
    )


def required_block_fragments(
    author_kind: BackgroundAuthorKind,
    stage: str,
) -> tuple[str, ...]:
    if stage == "snapshot_repair":
        return ("经历结束时间", IDENTITY_BLOCK_NAME, "已经确定的经历末段")
    fragments = ["故事现在", IDENTITY_BLOCK_NAME, _CREATIVE_POSTURE_BLOCK_NAME]
    if author_kind is BackgroundAuthorKind.WORLD:
        fragments.append("世界背景")
    else:
        fragments.extend(
            (
                (
                    ROLE_LOCATION_BLOCK_NAME
                    if author_kind is BackgroundAuthorKind.STORY_SOURCE
                    else ROLE_CURRENT_STATE_BLOCK_NAME
                ),
                RECENT_FOREGROUND_BLOCK_NAME,
            )
        )
    if author_kind in {BackgroundAuthorKind.KEYFRAME, BackgroundAuthorKind.ORDINARY}:
        fragments.extend(("这段生活经过的时间", "开局时间范围"))
    return tuple(fragments)


def _previous_view_fields(value: object) -> tuple[tuple[str, object], ...]:
    return tuple(
        (label, getattr(value, attribute, ""))
        for label, attribute in (
            ("时间", "narrative_time"),
            ("地点", "location"),
            ("正在做", "doing"),
            ("身体状态", "body_state"),
            ("心情", "mood"),
            ("打算", "intention"),
            ("当前牵挂", "current_concern"),
        )
        if str(getattr(value, attribute, "") or "").strip()
    )


def _source_markup(
    author_kind: BackgroundAuthorKind,
    frozen: FrozenBackgroundProjection,
    *,
    character_projection: str,
    identity_catalog_text: str,
    participant_references: Mapping[str, str],
) -> TrustedPromptMarkup:
    snapshot = frozen.snapshot
    opening = _is_opening_pass(author_kind, frozen.source)
    blocks = [
        prompt_time_block(
            snapshot["prompt_now"],
            timezone_name=str(snapshot.get("timezone_name") or ""),
        )
    ]
    if identity_catalog_text.strip():
        blocks.append(prompt_markup_block(IDENTITY_BLOCK_NAME, identity_catalog_text))
    blocks.extend(_foundation_blocks(snapshot))
    if author_kind is not BackgroundAuthorKind.WORLD:
        blocks.extend(
            _role_blocks(
                author_kind,
                snapshot,
                character_projection=character_projection,
                participant_references=participant_references,
            )
        )
    if author_kind in {
        BackgroundAuthorKind.STORY_SOURCE,
        BackgroundAuthorKind.KEYFRAME,
        BackgroundAuthorKind.ORDINARY,
    }:
        blocks.extend(
            story_source_blocks(
                snapshot.get("story_sources") or (),
                show_ordinals=author_kind
                in {BackgroundAuthorKind.KEYFRAME, BackgroundAuthorKind.ORDINARY},
            )
        )
    if author_kind in {BackgroundAuthorKind.KEYFRAME, BackgroundAuthorKind.ORDINARY}:
        interval = (
            snapshot.get("keyframe_frame_interval")
            if author_kind is BackgroundAuthorKind.KEYFRAME
            else snapshot.get("ordinary_frame_interval")
        )
        blocks.append(
            _opening_frame_window(author_kind) if opening else frame_window_block(interval)
        )
    return join_prompt_markup(blocks)


def _foundation_blocks(snapshot: dict[str, Any]) -> list[TrustedPromptMarkup]:
    blocks = [
        mapping_block("世界背景", snapshot.get("seed"), SEED_FIELDS),
        records_block("世界里的地点与资料", "资料", snapshot.get("lore"), LORE_FIELDS),
        world_changes_block(_state(snapshot.get("world_state"))),
        records_block(
            "创作边界",
            "边界",
            snapshot.get("boundaries"),
            BOUNDARY_FIELDS,
        ),
    ]
    posture = _creative_posture_block(snapshot.get("seed"))
    if str(posture).strip():
        blocks.insert(1, posture)
    return blocks


def _role_blocks(
    author_kind: BackgroundAuthorKind,
    snapshot: dict[str, Any],
    *,
    character_projection: str,
    participant_references: Mapping[str, str],
) -> list[TrustedPromptMarkup]:
    blocks: list[TrustedPromptMarkup] = []
    if character_projection.strip():
        blocks.append(
            prompt_markup_block(
                ROLE_PROFILE_BLOCK_NAME,
                trusted_identity_text(character_projection),
            )
        )
    view = snapshot.get("character_view")
    if view is not None:
        blocks.append(
            location_block(view)
            if author_kind is BackgroundAuthorKind.STORY_SOURCE
            else view_block(view)
        )
    if author_kind in {
        BackgroundAuthorKind.LIFE_DIRECTION,
        BackgroundAuthorKind.KEYFRAME,
        BackgroundAuthorKind.ORDINARY,
    }:
        recent = snapshot.get("recent_timeline") or ()
        if recent:
            blocks.append(
                timeline_block(
                    recent,
                    show_resolvable_ordinals=author_kind
                    in {BackgroundAuthorKind.KEYFRAME, BackgroundAuthorKind.ORDINARY},
                )
            )
    initial_direction = str(snapshot.get("initial_life_direction") or "").strip()
    if author_kind is BackgroundAuthorKind.LIFE_DIRECTION and initial_direction:
        blocks.append(
            prompt_markup_block(
                "初始人生方向",
                "这是一枚可选的开局种子，可以采用、改写或完全忽略：\n" + initial_direction,
            )
        )
    blocks.append(life_direction_block(_state(snapshot.get("life_state"))))
    blocks.append(
        foreground_handoff_block(
            snapshot.get("foreground_messages") or (),
            snapshot.get("foreground_runs") or (),
            participant_references=participant_references,
        )
    )
    return blocks


def _state(value: object) -> BackgroundAuthorState | None:
    return value if isinstance(value, BackgroundAuthorState) else None


def _projection(
    author_kind: BackgroundAuthorKind,
    source: BackgroundAuthorInput | FrozenBackgroundProjection,
) -> FrozenBackgroundProjection:
    if isinstance(source, FrozenBackgroundProjection):
        if source.source.author_kind is not author_kind:
            raise ValueError("frozen background projection belongs to another author")
        return source
    return freeze_background_projection(author_kind, source)


def _is_opening_pass(
    author_kind: BackgroundAuthorKind,
    source: BackgroundAuthorInput,
) -> bool:
    if str(source.initialization_state).upper() != "INITIALIZING":
        return False
    step = source.initialization_step
    if step is BackgroundInitializationStep.WORLD:
        return author_kind is BackgroundAuthorKind.WORLD
    if step is BackgroundInitializationStep.LIFE_DIRECTION:
        return author_kind is BackgroundAuthorKind.LIFE_DIRECTION
    if step is BackgroundInitializationStep.STORY_SOURCE:
        return author_kind is BackgroundAuthorKind.STORY_SOURCE
    if step is BackgroundInitializationStep.ORDINARY_CURRENT:
        expected = (
            BackgroundAuthorKind.ORDINARY
            if source.opening_keyframe_completed
            else BackgroundAuthorKind.KEYFRAME
        )
        return author_kind is expected
    return False


def _opening_frame_window(author_kind: BackgroundAuthorKind) -> TrustedPromptMarkup:
    if author_kind is BackgroundAuthorKind.KEYFRAME:
        fields = (
            ("可回溯范围", "按情节需要自由选择，不必填满"),
            ("结束位置", "故事所在时区前一自然日16:00"),
        )
    else:
        fields = (
            ("范围", "从开局时的角色状态到故事现在"),
            ("起点", "故事所在时区前一自然日16:00"),
            ("覆盖", "完整跨过一整晚并运行到故事现在"),
        )
    return prompt_markup_block("开局时间范围", prompt_field_lines(fields))


def _creative_posture_block(value: Any) -> TrustedPromptMarkup:
    source = value if isinstance(value, Mapping) else {}
    policy = str(source.get("expansion_policy") or "OPEN").strip().upper()
    expansion = (
        "在当前作者的职责范围内，优先扩写已有资料没有写明的角色生活、世界人物、"
        "地点、关系与历史；材料偶尔穿帮时直接继续创作，不核查或纠正。"
        if policy == "CANON_GUARDED"
        else "在当前作者的职责范围内，可以自由扩写已有资料没有写明的角色生活、世界"
        "人物、地点、关系与历史；材料偶尔穿帮时直接继续创作，不核查或纠正。"
    )
    real_participant_boundary = (
        "现实聊天中的对方和群里其他真实聊天身份绝不属于背景创作空间。聊天记录只证明"
        "已经发生的交流；可以写角色收到这些消息后自己的反应或决定，但绝对不要替现实"
        "人物新增任何台词、行动、到场、身份、经历、关系、想法或情绪，不要把约定、玩笑"
        "或设想写成他们随后真的参与了背景生活，也不要让角色与他们共同经历聊天记录中"
        "没有发生的事。背景作者可以继续创作的参与者只有角色本人、角色世界里已经存在的"
        "人物和新创造的虚构人物。"
    )
    return prompt_markup_block(
        _CREATIVE_POSTURE_BLOCK_NAME,
        f"{expansion}\n\n{real_participant_boundary}",
    )


def _fit(
    author_kind: BackgroundAuthorKind,
    definition: str,
    contract: str,
    task_input: TrustedPromptMarkup,
    budget: BackgroundPromptBudget | None,
) -> TrustedPromptMarkup:
    if budget is None:
        return task_input
    return fit_stage_markup(
        author_kind=author_kind,
        stage="creator",
        candidates=(("background_material", task_input),),
        limit=budget.stage_input_limit(
            task_definition=definition,
            output_contract=contract,
            output_reserve_tokens=DEFAULT_RESERVED_OUTPUT_TOKENS,
        ),
        required_name_fragments=required_block_fragments(author_kind, "creator"),
    )


__all__ = [
    "build_creator_prompt",
    "build_snapshot_repair_prompt",
    "required_block_fragments",
]
