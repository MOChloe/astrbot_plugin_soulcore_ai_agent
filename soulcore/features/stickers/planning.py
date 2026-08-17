"""Frozen inputs and semantic planning boundaries for sticker collection."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...contracts.ai_models import AIWorkPurpose
from ...contracts.vision import VisionInspectionMode
from ...shared.prompt_document import (
    TrustedPromptMarkup,
    join_prompt_markup,
    project_prompt_text,
    prompt_markup_block,
    prompt_markup_record,
)
from ..ai import run_structured_text_session
from ..ai.service import parse_model_turn
from ..media.ports import VisualCachePolicy
from .contracts import (
    STICKER_COLLECT_COMMAND_NAMES,
    StickerGenerationSpec,
    StickerTextFinishingDeferred,
    sticker_collect_output_contract,
)
from .domain import StickerSourceKind
from .policy import StickerRuntimeDisabled
from .text_modes import TEXT_MODE_INTEGRATED_TEXT


class StickerTextFinishingMixin:
    async def _finish_integrated_text(
        self,
        *,
        profile_id: str,
        instance_id: str,
        task_id: int,
        source_asset_id: str,
        spec: StickerGenerationSpec,
        initial_vision: Any | None = None,
        release_replaced_source: bool = True,
    ) -> str:
        vision = await self._integrated_initial_vision(
            profile_id, instance_id, source_asset_id, spec, initial_vision
        )
        if self._ocr_matches(spec.meme_text, str(vision.ocr_text or "")):
            return source_asset_id
        corrected = await self._try_integrated_text_correction(
            profile_id,
            instance_id,
            task_id,
            source_asset_id,
            spec,
            release_replaced_source,
        )
        if corrected:
            return corrected
        if release_replaced_source:
            await self._release_source_asset(source_asset_id, reason="INTEGRATED_TEXT_UNREADABLE")
        return ""

    async def _integrated_initial_vision(
        self,
        profile_id: str,
        instance_id: str,
        source_asset_id: str,
        spec: StickerGenerationSpec,
        initial_vision: Any | None,
    ) -> Any:
        if initial_vision is not None:
            return initial_vision
        await self._require_runtime_source(profile_id, instance_id, StickerSourceKind.GENERATED)
        try:
            result = await self.visual_service.describe_asset(
                profile_id=profile_id,
                instance_id=instance_id,
                asset_id=source_asset_id,
                foreground=False,
                cache_policy=VisualCachePolicy.USE,
                inspection_mode=VisionInspectionMode.OBJECTIVE,
            )
            await self._require_runtime_source(profile_id, instance_id, StickerSourceKind.GENERATED)
            return result
        except StickerRuntimeDisabled:
            raise
        except Exception as exc:
            raise StickerTextFinishingDeferred(
                source_asset_id,
                text_mode=TEXT_MODE_INTEGRATED_TEXT,
                cause=exc,
            ) from exc

    async def _try_integrated_text_correction(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        source_asset_id: str,
        spec: StickerGenerationSpec,
        release_source: bool,
    ) -> str:
        await self._require_runtime_source(profile_id, instance_id, StickerSourceKind.GENERATED)
        try:
            result = await self.visual_service.present_visual(
                profile_id=profile_id,
                instance_id=instance_id,
                run_id=task_id,
                counterpart_requirements="",
                scene_plan=(
                    "只局部重绘原图中文字区域，保持角色、表情、背景和其余像素构图不变；"
                    f"清除错误文字并准确写成：{spec.meme_text}。"
                ),
                image_count=1,
                aspect_ratio="1:1",
                size="auto",
                reference_asset_ids=[source_asset_id],
                reference_purposes=["文字编辑底图；只修正文字区域，不复制画风"],
                main_core_supports_vision=False,
                defer_inspection_to_sticker_check=True,
                require_raw_references=True,
                maximum_generation_backends=2,
            )
            await self._require_runtime_source(profile_id, instance_id, StickerSourceKind.GENERATED)
            corrected_ids = [str(value) for value in result.get("asset_ids", ()) if str(value)]
            if not corrected_ids:
                return ""
            return await self._verify_integrated_correction(
                profile_id,
                instance_id,
                source_asset_id,
                corrected_ids[0],
                spec,
                release_source,
            )
        except (StickerTextFinishingDeferred, StickerRuntimeDisabled):
            raise
        except Exception as exc:
            if self._unknown_generation_effect(exc):
                raise
            return ""

    async def _verify_integrated_correction(
        self,
        profile_id: str,
        instance_id: str,
        source_asset_id: str,
        candidate: str,
        spec: StickerGenerationSpec,
        release_source: bool,
    ) -> str:
        await self._require_runtime_source(profile_id, instance_id, StickerSourceKind.GENERATED)
        try:
            verification = await self.visual_service.describe_asset(
                profile_id=profile_id,
                instance_id=instance_id,
                asset_id=candidate,
                foreground=False,
                cache_policy=VisualCachePolicy.USE,
                inspection_mode=VisionInspectionMode.OBJECTIVE,
            )
            await self._require_runtime_source(profile_id, instance_id, StickerSourceKind.GENERATED)
        except StickerRuntimeDisabled:
            raise
        except Exception as exc:
            if release_source:
                await self._release_source_asset(
                    source_asset_id, reason="REPLACED_BY_TEXT_CORRECTION_PENDING"
                )
            raise StickerTextFinishingDeferred(
                candidate,
                text_mode=TEXT_MODE_INTEGRATED_TEXT,
                cause=exc,
            ) from exc
        if self._ocr_matches(spec.meme_text, str(verification.ocr_text or "")):
            if release_source:
                await self._release_source_asset(
                    source_asset_id, reason="REPLACED_BY_TEXT_CORRECTION"
                )
            return candidate
        await self._release_source_asset(candidate, reason="INTEGRATED_TEXT_CORRECTION_FAILED")
        return ""


@dataclass(frozen=True, slots=True)
class StickerCollectionGap:
    """One server-selected, model-readable inventory gap.

    ``key`` and the matching aliases are server-only.  The model receives only
    the natural goal and its source/identity boundaries.
    """

    key: str
    goal: str
    search_anchor: str
    expected_use: str
    expected_text: str
    visual_goal: str
    character_specific: bool = False
    animated_preferred: bool = False

    def collection_intent(self, administrator_preferences: str) -> dict[str, str]:
        return {
            "本轮要补": self.goal,
            "身份边界": "当前角色专属" if self.character_specific else "通用，不使用当前角色资料",
            "沟通用途": self.expected_use,
            "文字预期": self.expected_text,
            "画面方向": self.visual_goal,
            "管理员偏好与禁区": str(administrator_preferences or "").strip()[:3000],
        }


@dataclass(frozen=True, slots=True)
class _GapTemplate:
    key: str
    goal: str
    search_anchor: str
    expected_use: str
    expected_text: str
    aliases: tuple[str, ...]
    character_specific: bool = False


_GAP_TEMPLATES = (
    _GapTemplate(
        "ambient_light",
        "轻松可爱、没有明确答复意味的氛围表情",
        "轻松 可爱 氛围 表情包",
        "不承担回答，只用来活跃聊天节奏",
        "无字",
        ("可爱", "萌", "治愈", "氛围", "轻松"),
    ),
    _GapTemplate(
        "comfort",
        "能自然表达安慰或鼓励、又不绑定具体台词的通用反应",
        "安慰 鼓励 表情包",
        "安慰、打气或陪伴",
        "优先无字，也可以是极短且可复用的字幕",
        ("安慰", "鼓励", "打气", "抱抱", "陪伴"),
    ),
    _GapTemplate(
        "celebrate",
        "表达赞同、开心或小小庆祝的通用反应",
        "赞同 开心 庆祝 表情包",
        "赞同、开心或庆祝",
        "无字或极短字幕均可",
        ("赞同", "开心", "庆祝", "太好了", "鼓掌"),
    ),
    _GapTemplate(
        "confused",
        "表达疑惑、没听懂或想再确认一下的通用反应",
        "疑惑 没听懂 表情包",
        "疑问、困惑或请求确认",
        "优先无字，避免绑定具体问题",
        ("疑惑", "困惑", "没听懂", "问号", "确认"),
    ),
    _GapTemplate(
        "awkward",
        "表达尴尬、无语或一时接不上话的通用反应",
        "尴尬 无语 表情包",
        "尴尬、无语或短暂停顿",
        "无字或极短字幕均可",
        ("尴尬", "无语", "沉默", "接不上话", "汗"),
    ),
    _GapTemplate(
        "decline",
        "表达温和拒绝、不同意或劝停的通用反应",
        "拒绝 不同意 劝停 表情包",
        "拒绝、不同意或劝停",
        "允许短字幕，但不能只适用于单一事件",
        ("拒绝", "不同意", "不行", "劝停", "打住"),
    ),
    _GapTemplate(
        "spectate",
        "表达围观、看戏或轻微调侃的通用反应",
        "围观 看戏 调侃 表情包",
        "围观、看戏或轻微调侃",
        "优先无字",
        ("围观", "看戏", "吃瓜", "调侃"),
    ),
    _GapTemplate(
        "character_chibi",
        "当前角色专属的轻量 Q 版反应，动作和表情要能在聊天里反复使用",
        "Q版 角色反应 表情包",
        "以当前角色自己的形象表达轻量日常反应",
        "优先无字，避免把角色专属图变成固定台词海报",
        ("角色专属", "q版", "chibi"),
        character_specific=True,
    ),
)

_PLANNING_COMMAND_FIELDS = {
    "网页搜图": ("web_queries", "查询"),
    "生成前研究": ("generation_research_queries", "查询"),
    "生成概念": ("generation_prompts", "内容"),
}


def sticker_persona_fingerprint(persona: str) -> str:
    """Return the stable fingerprint stored with checked sticker items."""

    return hashlib.sha256(str(persona or "").encode()).hexdigest()


def _enabled_planning_commands(
    *,
    web_enabled: bool,
    generation_enabled: bool,
) -> list[str]:
    commands: list[str] = []
    if web_enabled:
        commands.append("网页搜图")
    if web_enabled and generation_enabled:
        commands.append("生成前研究")
    if generation_enabled:
        commands.append("生成概念")
    return commands


def _planning_commands_from_contract(output_contract: str) -> frozenset[str]:
    return frozenset(
        name for name in STICKER_COLLECT_COMMAND_NAMES if f"<{name}>" in output_contract
    )


def _empty_planning_result() -> dict[str, list[str]]:
    return {
        "web_queries": [],
        "generation_research_queries": [],
        "generation_prompts": [],
    }


def _parse_candidate_commands(commands: Sequence[Any]) -> dict[str, list[str]]:
    result = _empty_planning_result()
    for command in commands:
        target, field = _PLANNING_COMMAND_FIELDS[command.name]
        value = str(command.parameters.get(field) or "").strip()
        if not value or set(command.parameters) != {field}:
            raise ValueError(f"<{command.name}>只能填写且必须填写 [[{field}]]")
        result[target].append(value)
    return result


def _require_planning_commands_available(
    names: set[str],
    allowed_planning_commands: Sequence[str] | None,
) -> None:
    if allowed_planning_commands is None:
        return
    allowed = frozenset(str(name) for name in allowed_planning_commands)
    unknown = allowed - frozenset(STICKER_COLLECT_COMMAND_NAMES)
    if unknown:
        raise RuntimeError("表情候选输出合同含未知指令：" + "、".join(sorted(unknown)))
    unavailable = names - allowed
    if unavailable:
        raise ValueError("本轮不可提交未列为可用的指令：" + "、".join(sorted(unavailable)))


class StickerPlanningMixin:
    @staticmethod
    def _planning_prompt(**data: Any) -> TrustedPromptMarkup:
        gap = data["gap"]
        if not isinstance(gap, StickerCollectionGap):
            raise TypeError("gap must be StickerCollectionGap")
        requirements = str(data.get("requirements") or "").strip()
        commands = _enabled_planning_commands(
            web_enabled=bool(data.get("web_enabled")),
            generation_enabled=bool(data.get("generation_enabled")),
        )
        records: list[TrustedPromptMarkup] = [
            prompt_markup_record(
                "本轮补缺目标",
                (
                    ("要补的内容", gap.goal),
                    ("沟通用途", gap.expected_use),
                    ("文字预期", gap.expected_text),
                    ("画面方向", gap.visual_goal),
                    (
                        "身份边界",
                        "当前角色专属" if gap.character_specific else "通用，不使用当前角色资料",
                    ),
                    ("可用指令", "、".join(commands) or "无"),
                ),
                omit_empty=False,
            )
        ]
        if requirements:
            records.append(
                prompt_markup_record(
                    "收藏偏好",
                    (("内容", requirements[:5000]),),
                )
            )
        if gap.character_specific:
            records.append(
                prompt_markup_record(
                    "角色创作资料",
                    (
                        ("身份与外观", str(data.get("persona") or "")[:8000]),
                        ("背景", str(data.get("role_background") or "")[:3000]),
                        ("画面风格", str(data.get("world_texture") or "")[:3000]),
                        (
                            "身份参考",
                            str(data.get("identity_reference_note") or "") or "无",
                        ),
                    ),
                    omit_empty=True,
                )
            )
        return join_prompt_markup(records)

    @staticmethod
    def _planning_output_contract(
        base_contract: str,
        *,
        web_enabled: bool,
        generation_enabled: bool,
        character_specific: bool,
    ) -> str:
        if not str(base_contract or "").strip():
            raise ValueError("sticker collection output contract is empty")
        del character_specific
        return sticker_collect_output_contract(
            _enabled_planning_commands(
                web_enabled=web_enabled,
                generation_enabled=generation_enabled,
            )
        )

    @staticmethod
    def _web_query_tiers(
        *,
        plan: Mapping[str, Any],
        gap: StickerCollectionGap,
        requirements: str,
    ) -> list[list[str]]:
        planned = [
            str(value).strip()[:240]
            for value in list(plan.get("web_queries") or ())[:10]
            if str(value).strip()
        ]
        if not planned:
            return []
        constraint = re.sub(r"\s+", " ", str(requirements or "")).strip()[:160]

        def guarded(value: str, suffix: str) -> str:
            return " ".join(
                part
                for part in (
                    value,
                    gap.search_anchor,
                    constraint,
                    suffix,
                )
                if part
            )[:240]

        return [
            list(dict.fromkeys(guarded(value, "") for value in planned))[:10],
            list(dict.fromkeys(guarded(value, "表情包 梗图") for value in planned))[:10],
            list(
                dict.fromkeys(
                    guarded(
                        value,
                        "动图 GIF" if gap.animated_preferred else "聊天反应图 sticker",
                    )
                    for value in planned
                )
            )[:10],
        ]

    @classmethod
    def _select_collection_gap(
        cls,
        snapshot: Mapping[str, Any],
        *,
        task_id: int,
        requested_theme: str,
        requirements: str,
        character_name: str = "",
        web_enabled: bool,
    ) -> StickerCollectionGap:
        inventory = dict(snapshot.get("inventory") or snapshot)
        theme = re.sub(r"\s+", " ", str(requested_theme or "")).strip()[:500]
        preferences = re.sub(r"\s+", " ", str(requirements or "")).strip()
        character_specific = cls._explicit_character_focus(
            " ".join(part for part in (theme, preferences) if part),
            character_name,
        )
        if theme:
            template = _GapTemplate(
                "requested_theme",
                f"围绕“{theme}”补一张用途明确、可以反复使用的表情",
                f"{theme} 表情包",
                theme,
                "按主题本身决定；没有必要时不要强加文字",
                (),
                character_specific=character_specific,
            )
        else:
            template = cls._least_covered_template(inventory, task_id)
            character_specific = template.character_specific
        format_goal, expected_text, animated = cls._format_gap(
            inventory,
            preferences,
            default_text=template.expected_text,
            web_enabled=web_enabled,
        )
        existing_visuals = cls._existing_visual_concepts(inventory)
        visual_diversity = (
            "与以下近期已有画面的主体、动作或构图保持区别："
            + "；".join(f"“{value}”" for value in existing_visuals)
            if existing_visuals
            and int(inventory.get("visual_groups") or 0) < int(inventory.get("total") or 0)
            else ""
        )
        visual_goal = "；".join(
            value for value in (visual_diversity, format_goal) if str(value).strip()
        )
        return StickerCollectionGap(
            key=template.key,
            goal=f"{template.goal}；{format_goal}",
            search_anchor=f"{template.search_anchor} {format_goal}",
            expected_use=template.expected_use,
            expected_text=expected_text,
            visual_goal=visual_goal,
            character_specific=character_specific,
            animated_preferred=animated,
        )

    @staticmethod
    def _existing_visual_concepts(
        inventory: Mapping[str, Any],
        *,
        limit: int = 6,
    ) -> tuple[str, ...]:
        concepts: list[str] = []
        seen: set[str] = set()
        for row in inventory.get("coverage") or ():
            if not isinstance(row, Mapping):
                continue
            value = re.sub(r"\s+", " ", str(row.get("compact_description") or "")).strip()[:120]
            key = value.casefold()
            if not value or key in seen:
                continue
            seen.add(key)
            concepts.append(value)
            if len(concepts) >= max(0, int(limit)):
                break
        return tuple(concepts)

    @staticmethod
    def _explicit_character_focus(text: str, character_name: str) -> bool:
        normalized = str(text or "").casefold()
        name = str(character_name or "").strip().casefold()
        return bool(
            re.search(r"(角色专属|当前角色|本人形象|角色外观|q版|chibi|人物专属)", normalized)
            or (name and name in normalized)
        )

    @staticmethod
    def _least_covered_template(inventory: Mapping[str, Any], task_id: int) -> _GapTemplate:
        rows = list(inventory.get("coverage") or ())
        rotation = max(0, int(task_id))
        scores: list[tuple[int, int, _GapTemplate]] = []
        for index, template in enumerate(_GAP_TEMPLATES):
            matched = usage = 0
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                text = " ".join(
                    str(row.get(key) or "")
                    for key in (
                        "semantic_key",
                        "emotion",
                        "speech_act",
                        "vibe_tags",
                        "search_keywords",
                        "compact_description",
                        "collection_scope",
                    )
                ).casefold()
                is_match = (
                    str(row.get("collection_scope") or "") == "当前角色专属"
                    if template.character_specific
                    else any(alias.casefold() in text for alias in template.aliases)
                )
                if is_match:
                    matched += 1
                    usage += max(0, int(row.get("usage_count") or 0))
            tie_breaker = (index - rotation) % len(_GAP_TEMPLATES)
            scores.append((matched * 100 + min(usage, 99), tie_breaker, template))
        return min(scores, key=lambda value: (value[0], value[1]))[2]

    @staticmethod
    def _format_gap(
        inventory: Mapping[str, Any],
        preferences: str,
        *,
        default_text: str,
        web_enabled: bool,
    ) -> tuple[str, str, bool]:
        normalized = str(preferences or "").casefold()
        preference = StickerPlanningMixin._explicit_format_preference(
            normalized, default_text, web_enabled
        )
        if preference is not None:
            return preference
        preference = StickerPlanningMixin._requested_text_format(normalized)
        if preference is not None:
            return preference
        return StickerPlanningMixin._inventory_format_gap(
            inventory, default_text=default_text, web_enabled=web_enabled
        )

    @staticmethod
    def _requested_text_format(normalized: str) -> tuple[str, str, bool] | None:
        if re.search(r"(无字|不要文字|禁止文字)", normalized):
            return "画面本身要成立，不依赖文字", "无字", False
        if re.search(r"(带字|有字|字幕|短句)", normalized):
            return "需要文字时只用简短、可复用且符合禁区的内容", "短字幕", False
        return None

    @staticmethod
    def _inventory_format_gap(
        inventory: Mapping[str, Any], *, default_text: str, web_enabled: bool
    ) -> tuple[str, str, bool]:
        total = max(0, int(inventory.get("total") or 0))
        if StickerPlanningMixin._needs_more_animation(inventory, total, web_enabled):
            return "优先寻找动作清楚且循环自然的动图", default_text, True
        if StickerPlanningMixin._inventory_count_is_low(inventory, "text", total, 4):
            return "可用一句极短、可复用的字幕补足有字表达", "短字幕", False
        if StickerPlanningMixin._inventory_count_is_low(inventory, "no_text", total, 4):
            return "画面本身要成立，不依赖文字", "无字", False
        return "主体、动作和构图要清楚，适合在聊天中快速辨认", default_text, False

    @staticmethod
    def _needs_more_animation(inventory: Mapping[str, Any], total: int, web_enabled: bool) -> bool:
        return web_enabled and StickerPlanningMixin._inventory_count_is_low(
            inventory, "animated", total, 6
        )

    @staticmethod
    def _inventory_count_is_low(
        inventory: Mapping[str, Any], field: str, total: int, threshold: int
    ) -> bool:
        return total >= 4 and int(inventory.get(field) or 0) * threshold < total

    @staticmethod
    def _explicit_format_preference(
        normalized: str, default_text: str, web_enabled: bool
    ) -> tuple[str, str, bool] | None:
        if re.search(r"(动图|gif|动画表情)", normalized) and web_enabled:
            return "优先寻找动作清楚且循环自然的动图", default_text, True
        return None

    async def _run_model_commands(
        self,
        profile_id: str,
        instance: Any,
        capability: str,
        system_prompt: str,
        prompt: str,
        output_contract: str,
        *,
        owner_kind: str,
        owner_id: str,
        identity_context: Any | None = None,
        identity_catalog: Any | None = None,
        include_identity_context: bool = True,
    ) -> dict[str, Any]:
        if instance is None:
            raise ValueError("sticker instance unavailable")
        if capability == "sticker.collect":
            (await self._runtime_policy(profile_id, str(instance.instance_id))).require_collection()
        else:
            await self._require_runtime_enabled(profile_id, str(instance.instance_id))
        allowed_planning_commands = (
            _planning_commands_from_contract(output_contract)
            if "<无候选>" in output_contract
            else None
        )
        prompt_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        logical_step_key = f"sticker:{owner_kind}:{owner_id}:{capability}:{prompt_key}"
        if include_identity_context:
            if identity_context is None or identity_catalog is None:
                identity_context, identity_catalog = await self.identity.catalog(
                    profile_id, str(instance.instance_id)
                )
            system_prompt = self.identity.project_for_model(
                system_prompt,
                identity_catalog,
                scope=str(identity_context.scope),
            )
            projected_prompt = project_prompt_text(
                prompt,
                lambda value: self.identity.encode_for_model(value, identity_catalog),
            )
            if not isinstance(projected_prompt, TrustedPromptMarkup):
                projected_prompt = prompt_markup_block("任务材料", projected_prompt)
            prompt = join_prompt_markup(
                (
                    prompt_markup_block("人物引用", identity_catalog.prompt_text()),
                    projected_prompt,
                )
            )

        async def invoke(round_no: int, feedback: str) -> Any:
            projected_contract = output_contract
            projected_feedback = feedback
            if include_identity_context:
                projected_contract = self.identity.project_for_model(
                    output_contract,
                    identity_catalog,
                    scope=str(identity_context.scope),
                )
                projected_feedback = self.identity.project_for_model(
                    feedback,
                    identity_catalog,
                    scope=str(identity_context.scope),
                )
            return await self.model_gateway.generate_text(
                task_definition=system_prompt,
                task_input=prompt,
                output_contract=projected_contract,
                execution_record=projected_feedback,
                profile_id=profile_id,
                instance_id=str(instance.instance_id),
                capability=capability,
                owner_kind=owner_kind,
                owner_id=owner_id,
                idempotency_key=f"{logical_step_key}:round:{round_no}",
                work_purpose=(
                    AIWorkPurpose.STICKER_COLLECTION
                    if capability == "sticker.collect"
                    else AIWorkPurpose.STICKER_CHECK
                ),
                logical_stage_key=logical_step_key,
                round_no=round_no,
                operation_timeout_seconds=self.operation_timeout_seconds,
            )

        def validate(text: str) -> dict[str, Any]:
            return self._parse_model_commands(
                text,
                allowed_planning_commands=allowed_planning_commands,
            )

        validated = await run_structured_text_session(
            model_gateway=self.model_gateway,
            invoke=invoke,
            validate=validate,
        )
        if capability == "sticker.collect":
            (await self._runtime_policy(profile_id, str(instance.instance_id))).require_collection()
        else:
            await self._require_runtime_enabled(profile_id, str(instance.instance_id))
        return validated.value

    @staticmethod
    def _parse_model_commands(
        text: str,
        *,
        allowed_planning_commands: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        parsed = parse_model_turn(text)
        if parsed.errors:
            raise ValueError("输出格式有误：" + "；".join(parsed.errors))
        if parsed.working_text:
            raise ValueError("规定的表情指令标签之外不能有文字")
        names = {item.name for item in parsed.commands}
        if not parsed.commands:
            raise ValueError("必须写出至少一个允许的候选，或只写一个空的 <无候选> 块")
        if names == {"无候选"}:
            if len(parsed.commands) != 1 or parsed.commands[0].parameters:
                raise ValueError("<无候选>必须是唯一且不含字段的空块")
            return _empty_planning_result()
        if "无候选" in names:
            raise ValueError("<无候选>不能与任何候选指令同时出现")
        if names <= _PLANNING_COMMAND_FIELDS.keys():
            _require_planning_commands_available(names, allowed_planning_commands)
            return _parse_candidate_commands(parsed.commands)
        if names == {"表情设计"}:
            return {
                "generation_specs": [_generation_spec(item.parameters) for item in parsed.commands]
            }
        if names == {"表情检查"} and len(parsed.commands) == 1:
            return _sticker_check(dict(parsed.commands[0].parameters))
        raise ValueError("输出含有未允许的表情指令，或混用了彼此不兼容的指令")

    @staticmethod
    def _material_markup(
        values: Sequence[Any],
        *,
        available: bool,
    ) -> TrustedPromptMarkup:
        del available
        records: list[TrustedPromptMarkup] = []
        for value in values[:12]:
            if isinstance(value, Mapping):
                title = str(value.get("title") or "").strip()
                detail = next(
                    (
                        str(value.get(key) or "").strip()
                        for key in ("content", "snippet", "summary", "brief")
                        if str(value.get(key) or "").strip()
                    ),
                    "",
                )
            else:
                title, detail = "", str(value or "").strip()
            if not any((title, detail)):
                continue
            records.append(
                prompt_markup_record(
                    "参考资料",
                    (
                        ("标题", title[:240]),
                        ("内容", detail[:1200]),
                    ),
                )
            )
        return (
            prompt_markup_block("联网参考", join_prompt_markup(records))
            if records
            else TrustedPromptMarkup("")
        )

    @staticmethod
    def persona_fingerprint(persona: str) -> str:
        return sticker_persona_fingerprint(persona)


__all__ = [
    "StickerCollectionGap",
    "StickerPlanningMixin",
    "sticker_persona_fingerprint",
]


def _generation_spec(fields: Mapping[str, str]) -> dict[str, Any]:
    required = ("主体与身份", "动作表情", "构图与背景", "画面风格", "文字关系", "禁止项")
    missing = [name for name in required if not str(fields.get(name) or "").strip()]
    if missing:
        raise ValueError("表情设计缺少参数：" + "、".join(missing))
    text_mode = {
        "无文字": "NONE",
        "文字与画面不可分": "INTEGRATED_TEXT",
    }.get(fields["文字关系"])
    if text_mode is None:
        raise ValueError("表情设计的文字关系不是允许的中文选项")
    meme_text = str(fields.get("文字") or "").strip()
    caption_position = str(fields.get("字幕大致位置") or "").strip()
    safe_zone = str(fields.get("字幕安全区") or "").strip()
    if caption_position and caption_position not in {"上部", "下部", "画面内"}:
        raise ValueError("字幕大致位置只能是上部、下部或画面内")
    if text_mode == "NONE":
        if caption_position or safe_zone:
            raise ValueError("无文字的表情设计不得提供字幕位置或字幕安全区")
        meme_text = ""
    elif not meme_text:
        raise ValueError("有文字的表情设计必须提供文字")
    return {
        "style": fields["画面风格"],
        "subject_identity": fields["主体与身份"],
        "action_expression": fields["动作表情"],
        "composition": fields["构图与背景"],
        "meme_text": meme_text,
        "text_mode": text_mode,
        "caption_position": caption_position,
        "safe_zone": safe_zone,
        "negative_constraints": fields["禁止项"],
    }


def _sticker_check(fields: Mapping[str, str]) -> dict[str, Any]:
    mappings = {
        "accepted": "结论",
        "rejection_category": "拒绝类别",
        "reason": "原因",
        "compact_description": "表情释义",
        "usage_contexts": "适用语境",
        "vibe_tags": "整体观感",
        "emotion": "情绪",
        "speech_act": "言语作用",
        "intensity": "强度",
    }
    optional = {"拒绝类别", "适用语境", "整体观感", "情绪", "言语作用", "强度"}
    missing = [
        label for label in mappings.values() if label not in fields and label not in optional
    ]
    if missing:
        raise ValueError("表情检查缺少参数：" + "、".join(missing))
    result = {key: fields.get(label, "") for key, label in mappings.items()}
    conclusion = str(result["accepted"]).strip()
    if conclusion not in {"接纳", "拒绝"}:
        raise ValueError("表情检查结论只能是接纳或拒绝")
    result["accepted"] = conclusion == "接纳"
    if result["accepted"] and not str(result["vibe_tags"]).strip():
        raise ValueError("接纳的表情检查缺少参数：整体观感")
    categories = {
        "不安全内容": "UNSAFE_CONTENT",
        "不适合表情包": "NOT_A_STICKER",
        "搜集目标不符": "COLLECTION_INTENT_MISMATCH",
        "角色身份不符": "CHARACTER_IDENTITY_MISMATCH",
        "文字质量不合格": "TEXT_QUALITY",
        "收藏禁区": "ADMIN_PROHIBITION",
        "来源标记": "WATERMARK_PRESENT",
        "视觉证据不足": "INSUFFICIENT_VISUAL_EVIDENCE",
    }
    category = str(result["rejection_category"]).strip()
    if result["accepted"] and category:
        raise ValueError("接纳时不得提供拒绝类别")
    if not result["accepted"] and category not in categories:
        raise ValueError("拒绝时必须提供有效拒绝类别")
    result["rejection_category"] = categories.get(category, "")
    result["intensity"] = int(result["intensity"] or 0)
    result["usage_contexts"] = _split_values(result["usage_contexts"])
    result["vibe_tags"] = _split_values(result["vibe_tags"])
    return result


def _split_values(value: Any) -> list[str]:
    return [part.strip() for part in re.split(r"[,，、\n]", str(value or "")) if part.strip()]
