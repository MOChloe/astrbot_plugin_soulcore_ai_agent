"""Deterministic purpose projections over a frozen character snapshot."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from ...shared.token_meter import ConservativeTokenMeter
from ..identity import CHARACTER_PLACEHOLDER
from .domain import (
    CharacterCustomPrompts,
    CharacterFieldCategory,
    CharacterModel,
    CharacterModelSnapshot,
    CharacterProjection,
    CharacterProjectionSection,
    MainCoreModePrompts,
    MainCoreStylePrompts,
    ProjectionPurpose,
    ResponsePolishPrompts,
    StoryStylePrompts,
    UnsupportedProjectionPurpose,
)


@dataclass(frozen=True, slots=True)
class ProjectionPolicy:
    purpose: ProjectionPurpose
    categories: tuple[CharacterFieldCategory, ...]


_IDENTITY = CharacterFieldCategory.IDENTITY
_PERSONALITY = CharacterFieldCategory.PERSONALITY
_SOCIAL = CharacterFieldCategory.SOCIAL
_PREFERENCES = CharacterFieldCategory.PREFERENCES
_LANGUAGE = CharacterFieldCategory.LANGUAGE
_DIALOGUE_REFERENCE = CharacterFieldCategory.DIALOGUE_REFERENCE
_VISUAL = CharacterFieldCategory.VISUAL
_CAPABILITIES = CharacterFieldCategory.CAPABILITIES
_MAIN_CORE_CATEGORIES = (_IDENTITY, _PERSONALITY, _SOCIAL, _PREFERENCES, _LANGUAGE)

PROJECTION_POLICIES = {
    ProjectionPurpose.MAIN_CORE_WITH_POLISH: ProjectionPolicy(
        ProjectionPurpose.MAIN_CORE_WITH_POLISH,
        _MAIN_CORE_CATEGORIES,
    ),
    ProjectionPurpose.MAIN_CORE_DIRECT: ProjectionPolicy(
        ProjectionPurpose.MAIN_CORE_DIRECT,
        _MAIN_CORE_CATEGORIES,
    ),
    ProjectionPurpose.RESPONSE_POLISH: ProjectionPolicy(
        ProjectionPurpose.RESPONSE_POLISH,
        (_IDENTITY, _LANGUAGE, _DIALOGUE_REFERENCE),
    ),
    ProjectionPurpose.BACKGROUND_AUTHOR: ProjectionPolicy(
        ProjectionPurpose.BACKGROUND_AUTHOR,
        (_IDENTITY, _PERSONALITY, _SOCIAL, _PREFERENCES, _CAPABILITIES),
    ),
    ProjectionPurpose.WEB_PERSONALIZED: ProjectionPolicy(
        ProjectionPurpose.WEB_PERSONALIZED,
        (_IDENTITY, _PREFERENCES),
    ),
    ProjectionPurpose.VISUAL_GENERATION: ProjectionPolicy(
        ProjectionPurpose.VISUAL_GENERATION,
        (_VISUAL,),
    ),
    ProjectionPurpose.STICKER_PLANNING: ProjectionPolicy(
        ProjectionPurpose.STICKER_PLANNING,
        (_VISUAL, _PERSONALITY, _LANGUAGE, _SOCIAL),
    ),
}

_HEADERS = {
    _IDENTITY: "核心身份",
    _PERSONALITY: "性格与认知",
    _SOCIAL: "社交与边界",
    _PREFERENCES: "相关稳定偏好",
    _LANGUAGE: "语言表达",
    _DIALOGUE_REFERENCE: "角色原声台词参考（只用于表达风格，不是事实或已发生对话）",
    _VISUAL: "视觉身份",
    _CAPABILITIES: "能力与限制",
}
_IDENTITY_INLINE_FIELDS = frozenset({"姓名", "角色概述", "其他名字与称呼"})
_TERM_RE = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+", re.IGNORECASE)


class CharacterProjectionCompiler:
    def __init__(self, token_meter: ConservativeTokenMeter | None = None) -> None:
        self.token_meter = token_meter or ConservativeTokenMeter()

    def compile(
        self,
        snapshot: CharacterModelSnapshot,
        purpose: ProjectionPurpose | str,
        *,
        relevance_text: str = "",
    ) -> CharacterProjection:
        selected_purpose = self._purpose(purpose)
        policy = PROJECTION_POLICIES[selected_purpose]
        query = str(relevance_text or "")[:4_000]
        sections = self._sections(snapshot.model, policy, query)
        rendered = "\n\n".join(section.text for section in sections)
        categories = tuple(section.category for section in sections)
        custom_prompts = self._custom_prompts(snapshot.model, selected_purpose)
        fingerprint = self._projection_fingerprint(
            selected_purpose,
            categories,
            rendered,
            custom_prompts,
        )
        return CharacterProjection(
            profile_id=snapshot.profile_id,
            revision=snapshot.revision,
            content_fingerprint=snapshot.content_fingerprint,
            projection_fingerprint=fingerprint,
            purpose=selected_purpose,
            selected_categories=categories,
            sections=sections,
            rendered_text=rendered,
            token_count=self.token_meter.count_text(rendered),
            character_count=len(rendered),
            custom_prompts=custom_prompts,
        )

    def _sections(
        self,
        model: CharacterModel,
        policy: ProjectionPolicy,
        query: str,
    ) -> tuple[CharacterProjectionSection, ...]:
        lines_by_category = {
            category: self._category_lines(model, category, policy.purpose, query)
            for category in policy.categories
        }
        return tuple(
            CharacterProjectionSection(
                category,
                self._render_section(category, lines_by_category[category]),
                len(lines_by_category[category]),
            )
            for category in policy.categories
            if lines_by_category[category]
        )

    @staticmethod
    def _render_section(category: CharacterFieldCategory, lines: tuple[str, ...]) -> str:
        if category == _IDENTITY:
            return _render_identity(lines)
        if category == _DIALOGUE_REFERENCE:
            values = tuple(_split_labelled_line(line)[1] for line in lines)
            return _render_group(_HEADERS[category], values)
        return "\n\n".join(
            _render_group(label, values)
            for label, values in _group_labelled_lines(lines, fallback=_HEADERS[category])
        )

    def _category_lines(
        self,
        model: CharacterModel,
        category: CharacterFieldCategory,
        purpose: ProjectionPurpose,
        query: str,
    ) -> tuple[str, ...]:
        handlers = {
            _IDENTITY: self._identity_lines,
            _PERSONALITY: self._personality_lines,
            _SOCIAL: self._social_lines,
            _PREFERENCES: self._preference_lines,
            _LANGUAGE: self._language_lines,
            _DIALOGUE_REFERENCE: self._dialogue_reference_lines,
            _VISUAL: self._visual_lines,
            _CAPABILITIES: self._capability_lines,
        }
        return handlers[category](model, purpose, query)

    @staticmethod
    def _identity_lines(
        model: CharacterModel, purpose: ProjectionPurpose, _query: str
    ) -> tuple[str, ...]:
        identity = model.identity
        lines: list[str] = []
        if identity.name:
            lines.append(f"姓名：{CHARACTER_PLACEHOLDER}")
        if identity.overview:
            lines.append(f"角色概述：{identity.overview}")
        if identity.aliases:
            lines.append("其他名字与称呼：" + "、".join(identity.aliases))
        if purpose not in {
            ProjectionPurpose.RESPONSE_POLISH,
            ProjectionPurpose.WEB_PERSONALIZED,
        }:
            lines.extend(f"身份、经历与生活现状：{item}" for item in identity.facts)
        return tuple(lines)

    @staticmethod
    def _personality_lines(
        model: CharacterModel, purpose: ProjectionPurpose, _query: str
    ) -> tuple[str, ...]:
        value = model.personality
        fields = [
            ("性格与看重的事", value.traits_and_values),
            ("思考与行动方式", value.thinking_and_behavior),
            ("日常习惯与情绪表现", value.habits_and_emotions),
        ]
        if purpose == ProjectionPurpose.STICKER_PLANNING:
            fields = [
                ("日常习惯与情绪表现", value.habits_and_emotions),
                ("性格与看重的事", value.traits_and_values),
            ]
        return _labelled(fields)

    @staticmethod
    def _social_lines(
        model: CharacterModel, purpose: ProjectionPurpose, _query: str
    ) -> tuple[str, ...]:
        value = model.social
        if purpose is ProjectionPurpose.WEB_PERSONALIZED:
            return ()
        return _labelled(
            [
                ("与人相处的方式", value.interaction_style),
                ("关系边界与禁区", value.boundaries),
            ]
        )

    def _preference_lines(
        self, model: CharacterModel, _purpose: ProjectionPurpose, query: str
    ) -> tuple[str, ...]:
        value = model.preferences
        candidates = _labelled(
            [
                ("喜欢和感兴趣的事", value.likes_and_interests),
                ("不喜欢的事", value.dislikes),
            ]
        )
        return self._relevant(candidates, query, fallback_limit=6)

    def _language_lines(
        self, model: CharacterModel, purpose: ProjectionPurpose, _query: str
    ) -> tuple[str, ...]:
        value = model.language
        if purpose == ProjectionPurpose.STICKER_PLANNING:
            return _labelled([("平时怎么说话", value.speaking_style)])
        return _labelled(
            [
                ("平时怎么说话", value.speaking_style),
                ("发消息的习惯", value.messaging_habits),
                ("怎样称呼别人", value.address_habits),
            ]
        )

    @staticmethod
    def _dialogue_reference_lines(
        model: CharacterModel, _purpose: ProjectionPurpose, _query: str
    ) -> tuple[str, ...]:
        reference = model.dialogue_reference.strip()
        return (f"原声台词：{reference}",) if reference else ()

    @staticmethod
    def _visual_lines(
        model: CharacterModel, _purpose: ProjectionPurpose, _query: str
    ) -> tuple[str, ...]:
        value = model.visual
        return _labelled(
            [
                ("外貌与辨识特征", value.appearance),
                ("常见穿着", value.clothing),
                ("画面中不能出现的内容", value.visual_boundaries),
            ]
        )

    @staticmethod
    def _capability_lines(
        model: CharacterModel, _purpose: ProjectionPurpose, _query: str
    ) -> tuple[str, ...]:
        value = model.capabilities
        return _labelled(
            [
                ("会做什么", value.abilities),
                ("知道与不知道的事", value.knowledge_scope),
                ("做不到或受限制的事", value.limitations),
            ]
        )

    def _relevant(
        self, values: tuple[str, ...], query: str, *, fallback_limit: int
    ) -> tuple[str, ...]:
        query_terms = _terms(query)
        if not query_terms:
            return values[:fallback_limit]
        scored = [
            (len(query_terms.intersection(_terms(value))), index, value)
            for index, value in enumerate(values)
        ]
        matched = [item for item in scored if item[0] > 0]
        if not matched:
            return values[:fallback_limit]
        matched.sort(key=lambda item: (-item[0], item[1]))
        return tuple(item[2] for item in matched[:fallback_limit])

    @staticmethod
    def _purpose(value: ProjectionPurpose | str) -> ProjectionPurpose:
        try:
            return ProjectionPurpose(value)
        except ValueError as exc:
            raise UnsupportedProjectionPurpose(f"unsupported projection purpose: {value}") from exc

    @staticmethod
    def _custom_prompts(
        model: CharacterModel,
        purpose: ProjectionPurpose,
    ) -> CharacterCustomPrompts:
        if purpose in {
            ProjectionPurpose.MAIN_CORE_WITH_POLISH,
            ProjectionPurpose.MAIN_CORE_DIRECT,
        }:
            return CharacterCustomPrompts(
                main_core_modes=model.custom_prompts.main_core_modes,
                main_core_styles=model.custom_prompts.main_core_styles,
                story_styles=model.custom_prompts.story_styles,
                background_creation=model.custom_prompts.background_creation,
            )
        if purpose is ProjectionPurpose.RESPONSE_POLISH:
            return CharacterCustomPrompts(
                main_core_styles=MainCoreStylePrompts(
                    speaking_style=model.custom_prompts.main_core_styles.speaking_style,
                ),
                response_polish=model.custom_prompts.response_polish,
            )
        if purpose is ProjectionPurpose.BACKGROUND_AUTHOR:
            return CharacterCustomPrompts(
                story_styles=model.custom_prompts.story_styles,
                background_creation=model.custom_prompts.background_creation,
            )
        return CharacterCustomPrompts()

    @staticmethod
    def _projection_fingerprint(
        purpose: ProjectionPurpose,
        categories: tuple[CharacterFieldCategory, ...],
        rendered: str,
        custom_prompts: CharacterCustomPrompts,
    ) -> str:
        payload: dict[str, object] = {
            "purpose": purpose.value,
            "categories": [item.value for item in categories],
            "rendered": rendered,
        }
        custom_payload = _custom_prompts_fingerprint_payload(custom_prompts)
        if custom_payload:
            payload["custom_prompts"] = custom_payload
        material = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _labelled(fields: list[tuple[str, tuple[str, ...]]]) -> tuple[str, ...]:
    return tuple(f"{label}：{item}" for label, values in fields for item in values)


def _render_identity(lines: tuple[str, ...]) -> str:
    inline: list[str] = []
    grouped: dict[str, list[str]] = {}
    for line in lines:
        label, value = _split_labelled_line(line)
        if not label or label in _IDENTITY_INLINE_FIELDS:
            inline.append(f"{label}：{value}" if label else value)
            continue
        grouped.setdefault(label, []).append(value)
    sections: list[str] = []
    if inline:
        sections.append(_render_group(_HEADERS[_IDENTITY], tuple(inline)))
    sections.extend(_render_group(label, tuple(values)) for label, values in grouped.items())
    return "\n\n".join(sections)


def _group_labelled_lines(
    lines: tuple[str, ...], *, fallback: str
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    grouped: dict[str, list[str]] = {}
    for line in lines:
        label, value = _split_labelled_line(line)
        grouped.setdefault(label or fallback, []).append(value)
    return tuple((label, tuple(values)) for label, values in grouped.items())


def _split_labelled_line(line: str) -> tuple[str, str]:
    label, separator, value = str(line or "").partition("：")
    if not separator or not label.strip() or not value.strip():
        return "", str(line or "").strip()
    return label.strip(), value.strip()


def _render_group(label: str, values: tuple[str, ...]) -> str:
    return f"[{label}]\n" + "\n".join(f"- {value}" for value in values)


def _terms(value: str) -> frozenset[str]:
    result: set[str] = set()
    for match in _TERM_RE.findall(str(value or "").lower()):
        if match.isascii():
            result.add(match)
            continue
        result.update(match[index : index + 2] for index in range(max(1, len(match) - 1)))
    return frozenset(result)


def _custom_prompts_fingerprint_payload(
    custom_prompts: CharacterCustomPrompts,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    modes: MainCoreModePrompts = custom_prompts.main_core_modes
    if modes.self_initiated:
        payload["main_core_modes"] = {
            "self_initiated": modes.self_initiated,
        }
    styles: MainCoreStylePrompts = custom_prompts.main_core_styles
    if any(
        (
            styles.relationship_context,
            styles.speaking_style,
            styles.sticker_style,
            styles.thinking_style,
            styles.content_style,
            styles.conversation_content,
        )
    ):
        payload["main_core_styles"] = {
            "relationship_context": styles.relationship_context,
            "speaking_style": styles.speaking_style,
            "sticker_style": styles.sticker_style,
            "thinking_style": styles.thinking_style,
            "content_style": styles.content_style,
            "conversation_content": styles.conversation_content,
        }
    response_polish: ResponsePolishPrompts = custom_prompts.response_polish
    if response_polish.writing_correction:
        payload["response_polish"] = {
            "writing_correction": response_polish.writing_correction,
        }
    story_styles: StoryStylePrompts = custom_prompts.story_styles
    if any((story_styles.involvement, story_styles.stance)):
        payload["story_styles"] = {
            "involvement": story_styles.involvement,
            "stance": story_styles.stance,
        }
    background_creation = custom_prompts.background_creation
    if any(
        (
            background_creation.world_change,
            background_creation.story_boundary,
            background_creation.imagination,
            background_creation.temperature,
        )
    ):
        payload["background_creation"] = {
            "world_change": background_creation.world_change,
            "story_boundary": background_creation.story_boundary,
            "imagination": background_creation.imagination,
            "temperature": background_creation.temperature,
        }
    return payload


_PERSONA_GROUP_PRIORITY = {
    "核心身份": 0,
    "身份、经历与生活现状": 1,
    "关系边界与禁区": 2,
    "性格与看重的事": 3,
    "思考与行动方式": 4,
    "平时怎么说话": 5,
    "发消息的习惯": 6,
    "日常习惯与情绪表现": 7,
    "与人相处的方式": 8,
    "喜欢和感兴趣的事": 20,
    "不喜欢的事": 21,
    "怎样称呼别人": 22,
}


def budget_rendered_character_projection(
    rendered: str,
    *,
    max_tokens: int,
    token_meter: ConservativeTokenMeter,
) -> str:
    """Select complete semantic items, preserving the positive identity frame first."""

    groups = _projection_groups(rendered)
    if not groups or max_tokens < 1:
        return ""
    groups.sort(key=lambda item: (_PERSONA_GROUP_PRIORITY.get(item[0], 50), item[2]))
    selected: list[tuple[str, list[str]]] = []
    for label, values, _ordinal in groups:
        kept: list[str] = []
        for value in values:
            candidate = _render_selected_groups([*selected, (label, [*kept, value])])
            if token_meter.count_text(candidate) <= max_tokens:
                kept.append(value)
            elif label == "核心身份":
                bounded = _bounded_semantic_value(
                    label,
                    value,
                    prefix=[*selected, (label, list(kept))],
                    max_tokens=max_tokens,
                    token_meter=token_meter,
                )
                if bounded:
                    kept.append(bounded)
        if kept:
            selected.append((label, kept))
    return _render_selected_groups(selected)


def _bounded_semantic_value(
    label: str,
    value: str,
    *,
    prefix: list[tuple[str, list[str]]],
    max_tokens: int,
    token_meter: ConservativeTokenMeter,
) -> str:
    suffix = "…（此项按模型窗口有界保留）"
    low, high = 1, len(value)
    best = ""
    while low <= high:
        middle = (low + high) // 2
        candidate = value[:middle].rstrip() + suffix
        rendered = _render_selected_groups([*prefix[:-1], (label, [*prefix[-1][1], candidate])])
        if token_meter.count_text(rendered) <= max_tokens:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def _projection_groups(rendered: str) -> list[tuple[str, list[str], int]]:
    groups: list[tuple[str, list[str], int]] = []
    current_label = ""
    current_values: list[str] = []
    for line in str(rendered or "").splitlines():
        stripped = line.strip()
        label = _projection_group_label(stripped)
        if label is not None:
            _append_projection_group(groups, current_label, current_values)
            current_label = label
            current_values = []
        elif stripped.startswith("- ") and current_label:
            value = stripped[2:].strip()
            if value:
                current_values.append(value)
        elif stripped and current_label and current_values:
            current_values[-1] = f"{current_values[-1]}\n{stripped}"
    _append_projection_group(groups, current_label, current_values)
    return groups


def _projection_group_label(line: str) -> str | None:
    if line.startswith("[") and line.endswith("]") and len(line) > 2:
        return line[1:-1].strip()
    return None


def _append_projection_group(
    groups: list[tuple[str, list[str], int]],
    current_label: str,
    current_values: list[str],
) -> None:
    if current_label and current_values:
        groups.append((current_label, current_values, len(groups)))


def _render_selected_groups(groups: list[tuple[str, list[str]]]) -> str:
    return "\n\n".join(
        f"[{label}]\n" + "\n".join(f"- {value}" for value in values)
        for label, values in groups
        if values
    )


__all__ = [
    "CharacterProjectionCompiler",
    "PROJECTION_POLICIES",
    "ProjectionPolicy",
    "budget_rendered_character_projection",
]
