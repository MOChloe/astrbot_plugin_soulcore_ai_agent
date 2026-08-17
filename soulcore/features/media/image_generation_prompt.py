from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...shared.prompt_document import PromptBlock, compile_prompt_document, xml_text

_IMAGE_GENERATION_TASK = (
    "根据下方唯一一份最终画面规格生成一张完整图片。不得自行新增或推断人物、关系、事件、"
    "地点、物件或情绪含义；未列出的内容事实保持不具体化。可以补足不改变规格的景别、"
    "空间衔接及中性姿态或表情。"
)

_IMAGE_AUTHORITY = (
    "最终画面规格已经按职责分区；同一属性只采用最先适用的一层：\n"
    "1. “不可覆盖的事实”是唯一事实层，其中的身份、身体结构、固定标志物、视觉禁区、固定服装、"
    "世界规则和已成立事实不可被后文改写。\n"
    "2. “本轮明确要求”只在不违反事实层时生效，可以替换普通默认服装、媒介和画风。\n"
    "3. 参考图按“参考图1、参考图2……”对应实际图片，只为声明的用途提供视觉依据；未声明的身份、"
    "服装、物品、建筑、构图、画风、背景或姿势不进入成图。\n"
    "4. “未指定属性的视觉默认”只补充前面仍未确定的属性。默认绘图风格只决定媒介和画风；"
    "世界氛围只影响环境气质，不把叙事措辞变成新的画面事实。\n"
    "5. “构图落实细节”只组织尚未确定的场景关系、动作、视点、构图和光线，不复述或改写前四层。"
)


@dataclass(frozen=True, slots=True)
class ImageReferenceBinding:
    """Provider-order binding between one reference image and its declared role."""

    label: str
    purpose: str
    objective_content: str

    def render(self) -> str:
        return f"{self.label}\n用途：{self.purpose}\n客观内容：{self.objective_content}"


def generation_prompt(
    *,
    counterpart_requirements: str,
    scene_plan: str,
    world_facts: str,
    world_visual_defaults: str,
    drawing_style: str,
    persona: str,
    selected_visual_facts: str,
    reference_bindings: Sequence[ImageReferenceBinding],
) -> str:
    seen: set[str] = set()
    specification = "\n\n".join(
        section
        for section in (
            _render_spec_section(
                "不可覆盖的事实",
                (
                    ("角色视觉", persona),
                    ("世界事实与硬边界", world_facts),
                    ("本轮选定事实", selected_visual_facts),
                ),
                seen,
            ),
            _render_spec_section(
                "本轮明确要求",
                (("要求", counterpart_requirements),),
                seen,
            ),
            _render_reference_specs(reference_bindings, seen),
            _render_spec_section(
                "未指定属性的视觉默认",
                (
                    ("绘图风格", drawing_style),
                    ("世界氛围与偏好", world_visual_defaults),
                ),
                seen,
            ),
            _render_spec_section(
                "构图落实细节",
                (("细节", scene_plan),),
                seen,
            ),
        )
        if section
    )
    return compile_prompt_document(
        (
            PromptBlock(
                "任务定义",
                _IMAGE_GENERATION_TASK,
            ),
            PromptBlock("取舍规则", _IMAGE_AUTHORITY),
            PromptBlock("最终画面规格", xml_text(specification)),
        ),
        (),
    ).document


def _unique_spec_field(value: str, seen: set[str]) -> str:
    """Deduplicate only an unchanged complete field; punctuation is semantic content."""

    text = str(value or "").strip()
    if not text or text in seen:
        return ""
    seen.add(text)
    return text


def _render_spec_section(
    title: str,
    values: Sequence[tuple[str, str]],
    seen: set[str],
) -> str:
    lines: list[str] = []
    for label, value in values:
        unique = _unique_spec_field(value, seen)
        if unique:
            lines.append(f"{label}：{unique}")
    return f"[{title}]\n" + "\n".join(lines) if lines else ""


def _render_reference_specs(
    bindings: Sequence[ImageReferenceBinding],
    seen: set[str],
) -> str:
    rendered: list[str] = []
    for binding in bindings:
        lines = [binding.label, f"用途：{binding.purpose}"]
        objective = _unique_spec_field(binding.objective_content, seen)
        if objective:
            lines.append(f"客观内容：{objective}")
        rendered.append("\n".join(lines))
    return "[逐张参考用途与客观内容]\n" + "\n\n".join(rendered) if rendered else ""


__all__ = ["ImageReferenceBinding", "generation_prompt"]
