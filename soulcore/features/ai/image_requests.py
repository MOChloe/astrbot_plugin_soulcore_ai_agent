"""Image-generation request normalization."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from typing import Any

from ...contracts.ai_models import (
    AIBackendDescriptor,
    AICapabilityRequest,
    AIImageBackendCapabilities,
    AIImageContent,
)
from ...contracts.vision import VisionInspectionMode, VisionSequenceKind
from ...shared.prompt_document import compile_task_prompt
from .image_responses import invalid


def model(request: AICapabilityRequest, backend: AIBackendDescriptor, fallback: str) -> str:
    model = str(request.payload.get("model") or backend.model or fallback).strip()
    if not model:
        raise invalid("No model is configured for this image backend", backend)
    return model


def generation_input(
    request: AICapabilityRequest, backend: AIBackendDescriptor
) -> tuple[str, int, tuple[AIImageContent, ...]]:
    prompt = str(request.payload.get("prompt") or "").strip()
    if not prompt:
        raise invalid("image.generate requires a prompt", backend)
    count = int(request.payload.get("count") or 1)
    if count < 1 or count > 5:
        raise invalid("image count must be between 1 and 5", backend)
    references = payload_images(request.payload.get("references", ()))
    if references and backend.metadata.get("reference_image") is False:
        references = ()
    if len(references) > 1 and backend.metadata.get("multiple_references") is False:
        references = references[:1]
    maximum = max(1, min(5, int(backend.metadata.get("maximum_outputs") or 5)))
    if count > maximum:
        raise invalid(f"image count exceeds backend maximum of {maximum}", backend)
    return prompt, count, references


def validate_features(
    features: AIImageBackendCapabilities,
    payload: Mapping[str, Any],
    count: int,
    references: Sequence[AIImageContent],
    backend: AIBackendDescriptor,
) -> None:
    if not features.text_to_image:
        raise invalid("Backend does not declare text-to-image support", backend)
    if references and not features.reference_image:
        raise invalid("Backend does not declare reference-image support", backend)
    if len(references) > 1 and not features.multiple_references:
        raise invalid("Backend does not declare multiple-reference support", backend)
    if count > max(1, min(5, int(features.maximum_outputs))):
        raise invalid("Requested image count exceeds backend declaration", backend)
    ratio = str(payload.get("aspect_ratio") or "auto")
    if ratio != "auto" and features.supported_ratios and ratio not in features.supported_ratios:
        raise invalid("Requested aspect ratio is not supported by this backend", backend)
    size = str(payload.get("size") or "auto")
    if size != "auto" and features.supported_sizes and size not in features.supported_sizes:
        raise invalid("Requested image size is not supported by this backend", backend)


def payload_images(values: Any) -> tuple[AIImageContent, ...]:
    if isinstance(values, AIImageContent):
        return (values,)
    output: list[AIImageContent] = []
    for value in values or ():
        if isinstance(value, AIImageContent):
            output.append(value)
            continue
        if not isinstance(value, Mapping):
            continue
        raw = value.get("data") or b""
        if isinstance(raw, str):
            try:
                raw = base64.b64decode(raw, validate=True)
            except (ValueError, TypeError):
                raw = b""
        output.append(
            AIImageContent(
                str(value.get("mime_type") or "image/png"),
                raw if isinstance(raw, bytes) else b"",
                str(value.get("url") or ""),
                str(value.get("asset_id") or ""),
            )
        )
    return tuple(output)


def openai_image_parts(images: Sequence[AIImageContent]) -> list[Mapping[str, Any]]:
    parts: list[Mapping[str, Any]] = []
    for image in images:
        url = image.url
        if not url and image.data:
            url = f"data:{image.mime_type};base64,{base64.b64encode(image.data).decode('ascii')}"
        if url:
            parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def vision_prompt(
    sequence_kind: VisionSequenceKind | str = VisionSequenceKind.SINGLE_IMAGE,
    inspection_mode: VisionInspectionMode | str = VisionInspectionMode.OBJECTIVE,
) -> str:
    sequence = VisionSequenceKind(str(sequence_kind))
    mode = VisionInspectionMode(str(inspection_mode))
    return compile_task_prompt(
        task_definition=_vision_task_definition(sequence, mode),
        task_input="",
        output_contract=_vision_output_contract(sequence, mode),
    ).document


def _vision_task_definition(
    sequence: VisionSequenceKind,
    mode: VisionInspectionMode,
) -> str:
    sequence_instruction = {
        VisionSequenceKind.SINGLE_IMAGE: "你将收到一张静态图片。",
        VisionSequenceKind.ANIMATION_CONTACT_SHEET: (
            "你将收到一张动图分镜拼图，其中的画格来自同一段动图，"
            "从左到右、从上到下就是时间顺序。画格是连续过程的不同时刻，不是无关图片或"
            "同时发生的场景。"
        ),
        VisionSequenceKind.GIF_REPRESENTATIVE_FRAMES: (
            "你将收到同一段 GIF 动图的多张代表帧，传入顺序就是时间顺序。"
            "这些帧是连续过程的不同时刻，不是无关图片或同时发生的场景。"
        ),
    }[sequence]
    mode_instruction = {
        VisionInspectionMode.OBJECTIVE: "",
        VisionInspectionMode.STICKER_QUALITY: (
            "这些画面来自一张表情包候选。除了观察画面，还要判断是否存在网站名、水印、"
            "平台角标等非内容来源标记；只填写“是”或“否”，不抄录标记本身。"
            "本模式必须填写[[交流观感]]；即使它不像合格表情，也要说明图片通常会传出怎样的"
            "聊天质感，或为何几乎没有稳定的聊天表达效果。"
        ),
        VisionInspectionMode.OCR_DIAGNOSTIC: (
            "这张图片用于检查文字识别效果。以实际画面为准，优先逐字转录所有清晰正文，"
            "保留可见的字符、标点和换行；没有参考答案，不猜测看不清的字符。"
        ),
    }[mode]
    task_lines = [sequence_instruction]
    if mode_instruction:
        task_lines.append(mode_instruction)
    if sequence is not VisionSequenceKind.SINGLE_IMAGE:
        task_lines.append(
            "在[[动作变化]]中概括画格或帧能够确认的起点、关键转折、结果或循环方式；"
            "采样间没有展示的动作不猜测，没有可确认的变化就省略该字段。"
        )
    task_lines.append(_vision_picture_content_instruction(sequence))
    task_lines.extend(
        [
            (
                "允许按常识识别画面内容；无法仅凭画面确认的专名、来源、因果或意图不写成事实。"
                "[[主体身份]]、[[画面媒介]]和[[表情包特征]]只在确有内容且有助理解时填写，"
                "不为填满格式而重复画面内容。"
            ),
            (
                "[[交流观感]]只描述图片本身稳定呈现的整体聊天质感和通常会起到的交流效果。"
                "明显属于表情包、反应图或梗图时，优先写沙雕、抽象、土味、扯淡感、装正经、"
                "阴阳、敷衍、荒诞、装傻或欠揍等第一眼感受，不要只复述主体和动作；最多八十字。"
                "不得把它写成发送者此刻正在嘲讽、拒绝或敷衍，也不得评价作者或画中人物的品格。"
                "普通照片、截图或信息图片在非表情包质检模式下省略该字段。"
            ),
            (
                "[[正文文字]]指画面中承载内容或表达的文字；网站名、水印、平台角标等非内容来源标记"
                "不算正文。没有正文时选择“无正文文字”并省略正文文字；正文完整转录时选择"
                "“正文已转录”；有正文却不能完整辨认时选择“正文不能完整辨认”，写下能够确认的"
                "部分，一个字也无法确认时可以省略正文文字。"
            ),
            (
                "[[安全]]只表示画面能否在普通聊天中直接展示。露骨性内容、严重血腥伤害、自伤、"
                "仇恨侮辱或会造成人身伤害的危险行为填写“否”；关键区域严重模糊或遮挡到无法判断"
                "这些内容时也填写“否”；其余填写“是”。"
            ),
        ]
    )
    return "\n".join(task_lines)


def _vision_picture_content_instruction(sequence: VisionSequenceKind) -> str:
    return (
        (
            "简洁观察画面。在[[画面内容]]中用一段自足的文字概括主要内容，选择有助理解的"
            "主体外观、动作或表情、环境、物件与构图，不逐项盘点，也不重复[[正文文字]]。"
        )
        if sequence is VisionSequenceKind.SINGLE_IMAGE
        else (
            "先沿时间顺序把全部画格或帧作为一个完整过程阅读。在[[画面内容]]中用一段自足的"
            "文字概括整段动图主要发生什么、动作或表情怎样整体展开以及最终形成的效果；单个"
            "画格只作为过程证据，主体外观、环境和物件只选择理解完整动作所需的内容。不逐帧"
            "盘点，不用某一帧姿势代替整段含义，也不重复[[正文文字]]。"
        )
    )


def _vision_output_contract(
    sequence: VisionSequenceKind,
    mode: VisionInspectionMode,
) -> str:
    sequence_observation_field = (
        "[[动作变化]]（可选）：概括能够确认的起点、关键转折、结果或循环方式；"
        "不逐帧罗列，没有则省略\n"
        if sequence is not VisionSequenceKind.SINGLE_IMAGE
        else ""
    )
    source_marker_field = (
        "[[存在非内容来源标记]]（必填）：只能是“是”或“否”；只判断是否存在，不抄录标记内容\n"
        if mode is VisionInspectionMode.STICKER_QUALITY
        else ""
    )
    social_impression_field = (
        "[[交流观感]]（必填）：最多八十字，概括整体聊天质感及通常交流效果\n"
        if mode is VisionInspectionMode.STICKER_QUALITY
        else (
            "[[交流观感]]（可选）：最多八十字；仅在明显属于表情包、反应图或梗图时概括"
            "整体聊天质感及通常交流效果\n"
        )
    )
    return (
        "只输出一个“图像观察”块，以 <图像观察> 开始，以 </图像观察> 结束；开始和结束标签"
        "各自独占一行。块中的字段行写成 [[字段名]]: 内容。必填字段必须出现，可选字段没有内容时"
        "省略整行。\n\n"
        "字段规则：\n"
        "[[画面内容]]（必填）：用一段简洁、自足的文字概括主要画面，不重复正文文字\n"
        "[[主体身份]]（可选）：仅在画面足以确认专名且确实有助理解时填写\n"
        f"{sequence_observation_field}"
        "[[画面媒介]]（可选）：仅在媒介信息确实有助理解时填写\n"
        "[[正文文字]]（可选）：逐字转录能够确认的正文，不含非内容来源标记\n"
        "[[文字状态]]（必填）：只能是“无正文文字”“正文已转录”或“正文不能完整辨认”\n"
        "[[表情包特征]]（可选）：仅在画面明确呈现且有助理解时填写\n"
        f"{social_impression_field}"
        "[[安全]]（必填）：只能是“是”或“否”\n"
        "[[安全说明]]（[[安全]]为“否”时必填）：简述可见风险或无法判断的原因；"
        "为“是”时省略整行\n"
        f"{source_marker_field}"
    )


def openai_generation_fields(
    payload: Mapping[str, Any], model: str, prompt: str, count: int
) -> dict[str, Any]:
    output: dict[str, Any] = {"model": model, "prompt": prompt, "n": count}
    if str(payload.get("size") or "auto") != "auto":
        output["size"] = str(payload["size"])
    if payload.get("quality"):
        output["quality"] = str(payload["quality"])
    if payload.get("background"):
        output["background"] = str(payload["background"])
    return output


def optional_image_config(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    config = {}
    if str(payload.get("aspect_ratio") or "auto") != "auto":
        config["aspect_ratio"] = str(payload["aspect_ratio"])
    if str(payload.get("size") or "auto") != "auto":
        config["size"] = str(payload["size"])
    return {"image_config": config} if config else {}
