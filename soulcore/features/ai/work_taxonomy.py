"""Canonical taxonomy for AI work records visible in advanced settings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ...contracts.ai_models import AIWorkPurpose


class AIWorkNodeRole(StrEnum):
    BUSINESS_STAGE = "BUSINESS_STAGE"
    INTERNAL_ACTION = "INTERNAL_ACTION"
    SYSTEM_STAGE = "SYSTEM_STAGE"


class AIWorkNodeKind(StrEnum):
    MODEL = "MODEL"
    WEB = "WEB"
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"
    FILE = "FILE"
    COMMAND = "COMMAND"
    SYSTEM = "SYSTEM"


class AIWorkNodeStatus(StrEnum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    SKIPPED = "SKIPPED"
    FALLBACK = "FALLBACK"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


class AIProviderAttemptStatus(StrEnum):
    PREPARING = "PREPARING"
    IN_FLIGHT = "IN_FLIGHT"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


@dataclass(frozen=True, slots=True)
class AIWorkPurposeSpec:
    label: str
    reason: str
    kind: AIWorkNodeKind


_SPECS: dict[AIWorkPurpose, AIWorkPurposeSpec] = {
    AIWorkPurpose.MAIN_CORE: AIWorkPurposeSpec("主对话", "回应对方的消息", AIWorkNodeKind.MODEL),
    AIWorkPurpose.RESPONSE_POLISH: AIWorkPurposeSpec(
        "AI 润色", "整理即将发送的文字", AIWorkNodeKind.MODEL
    ),
    AIWorkPurpose.CONVERSATION_SUMMARY: AIWorkPurposeSpec(
        "对话摘要", "整理对话摘要", AIWorkNodeKind.MODEL
    ),
    AIWorkPurpose.KNOWLEDGE_ORGANIZATION: AIWorkPurposeSpec(
        "记忆与知识整理", "整理记忆与知识", AIWorkNodeKind.MODEL
    ),
    AIWorkPurpose.CHARACTER_PROFILE_IMPORT: AIWorkPurposeSpec(
        "角色资料整理", "把已有角色设定整理为可编辑资料", AIWorkNodeKind.MODEL
    ),
    AIWorkPurpose.TURN_CLASSIFICATION: AIWorkPurposeSpec(
        "消息回合判断", "判断消息是否属于同一回合", AIWorkNodeKind.MODEL
    ),
    AIWorkPurpose.GROUP_INTERJECTION: AIWorkPurposeSpec(
        "群聊插话判断", "判断是否应在群聊中回应", AIWorkNodeKind.MODEL
    ),
    AIWorkPurpose.GROUP_REPLY_RELOCATION: AIWorkPurposeSpec(
        "群聊回复落点复核", "复核正在组织的群聊回复是否仍然自然", AIWorkNodeKind.MODEL
    ),
    AIWorkPurpose.BACKGROUND_WORLD: AIWorkPurposeSpec(
        "世界层创作", "续写角色世界中的组织、人物与远方事件", AIWorkNodeKind.MODEL
    ),
    AIWorkPurpose.BACKGROUND_LIFE_DIRECTION: AIWorkPurposeSpec(
        "人生方向创作", "检查并续写角色的长期人生方向", AIWorkNodeKind.MODEL
    ),
    AIWorkPurpose.BACKGROUND_STORY_SOURCE: AIWorkPurposeSpec(
        "故事源创作", "创造可参与也可忽略的开放事件", AIWorkNodeKind.MODEL
    ),
    AIWorkPurpose.BACKGROUND_KEYFRAME: AIWorkPurposeSpec(
        "关键帧创作", "定期评估角色生活是否有真实变化", AIWorkNodeKind.MODEL
    ),
    AIWorkPurpose.BACKGROUND_ORDINARY: AIWorkPurposeSpec(
        "普通帧创作", "理解已送达对话并模拟角色当前生活", AIWorkNodeKind.MODEL
    ),
    AIWorkPurpose.FILE_GENERATION: AIWorkPurposeSpec("文件生成", "生成文件", AIWorkNodeKind.FILE),
    AIWorkPurpose.IMAGE_GENERATION: AIWorkPurposeSpec("图片生成", "生成图片", AIWorkNodeKind.IMAGE),
    AIWorkPurpose.IMAGE_UNDERSTANDING: AIWorkPurposeSpec(
        "图片理解", "理解图片内容", AIWorkNodeKind.IMAGE
    ),
    AIWorkPurpose.AUDIO_TRANSCRIPTION: AIWorkPurposeSpec(
        "语音识别", "将收到的语音转写为文字", AIWorkNodeKind.AUDIO
    ),
    AIWorkPurpose.AUDIO_SPEECH_GENERATION: AIWorkPurposeSpec(
        "语音生成", "将回复文字合成为语音", AIWorkNodeKind.AUDIO
    ),
    AIWorkPurpose.WEB_SEARCH: AIWorkPurposeSpec("联网搜索", "查询网页资料", AIWorkNodeKind.WEB),
    AIWorkPurpose.WEB_READ: AIWorkPurposeSpec("网页读取", "读取网页内容", AIWorkNodeKind.WEB),
    AIWorkPurpose.WEB_IMAGE_SEARCH: AIWorkPurposeSpec(
        "联网搜图", "搜索外部图片", AIWorkNodeKind.WEB
    ),
    AIWorkPurpose.STICKER_COLLECTION: AIWorkPurposeSpec(
        "表情包搜集", "自动搜集表情包", AIWorkNodeKind.IMAGE
    ),
    AIWorkPurpose.STICKER_CHECK: AIWorkPurposeSpec(
        "表情包检查", "检查表情包内容", AIWorkNodeKind.IMAGE
    ),
    AIWorkPurpose.TIMER_RUN: AIWorkPurposeSpec(
        "定时任务", "执行定时角色任务", AIWorkNodeKind.MODEL
    ),
    AIWorkPurpose.TIMER_LIFECYCLE_REVIEW: AIWorkPurposeSpec(
        "定时器有效性检查", "判断周期定时器是否仍有继续运行的意义", AIWorkNodeKind.MODEL
    ),
    AIWorkPurpose.ADMIN_MODEL_TEST: AIWorkPurposeSpec(
        "模型连接测试", "测试模型配置", AIWorkNodeKind.MODEL
    ),
    AIWorkPurpose.ADMIN_WEB_TEST: AIWorkPurposeSpec(
        "联网能力测试", "测试联网配置", AIWorkNodeKind.WEB
    ),
    AIWorkPurpose.MODEL_REQUEST: AIWorkPurposeSpec(
        "其他 AI 工作", "执行 AI 工作", AIWorkNodeKind.MODEL
    ),
}

_DURABLE_TASKS_WITH_OWN_WORKFLOW = frozenset({"MAIN_CORE", "BACKGROUND_AUTHOR"})


def durable_task_owns_workflow(task_type: str) -> bool:
    """Return whether the executor records its own complete business workflow."""

    return str(task_type or "").strip().upper() in _DURABLE_TASKS_WITH_OWN_WORKFLOW


def normalize_work_purpose(value: AIWorkPurpose | str | None) -> AIWorkPurpose:
    if isinstance(value, AIWorkPurpose):
        return value
    try:
        return AIWorkPurpose(str(value or "").strip().upper())
    except ValueError:
        return AIWorkPurpose.MODEL_REQUEST


def work_purpose_spec(value: AIWorkPurpose | str | None) -> AIWorkPurposeSpec:
    return _SPECS[normalize_work_purpose(value)]


def work_purpose_options() -> list[dict[str, str]]:
    return [{"value": purpose.value, "label": spec.label} for purpose, spec in _SPECS.items()]


__all__ = [
    "AIProviderAttemptStatus",
    "AIWorkNodeKind",
    "AIWorkNodeRole",
    "AIWorkNodeStatus",
    "AIWorkPurposeSpec",
    "durable_task_owns_workflow",
    "normalize_work_purpose",
    "work_purpose_options",
    "work_purpose_spec",
]
