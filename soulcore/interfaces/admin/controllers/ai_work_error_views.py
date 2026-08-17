"""Stable, user-facing error guidance for AI work records."""

from __future__ import annotations

from typing import Any

from ....contracts.ai_models import AIErrorCode

ERROR_GUIDANCE: dict[AIErrorCode, tuple[str, str, str]] = {
    AIErrorCode.INVALID_REQUEST: (
        "请求数据无效",
        "本阶段没有开始执行。",
        "检查阶段输入和必填参数。",
    ),
    AIErrorCode.BACKEND_NOT_FOUND: (
        "没有可用模型",
        "本阶段无法调用所需能力。",
        "检查用途对应的模型和能力池配置。",
    ),
    AIErrorCode.UNSUPPORTED_CAPABILITY: (
        "模型不支持此能力",
        "当前后端无法完成该阶段。",
        "换用支持该能力的模型。",
    ),
    AIErrorCode.AUTHENTICATION: (
        "模型接口鉴权失败",
        "模型请求没有执行成功。",
        "检查 API Key 和接口地址。",
    ),
    AIErrorCode.PERMISSION: (
        "模型接口权限不足",
        "模型请求被服务端拒绝。",
        "检查账号、模型权限和组织设置。",
    ),
    AIErrorCode.QUOTA_EXHAUSTED: (
        "模型额度不足",
        "本阶段无法继续调用模型。",
        "充值或切换可用模型。",
    ),
    AIErrorCode.RATE_LIMIT: (
        "模型接口限流",
        "当前尝试失败，系统可能已重试。",
        "降低并发或等待额度恢复。",
    ),
    AIErrorCode.NETWORK: (
        "模型网络连接失败",
        "请求没有得到有效响应。",
        "检查网络、代理和接口地址。",
    ),
    AIErrorCode.REMOTE_5XX: ("模型服务暂时异常", "服务端没有完成请求。", "稍后重试或切换后端。"),
    AIErrorCode.TIMEOUT: (
        "模型调用超时",
        "请求结果未知或未及时返回。",
        "检查服务延迟和阶段超时配置。",
    ),
    AIErrorCode.EMPTY_OUTPUT: (
        "模型返回空结果",
        "该轮输出无法进入业务流程。",
        "检查模型、Prompt 和内容安全策略。",
    ),
    AIErrorCode.CONTEXT_BUDGET: (
        "上下文超过模型容量",
        "请求在发送前被拒绝。",
        "减少上下文或选择更大窗口模型。",
    ),
    AIErrorCode.OUTPUT_CONTRACT: (
        "模型输出格式不合格",
        "该轮结果没有被业务接受。",
        "查看校验记录并调整 Prompt。",
    ),
    AIErrorCode.SAFETY_REFUSAL: (
        "模型拒绝生成",
        "该阶段没有得到可用结果。",
        "检查输入内容和模型安全策略。",
    ),
    AIErrorCode.COMMAND_TIMEOUT: (
        "内部动作超时",
        "相关工具动作未按时完成。",
        "检查工具服务和超时配置。",
    ),
    AIErrorCode.COMMAND_FAILED: (
        "内部动作失败",
        "相关工具结果没有成功产生。",
        "展开内部动作查看参数和结果。",
    ),
    AIErrorCode.COMMAND_PROTOCOL: (
        "内部动作格式错误",
        "模型动作没有被执行。",
        "查看解析与校验记录。",
    ),
    AIErrorCode.CIRCUIT_OPEN: (
        "模型接口已临时熔断",
        "系统暂时跳过该后端。",
        "检查连续失败原因或等待自动恢复。",
    ),
    AIErrorCode.CAPACITY_BUSY: (
        "模型并发已满",
        "本次调用未获得执行容量。",
        "稍后重试或增加可用后端。",
    ),
    AIErrorCode.ADAPTER_INCOMPATIBLE: (
        "模型适配器不兼容",
        "请求无法由当前适配器处理。",
        "检查适配器版本和模型类型。",
    ),
    AIErrorCode.PROMPT_CACHE_MARKER_UNSUPPORTED: (
        "缓存标记不受支持",
        "系统已在同一轮自动移除标记重试，不影响正常回复。",
        "无需操作；系统会在冷却期后重新协商。",
    ),
    AIErrorCode.INTERNAL: (
        "AI 子系统内部错误",
        "该阶段没有正常完成。",
        "查看高级诊断并提交完整错误信息。",
    ),
}

if set(ERROR_GUIDANCE) != set(AIErrorCode):  # pragma: no cover - import-time contract
    raise RuntimeError("AI error guidance must cover every AIErrorCode")


def known_error_guidance(code: str, message: str) -> dict[str, Any] | None:
    try:
        AIErrorCode(str(code or "").upper())
    except ValueError:
        return None
    return error_view(code, message)


def error_view(code: str, message: str) -> dict[str, Any] | None:
    normalized = str(code or "").upper()
    if not normalized and not message:
        return None
    try:
        title, impact, suggestion = ERROR_GUIDANCE[AIErrorCode(normalized)]
    except ValueError:
        title, impact, suggestion = (
            "处理阶段出现问题",
            "该问题可能只影响当前阶段，具体以运行状态为准。",
            "展开高级详情查看诊断码、输入和结果。",
        )
    return {
        "code": normalized,
        "title": title,
        "message": str(message or title),
        "impact": impact,
        "suggestion": suggestion,
    }


__all__ = ["error_view", "known_error_guidance"]
