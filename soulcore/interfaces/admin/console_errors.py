"""Stable, human-oriented error envelopes for advanced settings."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from ...features.role_package.domain import RolePackageError


class ConsoleValidationError(ValueError):
    """A user-correctable validation failure bound to specific form fields."""

    def __init__(self, message: str, *, field_errors: Mapping[str, str] | None = None) -> None:
        super().__init__(message)
        self.field_errors = dict(field_errors or {})


def require_successful_settings_result(result: Mapping[str, Any]) -> None:
    """Raise one field-aware error for an unsuccessful settings controller result."""

    if result.get("ok") is not False:
        return
    error = result.get("error")
    nested_message = error.get("message") if isinstance(error, Mapping) else error
    message = result.get("message") or nested_message or "这项设置没有保存成功"
    field_errors = error.get("field_errors") if isinstance(error, Mapping) else None
    raise ConsoleValidationError(
        str(message), field_errors=field_errors if isinstance(field_errors, Mapping) else None
    )


def console_error_envelope(
    error: BaseException,
    *,
    action: str,
    status_code: int,
) -> dict[str, Any]:
    """Translate one Page failure without leaking an internal exception dump."""

    raw = str(error or "").strip()
    lowered = raw.lower()
    code, title, message, impact, anchor = _classification(lowered, status_code)
    occurrence_source = f"{action}|{code}|{type(error).__name__}|{raw}"
    occurrence_id = hashlib.sha256(occurrence_source.encode("utf-8")).hexdigest()[:16]
    envelope = {
        "code": code,
        "title": title,
        "message": message,
        "impact": impact,
        "anchor": anchor,
        "occurrence_id": occurrence_id,
        "status_code": int(status_code),
        "recovery": _recovery(code),
    }
    if isinstance(error, (ConsoleValidationError, RolePackageError)):
        envelope["message"] = str(error)
        field_errors = (
            error.field_errors
            if isinstance(error, ConsoleValidationError)
            else ({error.field: str(error)} if error.field else {})
        )
        if field_errors:
            envelope["field_errors"] = field_errors
            envelope["recovery"] = []
    return envelope


def _classification(lowered: str, status_code: int) -> tuple[str, str, str, str, str]:
    for classifier in (
        _initialization_error,
        _availability_error,
        _account_error,
        _connection_error,
        _conflict_error,
    ):
        result = classifier(lowered)
        if result is not None:
            return result
    if status_code == 400:
        return (
            "invalid_request",
            "提交的内容无法保存",
            "请检查标红字段或当前操作需要的内容。",
            "本次操作没有生效，原有数据保持不变。",
            "current-field",
        )
    return (
        "internal_error",
        "SoulCore 没有完成这次操作",
        "请重试；如果问题持续出现，可导出支持报告。",
        "本次操作没有完成。",
        "current-action",
    )


def _initialization_error(lowered: str) -> tuple[str, str, str, str, str] | None:
    if "schema recovery required [migration_failed]" in lowered:
        return (
            "schema_recovery_required",
            "数据库升级没有完成",
            "升级事务已回滚，原数据库保持不变；请查看 AstrBot 插件日志后重试。",
            "SoulCore 当前不能处理消息；不要清空数据库。",
            "app-initialization",
        )
    if "schema recovery required [newer_schema]" in lowered:
        return (
            "schema_recovery_required",
            "数据库来自更高版本",
            "请安装创建该数据库的同版或更高版 SoulCore。",
            "SoulCore 当前不能处理消息；原数据库保持不变且不会提供清空动作。",
            "app-initialization",
        )
    if "schema recovery required" in lowered:
        return (
            "schema_recovery_required",
            "数据库需要在高级设置中选择恢复方式",
            "SoulCore 没有覆盖无法安全兼容的数据，请在这里选择备份后重建或直接重建。",
            "SoulCore 当前不能处理消息；原数据库仍保持不变。",
            "app-initialization",
        )
    if "initialization failed" in lowered:
        return (
            "initialization_failed",
            "SoulCore 启动失败",
            "插件启动时发生错误；请重载 SoulCore。若仍然失败，请检查 AstrBot 插件日志。",
            "SoulCore 当前不能处理消息或读取管理数据。",
            "app-initialization",
        )
    if "initialization is still in progress" in lowered or "is not initialized" in lowered:
        return (
            "initialization_pending",
            "SoulCore 正在启动",
            "插件刚刚安装或重载，正在准备数据库和后台服务；请稍后重试。",
            "当前页面数据暂时不能读取。",
            "app-initialization",
        )
    return None


def _availability_error(lowered: str) -> tuple[str, str, str, str, str] | None:
    if "open circuits" in lowered or "all matching ai backends" in lowered:
        return (
            "all_backends_open",
            "所有候选模型接口都在暂时停用",
            "这些接口刚刚连续失败，SoulCore 正在等待冷却后再尝试。",
            "当前模型调用尚未发送到服务商。",
            "model-interactions",
        )
    if "model_not_found" in lowered or "model not found" in lowered:
        return (
            "model_not_found",
            "服务商找不到这个模型",
            "请检查模型名称以及当前账号是否有权使用。",
            "使用该模型的功能无法运行。",
            "settings-models",
        )
    return None


def _account_error(lowered: str) -> tuple[str, str, str, str, str] | None:
    if any(value in lowered for value in ("401", "403", "authentication", "invalid token")):
        return (
            "authentication",
            "模型服务拒绝了当前密钥",
            "请在“模型与接口”中更新 API Key 后重试。",
            "需要该模型的对话或后台功能暂时无法运行。",
            "settings-models",
        )
    if "429" in lowered or "rate limit" in lowered:
        return (
            "rate_limit",
            "模型服务暂时限流",
            "请求过于频繁，请稍后重试或检查服务商限额。",
            "本次模型调用没有完成。",
            "model-interactions",
        )
    if any(value in lowered for value in ("quota", "billing", "insufficient")):
        return (
            "quota_exhausted",
            "模型服务额度不足",
            "请检查服务商余额、额度或计费状态。",
            "需要该接口的模型调用都会失败。",
            "settings-models",
        )
    return None


def _connection_error(lowered: str) -> tuple[str, str, str, str, str] | None:
    if "timeout" in lowered:
        return (
            "timeout",
            "请求等待时间过长",
            "服务没有在规定时间内响应，可以稍后重试。",
            "本次操作没有完成。",
            "current-action",
        )
    if any(value in lowered for value in ("network", "failed to fetch", "connection")):
        return (
            "network",
            "暂时无法连接服务",
            "请检查网络、API 地址和代理设置后重试。",
            "当前请求无法完成。",
            "app-network",
        )
    return None


def _conflict_error(lowered: str) -> tuple[str, str, str, str, str] | None:
    if "version" in lowered or "conflict" in lowered or "changed; reload before saving" in lowered:
        return (
            "version_conflict",
            "内容已在其他位置发生变化",
            "刷新当前区域后重新修改，避免覆盖较新的内容。",
            "本次修改没有保存。",
            "current-field",
        )
    return None


def _recovery(code: str) -> list[Mapping[str, str]]:
    if code in {"authentication", "quota_exhausted", "model_not_found"}:
        return [{"kind": "navigate", "label": "检查模型与接口", "target": "settings-models"}]
    return [{"kind": "retry", "label": "重试", "target": ""}]


__all__ = [
    "ConsoleValidationError",
    "console_error_envelope",
    "require_successful_settings_result",
]
