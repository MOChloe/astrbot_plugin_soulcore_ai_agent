"""AstrBot platform capability detection kept at the adapter boundary."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from ...features.delivery.capabilities import DeliveryCapability, QQAccountTier, QQEnvironment

QQ_OFFICIAL_ADAPTER_NAMES = frozenset({"qq_official", "qq_official_webhook"})
ONEBOT_ADAPTER_NAMES = frozenset({"aiocqhttp", "onebot", "napcat"})
PERSONAL_WECHAT_ADAPTER_NAMES = frozenset({"weixin_oc"})
QQ_OFFICIAL_ADAPTER_MODULES = (
    "astrbot.core.platform.sources.qqofficial",
    "astrbot.core.platform.sources.qq_official",
)
ONEBOT_ADAPTER_MODULES = ("astrbot.core.platform.sources.aiocqhttp",)
PERSONAL_WECHAT_ADAPTER_MODULES = ("astrbot.core.platform.sources.weixin_oc",)


def detect_delivery_capability(
    platform: Any,
    *,
    qq_environment: QQEnvironment | str | None = None,
    qq_account_tier: QQAccountTier | str | None = None,
) -> DeliveryCapability:
    """Read only non-secret adapter identity and explicit environment hints."""

    platform_id, adapter_name, config = _platform_identity(platform)
    qq_official, onebot, personal_wechat = _trusted_adapter_flags(platform, adapter_name)
    environment = _qq_environment(platform, config, qq_environment, qq_official)
    tier = _qq_tier(config, qq_account_tier)
    account_identity = _qq_account_identity(platform, config)
    return DeliveryCapability(
        platform_id=platform_id,
        adapter_name=adapter_name,
        qq_official=qq_official,
        onebot=onebot,
        personal_wechat=personal_wechat,
        qq_environment=environment,
        qq_account_tier=tier,
        qq_account_identity=account_identity,
        account_identity_confirmed=bool(account_identity),
        quote=qq_official or onebot,
        mention=onebot,
        retract_self=qq_official or onebot,
        returns_id=qq_official or onebot,
        inbound_recall_notice=onebot,
    )


def _platform_identity(platform: Any) -> tuple[str, str, Mapping[str, Any]]:
    meta = getattr(platform, "meta", None)
    meta = meta() if callable(meta) else meta
    config = getattr(platform, "config", None)
    return (
        str(_value(meta, "id") or getattr(platform, "platform_id", "")),
        str(_value(meta, "name") or "generic"),
        config if isinstance(config, Mapping) else {},
    )


def _trusted_adapter_flags(platform: Any, adapter_name: str) -> tuple[bool, bool, bool]:
    normalized = str(adapter_name or "").strip().lower().replace("-", "_")
    qq_official = normalized in QQ_OFFICIAL_ADAPTER_NAMES
    onebot = normalized in ONEBOT_ADAPTER_NAMES
    personal_wechat = normalized in PERSONAL_WECHAT_ADAPTER_NAMES
    if qq_official or onebot or personal_wechat:
        return qq_official, onebot, personal_wechat
    module = str(getattr(type(platform), "__module__", "") or "").strip().lower()
    return (
        any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in QQ_OFFICIAL_ADAPTER_MODULES
        ),
        any(
            module == prefix or module.startswith(f"{prefix}.") for prefix in ONEBOT_ADAPTER_MODULES
        ),
        any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in PERSONAL_WECHAT_ADAPTER_MODULES
        ),
    )


def personal_wechat_session_ready(platform: Any, target_id: str | None) -> bool:
    """Check only whether AstrBot holds an opaque iLink context for this contact.

    AstrBot's personal-WeChat adapter requires a context token captured from an
    inbound private message before ``send_by_session`` can address that user.
    Token values remain owned by AstrBot and are never returned or logged here.
    """

    target = str(target_id or "").strip()
    context_tokens = getattr(platform, "_context_tokens", None)
    if not target or not isinstance(context_tokens, Mapping):
        return False
    return bool(str(context_tokens.get(target) or "").strip())


def _qq_environment(
    platform: Any,
    config: Mapping[str, Any],
    value: QQEnvironment | str | None,
    qq_official: bool,
) -> QQEnvironment:
    if value is None and qq_official:
        # QQ sandbox membership is configured remotely per recipient.  The
        # WebSocket adapter's ``is_sandbox=False`` describes its HTTP endpoint,
        # not whether this openid belongs to the developer-platform sandbox.
        # Treat an undeclared QQ Official connection as sandbox; formal-account
        # quota rules require an explicit SoulCore environment declaration.
        value = config.get("soulcore_qq_environment") or QQEnvironment.SANDBOX
    default = QQEnvironment.FORMAL
    return _enum_or_default(QQEnvironment, value, default)


def _qq_tier(config: Mapping[str, Any], value: QQAccountTier | str | None) -> QQAccountTier:
    value = value or config.get("soulcore_qq_account_tier") or config.get("qq_account_tier")
    return _enum_or_default(QQAccountTier, value, QQAccountTier.UNCERTIFIED)


def _qq_account_identity(platform: Any, config: Mapping[str, Any]) -> str:
    appid = str(getattr(platform, "appid", "") or config.get("appid") or "").strip()
    if not appid:
        return ""
    return "appid-sha256:" + hashlib.sha256(f"qq_official:{appid}".encode()).hexdigest()


def _value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _enum_or_default(enum_type, value: Any, default):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value).lower())
    except (TypeError, ValueError):
        return default


__all__ = ["detect_delivery_capability", "personal_wechat_session_ready"]
