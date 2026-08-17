"""Administrator projection for recurring Timer lifecycle health."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ....features.timers.domain import TimerScope
from ....features.timers.service import TIMER_LIFECYCLE_REVIEW_CAPABILITY


async def timer_lifecycle_snapshot(
    timer_repository: Any,
    model_gateway: Any | None,
    *,
    profile_id: str,
    instance_id: str,
) -> dict[str, Any]:
    snapshot = await timer_repository.timer_lifecycle_snapshot(TimerScope(profile_id, instance_id))
    configured = await _model_configured(model_gateway, profile_id)
    recent = snapshot.get("recent") if isinstance(snapshot.get("recent"), Mapping) else None
    return {
        "active_recurring_count": int(snapshot.get("active_recurring_count") or 0),
        "pending_review_count": int(snapshot.get("pending_review_count") or 0),
        "auto_completed_count": int(snapshot.get("auto_completed_count") or 0),
        "recent_conclusion": _recent_value(recent, "decision", "status"),
        "recent_at": _recent_value(recent, "updated_at"),
        "last_error": _recent_value(recent, "error_code"),
        "model_configured": configured,
        "configuration_problem": "" if configured else "审查模型未配置",
    }


async def _model_configured(model_gateway: Any | None, profile_id: str) -> bool:
    if model_gateway is None:
        return False
    return (
        await model_gateway.resolve_backend_hint(
            capability=TIMER_LIFECYCLE_REVIEW_CAPABILITY,
            profile_id=profile_id,
        )
        is not None
    )


def _recent_value(recent: Mapping[str, Any] | None, *keys: str) -> str:
    if recent is None:
        return ""
    return str(next((recent.get(key) for key in keys if recent.get(key)), "") or "")


__all__ = ["timer_lifecycle_snapshot"]
