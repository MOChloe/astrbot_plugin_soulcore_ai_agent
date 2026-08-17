"""Deterministic notices appended after response polishing."""

from __future__ import annotations

from typing import Any

from ...contracts.system_notice import soulcore_system_notice


def image_failure_notice(decision: dict[str, Any]) -> str | None:
    failures = list(decision.get("image_generation_failures") or [])
    if not failures or _has_selected_image(decision):
        return None
    failure_codes = {
        str(item.get("error") or "").strip().lower() for item in failures if isinstance(item, dict)
    }
    if failure_codes & {"image_service_unavailable", "unavailable"}:
        return soulcore_system_notice("图片服务当前不可用，所以这次没有生成图片。请稍后再试。")
    if failure_codes & {"cancelled_or_timed_out", "timeout", "timeouterror"}:
        return soulcore_system_notice(
            "这次图片生成没有完成，可能是等待时间过长或任务被取消。请重新试一次。"
        )
    if failure_codes & {"no_inspected_output"}:
        return soulcore_system_notice(
            "生成的图片没有通过发送前检查，所以没有发送。请换一种画面描述再试。"
        )
    provider_tokens = ("model", "provider", "backend", "api", "http")
    if failure_codes and all(
        any(token in code for token in provider_tokens) for code in failure_codes
    ):
        return soulcore_system_notice("图片服务没有完成这次生成，所以没有图片可发送。请稍后再试。")
    return soulcore_system_notice("图片生成过程中出现问题，所以这次没有图片可发送。请重新试一次。")


def _has_selected_image(decision: dict[str, Any]) -> bool:
    return any(
        item.get("kind") == "IMAGE"
        for item in list(decision.get("expression_steps") or [])
        if isinstance(item, dict)
    )


__all__ = ["image_failure_notice"]
