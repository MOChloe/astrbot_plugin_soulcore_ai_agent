"""The single model-visible semantic recall command."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from ..ai.service import ModelVisibleCommandResult
from ..recall import RecallMode, RecallRequest
from .command_context import _active


async def recall_context(_event: Any, need: str) -> Any:
    """Recall facts, events and changes without exposing retrieval controls."""

    collector = _active()
    if collector.recall_query_calls >= 2:
        return "error: 本次行动最多显式回想两次"
    value = str(need or "").strip()
    if not value:
        return "error: 想回想的内容不能为空"
    if not collector.profile_id or not collector.instance_id:
        return "error: 当前交流无法回想已保存资料"
    service = collector.recall_service
    if service is None:
        return "error: 回想暂时不可用"
    collector.recall_query_calls += 1
    current_time = collector.player_profile_confirmed_at
    if not isinstance(current_time, datetime):
        current_time = datetime.now(UTC)
    try:
        bundle = await asyncio.wait_for(
            service.recall(
                RecallRequest(
                    profile_id=collector.profile_id,
                    instance_id=collector.instance_id,
                    need=value,
                    mode=RecallMode.EXPLICIT,
                    current_time=current_time,
                    recent_visible_context=tuple(collector.recent_visible_context[-6:]),
                    visible_source_fingerprints=frozenset(collector.visible_history_fingerprints),
                    excluded_document_keys=frozenset(
                        collector.visible_recall_document_keys | collector.recalled_document_keys
                    ),
                    token_budget=1200,
                )
            ),
            timeout=10.0,
        )
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        return ModelVisibleCommandResult("回想超时；请缩小想确认的内容后再试。")
    except Exception:
        return ModelVisibleCommandResult("回想暂时无法完成可靠核对。")
    collector.recalled_document_keys.update(bundle.document_keys)
    return ModelVisibleCommandResult(service.render(bundle, token_budget=1200))


__all__ = ["recall_context"]
