"""Sticker runtime status, task controls, and library administration."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from ....features.ai.durable_tasks import DurableAITaskManager
from ....features.conversation.ports import ConversationRepositoryPort
from ....features.profiles.ports import ProfilesRepositoryPort
from ....features.stickers.domain import StickerCheckVerdict, StickerItemStatus
from ....features.stickers.policy import StickerRuntimePolicy, load_sticker_runtime_policy
from ....features.stickers.ports import StickerRepositoryPort
from ....features.stickers.service import StickerCollectorPlugin
from ..presentation import ai_task_view, jsonable
from .sticker_admin_ports import StickerAITaskPort
from .sticker_references import StickerReferenceController


class StickerRuntimeAdminMixin:
    repository: StickerRepositoryPort
    profiles_repository: ProfilesRepositoryPort
    conversation_repository: ConversationRepositoryPort
    ai_repository: StickerAITaskPort
    ai_tasks: DurableAITaskManager
    sticker_collector: StickerCollectorPlugin
    references: StickerReferenceController

    async def sticker_runtime_snapshot(
        self,
        profile_id: str,
        instance_id: str,
        *,
        view: str = "candidates",
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        instance = await self.profiles_repository.get_character_instance(profile_id, instance_id)
        if instance is None:
            raise ValueError("unknown conversation instance")
        config = await self.repository.get_sticker_config(profile_id, instance.scope)
        policy = await load_sticker_runtime_policy(
            self.repository,
            self.profiles_repository,
            profile_id,
            instance_id=instance_id,
            config=config,
        )
        state = await self.repository.get_sticker_trigger_state(profile_id, instance_id)
        selected_view = self._sticker_view(view)
        current_page, size = max(1, int(page or 1)), max(1, min(20, int(page_size or 20)))
        page_result = await self._sticker_page(
            profile_id, instance_id, selected_view, current_page, size
        )
        now = datetime.now(UTC)
        page_items = list(page_result.get("items") or [])
        item_views = await self._sticker_item_views(
            profile_id, instance_id, selected_view, page_items, now
        )
        if selected_view in {"tasks", "errors"}:
            page_result = await self._sticker_task_page(
                profile_id, instance_id, selected_view, current_page, size
            )
            page_items = list(page_result["items"])
        active_task = await self._active_sticker_task(state)
        progress = await self._sticker_progress(profile_id, instance_id, state)
        stats = await self._sticker_stats(profile_id, instance_id, config)
        empty_page = {
            "total": 0,
            "page": current_page,
            "page_size": size,
            "page_count": 1,
        }
        serialized_page = jsonable(page_items)
        config_view = (await self.sticker_config_snapshot(profile_id, instance.scope))[
            "sticker_config"
        ]
        return self._sticker_runtime_view(
            profile_id,
            instance_id,
            selected_view,
            config,
            config_view,
            state,
            progress,
            stats,
            active_task,
            policy,
            page_result,
            empty_page,
            serialized_page,
            item_views,
            now,
        )

    async def sticker_config_snapshot(self, profile_id: str, scope: str) -> dict[str, Any]:
        raise NotImplementedError

    def _sticker_runtime_view(
        self,
        profile_id: str,
        instance_id: str,
        view: str,
        config: Any,
        config_view: Mapping[str, Any],
        state: Mapping[str, Any],
        progress: int,
        stats: Mapping[str, Any],
        active_task: Any,
        policy: StickerRuntimePolicy,
        page_result: Mapping[str, Any],
        empty_page: Mapping[str, Any],
        serialized_page: Any,
        item_views: list[dict[str, Any]],
        now: datetime,
    ) -> dict[str, Any]:
        last_error = str(state.get("last_error") or "")
        is_active = bool(active_task)
        normal_status = "正在搜集" if is_active else ("上次运行失败" if last_error else "等待触发")
        runtime_view = {
            "profile_id": profile_id,
            "instance_id": instance_id,
            "config": dict(config_view),
            "stats": stats,
            "trigger_state": {
                **jsonable(state),
                "turn_progress": progress,
                "turn_threshold": config.turn_threshold,
            },
            "active_task": ai_task_view(active_task) if active_task else {},
            "active": is_active,
            "status": normal_status,
            "status_tone": "danger" if last_error else ("success" if is_active else "neutral"),
            "current_stage": str((active_task or {}).get("status") or normal_status),
            "last_finished_at": state.get("last_success_at"),
            "next_trigger_at": state.get("cooldown_until"),
            "last_error": last_error,
            "blockers": (
                [] if policy.collection_enabled else ["表情包总开关或所有搜集来源当前未启用"]
            ),
            "view": view,
            "pagination": {
                key: page_result.get(key, empty_page[key])
                for key in ("total", "page", "page_size", "page_count")
            },
            "can_run": bool(policy.collection_enabled) and not bool(active_task),
            "can_stop": bool(active_task),
            "updated_at": now,
        }
        runtime_view.update(_runtime_page_sections(view, serialized_page, item_views))
        return runtime_view

    @staticmethod
    def _sticker_view(view: str) -> str:
        selected = str(view or "candidates").strip().lower()
        valid = {
            "candidates",
            "quarantine",
            "checks",
            "items",
            "archived",
            "tasks",
            "errors",
        }
        return selected if selected in valid else "candidates"

    async def _sticker_page(
        self, profile_id: str, instance_id: str, view: str, page: int, size: int
    ) -> dict[str, Any]:
        if view in {"candidates", "quarantine"}:
            statuses = (
                ("QUARANTINED",)
                if view == "quarantine"
                else ("PENDING", "CHECKING", "WAITING_CHECK")
            )
            return await self.repository.page_sticker_candidates(
                profile_id,
                instance_id,
                statuses=statuses,
                page=page,
                page_size=size,
            )
        if view == "checks":
            return await self.repository.page_sticker_checks(
                profile_id, instance_id, page=page, page_size=size
            )
        if view in {"items", "archived"}:
            statuses = ("ARCHIVED",) if view == "archived" else ("ACTIVE", "NEEDS_REVIEW")
            return await self.repository.page_sticker_items(
                profile_id,
                instance_id,
                statuses=statuses,
                page=page,
                page_size=size,
            )
        return {
            "items": [],
            "total": 0,
            "page": page,
            "page_size": size,
            "page_count": 1,
        }

    async def _sticker_item_views(
        self,
        profile_id: str,
        instance_id: str,
        view: str,
        items: list[Any],
        now: datetime,
    ) -> list[dict[str, Any]]:
        if view not in {"items", "archived"}:
            return []
        recent = await self.repository.sticker_recent_usage_stats(profile_id, instance_id)
        result = []
        for item in items:
            uses = int(recent.get(item.item_id, 0))
            result.append(
                {
                    **jsonable(item),
                    "recent_usage_count": uses,
                    "eviction_score": round(self._eviction_score(item, uses, now), 3),
                    "thumbnail_data_url": await self.references.sticker_item_thumbnail(
                        profile_id, instance_id, item.item_id
                    ),
                }
            )
        return result

    @staticmethod
    def _eviction_score(item: Any, recent_uses: int, now: datetime) -> float:
        age_days = (
            max(0.0, (now - item.last_used_at).total_seconds() / 86400.0)
            if item.last_used_at
            else 9999.0
        )
        player_bonus = 20.0 if item.source_kind.value == "PLAYER" else 0.0
        return (
            float(item.reinforcement_score) * 10.0
            + float(item.persona_score) * 5.0
            + recent_uses * 3.0
            + int(item.usage_count)
            - min(100.0, age_days / 3.0)
            + player_bonus
        )

    async def _sticker_task_page(
        self, profile_id: str, instance_id: str, view: str, page: int, size: int
    ) -> dict[str, Any]:
        tasks = await self.ai_repository.list_ai_tasks(
            profile_id=profile_id, instance_id=instance_id, limit=1000
        )
        rows = [
            ai_task_view(task)
            for task in tasks
            if str(task.get("task_type") or "").startswith("STICKER_")
        ]
        rows.sort(key=lambda row: int(row.get("task_id") or 0), reverse=True)
        if view == "errors":
            rows = [
                {
                    "task_id": row.get("task_id"),
                    "status": row.get("status"),
                    "error": row.get("last_error"),
                    "updated_at": row.get("updated_at"),
                }
                for row in rows
                if row.get("last_error")
            ]
        total, offset = len(rows), (page - 1) * size
        return {
            "items": rows[offset : offset + size],
            "total": total,
            "page": page,
            "page_size": size,
            "page_count": max(1, (total + size - 1) // size),
        }

    async def _active_sticker_task(self, state: Mapping[str, Any]) -> Any | None:
        if not state.get("active_task_id"):
            return None
        task = await self.ai_repository.get_ai_task(int(state["active_task_id"]))
        active = {
            "SCHEDULED",
            "READY",
            "RETRY_WAIT",
            "RUNNING",
            "PAUSE_REQUESTED",
            "PAUSED",
            "CANCEL_REQUESTED",
        }
        return task if task and str(task.get("status") or "") in active else None

    async def _sticker_progress(
        self, profile_id: str, instance_id: str, state: Mapping[str, Any]
    ) -> int:
        latest = await self.conversation_repository.get_latest_dialogue_message_id(
            profile_id, instance_id
        )
        return await self.conversation_repository.count_dialogue_turns(
            profile_id,
            instance_id,
            after_message_id=int(state.get("processed_through_message_id") or 0),
            through_message_id=latest,
        )

    async def _sticker_stats(
        self, profile_id: str, instance_id: str, config: Any
    ) -> dict[str, Any]:
        stats = await self.repository.sticker_stats(profile_id, instance_id)
        stats.update(
            {
                "item_count": int(stats.get("active", 0)),
                "capacity": config.library_limit,
            }
        )
        waiting = await self.repository.page_sticker_candidates(
            profile_id,
            instance_id,
            statuses=("WAITING_CHECK",),
            page=1,
            page_size=1,
        )
        stats["waiting_check"] = int(waiting.get("total") or 0)
        return stats

    async def run_sticker_collection(
        self,
        profile_id: str,
        instance_id: str,
        *,
        mode: str = "collect",
        theme: str = "",
    ) -> dict[str, Any]:
        if self.ai_tasks is None or self.sticker_collector is None:
            raise RuntimeError("表情包后台任务尚未就绪")
        instance = await self.profiles_repository.get_character_instance(profile_id, instance_id)
        if instance is None:
            raise ValueError("unknown conversation instance")
        config = await self.repository.get_sticker_config(profile_id, instance.scope)
        policy = await load_sticker_runtime_policy(
            self.repository,
            self.profiles_repository,
            profile_id,
            instance_id=instance_id,
            config=config,
        )
        normalized_mode = self._validate_sticker_run_mode(mode, policy)
        state = await self.repository.get_sticker_trigger_state(profile_id, instance_id)
        existing = await self._reuse_sticker_task(state)
        if existing:
            return existing
        frozen = await self.conversation_repository.get_latest_dialogue_message_id(
            profile_id, instance_id
        )
        library = await self.repository.ensure_sticker_library(
            profile_id, instance_id, library_kind="CORE"
        )
        task = await self.ai_repository.create_ai_task(
            profile_id,
            "STICKER_COLLECTION",
            instance_id=instance_id,
            task_class="BACKGROUND",
            capability="sticker.collect",
            priority=20,
            mutex_key="sticker-library:" + str(library["library_id"]),
            idempotency_key=f"sticker:manual:{instance_id}:{uuid.uuid4().hex}",
            input_data={
                "mode": normalized_mode,
                "theme": str(theme)[:500],
                "manual": True,
                "frozen_message_id": frozen,
            },
            recovery_policy="RESUME_CHECKPOINT",
            max_attempts=4,
        )
        await self.repository.update_sticker_trigger_state(
            profile_id,
            instance_id,
            frozen_through_message_id=frozen,
            active_task_id=int(task["task_id"]),
            cooldown_until=None,
            last_error="",
        )
        return {
            "ok": True,
            "queued": True,
            "task_id": int(task["task_id"]),
            "message": "表情包搜集任务已创建",
        }

    @staticmethod
    def _validate_sticker_run_mode(mode: str, policy: StickerRuntimePolicy) -> str:
        normalized = (
            "collect"
            if str(mode or "collect").lower() == "manual"
            else str(mode or "collect").lower()
        )
        if normalized != "collect":
            raise ValueError("sticker run mode must be collect")
        policy.require_collection()
        return "collect"

    async def _reuse_sticker_task(self, state: Mapping[str, Any]) -> dict[str, Any] | None:
        task_id = int(state.get("active_task_id") or 0)
        if not task_id:
            return None
        active = await self.ai_repository.get_ai_task(task_id)
        if active and str(active.get("status") or "") in {
            "SCHEDULED",
            "READY",
            "RETRY_WAIT",
            "PAUSED",
        }:
            active = await self.ai_repository.expedite_ai_task(
                task_id, actor_id="sticker-manual-run"
            )
        terminal = {
            "SUCCEEDED",
            "FAILED",
            "CANCELLED",
            "RECOVERY_REQUIRED",
            "DEFERRED",
        }
        if active and str(active.get("status") or "") not in terminal:
            return {
                "ok": True,
                "queued": True,
                "task_id": task_id,
                "message": "已有表情包任务，已请求立即执行",
            }
        return None

    async def stop_sticker_collection(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        if self.ai_tasks is None:
            raise RuntimeError("持久AI任务服务不可用")
        state = await self.repository.get_sticker_trigger_state(profile_id, instance_id)
        task_id = int(state.get("active_task_id") or 0)
        if not task_id:
            return {"ok": True, "stopped": False, "message": "当前没有活动任务"}
        result = await self.ai_tasks.cancel(
            task_id, actor_id="sticker-admin", reason="管理员停止表情包搜集"
        )
        if result is None or str(result.get("status") or "") == "CANCELLED":
            await self.repository.update_sticker_trigger_state(
                profile_id,
                instance_id,
                active_task_id=None,
                frozen_through_message_id=None,
            )
        return {
            "ok": True,
            "task_id": task_id,
            "status": str((result or {}).get("status") or "CANCELLED"),
            "message": "正在安全停止" if result else "已停止",
        }

    async def sticker_admin_action(
        self, profile_id: str, instance_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        action = str(payload.get("action") or "").lower()
        kind = str(payload.get("record_kind") or "").lower()
        record_id = str(payload.get("record_id") or "").strip()
        if not action or not record_id:
            raise ValueError("action and record_id are required")
        candidate_result = await self._candidate_action(
            action, kind, profile_id, instance_id, record_id
        )
        if candidate_result is not None:
            return candidate_result
        if kind not in {"items", "item", "library", "archived"}:
            raise ValueError("该操作只适用于正式表情包")
        return await self._library_action(action, profile_id, instance_id, record_id)

    async def _candidate_action(
        self, action: str, kind: str, profile_id: str, instance_id: str, record_id: str
    ) -> dict[str, Any] | None:
        if action == "accept":
            if self.sticker_collector is None:
                raise RuntimeError("表情包服务尚未就绪")
            item = await self.sticker_collector.admin_accept_candidate(
                profile_id,
                instance_id,
                record_id,
                reason="管理员在控制台覆盖AI语义／人设判断",
            )
            return {
                "ok": True,
                "item": jsonable(item),
                "message": "已按管理员覆盖规则接纳；文件、归属、隔离与有界去重仍已强制校验",
            }
        if action == "recheck":
            return await self._recheck_candidate(profile_id, instance_id, record_id)
        if action == "delete" and kind in {"candidates", "candidate", "quarantine"}:
            return await self._delete_candidate(profile_id, instance_id, record_id)
        if action == "reject":
            return await self._reject_candidate(profile_id, instance_id, record_id)
        return None

    async def _recheck_candidate(
        self, profile_id: str, instance_id: str, record_id: str
    ) -> dict[str, Any]:
        candidate = await self.repository.get_sticker_candidate(profile_id, instance_id, record_id)
        if candidate is None:
            raise ValueError("只能重新Check仍有来源证据的候选")
        policy = await load_sticker_runtime_policy(
            self.repository,
            self.profiles_repository,
            profile_id,
            instance_id=instance_id,
        )
        policy.require_source(candidate.source_kind)
        task = await self.ai_repository.create_ai_task(
            profile_id,
            "STICKER_CHECK",
            instance_id=instance_id,
            task_class="BACKGROUND",
            capability="sticker.check",
            priority=30,
            mutex_key=f"sticker-check:{record_id}",
            idempotency_key=f"sticker-check:{record_id}:{uuid.uuid4().hex}",
            input_data={"mode": "recheck", "candidate_id": record_id, "manual": True},
            recovery_policy="RESTART_SAFE",
            max_attempts=4,
        )
        return {
            "ok": True,
            "task_id": int(task["task_id"]),
            "message": "已进入正式Check任务",
        }

    async def _delete_candidate(
        self, profile_id: str, instance_id: str, record_id: str
    ) -> dict[str, Any]:
        await self.repository.delete_sticker_candidate(profile_id, instance_id, record_id)
        return {"ok": True, "message": "候选已移除，相关媒体已进入安全清理队列"}

    async def _reject_candidate(
        self, profile_id: str, instance_id: str, record_id: str
    ) -> dict[str, Any]:
        candidate = await self.repository.get_sticker_candidate(profile_id, instance_id, record_id)
        if candidate is None:
            raise ValueError("candidate not found")
        await self.repository.record_sticker_check(
            profile_id,
            instance_id,
            record_id,
            verdict=StickerCheckVerdict.REJECT,
            reason="ADMIN_REJECTED",
        )
        return {"ok": True, "message": "候选已拒绝"}

    async def _library_action(
        self, action: str, profile_id: str, instance_id: str, record_id: str
    ) -> dict[str, Any]:
        if action == "regenerate_description":
            policy = await load_sticker_runtime_policy(
                self.repository,
                self.profiles_repository,
                profile_id,
                instance_id=instance_id,
            )
            policy.require_enabled()
            item = await self.references.regenerate_item_description(
                profile_id, instance_id, record_id
            )
            return {
                "ok": True,
                "item": jsonable(item),
                "message": "描述已按原图重新生成，后续检索将使用新描述",
            }
        if action in {"reinforce", "unreinforce"}:
            item = await self.repository.reinforce_sticker_item(
                profile_id,
                instance_id,
                record_id,
                strength=5 if action == "reinforce" else -5,
                reason="高级设置操作",
                run_id="admin",
            )
            return {"ok": True, "item": jsonable(item), "message": "强化值已调整"}
        status = {
            "archive": StickerItemStatus.ARCHIVED,
            "delete": StickerItemStatus.DELETED,
            "restore": StickerItemStatus.ACTIVE,
        }.get(action)
        if status is None:
            raise ValueError("unknown sticker action")
        item = await self.repository.set_sticker_item_status(
            profile_id, instance_id, record_id, status
        )
        return {"ok": True, "item": jsonable(item), "message": "表情包状态已更新"}


def _runtime_page_sections(
    view: str, serialized_page: Any, item_views: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "candidates": serialized_page if view == "candidates" else [],
        "quarantine": serialized_page if view == "quarantine" else [],
        "checks": serialized_page if view == "checks" else [],
        "items": item_views if view == "items" else [],
        "archived": item_views if view == "archived" else [],
        "clusters": [],
        "duplicates": [],
        "usages": [],
        "tasks": serialized_page if view == "tasks" else [],
        "errors": serialized_page if view == "errors" else [],
    }


__all__ = ["StickerRuntimeAdminMixin"]
