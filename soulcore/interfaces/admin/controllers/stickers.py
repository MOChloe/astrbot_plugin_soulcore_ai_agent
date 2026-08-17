"""Sticker configuration and runtime administration composition."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from ....features.ai.durable_tasks import DurableAITaskManager
from ....features.conversation.ports import ConversationRepositoryPort
from ....features.media.storage import MediaStorageCoordinator
from ....features.profiles.ports import ProfilesRepositoryPort
from ....features.stickers.domain import (
    DEFAULT_STICKER_REQUIREMENTS,
    StickerConfig,
    StickerIntakeKind,
)
from ....features.stickers.ports import StickerRepositoryPort
from ....features.stickers.service import StickerCollectorPlugin
from .sticker_admin_ports import StickerAITaskPort
from .sticker_intake_lifecycle import StickerIntakeLifecycleMixin
from .sticker_intake_upload import StickerIntakeUploadMixin
from .sticker_references import StickerReferenceController
from .sticker_runtime_admin import StickerRuntimeAdminMixin


class StickerRuntimeTransitionMixin:
    async def _apply_sticker_runtime_transition(
        self, profile_id: str, scope: str, current: Any, updated: Any
    ) -> None:
        changes = _sticker_runtime_changes(current, updated)
        runtime_changed, automatic_runtime_changed = changes[:2]
        player_disabled, web_disabled, generation_disabled = changes[2:]
        if not runtime_changed:
            return
        await self._transition_sticker_instances(
            profile_id,
            scope,
            master_disabled=not bool(updated.enabled),
            automatic_runtime_changed=automatic_runtime_changed,
            player_disabled=player_disabled,
            web_disabled=web_disabled,
            generation_disabled=generation_disabled,
        )
        await self._transition_sticker_intake(
            profile_id,
            scope,
            master_disabled=not bool(updated.enabled),
            web_disabled=web_disabled,
        )

    async def _transition_sticker_instances(
        self,
        profile_id: str,
        scope: str,
        *,
        master_disabled: bool,
        automatic_runtime_changed: bool,
        player_disabled: bool,
        web_disabled: bool,
        generation_disabled: bool,
    ) -> None:
        current_time = datetime.now(UTC)
        for instance in await self.profiles_repository.list_character_instances(profile_id):
            if str(instance.scope) != str(scope):
                continue
            if automatic_runtime_changed:
                latest = await self.conversation_repository.get_latest_dialogue_message_id(
                    profile_id, instance.instance_id
                )
                await self.repository.update_sticker_trigger_state(
                    profile_id,
                    instance.instance_id,
                    processed_through_message_id=latest,
                    enabled_at=current_time,
                    last_success_at=None,
                    cooldown_until=None,
                )
            await self._cancel_disabled_sticker_tasks(
                profile_id,
                instance.instance_id,
                master_disabled=master_disabled,
                player_disabled=player_disabled,
                web_disabled=web_disabled,
                generation_disabled=generation_disabled,
            )

    async def _transition_sticker_intake(
        self,
        profile_id: str,
        scope: str,
        *,
        master_disabled: bool,
        web_disabled: bool,
    ) -> None:
        intake = await self.repository.get_active_sticker_intake_session(profile_id, scope)
        if intake is None or str(intake["status"]) == "FINALIZING":
            return
        if master_disabled:
            frozen = await self.repository.freeze_sticker_intake_session(
                str(intake["session_id"]),
                cancelled=True,
            )
            await self._cancel_intake_task(frozen, reason="表情包总开关已关闭")
            await self._finalize_frozen_intake(frozen)
            return
        if (
            web_disabled
            and str(intake["intake_kind"]) == StickerIntakeKind.SEARCH.value
            and str(intake["status"]) == "RUNNING"
        ):
            await self._cancel_intake_task(intake, reason="联网搜图已关闭")
            await self.repository.fail_sticker_intake_session(
                str(intake["session_id"]),
                error="联网搜图已关闭，已停止继续搜索",
            )

    async def _cancel_disabled_sticker_tasks(
        self,
        profile_id: str,
        instance_id: str,
        *,
        master_disabled: bool,
        player_disabled: bool,
        web_disabled: bool,
        generation_disabled: bool,
    ) -> None:
        if not (master_disabled or player_disabled or web_disabled or generation_disabled):
            return
        terminal = {"SUCCEEDED", "FAILED", "CANCELLED", "RECOVERY_REQUIRED", "DEFERRED"}
        tasks = await self.ai_repository.list_ai_tasks(
            profile_id=profile_id, instance_id=instance_id, limit=1000
        )
        for task in tasks:
            if await self._disabled_sticker_task_should_cancel(
                task,
                profile_id,
                instance_id,
                terminal=terminal,
                master_disabled=master_disabled,
                player_disabled=player_disabled,
                web_disabled=web_disabled,
                generation_disabled=generation_disabled,
            ):
                await self.ai_tasks.cancel(
                    int(task["task_id"]),
                    actor_id="sticker-runtime-gate",
                    reason="sticker runtime configuration disabled",
                )

    async def _disabled_sticker_task_should_cancel(
        self,
        task: Mapping[str, Any],
        profile_id: str,
        instance_id: str,
        *,
        terminal: set[str],
        master_disabled: bool,
        player_disabled: bool,
        web_disabled: bool,
        generation_disabled: bool,
    ) -> bool:
        if str(task.get("status") or "") in terminal:
            return False
        task_type = str(task.get("task_type") or "")
        managed = {"STICKER_COLLECTION", "STICKER_CHECK", "STICKER_INTAKE"}
        if task_type not in managed:
            return False
        if master_disabled:
            return True
        if task_type == "STICKER_COLLECTION":
            return web_disabled or generation_disabled
        if task_type == "STICKER_INTAKE":
            return await self._disabled_intake_task_should_cancel(task, web_disabled)
        return await self._disabled_check_task_should_cancel(
            task,
            profile_id,
            instance_id,
            player_disabled=player_disabled,
            web_disabled=web_disabled,
            generation_disabled=generation_disabled,
        )

    async def _disabled_intake_task_should_cancel(
        self,
        task: Mapping[str, Any],
        web_disabled: bool,
    ) -> bool:
        if not web_disabled:
            return False
        session_id = str(dict(task.get("input") or {}).get("session_id") or "")
        session = await self.repository.get_sticker_intake_session(session_id)
        return session is not None and str(session.get("intake_kind") or "") == "SEARCH"

    async def _disabled_check_task_should_cancel(
        self,
        task: Mapping[str, Any],
        profile_id: str,
        instance_id: str,
        *,
        player_disabled: bool,
        web_disabled: bool,
        generation_disabled: bool,
    ) -> bool:
        candidate_id = str(dict(task.get("input") or {}).get("candidate_id") or "")
        candidate = await self.repository.get_sticker_candidate(
            profile_id,
            instance_id,
            candidate_id,
        )
        if candidate is None:
            return False
        source = str(candidate.source_kind.value)
        if source == "PLAYER":
            return player_disabled
        if source == "WEB":
            return web_disabled
        if source == "GENERATED":
            return generation_disabled
        return False


def _sticker_runtime_changes(current: Any, updated: Any) -> tuple[bool, bool, bool, bool, bool]:
    runtime_changed = any(
        _sticker_runtime_field_changed(current, updated, field)
        for field in (
            "enabled",
            "player_collection_enabled",
            "web_collection_enabled",
            "generation_enabled",
        )
    )
    automatic_changed = any(
        _sticker_runtime_field_changed(current, updated, field)
        for field in ("enabled", "web_collection_enabled", "generation_enabled")
    )
    return (
        runtime_changed,
        automatic_changed,
        bool(current.player_collection_enabled) and not bool(updated.player_collection_enabled),
        bool(current.web_collection_enabled) and not bool(updated.web_collection_enabled),
        bool(current.generation_enabled) and not bool(updated.generation_enabled),
    )


def _sticker_runtime_field_changed(current: Any, updated: Any, field: str) -> bool:
    return bool(getattr(current, field)) != bool(getattr(updated, field))


class StickersAdminController(
    StickerIntakeUploadMixin,
    StickerIntakeLifecycleMixin,
    StickerRuntimeTransitionMixin,
    StickerRuntimeAdminMixin,
):
    def __init__(
        self,
        repository: StickerRepositoryPort,
        profiles_repository: ProfilesRepositoryPort,
        conversation_repository: ConversationRepositoryPort,
        ai_repository: StickerAITaskPort,
        ai_tasks: DurableAITaskManager,
        media_storage: MediaStorageCoordinator,
        sticker_collector: StickerCollectorPlugin,
        references: StickerReferenceController,
        is_terminating: Callable[[], bool],
    ) -> None:
        self.repository = repository
        self.profiles_repository = profiles_repository
        self.conversation_repository = conversation_repository
        self.ai_repository = ai_repository
        self.ai_tasks = ai_tasks
        self.media_storage = media_storage
        self.sticker_collector = sticker_collector
        self.references = references
        self.is_terminating = is_terminating
        self._intake_finalize_locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _sticker_config_view(config: StickerConfig) -> dict[str, Any]:
        return {
            "enabled": config.enabled,
            "player_collection_enabled": config.player_collection_enabled,
            "web_collection_enabled": config.web_collection_enabled,
            "generation_collection_enabled": config.generation_enabled,
            "trigger_mode": config.trigger_mode,
            "turn_threshold": config.turn_threshold,
            "elapsed_hours": config.elapsed_hours,
            "capacity": config.library_limit,
            "web_daily_limit": config.web_daily_limit,
            "generation_daily_limit": config.generated_daily_limit,
            "requirements": config.requirements,
            "version": config.version,
        }

    async def sticker_config_snapshot(self, profile_id: str, scope: str) -> dict[str, Any]:
        config = await self.repository.get_sticker_config(profile_id, scope)
        return {"sticker_config": self._sticker_config_view(config)}

    async def quick_setup_snapshot(
        self,
        profile_id: str,
        *,
        vision_ready: bool,
        web_ready: bool,
        image_ready: bool,
    ) -> dict[str, Any]:
        private, group, inventory = await asyncio.gather(
            self.repository.get_sticker_config(profile_id, "private"),
            self.repository.get_sticker_config(profile_id, "group"),
            self.repository.quick_setup_sticker_inventory(profile_id),
        )
        private_view = self._sticker_config_view(private)
        group_view = self._sticker_config_view(group)
        comparable = tuple(key for key in private_view if key != "version")
        return {
            "private": private_view,
            "group": group_view,
            "mixed": any(private_view[key] != group_view[key] for key in comparable),
            "inventory": {
                "private": int(inventory.get("private") or 0),
                "group": int(inventory.get("group") or 0),
                "total": int(inventory.get("total") or 0),
            },
            "dependencies": {
                "vision_ready": bool(vision_ready),
                "web_ready": bool(web_ready),
                "image_ready": bool(image_ready),
            },
        }

    async def quick_setup_configure(
        self,
        profile_id: str,
        payload: Mapping[str, Any],
        *,
        vision_ready: bool,
        web_ready: bool,
        image_ready: bool,
    ) -> dict[str, Any]:
        enabled = self._quick_setup_bool(payload, "enabled")
        player = self._quick_setup_bool(payload, "player_collection_enabled")
        web = self._quick_setup_bool(payload, "web_collection_enabled")
        generation = self._quick_setup_bool(payload, "generation_collection_enabled")
        if not enabled:
            player = web = generation = False
        inventory = await self.repository.quick_setup_sticker_inventory(profile_id)
        self._validate_quick_setup_choice(
            enabled=enabled,
            player=player,
            web=web,
            generation=generation,
            vision_ready=vision_ready,
            web_ready=web_ready,
            image_ready=image_ready,
            inventory_total=int(inventory.get("total") or 0),
        )
        raw_versions = payload.get("expected_versions")
        if not isinstance(raw_versions, Mapping):
            raise ValueError("快速设置缺少当前表情包配置版本")
        expected_versions = {
            "private": self._positive_version(raw_versions.get("private")),
            "group": self._positive_version(raw_versions.get("group")),
        }
        patch = {
            "enabled": enabled,
            "player_collection_enabled": player,
            "web_collection_enabled": web,
            "generation_enabled": generation,
            "trigger_mode": "TURNS_ONLY",
            "turn_threshold": payload.get("turn_threshold", 20),
            "elapsed_hours": 24,
            "library_limit": payload.get("capacity", 1000),
            "web_daily_limit": payload.get("web_daily_limit", 4),
            "generated_daily_limit": payload.get("generation_daily_limit", 1),
            "requirements": payload.get("requirements", DEFAULT_STICKER_REQUIREMENTS),
        }
        current, updated, changed = await self.repository.update_sticker_configs_atomically(
            profile_id,
            patch,
            expected_versions=expected_versions,
        )
        if changed:
            for scope in ("private", "group"):
                await self._apply_sticker_runtime_transition(
                    profile_id,
                    scope,
                    current[scope],
                    updated[scope],
                )
        return {
            "ok": True,
            "applied": changed,
            "stickers": await self.quick_setup_snapshot(
                profile_id,
                vision_ready=vision_ready,
                web_ready=web_ready,
                image_ready=image_ready,
            ),
        }

    async def validate_quick_setup(
        self,
        profile_id: str,
        *,
        vision_ready: bool,
        web_ready: bool,
        image_ready: bool,
    ) -> None:
        snapshot = await self.quick_setup_snapshot(
            profile_id,
            vision_ready=vision_ready,
            web_ready=web_ready,
            image_ready=image_ready,
        )
        for scope in ("private", "group"):
            config = snapshot[scope]
            self._validate_quick_setup_choice(
                enabled=bool(config["enabled"]),
                player=bool(config["player_collection_enabled"]),
                web=bool(config["web_collection_enabled"]),
                generation=bool(config["generation_collection_enabled"]),
                vision_ready=vision_ready,
                web_ready=web_ready,
                image_ready=image_ready,
                inventory_total=int(snapshot["inventory"]["total"]),
            )

    @staticmethod
    def _quick_setup_bool(payload: Mapping[str, Any], key: str) -> bool:
        value = payload.get(key)
        if not isinstance(value, bool):
            raise ValueError("表情包快速设置缺少明确选择")
        return value

    @staticmethod
    def _positive_version(value: Any) -> int:
        try:
            version = int(value)
        except (TypeError, ValueError):
            raise ValueError("表情包配置版本无效") from None
        if version < 1:
            raise ValueError("表情包配置版本无效")
        return version

    @staticmethod
    def _validate_quick_setup_choice(
        *,
        enabled: bool,
        player: bool,
        web: bool,
        generation: bool,
        vision_ready: bool,
        web_ready: bool,
        image_ready: bool,
        inventory_total: int,
    ) -> None:
        if not enabled:
            return
        acquisitions = player or web or generation
        if acquisitions and not vision_ready:
            raise ValueError("要收藏或制作新表情，请先完成图片理解设置")
        if web and not web_ready:
            raise ValueError("要联网寻找新表情，请先完成联网查询设置")
        if generation and not image_ready:
            raise ValueError("要自动生成新表情，请先完成图片生成设置")
        if not acquisitions and inventory_total < 1:
            raise ValueError("这个角色还没有可用表情，请允许一种收集方式或暂不开启")
        if not vision_ready and inventory_total < 1:
            raise ValueError("请先完成图片理解设置，角色才能安全使用表情包")

    async def save_sticker_config(
        self, profile_id: str, scope: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        current = await self.repository.get_sticker_config(profile_id, scope)
        expected = payload.get("expected_version")
        if expected not in (None, "") and int(expected) != int(current.version):
            raise ValueError("表情包配置已被其他请求修改，请刷新后重试")
        mapping = {
            "enabled": "enabled",
            "player_collection_enabled": "player_collection_enabled",
            "web_collection_enabled": "web_collection_enabled",
            "generation_collection_enabled": "generation_enabled",
            "trigger_mode": "trigger_mode",
            "turn_threshold": "turn_threshold",
            "elapsed_hours": "elapsed_hours",
            "capacity": "library_limit",
            "web_daily_limit": "web_daily_limit",
            "generation_daily_limit": "generated_daily_limit",
            "requirements": "requirements",
        }
        patch = {target: payload[source] for source, target in mapping.items() if source in payload}
        if not patch:
            raise ValueError("没有可保存的表情包配置")
        updated = await self.repository.update_sticker_config(profile_id, scope, patch)
        await self._apply_sticker_runtime_transition(profile_id, scope, current, updated)
        return await self.sticker_config_snapshot(profile_id, scope)

    async def clear_sticker_instance(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        instance = await self.profiles_repository.get_character_instance(profile_id, instance_id)
        if instance is None:
            raise ValueError("unknown conversation instance")
        for scoped in await self.profiles_repository.list_character_instances(profile_id):
            if str(scoped.scope) != str(instance.scope):
                continue
            await self._cancel_disabled_sticker_tasks(
                profile_id,
                str(scoped.instance_id),
                master_disabled=True,
                player_disabled=True,
                web_disabled=True,
                generation_disabled=True,
            )
        result = await self.repository.clear_sticker_instance_data(profile_id, instance_id)
        return {
            "ok": True,
            "message": "当前范围的共享表情库与当前聊天私有库已清空并重建",
            **(dict(result) if isinstance(result, Mapping) else {}),
        }


__all__ = ["StickersAdminController"]
