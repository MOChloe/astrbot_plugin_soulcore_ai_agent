"""Guided quick-setup actions for Advanced Settings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...contracts.thinking import require_thinking_policy
from .player_views import player_character_view, player_role_ref, player_role_view
from .quick_setup_contact import (
    configure_quick_setup_contact,
    quick_setup_contact_snapshot,
    validate_quick_setup_contact,
)

_RELATIONSHIP_CONTEXTS = {
    "cross_world_communication",
    "same_world_separate_lives",
}
_THINKING_STYLES = {
    "balanced",
    "center_on_the_other",
    "lovers",
    "devoted_lover",
}
_STICKER_STYLES = {"casual_fun", "only_when_fitting"}


class QuickSetupPageMixin:
    """Compose the guided setup workflow from existing feature surfaces."""

    async def _quick_setup_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        rows, profile_id = await self._quick_setup_role(payload)
        if not profile_id:
            return {
                "role": None,
                "required": False,
                "quick_setup_decided": True,
                "enabled": False,
            }
        profile = await self.profiles_repository.get_profile(profile_id)
        if profile is None:
            raise ValueError("当前角色配置不存在")
        role_row = self._quick_setup_role_row(rows, profile_id)
        decided = bool(profile.quick_setup_decided)
        return {
            "role": player_role_view(role_row, selected=True),
            "required": not decided,
            "quick_setup_decided": decided,
            "enabled": bool(profile.enabled),
        }

    async def _quick_setup_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        rows, profile_id = await self._quick_setup_role(payload)
        if not profile_id:
            raise ValueError("还没有可使用的角色")
        snapshot = await self.ai.handle("ai_quick_setup_snapshot", profile_id, payload)
        main = await self.profiles.main_config_snapshot(profile_id)
        profile = await self.profiles_repository.get_profile(profile_id)
        if profile is None:
            raise ValueError("当前角色配置不存在")
        web = await self.web.handle("quick_setup_snapshot", profile_id, payload)
        readiness = _quick_setup_readiness(snapshot, main, web)
        sections = await self._quick_setup_sections(
            profile_id,
            readiness,
            web,
        )
        return {
            **snapshot,
            "role": player_role_view(self._quick_setup_role_row(rows, profile_id), selected=True),
            "roles": [
                player_role_view(
                    item,
                    selected=self._row_profile_id(item) == profile_id,
                )
                for item in rows
            ],
            "quick_setup_decided": bool(profile.quick_setup_decided),
            **sections,
            "switches": {
                "enabled": bool(main.get("enabled")),
                "turn_buffer_enabled": bool(main.get("turn_buffer_enabled")),
                "response_polish_enabled": bool(main.get("response_polish_enabled")),
                "image_generation_enabled": bool(main.get("image_generation_enabled")),
            },
        }

    async def _quick_setup_sections(
        self,
        profile_id: str,
        readiness: Mapping[str, bool],
        web: Mapping[str, Any],
    ) -> dict[str, Any]:
        thinking = await self.thinking.quick_setup_snapshot(profile_id)
        stickers = await self.stickers.quick_setup_snapshot(
            profile_id,
            vision_ready=readiness["vision_ready"],
            web_ready=readiness["web_ready"],
            image_ready=readiness["image_ready"],
        )
        character = await self._quick_setup_character_view(profile_id)
        life = await self.background.quick_setup_life_snapshot(profile_id)
        contact = await quick_setup_contact_snapshot(
            self.timeline_repository,
            self.profiles_repository,
            profile_id,
        )
        return {
            "thinking": thinking,
            "web": dict(web),
            "stickers": stickers,
            "character": character,
            "life": life,
            "contact": contact,
        }

    async def _quick_setup_decision(self, payload: dict[str, Any]) -> dict[str, Any]:
        if str(payload.get("decision") or "").strip().lower() != "manual":
            raise ValueError("quick setup decision must be 'manual'")
        rows, profile_id = await self._quick_setup_role(payload)
        if not profile_id:
            raise ValueError("还没有可使用的角色")
        profile = await self.profiles.mark_quick_setup_decided(profile_id)
        await self.profiles_repository.set_console_preference(
            "role_settings.selected_profile_id", profile_id
        )
        role_row = self._quick_setup_role_row(rows, profile_id)
        return {
            "ok": True,
            "status": {
                "role": player_role_view(role_row, selected=True),
                "required": False,
                "quick_setup_decided": True,
                "enabled": bool(profile.enabled),
            },
        }

    async def _quick_setup_configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        rows, profile_id = await self._quick_setup_role(payload)
        if not profile_id:
            raise ValueError("还没有可使用的角色")
        result = await self.ai.handle("ai_quick_setup_configure", profile_id, payload)
        if bool(result.get("ok")) and bool(result.get("applied")):
            await self._apply_quick_setup_model_switch(profile_id, result)
            await self.profiles_repository.set_console_preference(
                "role_settings.selected_profile_id", profile_id
            )
            await self.sticker_repository.wake_waiting_sticker_checks(profile_id)
        snapshot_payload = {
            **payload,
            "role_ref": self._quick_setup_role_ref(rows, profile_id),
        }
        return {**result, "snapshot": await self._quick_setup_snapshot(snapshot_payload)}

    async def _apply_quick_setup_model_switch(
        self, profile_id: str, result: Mapping[str, Any]
    ) -> None:
        slot = str(result.get("slot") or "")
        disabled = str(result.get("action") or "") == "disable"
        if slot == "fast":
            await self.profiles.save_main_config(profile_id, turn_buffer_enabled=True)
        elif slot == "polish":
            await self.profiles.save_main_config(profile_id, response_polish_enabled=not disabled)
        elif slot == "image":
            await self.profiles.save_main_config(profile_id, image_generation_enabled=not disabled)

    async def _quick_setup_character_view(self, profile_id: str) -> dict[str, Any]:
        character_models = getattr(self, "character_models", None)
        if character_models is None:
            character = {"revision": 0, "sections": {}}
            prompt_selections: dict[str, Any] = {}
        else:
            raw_character = await character_models.snapshot(profile_id)
            character = player_character_view(raw_character)
            stored = raw_character.get("character_model")
            model = stored.get("model") if isinstance(stored, Mapping) else None
            selections = model.get("prompt_selections") if isinstance(model, Mapping) else None
            prompt_selections = dict(selections) if isinstance(selections, Mapping) else {}
        importer = getattr(self, "character_import", None)
        source = (
            await importer.source_view(profile_id)
            if importer is not None
            else {"available": False, "source_label": ""}
        )
        return {
            **character,
            "prompt_selections": prompt_selections,
            "astrbot_import": source,
        }

    async def _quick_setup_character_generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        _rows, profile_id = await self._quick_setup_role(payload)
        if not profile_id:
            raise ValueError("还没有可使用的角色")
        importer = getattr(self, "character_import", None)
        if importer is None:
            raise RuntimeError("当前 AstrBot 版本暂不支持读取角色资料")
        setup = await self.ai.handle("ai_quick_setup_snapshot", profile_id, payload)
        slots = setup.get("slots")
        main_slot = slots.get("main") if isinstance(slots, Mapping) else None
        main_model = main_slot.get("model") if isinstance(main_slot, Mapping) else None
        backend_id = (
            str(main_model.get("backend_id") or "") if isinstance(main_model, Mapping) else ""
        )
        character = await self.character_models.snapshot(profile_id)
        current = character.get("character_model")
        revision = int(current.get("revision") or 0) if isinstance(current, Mapping) else 0
        return await importer.generate(
            profile_id,
            backend_id=backend_id,
            request_id=str(payload.get("request_id") or ""),
            revision=revision,
        )

    async def _quick_setup_web_configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        rows, profile_id = await self._quick_setup_role(payload)
        if not profile_id:
            raise ValueError("还没有可使用的角色")
        result = await self.web.handle("quick_setup_configure", profile_id, payload)
        if bool(result.get("ok")) and bool(result.get("applied")):
            await self.profiles_repository.set_console_preference(
                "role_settings.selected_profile_id", profile_id
            )
            await self.sticker_repository.wake_waiting_sticker_checks(profile_id)
        snapshot_payload = {
            **payload,
            "role_ref": self._quick_setup_role_ref(rows, profile_id),
        }
        return {**result, "snapshot": await self._quick_setup_snapshot(snapshot_payload)}

    async def _quick_setup_sticker_configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        rows, profile_id = await self._quick_setup_role(payload)
        if not profile_id:
            raise ValueError("还没有可使用的角色")
        before = await self._quick_setup_snapshot(payload)
        dependencies = _quick_setup_dependencies(
            before,
            "暂时没有读到表情包依赖状态，请重新连接",
        )
        result = await self.stickers.quick_setup_configure(
            profile_id,
            payload,
            vision_ready=bool(dependencies.get("vision_ready")),
            web_ready=bool(dependencies.get("web_ready")),
            image_ready=bool(dependencies.get("image_ready")),
        )
        await self.profiles_repository.set_console_preference(
            "role_settings.selected_profile_id", profile_id
        )
        role_ref = self._quick_setup_role_ref(rows, profile_id)
        return {
            **result,
            "snapshot": await self._quick_setup_snapshot({**payload, "role_ref": role_ref}),
        }

    async def _quick_setup_life_configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        rows, profile_id = await self._quick_setup_role(payload)
        if not profile_id:
            raise ValueError("还没有可使用的角色")
        result = await self.background.quick_setup_life_configure(profile_id, payload)
        await self.profiles_repository.set_console_preference(
            "role_settings.selected_profile_id", profile_id
        )
        role_ref = self._quick_setup_role_ref(rows, profile_id)
        return {
            **result,
            "snapshot": await self._quick_setup_snapshot({**payload, "role_ref": role_ref}),
        }

    async def _quick_setup_contact_configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        rows, profile_id = await self._quick_setup_role(payload)
        if not profile_id:
            raise ValueError("还没有可使用的角色")
        result = await configure_quick_setup_contact(
            self.timeline_repository,
            self.profiles_repository,
            profile_id,
            payload,
        )
        await self.profiles_repository.set_console_preference(
            "role_settings.selected_profile_id", profile_id
        )
        role_ref = self._quick_setup_role_ref(rows, profile_id)
        return {
            **result,
            "snapshot": await self._quick_setup_snapshot({**payload, "role_ref": role_ref}),
        }

    async def _quick_setup_finish(self, payload: dict[str, Any]) -> dict[str, Any]:
        _rows, profile_id = await self._quick_setup_role(payload)
        if not profile_id:
            raise ValueError("还没有可使用的角色")
        setup = await self._quick_setup_snapshot(payload)
        _validate_quick_setup_main(setup)
        dependencies = _quick_setup_dependencies(setup)
        await self.stickers.validate_quick_setup(
            profile_id,
            vision_ready=bool(dependencies.get("vision_ready")),
            web_ready=bool(dependencies.get("web_ready")),
            image_ready=bool(dependencies.get("image_ready")),
        )
        validate_quick_setup_contact(setup.get("contact"))
        _validate_quick_setup_interaction(setup)
        policy = require_thinking_policy(payload.get("thinking_complexity"))
        await self.profiles.finish_quick_setup(
            profile_id,
            thinking_complexity=policy.complexity.value,
        )
        await self.profiles_repository.set_console_preference(
            "role_settings.selected_profile_id", profile_id
        )
        completed = {
            **setup,
            "quick_setup_decided": True,
            "thinking": {
                **dict(setup.get("thinking") or {}),
                "complexity": policy.complexity.value,
            },
            "switches": {**dict(setup.get("switches") or {}), "enabled": True},
        }
        return {"ok": True, "snapshot": completed}

    async def _quick_setup_role(
        self, payload: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], str]:
        """Resolve the explicit setup target without falling back to another role."""

        rows, profile_id = await self._player_roles(payload)
        requested_ref = str(payload.get("role_ref") or "").strip()
        if requested_ref and (not profile_id or player_role_ref(profile_id) != requested_ref):
            raise ValueError("要配置的角色已经变化，请重新选择")
        return rows, profile_id

    def _quick_setup_role_row(self, rows: list[dict[str, Any]], profile_id: str) -> dict[str, Any]:
        return next(
            (item for item in rows if self._row_profile_id(item) == profile_id),
            {"profile_id": profile_id},
        )

    def _quick_setup_role_ref(self, rows: list[dict[str, Any]], profile_id: str) -> str:
        return str(
            player_role_view(
                self._quick_setup_role_row(rows, profile_id),
                selected=True,
            )["role_ref"]
        )


def _quick_setup_readiness(
    snapshot: Mapping[str, Any], main: Mapping[str, Any], web: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        "vision_ready": _quick_setup_slot_ready(snapshot, "vision"),
        "image_ready": _quick_setup_slot_ready(snapshot, "image")
        and bool(main.get("image_generation_enabled")),
        "web_ready": bool(web.get("active")),
    }


def _quick_setup_slot_ready(snapshot: Mapping[str, Any], name: str) -> bool:
    slots = snapshot.get("slots")
    slot = slots.get(name) if isinstance(slots, Mapping) else None
    model = slot.get("model") if isinstance(slot, Mapping) else None
    return bool(
        isinstance(slot, Mapping)
        and slot.get("configured")
        and isinstance(model, Mapping)
        and model.get("enabled", True)
    )


def _quick_setup_dependencies(
    setup: Mapping[str, Any],
    error_message: str = "暂时没有读到表情包设置，请重新连接后再完成",
) -> Mapping[str, Any]:
    sticker_state = setup.get("stickers")
    dependencies = sticker_state.get("dependencies") if isinstance(sticker_state, Mapping) else None
    if not isinstance(dependencies, Mapping):
        raise ValueError(error_message)
    return dependencies


def _validate_quick_setup_main(setup: Mapping[str, Any]) -> None:
    slots = setup.get("slots")
    main_slot = slots.get("main") if isinstance(slots, Mapping) else None
    main_model = main_slot.get("model") if isinstance(main_slot, Mapping) else None
    if (
        not isinstance(main_slot, Mapping)
        or not bool(main_slot.get("configured"))
        or not isinstance(main_model, Mapping)
        or not bool(main_model.get("enabled", True))
    ):
        raise ValueError("请先让主力模型通过真实连接测试，再完成快速设置")


def _validate_quick_setup_interaction(setup: Mapping[str, Any]) -> None:
    styles, modes, polish, background = _quick_setup_prompt_selections(setup)
    _validate_quick_setup_styles(styles)
    if str(modes.get("self_initiated") or "") != "cater_to_interests":
        raise ValueError("角色的主动联系内容尚未保存，请重新确认相处方式")
    if str(polish.get("writing_correction") or "") != "remove_ai_formula":
        raise ValueError("角色的自然表达设置尚未保存，请重新确认相处方式")
    if str(background.get("story_boundary") or "") != "follow_original":
        raise ValueError("角色的故事边界尚未保存，请重新确认相处方式")
    _validate_quick_setup_sticker_style(setup, styles)


def _quick_setup_prompt_selections(
    setup: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    character = setup.get("character")
    selections = character.get("prompt_selections") if isinstance(character, Mapping) else None
    if not isinstance(selections, Mapping):
        raise ValueError("请先完成角色的相处方式设置")
    values = tuple(
        selections.get(name)
        for name in (
            "main_core_styles",
            "main_core_modes",
            "response_polish",
            "background_creation",
        )
    )
    if not all(isinstance(value, Mapping) for value in values):
        raise ValueError("请先完成角色的相处方式设置")
    styles, modes, polish, background = values
    assert isinstance(styles, Mapping)
    assert isinstance(modes, Mapping)
    assert isinstance(polish, Mapping)
    assert isinstance(background, Mapping)
    return styles, modes, polish, background


def _validate_quick_setup_styles(styles: Mapping[str, Any]) -> None:
    if str(styles.get("relationship_context") or "") not in _RELATIONSHIP_CONTEXTS:
        raise ValueError("请先确认你和角色生活在哪里")
    if str(styles.get("thinking_style") or "") not in _THINKING_STYLES:
        raise ValueError("请先确认你和角色一开始怎样相处")
    required = {
        "speaking_style": "natural_chat",
        "content_style": "background_as_subtext",
    }
    if any(str(styles.get(name) or "") != value for name, value in required.items()):
        raise ValueError("角色的自然聊天设置尚未保存，请重新确认相处方式")


def _validate_quick_setup_sticker_style(
    setup: Mapping[str, Any], styles: Mapping[str, Any]
) -> None:
    stickers = setup.get("stickers")
    private_stickers = stickers.get("private") if isinstance(stickers, Mapping) else None
    stickers_enabled = (
        bool(private_stickers.get("enabled")) if isinstance(private_stickers, Mapping) else False
    )
    if stickers_enabled and str(styles.get("sticker_style") or "") not in _STICKER_STYLES:
        raise ValueError("请先选择角色使用表情包的方式")


__all__ = ["QuickSetupPageMixin"]
