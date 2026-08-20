"""Stable Page action routing over domain-oriented administrator controllers."""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from datetime import datetime
from typing import Any, Protocol

from ...features.character_model.importing import PUBLIC_CHARACTER_FIELDS
from ...features.profiles.ports import ProfilesRepositoryPort
from ...features.stickers.ports import StickerRepositoryPort
from ...features.timeline.ports import TimelineRepositoryPort
from . import identity_confirmations as identity_ui
from .console_errors import require_successful_settings_result
from .console_views import (
    action_result_view,
    context_budget_view,
    feature_status_view,
    file_library_view,
    image_library_view,
    instance_workspace_view,
    knowledge_workspace_view,
    readiness_view,
    recall_lab_view,
    sticker_library_view,
    web_library_view,
)
from .controllers.character_models import CharacterModelsAdminController
from .controllers.diagnostics import DiagnosticsAdminController
from .controllers.knowledge import KnowledgeAdminController
from .controllers.media import MediaAdminController
from .controllers.operations import RuntimeOperationsController
from .controllers.player_profiles import PlayerProfilesAdminController
from .controllers.profile_settings import ProfileSettingsController
from .controllers.profiles import ProfilesAdminController
from .controllers.role_packages import RolePackageController
from .controllers.sticker_references import StickerReferenceController
from .controllers.stickers import StickersAdminController
from .controllers.thinking import ThinkingSettingsController
from .controllers.timeline import TimelineAdminController
from .controllers.timer_lifecycle import timer_lifecycle_snapshot
from .controllers.web import WebAdminController
from .delivery_attention import (
    acknowledge_delivery_failure,
    delivery_failure_preference_key,
    parse_delivery_failure_acknowledgements,
)
from .downloads import PageFileDownload
from .main_config_actions import handle_main_config_action
from .page_player_actions import PlayerPageActionsMixin
from .player_views import player_role_ref
from .presentation import jsonable
from .quick_setup_page import QuickSetupPageMixin
from .router import AdminActionRouter

_PROFILE_SETTINGS_SECTIONS = {
    "character",
    "player_character",
    "world",
    "world.lore",
    "world.boundary",
    "conversation",
}


class IdentityPageActionsMixin:
    async def _identity_annotations(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile_id = await self._profile_id(payload)
        action = str(payload.get("action") or "preview").strip().lower()
        section = str(payload.get("section") or "").strip().lower()
        confirmation_scope = identity_ui.settings_identity_scope(section, payload.get("scope"))
        if action in {"preview_existing", "apply_existing"}:
            return await self._existing_identity_annotations(action, profile_id, section, payload)
        value = payload.get("value")
        if not isinstance(value, (Mapping, list)):
            raise ValueError("value must be an object or list")
        context = await self._identity_annotation_context(profile_id, section, payload)
        candidates = identity_ui.candidates_for_payload(self.identity, value, context, payload)
        if action == "apply":
            return self._apply_identity_annotations(
                profile_id, confirmation_scope, section, value, candidates, payload
            )
        if action != "preview":
            raise ValueError("unsupported identity annotation action")
        return self._preview_identity_annotations(
            profile_id, confirmation_scope, section, value, candidates
        )

    async def _existing_identity_annotations(
        self,
        action: str,
        profile_id: str,
        section: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        worlds = self.background if section == "world" else None
        return await identity_ui.existing_settings_action(
            action,
            profile_id,
            section,
            payload,
            identity=self.identity,
            character_models=self.character_models,
            worlds=worlds,
            confirmations=self.identity_confirmations,
            save_settings=self._save_settings_value,
        )

    async def _identity_annotation_context(
        self, profile_id: str, section: str, payload: dict[str, Any]
    ) -> Any:
        instance_id = str(payload.get("instance_id") or "").strip()
        if not instance_id and str(payload.get("contact_ref") or "").strip():
            contact = await self._player_contact(profile_id, payload)
            instance_id = str(contact["instance_id"])
        if instance_id and section not in _PROFILE_SETTINGS_SECTIONS:
            return await self.identity.context(profile_id, instance_id)
        context = await self.identity.profile_context(profile_id)
        if section != "player_character":
            return context
        value = payload.get("value")
        patch = value.get("patch") if isinstance(value, Mapping) else None
        identity = patch.get("identity") if isinstance(patch, Mapping) else None
        proposed_name = (
            str(identity.get("name") or "").strip() if isinstance(identity, Mapping) else ""
        )
        return replace(context, character_name=proposed_name) if proposed_name else context

    def _apply_identity_annotations(
        self,
        profile_id: str,
        confirmation_scope: str,
        section: str,
        value: Any,
        candidates: Any,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        selected = {str(item) for item in payload.get("selected_ids", ()) or ()}
        transformed = self.identity.apply_annotations(value, candidates, selected)
        return {
            "ok": True,
            "value": transformed,
            "confirmation_token": self.identity_confirmations.grant(
                profile_id,
                confirmation_scope,
                section,
                transformed,
            ),
        }

    def _preview_identity_annotations(
        self,
        profile_id: str,
        confirmation_scope: str,
        section: str,
        value: Any,
        candidates: Any,
    ) -> dict[str, Any]:
        views = identity_ui.candidate_views(candidates)
        result: dict[str, Any] = {"ok": True, "candidates": views}
        if not views:
            result["confirmation_token"] = self.identity_confirmations.grant(
                profile_id,
                confirmation_scope,
                section,
                value,
            )
        return result


class LibraryPageActionsMixin:
    """Keep library-specific routing out of the root Page controller."""

    async def download(self, method: str, payload: Mapping[str, Any]) -> PageFileDownload:
        self.require_ready()
        if method == "role_package_download":
            role_ref, profile_id = await self._role_package_owner(payload)
            if self.role_packages is None:
                raise RuntimeError("角色包服务尚未就绪")
            return await self.role_packages.download(
                role_ref=role_ref,
                profile_id=profile_id,
                download_token=str(payload.get("download_token") or ""),
            )
        if method != "image_download":
            raise ValueError("unsupported page download")
        profile_id, scope = await self._scope(payload)
        instance = await self._instance(profile_id, scope, payload)
        return await self.media.image_download(
            profile_id,
            str(instance["instance_id"]),
            str(payload.get("asset_id") or ""),
        )

    async def _media(self, action: str, payload: dict[str, Any]) -> Any:
        profile_id, scope = await self._scope(payload)
        instance = await self._instance(profile_id, scope, payload)
        instance_id = instance["instance_id"]
        if action == "image_snapshot":
            return image_library_view(await self.media.image_snapshot(profile_id, instance_id))
        if action == "image_preview":
            return await self.media.image_preview(
                profile_id, instance_id, str(payload.get("asset_id") or "")
            )
        if action == "file_artifacts":
            return file_library_view(
                await self.media.file_artifact_snapshot(profile_id, instance_id)
            )
        await self.media.file_artifact_admin_action(profile_id, instance_id, payload)
        return action_result_view("文件操作已经完成")

    async def _knowledge(self, action: str, payload: dict[str, Any]) -> Any:
        profile_id, scope = await self._scope(payload)
        instance = await self._instance(profile_id, scope, payload)
        instance_id = instance["instance_id"]
        if action == "knowledge_snapshot":
            return knowledge_workspace_view(
                await self.knowledge.knowledge_snapshot(profile_id, instance_id)
            )
        if action == "knowledge_form":
            result = await self.knowledge.knowledge_form(
                profile_id, instance_id, str(payload.get("mode") or "dry").strip().lower()
            )
            if result.get("ok") is False:
                raise ValueError(str(result.get("reason") or "当前对话还不需要整理"))
            return action_result_view("知识整理已经开始")
        if action.startswith("recall_"):
            return await self._recall_admin_action(action, payload, profile_id, instance_id)
        if action == "knowledge_record":
            result = await self._update_knowledge_record(payload, profile_id, scope, instance)
            if result.get("ok") is False:
                raise ValueError("knowledge record changed concurrently; reload before saving")
            return action_result_view("知识内容已经更新")
        raise ValueError("unsupported knowledge action")

    async def _recall_admin_action(
        self,
        action: str,
        payload: dict[str, Any],
        profile_id: str,
        instance_id: str,
    ) -> Any:
        if action == "recall_probe":
            return recall_lab_view(
                await self.knowledge.recall_probe(
                    profile_id, instance_id, str(payload.get("query") or "")
                )
            )
        if action == "recall_configuration":
            return await self.knowledge.recall_configuration(profile_id, instance_id)
        if action == "recall_configuration_update":
            return await self.knowledge.recall_configuration_update(
                profile_id, instance_id, payload
            )
        if action == "recall_rebuild":
            return await self.knowledge.recall_rebuild(profile_id, instance_id)
        if action == "recall_integrity":
            return await self.knowledge.recall_integrity(profile_id, instance_id)
        if action == "recall_benchmark":
            return await self.knowledge.recall_benchmark(profile_id, instance_id)
        raise ValueError("unsupported knowledge action")

    async def _update_knowledge_record(
        self,
        payload: dict[str, Any],
        profile_id: str,
        scope: str,
        instance: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if str(payload.get("action") or "").lower() in {"create", "update"}:
            self.identity_confirmations.require(
                profile_id,
                str(instance.get("scope") or scope),
                "knowledge_record",
                payload.get("record"),
                str(payload.get("identity_confirmation_token") or ""),
            )
        return await self.knowledge.knowledge_record_action(
            profile_id, str(instance["instance_id"]), payload
        )

    async def _player_profile(self, action: str, payload: dict[str, Any]) -> Any:
        if self.player_profiles is None:
            raise RuntimeError("player profile administration is unavailable")
        profile_id, scope = await self._scope(payload)
        instance = await self._instance(profile_id, scope, payload)
        instance_id = str(instance["instance_id"])
        if action == "player_profile_snapshot":
            return await self.player_profiles.snapshot(profile_id, instance_id, scope, payload)
        if action == "player_profile_entry":
            return await self.player_profiles.entry_detail(profile_id, instance_id, scope, payload)
        operation = str(payload.get("action") or "").strip().lower()
        if operation in {"create", "update"}:
            semantic_value = (
                payload.get("record") if operation == "create" else payload.get("patch")
            )
            self.identity_confirmations.require(
                profile_id,
                scope,
                "player_profile_entry",
                semantic_value,
                str(payload.get("identity_confirmation_token") or ""),
            )
        return await self.player_profiles.action(profile_id, instance_id, scope, payload)


SettingsSaver = Callable[[], Awaitable[dict[str, Any]]]
WORLD_DEFINITION_FIELDS = (
    "world_brief",
    "world_rules",
    "life_direction",
    "world_texture",
    "expansion_policy",
)


class SettingsPageActionsMixin:
    """Keep settings snapshot and save orchestration out of the root page router."""

    async def _settings_snapshot_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile_id, scope = await self._scope(payload)
        return await self._settings_snapshot(profile_id, scope)

    async def _settings_snapshot(self, profile_id: str, scope: str) -> dict[str, Any]:
        main = await self.profiles.main_config_snapshot(profile_id)
        background_life = await self.background.quick_setup_life_snapshot(profile_id)
        character = await self.character_models.snapshot(profile_id)
        ai_packages = await self.ai.handle("ai_api_packages", profile_id, {})
        scope_config = await self.profiles.scope_config_snapshot(profile_id, scope)
        thinking = await self.thinking.snapshot(profile_id, scope_config, ai_packages)
        web_config = await self.web.handle("get_web_config", profile_id, {"scope": scope})
        web_providers = await self.web.handle("web_providers", profile_id, {"scope": scope})
        stickers = await self.stickers.sticker_config_snapshot(profile_id, scope)
        identity_reference = await self.sticker_references.sticker_reference_snapshot(
            profile_id, scope
        )
        world = await self.background.world_snapshot(profile_id)
        sections = {
            "main": main,
            "background_life": background_life,
            "character": character,
            "world": world,
            "models": ai_packages,
            "thinking": thinking,
            "conversation": scope_config,
            "capabilities": {
                "web": web_config,
                "web_providers": web_providers,
                "stickers": stickers,
                "identity_reference": identity_reference,
                "image_generation_enabled": bool(main.get("image_generation_enabled")),
                "file_artifacts_enabled": bool(main.get("file_artifacts_enabled")),
            },
        }
        return {
            "profile_id": profile_id,
            "scope": scope,
            "readiness": readiness_view(main, character, ai_packages, thinking),
            "sections": sections,
            "identity_confirmation_tokens": identity_ui.initial_settings_grants(
                self.identity_confirmations,
                profile_id,
                scope,
                {
                    "main": main,
                    "character": character,
                    "world": world,
                    "conversation": scope_config,
                },
            ),
        }

    async def _settings_section(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile_id, scope = await self._scope(payload)
        section = str(payload.get("section") or "").strip().lower()
        value = payload.get("value")
        if not isinstance(value, Mapping):
            raise ValueError("value must be an object")
        confirmation_scope = identity_ui.settings_identity_scope(section, scope)
        require_confirmation = (
            self.identity_confirmations.require_exact
            if section == "world"
            else self.identity_confirmations.require
        )
        require_confirmation(
            profile_id,
            confirmation_scope,
            section,
            value,
            str(payload.get("identity_confirmation_token") or ""),
        )
        merged = {
            **dict(value),
            "profile_id": profile_id,
            "scope": scope,
        }
        for name in ("instance_id", "expected_version", "idempotency_key"):
            if name in payload and name not in merged:
                merged[name] = payload[name]
        if "instance_id" not in merged and str(payload.get("contact_ref") or "").strip():
            merged["instance_id"] = str(
                (await self._player_contact(profile_id, payload))["instance_id"]
            )
        result = await self._save_settings_value(section, merged)
        require_successful_settings_result(result)
        readiness = None
        if section in {"main", "character", "thinking"}:
            try:
                readiness = (await self._settings_snapshot(profile_id, scope))["readiness"]
            except Exception:
                if section != "thinking":
                    raise
        return {
            "ok": True,
            "section": section,
            "saved_at": datetime.now().astimezone().isoformat(),
            "result": jsonable(result),
            "readiness": readiness,
            "identity_confirmation_token": self.identity_confirmations.grant(
                profile_id,
                confirmation_scope,
                section,
                value,
            ),
        }

    async def _save_settings_value(self, section: str, merged: dict[str, Any]) -> dict[str, Any]:
        handlers: dict[str, SettingsSaver] = {
            "main": lambda: self._main_config("save_main_config", merged),
            "character": lambda: self._character_model("save_character_model", merged),
            "player_character": lambda: self._player_character_settings(merged),
            "world": lambda: self._world_definition_settings(merged),
            "conversation": lambda: self._profile_action("save_scope_config", merged),
            "thinking": lambda: self.thinking.save(merged),
            "web": lambda: self._web("save_web_config", merged),
            "stickers": lambda: self._sticker("save_sticker_config", merged),
            "instance_chat": lambda: self._profile_action("save_instance_chat_policy", merged),
            "instance_contact": lambda: self._profile_action(
                "save_instance_contact_override", merged
            ),
            "platform_contact": lambda: self._profile_action(
                "save_platform_contact_policy", merged
            ),
        }
        handler = handlers.get(section)
        if handler is None:
            raise ValueError("unknown settings section")
        return await handler()

    async def _player_character_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Merge the bounded player editor patch without exposing advanced fields."""

        patch = payload.get("patch")
        if not isinstance(patch, Mapping) or not patch:
            raise ValueError("player character patch must be a non-empty object")
        allowed = {section: set(fields) for section, fields in PUBLIC_CHARACTER_FIELDS.items()}
        current = await self.character_models.snapshot(str(payload["profile_id"]))
        character = current.get("character_model") or {}
        model = copy.deepcopy(character.get("model") or {})
        for section, raw_fields in patch.items():
            if section not in allowed or not isinstance(raw_fields, Mapping):
                raise ValueError("player character patch contains an unsupported section")
            unknown = set(raw_fields) - allowed[section]
            if unknown:
                raise ValueError("player character patch contains unsupported fields")
            target = model.setdefault(section, {})
            if not isinstance(target, dict):
                raise ValueError("stored character section is unavailable")
            target.update(copy.deepcopy(dict(raw_fields)))
        return await self.character_models.save(
            str(payload["profile_id"]),
            {
                "expected_revision": int(payload.get("expected_revision") or 0),
                "idempotency_key": str(payload.get("idempotency_key") or ""),
                "model": model,
            },
        )

    async def _world_definition_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        missing = [name for name in WORLD_DEFINITION_FIELDS if name not in payload]
        if missing:
            raise ValueError("world settings require all definition fields")
        return await self.background.world_action(
            str(payload["profile_id"]),
            {
                "action": "save_definition",
                "expected_revision": payload.get("expected_revision"),
                "value": {name: payload[name] for name in WORLD_DEFINITION_FIELDS},
            },
        )


AI_ACTIONS = {
    "ai_work_records",
    "ai_work_record",
    "ai_work_attempt_debug",
    "ai_work_attempt_raw",
    "ai_backend_probe",
    "ai_capability_pool",
    "ai_api_packages",
    "ai_api_package",
    "ai_api_package_credential",
    "ai_api_model",
    "ai_api_package_probe",
    "ai_api_model_probe",
}
WEB_ACTIONS = {
    "get_web_config",
    "save_web_config",
    "web_providers",
    "web_provider",
    "web_provider_credential",
    "web_provider_probe",
    "web_image_probe",
    "web_search_test",
    "web_read_test",
    "web_snapshot",
}
STICKER_ACTIONS = {
    "get_sticker_config",
    "save_sticker_config",
    "sticker_references",
    "sticker_reference_action",
    "sticker_snapshot",
    "sticker_action",
    "sticker_run",
    "sticker_stop",
    "sticker_clear",
    "sticker_intake",
    "sticker_intake_start",
    "sticker_intake_upload",
    "sticker_intake_action",
    "sticker_intake_preview",
}
DIAGNOSTIC_ACTIONS = {"support_bundle"}
PROFILE_ACTIONS = {
    "get_scope_config",
    "save_scope_config",
    "instances",
    "instance_detail",
    "get_instance_contact_override",
    "save_instance_chat_policy",
    "save_instance_contact_override",
    "get_platform_contact_policy",
    "save_platform_contact_policy",
}
BRIDGE_ACTIONS = {
    "controlled_bridge_snapshot",
    "character_intent_detail",
    "character_intent_action",
}
DELIVERY_ACTIONS = {"delivery_failure_acknowledge"}
CONTEXT_ACTIONS = {"context_snapshot", "context_summary", "context_dry_run"}
MEDIA_ACTIONS = {
    "image_snapshot",
    "image_preview",
    "file_artifacts",
    "file_artifact_action",
}
KNOWLEDGE_ACTIONS = {
    "knowledge_snapshot",
    "knowledge_form",
    "recall_probe",
    "recall_configuration",
    "recall_configuration_update",
    "recall_rebuild",
    "recall_integrity",
    "recall_benchmark",
    "knowledge_record",
}
PLAYER_PROFILE_ACTIONS = {
    "player_profile_snapshot",
    "player_profile_entry",
    "player_profile_action",
}
MAIN_CONFIG_ACTIONS = {"get_main_config", "save_main_config"}
CHARACTER_MODEL_ACTIONS = {"get_character_model", "save_character_model"}


class BackgroundPageActionsMixin:
    background: Any
    identity_confirmations: Any
    _scope: Callable[[Mapping[str, Any]], Awaitable[tuple[str, str]]]
    _instance: Callable[[str, str, Mapping[str, Any]], Awaitable[dict[str, Any]]]

    def _background_handlers(self) -> dict[str, Any]:
        return {
            "background_workspace": self._background_workspace,
            "background_action": self._background_action,
        }

    async def _background_scope(self, payload: dict[str, Any]) -> tuple[str, str, str]:
        profile_id, scope = await self._scope(payload)
        instance = await self._instance(profile_id, scope, payload)
        return profile_id, scope, str(instance["instance_id"])

    async def _background_workspace(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile_id, _, instance_id = await self._background_scope(payload)
        return await self.background.workspace(profile_id, instance_id)

    async def _background_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action") or "").strip().lower()
        if action.startswith("seed_"):
            profile_id, _ = await self._scope(payload)
            self._require_seed_identity_confirmation(profile_id, action, payload)
            return await self.background.action(profile_id, "", payload)
        profile_id, _, instance_id = await self._background_scope(payload)
        return await self.background.action(profile_id, instance_id, payload)

    def _require_seed_identity_confirmation(
        self,
        profile_id: str,
        action: str,
        payload: Mapping[str, Any],
    ) -> None:
        section = {
            "seed_lore_create": "world.lore",
            "seed_lore_update": "world.lore",
            "seed_boundary_create": "world.boundary",
            "seed_boundary_update": "world.boundary",
        }.get(action)
        if section is None:
            return
        self.identity_confirmations.require_exact(
            profile_id,
            "profile",
            section,
            payload.get("value"),
            str(payload.get("identity_confirmation_token") or ""),
        )


class AIAdminControllerPort(Protocol):
    async def handle(
        self, method: str, profile_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]: ...


ReadyCheck = Callable[[], None]


def _console_profile_rows(values: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in values:
        item = jsonable(value)
        if not isinstance(item, dict):
            continue
        row = dict(item)
        profile_id = str(row.get("profile_id") or row.get("id") or "")
        row["role_ref"] = player_role_ref(profile_id)
        rows.append(row)
    return rows


def _selected_console_profile(
    rows: list[dict[str, Any]], payload: Mapping[str, Any], preferred: Any
) -> str:
    known_ids = [str(item.get("profile_id") or item.get("id") or "") for item in rows]
    requested = str(payload.get("profile_id") or "").strip()
    selected = requested if requested in known_ids else str(preferred or "")
    if selected in known_ids:
        return selected
    return next((item for item in known_ids if item), "")


def _console_scope(payload: Mapping[str, Any]) -> str:
    return (
        "group" if str(payload.get("scope") or "private").strip().lower() == "group" else "private"
    )


class AdminPageController(
    QuickSetupPageMixin,
    PlayerPageActionsMixin,
    IdentityPageActionsMixin,
    SettingsPageActionsMixin,
    BackgroundPageActionsMixin,
    LibraryPageActionsMixin,
):
    def __init__(
        self,
        *,
        profiles_repository: ProfilesRepositoryPort,
        timeline_repository: TimelineRepositoryPort,
        sticker_repository: StickerRepositoryPort,
        timer_repository: Any | None = None,
        model_gateway: Any | None = None,
        require_ready: ReadyCheck,
        profiles: ProfilesAdminController,
        character_models: CharacterModelsAdminController,
        profile_settings: ProfileSettingsController,
        ai: AIAdminControllerPort,
        timeline: TimelineAdminController,
        knowledge: KnowledgeAdminController,
        media: MediaAdminController,
        web: WebAdminController,
        stickers: StickersAdminController,
        sticker_references: StickerReferenceController,
        diagnostics: DiagnosticsAdminController,
        operations: RuntimeOperationsController,
        identity: Any,
        thinking: ThinkingSettingsController,
        background: Any,
        player_profiles: PlayerProfilesAdminController | None = None,
        character_import: Any | None = None,
        role_packages: RolePackageController | None = None,
    ) -> None:
        self.profiles_repository = profiles_repository
        self.timeline_repository = timeline_repository
        self.sticker_repository = sticker_repository
        self.timer_repository = timer_repository
        self.model_gateway = model_gateway
        self.require_ready = require_ready
        self.profiles = profiles
        self.character_models = character_models
        self.profile_settings = profile_settings
        self.ai = ai
        self.timeline = timeline
        self.knowledge = knowledge
        self.media = media
        self.web = web
        self.stickers = stickers
        self.sticker_references = sticker_references
        self.diagnostics = diagnostics
        self.operations = operations
        self.identity = identity
        self.player_profiles = player_profiles
        self.thinking = thinking
        self.background = background
        self.character_import = character_import
        self.role_packages = role_packages
        self.identity_confirmations = identity_ui.IdentityConfirmationGrants()
        self.router = AdminActionRouter(self._handlers())

    async def call(self, method: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.require_ready()
        return dict(await self.router.dispatch(method, dict(payload)))

    def _handlers(self) -> dict[str, Any]:
        handlers: dict[str, Any] = {
            "profiles": self._profiles,
            "console_bootstrap": self._console_bootstrap,
            "player_bootstrap": self._player_bootstrap,
            "player_guide_acknowledge": self._player_guide_acknowledge,
            "advanced_guide_acknowledge": self._advanced_guide_acknowledge,
            "player_now": self._player_now,
            "player_contacts": self._player_contacts,
            "player_relationship": self._player_relationship,
            "player_about": self._player_about,
            "release_notes": self._release_notes,
            "quick_setup_status": self._quick_setup_status,
            "quick_setup_snapshot": self._quick_setup_snapshot,
            "quick_setup_configure": self._quick_setup_configure,
            "quick_setup_web_configure": self._quick_setup_web_configure,
            "quick_setup_sticker_configure": self._quick_setup_sticker_configure,
            "quick_setup_life_configure": self._quick_setup_life_configure,
            "quick_setup_contact_configure": self._quick_setup_contact_configure,
            "quick_setup_character_generate": self._quick_setup_character_generate,
            "quick_setup_decision": self._quick_setup_decision,
            "quick_setup_finish": self._quick_setup_finish,
            "settings_snapshot": self._settings_snapshot_action,
            "settings_section": self._settings_section,
            "instance_workspace": self._instance_workspace,
            "save_console_profile": self._save_console_profile,
            "reset_instance": self._reset_instance,
            "identity_annotations": self._identity_annotations,
            "timer_lifecycle_snapshot": self._timer_lifecycle_snapshot,
            "role_package_export_prepare": self._role_package_export_prepare,
            "role_package_import_upload": self._role_package_import_upload,
            "role_package_import_apply": self._role_package_import_apply,
        }
        handlers.update(self._background_handlers())
        self._bind(handlers, MAIN_CONFIG_ACTIONS, self._main_config)
        self._bind(handlers, CHARACTER_MODEL_ACTIONS, self._character_model)
        self._bind(handlers, AI_ACTIONS, self._ai)
        self._bind(handlers, WEB_ACTIONS, self._web)
        self._bind(handlers, STICKER_ACTIONS, self._sticker)
        self._bind(handlers, DIAGNOSTIC_ACTIONS, self._diagnostic)
        self._bind(handlers, PROFILE_ACTIONS, self._profile_action)
        self._bind(handlers, BRIDGE_ACTIONS, self._bridge)
        self._bind(handlers, DELIVERY_ACTIONS, self._delivery)
        self._bind(handlers, CONTEXT_ACTIONS, self._context)
        self._bind(handlers, MEDIA_ACTIONS, self._media)
        self._bind(handlers, KNOWLEDGE_ACTIONS, self._knowledge)
        self._bind(handlers, PLAYER_PROFILE_ACTIONS, self._player_profile)
        return handlers

    async def _role_package_export_prepare(self, payload: dict[str, Any]) -> dict[str, Any]:
        role_ref, profile_id = await self._role_package_owner(payload)
        if self.role_packages is None:
            raise RuntimeError("角色包服务尚未就绪")
        return await self.role_packages.export_prepare(
            role_ref=role_ref,
            profile_id=profile_id,
        )

    async def _role_package_import_upload(self, payload: dict[str, Any]) -> dict[str, Any]:
        role_ref, profile_id = await self._role_package_owner(payload)
        if self.role_packages is None:
            raise RuntimeError("角色包服务尚未就绪")
        return await self.role_packages.upload(
            role_ref=role_ref,
            profile_id=profile_id,
            payload=payload,
        )

    async def _role_package_import_apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        role_ref, profile_id = await self._role_package_owner(payload)
        if self.role_packages is None:
            raise RuntimeError("角色包服务尚未就绪")
        result = await self.role_packages.apply(
            role_ref=role_ref,
            profile_id=profile_id,
            confirmation_token=str(payload.get("confirmation_token") or ""),
            idempotency_key=str(payload.get("idempotency_key") or ""),
        )
        # Import has already committed at this point. Model readiness is only a
        # follow-up hint, so a transient provider/status failure must not turn a
        # successful atomic import into an apparent failed request.
        try:
            setup = await self.ai.handle("ai_quick_setup_snapshot", profile_id, payload)
            slots = setup.get("slots") if isinstance(setup, Mapping) else None
            main = slots.get("main") if isinstance(slots, Mapping) else None
            result["model_setup_required"] = not (
                isinstance(main, Mapping) and bool(main.get("configured"))
            )
        except Exception:
            result["model_setup_required"] = True
        return result

    async def _role_package_owner(self, payload: Mapping[str, Any]) -> tuple[str, str]:
        if str(payload.get("profile_id") or "").strip():
            raise ValueError("角色包页面只接受不可逆 role_ref")
        role_ref = str(payload.get("role_ref") or "").strip()
        if not role_ref:
            raise ValueError("role_ref is required")
        rows, _selected = await self._player_roles({})
        for row in rows:
            profile_id = self._row_profile_id(row)
            if profile_id and player_role_ref(profile_id) == role_ref:
                return role_ref, profile_id
        raise ValueError("当前角色已经变化，请刷新页面后重试")

    async def _timer_lifecycle_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.timer_repository is None:
            raise RuntimeError("Timer lifecycle administration is unavailable")
        profile_id, scope = await self._scope(payload)
        instance = await self._instance(profile_id, scope, payload)
        return await timer_lifecycle_snapshot(
            self.timer_repository,
            self.model_gateway,
            profile_id=profile_id,
            instance_id=str(instance["instance_id"]),
        )

    @staticmethod
    def _bind(handlers: dict[str, Any], names: set[str], callback: Any) -> None:
        for name in names:

            async def handler(payload: dict[str, Any], action: str = name) -> Any:
                return await callback(action, payload)

            handlers[name] = handler

    async def _profiles(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        profiles = await self.profiles.sync_profiles()
        selected = await self.profiles_repository.get_console_preference(
            "role_settings.selected_profile_id"
        )
        return {
            "profiles": [jsonable(item) for item in profiles],
            "selected_profile_id": selected,
        }

    async def _save_console_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile_id = str(payload.get("profile_id") or "").strip()
        if not profile_id:
            raise ValueError("profile_id is required")
        await self.profiles.require_known_profile(profile_id)
        await self.profiles_repository.set_console_preference(
            "role_settings.selected_profile_id", profile_id
        )
        return {"ok": True, "selected_profile_id": profile_id}

    async def _console_bootstrap(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Load the bounded information required to mount the new console shell."""

        advanced_guide = await self._advanced_guide_state()
        profile_rows = _console_profile_rows(await self.profiles.sync_profiles())
        preferred = await self.profiles_repository.get_console_preference(
            "role_settings.selected_profile_id"
        )
        selected = (
            await self._player_profile_id(payload)
            if str(payload.get("role_ref") or "").strip()
            else _selected_console_profile(profile_rows, payload, preferred)
        )
        scope = _console_scope(payload)
        if not selected:
            return {
                "profiles": profile_rows,
                "selected_profile_id": "",
                "selected_role_ref": "",
                "scope": scope,
                "instances": [],
                "advanced_guide": advanced_guide,
                "readiness": {
                    "ready": False,
                    "status": "blocked",
                    "title": "还没有可管理的 AstrBot 配置档案",
                    "summary": "请先在 AstrBot 中创建并启用一个配置档案。",
                    "checks": [],
                    "issues": [],
                },
            }
        settings = await self._settings_snapshot(selected, scope)
        instances = await self.profiles.role_instances_snapshot(selected)
        return {
            "profiles": profile_rows,
            "selected_profile_id": selected,
            "selected_role_ref": player_role_ref(selected),
            "scope": scope,
            "instances": instances["sections"][scope],
            "advanced_guide": advanced_guide,
            "readiness": settings["readiness"],
        }

    async def _instance_workspace(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile_id, scope = await self._scope(payload)
        instance = await self._instance(profile_id, scope, payload)
        instance_id = instance["instance_id"]
        detail = await self._instance_detail(profile_id, instance_id, payload)
        identity_context = await self.identity.context(profile_id, instance_id)
        rendered_detail = self.identity.render_data(detail, identity_context)
        acknowledged_delivery_failures = parse_delivery_failure_acknowledgements(
            await self.profiles_repository.get_console_preference(
                delivery_failure_preference_key(profile_id, instance_id)
            )
        )
        context_budget = context_budget_view(
            await self.timeline.context_snapshot(profile_id, scope, instance_id)
        )
        return {
            "profile_id": profile_id,
            "scope": scope,
            "instance": {
                "instance_id": instance["instance_id"],
                "display_name": instance.get("display_name") or "当前对象",
                "scope": instance.get("scope"),
            },
            "workspace": instance_workspace_view(
                {
                    **rendered_detail,
                    "profile": {**rendered_detail.get("profile", {}), **instance},
                    "context_budget": context_budget,
                },
                acknowledged_delivery_failures=acknowledged_delivery_failures,
            ),
        }

    async def _delivery(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        if action != "delivery_failure_acknowledge":
            raise ValueError("unknown delivery action")
        profile_id, scope = await self._scope(payload)
        instance = await self._instance(profile_id, scope, payload)
        instance_id = instance["instance_id"]
        detail = await self._instance_detail(profile_id, instance_id, payload)
        await acknowledge_delivery_failure(
            self.profiles_repository,
            profile_id=profile_id,
            instance_id=instance_id,
            outbox=detail.get("outbox") or (),
            occurrence_id=payload.get("occurrence_id"),
        )
        return action_result_view("失败记录仍会保留，侧栏不再提醒这一次失败")

    async def _main_config(self, action: str, payload: dict[str, Any]) -> Any:
        profile_id = await self._profile_id(payload)
        return await handle_main_config_action(self.profiles, profile_id, action, payload)

    async def _character_model(self, action: str, payload: dict[str, Any]) -> Any:
        profile_id = await self._profile_id(payload)
        if action == "get_character_model":
            return await self.character_models.snapshot(profile_id)
        return await self.character_models.save(profile_id, payload)

    async def _ai(self, action: str, payload: dict[str, Any]) -> Any:
        profile_id = await self._profile_id(payload)
        if action in {"ai_work_record", "ai_work_attempt_debug", "ai_work_attempt_raw"}:
            scope = str(payload.get("scope") or "").strip().lower()
            if scope not in {"private", "group"}:
                raise ValueError("scope must be 'private' or 'group'")
            instance = await self._instance(profile_id, scope, payload)
            payload = {**payload, "instance_id": str(instance["instance_id"])}
        result = await self.ai.handle(action, profile_id, payload)
        wake_actions = {
            "ai_api_package",
            "ai_api_package_credential",
            "ai_api_model",
            "ai_api_package_probe",
            "ai_api_model_probe",
            "ai_backend_probe",
        }
        if action in wake_actions and bool(result.get("ok", True)):
            await self.sticker_repository.wake_waiting_sticker_checks(profile_id)
        return result

    async def _web(self, action: str, payload: dict[str, Any]) -> Any:
        result = await self.web.handle(action, await self._profile_id(payload), payload)
        return web_library_view(result) if action == "web_snapshot" else result

    async def _reset_instance(self, payload: dict[str, Any]) -> Any:
        profile_id, scope = await self._scope(payload)
        instance = await self._instance(profile_id, scope, payload)
        instance_id = str(instance["instance_id"])
        if str(payload.get("confirm_instance_id") or "") != instance_id:
            raise ValueError("confirm_instance_id must match the selected instance")
        mode = str(payload.get("mode") or "").strip().upper()
        expected_confirmation = {
            "ALL": "RESET_INSTANCE_ALL",
            "KEEP_STICKERS": "RESET_INSTANCE_KEEP_STICKERS",
        }.get(mode)
        if expected_confirmation is None:
            raise ValueError("reset mode must be ALL or KEEP_STICKERS")
        if str(payload.get("confirm") or "") != expected_confirmation:
            raise ValueError("conversation reset requires server confirmation")
        preserve_stickers = mode == "KEEP_STICKERS"
        await self.operations.reset_character_instance(
            profile_id,
            instance_id,
            preserve_stickers=preserve_stickers,
        )
        return action_result_view(
            "当前对话已重置，表情包已保留，正在重新初始化"
            if preserve_stickers
            else "当前对话已重置，表情包已清空，正在重新初始化"
        )

    async def _sticker(self, action: str, payload: dict[str, Any]) -> Any:
        profile_id, scope = await self._scope(payload)
        if action == "get_sticker_config":
            return await self.stickers.sticker_config_snapshot(profile_id, scope)
        if action == "save_sticker_config":
            return await self.stickers.save_sticker_config(profile_id, scope, payload)
        if action == "sticker_references":
            return await self.sticker_references.sticker_reference_snapshot(profile_id, scope)
        if action == "sticker_reference_action":
            return await self.sticker_references.sticker_reference_action(
                profile_id, scope, payload
            )
        instance = await self._instance(profile_id, scope, payload)
        return await self._sticker_instance(action, profile_id, instance["instance_id"], payload)

    async def _sticker_instance(
        self,
        action: str,
        profile_id: str,
        instance_id: str,
        payload: dict[str, Any],
    ) -> Any:
        intake_actions = {
            "sticker_intake",
            "sticker_intake_start",
            "sticker_intake_upload",
            "sticker_intake_action",
            "sticker_intake_preview",
        }
        if action == "sticker_snapshot":
            return await self._sticker_snapshot(profile_id, instance_id, payload)
        if action in intake_actions:
            return await self._sticker_intake_instance(
                action,
                profile_id,
                instance_id,
                payload,
            )
        if action == "sticker_action":
            await self.stickers.sticker_admin_action(profile_id, instance_id, payload)
            return action_result_view("表情包操作已经完成")
        if action == "sticker_stop":
            await self.stickers.stop_sticker_collection(profile_id, instance_id)
            return action_result_view("表情包自动搜集已经停止")
        if action == "sticker_clear":
            return await self._clear_sticker_instance(profile_id, instance_id, payload)
        await self.stickers.run_sticker_collection(
            profile_id,
            instance_id,
            mode=str(payload.get("mode") or "collect"),
            theme=str(payload.get("theme") or ""),
        )
        return action_result_view("表情包自动搜集已经开始运行")

    async def _sticker_intake_instance(
        self,
        action: str,
        profile_id: str,
        instance_id: str,
        payload: dict[str, Any],
    ) -> Any:
        if action == "sticker_intake":
            return await self.stickers.sticker_intake_snapshot(
                profile_id,
                instance_id,
                session_id=str(payload.get("session_id") or ""),
            )
        if action == "sticker_intake_preview":
            return await self.stickers.sticker_intake_preview(
                profile_id,
                instance_id,
                session_id=str(payload.get("session_id") or ""),
                entry_id=str(payload.get("entry_id") or ""),
            )
        handlers = {
            "sticker_intake_start": self.stickers.start_sticker_intake,
            "sticker_intake_upload": self.stickers.upload_sticker_intake_image,
            "sticker_intake_action": self.stickers.sticker_intake_action,
        }
        return await handlers[action](profile_id, instance_id, payload)

    async def _clear_sticker_instance(
        self,
        profile_id: str,
        instance_id: str,
        payload: dict[str, Any],
    ) -> Any:
        if str(payload.get("confirm_instance_id") or "") != instance_id:
            raise ValueError("confirm_instance_id must match the selected instance")
        if str(payload.get("confirm") or "") != "DELETE_STICKERS":
            raise ValueError("sticker clear requires server confirmation")
        return await self.stickers.clear_sticker_instance(profile_id, instance_id)

    async def _sticker_snapshot(
        self, profile_id: str, instance_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        view = str(payload.get("view") or "candidates").lower()
        snapshot = await self.stickers.sticker_runtime_snapshot(
            profile_id,
            instance_id,
            view=view,
            page=int(payload.get("page") or 1),
            page_size=int(payload.get("page_size") or 20),
        )
        return (
            feature_status_view("stickers", snapshot)
            if view == "tasks"
            else sticker_library_view(snapshot)
        )

    async def _diagnostic(self, action: str, payload: dict[str, Any]) -> Any:
        del action
        profile_id, scope = await self._scope(payload)
        instance = await self._instance(profile_id, scope, payload)
        return await self.diagnostics.support_bundle(
            profile_id,
            scope,
            instance_id=instance["instance_id"],
            include_model_content=str(payload.get("include_model_content") or "").lower()
            in {"1", "true", "yes", "on"},
        )

    async def _profile_action(self, action: str, payload: dict[str, Any]) -> Any:
        profile_id, scope = await self._scope(payload)
        if action == "get_scope_config":
            return await self.profiles.scope_config_snapshot(profile_id, scope)
        if action == "save_scope_config":
            return await self.profile_settings.save_scope_configuration(profile_id, scope, payload)
        if action == "instances":
            snapshot = await self.profiles.role_instances_snapshot(profile_id)
            return {
                "profile_id": profile_id,
                "scope": scope,
                "instances": snapshot["sections"][scope],
            }
        instance = await self._instance(profile_id, scope, payload)
        instance_id = instance["instance_id"]
        if action == "instance_detail":
            return await self._instance_detail(profile_id, instance_id, payload)
        if action == "get_instance_contact_override":
            return await self.profile_settings.instance_contact_override_snapshot(
                profile_id, instance_id
            )
        if action == "get_platform_contact_policy":
            return await self.profile_settings.platform_contact_policy_snapshot(
                profile_id, instance
            )
        if action == "save_platform_contact_policy":
            return await self.profile_settings.save_platform_contact_policy(
                profile_id, instance, payload
            )
        if action == "save_instance_chat_policy":
            return await self.profile_settings.save_instance_chat_policy(
                profile_id, instance_id, payload
            )
        return await self.profile_settings.save_instance_contact_override(
            profile_id, instance_id, payload
        )

    async def _instance_detail(
        self, profile_id: str, instance_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        snapshot = await self.profiles.character_instance_snapshot(profile_id, instance_id)
        diagnostics = await self.timeline.character_instance_diagnostics(
            profile_id,
            instance_id,
            message_page=max(1, int(payload.get("message_page") or 1)),
            message_page_size=max(5, min(int(payload.get("message_page_size") or 20), 100)),
        )
        return {**snapshot, **diagnostics}

    async def _bridge(self, action: str, payload: dict[str, Any]) -> Any:
        profile_id, scope = await self._scope(payload)
        instance = await self._instance(profile_id, scope, payload)
        instance_id = instance["instance_id"]
        if action == "controlled_bridge_snapshot":
            return await self.timeline.controlled_bridge_snapshot(profile_id, instance_id)
        if action == "character_intent_detail":
            detail = await self.timeline_repository.get_character_intent(
                profile_id, instance_id, str(payload.get("intent_id") or "").strip()
            )
            if detail is None:
                raise ValueError("character intent no longer exists")
            return jsonable(detail)
        if action == "character_intent_action":
            await self.timeline.character_intent_admin_action(profile_id, instance_id, payload)
            return action_result_view("角色意图已经更新")
        raise ValueError("unknown controlled bridge action")

    async def _context(self, action: str, payload: dict[str, Any]) -> Any:
        profile_id, scope = await self._scope(payload)
        instance_id = str(payload.get("instance_id") or "").strip()
        if not instance_id:
            config = await self.profiles.scope_config_snapshot(profile_id, scope)
            if action == "context_snapshot":
                return {**config, "diagnostics": None}
            raise ValueError("instance_id is required")
        await self._instance(profile_id, scope, payload)
        if action == "context_snapshot":
            return await self.timeline.context_snapshot(profile_id, scope, instance_id)
        if action == "context_summary":
            result = await self.timeline.force_context_summary(profile_id, instance_id)
            if result.get("ok") is False:
                raise ValueError(str(result.get("reason") or "当前对话还不需要整理"))
            return action_result_view("历史对话整理已经开始")
        return context_budget_view(await self.timeline.context_dry_run(profile_id, instance_id))

    async def _profile_id(self, payload: Mapping[str, Any]) -> str:
        profile_id = str(payload.get("profile_id") or "").strip()
        if profile_id:
            return profile_id
        if str(payload.get("role_ref") or "").strip():
            return await self._player_profile_id(payload)
        raise ValueError("profile_id is required")

    async def _scope(self, payload: Mapping[str, Any]) -> tuple[str, str]:
        profile_id = await self._profile_id(payload)
        scope = str(payload.get("scope") or "").strip().lower()
        if scope not in {"private", "group"} and str(payload.get("contact_ref") or "").strip():
            contact = await self._player_contact(profile_id, payload)
            scope = str(contact["scope"])
        if scope not in {"private", "group"}:
            raise ValueError("scope must be 'private' or 'group'")
        return profile_id, scope

    async def _instance(
        self, profile_id: str, scope: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        instance_id = str(payload.get("instance_id") or "").strip()
        if not instance_id and str(payload.get("contact_ref") or "").strip():
            contact = await self._player_contact(profile_id, payload)
            if str(contact.get("scope") or "") != scope:
                raise ValueError("contact does not belong to the selected scope")
            return contact
        if not instance_id:
            raise ValueError("instance_id is required")
        instance = await self.profiles.require_role_instance(profile_id, instance_id)
        if instance.get("scope") != scope:
            raise ValueError("instance does not belong to the selected scope")
        return instance
