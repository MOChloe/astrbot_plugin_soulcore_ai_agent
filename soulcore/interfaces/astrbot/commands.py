"""AstrBot command controller.

Decorators remain in main.py; this class owns command parsing and orchestration.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

    from .command_probes import CommandProbeController

from ...features.delivery.ports import DeliveryRepositoryPort
from ...features.knowledge.ports import KnowledgeRepositoryPort
from ...features.main_core.ports import MainCoreQueryPort
from ...features.profiles.ports import ProfilesRepositoryPort
from ...features.stickers.policy import load_sticker_runtime_policy
from ...features.stickers.ports import StickerRepositoryPort
from ...features.timeline.ports import TimelineRepositoryPort
from ...features.web.ports import WebRepositoryPort
from ...features.web.service import WebCommandContext, WebResearchService
from ..admin.controllers.diagnostics import DiagnosticsAdminController
from ..admin.controllers.knowledge import KnowledgeAdminController
from ..admin.controllers.media import MediaAdminController
from ..admin.controllers.operations import RuntimeOperationsController
from ..admin.controllers.profiles import ProfilesAdminController
from ..admin.controllers.stickers import StickersAdminController
from ..admin.controllers.timeline import TimelineAdminController
from ..admin.presentation import jsonable
from .profile import ProfileResolver
from .umo import CapturedUMO


class AICommandPort(Protocol):
    async def ai_manager_snapshot(self, profile_id: str) -> dict[str, Any]: ...
    async def handle(
        self, method: str, profile_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]: ...


class AITaskQueryPort(Protocol):
    async def list_ai_tasks(self, **values: object) -> list[Any]: ...
    async def probe_ai_backend(self, backend_id: str, *, profile_id: str) -> dict[str, Any]: ...


class OutboundCommandPort(Protocol):
    async def send_and_record_foreground(self, **kwargs: Any) -> bool: ...


class WebCommandPort(Protocol):
    async def handle(
        self, method: str, profile_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]: ...


class CommandController:
    def __init__(
        self,
        *,
        profiles_repository: ProfilesRepositoryPort,
        timeline_repository: TimelineRepositoryPort,
        main_core_queries: MainCoreQueryPort,
        knowledge_repository: KnowledgeRepositoryPort,
        delivery_repository: DeliveryRepositoryPort,
        ai_repository: AITaskQueryPort,
        sticker_repository: StickerRepositoryPort,
        web_repository: WebRepositoryPort,
        profile_resolver: ProfileResolver,
        profiles: ProfilesAdminController,
        timeline: TimelineAdminController,
        knowledge: KnowledgeAdminController,
        ai: AICommandPort,
        media: MediaAdminController,
        stickers: StickersAdminController,
        web: WebCommandPort,
        diagnostics: DiagnosticsAdminController,
        operations: RuntimeOperationsController,
        outbound: OutboundCommandPort,
        probes: CommandProbeController,
        web_research: WebResearchService,
    ) -> None:
        self.profiles_repository = profiles_repository
        self.timeline_repository = timeline_repository
        self.main_core_queries = main_core_queries
        self.knowledge_repository = knowledge_repository
        self.delivery_repository = delivery_repository
        self.ai_repository = ai_repository
        self.sticker_repository = sticker_repository
        self.web_repository = web_repository
        self.profile_resolver = profile_resolver
        self.profiles = profiles
        self.timeline = timeline
        self.knowledge = knowledge
        self.ai = ai
        self.media = media
        self.stickers = stickers
        self.web = web
        self.diagnostics_controller = diagnostics
        self.operations = operations
        self.outbound = outbound
        self.probes = probes
        self.web_research = web_research

    async def _command_profile(self, event: AstrMessageEvent, profile_id: str) -> str:
        if profile_id:
            await self.profiles.require_known_profile(profile_id)
            return profile_id
        return (await self.profile_resolver.resolve_event(event)).id

    async def _command_instance(self, event: AstrMessageEvent, profile_id: str) -> tuple[str, Any]:
        pid = await self._command_profile(event, profile_id)
        captured = CapturedUMO.parse(str(event.unified_msg_origin))
        if not captured.is_valid:
            raise ValueError("current command has no valid conversation instance")
        instance = await self.profiles_repository.ensure_character_instance(
            pid,
            captured.raw,
            platform_id=captured.platform_id,
            message_type=captured.message_type,
            target_id=captured.target_id,
            session_kind=captured.kind.value,
        )
        return pid, instance

    def _result(self, event: AstrMessageEvent, value: Any):
        return event.plain_result(
            json.dumps(jsonable(value), ensure_ascii=False, indent=2, default=str)
        )

    async def cmd_help(self, event: AstrMessageEvent):
        yield event.plain_result(
            "SoulCore 调试指令：doctor / profiles / status / state / targets / routes / "
            "instances / schedules / runs / run / context / summarize / "
            "knowledge / memory / world_info / outbox / ai / image / web / "
            "tick / probe"
            " / sticker；快捷测试：/sticker_status /sticker_collect [主题] "
            "/sticker_reinforce <item_id>；"
            "私聊重置：/soulcore重置（保留表情包）或 "
            "/soulcore重置并清空表情包"
        )

    async def cmd_doctor(self, event: AstrMessageEvent):
        pid = await self._command_profile(event, "")
        yield self._result(event, await self.diagnostics_controller.diagnostics(pid))

    async def cmd_profiles(self, event: AstrMessageEvent):
        yield self._result(event, await self.profiles.sync_profiles())

    async def cmd_status(self, event: AstrMessageEvent, profile_id: str = ""):
        pid, instance = await self._command_instance(event, profile_id)
        yield self._result(
            event,
            await self.profiles.character_instance_snapshot(pid, instance.instance_id),
        )

    async def cmd_state(self, event: AstrMessageEvent, profile_id: str = ""):
        pid, instance = await self._command_instance(event, profile_id)
        yield self._result(
            event,
            await self.profiles_repository.get_instance_state(pid, instance.instance_id),
        )

    async def cmd_targets(self, event: AstrMessageEvent, profile_id: str = ""):
        pid, instance = await self._command_instance(event, profile_id)
        yield self._result(event, instance)

    async def cmd_routes(self, event: AstrMessageEvent, profile_id: str = ""):
        pid, instance = await self._command_instance(event, profile_id)
        yield self._result(
            event,
            {"profile_id": pid, "instance_id": instance.instance_id, "umo": instance.route_umo},
        )

    async def cmd_instances(self, event: AstrMessageEvent, profile_id: str = ""):
        pid = await self._command_profile(event, profile_id)
        yield self._result(event, await self.profiles.role_instances_snapshot(pid))

    async def cmd_schedules(self, event: AstrMessageEvent, profile_id: str = ""):
        pid, instance = await self._command_instance(event, profile_id)
        yield self._result(
            event,
            await self.timeline_repository.list_instance_wakeups(pid, instance.instance_id),
        )

    async def cmd_runs(self, event: AstrMessageEvent, profile_id: str = "", limit: int = 10):
        pid, instance = await self._command_instance(event, profile_id)
        yield self._result(
            event,
            await self.main_core_queries.list_instance_runs(
                pid, instance.instance_id, max(1, min(limit, 50))
            ),
        )

    async def cmd_run(self, event: AstrMessageEvent, run_id: int):
        for profile in await self.profiles_repository.list_profiles():
            for instance in await self.profiles_repository.list_character_instances(
                profile.profile_id
            ):
                rows = await self.main_core_queries.list_instance_runs(
                    profile.profile_id, instance.instance_id, 100
                )
                row = next(
                    (item for item in rows if int(item.get("run_id") or 0) == run_id),
                    None,
                )
                if row:
                    yield self._result(event, row)
                    return
        yield event.plain_result("没有找到该 run_id。")

    async def cmd_context(self, event: AstrMessageEvent, profile_id: str = ""):
        pid, instance = await self._command_instance(event, profile_id)
        yield self._result(
            event,
            await self.timeline.context_snapshot(pid, instance.scope, instance.instance_id),
        )

    async def cmd_summarize(self, event: AstrMessageEvent, profile_id: str = "", mode: str = "dry"):
        if profile_id.lower() in {"commit", "dry"} and mode == "dry":
            mode, profile_id = profile_id, ""
        pid, instance = await self._command_instance(event, profile_id)
        value = (
            await self.timeline.force_context_summary(pid, instance.instance_id)
            if str(mode).lower() == "commit"
            else await self.timeline.context_snapshot(pid, instance.scope, instance.instance_id)
        )
        yield self._result(event, value)

    async def cmd_knowledge(
        self,
        event: AstrMessageEvent,
        action: str = "status",
        value: str = "",
        profile_id: str = "",
    ):
        """Read or safely trigger the instance-scoped knowledge system."""

        action = str(action or "status").strip().lower()
        if (
            action in {"status", "audit"}
            and not profile_id
            and value
            or action in {"dry", "commit"}
            and not profile_id
            and value
        ):
            profile_id, value = value, ""
        pid, instance = await self._command_instance(event, profile_id)
        result = await self._knowledge_action(pid, instance.instance_id, action, value)
        yield self._result(event, result)

    async def _knowledge_action(
        self, profile_id: str, instance_id: str, action: str, value: str
    ) -> dict[str, Any]:
        if action == "status":
            return await self.knowledge.knowledge_snapshot(profile_id, instance_id)
        if action in {"dry", "commit", "form"}:
            mode = self._knowledge_form_mode(action, value)
            return await self.knowledge.knowledge_form(profile_id, instance_id, mode)
        if action == "recall":
            return await self.knowledge.recall_probe(profile_id, instance_id, value)
        if action in {"memory", "world_info"}:
            kind = "memory" if action == "memory" else "world_info"
            return await self.knowledge.knowledge_record(
                profile_id, instance_id, kind, int(value or 0)
            )
        if action == "audit":
            rows = await self.knowledge_repository.list_knowledge_audit(
                profile_id, instance_id, limit=100
            )
            return {"audit": rows}
        return {"ok": False, "error": self._knowledge_usage()}

    @staticmethod
    def _knowledge_form_mode(action: str, value: str) -> str:
        if action == "form":
            return str(value or "dry").strip().lower()
        return "commit" if action == "commit" else "dry"

    @staticmethod
    def _knowledge_usage() -> str:
        return (
            "usage: /soulcore knowledge "
            "status|form dry|commit [profile_id]|recall <query>|"
            "memory <id>|world_info <id>|audit"
        )

    async def cmd_memory(self, event: AstrMessageEvent, memory_id: int, profile_id: str = ""):
        pid, instance = await self._command_instance(event, profile_id)
        yield self._result(
            event,
            await self.knowledge.knowledge_record(
                pid, instance.instance_id, "memory", int(memory_id)
            ),
        )

    async def cmd_world_info(
        self, event: AstrMessageEvent, world_info_id: int, profile_id: str = ""
    ):
        pid, instance = await self._command_instance(event, profile_id)
        yield self._result(
            event,
            await self.knowledge.knowledge_record(
                pid, instance.instance_id, "world_info", int(world_info_id)
            ),
        )

    async def cmd_outbox(self, event: AstrMessageEvent, profile_id: str = "", limit: int = 10):
        pid, instance = await self._command_instance(event, profile_id)
        yield self._result(
            event,
            await self.delivery_repository.list_instance_outbox(
                pid,
                instance.instance_id,
                limit=max(1, min(limit, 50)),
            ),
        )

    async def cmd_ai(
        self,
        event: AstrMessageEvent,
        action: str = "status",
        value: str = "",
        reason: str = "",
    ):
        """Inspect and control the unified AI manager."""

        pid = await self._command_profile(event, "")
        action = str(action or "status").strip().lower()
        if action == "status":
            result = await self.ai.ai_manager_snapshot(pid)
        elif action == "tasks":
            rows = await self.ai_repository.list_ai_tasks(
                profile_id=pid,
                status=str(value or "").strip() or None,
                limit=100,
            )
            result = {"tasks": rows}
        elif action == "task":
            result = await self.ai.handle("ai_task", pid, {"task_id": int(value or 0)})
        elif action == "providers":
            result = (await self.ai.ai_manager_snapshot(pid))["backends"]
        elif action == "probe":
            result = await self.ai.probe_ai_backend(str(value), profile_id=pid)
        elif action in {"pause", "resume", "cancel", "retry"}:
            result = await self.ai.handle(
                "ai_task_action",
                pid,
                {
                    "task_id": int(value or 0),
                    "action": action,
                    "reason": str(reason or "").strip(),
                },
            )
        else:
            result = {
                "ok": False,
                "error": (
                    "usage: /soulcore ai "
                    "status|tasks [status]|task <id>|providers|probe <backend>|"
                    "pause|resume|cancel|retry <id>"
                ),
            }
        yield self._result(event, result)

    async def cmd_image(
        self,
        event: AstrMessageEvent,
        action: str = "status",
        value: str = "",
    ):
        """Inspect the current instance's media system."""

        pid, instance = await self._command_instance(event, "")
        action = str(action or "status").strip().lower()
        if action in {"status", "assets"}:
            snapshot = await self.media.image_snapshot(pid, instance.instance_id)
            result = snapshot if action == "status" else {"assets": snapshot["assets"]}
        elif action == "task":
            result = await self.ai.handle("ai_task", pid, {"task_id": int(value or 0)})
        else:
            result = {
                "ok": False,
                "error": "usage: /soulcore image status|assets|task <task_id>",
            }
        yield self._result(event, result)

    async def cmd_sticker(
        self,
        event: AstrMessageEvent,
        action: str = "status",
        value: str = "",
    ):
        """Exercise the production sticker pipeline for the current route."""

        pid, instance = await self._command_instance(event, "")
        action = str(action or "status").strip().lower()
        value = str(value or "").strip()
        if action == "status":
            result = await self.stickers.sticker_runtime_snapshot(pid, instance.instance_id)
        elif action == "list":
            rows = await self.sticker_repository.list_sticker_items(
                pid, instance.instance_id, query=value, status="ACTIVE", limit=50
            )
            result = {"items": jsonable(rows), "count": len(rows)}
        elif action == "collect":
            result = await self.stickers.run_sticker_collection(
                pid, instance.instance_id, mode="collect", theme=value
            )
        elif action == "check":
            result = await self.stickers.sticker_admin_action(
                pid,
                instance.instance_id,
                {"action": "recheck", "record_kind": "candidate", "record_id": value},
            )
        elif action in {"reinforce", "unreinforce", "archive", "delete", "restore"}:
            result = await self.stickers.sticker_admin_action(
                pid,
                instance.instance_id,
                {"action": action, "record_kind": "item", "record_id": value},
            )
        elif action == "send":
            (
                await load_sticker_runtime_policy(
                    self.sticker_repository,
                    self.profiles_repository,
                    pid,
                    instance_id=instance.instance_id,
                )
            ).require_enabled()
            item = await self.sticker_repository.get_sticker_item(pid, instance.instance_id, value)
            if item is None or item.status.value != "ACTIVE":
                result = {"ok": False, "error": "找不到可用的正式表情包"}
            else:
                admin_run_id = int(datetime.now(UTC).timestamp() * 1_000_000)
                refs = await self.sticker_repository.create_sticker_run_refs(
                    pid,
                    instance.instance_id,
                    admin_run_id,
                    [item.item_id],
                    expires_at=datetime.now(UTC) + timedelta(hours=1),
                )
                if not refs:
                    result = {"ok": False, "error": "无法创建受控试发引用"}
                else:
                    sent = await self.outbound.send_and_record_foreground(
                        event=event,
                        profile_id=pid,
                        instance_id=instance.instance_id,
                        text="",
                        media_asset_ids=[],
                        sticker_ref_ids=[refs[0].sticker_ref],
                        file_asset_ids=[],
                        important_todo_ids=[],
                        run_id=admin_run_id,
                        idempotency_key=f"sticker-admin-send:{admin_run_id}",
                        metadata={"run_id": admin_run_id, "admin_sticker_test": True},
                    )
                    result = (
                        {
                            "ok": True,
                            "item_id": item.item_id,
                            "message": "已通过正式发送链试发表情包",
                        }
                        if sent
                        else {"ok": False, "error": "表情包系统已关闭，未发送"}
                    )
        else:
            result = {
                "ok": False,
                "error": (
                    "usage: /soulcore sticker status|list [关键词]|collect [主题]|"
                    "generate <主题>|check <candidate_id>|send <item_id>|"
                    "reinforce|unreinforce|archive|delete|restore <item_id>"
                ),
            }
        yield self._result(event, result)

    async def cmd_sticker_status_shortcut(self, event: AstrMessageEvent):
        """Shortcut: inspect the current instance sticker library."""

        pid, instance = await self._command_instance(event, "")
        yield self._result(
            event, await self.stickers.sticker_runtime_snapshot(pid, instance.instance_id)
        )

    async def cmd_sticker_collect_shortcut(self, event: AstrMessageEvent, theme: str = ""):
        """Shortcut: run web/generation collection through the full Check layer."""

        pid, instance = await self._command_instance(event, "")
        yield self._result(
            event,
            await self.stickers.run_sticker_collection(
                pid, instance.instance_id, mode="collect", theme=str(theme or "")
            ),
        )

    async def cmd_sticker_reinforce_shortcut(self, event: AstrMessageEvent, item_id: str = ""):
        """Shortcut: protect a known favourite from low-value eviction."""

        pid, instance = await self._command_instance(event, "")
        yield self._result(
            event,
            await self.stickers.sticker_admin_action(
                pid,
                instance.instance_id,
                {"action": "reinforce", "record_kind": "item", "record_id": item_id},
            ),
        )

    async def _web_search_command(self, pid: str, instance: Any, action: str, value: str):
        query = str(value or "").strip()
        if not query:
            return {"ok": False, "error": "usage: /soulcore web search <query>"}
        if self.web_research is None:
            return {"ok": False, "error": "web research service is unavailable"}
        profile = await self.profiles_repository.get_profile(pid)
        if profile is None or not bool(profile.web_search_enabled):
            return {"ok": False, "error": "当前角色已关闭联网查询"}
        context = WebCommandContext(
            self.web_research,
            profile_id=pid,
            instance_id=instance.instance_id,
            caller_id="admin-command",
            core_run_id=f"admin-{uuid.uuid4().hex}",
            intensity=str(profile.web_search_intensity),
        )
        response = (
            await context.search_images(query, "ANSWER_USER")
            if action == "images"
            else await context.search_web(query, "ANSWER_USER")
        )
        return response.as_result_data()

    async def cmd_web(
        self,
        event: AstrMessageEvent,
        action: str = "status",
        value: str = "",
    ):
        """Inspect or deliberately probe the profile-scoped web service."""
        pid, instance = await self._command_instance(event, "")
        action = str(action or "status").strip().lower()
        if action == "status":
            result = {
                "config": await self.web.handle("get_web_config", pid, {}),
                **await self.web.handle("web_providers", pid, {}),
            }
        elif action == "providers":
            result = await self.web.handle("web_providers", pid, {})
        elif action in {"search", "images"}:
            result = await self._web_search_command(pid, instance, action, value)
        elif action == "session":
            row = await self.web_repository.get_web_search_session(
                pid, instance.instance_id, str(value or ""), include_expired=True
            )
            result = {"session": jsonable(row)}
        elif action in {"probe", "probe-image"}:
            result = await self.web.handle(
                "web_image_probe" if action == "probe-image" else "web_provider_probe",
                pid,
                {"provider_id": str(value or ""), "confirm_cost": True},
            )
        else:
            result = {
                "ok": False,
                "error": (
                    "usage: /soulcore web status|providers|search|images <query>|"
                    "session <id>|probe|probe-image <provider_id>"
                ),
            }
        yield self._result(event, result)

    async def cmd_tick(self, event: AstrMessageEvent, profile_id: str = "", mode: str = "dry"):
        if profile_id.lower() in {"commit", "dry"} and mode == "dry":
            mode, profile_id = profile_id, ""
        pid = await self._command_profile(event, profile_id)
        commit = str(mode).lower() == "commit"
        captured = CapturedUMO.parse(str(event.unified_msg_origin))
        instance = (
            await self.profiles_repository.ensure_character_instance(
                pid,
                captured.raw,
                platform_id=captured.platform_id,
                message_type=captured.message_type,
                target_id=captured.target_id,
                session_kind=captured.kind.value,
            )
            if captured.is_valid
            else None
        )
        yield self._result(
            event,
            (
                await self.operations.trigger_instance_tick(
                    pid,
                    instance.instance_id,
                    commit=commit,
                    wait=True,
                    event=event,
                    force_proactive_delivery=False,
                )
                if instance is not None
                else {"ok": False, "error": "invalid_current_instance_route"}
            ),
        )

    async def cmd_reset(self, event: AstrMessageEvent):
        result = await self._reset_current_conversation(event, preserve_stickers=True)
        if result is not None:
            yield result

    async def cmd_reset_and_clear_stickers(self, event: AstrMessageEvent):
        result = await self._reset_current_conversation(event, preserve_stickers=False)
        if result is not None:
            yield result

    async def _reset_current_conversation(
        self,
        event: AstrMessageEvent,
        *,
        preserve_stickers: bool,
    ):
        pid, instance = await self._command_instance(event, "")
        if str(instance.scope) != "private":
            return None
        await self.operations.reset_character_instance(
            pid,
            instance.instance_id,
            preserve_stickers=preserve_stickers,
        )
        # The reset operation already queued and attempted the one allowed
        # initialization-start notice before workers resumed.
        return None

    async def cmd_probe(self, event: AstrMessageEvent, component: str, profile_id: str = ""):
        pid = await self._command_profile(event, profile_id)
        value = await self.probes.probe(event, pid, component)
        yield self._result(event, value)
