"""Cross-domain redacted diagnostics and support bundles."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol

from ....contracts.ai_models import AIErrorCode
from ....features.ai.prompt_debug import prompt_jsonable
from ....features.conversation.ports import ConversationRepositoryPort
from ....features.conversation.service import ConversationContextService
from ....features.delivery.ports import DeliveryRepositoryPort
from ....features.knowledge.service import KnowledgeFormationPlugin
from ....features.main_core.ports import MainCoreQueryPort
from ....features.media.ports import MediaRepositoryPort
from ....features.profiles.credentials import CredentialVault
from ....features.profiles.ports import ProfilesRepositoryPort
from ....features.recall import RecallService
from ....features.stickers.ports import StickerRepositoryPort
from ....features.timeline.ports import TimelineRepositoryPort
from ....features.timeline.scheduler import DurableSchedulerWorker
from ....features.web.ports import WebRepositoryPort
from ....shared.event_log import EventLogPort
from ..presentation import jsonable
from .knowledge import KnowledgeAdminController
from .profiles import ProfilesAdminController


class AITaskDiagnosticsPort(Protocol):
    async def list_ai_tasks(self, **values: object) -> list[Any]: ...
    async def list_ai_workflow_summaries(self, **values: object) -> list[Any]: ...
    async def list_ai_work_nodes(self, workflow_id: int) -> list[Any]: ...
    async def list_ai_provider_attempts(self, **values: object) -> list[Any]: ...


class AIAdminDiagnosticsPort(Protocol):
    async def ai_manager_snapshot(self, profile_id: str) -> dict[str, Any]: ...


class DiagnosticsAdminController:
    def __init__(
        self,
        *,
        profiles_repository: ProfilesRepositoryPort,
        timeline_repository: TimelineRepositoryPort,
        background_repository: Any,
        main_core_queries: MainCoreQueryPort,
        delivery_repository: DeliveryRepositoryPort,
        conversation_repository: ConversationRepositoryPort,
        ai_repository: AITaskDiagnosticsPort,
        web_repository: WebRepositoryPort,
        media_repository: MediaRepositoryPort,
        sticker_repository: StickerRepositoryPort,
        event_log: EventLogPort,
        profiles: ProfilesAdminController,
        knowledge: KnowledgeAdminController,
        ai: AIAdminDiagnosticsPort,
        scheduler: DurableSchedulerWorker,
        knowledge_plugin: KnowledgeFormationPlugin,
        recall: RecallService,
        context_service: ConversationContextService,
        credential_vault: CredentialVault,
        plugin_version: str,
        group_flow_repository: Any | None = None,
    ) -> None:
        self.profiles_repository = profiles_repository
        self.timeline_repository = timeline_repository
        self.background_repository = background_repository
        self.main_core_queries = main_core_queries
        self.delivery_repository = delivery_repository
        self.conversation_repository = conversation_repository
        self.ai_repository = ai_repository
        self.web_repository = web_repository
        self.media_repository = media_repository
        self.sticker_repository = sticker_repository
        self.event_log = event_log
        self.profiles = profiles
        self.knowledge = knowledge
        self.ai = ai
        self.scheduler = scheduler
        self.knowledge_plugin = knowledge_plugin
        self.recall = recall
        self.context_service = context_service
        self.credential_vault = credential_vault
        self.plugin_version = plugin_version
        self.group_flow_repository = group_flow_repository

    async def diagnostics(
        self, profile_id: str, *, instance_id: str | None = None
    ) -> dict[str, Any]:
        enabled = await self.profiles_repository.get_profile_soulcore_enabled(profile_id)
        instances = await self.profiles_repository.list_character_instances(profile_id)
        if instance_id is not None:
            instances = [item for item in instances if item.instance_id == instance_id]
            if not instances:
                raise ValueError("instance does not belong to the selected profile")
        states, runs, outbox, wakeups = await self._runtime_rows(profile_id, instances)
        profile = await self.profiles_repository.get_profile(profile_id)
        web_providers = await self.web_repository.list_web_search_providers(profile_id)
        web_ready, image_ready = self._web_readiness(web_providers)
        ai_snapshot = await self.ai.ai_manager_snapshot(profile_id)
        return {
            "doctor": self._doctor(enabled, profile, web_ready, image_ready, ai_snapshot),
            "main_config": {"profile_id": profile_id, "enabled": enabled},
            "web_research": self._web_diagnostics(profile, web_providers, web_ready, image_ready),
            "instances": [jsonable(item) for item in instances],
            "states": states,
            "routes": [jsonable(item) for item in instances],
            "runs": [jsonable(item) for item in runs],
            "outbox": [jsonable(item) for item in outbox],
            "wakeups": [jsonable(item) for item in wakeups],
            "scheduling": self._scheduling(instances, states),
            "group_flow": await self._group_flow_rows(profile_id, instances),
        }

    async def _group_flow_rows(self, profile_id: str, instances: list[Any]) -> list[dict[str, Any]]:
        repository = self.group_flow_repository
        if repository is None:
            return []
        rows = []
        for instance in instances:
            if instance.scope != "group":
                continue
            snapshot = await repository.diagnostic(profile_id, instance.instance_id)
            rows.append(
                {
                    "instance_id": instance.instance_id,
                    "window": jsonable(snapshot.window),
                    "next_window": jsonable(snapshot.next_window),
                    "algorithm": jsonable(snapshot.algorithm),
                }
            )
        return rows

    async def _runtime_rows(
        self, profile_id: str, instances: list[Any]
    ) -> tuple[list[dict[str, Any]], list[Any], list[Any], list[Any]]:
        states, runs, outbox, wakeups = [], [], [], []
        for instance in instances:
            iid = instance.instance_id
            state = await self.profiles_repository.get_instance_state(profile_id, iid)
            states.append({"instance_id": iid, **(jsonable(state) or {})})
            runs.extend(await self.main_core_queries.list_instance_runs(profile_id, iid, limit=5))
            outbox.extend(
                await self.delivery_repository.list_instance_outbox(profile_id, iid, limit=5)
            )
            wakeups.extend(
                await self.timeline_repository.list_instance_wakeups(profile_id, iid, limit=5)
            )
        runs.sort(key=lambda item: str(item.get("started_at") or ""), reverse=True)
        outbox.sort(key=lambda item: str(item.created_at or ""), reverse=True)
        wakeups.sort(key=lambda item: str(item.due_at or ""), reverse=True)
        return states, runs[:20], outbox[:20], wakeups[:20]

    def _web_readiness(self, providers: list[Any]) -> tuple[int, int]:
        ready = [
            item
            for item in providers
            if item.enabled
            and item.archived_at is None
            and self.credential_vault.describe(item.credential_id).configured
        ]
        image_kinds = {"TAVILY", "BOCHA", "BRAVE", "FIRECRAWL", "BAIDU_AI"}
        images = [item for item in ready if str(item.provider_kind).upper() in image_kinds]
        return len(ready), len(images)

    def _doctor(
        self,
        enabled: bool,
        profile: Any,
        web_ready: int,
        image_ready: int,
        ai_snapshot: Mapping[str, Any],
    ) -> list[dict[str, str]]:
        chat_pool: Mapping[str, Any] = next(
            (
                item
                for item in ai_snapshot.get("capability_pools") or []
                if item.get("capability") == "chat.completion"
            ),
            {},
        )
        direct_text_backends = len(chat_pool.get("backend_ids") or [])
        knowledge_ready = self.knowledge_plugin is not None and self.recall is not None
        web_enabled = bool(profile and profile.web_search_enabled)
        return [
            self._profile_switch_status(enabled),
            {"name": "database", "status": "ok", "message": "SQLite ready"},
            {
                "name": "direct_text_backend",
                "status": "ok" if direct_text_backends else "error",
                "message": f"{direct_text_backends} SoulCore direct text backend(s)",
            },
            self._worker_status("scheduler", self.scheduler, "durable worker active"),
            {"name": "context", "status": "ok", "message": "SoulCore-owned context manager active"},
            self._knowledge_status(knowledge_ready),
            self._web_status(web_enabled, web_ready, image_ready),
        ]

    @staticmethod
    def _profile_switch_status(enabled: bool) -> dict[str, str]:
        return {
            "name": "profile_master_switch",
            "status": "ok" if enabled else "warning",
            "message": (
                "SoulCore enabled for this AstrBot profile"
                if enabled
                else "SoulCore disabled; runtime work is paused and data is preserved"
            ),
        }

    @staticmethod
    def _knowledge_status(ready: bool) -> dict[str, str]:
        return {
            "name": "knowledge",
            "status": "ok" if ready else "error",
            "message": (
                "instance-scoped Memory/KnowledgeFact services active"
                if ready
                else "knowledge service unavailable"
            ),
        }

    @staticmethod
    def _web_status(enabled: bool, web_ready: int, image_ready: int) -> dict[str, str]:
        if not enabled:
            return {
                "name": "web_research",
                "status": "disabled",
                "message": "web research disabled for this profile",
            }
        return {
            "name": "web_research",
            "status": "ok" if web_ready else "warning",
            "message": f"web research enabled with {web_ready} web / {image_ready} image interface(s)",
        }

    @staticmethod
    def _worker_status(name: str, worker: DurableSchedulerWorker, success: str) -> dict[str, str]:
        running = bool(worker and worker.running)
        return {
            "name": name,
            "status": "ok" if running else "error",
            "message": success if running else str(worker.last_error or "not running"),
        }

    @staticmethod
    def _web_diagnostics(
        profile: Any, providers: list[Any], web_ready: int, image_ready: int
    ) -> dict[str, Any]:
        return {
            "enabled": bool(profile and profile.web_search_enabled),
            "intensity": str(profile.web_search_intensity if profile is not None else "STANDARD"),
            "provider_count": len(providers),
            "ready_provider_count": web_ready,
            "ready_image_provider_count": image_ready,
        }

    @staticmethod
    def _scheduling(instances: list[Any], states: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "instance_count": len(instances),
            "normal_count": sum(1 for state in states if not state.get("low_frequency_mode")),
            "low_frequency_count": sum(1 for state in states if state.get("low_frequency_mode")),
        }

    async def support_bundle(
        self,
        profile_id: str,
        scope: str,
        *,
        instance_id: str,
        include_model_content: bool = False,
    ) -> dict[str, Any]:
        config = await self.profiles.scope_config_snapshot(profile_id, scope)
        instances = await self._support_instances(profile_id, scope, instance_id)
        runtimes = [
            await self._support_runtime(
                profile_id,
                item,
                include_model_content=include_model_content,
            )
            for item in instances
        ]
        ai_snapshot = await self._redacted_ai_snapshot(profile_id, instance_id)
        web = await self._web_support(profile_id, instance_id)
        logs = await self.event_log.list_logs(profile_id, instance_id=instance_id, limit=500)
        bundle = {
            "format": "soulcore-support-bundle",
            "generated_at": datetime.now().astimezone().isoformat(),
            "plugin_version": self.plugin_version,
            "profile_id": profile_id,
            "scope": scope,
            **config,
            "instances": runtimes,
            "diagnostics": await self.diagnostics(profile_id, instance_id=instance_id),
            "ai_manager": ai_snapshot,
            "web_research": web,
            "logs": jsonable(logs),
            "log_retention_limit": 1000,
            "model_content_included": bool(include_model_content),
        }
        return _support_safe(bundle, include_model_content=include_model_content)

    async def _support_instances(
        self, profile_id: str, scope: str, instance_id: str
    ) -> list[dict[str, Any]]:
        snapshot = await self.profiles.role_instances_snapshot(profile_id)
        instances = list(snapshot["sections"][scope])
        selected = [item for item in instances if item["instance_id"] == instance_id]
        if not selected:
            raise ValueError("instance does not belong to the selected scope")
        return selected

    async def _support_runtime(
        self,
        profile_id: str,
        instance: Mapping[str, Any],
        *,
        include_model_content: bool,
    ) -> dict[str, Any]:
        iid = str(instance["instance_id"])
        scope_config = await self.profiles_repository.get_scope_config(
            profile_id, instance["scope"]
        )
        return {
            "instance": instance,
            "state": jsonable(await self.profiles_repository.get_instance_state(profile_id, iid)),
            "background": self._support_background_view(
                await self.background_repository.load_background_workspace(profile_id, iid)
            ),
            "runs": [
                self._support_run_view(run, include_model_content=include_model_content)
                for run in await self.main_core_queries.list_instance_runs(
                    profile_id, iid, limit=20
                )
            ],
            "wakeups": jsonable(
                await self.timeline_repository.list_instance_wakeups(profile_id, iid, limit=20)
            ),
            "outbox": jsonable(
                await self.delivery_repository.list_instance_outbox(profile_id, iid, limit=20)
            ),
            "context": jsonable(
                await self.context_service.diagnostics(profile_id, iid, scope_config)
            ),
            **await self._support_context(
                profile_id,
                iid,
                include_model_content=include_model_content,
            ),
            "knowledge": await self.knowledge.knowledge_support_snapshot(profile_id, iid),
            "stickers": await self.sticker_support_snapshot(profile_id, iid),
            "media": await self._support_media(profile_id, iid),
        }

    @staticmethod
    def _support_background_view(source: Mapping[str, Any]) -> dict[str, Any]:
        """Export health and scheduling without dumping private author material."""

        instance = source.get("instance")
        authors = source.get("authors")
        story_sources = source.get("story_sources")
        timeline = source.get("timeline")
        current_view = source.get("current_view")
        if not isinstance(instance, Mapping):
            raise ValueError("background instance diagnostic is unavailable")
        if not isinstance(authors, list):
            raise ValueError("background author diagnostics are unavailable")
        if not isinstance(story_sources, list) or not isinstance(timeline, list):
            raise ValueError("background content diagnostics are unavailable")
        author_views = []
        for item in authors:
            if not isinstance(item, Mapping):
                raise ValueError("background author diagnostic is invalid")
            author_views.append(
                {
                    "author_kind": item.get("author_kind"),
                    "status": item.get("status"),
                    "next_due_at": jsonable(item.get("next_due_at")),
                    "hard_due_at": jsonable(item.get("hard_due_at")),
                    "last_success_at": jsonable(item.get("last_success_at")),
                    "failure_count": int(item.get("failure_count") or 0),
                    "error_category": _support_error_category(item.get("last_error")),
                }
            )
        return {
            "enabled": bool(instance.get("enabled")),
            "initialization_state": instance.get("initialization_state"),
            "simulated_through_at": jsonable(instance.get("simulated_through_at")),
            "last_foreground_at": jsonable(instance.get("last_foreground_at")),
            "authors": author_views,
            "story_source_count": len(story_sources),
            "timeline_event_count": len(timeline),
            "current_view_present": bool(current_view),
        }

    async def _support_context(
        self,
        profile_id: str,
        instance_id: str,
        *,
        include_model_content: bool,
    ) -> dict[str, Any]:
        result = {
            "context_latest_build": jsonable(
                await self.conversation_repository.get_context_build_report(profile_id, instance_id)
            ),
        }
        if include_model_content:
            tasks = await self.ai_repository.list_ai_tasks(
                profile_id=profile_id, instance_id=instance_id, limit=100
            )
            result["ai_tasks"] = [self._support_ai_task_view(item) for item in tasks]
            result["context_ledger"] = jsonable(
                await self.conversation_repository.list_instance_messages(
                    profile_id,
                    instance_id,
                    limit=100,
                    ascending=False,
                    context_eligible_only=False,
                )
            )
            latest_summary = await self.conversation_repository.get_latest_dialogue_summary(
                profile_id, instance_id
            )
            result["context_summary"] = jsonable(latest_summary) if latest_summary else None
            workflows = await self.ai_repository.list_ai_workflow_summaries(
                profile_id=profile_id,
                instance_id=instance_id,
                limit=100,
            )
            result["ai_workflows"] = [
                [
                    self._support_ai_node_view(
                        node,
                        await self.ai_repository.list_ai_provider_attempts(
                            node_id=int(node["node_id"])
                        ),
                    )
                    for node in await self.ai_repository.list_ai_work_nodes(
                        int(workflow["workflow_id"])
                    )
                ]
                for workflow in workflows
            ]
            result["ai_workflows"] = [
                {
                    "workflow": self._support_ai_workflow_view(workflow),
                    "stages": stages,
                }
                for workflow, stages in zip(workflows, result["ai_workflows"], strict=True)
            ]
        return result

    @staticmethod
    def _support_run_view(run: Mapping[str, Any], *, include_model_content: bool) -> dict[str, Any]:
        """Project MainCore runs without exporting their internal request DTO by default."""

        result = {
            "source": run.get("source"),
            "status": run.get("status"),
            "started_at": jsonable(run.get("started_at")),
            "finished_at": jsonable(run.get("finished_at")),
            "error_category": _support_error_category(run.get("error")),
        }
        if include_model_content:
            result.update(
                {
                    "reason": run.get("reason"),
                    "request": prompt_jsonable(run.get("request")),
                    "decision": prompt_jsonable(run.get("decision")),
                }
            )
        return result

    @staticmethod
    def _support_ai_task_view(task: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: jsonable(task.get(key))
            for key in (
                "task_type",
                "status",
                "attempt_count",
                "max_attempts",
                "due_at",
                "created_at",
                "updated_at",
                "error_code",
            )
        }

    @staticmethod
    def _support_ai_workflow_view(workflow: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: jsonable(workflow.get(key))
            for key in (
                "primary_purpose",
                "status",
                "started_at",
                "finished_at",
                "final_error_code",
            )
        }

    @classmethod
    def _support_ai_node_view(
        cls,
        node: Mapping[str, Any],
        attempts: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "role": node.get("node_role"),
            "kind": node.get("node_kind"),
            "purpose": node.get("purpose"),
            "status": node.get("status"),
            "started_at": jsonable(node.get("started_at")),
            "finished_at": jsonable(node.get("finished_at")),
            "error_code": node.get("error_code"),
            "provider_attempts": [cls._support_ai_attempt_view(item) for item in attempts],
        }

    @staticmethod
    def _support_ai_attempt_view(attempt: Mapping[str, Any]) -> dict[str, Any]:
        request = attempt.get("request")
        response = attempt.get("response")
        safe_request = dict(request) if isinstance(request, Mapping) else {}
        safe_response = dict(response) if isinstance(response, Mapping) else {}
        return {
            "round": attempt.get("round_no"),
            "attempt": attempt.get("attempt_no"),
            "status": attempt.get("status"),
            "model": attempt.get("model_id"),
            "sent_at": jsonable(attempt.get("sent_at")),
            "finished_at": jsonable(attempt.get("finished_at")),
            "tokens": {
                "input": attempt.get("input_tokens"),
                "output": attempt.get("output_tokens"),
            },
            "error_code": attempt.get("error_code"),
            "model_exchange": prompt_jsonable(
                {
                    "request": {
                        key: safe_request.get(key)
                        for key in (
                            "logical_prompt",
                            "context_text",
                            "turn_text",
                            "capability_input",
                            "provider_envelope",
                        )
                    },
                    "response": {
                        key: safe_response.get(key)
                        for key in (
                            "text",
                            "finish_reason",
                            "capability_output",
                            "provider_envelope",
                        )
                    },
                }
            ),
        }

    async def _support_media(self, profile_id: str, instance_id: str) -> list[dict[str, Any]]:
        assets = await self.media_repository.list_media_assets(profile_id, instance_id, limit=100)
        return [
            {
                "asset_id": asset.asset_id,
                "origin": asset.origin.value,
                "purpose": asset.purpose.value,
                "file_status": asset.file_status.value,
                "inspection_status": asset.inspection_status.value,
                "delivery_status": asset.delivery_status,
                "byte_size": asset.byte_size,
                "width": asset.width,
                "height": asset.height,
                "core_run_id": asset.core_run_id,
                "created_at": jsonable(asset.created_at),
            }
            for asset in assets
        ]

    async def _redacted_ai_snapshot(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        snapshot = await self.ai.ai_manager_snapshot(profile_id)
        backends = list(snapshot.get("backends") or [])
        for backend in backends:
            backend.pop("credential_last4", None)
            backend.pop("credential_source", None)
            backend["credential_configured"] = bool(backend.get("credential_configured"))
        tasks = await self.ai_repository.list_ai_tasks(
            profile_id=profile_id,
            instance_id=instance_id,
            limit=1000,
        )
        counts: dict[str, int] = {}
        errors: dict[str, int] = {}
        for task in tasks:
            status = str(task.get("status") or "UNKNOWN")
            counts[status] = counts.get(status, 0) + 1
            if task.get("last_error"):
                code = str(task.get("error_code") or "UNKNOWN")
                errors[code] = errors.get(code, 0) + 1
        return {
            "backends": backends,
            "capability_pools": list(snapshot.get("capability_pools") or []),
            "pauses": list(snapshot.get("pauses") or []),
            "instance_tasks": {
                "total": len(tasks),
                "counts": counts,
                "error_summary": [
                    {"category": key, "error_code": key, "count": value}
                    for key, value in sorted(errors.items())
                ],
            },
        }

    async def _web_support(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        providers = await self.web_repository.list_web_search_providers(profile_id)
        sessions = await self.web_repository.list_web_search_sessions(
            profile_id, instance_id=instance_id, limit=100
        )
        return {
            "providers": [self._web_provider_view(item) for item in providers],
            "sessions": [self._web_session_view(item) for item in sessions],
            "private_query_and_page_text_omitted": True,
        }

    @staticmethod
    def _web_provider_view(item: Any) -> dict[str, Any]:
        kind = str(item.provider_kind).upper()
        return {
            "provider_id": item.provider_id,
            "provider_kind": item.provider_kind,
            "enabled": item.enabled,
            "read_enabled": item.read_enabled,
            "supports_image_search": kind in {"TAVILY", "BOCHA", "BRAVE", "FIRECRAWL", "BAIDU_AI"},
            "priority": item.priority,
            "credential_configured": bool(item.credential_id),
        }

    @staticmethod
    def _web_session_view(item: Any) -> dict[str, Any]:
        return {
            "session_id": item.session_id,
            "instance_id": item.instance_id,
            "purpose": item.purpose.value,
            "search_kind": item.search_kind.value,
            "status": item.status.value,
            "provider_count": item.provider_count,
            "result_count": item.result_count,
            "started_at": jsonable(item.started_at),
        }

    async def sticker_support_snapshot(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        stats = await self.sticker_repository.sticker_stats(profile_id, instance_id)
        tasks = await self.ai_repository.list_ai_tasks(
            profile_id=profile_id, instance_id=instance_id, limit=100
        )
        safe_tasks = [
            self._safe_sticker_task(profile_id, instance_id, task)
            for task in tasks
            if str(task.get("task_type") or "") in {"STICKER_COLLECTION", "STICKER_CHECK"}
        ]
        return {
            "counts": stats,
            "tasks": safe_tasks,
            "private_images_paths_hashes_urls_ocr_and_descriptions_omitted": True,
        }

    @staticmethod
    def _safe_sticker_task(
        profile_id: str, instance_id: str, task: Mapping[str, Any]
    ) -> dict[str, str]:
        row = jsonable(task) or {}
        task_id = str(row.get("task_id") or "")
        digest = hashlib.sha256(f"{profile_id}\0{instance_id}\0{task_id}".encode()).hexdigest()[:16]
        return {
            "diagnostic_id": f"stkdiag_{digest}",
            "task_type": str(row.get("task_type") or ""),
            "status": str(row.get("status") or ""),
            "error_category": _support_error_category(row.get("error")),
        }


_SUPPORT_ERROR_CATEGORIES = frozenset(
    {
        *(code.value for code in AIErrorCode),
        "AIInvocationError",
        "AssertionError",
        "BackgroundOutputError",
        "BootstrapRollbackError",
        "CharacterModelError",
        "CommandProtocolError",
        "ConnectionError",
        "ContextBudgetError",
        "CreativeOutputError",
        "FileNotFoundError",
        "ImageGenerationDisabledError",
        "ImageGenerationRequestError",
        "KeyError",
        "OpenAIHTTPStatusError",
        "OpenAITransportError",
        "OSError",
        "ResponsePolishContractError",
        "RuntimeError",
        "RuntimeOwnershipError",
        "SocialSnapshotError",
        "StickerDeliveryPreparationError",
        "TimeoutError",
        "TimerAdmissionFenceError",
        "TimerDomainError",
        "TypeError",
        "ValueError",
        "WebHTTPStatusError",
        "WebImageInspectionError",
        "WebResearchError",
        "WebTransportError",
        "WorkCheckpointError",
        "WorkCheckpointStorageError",
        "WorkContinuityError",
    }
)


def _support_error_category(error: object) -> str:
    raw_error = str(error or "")
    if not raw_error:
        return ""
    candidate = raw_error.partition(":")[0].strip()
    return candidate if candidate in _SUPPORT_ERROR_CATEGORIES else "UNKNOWN"


_SUPPORT_MODEL_CONTENT_KEYS = {
    "checkpoint",
    "content",
    "checkpoint_json",
    "context_ledger",
    "context_text",
    "contact_evidence",
    "foreground_context_notes",
    "input",
    "input_json",
    "logical_prompt",
    "messages",
    "metadata_json",
    "payload",
    "payload_json",
    "plain_text",
    "progress_json",
    "prompt",
    "provider_envelope",
    "provider_request",
    "provider_response",
    "request_json",
    "result",
    "result_json",
    "response_json",
    "progress",
    "turn_text",
    "user_message",
}


def _support_safe(value: Any, *, include_model_content: bool) -> Any:
    """Force secret/binary redaction and omit model/chat bodies by default."""

    safe = prompt_jsonable(value)
    if include_model_content:
        return safe

    def omit_content(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): (
                    "[正文未包含]"
                    if str(key).lower() in _SUPPORT_MODEL_CONTENT_KEYS
                    else omit_content(child)
                )
                for key, child in item.items()
            }
        if isinstance(item, list):
            return [omit_content(child) for child in item]
        return item

    return omit_content(safe)
