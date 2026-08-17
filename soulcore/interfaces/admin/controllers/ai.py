"""Administrator control plane for durable AI work and routing."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from typing import Any, Protocol

from ....features.ai.durable_tasks import DurableAITaskManager
from ....features.ai.ports import AIAdminQueryRepositoryPort
from ....features.ai.service import AIManager
from ....features.delivery.ports import DeliveryRepositoryPort
from ....features.main_core.ports import MainCoreQueryPort
from ....features.profiles.credentials import CredentialVault
from ..presentation import (
    ai_task_run_ids,
    ai_task_view,
    core_run_detail_view,
    jsonable,
    outbox_detail_view,
    positive_int,
)
from .ai_quick_setup import AIQuickSetupController
from .ai_support import AI_CAPABILITIES
from .ai_work_records import AIWorkRecordsController


class AIConfigurationControllerPort(Protocol):
    async def snapshot(self, profile_id: str) -> dict[str, Any]: ...
    async def save_package(self, payload: Mapping[str, Any], profile_id: str) -> dict[str, Any]: ...
    async def save_credential(
        self, payload: Mapping[str, Any], profile_id: str
    ) -> dict[str, Any]: ...
    async def save_model(self, payload: Mapping[str, Any], profile_id: str) -> dict[str, Any]: ...


class AIProbeControllerPort(Protocol):
    async def probe_backend(self, backend_id: str, profile_id: str) -> dict[str, Any]: ...
    async def probe_package(
        self, package_id: str, profile_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]: ...
    async def probe_model(
        self, backend_id: str, profile_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]: ...


ReloadBackends = Callable[[], Awaitable[None]]


class AIAdminController:
    def __init__(
        self,
        *,
        repository: AIAdminQueryRepositoryPort,
        main_core_queries: MainCoreQueryPort,
        delivery_repository: DeliveryRepositoryPort,
        ai_tasks: DurableAITaskManager,
        ai_manager: AIManager,
        credential_vault: CredentialVault,
        configuration: AIConfigurationControllerPort,
        probes: AIProbeControllerPort,
        reload_backends: ReloadBackends,
    ) -> None:
        self.repository = repository
        self.main_core_queries = main_core_queries
        self.delivery_repository = delivery_repository
        self.ai_tasks = ai_tasks
        self.ai_manager = ai_manager
        self.credential_vault = credential_vault
        self.configuration = configuration
        self.probes = probes
        self.reload_backends = reload_backends
        self.ai_work_records = AIWorkRecordsController(repository)
        self.quick_setup = AIQuickSetupController(
            repository,
            configuration,
            probes,
            self._save_capability_pool,
            getattr(configuration, "runtime_context", None),
        )

    async def probe_ai_backend(self, backend_id: str, *, profile_id: str) -> dict[str, Any]:
        return await self.probes.probe_backend(backend_id, profile_id)

    async def handle(
        self, method: str, profile_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        handlers: dict[str, Callable[[], Awaitable[dict[str, Any]]]] = {
            "ai_manager": lambda: self.ai_manager_snapshot(profile_id),
            "ai_work_records": lambda: self._ai_work_records(profile_id, payload),
            "ai_work_record": lambda: self._ai_work_record(profile_id, payload),
            "ai_work_attempt_debug": lambda: self._ai_work_attempt_debug(profile_id, payload),
            "ai_work_attempt_raw": lambda: self._ai_work_attempt_raw(profile_id, payload),
            "ai_pause": lambda: self._pause(payload),
            "ai_quick_setup_snapshot": lambda: self.quick_setup.snapshot(profile_id),
            "ai_quick_setup_configure": lambda: self.quick_setup.configure(profile_id, payload),
            "ai_api_packages": lambda: self.configuration.snapshot(profile_id),
            "ai_api_package": lambda: self.configuration.save_package(payload, profile_id),
            "ai_api_package_credential": lambda: self.configuration.save_credential(
                payload, profile_id
            ),
            "ai_api_model": lambda: self._save_model_action(payload, profile_id),
            "ai_capability_pool": lambda: self._save_capability_pool(payload, profile_id),
            "ai_api_package_probe": lambda: self.probes.probe_package(
                str(payload.get("package_id") or "").strip(), profile_id, payload
            ),
            "ai_api_model_probe": lambda: self.probes.probe_model(
                self._model_probe_id(payload), profile_id, payload
            ),
            "ai_backend_probe": lambda: self.probes.probe_backend(
                str(payload.get("backend_id") or "").strip(), profile_id
            ),
        }
        handler = handlers.get(method)
        if handler is None:
            raise ValueError(f"unknown AI manager action: {method}")
        return await handler()

    async def _save_model_action(
        self, payload: Mapping[str, Any], profile_id: str
    ) -> dict[str, Any]:
        if str(payload.get("action") or "").strip().lower() == "reorder_capability_pool":
            return await self._save_capability_pool(payload, profile_id)
        return await self.configuration.save_model(payload, profile_id)

    async def _ai_work_records(self, profile_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return await self.ai_work_records.list(profile_id, payload)

    async def _ai_work_record(self, profile_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return await self.ai_work_records.detail(profile_id, payload)

    async def _ai_work_attempt_raw(
        self, profile_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        return await self.ai_work_records.raw(profile_id, payload)

    async def _ai_work_attempt_debug(
        self, profile_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        return await self.ai_work_records.debug(profile_id, payload)

    async def _tasks(self, profile_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        limit = max(1, min(int(payload.get("limit") or 100), 500))
        rows = await self.repository.list_ai_tasks(
            profile_id=profile_id,
            instance_id=str(payload.get("instance_id") or "").strip() or None,
            status=str(payload.get("status") or "").strip() or None,
            task_type=str(payload.get("task_type") or "").strip() or None,
            limit=limit,
        )
        backend_id = str(payload.get("backend_id") or "").strip()
        if backend_id:
            rows = [row for row in rows if row.get("backend_id") == backend_id]
        return {"tasks": [ai_task_view(row) for row in rows], "next_cursor": None}

    async def _task(self, profile_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        task_id = int(payload.get("task_id") or 0)
        task = await self.repository.get_ai_task(task_id)
        if task is None or task.get("profile_id") != profile_id:
            raise ValueError("unknown AI task")
        related = await self._related_runtime(profile_id, task_id, task)
        return {
            "task": ai_task_view(task),
            "execution": jsonable(
                {key: task.get(key) for key in ("step_key", "checkpoint", "progress", "result")}
            ),
            "timeline": jsonable(
                await self.repository.list_ai_task_audit(
                    task_id=task_id, profile_id=profile_id, limit=200
                )
            ),
            "attempts": jsonable(await self.repository.list_ai_task_attempts(task_id, limit=100)),
            "related": related,
        }

    async def _related_runtime(
        self, profile_id: str, task_id: int, task: Mapping[str, Any]
    ) -> dict[str, Any]:
        instance_id = str(task.get("instance_id") or "").strip()
        if not instance_id:
            return {"core_runs": [], "outbox": []}
        runs = await self.main_core_queries.list_instance_runs(profile_id, instance_id, limit=500)
        run_ids = ai_task_run_ids(task)
        for run in runs:
            run_id = self._referenced_run_id(run, task_id)
            if run_id:
                run_ids.add(run_id)
        related_runs = [row for row in runs if positive_int(row.get("run_id")) in run_ids]
        outbox = (
            await self.delivery_repository.list_instance_outbox(profile_id, instance_id, limit=500)
            if run_ids
            else []
        )
        prefixes = tuple(
            prefix
            for run_id in sorted(run_ids)
            for prefix in (f"core-run:{run_id}:", f"instance-run:{run_id}:")
        )
        return {
            "core_runs": [core_run_detail_view(row) for row in related_runs],
            "outbox": [
                outbox_detail_view(item)
                for item in outbox
                if str(item.idempotency_key or "").startswith(prefixes)
            ],
        }

    @staticmethod
    def _referenced_run_id(run: Mapping[str, Any], task_id: int) -> int | None:
        request = run.get("request")
        if not isinstance(request, Mapping):
            return None
        metadata = request.get("metadata")
        if not isinstance(metadata, Mapping):
            return None
        if positive_int(metadata.get("ai_task_id")) != task_id:
            return None
        return positive_int(run.get("run_id"))

    async def _task_action(self, profile_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        task_id = int(payload.get("task_id") or 0)
        current = await self.repository.get_ai_task(task_id)
        if current is None or current.get("profile_id") != profile_id:
            raise ValueError("unknown AI task")
        expected = payload.get("expected_version")
        version = None if expected is None or expected == "" else int(str(expected))
        if version is not None and version != int(current.get("version") or 1):
            raise ValueError("AI task version conflict; refresh before retrying")
        result = await self._perform_task_action(
            task_id,
            str(payload.get("action") or "").strip().lower(),
            str(payload.get("reason") or "").strip(),
            version,
        )
        if result is None:
            raise ValueError("AI task is not in a state that accepts this action")
        return {"ok": True, "task": jsonable(result)}

    async def _perform_task_action(
        self, task_id: int, action: str, reason: str, version: int | None
    ) -> Any:
        actor = "astrbot-admin-page"
        if action == "pause":
            return await self.ai_tasks.pause(
                task_id, actor_id=actor, reason=reason, expected_version=version
            )
        if action == "resume":
            return await self.ai_tasks.resume(task_id, actor_id=actor, expected_version=version)
        if action == "cancel":
            return await self.ai_tasks.cancel(
                task_id, actor_id=actor, reason=reason, expected_version=version
            )
        if action == "retry":
            return await self.ai_tasks.manual_retry(
                task_id, actor_id=actor, expected_version=version
            )
        raise ValueError("action must be pause, resume, cancel or retry")

    async def _pause(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        mapping = {
            "GLOBAL": "GLOBAL",
            "BACKGROUND": "BACKGROUND",
            "BACKEND": "BACKEND",
            "CAPABILITY": "CAPABILITY",
        }
        raw = str(payload.get("scope_type") or "").upper()
        scope = mapping.get(raw)
        if scope is None:
            raise ValueError("unsupported AI pause scope")
        key = str(payload.get("scope_id") or "").strip()
        paused = bool(payload.get("paused", True))
        result = await self.repository.set_ai_manager_pause(
            scope,
            scope_key=key,
            paused=paused,
            reason=str(payload.get("reason") or "").strip(),
            actor_id="astrbot-admin-page",
        )
        cancelled = self.ai_manager.emergency_stop() if scope == "GLOBAL" and paused else 0
        return {"ok": True, "pause": jsonable(result), "cancel_requested_invocations": cancelled}

    async def _save_capability_pool(
        self, payload: Mapping[str, Any], profile_id: str
    ) -> dict[str, Any]:
        capability = str(payload.get("capability") or "").strip().lower()
        if capability not in AI_CAPABILITIES:
            raise ValueError("unsupported AI capability")
        backend_ids = self._pool_backend_ids(payload)
        by_id = self._profile_backends(await self.repository.list_ai_backends(), profile_id)
        self._validate_pool_backends(backend_ids, by_id, capability)
        await self._replace_capability_members(
            capability,
            backend_ids,
            payload,
            managed_backend_ids=set(by_id),
        )
        await self.reload_backends()
        return {"ok": True, "capability": capability, "backend_ids": backend_ids}

    @staticmethod
    def _pool_backend_ids(payload: Mapping[str, Any]) -> list[str]:
        return [str(item).strip() for item in payload.get("backend_ids") or [] if str(item).strip()]

    @staticmethod
    def _profile_backends(
        backends: list[Mapping[str, Any]],
        profile_id: str,
    ) -> dict[str, Mapping[str, Any]]:
        return {
            str(row["backend_id"]): row
            for row in backends
            if str(dict(row.get("metadata") or {}).get("profile_id") or "") == profile_id
        }

    def _validate_pool_backends(
        self,
        backend_ids: list[str],
        by_id: Mapping[str, Mapping[str, Any]],
        capability: str,
    ) -> None:
        unknown = [item for item in backend_ids if item not in by_id]
        if unknown:
            raise ValueError("capability pool contains an unknown backend")
        incompatible = [
            item
            for item in backend_ids
            if not self._backend_supports_capability(by_id[item], capability)
        ]
        if incompatible:
            raise ValueError(
                "capability pool contains incompatible backends: " + ", ".join(incompatible)
            )

    async def _replace_capability_members(
        self,
        capability: str,
        backend_ids: list[str],
        payload: Mapping[str, Any],
        *,
        managed_backend_ids: set[str],
    ) -> None:
        selected = set(backend_ids)
        for row in await self.repository.list_ai_capability_pool(capability):
            if row["backend_id"] in managed_backend_ids and row["backend_id"] not in selected:
                await self.repository.upsert_ai_capability_pool(
                    capability,
                    row["backend_id"],
                    priority=int(row.get("priority") or 0),
                    enabled=False,
                    config=dict(row.get("config") or {}),
                )
        for index, backend_id in enumerate(backend_ids):
            await self.repository.upsert_ai_capability_pool(
                capability,
                backend_id,
                priority=index + 1,
                enabled=bool(payload.get("enabled", True)),
            )

    @staticmethod
    def _backend_supports_capability(backend: Mapping[str, Any], capability: str) -> bool:
        metadata = dict(backend.get("metadata") or {})
        declared = {
            str(item).strip().lower()
            for item in metadata.get("capabilities", ())
            if str(item).strip()
        }
        wanted = str(capability).strip().lower()
        if wanted in declared:
            return True
        return wanted in declared

    async def ai_manager_snapshot(self, profile_id: str) -> dict[str, Any]:
        tasks = await self.repository.list_ai_tasks(profile_id=profile_id, limit=1000)
        counts, errors = self._task_counts(tasks)
        await self._add_invocation_errors(profile_id, errors)
        pauses = await self.repository.list_ai_manager_pauses()
        backends = await self._backend_views(pauses)
        pools = await self._capability_views()
        return {
            "overview": self._overview(tasks, counts, pauses),
            "backends": backends,
            "capability_pools": pools,
            "pauses": jsonable(pauses),
            "error_summary": [
                {"category": key, "error_code": key, "count": value}
                for key, value in sorted(errors.items())
            ],
            "capability_options": list(AI_CAPABILITIES),
        }

    @staticmethod
    def _task_counts(tasks: list[Mapping[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
        counts: dict[str, int] = {}
        errors: dict[str, int] = {}
        for task in tasks:
            status = str(task.get("status") or "UNKNOWN")
            counts[status] = counts.get(status, 0) + 1
            if task.get("last_error"):
                code = str(task.get("error_code") or "UNKNOWN")
                errors[code] = errors.get(code, 0) + 1
        return counts, errors

    async def _add_invocation_errors(self, profile_id: str, errors: dict[str, int]) -> None:
        for workflow in await self.repository.list_ai_workflow_summaries(
            profile_id=profile_id, limit=100
        ):
            code = str(workflow.get("final_error_code") or "").strip()
            if code:
                errors[code] = errors.get(code, 0) + 1

    async def _backend_views(self, pauses: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        paused = {
            str(item.get("scope_key") or "")
            for item in pauses
            if item.get("pause_scope") == "BACKEND" and item.get("paused")
        }
        return [self._backend_view(row, paused) for row in await self.repository.list_ai_backends()]

    def _backend_view(self, backend: Mapping[str, Any], paused: set[str]) -> dict[str, Any]:
        metadata = dict(backend.get("metadata") or {})
        credential = self._backend_credential(backend)
        source = "direct"
        return {
            **jsonable(backend),
            "source": source,
            "editable": True,
            "priority": int(metadata.get("priority") or 100),
            "status": backend.get("health_status") or "UNKNOWN",
            "capabilities": list(metadata.get("capabilities") or []),
            "model": metadata.get("model") or "",
            "base_url": metadata.get("base_url") or "",
            "adapter_config": metadata.get("adapter_config") or {},
            "reference_image": metadata.get("reference_image"),
            "multiple_references": metadata.get("multiple_references"),
            "maximum_outputs": metadata.get("maximum_outputs") or 5,
            "paused": backend["backend_id"] in paused,
            **credential,
        }

    def _backend_credential(self, backend: Mapping[str, Any]) -> dict[str, Any]:
        return self._credential_snapshot(str(backend.get("credential_id") or ""))

    def _credential_snapshot(self, credential_id: str) -> dict[str, Any]:
        if not credential_id:
            return {
                "credential_configured": False,
                "credential_last4": "",
                "credential_source": "missing",
                "credential_error": "missing_reference",
            }
        try:
            info = self.credential_vault.describe(credential_id)
        except (TypeError, ValueError):
            return {
                "credential_configured": False,
                "credential_last4": "",
                "credential_source": "invalid",
                "credential_error": "invalid_reference",
            }
        return {
            "credential_configured": info.configured,
            "credential_last4": info.last4,
            "credential_source": info.source,
            "credential_error": "" if info.configured else "not_configured",
        }

    async def _capability_views(self) -> list[dict[str, Any]]:
        result = []
        for capability in AI_CAPABILITIES:
            members = await self.repository.list_ai_capability_pool(capability)
            result.append(
                {
                    "capability": capability,
                    "enabled": any(bool(item.get("enabled")) for item in members),
                    "backend_ids": [item["backend_id"] for item in members if item.get("enabled")],
                    "members": jsonable(members),
                }
            )
        return result

    @staticmethod
    def _overview(
        tasks: list[Mapping[str, Any]],
        counts: Mapping[str, int],
        pauses: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        now = datetime.now().astimezone()
        failed_24h = sum(1 for task in tasks if AIAdminController._failed_recently(task, now))
        return {
            "total": len(tasks),
            "running": counts.get("RUNNING", 0),
            "queued": sum(counts.get(item, 0) for item in ("READY", "SCHEDULED", "RETRY_WAIT")),
            "pending": counts.get("READY", 0) + counts.get("SCHEDULED", 0),
            "retrying": counts.get("RETRY_WAIT", 0),
            "blocked": counts.get("RECOVERY_REQUIRED", 0) + counts.get("PAUSED", 0),
            "paused": counts.get("PAUSED", 0),
            "failed": counts.get("FAILED", 0),
            "failed_24h": failed_24h,
            "recovery_required": counts.get("RECOVERY_REQUIRED", 0),
            "background_paused": any(
                item.get("pause_scope") == "BACKGROUND" and item.get("paused") for item in pauses
            ),
            "global_paused": any(
                item.get("pause_scope") == "GLOBAL" and item.get("paused") for item in pauses
            ),
        }

    @staticmethod
    def _failed_recently(task: Mapping[str, Any], now: datetime) -> bool:
        if task.get("status") != "FAILED" or not task.get("finished_at"):
            return False
        try:
            finished = datetime.fromisoformat(str(task["finished_at"]))
            finished = finished.replace(tzinfo=now.tzinfo) if finished.tzinfo is None else finished
            return (now - finished.astimezone(now.tzinfo)).total_seconds() <= 86400
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _model_probe_id(payload: Mapping[str, Any]) -> str:
        return str(payload.get("backend_id") or "").strip()
