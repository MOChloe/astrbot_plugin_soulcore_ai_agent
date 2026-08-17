"""Administrator controller for profile-scoped web providers and probes."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from typing import Any

from ....contracts.ai_models import (
    AICapabilityEffect,
    AICapabilityRequest,
    AIExecutionMode,
    AIRetryPolicy,
    AIWorkPurpose,
)
from ....contracts.web import WebDiagnosticOverride
from ....features.profiles.credentials import CredentialVault
from ....features.profiles.ports import ProfilesRepositoryPort
from ....features.web.domain import WebSearchProviderRecord
from ....features.web.ports import WebAIManagerPort, WebRepositoryPort
from ..presentation import jsonable

ReloadAI = Callable[[], Awaitable[None]]

_QUICK_SETUP_PROVIDER_KINDS = {
    "TAVILY": ("Tavily", True),
    "BOCHA": ("博查", False),
    "BRAVE": ("Brave Search", False),
    "FIRECRAWL": ("Firecrawl", True),
    "BAIDU_AI": ("百度 AI 搜索", False),
    "EXA": ("Exa", True),
}


class WebAdminController:
    def __init__(
        self,
        repository: WebRepositoryPort,
        profiles_repository: ProfilesRepositoryPort,
        ai_manager: WebAIManagerPort,
        credential_vault: CredentialVault,
        reload_ai_backends: ReloadAI,
    ) -> None:
        self.repository = repository
        self.profiles_repository = profiles_repository
        self.ai_manager = ai_manager
        self.credential_vault = credential_vault
        self.reload_ai_backends = reload_ai_backends

    async def handle(
        self,
        method: str,
        profile_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        profile = await self.profiles_repository.get_profile(profile_id)
        if profile is None:
            raise ValueError("找不到当前 AstrBot 配置档案")
        if method == "get_web_config":
            return self._config_view(profile)
        if method == "quick_setup_snapshot":
            return await self._quick_setup_snapshot(profile)
        if method == "quick_setup_configure":
            return await self._quick_setup_configure(profile, payload)
        if method in {"save_web_config", "web_config"}:
            return await self._save_config(profile, payload)
        if method == "web_providers":
            return await self._providers(profile_id)
        if method == "web_provider":
            return await self._provider_action(profile_id, payload)
        if method == "web_provider_credential":
            return await self._save_credential(profile_id, payload)
        if method in {"web_provider_probe", "web_image_probe", "web_read_test"}:
            return await self._probe(
                method,
                profile_id,
                payload,
                diagnostic_override=WebDiagnosticOverride.ADMIN_PROVIDER_PROBE,
            )
        if method == "web_search_test":
            raise ValueError("请在具体联网接口上使用“测试搜索”")
        if method == "web_snapshot":
            return await self._snapshot(profile_id, payload)
        raise ValueError(f"unknown web page action: {method}")

    @staticmethod
    def _config_view(value: Any) -> dict[str, Any]:
        return {
            "enabled": bool(value.web_search_enabled),
            "intensity": str(value.web_search_intensity),
            "version": str(value.updated_at.isoformat() if value.updated_at else "0"),
        }

    async def _save_config(self, current: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
        profile_id = str(current.profile_id)
        updated = await self.repository.update_web_search_settings(
            profile_id,
            enabled=payload.get("enabled"),
            intensity=payload.get("intensity"),
        )
        return self._config_view(updated)

    async def _provider_view(self, row: Any) -> dict[str, Any]:
        info = self.credential_vault.describe(str(row.credential_id)) if row.credential_id else None
        status = (
            "DISABLED"
            if not row.enabled
            else "CONFIGURED"
            if info and info.configured
            else "UNCONFIGURED"
        )
        kind = str(row.provider_kind).upper()
        supports_images = kind in {
            "TAVILY",
            "BOCHA",
            "BRAVE",
            "FIRECRAWL",
            "BAIDU_AI",
        }
        return {
            **jsonable(row),
            "supports_image_search": supports_images,
            "status": status,
            "credential_configured": bool(info and info.configured),
            "credential_hint": f"•••• {info.last4}" if info and info.last4 else "未配置",
            "credential_mode": "env_name" if info and info.source == "env" else "secret",
            "env_name": info.reference if info and info.source == "env" else "",
        }

    async def _providers(self, profile_id: str) -> dict[str, Any]:
        rows = await self.repository.list_web_search_providers(profile_id)
        views = [await self._provider_view(row) for row in rows]
        enabled = [item for item in views if item["enabled"]]
        return {
            "providers": views,
            "effective_orders": {
                "web": enabled,
                "images": [item for item in enabled if item["supports_image_search"]],
            },
        }

    async def _quick_setup_snapshot(self, profile: Any) -> dict[str, Any]:
        """Return only the provider facts needed by the guided setup."""

        profile_id = str(profile.profile_id)
        providers = [
            await self._provider_view(item)
            for item in await self.repository.list_web_search_providers(profile_id)
        ]
        configured = [item for item in providers if bool(item["credential_configured"])]
        ready = [item for item in configured if bool(item["enabled"])]
        preferred = next(
            (item for item in ready if bool(profile.web_search_enabled)),
            ready[0] if ready else configured[0] if configured else None,
        )
        quick_provider_id = await self._quick_setup_provider_id(profile_id)
        draft = next(
            (item for item in providers if str(item["provider_id"]) == quick_provider_id),
            None,
        )
        return {
            "enabled": bool(profile.web_search_enabled),
            "configured": bool(configured),
            "active": bool(profile.web_search_enabled and ready),
            "provider_count": len(configured),
            "provider": self._guided_provider_view(preferred),
            "draft_provider": self._guided_provider_view(draft),
            "quick_provider_id": quick_provider_id,
            "uses_main_model": True,
        }

    async def _quick_setup_configure(
        self, profile: Any, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Apply one simple web choice without exposing provider administration."""

        profile_id = str(profile.profile_id)
        action = str(payload.get("action") or "configure").strip().lower()
        if action == "disable":
            await self.repository.update_web_search_settings(
                profile_id, enabled=False, intensity=None
            )
            return {"ok": True, "applied": True, "action": action, "probe": None}
        if action not in {"configure", "use_existing"}:
            raise ValueError("快速设置中的联网操作无效")
        if not bool(payload.get("confirm_cost")):
            raise ValueError("联网测试会发起一次真实查询，请先确认")
        if action == "use_existing":
            return await self._quick_setup_use_existing(profile_id, payload)
        return await self._quick_setup_create(profile_id, payload)

    async def _quick_setup_use_existing(
        self, profile_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        provider_id = str(payload.get("provider_id") or "").strip()
        row = await self.repository.get_web_search_provider(profile_id, provider_id)
        info = self.credential_vault.describe(str(row.credential_id)) if row else None
        if row is None or info is None or not info.configured:
            return self._quick_setup_failure(action="use_existing", error="原有联网接口已不可用")
        was_enabled = bool(row.enabled)
        current = row
        try:
            if not was_enabled:
                current = await self.repository.set_web_search_provider_enabled(
                    profile_id,
                    provider_id,
                    True,
                    expected_version=int(row.version),
                )
                await self.reload_ai_backends()
            probe = await self._probe(
                "web_provider_probe",
                profile_id,
                {"provider_id": provider_id, "confirm_cost": True},
                diagnostic_override=WebDiagnosticOverride.ADMIN_PROVIDER_PROBE,
            )
            await self.repository.update_web_search_settings(
                profile_id, enabled=True, intensity=None
            )
        except Exception as exc:
            if not was_enabled:
                await self._restore_provider_enabled(profile_id, current, False)
            return self._quick_setup_failure(action="use_existing", error=str(exc))
        return {
            "ok": True,
            "applied": True,
            "action": "use_existing",
            "provider": self._guided_provider_view(await self._provider_view(current)),
            "probe": probe,
        }

    async def _quick_setup_create(
        self, profile_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        provider_kind = str(payload.get("provider_kind") or "TAVILY").strip().upper()
        metadata = _QUICK_SETUP_PROVIDER_KINDS.get(provider_kind)
        secret = str(payload.get("secret") or "")
        if metadata is None:
            raise ValueError("不支持这个联网查询服务")
        if not secret.strip():
            raise ValueError("请填写联网服务的 API Key")

        provider_id = await self._quick_setup_provider_id(profile_id)
        existing = await self.repository.get_web_search_provider(profile_id, provider_id)
        before = replace(existing, config=dict(existing.config)) if existing is not None else None
        credential_backup = (
            self._credential_backup(str(existing.credential_id)) if existing else None
        )
        saved: Any | None = None
        try:
            saved = await self._upsert_provider(
                profile_id,
                provider_id,
                int(existing.version) if existing is not None else None,
                {
                    "provider_kind": provider_kind,
                    "display_name": metadata[0],
                    "priority": 1,
                    "enabled": True,
                    "read_enabled": metadata[1],
                },
            )
            self.credential_vault.set_secret(str(saved.credential_id), secret)
            await self.reload_ai_backends()
            probe = await self._probe(
                "web_provider_probe",
                profile_id,
                {"provider_id": provider_id, "confirm_cost": True},
                diagnostic_override=WebDiagnosticOverride.ADMIN_PROVIDER_PROBE,
            )
            await self.repository.update_web_search_settings(
                profile_id, enabled=True, intensity=None
            )
        except Exception as exc:
            await self._rollback_quick_setup_provider(
                profile_id,
                before=before,
                saved=saved,
                credential_backup=credential_backup,
            )
            return self._quick_setup_failure(
                action="configure",
                error=str(exc),
                redaction_value=secret,
            )
        return {
            "ok": True,
            "applied": True,
            "action": "configure",
            "provider": self._guided_provider_view(await self._provider_view(saved)),
            "probe": probe,
        }

    async def _rollback_quick_setup_provider(
        self,
        profile_id: str,
        *,
        before: WebSearchProviderRecord | None,
        saved: Any | None,
        credential_backup: tuple[str, str] | None,
    ) -> None:
        if saved is None:
            return
        current = await self.repository.get_web_search_provider(profile_id, str(saved.provider_id))
        if current is None:
            return
        if before is None:
            if current.enabled:
                await self.repository.set_web_search_provider_enabled(
                    profile_id,
                    str(current.provider_id),
                    False,
                    expected_version=int(current.version),
                )
            self.credential_vault.delete(str(current.credential_id))
        else:
            await self.repository.upsert_web_search_provider(
                before, expected_version=int(current.version)
            )
            self._restore_credential(str(before.credential_id), credential_backup)
        await self.reload_ai_backends()

    async def _restore_provider_enabled(self, profile_id: str, row: Any, enabled: bool) -> None:
        current = await self.repository.get_web_search_provider(profile_id, str(row.provider_id))
        if current is not None and bool(current.enabled) != enabled:
            await self.repository.set_web_search_provider_enabled(
                profile_id,
                str(current.provider_id),
                enabled,
                expected_version=int(current.version),
            )
            await self.reload_ai_backends()

    def _credential_backup(self, credential_id: str) -> tuple[str, str]:
        info = self.credential_vault.describe(credential_id)
        if info.source == "env":
            return ("env", str(info.reference))
        if info.source == "file" and info.configured:
            return ("file", self.credential_vault.resolve(credential_id))
        return ("missing", "")

    def _restore_credential(self, credential_id: str, backup: tuple[str, str] | None) -> None:
        mode, value = backup or ("missing", "")
        if mode == "env":
            self.credential_vault.set_env_reference(credential_id, value)
        elif mode == "file":
            self.credential_vault.set_secret(credential_id, value)
        else:
            self.credential_vault.delete(credential_id)

    async def _quick_setup_provider_id(self, profile_id: str) -> str:
        digest = hashlib.sha256(profile_id.encode("utf-8")).hexdigest()[:20]
        base = f"web-quick:{digest}"
        rows = await self.repository.list_web_search_providers(profile_id, include_archived=True)
        active = [
            str(item.provider_id)
            for item in rows
            if item.archived_at is None
            and (str(item.provider_id) == base or str(item.provider_id).startswith(f"{base}:"))
        ]
        if active:
            return sorted(active)[0]
        used = {str(item.provider_id) for item in rows}
        if base not in used:
            return base
        index = 2
        while f"{base}:{index}" in used:
            index += 1
        return f"{base}:{index}"

    @staticmethod
    def _guided_provider_view(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return {
            "provider_id": str(value.get("provider_id") or ""),
            "provider_kind": str(value.get("provider_kind") or "").upper(),
            "display_name": str(value.get("display_name") or ""),
            "enabled": bool(value.get("enabled")),
            "version": int(value.get("version") or 0),
            "credential_configured": bool(value.get("credential_configured")),
            "credential_hint": str(value.get("credential_hint") or ""),
            "supports_image_search": bool(value.get("supports_image_search")),
            "supports_page_reading": bool(value.get("read_enabled")),
        }

    @staticmethod
    def _quick_setup_failure(
        *, action: str, error: str, redaction_value: str = ""
    ) -> dict[str, Any]:
        message = str(error or "联网接口连接测试失败")
        if redaction_value:
            message = message.replace(redaction_value, "[已隐藏]")
        return {
            "ok": True,
            "applied": False,
            "action": action,
            "probe": {"ok": False, "error": message},
            "can_skip": True,
        }

    async def _provider_action(self, profile_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action") or "").lower()
        provider_id = str(payload.get("provider_id") or "").strip()
        expected_raw = payload.get("expected_version")
        expected = int(expected_raw) if expected_raw not in (None, "", 0, "0") else None
        if action == "archive":
            if not bool(payload.get("confirm")):
                raise ValueError("归档联网接口前需要二次确认")
            row = await self.repository.archive_web_search_provider(
                profile_id, provider_id, expected_version=expected
            )
        elif action in {"enable", "disable"}:
            row = await self.repository.set_web_search_provider_enabled(
                profile_id, provider_id, action == "enable", expected_version=expected
            )
        elif action in {"create", "update"}:
            row = await self._upsert_provider(profile_id, provider_id, expected, payload)
        else:
            raise ValueError("不支持的联网接口操作")
        await self.reload_ai_backends()
        view = await self._provider_view(row)
        return {"ok": True, "provider_id": row.provider_id, "version": row.version, **view}

    async def _upsert_provider(
        self,
        profile_id: str,
        provider_id: str,
        expected_version: int | None,
        payload: Mapping[str, Any],
    ) -> Any:
        existing = (
            await self.repository.get_web_search_provider(profile_id, provider_id)
            if provider_id
            else None
        )
        provider_id = provider_id or f"web-provider:{uuid.uuid4().hex}"
        provider_kind = str(payload.get("provider_kind") or "TAVILY").upper()
        credential_id = str(
            existing.credential_id if existing else f"web-provider-{uuid.uuid4().hex}"
        )
        record = WebSearchProviderRecord(
            provider_id=provider_id,
            profile_id=profile_id,
            provider_kind=provider_kind,
            display_name=str(payload.get("display_name") or provider_kind).strip(),
            backend_id=str(existing.backend_id if existing else f"web:{uuid.uuid4().hex}"),
            credential_id=credential_id,
            priority=max(1, int(payload.get("priority") or 1)),
            enabled=bool(payload.get("enabled", True)),
            read_enabled=bool(payload.get("read_enabled", False)),
            config=dict(existing.config if existing else {}),
        )
        return await self.repository.upsert_web_search_provider(
            record, expected_version=expected_version
        )

    async def _save_credential(self, profile_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        provider_id = str(payload.get("provider_id") or "").strip()
        row = await self.repository.get_web_search_provider(profile_id, provider_id)
        if row is None:
            raise ValueError("找不到该联网接口")
        mode = str(payload.get("credential_mode") or "secret")
        if mode == "env_name":
            info = self.credential_vault.set_env_reference(
                row.credential_id, str(payload.get("env_name") or "")
            )
        else:
            info = self.credential_vault.set_secret(
                row.credential_id, str(payload.get("secret") or "")
            )
        await self.reload_ai_backends()
        return {
            "ok": True,
            "configured": info.configured,
            "last4": info.last4,
            "source": info.source,
        }

    async def _probe(
        self,
        method: str,
        profile_id: str,
        payload: Mapping[str, Any],
        *,
        diagnostic_override: WebDiagnosticOverride,
    ) -> dict[str, Any]:
        if diagnostic_override is not WebDiagnosticOverride.ADMIN_PROVIDER_PROBE:
            raise ValueError("联网接口诊断缺少明确的管理端授权")
        if not bool(payload.get("confirm_cost")):
            raise ValueError("测试会产生一次真实 API 调用，请先确认")
        provider_id = str(payload.get("provider_id") or "").strip()
        row = await self.repository.get_web_search_provider(profile_id, provider_id)
        if row is None or not row.enabled:
            raise ValueError("联网接口不存在或未启用")
        capability = "web.image_search" if method == "web_image_probe" else "web.search"
        searched = await self.ai_manager.invoke_capability(
            self._probe_request(
                capability,
                profile_id,
                provider_id,
                row.backend_id,
                diagnostic_override=diagnostic_override,
            )
        )
        output = searched.output
        if method in {"web_provider_probe", "web_image_probe"}:
            return {"ok": True, "results": [jsonable(item) for item in output.items]}
        items = list(output.items)
        if not row.read_enabled or not items:
            raise ValueError("该接口未启用正文读取，或测试搜索没有返回可读页面")
        read = await self.ai_manager.invoke_capability(
            self._read_probe_request(
                profile_id,
                provider_id,
                row.backend_id,
                items[0].url,
                diagnostic_override=diagnostic_override,
            )
        )
        return {"ok": True, "results": [jsonable(read.output)]}

    @staticmethod
    def _probe_request(
        capability: str,
        profile_id: str,
        provider_id: str,
        backend_id: str,
        *,
        diagnostic_override: WebDiagnosticOverride,
    ) -> AICapabilityRequest:
        return AICapabilityRequest(
            invocation_id=f"web-probe:{uuid.uuid4().hex}",
            capability=capability,
            work_purpose=AIWorkPurpose.ADMIN_WEB_TEST,
            logical_stage_key=f"web-probe:{provider_id}:{uuid.uuid4().hex}",
            payload={
                "query": "SoulCore connectivity test",
                "depth": "quick",
                "freshness": "auto",
                "max_results": 2,
            },
            backend_ids=(str(backend_id),),
            effect=AICapabilityEffect.READ_ONLY,
            execution_mode=AIExecutionMode.FOREGROUND_SYNC,
            profile_id=profile_id,
            owner_kind="WEB_PROBE",
            owner_id=provider_id,
            idempotency_key=f"web-probe:{uuid.uuid4().hex}",
            retry_policy=AIRetryPolicy(max_attempts=1, backend_timeout_seconds=300),
            metadata={"diagnostic_override": diagnostic_override.value},
        )

    @staticmethod
    def _read_probe_request(
        profile_id: str,
        provider_id: str,
        backend_id: str,
        url: str,
        *,
        diagnostic_override: WebDiagnosticOverride,
    ) -> AICapabilityRequest:
        return AICapabilityRequest(
            invocation_id=f"web-read-probe:{uuid.uuid4().hex}",
            capability="web.read",
            work_purpose=AIWorkPurpose.ADMIN_WEB_TEST,
            logical_stage_key=f"web-read-probe:{provider_id}:{uuid.uuid4().hex}",
            payload={
                "url": str(url),
                "focus": "SoulCore connectivity",
                "max_characters": 1500,
            },
            backend_ids=(str(backend_id),),
            effect=AICapabilityEffect.READ_ONLY,
            execution_mode=AIExecutionMode.FOREGROUND_SYNC,
            profile_id=profile_id,
            owner_kind="WEB_PROBE",
            owner_id=provider_id,
            idempotency_key=f"web-read-probe:{uuid.uuid4().hex}",
            retry_policy=AIRetryPolicy(max_attempts=1, backend_timeout_seconds=300),
            metadata={"diagnostic_override": diagnostic_override.value},
        )

    async def _snapshot(self, profile_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        page = max(1, int(payload.get("page") or 1))
        page_size = max(1, min(50, int(payload.get("page_size") or 10)))
        status = str(payload.get("status") or "").strip() or None
        rows = await self.repository.list_web_search_sessions(
            profile_id, status=status, limit=page_size, offset=(page - 1) * page_size
        )
        total = await self.repository.count_web_search_sessions(profile_id)
        providers = [
            await self._provider_view(item)
            for item in await self.repository.list_web_search_providers(profile_id)
        ]
        sessions = [
            {
                **jsonable(item),
                "query_preview": str(item.query)[:80] if item.query else "已按保留规则清理",
                "provider_summary": f"{item.result_count} 个结果",
                "created_at": item.started_at,
            }
            for item in rows
        ]
        return {
            "summary": {
                "ready_providers": sum(1 for item in providers if item["enabled"]),
                "running": sum(1 for item in rows if str(item.status.value) == "RUNNING"),
                "sessions_24h": total,
                "partial": sum(1 for item in rows if str(item.status.value) == "PARTIAL"),
                "total": total,
            },
            "providers": providers,
            "sessions": sessions,
            "results": [],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": max(1, (total + page_size - 1) // page_size),
            },
        }
