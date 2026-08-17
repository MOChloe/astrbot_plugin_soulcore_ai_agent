"""Timeline, controlled-bridge and conversation-context administration."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from ....contracts.runtime_limits import DURABLE_AI_MAX_ATTEMPTS
from ....features.ai.ports import DurableTaskRepositoryPort
from ....features.conversation.ports import ConversationRepositoryPort
from ....features.conversation.service import ConversationContextService
from ....features.delivery.ports import DeliveryRepositoryPort
from ....features.main_core.service import MainCoreRunner
from ....features.profiles.ports import ProfilesRepositoryPort
from ....features.timeline.ports import TimelineRepositoryPort
from ...astrbot import DeliveryTransport
from ..presentation import jsonable
from .profiles import ProfilesAdminController


class TimelineAdminController:
    def __init__(
        self,
        profiles_repository: ProfilesRepositoryPort,
        timeline_repository: TimelineRepositoryPort,
        conversation_repository: ConversationRepositoryPort,
        ai_repository: DurableTaskRepositoryPort,
        delivery_repository: DeliveryRepositoryPort,
        profiles: ProfilesAdminController,
        delivery: DeliveryTransport,
        runner: MainCoreRunner,
        context_service: ConversationContextService,
    ) -> None:
        self.profiles_repository = profiles_repository
        self.timeline_repository = timeline_repository
        self.conversation_repository = conversation_repository
        self.ai_repository = ai_repository
        self.delivery_repository = delivery_repository
        self.profiles = profiles
        self.delivery = delivery
        self.runner = runner
        self.context_service = context_service

    async def character_instance_diagnostics(
        self,
        profile_id: str,
        instance_id: str,
        *,
        message_page: int = 1,
        message_page_size: int = 20,
    ) -> dict[str, Any]:
        instance = await self.profiles.require_role_instance(profile_id, instance_id)
        result = self._diagnostic_base()
        await self._add_state_views(result, profile_id, instance_id)
        await self._add_delivery_views(result, profile_id, instance)
        await self._add_message_views(
            result, profile_id, instance_id, message_page, message_page_size
        )
        await self._add_runtime_lists(result, profile_id, instance_id)
        return result

    @staticmethod
    def _diagnostic_base() -> dict[str, Any]:
        return {
            "state": {},
            "runs": [],
            "messages": [],
            "message_stats": {
                "total": 0,
                "inbound": 0,
                "outbound": 0,
                "internal_memo": 0,
                "latest_at": None,
            },
            "message_pagination": {"page": 1, "page_size": 20, "total_pages": 1},
            "outbox": [],
            "wakeups": [],
            "contact_clock": {},
            "delivery_capability": {},
            "qpm": [],
            "doctor": [],
            "state_message_gate": {},
            "character_intents": [],
        }

    async def _add_state_views(
        self, result: dict[str, Any], profile_id: str, instance_id: str
    ) -> None:
        state = await self.profiles_repository.get_instance_state(profile_id, instance_id)
        result["state"] = {
            **(jsonable(state) or {}),
        }
        result["contact_clock"] = (
            jsonable(await self.timeline_repository.get_contact_state(profile_id, instance_id))
            or {}
        )
        await self._add_controlled_bridge_views(result, profile_id, instance_id)

    async def _add_controlled_bridge_views(
        self, result: dict[str, Any], profile_id: str, instance_id: str
    ) -> None:
        result["state_message_gate"] = {
            "policy": jsonable(
                await self.timeline_repository.resolve_state_gate_policy(profile_id, instance_id)
            ),
            "snapshot": jsonable(
                await self.timeline_repository.get_state_gate_snapshot(profile_id, instance_id)
            ),
        }
        result["character_intents"] = jsonable(
            await self.timeline_repository.list_character_intents(
                profile_id, instance_id, active_only=False, limit=100
            )
        )

    async def _add_delivery_views(
        self,
        result: dict[str, Any],
        profile_id: str,
        instance: Mapping[str, Any],
    ) -> None:
        delivery_policy = await self.timeline_repository.get_delivery_policy(
            profile_id, str(instance["scope"])
        )
        group_limit = int((delivery_policy or {}).get("group_send_qpm_limit") or 20)
        result["delivery_capability"] = {
            "platform_id": instance["platform_id"],
            "session_kind": instance["session_kind"],
            "configured_group_qpm": group_limit,
            "soulcore_paths_only": True,
        }
        try:
            result["qpm"] = jsonable(
                await self.delivery.qpm_snapshots(
                    str(instance["route_umo"]),
                    profile_id=profile_id,
                    configured_group_limit=group_limit,
                )
            )
        except Exception:
            result["qpm"] = []

    async def _add_message_views(
        self,
        result: dict[str, Any],
        profile_id: str,
        instance_id: str,
        message_page: int,
        message_page_size: int,
    ) -> None:
        stats = await self.conversation_repository.instance_message_stats(profile_id, instance_id)
        size = max(5, min(int(message_page_size), 100))
        total = int(stats.get("total") or 0)
        total_pages = max(1, (total + size - 1) // size)
        page = max(1, min(int(message_page), total_pages))
        result["messages"] = jsonable(
            await self.conversation_repository.list_instance_messages(
                profile_id,
                instance_id,
                limit=size,
                offset=(page - 1) * size,
                ascending=False,
                context_eligible_only=False,
            )
        )
        result["message_stats"] = jsonable(stats)
        result["message_pagination"] = {
            "page": page,
            "page_size": size,
            "total_pages": total_pages,
            "total": total,
        }

    async def _add_runtime_lists(
        self, result: dict[str, Any], profile_id: str, instance_id: str
    ) -> None:
        result["runs"] = jsonable(
            await self.timeline_repository.list_instance_runs(profile_id, instance_id, limit=20)
        )
        result["outbox"] = jsonable(
            await self.delivery_repository.list_instance_outbox(profile_id, instance_id, limit=20)
        )
        result["wakeups"] = jsonable(
            await self.timeline_repository.list_instance_wakeups(profile_id, instance_id, limit=20)
        )
        fragments = await self.runner.outbox.list_message_fragments(
            profile_id, instance_id, limit=100
        )
        result["platform_message_fragments"] = [
            {
                "message_ref": item.message_ref,
                "ledger_message_id": item.ledger_message_id,
                "fragment_ordinal": item.fragment_ordinal,
                "content_kind": item.content_kind,
                "content_projection": item.content_projection,
                "direction": item.direction.value,
                "native_reply_supported": item.native_reply_supported,
                "member_mention_supported": item.member_mention_supported,
                "self_retraction_supported": item.self_retraction_supported,
                "returns_platform_message_id": item.returns_platform_message_id,
                "accepted_at": jsonable(item.accepted_at),
                "retractable_until": jsonable(item.retractable_until),
                "retraction_status": jsonable(item.retraction_status),
            }
            for item in fragments
        ]
        result["message_retraction_actions"] = jsonable(
            await self.runner.outbox.list_retraction_actions(profile_id, instance_id, limit=100)
        )

    async def controlled_bridge_snapshot(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        """Return the bounded phase-2 admin view for one isolated instance."""
        assert self.timeline_repository is not None
        await self.profiles.require_role_instance(profile_id, instance_id)
        policy = await self.timeline_repository.resolve_state_gate_policy(profile_id, instance_id)
        gate = await self.timeline_repository.get_state_gate_snapshot(profile_id, instance_id)
        intents = await self.timeline_repository.list_character_intents(
            profile_id, instance_id, active_only=False, limit=100
        )
        return {
            "profile_id": profile_id,
            "instance_id": instance_id,
            "state_message_gate": {
                "policy": jsonable(policy),
                "snapshot": jsonable(gate),
            },
            "character_intents": jsonable(intents),
            "limits": {"active_character_intents": 32},
        }

    async def character_intent_admin_action(
        self, profile_id: str, instance_id: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        assert self.timeline_repository is not None
        await self.profiles.require_role_instance(profile_id, instance_id)
        action = str(payload.get("action") or "").strip().lower()
        reason = str(payload.get("reason") or "").strip()
        if action == "create":
            await self._create_character_intent(profile_id, instance_id, payload, reason)
        else:
            await self._mutate_character_intent(profile_id, instance_id, action, payload, reason)
        return await self.controlled_bridge_snapshot(profile_id, instance_id)

    async def _create_character_intent(
        self,
        profile_id: str,
        instance_id: str,
        payload: Mapping[str, Any],
        reason: str,
    ) -> None:
        assert self.timeline_repository is not None
        kind = str(payload.get("intent_kind") or "").strip().upper()
        goal = str(payload.get("goal") or "").strip()
        if kind not in {"FUTURE_THOUGHT", "ACTION_INTENT"}:
            raise ValueError("intent_kind must be FUTURE_THOUGHT or ACTION_INTENT")
        if not goal or not reason:
            raise ValueError("admin intent creation requires goal and reason")
        creation = self._intent_creation(payload, kind, goal, reason)
        await self.timeline_repository.apply_character_intent_mutations(
            profile_id, instance_id, actor_kind="ADMIN", creations=[creation]
        )

    @staticmethod
    def _intent_creation(
        payload: Mapping[str, Any], kind: str, goal: str, reason: str
    ) -> dict[str, Any]:
        return {
            "intent_kind": kind,
            "origin_kind": "ADMIN",
            "goal": goal,
            "summary": str(payload.get("summary") or goal).strip(),
            "motivation": str(payload.get("motivation") or "").strip(),
            "constraints": list(payload.get("constraints") or []),
            "priority": float(payload.get("priority", 0.5)),
            "not_before_at": payload.get("not_before_at"),
            "target_at": payload.get("target_at"),
            "expires_at": payload.get("expires_at"),
            "conflict_key": str(payload.get("conflict_key") or "").strip(),
            "creation_key": str(payload.get("creation_key") or f"admin:{uuid.uuid4().hex}"),
            "change_reason": reason,
            "evidence": [{"evidence_kind": "ADMIN", "metadata": {"reason": reason}}],
        }

    async def _mutate_character_intent(
        self,
        profile_id: str,
        instance_id: str,
        action: str,
        payload: Mapping[str, Any],
        reason: str,
    ) -> None:
        assert self.timeline_repository is not None
        intent_id, expected = self._intent_identity(payload)
        if action == "delete":
            deleted = await self.timeline_repository.delete_character_intent(
                profile_id, instance_id, intent_id, expected_version=expected
            )
            if not deleted:
                raise ValueError("only a terminal intent at the expected version can be deleted")
            return
        if not reason:
            raise ValueError("character intent admin action requires a reason")
        operation = self._intent_operation(action, payload, intent_id, expected, reason)
        await self.timeline_repository.apply_character_intent_mutations(
            profile_id, instance_id, actor_kind="ADMIN", operations=[operation]
        )

    @staticmethod
    def _intent_identity(payload: Mapping[str, Any]) -> tuple[str, int]:
        intent_id = str(payload.get("intent_id") or "").strip()
        expected = int(payload.get("expected_version") or 0)
        if not intent_id or expected < 1:
            raise ValueError("intent_id and expected_version are required")
        return intent_id, expected

    def _intent_operation(
        self,
        action: str,
        payload: Mapping[str, Any],
        intent_id: str,
        expected: int,
        reason: str,
    ) -> dict[str, Any]:
        if action != "update":
            return self._intent_transition(action, intent_id, expected, reason)
        operation: dict[str, Any] = {
            "intent_id": intent_id,
            "expected_version": expected,
            "operation": "UPDATE",
            "reason": reason,
        }
        fields = (
            "goal",
            "summary",
            "motivation",
            "constraints",
            "priority",
            "not_before_at",
            "target_at",
            "expires_at",
            "next_review_at",
        )
        operation.update({key: payload[key] for key in fields if key in payload})
        return operation

    @staticmethod
    def _intent_transition(
        action: str, intent_id: str, expected: int, reason: str
    ) -> dict[str, Any]:
        transitions = {
            "start": "IN_PROGRESS",
            "block": "BLOCKED",
            "complete": "COMPLETED",
            "consume": "CONSUMED",
            "cancel": "CANCELLED",
            "expire": "EXPIRED",
            "supersede": "SUPERSEDED",
        }
        if action not in transitions:
            raise ValueError("unsupported character intent admin action")
        return {
            "intent_id": intent_id,
            "expected_version": expected,
            "operation": "TRANSITION",
            "to_status": transitions[action],
            "reason": reason,
        }

    async def context_snapshot(
        self, profile_id: str, scope: str, instance_id: str
    ) -> dict[str, Any]:
        if self.context_service is None:
            raise RuntimeError("SoulCore context service is unavailable")
        config = await self.profiles_repository.get_scope_config(profile_id, scope)
        if config is None:
            raise ValueError("scope context configuration is unavailable")
        diagnostics = await self.context_service.diagnostics(profile_id, instance_id, config)
        instance = await self.profiles_repository.get_character_instance(profile_id, instance_id)
        if instance is not None and self.runner is not None:
            await self._add_provider_context_diagnostics(config, instance, diagnostics)
        return {
            "profile_id": profile_id,
            "scope": scope,
            "instance_id": instance_id,
            "config": jsonable(config),
            "diagnostics": jsonable(diagnostics),
        }

    async def _add_provider_context_diagnostics(
        self, config: Any, instance: Any, diagnostics: dict[str, Any]
    ) -> None:
        try:
            hint = await self.runner.resolve_backend_hint(
                config, instance.route_umo, capability="chat.completion"
            )
            metadata = dict(hint.metadata) if hint is not None else {}
            raw_limit = metadata.get("max_context_tokens")
            provider_limit = (
                int(raw_limit) if str(raw_limit or "").isdigit() and int(raw_limit) > 0 else None
            )
            diagnostics["provider_context_window"] = provider_limit
            effective_budget = int(
                dict(diagnostics.get("budget") or {}).get("max_context_tokens")
                or config.max_context_tokens
            )
            diagnostics["effective_max_tokens"] = min(
                effective_budget,
                provider_limit or effective_budget,
            )
            diagnostics["resolved_backend_id"] = (
                str(hint.backend_id or "") if hint is not None else ""
            )
            if provider_limit is None:
                diagnostics["provider_warning"] = (
                    "Provider上下文窗口未知，当前按用户MaxToken执行保守预算。"
                )
        except Exception as exc:
            diagnostics["provider_warning"] = f"{type(exc).__name__}: {exc}"

    async def force_context_summary(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        if not await self.context_service.runtime_gate.is_enabled(profile_id):
            message = "SoulCore 总开关已关闭，未创建对话摘要任务"
            return {
                "ok": False,
                "queued": False,
                "code": "profile_disabled",
                "reason": message,
                "error": message,
            }
        summary = await self.conversation_repository.get_latest_dialogue_summary(
            profile_id, instance_id
        )
        window = await self.context_service.dialogue_summary_window(
            profile_id,
            instance_id,
            covered_through_message_id=(summary.covered_through_message_id if summary else None),
        )
        target = int(window.get("target_message_id") or 0)
        if target < 1:
            turn_count = int(window.get("pending_turn_count") or 0)
            return {
                "ok": False,
                "queued": False,
                "reason": "至少需要最近20个完整说话轮次以外的更早对话才能生成摘要",
                "error": "至少需要最近20个完整说话轮次以外的更早对话才能生成摘要",
                "turn_count": turn_count,
                "message_count": int(window.get("pending_message_count") or 0),
            }
        task = await self.ai_repository.create_ai_task(
            profile_id,
            "DIALOGUE_SUMMARY",
            instance_id=instance_id,
            task_class="BACKGROUND",
            capability="text.completion",
            priority=-20,
            mutex_key=f"dialogue-summary:{instance_id}",
            idempotency_key=(f"dialogue-summary:{target}:{summary.summary_id if summary else 0}"),
            input_data={
                "target_message_id": target,
                "base_summary_id": summary.summary_id if summary else None,
            },
            recovery_policy="RESUME_CHECKPOINT",
            max_attempts=DURABLE_AI_MAX_ATTEMPTS,
        )
        return {"ok": True, "queued": True, "task": jsonable(task)}

    async def context_dry_run(self, profile_id: str, instance_id: str) -> dict[str, Any]:
        if self.context_service is None or self.runner is None:
            raise RuntimeError("SoulCore context service is unavailable")
        instance = await self.profiles_repository.get_character_instance(profile_id, instance_id)
        if instance is None:
            raise ValueError("unknown conversation instance")
        role = await self.profiles_repository.get_scope_config(profile_id, instance.scope)
        if role is None:
            raise ValueError("scope context configuration is unavailable")
        hint = await self.runner.resolve_backend_hint(
            role, instance.route_umo, capability="chat.completion"
        )
        hint_metadata = dict(hint.metadata) if hint is not None else {}
        raw_limit = hint_metadata.get("max_context_tokens")
        provider_limit = (
            int(raw_limit) if str(raw_limit or "").isdigit() and int(raw_limit) > 0 else None
        )
        model_id = str(
            (hint.model or hint.backend_id) if hint is not None else "unresolved-backend"
        )
        run_prompt = "管理员执行上下文预算 Dry Run，不产生真实消息。"
        prepared = await self.context_service.prepare(
            profile_id=profile_id,
            instance_id=instance_id,
            role=role,
            run_prompt=run_prompt,
            model_id=model_id,
            provider_context_limit=provider_limit,
        )
        return {
            "ok": True,
            "report": jsonable(prepared.compiled.report),
            "selected_context_count": len(prepared.compiled.items),
        }
