"""Outbox dispatch preparation and generation-fence checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...contracts.models import OutboxItem, OutboxStatus
from ...shared.event_log import record_event
from ...shared.time import utcnow
from ..files.service import FILE_ARTIFACTS_DISABLED_REASON
from ..stickers.service import load_sticker_runtime_policy
from .qpm import QPMDispatchFence
from .routes import CapturedUMO
from .voice_artifacts import VoiceArtifact


@dataclass(slots=True)
class OutboxDispatchContext:
    profile_id: str
    instance_id: str
    item: OutboxItem
    current: OutboxItem
    content: str
    components: list[dict[str, Any]]
    file_components: list[dict[str, Any]]
    important_todo_ids: list[str]
    identity_template: str = ""
    sticker_deliveries: list[dict[str, Any]] = field(default_factory=list)
    file_deliveries: list[dict[str, Any]] = field(default_factory=list)
    pending_ledger_message: Any | None = None
    expression_qpm_reservation: Any | None = None
    expression_qpm_dispatch_fence: QPMDispatchFence | None = None
    expression_send_permit_id: int | None = None
    expression_send_permit_started: bool = False
    delivery_chunks: int = 0
    delivery_attempted_chunks: int = 0
    delivery_receipts: tuple[Any, ...] = ()
    authoritative_delivery_result: Any | None = None
    sticker_components_removed: bool = False
    voice_requested: bool = False
    voice_artifact: VoiceArtifact | None = None
    voice_cleanup_id: int | None = None
    voice_fallback_reason: str = ""
    voice_delivered: bool = False
    retain_voice_artifact: bool = False
    runtime_policy_rejection_reason: str = ""


class OutboxDispatchPreparationMixin:
    async def _prepare_dispatch_context(
        self,
        profile_id: str,
        instance_id: str,
        item: OutboxItem,
    ) -> OutboxDispatchContext | None:
        current = await self.outbox.get_instance_outbox(profile_id, instance_id, item.outbox_id)
        if current is None or current.status is not OutboxStatus.PENDING:
            return None
        not_before = current.not_before_at
        if not_before is not None and not_before > utcnow():
            return None
        components = list(item.payload.get("components") or [])
        identity_template = str(item.payload.get("content") or "").strip()
        content = identity_template
        identity_service = getattr(self.context_service, "identity", None)
        if identity_service is not None and identity_template:
            identity_context = await identity_service.context(profile_id, instance_id)
            content = identity_service.render(identity_template, identity_context).strip()
        return OutboxDispatchContext(
            profile_id=profile_id,
            instance_id=instance_id,
            item=item,
            current=current,
            content=content,
            identity_template=identity_template,
            components=components,
            file_components=[
                component
                for component in components
                if str(component.get("type") or "") == "file_artifact"
            ],
            important_todo_ids=self._important_todo_ids(item.payload),
        )

    async def _dependency_allows_dispatch(self, ctx: OutboxDispatchContext) -> bool:
        dependency_key = str(ctx.current.depends_on_idempotency_key or "").strip()
        if not dependency_key:
            return True
        dependency = await self.outbox.get_instance_outbox_by_idempotency_key(
            ctx.profile_id, ctx.instance_id, dependency_key
        )
        is_file = self._is_file_expression(ctx.current.payload)
        if dependency is None or dependency.status in {
            OutboxStatus.FAILED,
            OutboxStatus.CANCELLED,
            OutboxStatus.PARTIALLY_ATTEMPTED,
        }:
            if not is_file:
                return True
            await self._fail_missing_dependency(ctx, dependency_key)
            return False
        if dependency.status in {OutboxStatus.PENDING, OutboxStatus.SENDING}:
            return False
        return dependency.status in {
            OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED,
            OutboxStatus.UNKNOWN_AFTER_CRASH,
        }

    @staticmethod
    def _is_file_expression(payload: dict[str, Any]) -> bool:
        if str(payload.get("expression_kind") or "").upper() == "FILE":
            return True
        if str(payload.get("file_delivery_role") or "").upper() in {
            "ANNOUNCEMENT",
            "ARTIFACT",
        }:
            return True
        return any(
            isinstance(component, dict)
            and str(component.get("type") or "").lower() == "file_artifact"
            for component in list(payload.get("components") or [])
        )

    async def _fail_missing_dependency(
        self, ctx: OutboxDispatchContext, dependency_key: str
    ) -> None:
        await self._transition_scoped_outbox(
            ctx.profile_id,
            ctx.instance_id,
            ctx.current.outbox_id,
            OutboxStatus.FAILED,
            error="required_file_announcement_failed",
        )
        if ctx.important_todo_ids:
            await self.files.settle_file_todos(
                ctx.profile_id,
                ctx.instance_id,
                ctx.important_todo_ids,
                status="PENDING",
                error="required_file_announcement_failed",
            )
        await record_event(
            self.event_log,
            profile_id=ctx.profile_id,
            instance_id=ctx.instance_id,
            level="WARN",
            category="delivery",
            message="文件说明未成功尝试，已阻止裸文件投递",
            details={"outbox_id": ctx.current.outbox_id, "dependency": dependency_key},
        )

    async def _preclaim_dispatch_ready(self, ctx: OutboxDispatchContext) -> bool:
        if self._is_file_expression(ctx.current.payload) and not (
            await self.files.get_profile_file_artifacts_enabled(ctx.profile_id)
        ):
            await self._record_outbox_wait_once(
                ctx.profile_id,
                ctx.instance_id,
                ctx.item.outbox_id,
                FILE_ARTIFACTS_DISABLED_REASON,
            )
            return False
        if not await self._activity_epoch_matches(ctx, after_claim=False):
            return False
        if not await self._outbox_belongs_to_bound_instance(
            ctx.profile_id, ctx.instance_id, ctx.item
        ):
            await self._reject_route_mismatch(ctx)
            return False
        return await self._adapter_route_ready(ctx)

    async def _adapter_route_ready(self, ctx: OutboxDispatchContext) -> bool:
        captured = CapturedUMO.parse(ctx.item.umo)
        if not self.delivery.route_ready(captured):
            await self._record_outbox_wait_once(
                ctx.profile_id,
                ctx.instance_id,
                ctx.item.outbox_id,
                "adapter_route_not_ready",
            )
            return False
        return True

    async def _reject_route_mismatch(self, ctx: OutboxDispatchContext) -> None:
        await self._transition_scoped_outbox(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.outbox_id,
            OutboxStatus.FAILED,
            error="instance_route_mismatch",
        )
        await record_event(
            self.event_log,
            profile_id=ctx.profile_id,
            instance_id=ctx.instance_id,
            level="ERROR",
            category="delivery",
            message="Outbox 路由与角色实例不匹配",
            details={"outbox_id": ctx.item.outbox_id},
        )

    async def _claim_dispatch(self, ctx: OutboxDispatchContext) -> bool:
        batch_id = str(ctx.current.expression_batch_id or "").strip()
        if batch_id:
            claimed, message = await self.outbox.begin_instance_outbox_dispatch(
                ctx.profile_id,
                ctx.instance_id,
                ctx.item.outbox_id,
                context_message=self._pending_expression_context_message(ctx),
            )
            ctx.pending_ledger_message = message
            return bool(claimed)
        return bool(
            await self.outbox.claim_instance_outbox(
                ctx.profile_id, ctx.instance_id, ctx.item.outbox_id
            )
        )

    def _pending_expression_context_message(
        self, ctx: OutboxDispatchContext
    ) -> dict[str, Any] | None:
        """Build the real ledger row before the platform call begins.

        PENDING is intentionally excluded from normal context projection.  It
        only becomes visible to a later Main Core after settlement proves that
        a platform attempt started.
        """

        if not bool(ctx.item.payload.get("context_record", True)):
            return None
        plain_text = "\n".join(
            value
            for value in (
                self._history_delivery_text(ctx),
                *(row["projection"] for row in ctx.sticker_deliveries),
            )
            if value
        )
        components = [
            component
            for component in ctx.components
            if str(component.get("type") or "") != "sticker_ref"
        ] + [{"type": "sticker", "projection": row["projection"]} for row in ctx.sticker_deliveries]
        payload = ctx.item.payload
        return {
            "role": "assistant",
            "sender_id": "soulcore",
            "sender_name": "SoulCore",
            "plain_text": plain_text,
            "identity_template": ctx.identity_template,
            "internal_memo": str(payload.get("internal_memo") or "").strip(),
            "components": components,
            "idempotency_key": f"outbox:{ctx.item.outbox_id}",
            "metadata": {
                "outbox_id": ctx.item.outbox_id,
                "origin_kind": str(payload.get("origin_kind") or "CORE_RUN"),
                "expression_batch_id": str(payload.get("expression_batch_id") or ""),
                "expression_ordinal": int(payload.get("expression_ordinal") or 0),
                **self._scene_narration_metadata(payload),
            },
            "knowledge_eligibility": "HELD",
            "knowledge_eligibility_reason": "delivery_unconfirmed",
        }

    @staticmethod
    def _scene_narration_metadata(payload: dict[str, Any]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for metadata_key in ("scene_narration_before", "scene_narration_after"):
            values = [
                str(value or "").strip()
                for value in payload.get(metadata_key) or ()
                if str(value or "").strip()
            ]
            if values:
                result[metadata_key] = values
        return result

    @staticmethod
    def _history_delivery_text(ctx: OutboxDispatchContext) -> str:
        if ctx.voice_delivered and ctx.content:
            return f"（语音）{ctx.content}"
        return ctx.content

    async def _resolve_dispatch_components(self, ctx: OutboxDispatchContext) -> bool:
        try:
            await self._resolve_sticker_components(ctx)
            await self._resolve_file_components(ctx)
            if ctx.file_components and self.file_artifact_service is None:
                raise RuntimeError("file delivery requires an artifact service")
            return True
        except Exception as exc:
            await self._fail_component_resolution(ctx, exc)
            return False

    async def _resolve_sticker_components(self, ctx: OutboxDispatchContext) -> None:
        if not await self._sticker_components_enabled(ctx):
            self._remove_sticker_components(ctx)
            return
        for component in ctx.components:
            if str(component.get("type") or "") != "sticker_ref":
                continue
            if self.context_service is None:
                raise RuntimeError("sticker delivery requires an instance context")
            sticker_ref = str(component.get("sticker_ref") or "")
            run_id = int(component.get("run_id") or 0)
            resolved = await self.stickers.resolve_sticker_run_refs(
                ctx.profile_id, ctx.instance_id, run_id, [sticker_ref]
            )
            if len(resolved) != 1:
                raise RuntimeError("sticker reference is unavailable")
            item = resolved[0]
            ctx.sticker_deliveries.append(
                {
                    "sticker_ref": sticker_ref,
                    "run_id": run_id,
                    "item_id": item.item_id,
                    "asset_id": item.asset_id,
                    "projection": f"[表情包] {item.compact_description}",
                }
            )

    async def _sticker_components_enabled(self, ctx: OutboxDispatchContext) -> bool:
        if not any(
            str(component.get("type") or "") == "sticker_ref" for component in ctx.components
        ):
            return True
        policy = await load_sticker_runtime_policy(
            self.stickers,
            self.profiles,
            ctx.profile_id,
            instance_id=ctx.instance_id,
        )
        return policy.enabled

    @staticmethod
    def _remove_sticker_components(ctx: OutboxDispatchContext) -> None:
        before = len(ctx.components)
        ctx.components = [
            component
            for component in ctx.components
            if str(component.get("type") or "") != "sticker_ref"
        ]
        if len(ctx.components) != before or ctx.sticker_deliveries:
            ctx.sticker_components_removed = True
        ctx.sticker_deliveries.clear()

    async def _resolve_file_components(self, ctx: OutboxDispatchContext) -> None:
        if not ctx.important_todo_ids:
            return
        rows = await self.files.get_file_assets_for_todos(
            ctx.profile_id, ctx.instance_id, ctx.important_todo_ids
        )
        if {str(row.get("todo_id") or "") for row in rows} != set(ctx.important_todo_ids):
            raise RuntimeError("important todo ownership changed")
        ctx.file_deliveries = [row for row in rows if row.get("asset_id")]
        actual_assets = {
            str(row.get("asset_id") or "") for row in ctx.file_deliveries if row.get("asset_id")
        }
        expected_assets = {
            str(component.get("asset_id") or "") for component in ctx.file_components
        }
        if actual_assets != expected_assets:
            raise RuntimeError("file artifact ownership changed")
        await self.files.settle_file_todos(
            ctx.profile_id,
            ctx.instance_id,
            ctx.important_todo_ids,
            status="DELIVERY_PENDING",
        )

    async def _fail_component_resolution(self, ctx: OutboxDispatchContext, exc: Exception) -> None:
        error = f"component_resolution_failed:{type(exc).__name__}"
        await self._transition_scoped_outbox(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.outbox_id,
            OutboxStatus.FAILED,
            error=error,
        )
        await self._settle_contact_outbox(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.payload,
            delivered=False,
            reason="component_resolution_failed",
        )
        if ctx.important_todo_ids:
            await self.files.settle_file_todos(
                ctx.profile_id,
                ctx.instance_id,
                ctx.important_todo_ids,
                status="PENDING",
                error=error,
            )

    async def _dispatch_has_content(self, ctx: OutboxDispatchContext) -> bool:
        if ctx.content or ctx.components:
            return True
        reason = "sticker_disabled" if ctx.sticker_components_removed else "empty_content"
        status = OutboxStatus.CANCELLED if ctx.sticker_components_removed else OutboxStatus.FAILED
        await self._transition_scoped_outbox(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.outbox_id,
            status,
            error=reason,
        )
        await self._settle_contact_outbox(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.payload,
            delivered=False,
            reason=reason,
        )
        if ctx.important_todo_ids:
            await self.files.settle_file_todos(
                ctx.profile_id,
                ctx.instance_id,
                ctx.important_todo_ids,
                status="PENDING",
                error=reason,
            )
        await record_event(
            self.event_log,
            profile_id=ctx.profile_id,
            instance_id=ctx.instance_id,
            level="ERROR",
            category="delivery",
            message="Outbox 内容为空，已拒绝投递",
            details={"outbox_id": ctx.item.outbox_id},
        )
        return False

    async def _activity_epoch_matches(
        self, ctx: OutboxDispatchContext, *, after_claim: bool
    ) -> bool:
        state = await self.profiles.get_instance_state(ctx.profile_id, ctx.instance_id)
        if state.activity_epoch == ctx.current.activity_epoch:
            return True
        if await self._allows_preserved_expression_dispatch(ctx):
            return True
        await self._reject_superseded_dispatch(ctx, finalize_media=after_claim)
        return False

    async def _allows_preserved_expression_dispatch(self, ctx: OutboxDispatchContext) -> bool:
        return bool(
            await self.outbox.allows_preserved_expression_dispatch(
                ctx.profile_id,
                ctx.instance_id,
                ctx.current.outbox_id,
            )
        )

    async def _reject_superseded_dispatch(
        self, ctx: OutboxDispatchContext, *, finalize_media: bool
    ) -> None:
        await self._transition_scoped_outbox(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.outbox_id,
            OutboxStatus.FAILED,
            error="superseded_by_new_inbound_activity",
        )
        await self._settle_contact_outbox(
            ctx.profile_id,
            ctx.instance_id,
            ctx.item.payload,
            delivered=False,
            reason="superseded_by_new_inbound_activity",
            superseded=True,
        )
        if finalize_media:
            await self._fail_image_components(ctx, "superseded_by_new_inbound_activity")
        if ctx.important_todo_ids:
            await self.files.settle_file_todos(
                ctx.profile_id,
                ctx.instance_id,
                ctx.important_todo_ids,
                status="PENDING",
                error="superseded_by_new_inbound_activity",
            )
        if not finalize_media:
            await record_event(
                self.event_log,
                profile_id=ctx.profile_id,
                instance_id=ctx.instance_id,
                level="WARN",
                category="delivery",
                message="Outbox 已被较新的用户消息取消",
                details={"outbox_id": ctx.item.outbox_id},
            )

    async def _fail_image_components(self, ctx: OutboxDispatchContext, error: str) -> None:
        for component in ctx.components:
            if str(component.get("type") or "") == "image_asset":
                await self.media.finalize_media_delivery(
                    ctx.profile_id,
                    ctx.instance_id,
                    str(component.get("asset_id") or ""),
                    "FAILED",
                    error=error,
                )


__all__ = ["OutboxDispatchContext", "OutboxDispatchPreparationMixin"]
