from __future__ import annotations

from dataclasses import dataclass

from ....contracts.deferred_gate import DeferredGateCommitFence
from ....contracts.group_flow import GroupRunFence
from ....contracts.inbound_recall import InboundRecallCommitFence
from ....contracts.turn_buffer import TurnBufferCommitFence
from ...stickers.service import StickerImportIntent, StickerSourceKind
from .commit_transaction import (
    InstanceCoreCommitContext,
    InstanceCoreResultTransaction,
)
from .instance_reset import InstanceResetTransaction
from .support import (
    Any,
    _dt,
    _now,
)


@dataclass(frozen=True, slots=True)
class _ParsedCommitFences:
    turn_buffer: TurnBufferCommitFence | None
    group_run: GroupRunFence | None
    inbound_recall: tuple[InboundRecallCommitFence, ...]
    deferred_gate: DeferredGateCommitFence | None


def _parse_commit_fences(
    *,
    turn_buffer_fence: dict[str, Any] | None,
    group_run_fence: dict[str, Any] | None,
    inbound_recall_fences: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    deferred_gate_fence: dict[str, Any] | None,
) -> _ParsedCommitFences | None:
    parsed_turn_buffer = TurnBufferCommitFence.from_metadata(turn_buffer_fence)
    parsed_group_run = GroupRunFence.from_metadata(group_run_fence)
    parsed_inbound_recall = tuple(
        InboundRecallCommitFence.from_metadata(value) for value in inbound_recall_fences
    )
    parsed_deferred_gate = DeferredGateCommitFence.from_metadata(deferred_gate_fence)
    if turn_buffer_fence is not None and parsed_turn_buffer is None:
        return None
    if group_run_fence is not None and parsed_group_run is None:
        return None
    if any(fence is None for fence in parsed_inbound_recall):
        return None
    if deferred_gate_fence is not None and parsed_deferred_gate is None:
        return None
    if parsed_turn_buffer is not None and parsed_group_run is not None:
        return None
    return _ParsedCommitFences(
        turn_buffer=parsed_turn_buffer,
        group_run=parsed_group_run,
        inbound_recall=tuple(fence for fence in parsed_inbound_recall if fence is not None),
        deferred_gate=parsed_deferred_gate,
    )


class CoreCommitCommands:
    async def reset_character_instance_runtime(
        self,
        profile_id: str,
        instance_id: str,
        *,
        preserve_stickers: bool = True,
    ) -> dict[str, Any]:
        if await self._profiles.get_character_instance(profile_id, instance_id) is None:
            raise KeyError((profile_id, instance_id))
        result = await self.uow.run(
            InstanceResetTransaction(
                profile_id=profile_id,
                instance_id=instance_id,
                preserve_stickers=bool(preserve_stickers),
                now=_dt(_now()),
            )
        )
        await self.db.publish_backup_after_commit(operation="instance_runtime_reset")
        return result


class InstanceCoreResultCommands:
    async def commit_instance_core_result(
        self,
        instance_id: str,
        run_id: int,
        expected_state_epoch: int,
        expected_activity_epoch: int,
        outbound_actions: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        decision: dict[str, Any] | None = None,
        selected_media_asset_ids: list[str] | tuple[str, ...] = (),
        selected_important_todo_ids: list[str] | tuple[str, ...] = (),
        file_generation_requests: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        work_checkpoint_snapshot: Any | None = None,
        work_controlled_resource_refs: list[str] | tuple[str, ...] = (),
        *,
        profile_id: str | None = None,
        player_profile_mutations: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        timer_command_context: Any | None = None,
        temporary_absence: dict[str, Any] | None = None,
        sticker_import_intents: (list[StickerImportIntent] | tuple[StickerImportIntent, ...]) = (),
        sticker_disable_item_ids: list[str] | tuple[str, ...] = (),
        expression_batch: dict[str, Any] | None = None,
        contact_silent_deferral: dict[str, Any] | None = None,
        turn_buffer_fence: dict[str, Any] | None = None,
        group_run_fence: dict[str, Any] | None = None,
        inbound_recall_fences: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
        deferred_gate_fence: dict[str, Any] | None = None,
        model_visible_message_ids: list[int] | tuple[int, ...] = (),
        delivery_output_budget: int | None = None,
    ) -> bool:
        fences = _parse_commit_fences(
            turn_buffer_fence=turn_buffer_fence,
            group_run_fence=group_run_fence,
            inbound_recall_fences=inbound_recall_fences,
            deferred_gate_fence=deferred_gate_fence,
        )
        if fences is None:
            return False
        profile_id = await self._resolve_instance_profile(profile_id, instance_id)
        instance = await self._profiles.get_character_instance(profile_id, instance_id)
        if instance is None:
            raise KeyError((profile_id, instance_id))
        normalized_file_requests = self._normalize_file_requests(file_generation_requests)
        normalized_sticker_intents = self._normalize_sticker_import_intents(sticker_import_intents)
        selected_media, selected_todos = self._validate_core_result_inputs(
            outbound_actions,
            selected_media_asset_ids,
            selected_important_todo_ids,
            normalized_file_requests,
        )
        context = InstanceCoreCommitContext(
            profile_id=profile_id,
            instance_id=instance_id,
            instance=instance,
            run_id=run_id,
            expected_state_epoch=expected_state_epoch,
            expected_activity_epoch=expected_activity_epoch,
            outbound_actions=list(outbound_actions),
            decision=decision,
            expression_batch=dict(expression_batch) if expression_batch else None,
            selected_media=selected_media,
            selected_todos=selected_todos,
            file_generation_requests=normalized_file_requests,
            work_checkpoint_snapshot=work_checkpoint_snapshot,
            work_controlled_resource_refs=self._nonempty_strings(work_controlled_resource_refs),
            player_profile_mutations=list(player_profile_mutations),
            timer_command_context=timer_command_context,
            temporary_absence=(dict(temporary_absence) if temporary_absence else None),
            sticker_import_intents=normalized_sticker_intents,
            sticker_disable_item_ids=tuple(
                dict.fromkeys(
                    str(item).strip() for item in sticker_disable_item_ids if str(item).strip()
                )
            ),
            contact_silent_deferral=(
                dict(contact_silent_deferral) if contact_silent_deferral else None
            ),
            turn_buffer_fence=fences.turn_buffer,
            group_run_fence=fences.group_run,
            inbound_recall_fences=fences.inbound_recall,
            deferred_gate_fence=fences.deferred_gate,
            model_visible_message_ids=self._positive_integers(model_visible_message_ids),
            delivery_output_budget=(
                max(0, int(delivery_output_budget)) if delivery_output_budget is not None else None
            ),
            now=_dt(_now()),
        )
        committed = await self.uow.run(InstanceCoreResultTransaction(self, context))
        if committed:
            await self.db.publish_backup_after_commit()
        return committed

    @staticmethod
    def _nonempty_strings(values: list[str] | tuple[str, ...]) -> frozenset[str]:
        return frozenset(filter(None, map(str, values)))

    @staticmethod
    def _positive_integers(values: list[int] | tuple[int, ...]) -> frozenset[int]:
        return frozenset(value for value in map(int, values) if value > 0)

    @staticmethod
    def _normalize_sticker_import_intents(
        intents: list[StickerImportIntent] | tuple[StickerImportIntent, ...],
    ) -> tuple[StickerImportIntent, ...]:
        normalized: dict[str, StickerImportIntent] = {}
        for value in intents:
            if not isinstance(value, StickerImportIntent):
                raise ValueError("sticker import intents must use the run-scoped contract")
            source_ref = str(value.source_ref).strip()
            source_asset_id = str(value.source_asset_id).strip()
            if not source_ref or not source_asset_id:
                raise ValueError("sticker import intent requires a source reference and asset")
            intent = StickerImportIntent(
                source_ref=source_ref,
                source_kind=StickerSourceKind(str(value.source_kind).upper()),
                source_asset_id=source_asset_id,
            )
            if intent.source_kind is StickerSourceKind.UPLOAD:
                raise ValueError("MainCore sticker imports cannot use upload sources")
            previous = normalized.get(source_ref)
            if previous is not None and previous != intent:
                raise ValueError("sticker import source reference has conflicting intents")
            normalized[source_ref] = intent
        return tuple(normalized.values())

    @staticmethod
    def _normalize_file_requests(
        requests: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> list[dict[str, Any]]:
        normalized = [dict(item) for item in requests]
        for index, item in enumerate(normalized, start=1):
            item.setdefault("request_ref", f"file_request_{index}")
        return normalized

    async def _resolve_instance_profile(self, profile_id: str | None, instance_id: str) -> str:
        if profile_id is not None:
            return profile_id
        owners = await self.db.fetch_all(
            "SELECT profile_id FROM character_instances WHERE instance_id = ?",
            (instance_id,),
        )
        if not owners:
            raise KeyError(instance_id)
        if len(owners) != 1:
            raise ValueError("instance_id is ambiguous; profile_id is required")
        return str(owners[0]["profile_id"])

    @staticmethod
    def _validate_core_result_inputs(
        outbound_actions: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        media_ids: list[str] | tuple[str, ...],
        todo_ids: list[str] | tuple[str, ...],
        file_requests: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> tuple[list[str], list[str]]:
        selected_media = list(dict.fromkeys(str(item) for item in media_ids))
        if len(selected_media) > 5:
            raise ValueError("at most five generated media assets may be selected")
        selected_todos = list(dict.fromkeys(str(item) for item in todo_ids if str(item)))
        if len(selected_todos) > 3:
            raise ValueError("at most three important file todos may be selected")
        if len(file_requests) > 3:
            raise ValueError("at most three file generation requests may be committed")
        return selected_media, selected_todos
