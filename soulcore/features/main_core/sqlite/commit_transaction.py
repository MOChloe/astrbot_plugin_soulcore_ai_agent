from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from ....contracts.deferred_gate import DeferredGateCommitFence
from ....contracts.group_flow import GroupRunFence
from ....contracts.inbound_recall import InboundRecallCommitFence
from ....contracts.turn_buffer import TurnBufferCommitFence
from ....storage.sqlite.dialogue_turns import INTERNAL_TIMELINE_DELIVERY_STATUS
from ...delivery.service import outbox_todo_ids
from ...identity import validate_identity_template
from ...stickers.service import StickerImportIntent
from ...timeline.service import (
    TemporaryAbsenceExpiryWake,
    temporary_absence_expiry_payload,
)
from ..work_continuity import MainCoreWorkSnapshot
from .commit_contact import ContactSilentDeferralSettlement
from .commit_deferred_gate import DeferredGateCommitSettlement
from .commit_file_work import FileWorkCheckpointMixin
from .commit_files import FileRequestWriter
from .commit_group_flow import GroupFlowCommitSettlement
from .expression_transaction import ExpressionTransactionMixin
from .player_profiles import PlayerProfileTransactionWriter
from .support import (
    Any,
    OutboxStatus,
    RunStatus,
    _dt,
    _dump,
    _parse,
)


@dataclass(frozen=True, slots=True)
class InboundRecallCommitSettlement:
    profile_id: str
    instance_id: str
    expected_activity_epoch: int
    fences: tuple[InboundRecallCommitFence, ...]
    now: str

    def claims_are_current(self, conn: sqlite3.Connection) -> bool:
        if any(fence.activity_epoch != self.expected_activity_epoch for fence in self.fences):
            return False
        for fence in self.fences:
            row = conn.execute(
                """SELECT 1 FROM inbound_message_recall_states state
                JOIN instance_messages message
                  ON message.profile_id = state.profile_id
                 AND message.instance_id = state.instance_id
                 AND message.message_id = state.ledger_message_id
                WHERE state.profile_id = ? AND state.instance_id = ?
                  AND state.ledger_message_id = ? AND state.status = 'RELEASED'
                  AND state.lease_token = ? AND message.delivery_status = 'RECEIVED'""",
                (
                    self.profile_id,
                    self.instance_id,
                    fence.ledger_message_id,
                    fence.lease_token,
                ),
            ).fetchone()
            if row is None:
                return False
        return True

    def resolve(self, conn: sqlite3.Connection) -> None:
        for fence in self.fences:
            changed = conn.execute(
                """UPDATE inbound_message_recall_states
                SET status = 'DISPATCHED', activity_epoch = ?,
                    dispatched_at = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND ledger_message_id = ?
                  AND status = 'RELEASED' AND lease_token = ?""",
                (
                    self.expected_activity_epoch,
                    self.now,
                    self.now,
                    self.profile_id,
                    self.instance_id,
                    fence.ledger_message_id,
                    fence.lease_token,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError("inbound-recall ownership changed during Main Core commit")


@dataclass(frozen=True, slots=True)
class TurnBufferCommitSettlement:
    profile_id: str
    instance_id: str
    expected_activity_epoch: int
    fence: TurnBufferCommitFence | None
    now: str

    def claim_is_current(self, conn: sqlite3.Connection) -> bool:
        fence = self.fence
        if fence is None:
            return True
        if fence.activity_epoch != self.expected_activity_epoch:
            return False
        row = conn.execute(
            """SELECT 1 FROM conversation_turn_buffer_batches
            WHERE profile_id = ? AND instance_id = ? AND batch_id = ?
              AND status = 'CLAIMED' AND generation = ? AND activity_epoch = ?
              AND lease_token = ? AND version = ? AND main_core_task_ref = ?""",
            (
                self.profile_id,
                self.instance_id,
                fence.batch_id,
                fence.generation,
                fence.activity_epoch,
                fence.lease_token,
                fence.version,
                fence.main_core_task_ref,
            ),
        ).fetchone()
        return row is not None

    def resolve(self, conn: sqlite3.Connection) -> None:
        fence = self.fence
        if fence is None:
            return
        changed = conn.execute(
            """UPDATE conversation_turn_buffer_batches SET status = 'RESOLVED',
            due_at = NULL, lease_owner = NULL, lease_until = NULL,
            lease_token = lease_token + 1,
            resolution_outcome = 'MAIN_CORE_COMMITTED',
            version = version + 1, updated_at = ?, resolved_at = ?
            WHERE profile_id = ? AND instance_id = ? AND batch_id = ?
              AND status = 'CLAIMED' AND generation = ? AND activity_epoch = ?
              AND lease_token = ? AND version = ? AND main_core_task_ref = ?""",
            (
                self.now,
                self.now,
                self.profile_id,
                self.instance_id,
                fence.batch_id,
                fence.generation,
                fence.activity_epoch,
                fence.lease_token,
                fence.version,
                fence.main_core_task_ref,
            ),
        ).rowcount
        if changed != 1:
            raise RuntimeError("turn-buffer ownership changed during Main Core commit")
        released = conn.execute(
            """UPDATE instance_messages SET knowledge_eligibility = 'ELIGIBLE',
            knowledge_eligibility_reason = ''
            WHERE profile_id = ? AND instance_id = ?
              AND knowledge_eligibility = 'HELD'
              AND knowledge_eligibility_reason = 'inbound_turn_buffer_pending'
              AND message_id IN (SELECT message_id
                FROM conversation_turn_buffer_members WHERE batch_id = ?)""",
            (self.profile_id, self.instance_id, fence.batch_id),
        ).rowcount
        if released:
            conn.execute(
                """UPDATE knowledge_processing_state SET
                processing_version = processing_version + 1, updated_at = ?
                WHERE profile_id = ? AND instance_id = ?""",
                (self.now, self.profile_id, self.instance_id),
            )


@dataclass(frozen=True, slots=True)
class InstanceCoreCommitContext:
    profile_id: str
    instance_id: str
    instance: Any
    run_id: int
    expected_state_epoch: int
    expected_activity_epoch: int
    outbound_actions: list[dict[str, Any]]
    decision: dict[str, Any] | None
    expression_batch: dict[str, Any] | None
    selected_media: list[str]
    selected_todos: list[str]
    file_generation_requests: list[dict[str, Any]]
    work_checkpoint_snapshot: MainCoreWorkSnapshot | None
    work_controlled_resource_refs: frozenset[str]
    player_profile_mutations: list[dict[str, Any]]
    timer_command_context: Any | None
    temporary_absence: dict[str, Any] | None
    sticker_import_intents: tuple[StickerImportIntent, ...]
    sticker_disable_item_ids: tuple[str, ...]
    contact_silent_deferral: dict[str, Any] | None
    turn_buffer_fence: TurnBufferCommitFence | None
    group_run_fence: GroupRunFence | None
    inbound_recall_fences: tuple[InboundRecallCommitFence, ...]
    deferred_gate_fence: DeferredGateCommitFence | None
    model_visible_message_ids: frozenset[int]
    delivery_output_budget: int | None
    now: str


def _temporary_absence_policy(
    conn: sqlite3.Connection,
    context: InstanceCoreCommitContext,
) -> Any:
    policy = conn.execute(
        """SELECT
            COALESCE(override.enabled, policy.enabled) AS enabled,
            COALESCE(override.max_gate_hours, policy.max_gate_hours) AS max_gate_hours
        FROM character_instances AS instance
        JOIN scope_state_gate_policies AS policy
          ON policy.profile_id = instance.profile_id AND policy.scope = instance.scope
        LEFT JOIN instance_state_gate_overrides AS override
          ON override.profile_id = instance.profile_id
         AND override.instance_id = instance.instance_id
        WHERE instance.profile_id = ? AND instance.instance_id = ?""",
        (context.profile_id, context.instance_id),
    ).fetchone()
    if policy is None or not bool(policy["enabled"]):
        raise ValueError("temporary absence is not enabled for this conversation")
    return policy


def _temporary_absence_due_at(
    intent: dict[str, Any],
    *,
    committed_at: datetime,
) -> tuple[str, datetime]:
    reason = str(intent.get("reason") or "").strip()
    if not reason or len(reason) > 1000:
        raise ValueError("temporary absence reason is invalid")
    rule = dict(intent.get("rule") or {})
    kind = str(rule.get("kind") or "").upper()
    if kind == "RELATIVE":
        return reason, committed_at + timedelta(seconds=int(rule.get("delay_seconds") or 0))
    if kind == "ABSOLUTE":
        due_at = _parse(str(rule.get("at") or ""))
        if due_at is None:
            raise ValueError("temporary absence end time is invalid")
        return reason, due_at
    raise ValueError("temporary absence time rule is invalid")


class InstanceCoreResultTransaction(FileWorkCheckpointMixin, ExpressionTransactionMixin):
    def __init__(self, owner: Any, context: InstanceCoreCommitContext) -> None:
        self.owner = owner
        self.context = context

    def __call__(self, conn: sqlite3.Connection) -> bool:
        buffer_settlement = TurnBufferCommitSettlement(
            self.context.profile_id,
            self.context.instance_id,
            self.context.expected_activity_epoch,
            self.context.turn_buffer_fence,
            self.context.now,
        )
        group_settlement = GroupFlowCommitSettlement(
            profile_id=self.context.profile_id,
            instance_id=self.context.instance_id,
            fence=self.context.group_run_fence,
            source_run_id=self.context.run_id,
            segment_index=(self.context.expression_batch or {}).get("segment_index"),
            has_visible_output=bool(self.context.outbound_actions or self.context.expression_batch),
            final_commit=True,
            now=self.context.now,
        )
        recall_settlement = InboundRecallCommitSettlement(
            self.context.profile_id,
            self.context.instance_id,
            self.context.expected_activity_epoch,
            self.context.inbound_recall_fences,
            self.context.now,
        )
        deferred_settlement = DeferredGateCommitSettlement(
            self.owner,
            self.context.profile_id,
            self.context.instance_id,
            self.context.expected_activity_epoch,
            self.context.run_id,
            self.context.deferred_gate_fence,
            self.context.now,
        )
        contact_settlement = ContactSilentDeferralSettlement(
            self.context.profile_id,
            self.context.instance_id,
            self.context.contact_silent_deferral,
            self.context.now,
        )
        if not self._can_commit(
            conn,
            buffer_settlement,
            group_settlement,
            contact_settlement,
            recall_settlement,
            deferred_settlement,
        ):
            return False
        self._validate_delivery_output_budget()
        self._mark_model_visible_messages(conn)
        self._apply_timer_intents(conn)
        self._apply_temporary_absence(conn)
        PlayerProfileTransactionWriter(self.context).apply(conn)
        self._select_media(conn)
        self._select_todos(conn)
        self._validate_file_feature(conn)
        self._apply_sticker_import_intents(conn)
        self._apply_sticker_disable_intents(conn)
        file_writer = FileRequestWriter(self.owner, self.context)
        created_file_jobs: list[tuple[str, dict[str, Any]]] = []
        for index, request in enumerate(self.context.file_generation_requests):
            created_file_jobs.append((file_writer.create(conn, index, request), request))
        self._freeze_file_work(conn, created_file_jobs)
        contact_settlement.resolve(conn)
        new_epoch = self._update_core_state(conn)
        self._create_expression_batch(conn)
        if self.context.expression_batch:
            self._create_expression_timeline(conn)
        else:
            for index, action in enumerate(self.context.outbound_actions):
                self._create_outbound(conn, index, action)
        buffer_settlement.resolve(conn)
        group_settlement.resolve(conn)
        recall_settlement.resolve(conn)
        deferred_settlement.resolve(conn)
        self._finish_run(conn, new_epoch)
        return True

    def _validate_delivery_output_budget(self) -> None:
        budget = self.context.delivery_output_budget
        batch = self.context.expression_batch
        if budget is None or batch is None:
            return
        output_count = int(batch.get("output_count") or 0)
        if output_count > budget:
            raise ValueError("expression batch exceeds the run-scoped delivery output budget")

    def _mark_model_visible_messages(self, conn: sqlite3.Connection) -> None:
        ids = tuple(sorted(self.context.model_visible_message_ids))
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        conn.execute(
            f"""UPDATE inbound_message_recall_states
            SET committed_full_at = ?, committed_run_id = ?, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?
              AND ledger_message_id IN ({placeholders})
              AND status != 'RECALLED'
              AND EXISTS (
                SELECT 1 FROM instance_messages message
                WHERE message.profile_id = inbound_message_recall_states.profile_id
                  AND message.instance_id = inbound_message_recall_states.instance_id
                  AND message.message_id = inbound_message_recall_states.ledger_message_id
                  AND message.delivery_status = 'RECEIVED'
              )""",
            (
                self.context.now,
                self.context.run_id,
                self.context.now,
                self.context.profile_id,
                self.context.instance_id,
                *ids,
            ),
        )

    def _apply_timer_intents(self, conn: sqlite3.Connection) -> None:
        timer_context = self.context.timer_command_context
        if timer_context is None:
            return
        committed_at = _parse(self.context.now)
        if committed_at is None:
            raise ValueError("Timer final commit requires an aware transaction timestamp")
        timer_context.commit_in_transaction(
            conn,
            committed_at=committed_at,
            on_created=lambda event: self._append_future_arrangement_timeline_event(
                conn,
                event,
                committed_at=committed_at,
            ),
        )

    def _append_future_arrangement_timeline_event(
        self,
        conn: sqlite3.Connection,
        event: Any,
        *,
        committed_at: datetime,
    ) -> None:
        template = str(
            validate_identity_template(
                (
                    "[已记下未来的事，未发送给对方] "
                    f"时间：{event.schedule_summary}；"
                    f"到时候做什么：{event.action_template}"
                ),
                scope=str(self.context.instance.scope),
            )
        )
        conn.execute(
            """INSERT INTO instance_messages(
                profile_id, instance_id, direction, role, internal_memo, sender_id,
                sender_name, plain_text, identity_template, components_json, delivery_status,
                idempotency_key, metadata_json, occurred_at, created_at,
                knowledge_eligibility, knowledge_eligibility_reason,
                expression_batch_id, expression_ordinal
            ) VALUES (?, ?, 'OUTBOUND', 'system', '', 'soulcore', '', ?, ?, '[]', ?, ?, ?,
                ?, ?, 'ELIGIBLE', '', NULL, NULL)""",
            (
                self.context.profile_id,
                self.context.instance_id,
                template,
                template,
                INTERNAL_TIMELINE_DELIVERY_STATUS,
                str(event.idempotency_key),
                _dump(
                    {
                        "timeline_event_kind": "future_arrangement",
                        "visibility": "model_private",
                        "source_run_id": self.context.run_id,
                    }
                ),
                _dt(committed_at),
                self.context.now,
            ),
        )

    def _apply_temporary_absence(self, conn: sqlite3.Connection) -> None:
        intent = self.context.temporary_absence
        if intent is None:
            return
        committed_at = _parse(self.context.now)
        if committed_at is None:
            raise ValueError("temporary absence requires an aware transaction timestamp")
        policy = _temporary_absence_policy(conn, self.context)
        reason, due_at = _temporary_absence_due_at(intent, committed_at=committed_at)
        duration = due_at - committed_at
        maximum = timedelta(hours=min(24, max(1, int(policy["max_gate_hours"]))))
        if duration < timedelta(minutes=1) or duration > maximum:
            raise ValueError("temporary absence duration is outside the configured boundary")
        changed = conn.execute(
            """UPDATE instance_state_gate_snapshots SET action = 'DEFER',
            reason_code = 'TEMPORARY_ABSENCE', expression_context = ?,
            not_before_at = ?, until_at = ?, source_run_id = ?,
            generation = generation + 1, version = version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?""",
            (
                reason,
                self.context.now,
                _dt(due_at),
                self.context.run_id,
                self.context.now,
                self.context.profile_id,
                self.context.instance_id,
            ),
        ).rowcount
        if changed != 1:
            raise RuntimeError("temporary absence snapshot is unavailable")
        snapshot = conn.execute(
            """SELECT generation FROM instance_state_gate_snapshots
            WHERE profile_id = ? AND instance_id = ?""",
            (self.context.profile_id, self.context.instance_id),
        ).fetchone()
        if snapshot is None:
            raise RuntimeError("temporary absence snapshot disappeared")
        state = conn.execute(
            """SELECT activity_epoch FROM instance_core_state
            WHERE profile_id = ? AND instance_id = ?""",
            (self.context.profile_id, self.context.instance_id),
        ).fetchone()
        if state is None:
            raise RuntimeError("temporary absence activity state disappeared")
        marker = TemporaryAbsenceExpiryWake(
            gate_generation=int(snapshot["generation"]),
            activity_epoch=int(state["activity_epoch"]),
            source_run_id=self.context.run_id,
        )
        conn.execute(
            """INSERT INTO instance_wakeups(
                profile_id, instance_id, source, due_at, reason,
                conversation_ref, idempotency_key, payload_json, status,
                intent_kind, created_at, updated_at
            ) VALUES (?, ?, 'PLUGIN_WAKE', ?, ?, ?, ?, ?, 'PENDING',
                'PLUGIN_WAKE', ?, ?)
            ON CONFLICT(profile_id, instance_id, idempotency_key)
                WHERE idempotency_key IS NOT NULL DO NOTHING""",
            (
                self.context.profile_id,
                self.context.instance_id,
                _dt(due_at),
                "此前主动暂离的时间自然结束",
                self.context.instance.route_umo,
                marker.idempotency_key,
                _dump(
                    temporary_absence_expiry_payload(
                        reason=reason,
                        started_at=committed_at,
                        planned_until=due_at,
                        gate_generation=marker.gate_generation,
                        activity_epoch=marker.activity_epoch,
                        source_run_id=marker.source_run_id,
                    )
                ),
                self.context.now,
                self.context.now,
            ),
        )

    def _apply_sticker_import_intents(self, conn: sqlite3.Connection) -> None:
        for intent in self.context.sticker_import_intents:
            self.owner._core_commit_transactions.commit_sticker(
                conn,
                profile_id=self.context.profile_id,
                instance_id=self.context.instance_id,
                run_id=self.context.run_id,
                intent=intent,
                now=self.context.now,
            )

    def _apply_sticker_disable_intents(self, conn: sqlite3.Connection) -> None:
        committed_at = _parse(self.context.now)
        if committed_at is None:
            raise ValueError("sticker disable commit requires an aware transaction timestamp")
        for item_id in self.context.sticker_disable_item_ids:
            self.owner._core_commit_transactions.disable_sticker(
                conn,
                self.context.profile_id,
                self.context.instance_id,
                item_id,
                now=committed_at,
            )

    def _can_commit(
        self,
        conn: sqlite3.Connection,
        buffer_settlement: TurnBufferCommitSettlement,
        group_settlement: GroupFlowCommitSettlement,
        contact_settlement: ContactSilentDeferralSettlement,
        recall_settlement: InboundRecallCommitSettlement,
        deferred_settlement: DeferredGateCommitSettlement,
    ) -> bool:
        return bool(
            self._snapshot_is_current(conn)
            and buffer_settlement.claim_is_current(conn)
            and group_settlement.claim_is_current(conn)
            and contact_settlement.claim_is_current(conn)
            and recall_settlement.claims_are_current(conn)
            and deferred_settlement.claim_is_current(conn)
        )

    def _snapshot_is_current(self, conn: sqlite3.Connection) -> bool:
        context = self.context
        state = conn.execute(
            """SELECT state.state_epoch, state.activity_epoch,
                profile.enabled AS profile_enabled,
                instance.initialization_state,
                COALESCE(chat_policy.soulcore_enabled, 1) AS instance_enabled,
                COALESCE(chat_policy.image_send_enabled, 1) AS image_send_enabled
            FROM instance_core_state AS state
            JOIN role_profiles AS profile ON profile.profile_id = state.profile_id
            JOIN character_instances AS instance
              ON instance.profile_id = state.profile_id
             AND instance.instance_id = state.instance_id
            LEFT JOIN instance_chat_policies AS chat_policy
              ON chat_policy.profile_id = state.profile_id
             AND chat_policy.instance_id = state.instance_id
            WHERE state.profile_id = ? AND state.instance_id = ?""",
            (context.profile_id, context.instance_id),
        ).fetchone()
        run = conn.execute(
            """SELECT status FROM instance_core_runs
            WHERE profile_id = ? AND instance_id = ? AND run_id = ?""",
            (context.profile_id, context.instance_id, context.run_id),
        ).fetchone()
        return bool(
            state is not None
            and run is not None
            and bool(state["profile_enabled"])
            and bool(state["instance_enabled"])
            and (not context.selected_media or bool(state["image_send_enabled"]))
            and state["initialization_state"] == "READY"
            and run["status"] == RunStatus.RUNNING.value
            and state["state_epoch"] == context.expected_state_epoch
            and (
                context.group_run_fence is not None
                or state["activity_epoch"] == context.expected_activity_epoch
            )
        )

    def _select_media(self, conn: sqlite3.Connection) -> None:
        context = self.context
        if not context.selected_media:
            return
        placeholders = ",".join("?" for _ in context.selected_media)
        rows = list(
            conn.execute(
                f"""SELECT asset_id FROM media_assets
            WHERE profile_id = ? AND instance_id = ? AND core_run_id = ?
              AND origin = 'GENERATED' AND file_status = 'AVAILABLE'
              AND inspection_status = 'READY'
              AND asset_id IN ({placeholders})""",
                (
                    context.profile_id,
                    context.instance_id,
                    context.run_id,
                    *context.selected_media,
                ),
            )
        )
        if {row["asset_id"] for row in rows} != set(context.selected_media):
            raise ValueError("selected assets must be inspected outputs of the current run")
        conn.execute(
            f"""UPDATE media_assets SET delivery_status = 'SELECTED', updated_at = ?
            WHERE asset_id IN ({placeholders})""",
            (context.now, *context.selected_media),
        )

    def _select_todos(self, conn: sqlite3.Connection) -> None:
        context = self.context
        if not context.selected_todos:
            return
        placeholders = ",".join("?" for _ in context.selected_todos)
        rows = list(
            conn.execute(
                f"""SELECT t.todo_id, t.kind, t.file_asset_id
            FROM important_todos t
            LEFT JOIN file_assets f ON f.asset_id = t.file_asset_id
            WHERE t.profile_id = ? AND t.instance_id = ?
              AND t.status = 'PENDING' AND t.todo_id IN ({placeholders})
              AND (t.kind = 'FILE_FAILED' OR (
                t.kind = 'FILE_READY' AND f.file_status = 'AVAILABLE'
              ))""",
                (context.profile_id, context.instance_id, *context.selected_todos),
            )
        )
        if {str(row["todo_id"]) for row in rows} != set(context.selected_todos):
            raise ValueError("selected important todos must be pending and owned by this instance")
        conn.execute(
            f"""UPDATE important_todos SET status = 'SELECTED',
                selected_run_id = ?, selected_activity_epoch = ?,
                version = version + 1, updated_at = ?
            WHERE todo_id IN ({placeholders})""",
            (
                context.run_id,
                context.expected_activity_epoch,
                context.now,
                *context.selected_todos,
            ),
        )
        self._select_todo_assets(conn, rows)

    def _select_todo_assets(self, conn: sqlite3.Connection, todo_rows: list[sqlite3.Row]) -> None:
        asset_ids = [
            str(row["file_asset_id"]) for row in todo_rows if row["file_asset_id"] is not None
        ]
        if not asset_ids:
            return
        placeholders = ",".join("?" for _ in asset_ids)
        conn.execute(
            f"""UPDATE file_assets SET delivery_status = 'SELECTED',
                updated_at = ? WHERE asset_id IN ({placeholders})""",
            (self.context.now, *asset_ids),
        )

    def _validate_file_feature(self, conn: sqlite3.Connection) -> None:
        context = self.context
        if not context.file_generation_requests and not context.selected_todos:
            return
        feature = conn.execute(
            "SELECT file_artifacts_enabled FROM role_profiles WHERE profile_id = ?",
            (context.profile_id,),
        ).fetchone()
        if feature is None or not bool(feature[0]):
            raise ValueError("file artifacts are disabled for this profile")

    def _update_core_state(self, conn: sqlite3.Connection) -> int:
        context = self.context
        new_epoch = context.expected_state_epoch + 1
        conn.execute(
            """UPDATE instance_core_state SET
                state_epoch = ?, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?""",
            (
                new_epoch,
                context.now,
                context.profile_id,
                context.instance_id,
            ),
        )
        return new_epoch

    def _create_outbound(
        self,
        conn: sqlite3.Connection,
        index: int,
        raw: dict[str, Any],
    ) -> None:
        context = self.context
        route_umo, payload, key, expression = self._normalize_outbound(index, raw)
        if route_umo != context.instance.route_umo:
            raise ValueError("instance outbound route must equal instance.route_umo")
        inserted = conn.execute(
            """INSERT INTO instance_outbox(
                profile_id, instance_id, workflow_id, route_umo, payload_json, status,
                idempotency_key, activity_epoch, origin_kind,
                origin_run_id, expression_batch_id, expression_ordinal,
                expression_step_ordinal, not_before_at, interrupt_policy,
                depends_on_idempotency_key, created_at, updated_at
            ) VALUES (?, ?, (SELECT workflow_id FROM instance_core_runs WHERE run_id = ?),
                ?, ?, ?, ?, ?, 'CORE_RUN', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id, instance_id, idempotency_key) DO NOTHING""",
            (
                context.profile_id,
                context.instance_id,
                context.run_id,
                context.instance.route_umo,
                _dump(payload),
                OutboxStatus.PENDING.value,
                key,
                context.expected_activity_epoch,
                context.run_id,
                expression["batch_id"],
                expression["ordinal"],
                expression["step_ordinal"],
                expression["not_before_at"],
                expression["interrupt_policy"],
                expression["depends_on_idempotency_key"],
                context.now,
                context.now,
            ),
        )
        outbox_id = int(
            conn.execute(
                """SELECT outbox_id FROM instance_outbox
                WHERE profile_id = ? AND instance_id = ? AND idempotency_key = ?""",
                (context.profile_id, context.instance_id, key),
            ).fetchone()[0]
        )
        if inserted.rowcount:
            self.owner._core_commit_transactions.bind_todos(
                conn,
                profile_id=context.profile_id,
                instance_id=context.instance_id,
                outbox_id=outbox_id,
                todo_ids=outbox_todo_ids(payload),
                selected_run_id=context.run_id,
            )

    def _normalize_outbound(
        self, index: int, raw: dict[str, Any]
    ) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
        context = self.context
        default_key = f"instance-run:{context.run_id}:outbound:{index}"
        route_umo = str(raw.get("route_umo") or context.instance.route_umo)
        payload = dict(raw.get("payload") or {})
        if "content" in raw and "content" not in payload:
            payload["content"] = raw["content"]
        scope = str(context.instance.scope)
        if "content" in payload:
            payload["content"] = str(
                validate_identity_template(str(payload.get("content") or ""), scope=scope)
            )
        if "internal_memo" in payload:
            payload["internal_memo"] = str(
                validate_identity_template(str(payload.get("internal_memo") or ""), scope=scope)
            )
        self._normalize_scene_narration_payload(payload, scope=scope)
        self._tag_group_window(payload)
        return (
            route_umo,
            payload,
            str(raw.get("idempotency_key") or default_key),
            (self._outbound_expression(raw)),
        )

    @staticmethod
    def _normalize_scene_narration_payload(payload: dict[str, Any], *, scope: str) -> None:
        for metadata_key in ("scene_narration_before", "scene_narration_after"):
            if metadata_key not in payload:
                continue
            value = payload.get(metadata_key)
            raw_values = (
                (value,) if isinstance(value, str) else value if isinstance(value, list) else ()
            )
            payload[metadata_key] = [
                str(validate_identity_template(str(item or ""), scope=scope))
                for item in raw_values
                if str(item or "").strip()
            ]

    def _tag_group_window(self, payload: dict[str, Any]) -> None:
        fence = self.context.group_run_fence
        if fence is not None:
            payload["group_window_id"] = fence.window_id

    def _outbound_expression(self, raw: dict[str, Any]) -> dict[str, Any]:
        delay_seconds = max(0, int(raw.get("not_before_after_seconds") or 0))
        not_before = self._expression_not_before(delay_seconds)
        interrupt_policy = (
            "PRESERVE"
            if self.context.group_run_fence is not None
            else str(raw.get("interrupt_policy") or "PRESERVE").upper()
        )
        return {
            "batch_id": str(raw.get("expression_batch_id") or "").strip() or None,
            "ordinal": raw.get("expression_ordinal"),
            "step_ordinal": raw.get("expression_step_ordinal"),
            "not_before_at": not_before,
            "interrupt_policy": interrupt_policy,
            "depends_on_idempotency_key": (
                str(raw.get("depends_on_idempotency_key") or "").strip() or None
            ),
        }

    def _finish_run(self, conn: sqlite3.Connection, new_epoch: int) -> None:
        context = self.context
        cursor = conn.execute(
            """UPDATE instance_core_runs SET status = ?, decision_json = ?,
                committed_state_epoch = ?, finished_at = ?
            WHERE profile_id = ? AND instance_id = ? AND run_id = ?
                AND status = ?""",
            (
                RunStatus.COMPLETED.value,
                _dump(context.decision or {}),
                new_epoch,
                context.now,
                context.profile_id,
                context.instance_id,
                context.run_id,
                RunStatus.RUNNING.value,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("instance run changed while committing")
