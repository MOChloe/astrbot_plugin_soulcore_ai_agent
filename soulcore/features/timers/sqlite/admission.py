"""Atomic SQLite admission and occupancy transitions for Timer runs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from ....storage.sqlite.codec import encode_datetime
from ....storage.sqlite.expression_batch_lifecycle import (
    terminal_timer_occurrence_status,
)
from ....storage.sqlite.repository import SqliteRepository
from ..admission import (
    BeginTimerProviderCommand,
    ClaimNextTimerCommand,
    CompleteTimerNoOpCommand,
    HandoffTimerExpressionCommand,
    RetryTimerRunCommand,
    SupersedeTimerRunCommand,
    TimerAdmissionFenceError,
    TimerAdmissionResult,
    TimerClaimOutcome,
    TimerProviderOutcome,
    TimerProviderResult,
    TimerRunFence,
    TimerSettlementResult,
)
from ..domain import TimerScope, require_aware
from ..task_identity import timer_run_task_idempotency_key
from .codec import decode_occupancy, decode_occurrence
from .rule_completion import complete_one_shot_rule_for_occurrence


class TimerAdmissionRecoveryMixin:
    @staticmethod
    def _recover_handoff(
        conn: sqlite3.Connection,
        occupancy: sqlite3.Row,
        occurrence: sqlite3.Row,
        now: datetime,
    ) -> None:
        conn.execute(
            """UPDATE instance_main_core_occupancies SET kind = 'EXPRESSION',
            resource_ref = ?, version = version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND occupancy_id = ?
            AND version = ? AND generation = ? AND status = 'ACTIVE'""",
            (
                occurrence["delivery_ref"],
                encode_datetime(now),
                occupancy["profile_id"],
                occupancy["instance_id"],
                occupancy["occupancy_id"],
                occupancy["version"],
                occupancy["generation"],
            ),
        )

    def _recover_committed_run(
        self,
        conn: sqlite3.Connection,
        occupancy: sqlite3.Row,
        occurrence: sqlite3.Row,
        now: datetime,
    ) -> bool:
        run_id = self._execution_run_id(occurrence["execution_ref"])
        run = self._core_run(conn, occurrence)
        if run is None or run["status"] != "COMPLETED":
            return False
        assert run_id is not None
        batch = conn.execute(
            """SELECT batch_id FROM instance_expression_batches WHERE profile_id = ?
            AND instance_id = ? AND source_run_id = ?""",
            (occurrence["profile_id"], occurrence["instance_id"], run_id),
        ).fetchone()
        if batch is None:
            self._complete_occurrence(conn, occurrence)
            self._release_occupancy(conn, occupancy, now)
            return True
        self._recover_committed_expression(conn, occupancy, occurrence, str(batch["batch_id"]), now)
        return True

    def _recover_committed_expression(
        self,
        conn: sqlite3.Connection,
        occupancy: sqlite3.Row,
        occurrence: sqlite3.Row,
        batch_id: str,
        now: datetime,
    ) -> None:
        conn.execute(
            """UPDATE timer_occurrences SET status = 'WAITING_DELIVERY', delivery_ref = ?,
            version = version + 1 WHERE profile_id = ? AND instance_id = ?
            AND occurrence_id = ? AND version = ? AND generation = ? AND status = 'RUNNING'""",
            (
                batch_id,
                occurrence["profile_id"],
                occurrence["instance_id"],
                occurrence["occurrence_id"],
                occurrence["version"],
                occurrence["generation"],
            ),
        )
        updated = self._occupancy(conn, occupancy)
        scope = TimerScope(str(occurrence["profile_id"]), str(occurrence["instance_id"]))
        current = self._occurrence(conn, scope, str(occurrence["occurrence_id"]))
        assert updated is not None and current is not None
        self._recover_handoff(conn, updated, current, now)
        self._settle_terminal_expression_occupancy(conn, batch_id, encode_datetime(now))

    def _resolve_recovering(
        self,
        conn: sqlite3.Connection,
        occurrence: sqlite3.Row,
        now: datetime,
    ) -> bool:
        if occurrence["status"] != "RECOVERING":
            return False
        if self._recover_committed_recovery(conn, occurrence, now):
            return True
        run = self._core_run(conn, occurrence)
        task, _ = self._timer_task_for_occurrence(conn, occurrence, run)
        if self._task_owns_generation(task):
            return self._resume_recovering_retry(conn, occurrence)
        changed = conn.execute(
            """UPDATE timer_occurrences SET status = 'WAITING',
            generation = generation + 1, execution_ref = NULL, delivery_ref = NULL,
            recovery_from = NULL, version = version + 1
            WHERE profile_id = ? AND instance_id = ? AND occurrence_id = ?
            AND version = ? AND generation = ? AND status = 'RECOVERING'""",
            (
                occurrence["profile_id"],
                occurrence["instance_id"],
                occurrence["occurrence_id"],
                occurrence["version"],
                occurrence["generation"],
            ),
        ).rowcount
        if changed != 1:
            raise TimerAdmissionFenceError("Timer recovery requeue lost")
        return True

    @staticmethod
    def _resume_recovering_retry(
        conn: sqlite3.Connection,
        occurrence: sqlite3.Row,
    ) -> bool:
        changed = conn.execute(
            """UPDATE timer_occurrences SET status = 'WAITING',
            execution_ref = NULL, delivery_ref = NULL, recovery_from = NULL,
            version = version + 1 WHERE profile_id = ? AND instance_id = ?
            AND occurrence_id = ? AND version = ? AND generation = ?
            AND status = 'RECOVERING'""",
            (
                occurrence["profile_id"],
                occurrence["instance_id"],
                occurrence["occurrence_id"],
                occurrence["version"],
                occurrence["generation"],
            ),
        ).rowcount
        if changed != 1:
            raise TimerAdmissionFenceError("Timer live retry recovery lost")
        return True

    def _recover_committed_recovery(
        self,
        conn: sqlite3.Connection,
        occurrence: sqlite3.Row,
        now: datetime,
    ) -> bool:
        run_id = self._execution_run_id(occurrence["execution_ref"])
        run = self._core_run(conn, occurrence)
        if run_id is None or run is None or run["status"] != "COMPLETED":
            return False
        batch = conn.execute(
            """SELECT batch_id FROM instance_expression_batches WHERE profile_id = ?
            AND instance_id = ? AND source_run_id = ?""",
            (occurrence["profile_id"], occurrence["instance_id"], run_id),
        ).fetchone()
        if batch is None:
            changed = conn.execute(
                """UPDATE timer_occurrences SET status = 'COMPLETED', recovery_from = NULL,
                version = version + 1 WHERE profile_id = ? AND instance_id = ?
                AND occurrence_id = ? AND version = ? AND generation = ?
                AND status = 'RECOVERING'""",
                (
                    occurrence["profile_id"],
                    occurrence["instance_id"],
                    occurrence["occurrence_id"],
                    occurrence["version"],
                    occurrence["generation"],
                ),
            ).rowcount
            if changed != 1:
                raise TimerAdmissionFenceError("Timer recovery completion lost")
            complete_one_shot_rule_for_occurrence(conn, occurrence)
            return True
        batch_id = str(batch["batch_id"])
        self._restore_expression_occupancy(conn, occurrence, run_id, batch_id, now)
        changed = conn.execute(
            """UPDATE timer_occurrences SET status = 'WAITING_DELIVERY', delivery_ref = ?,
            recovery_from = NULL, version = version + 1 WHERE profile_id = ?
            AND instance_id = ? AND occurrence_id = ? AND version = ?
            AND generation = ? AND status = 'RECOVERING'""",
            (
                batch_id,
                occurrence["profile_id"],
                occurrence["instance_id"],
                occurrence["occurrence_id"],
                occurrence["version"],
                occurrence["generation"],
            ),
        ).rowcount
        if changed != 1:
            raise TimerAdmissionFenceError("Timer recovery expression handoff lost")
        self._settle_terminal_expression_occupancy(conn, batch_id, encode_datetime(now))
        return True

    def _restore_expression_occupancy(
        self,
        conn: sqlite3.Connection,
        occurrence: sqlite3.Row,
        run_id: int,
        batch_id: str,
        now: datetime,
    ) -> None:
        scope = TimerScope(str(occurrence["profile_id"]), str(occurrence["instance_id"]))
        if self._active(conn, scope) is not None:
            raise TimerAdmissionFenceError("Timer recovery found a competing occupancy")
        previous = conn.execute(
            """SELECT * FROM instance_main_core_occupancies WHERE profile_id = ?
            AND instance_id = ? AND resource_ref = ? AND generation = ?
            ORDER BY created_at DESC LIMIT 1""",
            (
                occurrence["profile_id"],
                occurrence["instance_id"],
                occurrence["occurrence_id"],
                occurrence["generation"],
            ),
        ).fetchone()
        stamp = encode_datetime(now)
        if previous is not None:
            changed = conn.execute(
                """UPDATE instance_main_core_occupancies SET kind = 'EXPRESSION',
                resource_ref = ?, status = 'ACTIVE', version = version + 1,
                lease_expires_at = ?, updated_at = ?, released_at = NULL
                WHERE profile_id = ? AND instance_id = ? AND occupancy_id = ?
                AND version = ? AND generation = ? AND kind = 'TIMER'
                AND status IN ('EXPIRED','RELEASED')""",
                (
                    batch_id,
                    stamp,
                    stamp,
                    previous["profile_id"],
                    previous["instance_id"],
                    previous["occupancy_id"],
                    previous["version"],
                    previous["generation"],
                ),
            ).rowcount
            if changed != 1:
                raise TimerAdmissionFenceError("Timer recovery occupancy restore lost")
            return
        conn.execute(
            """INSERT INTO instance_main_core_occupancies(
            profile_id, instance_id, occupancy_id, kind, resource_ref, status,
            version, generation, lease_owner, lease_token, lease_expires_at,
            created_at, updated_at) VALUES (?, ?, ?, 'EXPRESSION', ?, 'ACTIVE',
            1, ?, 'timer-runtime-recovery', ?, ?, ?, ?)""",
            (
                occurrence["profile_id"],
                occurrence["instance_id"],
                f"timer-recovery:{run_id}:{occurrence['generation']}",
                batch_id,
                occurrence["generation"],
                f"recovery:{occurrence['generation']}",
                stamp,
                stamp,
                stamp,
            ),
        )

    @staticmethod
    def _core_run(conn: sqlite3.Connection, occurrence: sqlite3.Row) -> sqlite3.Row | None:
        run_id = TimerAdmissionRecoveryMixin._execution_run_id(occurrence["execution_ref"])
        if run_id is None:
            return None
        return conn.execute(
            """SELECT status, request_json FROM instance_core_runs WHERE profile_id = ?
            AND instance_id = ? AND run_id = ?""",
            (occurrence["profile_id"], occurrence["instance_id"], run_id),
        ).fetchone()

    @staticmethod
    def _timer_task_for_occurrence(
        conn: sqlite3.Connection,
        occurrence: sqlite3.Row,
        run: sqlite3.Row | None,
    ) -> tuple[sqlite3.Row | None, bool]:
        request = TimerAdmissionRecoveryMixin._json_object(
            run["request_json"] if run is not None else None
        )
        metadata = request.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        try:
            task_id = int(metadata.get("ai_task_id") or 0)
        except (TypeError, ValueError):
            task_id = 0
        if task_id > 0:
            task = conn.execute(
                """SELECT task_id, status, lease_until, due_at FROM ai_tasks WHERE task_id = ?
                AND profile_id = ? AND instance_id = ? AND task_type = 'TIMER_RUN'""",
                (task_id, occurrence["profile_id"], occurrence["instance_id"]),
            ).fetchone()
            return task, True
        task = conn.execute(
            """SELECT task_id, status, lease_until, due_at
            FROM ai_tasks WHERE profile_id = ?
            AND instance_id = ? AND task_type = 'TIMER_RUN'
            AND idempotency_key = ? AND generation = ?""",
            (
                occurrence["profile_id"],
                occurrence["instance_id"],
                timer_run_task_idempotency_key(
                    str(occurrence["profile_id"]),
                    str(occurrence["instance_id"]),
                    str(occurrence["occurrence_id"]),
                    int(occurrence["generation"]),
                ),
                int(occurrence["generation"]) + 1,
            ),
        ).fetchone()
        return task, task is not None

    @staticmethod
    def _json_object(value: object) -> dict[str, object]:
        try:
            loaded = json.loads(str(value or "{}"))
        except ValueError:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    @staticmethod
    def _task_owns_generation(task: sqlite3.Row | None) -> bool:
        if task is None:
            return False
        return str(task["status"]) in {
            "SCHEDULED",
            "READY",
            "RUNNING",
            "PAUSE_REQUESTED",
            "PAUSED",
            "CANCEL_REQUESTED",
            "RETRY_WAIT",
            "RECOVERY_REQUIRED",
        }

    @staticmethod
    def _task_has_execution_lease(task: sqlite3.Row | None, now: datetime) -> bool:
        if task is None:
            return False
        return bool(
            str(task["status"]) in {"RUNNING", "PAUSE_REQUESTED", "CANCEL_REQUESTED"}
            and task["lease_until"] is not None
            and task["lease_until"] > encode_datetime(now)
        )

    @staticmethod
    def _extend_running_lease(
        conn: sqlite3.Connection,
        occupancy: sqlite3.Row,
        task: sqlite3.Row,
        now: datetime,
    ) -> bool:
        lease_until = str(task["lease_until"] or "")
        if not lease_until or lease_until <= str(occupancy["lease_expires_at"]):
            return False
        changed = conn.execute(
            """UPDATE instance_main_core_occupancies SET lease_expires_at = ?, updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND occupancy_id = ?
            AND version = ? AND generation = ? AND kind = 'TIMER' AND status = 'ACTIVE'
            AND lease_expires_at < ?""",
            (
                lease_until,
                encode_datetime(now),
                occupancy["profile_id"],
                occupancy["instance_id"],
                occupancy["occupancy_id"],
                occupancy["version"],
                occupancy["generation"],
                lease_until,
            ),
        ).rowcount
        return changed == 1

    @staticmethod
    def _execution_run_id(value: object) -> int | None:
        text = str(value or "")
        if not text.startswith("core-run:"):
            return None
        try:
            run_id = int(text.removeprefix("core-run:"))
        except ValueError:
            return None
        return run_id if run_id > 0 else None

    def _mark_running_recovery(
        self,
        conn: sqlite3.Connection,
        occupancy: sqlite3.Row,
        occurrence: sqlite3.Row,
        now: datetime,
    ) -> bool:
        changed = conn.execute(
            """UPDATE timer_occurrences SET status = 'RECOVERING', recovery_from = 'RUNNING',
            version = version + 1 WHERE profile_id = ? AND instance_id = ?
            AND occurrence_id = ? AND version = ? AND generation = ? AND status = 'RUNNING'""",
            (
                occurrence["profile_id"],
                occurrence["instance_id"],
                occurrence["occurrence_id"],
                occurrence["version"],
                occurrence["generation"],
            ),
        ).rowcount
        if changed != 1:
            return False
        if not self._expire_occupancy(conn, occupancy, now):
            raise TimerAdmissionFenceError("Timer recovery occupancy expiry lost")
        return True


_PLAYER_BUFFER = """SELECT 1 FROM conversation_turn_buffer_batches
    WHERE profile_id = ? AND instance_id = ? AND (
      status IN ('PENDING','CLASSIFYING','WAITING')
      OR (status = 'CLAIMED' AND main_core_task_ref IS NULL)
    ) LIMIT 1"""
_PLAYER_GROUP = """SELECT 1 FROM group_flow_windows
    WHERE profile_id = ? AND instance_id = ? AND (
      status IN ('COLLECTING','JUDGING','READY')
      OR (status = 'RUNNING' AND main_core_task_ref IS NULL)
    ) LIMIT 1"""
_PLAYER_HELD = """SELECT 1 FROM instance_messages
    WHERE profile_id = ? AND instance_id = ? AND direction = 'INBOUND'
      AND knowledge_eligibility = 'HELD' LIMIT 1"""


def has_waiting_player_work(conn: sqlite3.Connection, profile_id: str, instance_id: str) -> bool:
    params = (profile_id, instance_id)
    return any(conn.execute(sql, params).fetchone() is not None for sql in _player_queries())


def _player_queries() -> tuple[str, str, str]:
    return _PLAYER_BUFFER, _PLAYER_GROUP, _PLAYER_HELD


def settle_terminal_expression_occupancy(
    conn: sqlite3.Connection, batch_id: str | None, now: str
) -> bool:
    batch_status = _locally_terminal_batch_status(conn, batch_id)
    if batch_status is None:
        return False
    occupancy = conn.execute(
        """SELECT * FROM instance_main_core_occupancies
        WHERE kind = 'EXPRESSION' AND resource_ref = ? AND status = 'ACTIVE'""",
        (batch_id,),
    ).fetchone()
    if occupancy is None:
        return False
    occurrence = conn.execute(
        """SELECT * FROM timer_occurrences WHERE profile_id = ? AND instance_id = ?
        AND delivery_ref = ? AND status = 'WAITING_DELIVERY' AND generation = ?""",
        (
            occupancy["profile_id"],
            occupancy["instance_id"],
            batch_id,
            occupancy["generation"],
        ),
    ).fetchone()
    if occurrence is None:
        return False
    occurrence_changed = conn.execute(
        """UPDATE timer_occurrences SET status = ?, version = version + 1
        WHERE profile_id = ? AND instance_id = ? AND occurrence_id = ?
        AND version = ? AND generation = ? AND status = 'WAITING_DELIVERY'""",
        (
            terminal_timer_occurrence_status(batch_status),
            occurrence["profile_id"],
            occurrence["instance_id"],
            occurrence["occurrence_id"],
            occurrence["version"],
            occurrence["generation"],
        ),
    ).rowcount
    if occurrence_changed != 1:
        return False
    if terminal_timer_occurrence_status(batch_status) == "COMPLETED":
        complete_one_shot_rule_for_occurrence(conn, occurrence)
    changed = conn.execute(
        """UPDATE instance_main_core_occupancies SET status = 'RELEASED',
        version = version + 1, updated_at = ?, released_at = ?
        WHERE profile_id = ? AND instance_id = ? AND occupancy_id = ?
        AND version = ? AND generation = ? AND kind = 'EXPRESSION'
        AND resource_ref = ? AND status = 'ACTIVE'""",
        (
            now,
            now,
            occupancy["profile_id"],
            occupancy["instance_id"],
            occupancy["occupancy_id"],
            occupancy["version"],
            occupancy["generation"],
            batch_id,
        ),
    ).rowcount
    if changed != 1:
        raise TimerAdmissionFenceError("Timer expression occupancy release lost")
    return True


def _locally_terminal_batch_status(
    conn: sqlite3.Connection,
    batch_id: str | None,
) -> str | None:
    if not batch_id:
        return None
    batch = conn.execute(
        "SELECT status FROM instance_expression_batches WHERE batch_id = ?", (batch_id,)
    ).fetchone()
    if batch is None or str(batch["status"]) == "ACTIVE":
        return None
    pending_outbox = conn.execute(
        """SELECT 1 FROM instance_outbox WHERE expression_batch_id = ?
        AND status IN ('PENDING','SENDING') LIMIT 1""",
        (batch_id,),
    ).fetchone()
    pending_retraction = conn.execute(
        """SELECT 1 FROM message_retraction_actions WHERE expression_batch_id = ?
        AND status IN ('PENDING','SENDING') LIMIT 1""",
        (batch_id,),
    ).fetchone()
    if pending_outbox is not None or pending_retraction is not None:
        return None
    return str(batch["status"])


class SqliteTimerAdmissionRepository(TimerAdmissionRecoveryMixin, SqliteRepository):
    _settle_terminal_expression_occupancy = staticmethod(settle_terminal_expression_occupancy)

    async def is_instance_occupied(self, scope: TimerScope) -> bool:
        return await self.uow.run(lambda conn: self._active(conn, scope) is not None)

    async def claim_next_timer(self, command: ClaimNextTimerCommand) -> TimerAdmissionResult:
        return await self.uow.run(lambda conn: self._claim_next(conn, command))

    def _claim_next(
        self, conn: sqlite3.Connection, command: ClaimNextTimerCommand
    ) -> TimerAdmissionResult:
        active = self._active(conn, command.scope)
        replay = self._replay(active, conn, command)
        if replay is not None:
            return replay
        if active is not None:
            active = self._expire_unstarted_claim(conn, active, command.now)
        if active is not None:
            return TimerAdmissionResult(TimerClaimOutcome.OCCUPIED)
        if has_waiting_player_work(conn, command.scope.profile_id, command.scope.instance_id):
            return TimerAdmissionResult(TimerClaimOutcome.PLAYER_WAITING)
        occurrence = self._next_waiting(conn, command.scope)
        if occurrence is None:
            return TimerAdmissionResult(TimerClaimOutcome.EMPTY)
        return self._claim_occurrence(conn, occurrence, command)

    async def begin_timer_provider(self, command: BeginTimerProviderCommand) -> TimerProviderResult:
        return await self.uow.run(lambda conn: self._begin_provider(conn, command))

    def _begin_provider(
        self, conn: sqlite3.Connection, command: BeginTimerProviderCommand
    ) -> TimerProviderResult:
        occurrence, occupancy = self._fenced(conn, command.fence, command.now, "CLAIMED")
        if has_waiting_player_work(
            conn, command.fence.scope.profile_id, command.fence.scope.instance_id
        ):
            return self._supersede(conn, occurrence, occupancy, command.now)
        now = encode_datetime(command.now)
        conn.execute(
            """UPDATE timer_occurrences SET status = 'RUNNING', execution_ref = ?,
            version = version + 1 WHERE profile_id = ? AND instance_id = ?
            AND occurrence_id = ? AND version = ? AND generation = ?""",
            (
                command.execution_ref.value,
                occurrence["profile_id"],
                occurrence["instance_id"],
                occurrence["occurrence_id"],
                occurrence["version"],
                occurrence["generation"],
            ),
        )
        self._touch_occupancy(conn, occupancy, now)
        return self._provider_result(conn, command.fence.scope, command.fence.occurrence_id.value)

    async def complete_timer_noop(self, command: CompleteTimerNoOpCommand) -> TimerSettlementResult:
        return await self.uow.run(lambda conn: self._complete_noop(conn, command))

    def _complete_noop(
        self, conn: sqlite3.Connection, command: CompleteTimerNoOpCommand
    ) -> TimerSettlementResult:
        occurrence, occupancy = self._fenced(conn, command.fence, command.now, "RUNNING")
        self._complete_occurrence(conn, occurrence)
        self._release_occupancy(conn, occupancy, command.now)
        return self._settlement(conn, command.fence.scope, command.fence.occurrence_id.value)

    async def supersede_timer_run(self, command: SupersedeTimerRunCommand) -> TimerSettlementResult:
        return await self.uow.run(lambda conn: self._supersede_run(conn, command))

    async def retry_timer_run(self, command: RetryTimerRunCommand) -> TimerSettlementResult:
        return await self.uow.run(lambda conn: self._retry_run(conn, command))

    def _retry_run(
        self, conn: sqlite3.Connection, command: RetryTimerRunCommand
    ) -> TimerSettlementResult:
        occurrence, occupancy = self._fenced(conn, command.fence, command.now, "RUNNING")
        changed = conn.execute(
            """UPDATE timer_occurrences SET status = 'WAITING',
            execution_ref = NULL, delivery_ref = NULL, recovery_from = NULL,
            version = version + 1 WHERE profile_id = ? AND instance_id = ?
            AND occurrence_id = ? AND version = ? AND generation = ?
            AND status = 'RUNNING'""",
            (
                occurrence["profile_id"],
                occurrence["instance_id"],
                occurrence["occurrence_id"],
                occurrence["version"],
                occurrence["generation"],
            ),
        ).rowcount
        if changed != 1:
            raise TimerAdmissionFenceError("Timer retry release lost")
        self._release_occupancy(conn, occupancy, command.now)
        return self._settlement(conn, command.fence.scope, command.fence.occurrence_id.value)

    def _supersede_run(
        self, conn: sqlite3.Connection, command: SupersedeTimerRunCommand
    ) -> TimerSettlementResult:
        occurrence, occupancy = self._fenced_any(
            conn, command.fence, command.now, {"CLAIMED", "RUNNING"}
        )
        changed = conn.execute(
            """UPDATE timer_occurrences SET status = 'WAITING', generation = generation + 1,
            execution_ref = NULL, delivery_ref = NULL, recovery_from = NULL,
            version = version + 1 WHERE profile_id = ? AND instance_id = ?
            AND occurrence_id = ? AND version = ? AND generation = ?
            AND status IN ('CLAIMED','RUNNING')""",
            (
                occurrence["profile_id"],
                occurrence["instance_id"],
                occurrence["occurrence_id"],
                occurrence["version"],
                occurrence["generation"],
            ),
        ).rowcount
        if changed != 1:
            raise TimerAdmissionFenceError("Timer supersession lost")
        self._release_occupancy(conn, occupancy, command.now)
        return self._settlement(conn, command.fence.scope, command.fence.occurrence_id.value)

    async def handoff_timer_expression(
        self, command: HandoffTimerExpressionCommand
    ) -> TimerSettlementResult:
        return await self.uow.run(lambda conn: self._handoff(conn, command))

    def _handoff(
        self, conn: sqlite3.Connection, command: HandoffTimerExpressionCommand
    ) -> TimerSettlementResult:
        occurrence, occupancy = self._fenced(conn, command.fence, command.now, "RUNNING")
        self._validate_batch(conn, command, occurrence)
        now = encode_datetime(command.now)
        conn.execute(
            """UPDATE timer_occurrences SET status = 'WAITING_DELIVERY', delivery_ref = ?,
            version = version + 1 WHERE profile_id = ? AND instance_id = ?
            AND occurrence_id = ? AND version = ? AND generation = ?""",
            (
                command.delivery_ref.value,
                occurrence["profile_id"],
                occurrence["instance_id"],
                occurrence["occurrence_id"],
                occurrence["version"],
                occurrence["generation"],
            ),
        )
        conn.execute(
            """UPDATE instance_main_core_occupancies SET kind = 'EXPRESSION',
            resource_ref = ?, version = version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND occupancy_id = ?
            AND version = ? AND generation = ? AND status = 'ACTIVE'""",
            (
                command.delivery_ref.value,
                now,
                occupancy["profile_id"],
                occupancy["instance_id"],
                occupancy["occupancy_id"],
                occupancy["version"],
                occupancy["generation"],
            ),
        )
        settle_terminal_expression_occupancy(conn, command.delivery_ref.value, now)
        return self._settlement(conn, command.fence.scope, command.fence.occurrence_id.value)

    async def reconcile_timer_occupancy(self, scope: TimerScope, *, now: datetime) -> int:
        require_aware(now)
        return await self.uow.run(lambda conn: self._reconcile(conn, scope, now))

    def _reconcile(self, conn: sqlite3.Connection, scope: TimerScope, now: datetime) -> int:
        active = self._active(conn, scope)
        if active is None:
            recovering = self._recovering_occurrence(conn, scope)
            if recovering is None:
                return 0
            return int(self._resolve_recovering(conn, recovering, now))
        if active["kind"] == "TIMER":
            occurrence = self._occurrence_for_occupancy(conn, active)
            if occurrence is None:
                return int(self._expire_occupancy(conn, active, now))
            if occurrence["status"] == "CLAIMED":
                return int(self._expire_unstarted_claim(conn, active, now) is None)
            if occurrence["status"] == "WAITING_DELIVERY" and occurrence["delivery_ref"]:
                self._recover_handoff(conn, active, occurrence, now)
                settle_terminal_expression_occupancy(
                    conn, str(occurrence["delivery_ref"]), encode_datetime(now)
                )
                return 1
            if occurrence["status"] == "RUNNING":
                return int(self._reconcile_running(conn, scope, active, occurrence, now))
        if active["kind"] == "EXPRESSION":
            return int(
                settle_terminal_expression_occupancy(
                    conn, str(active["resource_ref"]), encode_datetime(now)
                )
            )
        return 0

    def _reconcile_running(
        self,
        conn: sqlite3.Connection,
        scope: TimerScope,
        active: sqlite3.Row,
        occurrence: sqlite3.Row,
        now: datetime,
    ) -> bool:
        run = self._core_run(conn, occurrence)
        task, has_task_link = self._timer_task_for_occurrence(conn, occurrence, run)
        has_execution_lease = self._task_has_execution_lease(task, now)
        if has_execution_lease:
            return self._extend_running_lease(conn, active, task, now)
        if self._recover_committed_run(conn, active, occurrence, now):
            return True
        run_is_terminal = run is not None and str(run["status"]) != "RUNNING"
        should_recover = (
            run_is_terminal
            or (has_task_link and not has_execution_lease)
            or active["lease_expires_at"] <= encode_datetime(now)
        )
        if not should_recover or not self._mark_running_recovery(conn, active, occurrence, now):
            return False
        recovering = self._occurrence(conn, scope, str(occurrence["occurrence_id"]))
        if recovering is None or recovering["status"] != "RECOVERING":
            raise TimerAdmissionFenceError("Timer recovery transition lost")
        self._resolve_recovering(conn, recovering, now)
        return True

    @staticmethod
    def _active(conn: sqlite3.Connection, scope: TimerScope) -> sqlite3.Row | None:
        return conn.execute(
            """SELECT * FROM instance_main_core_occupancies WHERE profile_id = ?
            AND instance_id = ? AND status = 'ACTIVE'""",
            (scope.profile_id, scope.instance_id),
        ).fetchone()

    @staticmethod
    def _recovering_occurrence(conn: sqlite3.Connection, scope: TimerScope) -> sqlite3.Row | None:
        return conn.execute(
            """SELECT * FROM timer_occurrences WHERE profile_id = ? AND instance_id = ?
            AND status = 'RECOVERING'
            ORDER BY original_due_at, created_sequence, stable_ref LIMIT 1""",
            (scope.profile_id, scope.instance_id),
        ).fetchone()

    def _replay(
        self,
        active: sqlite3.Row | None,
        conn: sqlite3.Connection,
        command: ClaimNextTimerCommand,
    ) -> TimerAdmissionResult | None:
        if active is None or (
            active["occupancy_id"] != command.occupancy_id
            or active["lease_owner"] != command.lease_owner
            or active["lease_token"] != command.lease_token
            or active["kind"] != "TIMER"
        ):
            return None
        occurrence = self._occurrence_for_occupancy(conn, active)
        if (
            occurrence is None
            or occurrence["status"] != "CLAIMED"
            or active["lease_expires_at"] <= encode_datetime(command.now)
        ):
            return None
        return self._admission_result(occurrence, active, replayed=True)

    def _expire_unstarted_claim(
        self, conn: sqlite3.Connection, active: sqlite3.Row, now: datetime
    ) -> sqlite3.Row | None:
        if active["kind"] != "TIMER" or active["lease_expires_at"] > encode_datetime(now):
            return active
        occurrence = self._occurrence_for_occupancy(conn, active)
        if occurrence is None or occurrence["status"] != "CLAIMED":
            return active
        task, _ = self._timer_task_for_occurrence(conn, occurrence, None)
        generation_increment = 0 if self._task_owns_generation(task) else 1
        conn.execute(
            """UPDATE timer_occurrences SET status = 'WAITING', generation = generation + ?,
            version = version + 1 WHERE profile_id = ? AND instance_id = ?
            AND occurrence_id = ? AND status = 'CLAIMED' AND version = ? AND generation = ?""",
            (
                generation_increment,
                occurrence["profile_id"],
                occurrence["instance_id"],
                occurrence["occurrence_id"],
                occurrence["version"],
                occurrence["generation"],
            ),
        )
        self._expire_occupancy(conn, active, now)
        return None

    @staticmethod
    def _next_waiting(conn: sqlite3.Connection, scope: TimerScope) -> sqlite3.Row | None:
        return conn.execute(
            """SELECT * FROM timer_occurrences WHERE profile_id = ? AND instance_id = ?
            AND status = 'WAITING' ORDER BY original_due_at, created_sequence, stable_ref LIMIT 1""",
            (scope.profile_id, scope.instance_id),
        ).fetchone()

    def _claim_occurrence(
        self, conn: sqlite3.Connection, occurrence: sqlite3.Row, command: ClaimNextTimerCommand
    ) -> TimerAdmissionResult:
        now = encode_datetime(command.now)
        changed = conn.execute(
            """UPDATE timer_occurrences SET status = 'CLAIMED', version = version + 1
            WHERE profile_id = ? AND instance_id = ? AND occurrence_id = ?
            AND status = 'WAITING' AND version = ? AND generation = ?""",
            (
                occurrence["profile_id"],
                occurrence["instance_id"],
                occurrence["occurrence_id"],
                occurrence["version"],
                occurrence["generation"],
            ),
        ).rowcount
        if changed != 1:
            raise TimerAdmissionFenceError("Timer occurrence claim lost")
        conn.execute(
            """INSERT INTO instance_main_core_occupancies(
            profile_id, instance_id, occupancy_id, kind, resource_ref, status,
            version, generation, lease_owner, lease_token, lease_expires_at,
            created_at, updated_at) VALUES (?, ?, ?, 'TIMER', ?, 'ACTIVE',
            1, ?, ?, ?, ?, ?, ?)""",
            (
                command.scope.profile_id,
                command.scope.instance_id,
                command.occupancy_id,
                occurrence["occurrence_id"],
                occurrence["generation"],
                command.lease_owner,
                command.lease_token,
                encode_datetime(command.lease_expires_at),
                now,
                now,
            ),
        )
        current = self._occurrence(conn, command.scope, str(occurrence["occurrence_id"]))
        active = self._active(conn, command.scope)
        assert current is not None and active is not None
        return self._admission_result(current, active)

    def _fenced(
        self, conn: sqlite3.Connection, fence: TimerRunFence, now: datetime, status: str
    ) -> tuple[sqlite3.Row, sqlite3.Row]:
        occurrence = self._occurrence(conn, fence.scope, fence.occurrence_id.value)
        occupancy = self._active(conn, fence.scope)
        if occurrence is None or occupancy is None:
            raise TimerAdmissionFenceError("stale Timer admission fence")
        if not self._matches_fence(occurrence, occupancy, fence, now, status):
            raise TimerAdmissionFenceError("stale Timer admission fence")
        return occurrence, occupancy

    def _fenced_any(
        self,
        conn: sqlite3.Connection,
        fence: TimerRunFence,
        now: datetime,
        statuses: set[str],
    ) -> tuple[sqlite3.Row, sqlite3.Row]:
        occurrence = self._occurrence(conn, fence.scope, fence.occurrence_id.value)
        occupancy = self._active(conn, fence.scope)
        if occurrence is None or occupancy is None or occurrence["status"] not in statuses:
            raise TimerAdmissionFenceError("stale Timer admission fence")
        if not self._matches_fence(occurrence, occupancy, fence, now, str(occurrence["status"])):
            raise TimerAdmissionFenceError("stale Timer admission fence")
        return occurrence, occupancy

    @staticmethod
    def _matches_fence(
        occurrence: sqlite3.Row,
        occupancy: sqlite3.Row,
        fence: TimerRunFence,
        now: datetime,
        status: str,
    ) -> bool:
        return bool(
            occurrence["status"] == status
            and occurrence["version"] == fence.occurrence_version
            and occurrence["generation"] == fence.generation
            and occupancy["occupancy_id"] == fence.occupancy_id
            and occupancy["version"] == fence.occupancy_version
            and occupancy["generation"] == fence.generation
            and occupancy["lease_owner"] == fence.lease_owner
            and occupancy["lease_token"] == fence.lease_token
            and occupancy["kind"] == "TIMER"
            and occupancy["lease_expires_at"] > encode_datetime(now)
        )

    def _supersede(
        self,
        conn: sqlite3.Connection,
        occurrence: sqlite3.Row,
        occupancy: sqlite3.Row,
        now: datetime,
    ) -> TimerProviderResult:
        conn.execute(
            """UPDATE timer_occurrences SET status = 'WAITING', generation = generation + 1,
            version = version + 1 WHERE profile_id = ? AND instance_id = ?
            AND occurrence_id = ? AND version = ? AND generation = ?""",
            (
                occurrence["profile_id"],
                occurrence["instance_id"],
                occurrence["occurrence_id"],
                occurrence["version"],
                occurrence["generation"],
            ),
        )
        self._release_occupancy(conn, occupancy, now)
        current = self._occurrence(
            conn,
            TimerScope(str(occurrence["profile_id"]), str(occurrence["instance_id"])),
            str(occurrence["occurrence_id"]),
        )
        released = self._occupancy(conn, occupancy)
        assert current is not None and released is not None
        return TimerProviderResult(
            TimerProviderOutcome.SUPERSEDED,
            decode_occurrence(dict(current)),
            decode_occupancy(dict(released)),
            None,
        )

    def _provider_result(
        self, conn: sqlite3.Connection, scope: TimerScope, occurrence_id: str
    ) -> TimerProviderResult:
        occurrence = self._occurrence(conn, scope, occurrence_id)
        occupancy = self._active(conn, scope)
        assert occurrence is not None and occupancy is not None
        fence = self._fence(occurrence, occupancy)
        return TimerProviderResult(
            TimerProviderOutcome.STARTED,
            decode_occurrence(dict(occurrence)),
            decode_occupancy(dict(occupancy)),
            fence,
        )

    @staticmethod
    def _touch_occupancy(conn: sqlite3.Connection, occupancy: sqlite3.Row, now: str) -> None:
        conn.execute(
            """UPDATE instance_main_core_occupancies SET version = version + 1,
            updated_at = ? WHERE profile_id = ? AND instance_id = ? AND occupancy_id = ?
            AND version = ? AND generation = ? AND status = 'ACTIVE'""",
            (
                now,
                occupancy["profile_id"],
                occupancy["instance_id"],
                occupancy["occupancy_id"],
                occupancy["version"],
                occupancy["generation"],
            ),
        )

    @staticmethod
    def _complete_occurrence(conn: sqlite3.Connection, occurrence: sqlite3.Row) -> None:
        changed = conn.execute(
            """UPDATE timer_occurrences SET status = 'COMPLETED', version = version + 1
            WHERE profile_id = ? AND instance_id = ? AND occurrence_id = ?
            AND version = ? AND generation = ? AND status = 'RUNNING'""",
            (
                occurrence["profile_id"],
                occurrence["instance_id"],
                occurrence["occurrence_id"],
                occurrence["version"],
                occurrence["generation"],
            ),
        ).rowcount
        if changed == 1:
            complete_one_shot_rule_for_occurrence(conn, occurrence)

    @staticmethod
    def _release_occupancy(conn: sqlite3.Connection, occupancy: sqlite3.Row, now: datetime) -> None:
        stamp = encode_datetime(now)
        conn.execute(
            """UPDATE instance_main_core_occupancies SET status = 'RELEASED',
            version = version + 1, updated_at = ?, released_at = ?
            WHERE profile_id = ? AND instance_id = ? AND occupancy_id = ?
            AND version = ? AND generation = ? AND status = 'ACTIVE'""",
            (
                stamp,
                stamp,
                occupancy["profile_id"],
                occupancy["instance_id"],
                occupancy["occupancy_id"],
                occupancy["version"],
                occupancy["generation"],
            ),
        )

    @staticmethod
    def _expire_occupancy(conn: sqlite3.Connection, occupancy: sqlite3.Row, now: datetime) -> bool:
        stamp = encode_datetime(now)
        changed = conn.execute(
            """UPDATE instance_main_core_occupancies SET status = 'EXPIRED',
            version = version + 1, updated_at = ?, released_at = ?
            WHERE profile_id = ? AND instance_id = ? AND occupancy_id = ?
            AND version = ? AND generation = ? AND status = 'ACTIVE'""",
            (
                stamp,
                stamp,
                occupancy["profile_id"],
                occupancy["instance_id"],
                occupancy["occupancy_id"],
                occupancy["version"],
                occupancy["generation"],
            ),
        ).rowcount
        return changed == 1

    @staticmethod
    def _validate_batch(
        conn: sqlite3.Connection,
        command: HandoffTimerExpressionCommand,
        occurrence: sqlite3.Row,
    ) -> None:
        batch = conn.execute(
            """SELECT * FROM instance_expression_batches WHERE batch_id = ?
            AND profile_id = ? AND instance_id = ? AND source_run_id = ?""",
            (
                command.delivery_ref.value,
                command.fence.scope.profile_id,
                command.fence.scope.instance_id,
                command.source_run_id,
            ),
        ).fetchone()
        if batch is None or occurrence["execution_ref"] != f"core-run:{command.source_run_id}":
            raise TimerAdmissionFenceError("expression batch does not belong to Timer run")

    def _settlement(
        self, conn: sqlite3.Connection, scope: TimerScope, occurrence_id: str
    ) -> TimerSettlementResult:
        occurrence = self._occurrence(conn, scope, occurrence_id)
        assert occurrence is not None
        occupancy = conn.execute(
            """SELECT * FROM instance_main_core_occupancies WHERE profile_id = ?
            AND instance_id = ? AND resource_ref IN (?, ?)
            ORDER BY created_at DESC LIMIT 1""",
            (scope.profile_id, scope.instance_id, occurrence_id, occurrence["delivery_ref"]),
        ).fetchone()
        assert occupancy is not None
        return TimerSettlementResult(
            decode_occurrence(dict(occurrence)), decode_occupancy(dict(occupancy))
        )

    @staticmethod
    def _occurrence(
        conn: sqlite3.Connection, scope: TimerScope, occurrence_id: str
    ) -> sqlite3.Row | None:
        return conn.execute(
            """SELECT * FROM timer_occurrences WHERE profile_id = ? AND instance_id = ?
            AND occurrence_id = ?""",
            (scope.profile_id, scope.instance_id, occurrence_id),
        ).fetchone()

    @staticmethod
    def _occurrence_for_occupancy(
        conn: sqlite3.Connection, occupancy: sqlite3.Row
    ) -> sqlite3.Row | None:
        return conn.execute(
            """SELECT * FROM timer_occurrences WHERE profile_id = ? AND instance_id = ?
            AND occurrence_id = ?""",
            (occupancy["profile_id"], occupancy["instance_id"], occupancy["resource_ref"]),
        ).fetchone()

    @staticmethod
    def _occupancy(conn: sqlite3.Connection, previous: sqlite3.Row) -> sqlite3.Row | None:
        return conn.execute(
            """SELECT * FROM instance_main_core_occupancies WHERE profile_id = ?
            AND instance_id = ? AND occupancy_id = ?""",
            (previous["profile_id"], previous["instance_id"], previous["occupancy_id"]),
        ).fetchone()

    @staticmethod
    def _fence(occurrence: sqlite3.Row, occupancy: sqlite3.Row) -> TimerRunFence:
        return TimerRunFence(
            TimerScope(str(occurrence["profile_id"]), str(occurrence["instance_id"])),
            decode_occurrence(dict(occurrence)).occurrence_id,
            int(occurrence["version"]),
            int(occurrence["generation"]),
            str(occupancy["occupancy_id"]),
            int(occupancy["version"]),
            str(occupancy["lease_owner"]),
            str(occupancy["lease_token"]),
        )

    def _admission_result(
        self, occurrence: sqlite3.Row, occupancy: sqlite3.Row, *, replayed: bool = False
    ) -> TimerAdmissionResult:
        return TimerAdmissionResult(
            TimerClaimOutcome.CLAIMED,
            decode_occurrence(dict(occurrence)),
            decode_occupancy(dict(occupancy)),
            self._fence(occurrence, occupancy),
            replayed,
        )


__all__ = [
    "SqliteTimerAdmissionRepository",
    "has_waiting_player_work",
    "settle_terminal_expression_occupancy",
]
