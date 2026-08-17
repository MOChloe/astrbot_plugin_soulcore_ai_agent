from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

from ....contracts.delivery_visibility import DIALOGUE_CONTINUITY_OUTBOUND_STATUSES
from ....contracts.group_flow import (
    GroupFlowDiagnostic,
    GroupFlowSourceMessage,
    GroupFlowStatus,
    GroupFlowWindow,
    GroupReplyRelocationCheck,
)
from ....contracts.message_reference import with_inbound_reply_projection
from ....storage.sqlite.codec import _dt, _parse
from ....storage.sqlite.inbound_admission_transactions import InboundAdmissionTransactions
from ....storage.sqlite.repository import SqliteRepository
from ..policy import group_interjection_check_probability
from .append import GroupFlowAppendSql
from .claims import GroupFlowClaimSql
from .policy import GroupFlowPolicySql
from .relocation_apply import GroupReplyRelocationApplySql
from .relocation_claim import GroupReplyRelocationClaimSql
from .settlement import GroupFlowSettlementSql


class GroupFlowOperationsSql(GroupFlowClaimSql, GroupFlowSettlementSql):
    """Transactional window progression: admission, claims, settlement and recovery."""


class GroupReplyRelocationStateSql:
    async def record_reply_relocation_decision(
        self,
        check: GroupReplyRelocationCheck,
        *,
        recheck_after_seconds: int,
        error_code: str,
        now: datetime,
    ) -> bool:
        delay = max(0, min(60, int(recheck_after_seconds)))
        now_text = _dt(now)
        recheck_at = _dt(now + timedelta(seconds=delay)) if delay else None
        assert now_text is not None
        changed = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE group_reply_relocation_states SET
                last_checked_message_id = ?, candidate_through_message_id = ?,
                candidate_recheck_at = ?, check_owner = NULL, check_until = NULL,
                check_through_message_id = NULL, check_final = 0,
                last_error_code = ?, updated_at = ?
                WHERE window_id = ? AND check_token = ?
                  AND check_through_message_id = ? AND check_owner IS NOT NULL
                  AND relocation_count = 0
                  AND EXISTS (
                    SELECT 1 FROM group_flow_windows protected
                    WHERE protected.window_id = group_reply_relocation_states.window_id
                      AND protected.status IN ('RUNNING','WAITING_FIRST_ATTEMPT')
                      AND protected.version = ? AND protected.lease_token = ?
                      AND protected.frozen_through_message_id = ?
                      AND protected.main_core_task_ref = ?
                  )
                  AND EXISTS (
                    SELECT 1 FROM group_flow_windows delta
                    WHERE delta.window_id = ? AND delta.status = 'COLLECTING'
                      AND delta.last_message_id = ?
                  )""",
                (
                    check.delta_through_message_id,
                    check.delta_through_message_id if delay else None,
                    recheck_at,
                    str(error_code or "")[:120],
                    now_text,
                    check.fence.window_id,
                    check.check_token,
                    check.delta_through_message_id,
                    check.fence.version,
                    check.fence.lease_token,
                    check.fence.frozen_through_message_id,
                    check.fence.main_core_task_ref,
                    check.delta_window_id,
                    check.delta_through_message_id,
                ),
            ),
            transaction=True,
        )
        return changed.rowcount == 1

    async def release_reply_relocation_check(
        self, check: GroupReplyRelocationCheck, *, now: datetime
    ) -> bool:
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE group_reply_relocation_states SET check_owner = NULL,
                check_until = NULL, check_through_message_id = NULL, check_final = 0,
                updated_at = ? WHERE window_id = ? AND check_token = ?""",
                (_dt(now), check.fence.window_id, check.check_token),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1


class GroupReplyRelocationSql(
    GroupReplyRelocationClaimSql,
    GroupReplyRelocationApplySql,
    GroupReplyRelocationStateSql,
):
    """Assemble the independently-owned relocation claim, apply, and state surfaces."""


class SqliteGroupFlowRepository(
    GroupFlowPolicySql,
    GroupFlowAppendSql,
    GroupFlowOperationsSql,
    GroupReplyRelocationSql,
    SqliteRepository,
):
    """SQLite implementation; all state transitions remain transaction fenced."""

    def __init__(
        self,
        engine,
        inbound_admission: InboundAdmissionTransactions,
    ) -> None:
        SqliteRepository.__init__(self, engine)
        self._inbound_admission = inbound_admission

    async def get_window(
        self, profile_id: str, instance_id: str, window_id: str
    ) -> GroupFlowWindow | None:
        row = await self.db.fetch_one(
            """SELECT * FROM group_flow_windows WHERE profile_id = ?
            AND instance_id = ? AND window_id = ?""",
            (profile_id, instance_id, window_id),
        )
        return await self._window(row) if row is not None else None

    async def load_window_messages(
        self, profile_id: str, instance_id: str, window_id: str
    ) -> tuple[GroupFlowSourceMessage, ...]:
        rows = await self.db.fetch_all(
            """SELECT message.message_id, message.sender_id, message.sender_name,
            message.plain_text, message.components_json, message.occurred_at,
            message.role, message.direction, message.delivery_status,
            member.media_cluster_keys_json
            FROM group_flow_window_members member
            JOIN instance_messages message
              ON message.profile_id = member.profile_id
             AND message.instance_id = member.instance_id
             AND message.message_id = member.message_id
            WHERE member.profile_id = ? AND member.instance_id = ?
              AND member.window_id = ? ORDER BY member.ordinal""",
            (profile_id, instance_id, window_id),
        )
        return tuple(self._source_message(row) for row in rows)

    async def load_judge_messages(
        self, profile_id: str, instance_id: str, window_id: str
    ) -> tuple[GroupFlowSourceMessage, ...]:
        placeholders = ",".join("?" for _ in DIALOGUE_CONTINUITY_OUTBOUND_STATUSES)
        window = await self.get_window(profile_id, instance_id, window_id)
        if window is None:
            return ()
        context_rows = await self.db.fetch_all(
            f"""SELECT message_id, sender_id,
            CASE WHEN direction = 'OUTBOUND' THEN COALESCE(
                NULLIF(profile.name, ''),
                sender_name
            ) ELSE sender_name END AS sender_name,
            plain_text,
            components_json, occurred_at, role, direction, delivery_status,
            '[]' AS media_cluster_keys_json FROM instance_messages message
            JOIN role_profiles profile ON profile.profile_id = message.profile_id
            WHERE message.profile_id = ? AND message.instance_id = ? AND message.message_id < ?
            AND (message.direction = 'INBOUND' OR (
                message.direction = 'OUTBOUND' AND message.role = 'assistant'
                AND message.delivery_status IN ({placeholders})
              )
            ) ORDER BY message.message_id DESC LIMIT 64""",
            (
                profile_id,
                instance_id,
                window.first_message_id,
                *DIALOGUE_CONTINUITY_OUTBOUND_STATUSES,
            ),
        )
        window_messages = await self.load_window_messages(profile_id, instance_id, window_id)
        prefix = tuple(self._source_message(row) for row in reversed(context_rows))
        return (*prefix, *window_messages)

    async def diagnostic(self, profile_id: str, instance_id: str) -> GroupFlowDiagnostic:
        rows = await self.db.fetch_all(
            """SELECT * FROM group_flow_windows WHERE profile_id = ? AND instance_id = ?
            AND status IN ('COLLECTING','JUDGING','READY','RUNNING','WAITING_FIRST_ATTEMPT')
            ORDER BY CASE status WHEN 'COLLECTING' THEN 1 ELSE 0 END, created_at""",
            (profile_id, instance_id),
        )
        windows = [await self._window(row) for row in rows]
        pipeline = next(
            (item for item in windows if item.status != GroupFlowStatus.COLLECTING), None
        )
        collecting = next(
            (item for item in windows if item.status == GroupFlowStatus.COLLECTING), None
        )
        primary = pipeline or collecting
        algorithm: dict[str, object] = {}
        state = await self.db.fetch_one(
            """SELECT activity_released_through_message_id
            FROM group_flow_instance_state WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        )
        if primary is not None:
            algorithm = {
                "rate_ewma": primary.rate_ewma,
                "repeat_ratio": primary.repeat_ratio,
                "judge_threshold": primary.judge_threshold,
                "next_judge_at": primary.next_judge_at,
                "quiet_due_at": primary.quiet_due_at,
                "dynamic_due_at": primary.dynamic_due_at,
                "interjection_check_probability": group_interjection_check_probability(
                    primary.rate_ewma
                ),
                "activity_released_through_message_id": (
                    int(state["activity_released_through_message_id"])
                    if state and state["activity_released_through_message_id"] is not None
                    else None
                ),
            }
        return GroupFlowDiagnostic(
            window=primary,
            next_window=collecting if pipeline is not None else None,
            algorithm=algorithm,
        )

    async def _window(self, row: sqlite3.Row) -> GroupFlowWindow:
        members = await self.db.fetch_all(
            """SELECT message_id FROM group_flow_window_members
            WHERE window_id = ? ORDER BY ordinal""",
            (row["window_id"],),
        )
        return GroupFlowWindow(
            window_id=str(row["window_id"]),
            profile_id=str(row["profile_id"]),
            instance_id=str(row["instance_id"]),
            status=GroupFlowStatus(str(row["status"])),
            message_ids=tuple(int(item["message_id"]) for item in members),
            first_message_id=int(row["first_message_id"]),
            last_message_id=int(row["last_message_id"]),
            message_count=int(row["message_count"]),
            rate_ewma=float(row["rate_ewma"]),
            repeat_ratio=float(row["repeat_ratio"]),
            judge_threshold=int(row["judge_threshold"]),
            judge_through_message_id=self._optional_int(row["judge_through_message_id"]),
            frozen_through_message_id=self._optional_int(row["frozen_through_message_id"]),
            next_judge_at=_parse(row["next_judge_at"]),
            quiet_due_at=_parse(row["quiet_due_at"]),
            dynamic_due_at=_parse(row["dynamic_due_at"]),
            direct_due_at=_parse(row["direct_due_at"]),
            direct_address=bool(row["direct_address"]),
            judge_result=str(row["judge_result"]),
            judge_error_code=str(row["judge_error_code"]),
            lease_owner=str(row["lease_owner"]) if row["lease_owner"] is not None else None,
            lease_token=int(row["lease_token"]),
            lease_until=_parse(row["lease_until"]),
            main_core_task_ref=(
                str(row["main_core_task_ref"]) if row["main_core_task_ref"] is not None else None
            ),
            first_attempt_started_at=_parse(row["first_attempt_started_at"]),
            error_code=str(row["error_code"]),
            version=int(row["version"]),
            created_at=_parse(row["created_at"]),
            updated_at=_parse(row["updated_at"]),
            resolved_at=_parse(row["resolved_at"]),
            resolution_outcome=str(row["resolution_outcome"]),
        )

    @classmethod
    def _source_message(cls, row: sqlite3.Row) -> GroupFlowSourceMessage:
        cluster_keys = cls._string_tuple(row["media_cluster_keys_json"])
        components = cls._components(row["components_json"])
        return GroupFlowSourceMessage(
            message_id=int(row["message_id"]),
            sender_id=str(row["sender_id"]),
            sender_name=str(row["sender_name"]),
            plain_text=with_inbound_reply_projection(str(row["plain_text"]), components),
            media_kinds=cls._media_types(components),
            media_cluster_keys=cluster_keys,
            occurred_at=_parse(row["occurred_at"]),
            role=str(row["role"]),
            direction=str(row["direction"]),
            delivery_status=str(row["delivery_status"]),
        )

    @staticmethod
    def _components(value: str) -> list[dict[str, object]]:
        try:
            components = json.loads(value or "[]")
        except (TypeError, ValueError):
            return []
        if not isinstance(components, list):
            return []
        return [item for item in components if isinstance(item, dict)]

    @staticmethod
    def _media_types(components: list[dict[str, object]]) -> tuple[str, ...]:
        return tuple(
            str(item.get("type") or "").strip().upper()
            for item in components
            if str(item.get("type") or "").strip().lower()
            in {"image", "record", "audio", "voice", "file", "video"}
        )

    @staticmethod
    def _string_tuple(value: str) -> tuple[str, ...]:
        try:
            parsed = json.loads(value or "[]")
        except (TypeError, ValueError):
            return ()
        return tuple(str(item) for item in parsed) if isinstance(parsed, list) else ()

    @staticmethod
    def _optional_int(value: object) -> int | None:
        return int(value) if value is not None else None


__all__ = ["SqliteGroupFlowRepository"]
