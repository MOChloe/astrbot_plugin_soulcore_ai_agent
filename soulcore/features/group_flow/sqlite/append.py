from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime

from ....contracts.delivery_visibility import DIALOGUE_CONTINUITY_OUTBOUND_STATUSES
from ....contracts.group_flow import (
    GroupFlowInboundMessage,
    GroupFlowPolicy,
    GroupFlowWindow,
)
from ....storage.sqlite.codec import _dt, _parse
from ..cleaning import media_cluster_keys_match, normalized_text_fingerprint
from ..policy import GroupSchedule, build_schedule, reply_gap_due_at


class GroupFlowAppendSql:
    async def append_message(
        self,
        profile_id: str,
        instance_id: str,
        message: GroupFlowInboundMessage,
        *,
        policy: GroupFlowPolicy,
        now: datetime,
    ) -> GroupFlowWindow:
        def operation(conn: sqlite3.Connection) -> str:
            return self._append_message_sql(
                conn,
                profile_id,
                instance_id,
                message,
                policy=policy,
                now=now,
            )

        window_id = await self.uow.run(operation)
        window = await self.get_window(profile_id, instance_id, window_id)
        if window is None:
            raise RuntimeError("group flow append did not create a window")
        return window

    def _append_message_sql(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        message: GroupFlowInboundMessage,
        *,
        policy: GroupFlowPolicy,
        now: datetime,
    ) -> str:
        existing = conn.execute(
            """SELECT window_id FROM group_flow_window_members
            WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
            (profile_id, instance_id, message.message_id),
        ).fetchone()
        if existing is not None:
            return str(existing["window_id"])
        ledger = self._ledger_message(conn, profile_id, instance_id, message.message_id)
        state = self._state(conn, profile_id, instance_id)
        current = conn.execute(
            """SELECT * FROM group_flow_windows WHERE profile_id = ? AND instance_id = ?
            AND status = 'COLLECTING'""",
            (profile_id, instance_id),
        ).fetchone()
        now_text = _dt(now)
        assert now_text is not None
        window_id, ordinal, first_at = self._collection_target(
            conn, profile_id, instance_id, message, ledger, current, now_text
        )
        occurred_at = _parse(ledger["occurred_at"]) or message.occurred_at
        self._insert_member(
            conn, window_id, profile_id, instance_id, message, ledger, ordinal, now_text
        )
        count, repeat_ratio = self._repeat_stats(conn, window_id)
        last_visible = self._last_visible(conn, profile_id, instance_id, state)
        direct = bool(message.direct_address) or bool(current and current["direct_address"])
        schedule = build_schedule(
            policy,
            first_at=first_at,
            last_at=occurred_at,
            previous_rate=float(state["rate_ewma"]) if state is not None else 0.0,
            previous_at=_parse(state["last_inbound_at"]) if state is not None else None,
            repeat_ratio=repeat_ratio,
            direct_address=direct,
            last_visible_at=last_visible,
            now=now,
        )
        self._persist_append_schedule(
            conn,
            profile_id,
            instance_id,
            window_id,
            message.message_id,
            ledger,
            schedule,
            count=count,
            repeat_ratio=repeat_ratio,
            direct=direct,
            last_visible=last_visible,
            gap_due=reply_gap_due_at(policy, last_visible_at=last_visible),
            now_text=now_text,
        )
        conn.execute(
            """UPDATE instance_messages SET knowledge_eligibility = 'HELD',
            knowledge_eligibility_reason = 'group_flow_pending'
            WHERE profile_id = ? AND instance_id = ? AND message_id = ?
              AND knowledge_eligibility != 'EXCLUDED'""",
            (profile_id, instance_id, message.message_id),
        )
        return window_id

    @staticmethod
    def _collection_target(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        message: GroupFlowInboundMessage,
        ledger: sqlite3.Row,
        current: sqlite3.Row | None,
        now_text: str,
    ) -> tuple[str, int, datetime]:
        if current is not None:
            first = conn.execute(
                """SELECT occurred_at FROM group_flow_window_members
                WHERE window_id = ? ORDER BY ordinal LIMIT 1""",
                (current["window_id"],),
            ).fetchone()
            return (
                str(current["window_id"]),
                int(current["message_count"]),
                _parse(first["occurred_at"]) or message.occurred_at,
            )
        window_id = f"group:{uuid.uuid4().hex}"
        conn.execute(
            """INSERT INTO group_flow_windows(
            window_id, profile_id, instance_id, first_message_id, last_message_id,
            message_count, direct_address, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)""",
            (
                window_id,
                profile_id,
                instance_id,
                message.message_id,
                message.message_id,
                int(message.direct_address),
                now_text,
                now_text,
            ),
        )
        return window_id, 0, _parse(ledger["occurred_at"]) or message.occurred_at

    def _insert_member(
        self,
        conn: sqlite3.Connection,
        window_id: str,
        profile_id: str,
        instance_id: str,
        message: GroupFlowInboundMessage,
        ledger: sqlite3.Row,
        ordinal: int,
        now_text: str,
    ) -> None:
        media_kinds = self._media_kinds(ledger["components_json"])
        keys = tuple(
            str(value).strip() for value in message.media_cluster_keys if str(value).strip()
        )
        cluster_keys = self._canonical_media_keys(conn, window_id, keys)
        fingerprint = normalized_text_fingerprint(
            str(ledger["plain_text"]), media_kinds, cluster_keys, message_id=message.message_id
        )
        conn.execute(
            """INSERT INTO group_flow_window_members(
            window_id, profile_id, instance_id, message_id, ordinal,
            normalized_fingerprint, media_cluster_keys_json, sender_id, occurred_at, added_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                window_id,
                profile_id,
                instance_id,
                message.message_id,
                ordinal,
                fingerprint,
                json.dumps(cluster_keys, ensure_ascii=False, separators=(",", ":")),
                str(ledger["sender_id"]),
                ledger["occurred_at"],
                now_text,
            ),
        )

    def _persist_append_schedule(
        self,
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        window_id: str,
        message_id: int,
        ledger: sqlite3.Row,
        schedule: GroupSchedule,
        *,
        count: int,
        repeat_ratio: float,
        direct: bool,
        last_visible: datetime | None,
        gap_due: datetime | None,
        now_text: str,
    ) -> None:
        conn.execute(
            """UPDATE group_flow_windows SET last_message_id = ?, message_count = ?,
            rate_ewma = ?, repeat_ratio = ?, judge_threshold = ?, next_judge_at = ?,
            quiet_due_at = ?, dynamic_due_at = ?, direct_due_at = ?, direct_address = ?,
            version = version + 1, updated_at = ? WHERE window_id = ? AND status = 'COLLECTING'""",
            (
                message_id,
                count,
                schedule.rate_ewma,
                repeat_ratio,
                schedule.judge_threshold,
                _dt(self._later(schedule.next_judge_at, gap_due)),
                _dt(self._later(schedule.quiet_due_at, gap_due)),
                _dt(self._later(schedule.dynamic_due_at, gap_due)),
                _dt(schedule.direct_due_at),
                int(direct),
                now_text,
                window_id,
            ),
        )
        conn.execute(
            """INSERT INTO group_flow_instance_state(
            profile_id, instance_id, rate_ewma, last_inbound_at,
            last_visible_assistant_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id, instance_id) DO UPDATE SET
            rate_ewma = excluded.rate_ewma, last_inbound_at = excluded.last_inbound_at,
            last_visible_assistant_at = COALESCE(
                excluded.last_visible_assistant_at,
                group_flow_instance_state.last_visible_assistant_at
            ), updated_at = excluded.updated_at""",
            (
                profile_id,
                instance_id,
                schedule.rate_ewma,
                ledger["occurred_at"],
                _dt(last_visible),
                now_text,
            ),
        )

    @staticmethod
    def _ledger_message(
        conn: sqlite3.Connection, profile_id: str, instance_id: str, message_id: int
    ) -> sqlite3.Row:
        instance = conn.execute(
            """SELECT scope FROM character_instances
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        ).fetchone()
        if instance is None or str(instance["scope"]) != "group":
            raise ValueError("group flow only accepts group instances")
        row = conn.execute(
            """SELECT message_id, sender_id, plain_text, components_json, occurred_at
            FROM instance_messages WHERE profile_id = ? AND instance_id = ?
            AND message_id = ? AND direction = 'INBOUND'
            AND delivery_status = 'RECEIVED'""",
            (profile_id, instance_id, int(message_id)),
        ).fetchone()
        if row is None:
            raise ValueError("group flow requires a received inbound ledger message")
        return row

    @staticmethod
    def _state(conn: sqlite3.Connection, profile_id: str, instance_id: str) -> sqlite3.Row | None:
        return conn.execute(
            """SELECT * FROM group_flow_instance_state
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        ).fetchone()

    @staticmethod
    def _last_visible(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        state: sqlite3.Row | None,
    ) -> datetime | None:
        if state is not None and state["last_visible_assistant_at"]:
            return _parse(state["last_visible_assistant_at"])
        placeholders = ",".join("?" for _ in DIALOGUE_CONTINUITY_OUTBOUND_STATUSES)
        row = conn.execute(
            f"""SELECT MAX(occurred_at) AS occurred_at FROM instance_messages
            WHERE profile_id = ? AND instance_id = ? AND direction = 'OUTBOUND'
            AND role = 'assistant'
            AND delivery_status IN ({placeholders})""",
            (profile_id, instance_id, *DIALOGUE_CONTINUITY_OUTBOUND_STATUSES),
        ).fetchone()
        return _parse(row["occurred_at"]) if row and row["occurred_at"] else None

    @staticmethod
    def _repeat_stats(conn: sqlite3.Connection, window_id: str) -> tuple[int, float]:
        row = conn.execute(
            """SELECT COUNT(*) AS total, COUNT(DISTINCT normalized_fingerprint) AS unique_count
            FROM group_flow_window_members WHERE window_id = ?""",
            (window_id,),
        ).fetchone()
        total = int(row["total"])
        return total, max(0.0, (total - int(row["unique_count"])) / total)

    @staticmethod
    def _canonical_media_keys(
        conn: sqlite3.Connection, window_id: str, cluster_keys: tuple[str, ...]
    ) -> tuple[str, ...]:
        if not cluster_keys:
            return ()
        rows = conn.execute(
            """SELECT media_cluster_keys_json FROM group_flow_window_members
            WHERE window_id = ? AND media_cluster_keys_json <> '[]' ORDER BY ordinal""",
            (window_id,),
        )
        for row in rows:
            try:
                parsed = json.loads(row["media_cluster_keys_json"] or "[]")
            except (TypeError, ValueError):
                continue
            existing = tuple(str(value) for value in parsed) if isinstance(parsed, list) else ()
            if media_cluster_keys_match(existing, cluster_keys):
                return existing
        return cluster_keys

    @staticmethod
    def _media_kinds(value: str) -> tuple[str, ...]:
        try:
            components = json.loads(value or "[]")
        except (TypeError, ValueError):
            return ()
        if not isinstance(components, list):
            return ()
        return tuple(
            str(item.get("type") or "").strip().upper()
            for item in components
            if isinstance(item, dict)
            and str(item.get("type") or "").strip().lower()
            in {"image", "record", "audio", "voice", "file", "video"}
        )

    @staticmethod
    def _later(value: datetime, boundary: datetime | None) -> datetime:
        return value if boundary is None or value >= boundary else boundary


__all__ = ["GroupFlowAppendSql"]
