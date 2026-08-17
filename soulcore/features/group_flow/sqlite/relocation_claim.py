from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

from ....contracts.group_flow import GroupReplyRelocationCheck, GroupRunFence
from ....storage.sqlite.codec import _dt


def _expire_claims(conn: sqlite3.Connection, now_text: str) -> None:
    conn.execute(
        """UPDATE group_reply_relocation_states SET check_owner = NULL,
        check_until = NULL, check_through_message_id = NULL, check_final = 0,
        updated_at = ? WHERE check_owner IS NOT NULL AND check_until <= ?""",
        (now_text, now_text),
    )


def _ensure_relocation_states(conn: sqlite3.Connection, now_text: str) -> None:
    conn.execute(
        """INSERT INTO group_reply_relocation_states(
            window_id, profile_id, instance_id, created_at, updated_at
        )
        SELECT protected.window_id, protected.profile_id, protected.instance_id, ?, ?
        FROM group_flow_windows protected
        WHERE protected.status IN ('RUNNING','WAITING_FIRST_ATTEMPT')
        ON CONFLICT(window_id) DO NOTHING""",
        (now_text, now_text),
    )


def _claimable_relocation_rows(
    conn: sqlite3.Connection, now_text: str, limit: int
) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """SELECT protected.window_id, delta.last_message_id,
            CASE WHEN state.candidate_recheck_at IS NULL THEN 0 ELSE 1 END AS final
            FROM group_flow_windows protected
            JOIN group_reply_relocation_states state
              ON state.window_id = protected.window_id
            JOIN group_flow_windows delta
              ON delta.profile_id = protected.profile_id
             AND delta.instance_id = protected.instance_id
             AND delta.status = 'COLLECTING'
            JOIN role_profiles profile ON profile.profile_id = protected.profile_id
            WHERE protected.status IN ('RUNNING','WAITING_FIRST_ATTEMPT')
              AND protected.first_attempt_started_at IS NULL
              AND protected.main_core_task_ref IS NOT NULL
              AND state.relocation_count = 0 AND state.check_owner IS NULL
              AND profile.enabled = 1
              AND NOT EXISTS (SELECT 1 FROM instance_chat_policies chat_policy
                WHERE chat_policy.profile_id = protected.profile_id
                  AND chat_policy.instance_id = protected.instance_id
                  AND chat_policy.soulcore_enabled = 0)
              AND (
                (state.candidate_recheck_at IS NOT NULL
                 AND state.candidate_recheck_at <= ?)
                OR
                (state.candidate_recheck_at IS NULL
                 AND delta.last_message_id > COALESCE(state.last_checked_message_id, 0)
                 AND (
                   (delta.next_judge_at IS NOT NULL
                    AND delta.next_judge_at <= ?
                    AND delta.message_count >= delta.judge_threshold)
                   OR (delta.quiet_due_at IS NOT NULL AND delta.quiet_due_at <= ?)
                   OR (delta.direct_due_at IS NOT NULL AND delta.direct_due_at <= ?)
                 ))
              )
            ORDER BY COALESCE(
                state.candidate_recheck_at,
                delta.direct_due_at,
                delta.next_judge_at,
                delta.quiet_due_at
            ), protected.window_id
            LIMIT ?""",
            (now_text, now_text, now_text, now_text, max(0, int(limit))),
        )
    )


def _claim_relocation_rows(
    conn: sqlite3.Connection, rows: list[sqlite3.Row], owner: str, lease_until: str, now_text: str
) -> list[tuple[str, int]]:
    claimed: list[tuple[str, int]] = []
    for row in rows:
        window_id = str(row["window_id"])
        changed = conn.execute(
            """UPDATE group_reply_relocation_states SET check_owner = ?,
            check_until = ?, check_token = check_token + 1,
            check_through_message_id = ?, check_final = ?, updated_at = ?
            WHERE window_id = ? AND check_owner IS NULL AND relocation_count = 0""",
            (
                owner,
                lease_until,
                int(row["last_message_id"]),
                int(row["final"]),
                now_text,
                window_id,
            ),
        ).rowcount
        if changed:
            token = conn.execute(
                "SELECT check_token FROM group_reply_relocation_states WHERE window_id = ?",
                (window_id,),
            ).fetchone()
            claimed.append((window_id, int(token["check_token"])))
    return claimed


def _claim_relocation_checks(
    conn: sqlite3.Connection, owner: str, now_text: str, lease_until: str, limit: int
) -> list[tuple[str, int]]:
    _expire_claims(conn, now_text)
    _ensure_relocation_states(conn, now_text)
    rows = _claimable_relocation_rows(conn, now_text, limit)
    return _claim_relocation_rows(conn, rows, owner, lease_until, now_text)


class GroupReplyRelocationClaimSql:
    async def claim_reply_relocation_checks(
        self,
        *,
        now: datetime,
        worker_id: str,
        limit: int,
        lease_seconds: int,
    ) -> tuple[GroupReplyRelocationCheck, ...]:
        owner = str(worker_id).strip()
        if not owner:
            raise ValueError("group reply relocation worker_id cannot be empty")
        now_text = _dt(now)
        lease_until = _dt(now + timedelta(seconds=max(1, int(lease_seconds))))
        assert now_text is not None and lease_until is not None
        identities = await self.uow.run(
            lambda conn: _claim_relocation_checks(conn, owner, now_text, lease_until, limit)
        )
        checks = [
            check
            for window_id, token in identities
            if (check := await self._load_reply_relocation_check(window_id, token)) is not None
        ]
        return tuple(checks)

    async def _load_reply_relocation_check(
        self, window_id: str, check_token: int
    ) -> GroupReplyRelocationCheck | None:
        row = await self.db.fetch_one(
            """SELECT protected.*, state.check_token, state.check_through_message_id,
            state.check_final, delta.window_id AS delta_window_id
            FROM group_flow_windows protected
            JOIN group_reply_relocation_states state ON state.window_id = protected.window_id
            JOIN group_flow_windows delta
              ON delta.profile_id = protected.profile_id
             AND delta.instance_id = protected.instance_id
             AND delta.status = 'COLLECTING'
            WHERE protected.window_id = ? AND state.check_token = ?
              AND state.check_owner IS NOT NULL""",
            (window_id, int(check_token)),
        )
        if row is None or row["frozen_through_message_id"] is None:
            return None
        return GroupReplyRelocationCheck(
            profile_id=str(row["profile_id"]),
            instance_id=str(row["instance_id"]),
            fence=GroupRunFence(
                window_id=str(row["window_id"]),
                frozen_through_message_id=int(row["frozen_through_message_id"]),
                lease_token=int(row["lease_token"]),
                version=int(row["version"]),
                main_core_task_ref=str(row["main_core_task_ref"]),
            ),
            delta_window_id=str(row["delta_window_id"]),
            delta_through_message_id=int(row["check_through_message_id"]),
            check_token=int(row["check_token"]),
            final_recheck=bool(row["check_final"]),
            pending_first_text=await self._pending_first_text(
                str(row["profile_id"]), str(row["instance_id"]), str(row["window_id"])
            ),
        )

    async def _pending_first_text(self, profile_id: str, instance_id: str, window_id: str) -> str:
        rows = await self.db.fetch_all(
            """SELECT payload_json FROM instance_outbox
            WHERE profile_id = ? AND instance_id = ? AND status = 'PENDING'
              AND json_extract(payload_json, '$.group_window_id') = ?
            ORDER BY COALESCE(expression_step_ordinal, expression_ordinal), outbox_id""",
            (profile_id, instance_id, window_id),
        )
        for row in rows:
            try:
                payload = json.loads(str(row["payload_json"] or "{}"))
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict) and (text := str(payload.get("content") or "").strip()):
                return text
        return ""


__all__ = ["GroupReplyRelocationClaimSql"]
