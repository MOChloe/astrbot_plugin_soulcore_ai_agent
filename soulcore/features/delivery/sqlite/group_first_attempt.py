"""Narrow SQLite transactions for the first group-window delivery boundary."""

from __future__ import annotations

import sqlite3
from typing import Any

from ....contracts.models import OutboxStatus
from .support import _dump, _load

GROUP_BURST_AUTO_QUOTE_THRESHOLD = 6
_ATTEMPTED_OUTBOX_STATUSES = (
    OutboxStatus.PLATFORM_ACCEPTED_UNCONFIRMED.value,
    OutboxStatus.PARTIALLY_ATTEMPTED.value,
    OutboxStatus.UNKNOWN_AFTER_CRASH.value,
)


class BeginGroupAwareDispatchPermit:
    def __init__(
        self,
        permit_id: int,
        *,
        now: str,
        expires_at: str,
        profile_id: str,
        instance_id: str,
        group_window_id: str,
        outbox_id: int | None = None,
        auto_quote_threshold: int = GROUP_BURST_AUTO_QUOTE_THRESHOLD,
    ) -> None:
        self.permit_id = int(permit_id)
        self.now = now
        self.expires_at = expires_at
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.group_window_id = group_window_id
        self.outbox_id = int(outbox_id) if outbox_id is not None else None
        self.auto_quote_threshold = max(1, int(auto_quote_threshold))

    def __call__(self, conn: sqlite3.Connection) -> dict[str, Any]:
        if self.group_window_id and self._group_fence_needs_release(conn) is None:
            return {"started": False}
        cursor = conn.execute(
            """UPDATE platform_send_permits SET status = 'DISPATCHING',
            dispatched_at = ?, expires_at = ?, updated_at = ? WHERE permit_id = ?
              AND (? IS NULL OR (
                profile_id = ? AND instance_id = ?
                AND origin_kind = 'EXPRESSION_ITEM' AND origin_id = ?
              ))
              AND status = 'RESERVED' AND lease_until > ?""",
            (
                self.now,
                self.expires_at,
                self.now,
                self.permit_id,
                self.outbox_id,
                self.profile_id,
                self.instance_id,
                f"expression-outbox:{self.outbox_id}",
                self.now,
            ),
        )
        if cursor.rowcount != 1:
            return {"started": False}
        return {"started": True}

    def _load_group_outbox(self, conn: sqlite3.Connection) -> sqlite3.Row:
        row = conn.execute(
            """SELECT * FROM instance_outbox WHERE profile_id = ? AND instance_id = ?
            AND outbox_id = ? AND status = 'SENDING'
            AND expression_batch_id IS NOT NULL
            AND json_extract(payload_json, '$.group_window_id') = ?""",
            (
                self.profile_id,
                self.instance_id,
                self.outbox_id,
                self.group_window_id,
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("group expression outbox is not ready for platform dispatch")
        return row

    def _persist_burst_anchor(
        self,
        conn: sqlite3.Connection,
        outbox: sqlite3.Row,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        if _valid_reply_target(payload.get("reply_target")):
            return payload, 0
        previous = self._previous_attempted_outbox(conn, outbox)
        baseline = (
            int(previous["context_message_id"])
            if previous is not None
            else self._frozen_message_id(conn)
        )
        inbound_count = self._received_inbound_count(conn, baseline)
        if inbound_count < self.auto_quote_threshold:
            return payload, inbound_count
        target = self._previous_outbound_target(conn, previous) if previous is not None else None
        if target is None:
            target = self._frozen_source_target(conn, route_umo=str(outbox["route_umo"]))
        if target is None:
            return payload, inbound_count
        updated = dict(payload)
        updated["reply_target"] = target
        changed = conn.execute(
            """UPDATE instance_outbox SET payload_json = ?, updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND outbox_id = ?
              AND status = 'SENDING' AND payload_json = ?""",
            (
                _dump(updated),
                self.now,
                self.profile_id,
                self.instance_id,
                self.outbox_id,
                outbox["payload_json"],
            ),
        ).rowcount
        if changed != 1:
            raise RuntimeError("group burst reply anchor persistence lost")
        return updated, inbound_count

    def _previous_attempted_outbox(
        self,
        conn: sqlite3.Connection,
        outbox: sqlite3.Row,
    ) -> sqlite3.Row | None:
        placeholders = ",".join("?" for _ in _ATTEMPTED_OUTBOX_STATUSES)
        return conn.execute(
            f"""SELECT * FROM instance_outbox WHERE profile_id = ? AND instance_id = ?
            AND expression_batch_id = ? AND expression_ordinal < ?
            AND context_message_id IS NOT NULL AND status IN ({placeholders})
            ORDER BY expression_ordinal DESC LIMIT 1""",
            (
                self.profile_id,
                self.instance_id,
                outbox["expression_batch_id"],
                int(outbox["expression_ordinal"]),
                *_ATTEMPTED_OUTBOX_STATUSES,
            ),
        ).fetchone()

    def _frozen_message_id(self, conn: sqlite3.Connection) -> int:
        row = conn.execute(
            """SELECT frozen_through_message_id FROM group_flow_windows
            WHERE window_id = ? AND profile_id = ? AND instance_id = ?""",
            (self.group_window_id, self.profile_id, self.instance_id),
        ).fetchone()
        if row is None or row["frozen_through_message_id"] is None:
            raise RuntimeError("group window has no frozen inbound boundary")
        return int(row["frozen_through_message_id"])

    def _received_inbound_count(self, conn: sqlite3.Connection, baseline: int) -> int:
        row = conn.execute(
            """SELECT COUNT(*) AS total FROM instance_messages
            WHERE profile_id = ? AND instance_id = ? AND message_id > ?
              AND direction = 'INBOUND' AND role = 'user'
              AND delivery_status = 'RECEIVED'""",
            (self.profile_id, self.instance_id, int(baseline)),
        ).fetchone()
        return int(row["total"] if row is not None else 0)

    def _previous_outbound_target(
        self,
        conn: sqlite3.Connection,
        previous: sqlite3.Row,
    ) -> dict[str, Any] | None:
        return _target_for_ledger_message(
            conn,
            self.profile_id,
            self.instance_id,
            int(previous["context_message_id"]),
            route_umo=str(previous["route_umo"]),
            direction="OUTBOUND",
        )

    def _frozen_source_target(
        self,
        conn: sqlite3.Connection,
        *,
        route_umo: str,
    ) -> dict[str, Any] | None:
        row = conn.execute(
            """SELECT member.message_id FROM group_flow_window_members member
            JOIN group_flow_windows window ON window.window_id = member.window_id
            JOIN instance_messages message
              ON message.profile_id = member.profile_id
             AND message.instance_id = member.instance_id
             AND message.message_id = member.message_id
            WHERE member.window_id = ? AND member.profile_id = ? AND member.instance_id = ?
              AND member.message_id <= window.frozen_through_message_id
              AND message.direction = 'INBOUND' AND message.role = 'user'
              AND message.delivery_status = 'RECEIVED'
              AND EXISTS (
                SELECT 1 FROM instance_message_fragments fragment
                WHERE fragment.profile_id = member.profile_id
                  AND fragment.instance_id = member.instance_id
                  AND fragment.ledger_message_id = member.message_id
                  AND fragment.direction = 'INBOUND'
                  AND fragment.route_umo = ?
                  AND COALESCE(fragment.retraction_status, '') != 'RETRACTED'
              )
            ORDER BY member.ordinal DESC LIMIT 1""",
            (self.group_window_id, self.profile_id, self.instance_id, route_umo),
        ).fetchone()
        if row is None:
            return None
        return _target_for_ledger_message(
            conn,
            self.profile_id,
            self.instance_id,
            int(row["message_id"]),
            direction="INBOUND",
            route_umo=route_umo,
        )

    def _group_fence_needs_release(self, conn: sqlite3.Connection) -> bool | None:
        row = conn.execute(
            """SELECT status, resolution_outcome FROM group_flow_windows
            WHERE window_id = ? AND profile_id = ? AND instance_id = ?""",
            (self.group_window_id, self.profile_id, self.instance_id),
        ).fetchone()
        if row is None:
            return None
        if row["status"] == "WAITING_FIRST_ATTEMPT":
            return True
        if row["status"] == "RESOLVED" and row["resolution_outcome"] == "ADAPTER_CALL_STARTED":
            # One Main Core expression can contain several physical chat
            # bubbles.  Only the first platform call releases the group
            # window; later bubbles from that same committed expression must
            # still be allowed to start their own send permits.
            return False
        return None

    def _release_group_fence(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute(
            """UPDATE group_flow_windows SET status = 'RESOLVED',
            first_attempt_started_at = ?, resolution_outcome = 'ADAPTER_CALL_STARTED',
            resolved_at = ?, updated_at = ?, version = version + 1
            WHERE window_id = ? AND profile_id = ? AND instance_id = ?
              AND status = 'WAITING_FIRST_ATTEMPT'""",
            (
                self.now,
                self.now,
                self.now,
                self.group_window_id,
                self.profile_id,
                self.instance_id,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("group first-attempt fence changed during permit transition")
        _release_window_messages(
            conn,
            self.profile_id,
            self.instance_id,
            self.group_window_id,
            reason="group_flow_first_attempt_started",
            now=self.now,
        )
        conn.execute(
            """INSERT INTO group_flow_instance_state(
              profile_id, instance_id, last_visible_assistant_at,
              last_resolved_message_id, updated_at
            ) VALUES (?, ?, ?, (
              SELECT frozen_through_message_id FROM group_flow_windows WHERE window_id = ?
            ), ?) ON CONFLICT(profile_id, instance_id) DO UPDATE SET
              last_visible_assistant_at = excluded.last_visible_assistant_at,
              last_resolved_message_id = excluded.last_resolved_message_id,
              updated_at = excluded.updated_at""",
            (
                self.profile_id,
                self.instance_id,
                self.now,
                self.group_window_id,
                self.now,
            ),
        )


def mark_group_first_attempt_started(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    group_window_id: str,
    now: str,
) -> bool:
    """Release one group window only after its platform sender was entered."""

    operation = BeginGroupAwareDispatchPermit(
        0,
        now=now,
        expires_at=now,
        profile_id=str(profile_id),
        instance_id=str(instance_id),
        group_window_id=str(group_window_id),
    )
    release = operation._group_fence_needs_release(conn)
    if release is None:
        raise RuntimeError("group first-attempt fence changed before platform call settlement")
    if release:
        operation._release_group_fence(conn)
    return bool(release)


class PrepareGroupDispatchAnchor:
    """Persist burst addressing while the platform-call permit stays RESERVED."""

    def __init__(
        self,
        permit_id: int,
        *,
        now: str,
        profile_id: str,
        instance_id: str,
        group_window_id: str,
        outbox_id: int,
        auto_quote_threshold: int = GROUP_BURST_AUTO_QUOTE_THRESHOLD,
    ) -> None:
        self.operation = BeginGroupAwareDispatchPermit(
            permit_id,
            now=now,
            expires_at=now,
            profile_id=profile_id,
            instance_id=instance_id,
            group_window_id=group_window_id,
            outbox_id=outbox_id,
            auto_quote_threshold=auto_quote_threshold,
        )

    def __call__(self, conn: sqlite3.Connection) -> dict[str, Any]:
        permit = conn.execute(
            """SELECT status, lease_until FROM platform_send_permits
            WHERE permit_id = ? AND profile_id = ? AND instance_id = ?
              AND origin_kind = 'EXPRESSION_ITEM' AND origin_id = ?""",
            (
                self.operation.permit_id,
                self.operation.profile_id,
                self.operation.instance_id,
                f"expression-outbox:{self.operation.outbox_id}",
            ),
        ).fetchone()
        if permit is None:
            return {"prepared": False, "reason": "INVALID_IDENTITY"}
        status = str(permit["status"])
        if status in {"DISPATCHING", "ATTEMPTED_UNKNOWN"}:
            return {"prepared": False, "reason": "ALREADY_STARTED"}
        if status != "RESERVED":
            return {"prepared": False, "reason": "NOT_RESERVED"}
        if str(permit["lease_until"]) <= self.operation.now:
            return {"prepared": False, "reason": "EXPIRED"}
        outbox = self.operation._load_group_outbox(conn)
        payload = dict(_load(outbox["payload_json"]) or {})
        payload, inbound_count = self.operation._persist_burst_anchor(conn, outbox, payload)
        return {
            "prepared": True,
            "reason": "PREPARED",
            "payload": payload,
            "inbound_count": inbound_count,
        }


class ResolveUndeliverableGroupWindow:
    def __init__(self, profile_id: str, instance_id: str, group_window_id: str, now: str) -> None:
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.group_window_id = group_window_id
        self.now = now

    def __call__(self, conn: sqlite3.Connection) -> bool:
        remaining = conn.execute(
            """SELECT 1 FROM instance_outbox WHERE profile_id = ? AND instance_id = ?
            AND json_extract(payload_json, '$.group_window_id') = ?
            AND status IN ('PENDING', 'SENDING') LIMIT 1""",
            (self.profile_id, self.instance_id, self.group_window_id),
        ).fetchone()
        if remaining is not None:
            return False
        cursor = conn.execute(
            """UPDATE group_flow_windows SET status = 'RESOLVED',
            resolution_outcome = 'NO_DELIVERABLE_OUTPUT', resolved_at = ?,
            updated_at = ?, version = version + 1 WHERE window_id = ?
            AND profile_id = ? AND instance_id = ?
            AND status = 'WAITING_FIRST_ATTEMPT'""",
            (
                self.now,
                self.now,
                self.group_window_id,
                self.profile_id,
                self.instance_id,
            ),
        )
        if cursor.rowcount != 1:
            return False
        _release_window_messages(
            conn,
            self.profile_id,
            self.instance_id,
            self.group_window_id,
            reason="group_flow_no_deliverable_output",
            now=self.now,
        )
        conn.execute(
            """INSERT INTO group_flow_instance_state(
              profile_id, instance_id, last_resolved_message_id, updated_at
            ) VALUES (?, ?, (
              SELECT frozen_through_message_id FROM group_flow_windows WHERE window_id = ?
            ), ?) ON CONFLICT(profile_id, instance_id) DO UPDATE SET
              last_resolved_message_id = excluded.last_resolved_message_id,
              updated_at = excluded.updated_at""",
            (
                self.profile_id,
                self.instance_id,
                self.group_window_id,
                self.now,
            ),
        )
        return True


def resolve_retract_only_group_window(
    conn: sqlite3.Connection,
    batch_id: str,
    now: str,
) -> bool:
    batch = conn.execute(
        """SELECT batch.profile_id, batch.instance_id,
          json_extract(run.request_json, '$.metadata.group_run_fence.window_id')
            AS group_window_id,
          json_extract(run.request_json, '$.metadata.group_run_fence.frozen_through_message_id')
            AS frozen_through_message_id,
          json_extract(run.request_json, '$.metadata.group_run_fence.lease_token')
            AS lease_token,
          json_extract(run.request_json, '$.metadata.group_run_fence.version')
            AS fence_version,
          json_extract(run.request_json, '$.metadata.group_run_fence.main_core_task_ref')
            AS main_core_task_ref
        FROM instance_expression_batches batch
        JOIN instance_core_runs run
          ON run.profile_id = batch.profile_id
         AND run.instance_id = batch.instance_id
         AND run.run_id = batch.source_run_id
        WHERE batch.batch_id = ?""",
        (str(batch_id),),
    ).fetchone()
    if batch is None or not str(batch["group_window_id"] or "").strip():
        return False
    visible = conn.execute(
        "SELECT 1 FROM instance_outbox WHERE expression_batch_id = ? LIMIT 1",
        (str(batch_id),),
    ).fetchone()
    if visible is not None:
        return False
    unfinished = conn.execute(
        """SELECT 1 FROM message_retraction_actions
        WHERE expression_batch_id = ? AND status IN ('PENDING', 'SENDING') LIMIT 1""",
        (str(batch_id),),
    ).fetchone()
    if unfinished is not None:
        return False
    window = conn.execute(
        """SELECT window_id FROM group_flow_windows
        WHERE profile_id = ? AND instance_id = ?
          AND window_id = ?
          AND status = 'WAITING_FIRST_ATTEMPT'
          AND frozen_through_message_id = ?
          AND lease_token = ?
          AND version = ? + 1
          AND main_core_task_ref = ?""",
        (
            str(batch["profile_id"]),
            str(batch["instance_id"]),
            str(batch["group_window_id"]),
            int(batch["frozen_through_message_id"]),
            int(batch["lease_token"]),
            int(batch["fence_version"]),
            str(batch["main_core_task_ref"]),
        ),
    ).fetchone()
    if window is None:
        return False
    return ResolveUndeliverableGroupWindow(
        str(batch["profile_id"]),
        str(batch["instance_id"]),
        str(window["window_id"]),
        now,
    )(conn)


def _release_window_messages(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    window_id: str,
    *,
    reason: str,
    now: str,
) -> None:
    changed = conn.execute(
        """UPDATE instance_messages SET knowledge_eligibility = 'ELIGIBLE',
        knowledge_eligibility_reason = ? WHERE profile_id = ? AND instance_id = ?
        AND knowledge_eligibility = 'HELD'
        AND knowledge_eligibility_reason = 'group_flow_pending'
        AND message_id IN (
          SELECT message_id FROM group_flow_window_members WHERE window_id = ?
        )""",
        (reason, profile_id, instance_id, window_id),
    ).rowcount
    if changed:
        conn.execute(
            """UPDATE knowledge_processing_state SET
            processing_version = processing_version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?""",
            (now, profile_id, instance_id),
        )


def _valid_reply_target(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return bool(
        str(value.get("message_ref") or "").strip()
        and str(value.get("route_umo") or "").strip()
        and (
            str(value.get("platform_message_id") or "").strip()
            or str(value.get("platform_reference_id") or "").strip()
        )
    )


def _target_for_ledger_message(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    ledger_message_id: int,
    *,
    direction: str,
    route_umo: str = "",
) -> dict[str, Any] | None:
    params: list[Any] = [profile_id, instance_id, int(ledger_message_id), direction]
    route_clause = ""
    if route_umo:
        route_clause = " AND fragment.route_umo = ?"
        params.append(route_umo)
    row = conn.execute(
        f"""SELECT fragment.*, message.sender_name, message.plain_text
        FROM instance_message_fragments fragment
        JOIN instance_messages message
          ON message.profile_id = fragment.profile_id
         AND message.instance_id = fragment.instance_id
         AND message.message_id = fragment.ledger_message_id
        WHERE fragment.profile_id = ? AND fragment.instance_id = ?
          AND fragment.ledger_message_id = ? AND fragment.direction = ?
          AND COALESCE(fragment.retraction_status, '') != 'RETRACTED'
          {route_clause}
        ORDER BY fragment.fragment_ordinal LIMIT 1""",
        params,
    ).fetchone()
    if row is None:
        return None
    platform_message_id = str(row["platform_message_id"] or "").strip()
    platform_reference_id = str(row["platform_reference_id"] or "").strip()
    message_ref = str(row["message_ref"] or "").strip()
    fragment_route = str(row["route_umo"] or "").strip()
    if not all((message_ref, fragment_route, platform_message_id)):
        return None
    native_target = ""
    if bool(row["native_reply_supported"]):
        native_target = platform_reference_id or platform_message_id
    projection = str(row["content_projection"] or row["plain_text"] or "").strip()[:120]
    return {
        "message_ref": message_ref,
        "platform_message_id": platform_message_id,
        "platform_reference_id": platform_reference_id,
        "native_reply_target_id": native_target,
        "sender_display_name": str(row["sender_name"] or "")[:80],
        "content_kind": str(row["content_kind"] or "OTHER"),
        "content_projection": projection,
        "platform_instance_id": str(row["platform_instance_id"] or ""),
        "route_umo": fragment_route,
    }


__all__ = [
    "BeginGroupAwareDispatchPermit",
    "GROUP_BURST_AUTO_QUOTE_THRESHOLD",
    "ResolveUndeliverableGroupWindow",
    "mark_group_first_attempt_started",
    "resolve_retract_only_group_window",
]
