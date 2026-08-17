from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime

from ....contracts.group_flow import (
    GroupFlowInboundMessage,
    GroupFlowPolicy,
)
from ....storage.sqlite.codec import _parse
from ....storage.sqlite.inbound_admission_transactions import InboundAdmissionTransactions
from ....storage.sqlite.repository_lifecycle import KnowledgeTaskSql


class GroupFlowOrphanRecoverySql(KnowledgeTaskSql):
    """Recover durable inbound admissions left between ledger and group-flow handoff."""

    _inbound_admission: InboundAdmissionTransactions

    def _recover_orphaned_messages(
        self,
        conn: sqlite3.Connection,
        *,
        now: datetime,
        now_text: str,
    ) -> int:
        rows = self._orphaned_message_rows(conn, now_text=now_text)
        policies: dict[str, GroupFlowPolicy] = {}
        recovery_owner = f"group-flow-recovery:{uuid.uuid4().hex}"
        recovered = 0
        for row in rows:
            recovered += int(
                self._recover_orphaned_message(
                    conn,
                    row,
                    policies=policies,
                    recovery_owner=recovery_owner,
                    now=now,
                    now_text=now_text,
                )
            )
        return recovered

    def _recover_orphaned_message(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        policies: dict[str, GroupFlowPolicy],
        recovery_owner: str,
        now: datetime,
        now_text: str,
    ) -> bool:
        profile_id = str(row["profile_id"])
        instance_id = str(row["instance_id"])
        message_id = int(row["message_id"])
        token = self._claim_orphaned_admission(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            message_id=message_id,
            recovery_owner=recovery_owner,
            now=now,
        )
        if token is None:
            return False
        if not bool(row["has_window_member"]):
            policy = self._recovery_policy(conn, policies, profile_id, now_text)
            if policy is None:
                return False
            occurred_at = _parse(row["occurred_at"]) or now
            self._append_message_sql(
                conn,
                profile_id,
                instance_id,
                _inbound_message(row, occurred_at),
                policy=policy,
                now=occurred_at,
            )
        # The durable window must exist before group admission is applied.
        # Otherwise ledger-only crash recovery would advance activity as if
        # this were a private message and bypass the interjection judge.
        admission = self._inbound_admission.apply(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            message_id=message_id,
            now=now,
            group_scope=True,
            refresh_knowledge_task=self._refresh_knowledge_task_sql,
            lease_owner=recovery_owner,
            lease_token=token,
        )
        if not admission.ownership_valid:
            raise RuntimeError("group-flow orphan admission ownership changed")
        self._complete_orphaned_admission(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            message_id=message_id,
            recovery_owner=recovery_owner,
            token=token,
            now=now,
        )
        return True

    @staticmethod
    def _orphaned_message_rows(
        conn: sqlite3.Connection,
        *,
        now_text: str,
    ) -> list[sqlite3.Row]:
        return list(
            conn.execute(
                """SELECT message.*,
                  EXISTS (
                    SELECT 1 FROM group_flow_window_members member
                    WHERE member.profile_id = message.profile_id
                      AND member.instance_id = message.instance_id
                      AND member.message_id = message.message_id
                  ) AS has_window_member
                FROM instance_messages message
                JOIN character_instances instance
                  ON instance.profile_id = message.profile_id
                 AND instance.instance_id = message.instance_id
                WHERE instance.scope = 'group'
                  AND message.direction = 'INBOUND'
                  AND message.delivery_status = 'RECEIVED'
                  AND message.knowledge_eligibility = 'HELD'
                  AND message.knowledge_eligibility_reason = 'group_flow_pending'
                  AND json_extract(message.metadata_json,
                    '$.inbound_admission.status') = 'ADMITTING'
                  AND json_extract(message.metadata_json,
                    '$.inbound_admission.lease_until') <= ?
                ORDER BY message.message_id LIMIT 100""",
                (now_text,),
            )
        )

    def _claim_orphaned_admission(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        instance_id: str,
        message_id: int,
        recovery_owner: str,
        now: datetime,
    ) -> int | None:
        return self._inbound_admission.claim_expired(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            message_id=message_id,
            lease_owner=recovery_owner,
            now=now,
            lease_seconds=30,
        )

    def _recovery_policy(
        self,
        conn: sqlite3.Connection,
        policies: dict[str, GroupFlowPolicy],
        profile_id: str,
        now_text: str,
    ) -> GroupFlowPolicy | None:
        policy = policies.get(profile_id)
        if policy is not None:
            return policy
        conn.execute(
            """INSERT OR IGNORE INTO group_flow_policies(
            profile_id, scope, created_at, updated_at
            ) VALUES (?, 'group', ?, ?)""",
            (profile_id, now_text, now_text),
        )
        row = conn.execute(
            """SELECT * FROM group_flow_policies
            WHERE profile_id = ? AND scope = 'group'""",
            (profile_id,),
        ).fetchone()
        if row is None:
            return None
        policy = self._policy(row)
        policies[profile_id] = policy
        return policy

    def _complete_orphaned_admission(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        instance_id: str,
        message_id: int,
        recovery_owner: str,
        token: int,
        now: datetime,
    ) -> None:
        if self._inbound_admission.complete(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            message_id=message_id,
            lease_owner=recovery_owner,
            lease_token=token,
            now=now,
        ):
            return
        raise RuntimeError("group-flow orphan handoff ownership changed")


def _inbound_message(row: sqlite3.Row, occurred_at: datetime) -> GroupFlowInboundMessage:
    components = _json_list(row["components_json"])
    metadata = _json_object(row["metadata_json"])
    return GroupFlowInboundMessage(
        message_id=int(row["message_id"]),
        occurred_at=occurred_at,
        sender_id=str(row["sender_id"] or ""),
        sender_name=str(row["sender_name"] or ""),
        plain_text=str(row["plain_text"] or ""),
        media_kinds=_media_kinds(components),
        media_cluster_keys=(),
        direct_address=bool(metadata.get("direct_address"))
        or _resolved_assistant_reference(components),
    )


def _json_object(value: object) -> dict[str, object]:
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _json_list(value: object) -> list[dict[str, object]]:
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [dict(item) for item in parsed if isinstance(item, dict)]


def _media_kinds(components: list[dict[str, object]]) -> tuple[str, ...]:
    return tuple(
        str(item.get("type") or "").strip().upper()
        for item in components
        if str(item.get("type") or "").strip().lower()
        in {"image", "record", "audio", "voice", "file", "video"}
    )


def _resolved_assistant_reference(components: list[dict[str, object]]) -> bool:
    return any(
        str(item.get("type") or "").strip().lower() == "inbound_reply_reference"
        and str(item.get("status") or "").strip().lower() == "resolved"
        and str(item.get("target_role") or "").strip().lower() == "assistant"
        for item in components
    )


__all__ = ["GroupFlowOrphanRecoverySql"]
