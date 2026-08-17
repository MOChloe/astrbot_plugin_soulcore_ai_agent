from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Any

from ....storage.sqlite.codec import _parse
from ....storage.sqlite.dialogue_turns import context_eligible_sql
from ....storage.sqlite.inbound_admission_transactions import InboundAdmissionTransactions
from ...delivery.service import CapturedUMO
from ..algorithm import INBOUND_RECALL_ALGORITHM_VERSION, INBOUND_RECALL_GRACE_SECONDS
from .records import normalize_scope, required_datetime


class RecoverInboundRecallOrphans:
    def __init__(
        self,
        *,
        inbound_admission: InboundAdmissionTransactions,
        now: datetime,
        now_text: str,
    ) -> None:
        self.inbound_admission = inbound_admission
        self.now = now
        self.now_text = now_text
        self.recovery_owner = f"inbound-recall-recovery:{uuid.uuid4().hex}"

    def __call__(self, conn: sqlite3.Connection) -> int:
        recovered = 0
        for row in self._candidate_rows(conn):
            recovered += self._recover_row(conn, row)
        return recovered

    def _candidate_rows(self, conn: sqlite3.Connection) -> list[sqlite3.Row]:
        return list(
            conn.execute(
                """SELECT message.*, instance.route_umo AS instance_route_umo,
                instance.scope AS instance_scope,
                fragment.platform_instance_id AS fragment_platform_instance_id,
                fragment.route_umo AS fragment_route_umo,
                fragment.platform_message_id AS fragment_platform_message_id
                FROM instance_messages message
                JOIN character_instances instance
                  ON instance.profile_id = message.profile_id
                 AND instance.instance_id = message.instance_id
                LEFT JOIN instance_message_fragments fragment
                  ON fragment.profile_id = message.profile_id
                 AND fragment.instance_id = message.instance_id
                 AND fragment.ledger_message_id = message.message_id
                 AND fragment.direction = 'INBOUND'
                 AND fragment.fragment_ordinal = 0
                WHERE message.direction = 'INBOUND'
                  AND message.delivery_status = 'PENDING_RECALL_GRACE'
                  AND message.knowledge_eligibility = 'HELD'
                  AND message.knowledge_eligibility_reason = 'inbound_recall_grace'
                  AND json_extract(message.metadata_json,
                    '$.inbound_admission.status') = 'ADMITTING'
                  AND json_extract(message.metadata_json,
                    '$.inbound_admission.lease_until') <= ?
                ORDER BY message.message_id LIMIT 100""",
                (self.now_text,),
            )
        )

    def _recover_row(self, conn: sqlite3.Connection, row: sqlite3.Row) -> int:
        metadata = _json_object(row["metadata_json"])
        routing = self._routing_identity(row, metadata)
        if routing is None:
            return 0
        route_umo, platform_instance_id, platform_message_id = routing
        token = self._claim(conn, row)
        if token is None:
            return 0
        received_at = _parse(str(row["occurred_at"])) or self.now
        cursor = self._insert_hold(
            conn,
            row,
            metadata,
            route_umo,
            platform_instance_id,
            platform_message_id,
            received_at,
        )
        completed = self.inbound_admission.complete(
            conn,
            profile_id=str(row["profile_id"]),
            instance_id=str(row["instance_id"]),
            message_id=int(row["message_id"]),
            lease_owner=self.recovery_owner,
            lease_token=token,
            now=self.now,
            status="HANDED_OFF",
        )
        if not completed:
            raise RuntimeError("inbound-recall orphan handoff ownership changed")
        return max(1, int(cursor.rowcount))

    @staticmethod
    def _routing_identity(
        row: sqlite3.Row,
        metadata: dict[str, Any],
    ) -> tuple[str, str, str] | None:
        route_umo = str(
            metadata.get("route_umo")
            or row["fragment_route_umo"]
            or row["instance_route_umo"]
            or ""
        ).strip()
        captured = CapturedUMO.parse(route_umo)
        platform_instance_id = str(
            metadata.get("platform_instance_id")
            or row["fragment_platform_instance_id"]
            or captured.platform_id
            or ""
        ).strip()
        platform_message_id = str(
            metadata.get("platform_message_id") or row["fragment_platform_message_id"] or ""
        ).strip()
        if not route_umo or not platform_instance_id or not platform_message_id:
            return None
        return route_umo, platform_instance_id, platform_message_id

    def _claim(self, conn: sqlite3.Connection, row: sqlite3.Row) -> int | None:
        return self.inbound_admission.claim_expired(
            conn,
            profile_id=str(row["profile_id"]),
            instance_id=str(row["instance_id"]),
            message_id=int(row["message_id"]),
            lease_owner=self.recovery_owner,
            now=self.now,
            lease_seconds=30,
        )

    def _insert_hold(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        metadata: dict[str, Any],
        route_umo: str,
        platform_instance_id: str,
        platform_message_id: str,
        received_at: datetime,
    ) -> sqlite3.Cursor:
        received_text = required_datetime(received_at)
        previous = conn.execute(
            f"""SELECT occurred_at FROM instance_messages
            WHERE profile_id = ? AND instance_id = ? AND message_id <> ?
              AND occurred_at <= ? AND {context_eligible_sql()}
              AND (direction = 'OUTBOUND' OR role = 'user')
            ORDER BY occurred_at DESC, message_id DESC LIMIT 1""",
            (
                row["profile_id"],
                row["instance_id"],
                int(row["message_id"]),
                received_text,
            ),
        ).fetchone()
        return conn.execute(
            """INSERT INTO inbound_message_recall_states(
                profile_id, instance_id, ledger_message_id, platform_instance_id,
                route_umo, platform_message_id, scope, direct_address,
                received_at, grace_until, previous_activity_at, status,
                algorithm_version, original_plain_text, original_components_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'HELD', ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id, instance_id, ledger_message_id) DO NOTHING""",
            (
                row["profile_id"],
                row["instance_id"],
                int(row["message_id"]),
                platform_instance_id,
                route_umo,
                platform_message_id,
                normalize_scope(str(metadata.get("scope") or row["instance_scope"])),
                int(bool(metadata.get("direct_address"))),
                received_text,
                required_datetime(received_at + timedelta(seconds=INBOUND_RECALL_GRACE_SECONDS)),
                str(previous["occurred_at"]) if previous is not None else None,
                INBOUND_RECALL_ALGORITHM_VERSION,
                str(row["plain_text"] or ""),
                str(row["components_json"] or "[]"),
                self.now_text,
                self.now_text,
            ),
        )


def _json_object(value: object) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}
