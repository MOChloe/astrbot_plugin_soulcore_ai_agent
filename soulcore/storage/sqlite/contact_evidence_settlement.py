"""Atomic resolution of evidence reserved for one proactive contact attempt."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import Any


def _load(value: Any) -> Any:
    if value in (None, ""):
        return {}
    if isinstance(value, (dict, list)):
        return value
    return json.loads(str(value))


class ContactEvidenceSettlement:
    def __init__(
        self,
        *,
        profile_id: str,
        instance_id: str,
        attempt_ref: str,
        generation: int,
        result: str,
        target: str,
        point: str,
    ) -> None:
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.attempt_ref = attempt_ref
        self.generation = generation
        self.result = result
        self.target = target
        self.point = point

    def __call__(self, conn: sqlite3.Connection) -> int:
        rows = conn.execute(
            """SELECT * FROM contact_evidence_reservations WHERE profile_id = ?
            AND instance_id = ? AND attempt_ref = ? AND contact_generation = ?
            AND status = 'RESERVED'""",
            (self.profile_id, self.instance_id, self.attempt_ref, self.generation),
        ).fetchall()
        for row in rows:
            self._settle_reservation(conn, row)
        self._clear_deferred_snapshot(conn)
        return len(rows)

    def _settle_reservation(self, conn: sqlite3.Connection, row: Mapping[str, Any]) -> None:
        conn.execute(
            """UPDATE contact_evidence_reservations SET status = ?,
            resolved_at = ?, resolution_reason = ?, version = version + 1
            WHERE reservation_id = ? AND status = 'RESERVED'""",
            (self.target, self.point, self.result.lower(), row["reservation_id"]),
        )
        kind = str(row["evidence_kind"])
        if kind == "ROLE_TIMELINE_EVENT" and self._consumes_evidence:
            self._settle_timeline_event(conn, row["evidence_ref"])

    @property
    def _consumes_evidence(self) -> bool:
        return self.result in {"DELIVERED", "SUPERSEDED"}

    def _settle_timeline_event(self, conn: sqlite3.Connection, reference: Any) -> None:
        try:
            event_id = int(reference)
        except (TypeError, ValueError):
            raise ValueError("ROLE_TIMELINE_EVENT evidence_ref must be numeric") from None
        conn.execute(
            """UPDATE instance_contact_state SET timeline_event_watermark =
            MAX(timeline_event_watermark, ?), version = version + 1,
            updated_at = ? WHERE profile_id = ? AND instance_id = ?""",
            (event_id, self.point, self.profile_id, self.instance_id),
        )

    def _clear_deferred_snapshot(self, conn: sqlite3.Connection) -> None:
        if not self._consumes_evidence:
            return
        state = conn.execute(
            """SELECT deferred_evidence_json FROM instance_contact_state
            WHERE profile_id = ? AND instance_id = ?""",
            (self.profile_id, self.instance_id),
        ).fetchone()
        frozen = _load(state["deferred_evidence_json"]) if state else {}
        if not self._owns_snapshot(frozen):
            return
        conn.execute(
            """UPDATE instance_contact_state SET deferred_evidence_json = '{}',
            version = version + 1, updated_at = ? WHERE profile_id = ? AND instance_id = ?""",
            (self.point, self.profile_id, self.instance_id),
        )

    def _owns_snapshot(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        return (
            str(value.get("attempt_ref") or "") == self.attempt_ref
            and int(value.get("generation") or 0) == self.generation
        )


__all__ = ["ContactEvidenceSettlement"]
