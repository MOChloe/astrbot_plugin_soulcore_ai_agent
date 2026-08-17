from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from ....storage.sqlite.contact_evidence_settlement import ContactEvidenceSettlement
from .support import _dt, _dump, _load, sqlite3


class ContactSilentDeferralSettlement:
    """Resolve a declined autonomous contact inside the Main Core commit."""

    def __init__(
        self,
        profile_id: str,
        instance_id: str,
        value: Mapping[str, Any] | None,
        now: str,
    ) -> None:
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.value = dict(value) if value else None
        self.now = now

    @property
    def active(self) -> bool:
        return self.value is not None

    @property
    def attempt_ref(self) -> str:
        return str((self.value or {}).get("attempt_ref") or "").strip()

    @property
    def generation(self) -> int:
        return int((self.value or {}).get("generation") or 0)

    @property
    def task_id(self) -> int | None:
        value = int((self.value or {}).get("task_id") or 0)
        return value or None

    @property
    def retry_at(self) -> datetime | None:
        value = (self.value or {}).get("retry_at")
        if value is None or isinstance(value, datetime):
            return value
        raise ValueError("contact silent deferral retry_at must be datetime or null")

    @property
    def next_reroll_count(self) -> int:
        return max(0, int((self.value or {}).get("next_reroll_count") or 0))

    def claim_is_current(self, conn: sqlite3.Connection) -> bool:
        if not self.active:
            return True
        if not self.attempt_ref or self.generation < 1:
            raise ValueError("contact silent deferral requires attempt identity")
        if self.retry_at is not None and self.next_reroll_count < 1:
            raise ValueError("contact silent reroll requires a positive reroll count")
        attempt = conn.execute(
            """SELECT status, task_id FROM contact_attempts WHERE profile_id = ?
            AND instance_id = ? AND attempt_ref = ? AND generation = ?""",
            (self.profile_id, self.instance_id, self.attempt_ref, self.generation),
        ).fetchone()
        if attempt is None or attempt["status"] != "READY":
            return False
        if self.task_id is not None and attempt["task_id"] not in (None, self.task_id):
            return False
        state = conn.execute(
            """SELECT deferred_evidence_json FROM instance_contact_state
            WHERE profile_id = ? AND instance_id = ?""",
            (self.profile_id, self.instance_id),
        ).fetchone()
        snapshot = _load(state["deferred_evidence_json"]) if state else {}
        return bool(
            isinstance(snapshot, dict)
            and str(snapshot.get("attempt_ref") or "") == self.attempt_ref
            and int(snapshot.get("generation") or 0) == self.generation
        )

    def resolve(self, conn: sqlite3.Connection) -> None:
        if not self.active:
            return
        ContactEvidenceSettlement(
            profile_id=self.profile_id,
            instance_id=self.instance_id,
            attempt_ref=self.attempt_ref,
            generation=self.generation,
            result="DEFERRED",
            target="RELEASED",
            point=self.now,
        )(conn)
        attempt = conn.execute(
            """UPDATE contact_attempts SET status = 'FINALIZED', attempted = 0,
            success = 0, answered = 0, task_id = COALESCE(task_id, ?),
            finalized_at = ? WHERE profile_id = ? AND instance_id = ?
              AND attempt_ref = ? AND generation = ? AND status = 'READY'""",
            (
                self.task_id,
                self.now,
                self.profile_id,
                self.instance_id,
                self.attempt_ref,
                self.generation,
            ),
        )
        if attempt.rowcount != 1:
            raise RuntimeError("contact attempt changed while deferring silent Main Core result")
        retry_text = _dt(self.retry_at) if self.retry_at is not None else None
        marker = {"reroll_count": self.next_reroll_count} if retry_text is not None else {}
        state = conn.execute(
            """UPDATE instance_contact_state SET next_check_at = CASE
                WHEN ? IS NULL THEN next_check_at
                WHEN next_check_at IS NULL OR next_check_at > ? THEN ?
                ELSE next_check_at END,
            deferred_evidence_json = ?, last_result = 'DEFERRED',
            last_reason = ?, last_committed_task_id = COALESCE(?, last_committed_task_id),
            version = version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?""",
            (
                retry_text,
                retry_text,
                retry_text,
                _dump(marker),
                (
                    "main_core_silent_reroll"
                    if retry_text is not None
                    else "main_core_silent_no_opening"
                ),
                self.task_id,
                self.now,
                self.profile_id,
                self.instance_id,
            ),
        )
        if state.rowcount != 1:
            raise RuntimeError("contact state disappeared while deferring silent Main Core result")


__all__ = ["ContactSilentDeferralSettlement"]
