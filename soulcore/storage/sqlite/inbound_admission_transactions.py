"""Composition-owned cross-feature transaction steps for inbound admission."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from ...features.delivery.sqlite.inbound_admission import (
    InboundAdmissionResult,
    apply_inbound_admission_sql,
    claim_expired_inbound_admission_sql,
    complete_inbound_admission_sql,
    renew_inbound_admission_sql,
)

KnowledgeRefresh = Callable[..., sqlite3.Row | None]
ContactAnswer = Callable[..., sqlite3.Row | None]


@dataclass(frozen=True, slots=True)
class InboundAdmissionTransactions:
    """Keep admission and contact-answer settlement on the caller's connection."""

    mark_contact_answered: ContactAnswer

    def apply(
        self,
        conn: sqlite3.Connection,
        *,
        profile_id: str,
        instance_id: str,
        message_id: int,
        now: datetime,
        group_scope: bool,
        refresh_knowledge_task: KnowledgeRefresh,
        lease_owner: str | None = None,
        lease_token: int | None = None,
    ) -> InboundAdmissionResult:
        return apply_inbound_admission_sql(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            message_id=message_id,
            now=now,
            group_scope=group_scope,
            refresh_knowledge_task=refresh_knowledge_task,
            mark_contact_answered=self.mark_contact_answered,
            lease_owner=lease_owner,
            lease_token=lease_token,
        )

    claim_expired = staticmethod(claim_expired_inbound_admission_sql)
    complete = staticmethod(complete_inbound_admission_sql)
    renew = staticmethod(renew_inbound_admission_sql)


__all__ = ["InboundAdmissionTransactions"]
