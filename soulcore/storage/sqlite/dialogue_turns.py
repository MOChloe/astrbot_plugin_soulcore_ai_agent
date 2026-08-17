from __future__ import annotations

from ...contracts.delivery_visibility import (
    DIALOGUE_CONTINUITY_OUTBOUND_STATUSES,
    sql_status_values,
)

CONTEXT_ELIGIBLE_INBOUND_STATUSES = ("RECEIVED",)
INTERNAL_TIMELINE_DELIVERY_STATUS = "INTERNAL_TIMELINE"
CONTEXT_ONLY_OUTBOUND_STATUSES = (INTERNAL_TIMELINE_DELIVERY_STATUS,)


def _column(message_alias: str, name: str) -> str:
    alias = str(message_alias or "").strip()
    if alias and not alias.isidentifier():
        raise ValueError("message_alias must be a SQL identifier")
    return f"{alias}.{name}" if alias else name


def context_eligible_sql(message_alias: str = "") -> str:
    direction = _column(message_alias, "direction")
    delivery_status = _column(message_alias, "delivery_status")
    inbound = ",".join(f"'{value}'" for value in CONTEXT_ELIGIBLE_INBOUND_STATUSES)
    outbound = sql_status_values(
        (*DIALOGUE_CONTINUITY_OUTBOUND_STATUSES, *CONTEXT_ONLY_OUTBOUND_STATUSES)
    )
    return (
        f"(({direction} = 'INBOUND' AND {delivery_status} IN ({inbound})) "
        f"OR ({direction} = 'OUTBOUND' AND {delivery_status} IN ({outbound})))"
    )


def dialogue_progress_eligible_sql(message_alias: str = "m") -> str:
    """Return SQL for a delivered message whose buffering decision has settled."""

    knowledge_eligibility = _column(message_alias, "knowledge_eligibility")
    return f"({context_eligible_sql(message_alias)} AND {knowledge_eligibility} = 'ELIGIBLE')"


def dialogue_turn_key_sql(message_alias: str = "m") -> str:
    """Return the one authoritative SQL key for a visible speaker turn."""

    message_id = _column(message_alias, "message_id")
    profile_id = _column(message_alias, "profile_id")
    instance_id = _column(message_alias, "instance_id")
    direction = _column(message_alias, "direction")
    expression_batch_id = _column(message_alias, "expression_batch_id")
    return f"""CASE
        WHEN {direction} = 'INBOUND' THEN COALESCE(
            'INBOUND:PRIVATE:' || (
                SELECT member.batch_id
                FROM conversation_turn_buffer_members member
                JOIN conversation_turn_buffer_batches batch
                  ON batch.batch_id = member.batch_id
                 AND batch.profile_id = member.profile_id
                 AND batch.instance_id = member.instance_id
                WHERE member.profile_id = {profile_id}
                  AND member.instance_id = {instance_id}
                  AND member.message_id = {message_id}
                ORDER BY batch.updated_at DESC, member.batch_id DESC
                LIMIT 1
            ),
            'INBOUND:GROUP:' || (
                SELECT member.window_id
                FROM group_flow_window_members member
                WHERE member.profile_id = {profile_id}
                  AND member.instance_id = {instance_id}
                  AND member.message_id = {message_id}
                LIMIT 1
            ),
            'INBOUND:MESSAGE:' || CAST({message_id} AS TEXT)
        )
        WHEN {direction} = 'OUTBOUND' THEN COALESCE(
            'OUTBOUND:EXPRESSION:' || NULLIF(TRIM({expression_batch_id}), ''),
            'OUTBOUND:MESSAGE:' || CAST({message_id} AS TEXT)
        )
        ELSE 'MESSAGE:' || CAST({message_id} AS TEXT)
    END"""


__all__ = [
    "CONTEXT_ELIGIBLE_INBOUND_STATUSES",
    "CONTEXT_ONLY_OUTBOUND_STATUSES",
    "INTERNAL_TIMELINE_DELIVERY_STATUS",
    "context_eligible_sql",
    "dialogue_progress_eligible_sql",
    "dialogue_turn_key_sql",
]
