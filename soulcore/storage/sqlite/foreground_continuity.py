"""Shared SQLite predicates for model-visible foreground continuity."""

from ...contracts.delivery_visibility import (
    DIALOGUE_CONTINUITY_OUTBOUND_STATUSES,
    sql_status_values,
)

_OUTBOUND_STATUS_SQL = sql_status_values(DIALOGUE_CONTINUITY_OUTBOUND_STATUSES)


def _message_alias(message_alias: str) -> str:
    alias = str(message_alias or "").strip()
    if not alias.isidentifier():
        raise ValueError("message alias must be a SQL identifier")
    return alias


def foreground_message_is_background_cursor_target_sql(message_alias: str) -> str:
    """Return the predicate for projected rows an Ordinary cursor must scan."""

    alias = _message_alias(message_alias)
    return f"""
    (
      {alias}.role IN ('user', 'assistant')
      AND {alias}.knowledge_eligibility IN ('ELIGIBLE', 'HELD')
      AND CAST(json_extract(
            {alias}.metadata_json,
            '$.background_foreground_projection.version'
          ) AS INTEGER) = 1
    )
    """.strip()


def foreground_message_is_background_evidence_sql(message_alias: str) -> str:
    """Return the predicate for real dialogue evidence shown to background authors."""

    alias = _message_alias(message_alias)
    return f"""
    (
      CAST(json_extract(
            {alias}.metadata_json,
            '$.background_foreground_projection.version'
          ) AS INTEGER) = 1
      AND (
        (
          {alias}.role = 'user'
          AND {alias}.direction = 'INBOUND'
          AND {alias}.delivery_status = 'RECEIVED'
          AND {alias}.knowledge_eligibility = 'ELIGIBLE'
        )
        OR (
          {alias}.role = 'assistant'
          AND {alias}.direction = 'OUTBOUND'
          AND {alias}.knowledge_eligibility IN ('ELIGIBLE', 'HELD')
          AND (
            NULLIF(TRIM({alias}.internal_memo), '') IS NOT NULL
            OR UPPER({alias}.delivery_status) IN ({_OUTBOUND_STATUS_SQL})
          )
        )
      )
    )
    """.strip()


FOREGROUND_RUN_HAS_BACKGROUND_PROJECTION_SQL = """
(
    EXISTS (
        SELECT 1
        FROM json_each(
            core_run.decision_json,
            '$.foreground_continuity'
        ) AS continuity
        WHERE continuity.type = 'object'
    )
)
""".strip()


__all__ = [
    "FOREGROUND_RUN_HAS_BACKGROUND_PROJECTION_SQL",
    "foreground_message_is_background_cursor_target_sql",
    "foreground_message_is_background_evidence_sql",
]
