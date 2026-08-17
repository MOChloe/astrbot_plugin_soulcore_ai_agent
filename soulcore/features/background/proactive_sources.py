"""Read-only discovery of predictable proactive MainCore schedules.

The scanner deliberately projects only durable scheduling metadata.  Nothing
from this module is rendered into a model prompt.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from .prewarm import (
    PROACTIVE_FRAME_GAP_MINUTES,
    ProactiveFrameCandidate,
    ProactiveFrameSourceKind,
)

PREDICTABLE_PROACTIVE_SOURCE_LIMIT = 100
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PredictableProactiveSource:
    """One durable source whose MainCore admission time is already known."""

    candidate: ProactiveFrameCandidate
    record_version: int
    contact_generation: int = 0
    contact_last_success_at: datetime | None = None
    contact_daily_bucket: str = ""
    contact_daily_success_count: int = 0
    contact_consecutive_unanswered: int = 0


class PredictableSourceDatabasePort(Protocol):
    async def fetch_all(
        self,
        sql: str,
        params: tuple[object, ...] = (),
    ) -> list[Mapping[str, Any]]: ...


_TASK_SOURCE_SQL = """
CASE WHEN json_valid(task.input_json)
     THEN json_extract(task.input_json, '$.payload.source') ELSE NULL END
"""
_TASK_CONTACT_REF_SQL = """
CASE WHEN json_valid(task.input_json)
     THEN json_extract(
         task.input_json, '$.payload.metadata.contact_attempt_ref'
     ) ELSE NULL END
"""
_TASK_WAKEUP_ID_SQL = """
CASE WHEN json_valid(task.input_json)
     THEN json_extract(task.input_json, '$.payload.wakeup_id') ELSE NULL END
"""
_TASK_PREWARM_REF_SQL = """
CASE WHEN json_valid(task.input_json)
     THEN json_extract(
         task.input_json, '$.payload._proactive_frame_schedule.source_ref'
     ) ELSE NULL END
"""
_TASK_PREWARM_PLANNED_SQL = """
CASE WHEN json_valid(task.input_json)
     THEN json_extract(
         task.input_json, '$.payload._proactive_frame_schedule.planned_main_core_at'
     ) ELSE NULL END
"""
_TASK_EFFECTIVE_PLANNED_SQL = (
    f"COALESCE(NULLIF(TRIM(({_TASK_PREWARM_PLANNED_SQL})), ''), task.due_at)"
)


PREDICTABLE_PROACTIVE_SOURCES_SQL = f"""
WITH candidate AS (
    SELECT occurrence.profile_id, occurrence.instance_id,
           'TIMER' AS source_kind,
           'timer-occurrence:' || occurrence.stable_ref AS source_ref,
           occurrence.original_due_at AS planned_at,
           occurrence.original_due_at AS seed_planned_at,
           occurrence.version AS record_version,
           0 AS contact_generation,
           NULL AS contact_last_success_at,
           '' AS contact_daily_bucket,
           0 AS contact_daily_success_count,
           0 AS contact_consecutive_unanswered
    FROM timer_occurrences AS occurrence
    JOIN timer_rules AS rule
      ON rule.profile_id = occurrence.profile_id
     AND rule.instance_id = occurrence.instance_id
     AND rule.rule_id = occurrence.rule_id
    WHERE occurrence.status = 'SCHEDULED'
      AND rule.status = 'ACTIVE'
      AND occurrence.original_due_at > ?
      AND occurrence.original_due_at <= ?

    UNION ALL

    SELECT wakeup.profile_id, wakeup.instance_id,
           'WAKEUP' AS source_kind,
           'instance-wakeup:' || CAST(wakeup.wakeup_id AS TEXT) AS source_ref,
           wakeup.due_at AS planned_at,
           wakeup.due_at AS seed_planned_at,
           wakeup.version AS record_version,
           0 AS contact_generation,
           NULL AS contact_last_success_at,
           '' AS contact_daily_bucket,
           0 AS contact_daily_success_count,
           0 AS contact_consecutive_unanswered
    FROM instance_wakeups AS wakeup
    WHERE wakeup.status = 'PENDING'
      AND wakeup.source IN ('PLUGIN_WAKE', 'TIMER')
      AND wakeup.due_at > ?
      AND wakeup.due_at <= ?

    UNION ALL

    SELECT contact.profile_id, contact.instance_id,
           'CONTACT' AS source_kind,
           'contact-check:' || contact.profile_id || ':'
             || contact.instance_id || ':' || contact.next_check_at AS source_ref,
           contact.next_check_at AS planned_at,
           contact.next_check_at AS seed_planned_at,
           contact.version AS record_version,
           contact.generation AS contact_generation,
           contact.last_success_at AS contact_last_success_at,
           contact.daily_bucket AS contact_daily_bucket,
           contact.daily_success_count AS contact_daily_success_count,
           contact.consecutive_unanswered AS contact_consecutive_unanswered
    FROM instance_contact_state AS contact
    WHERE contact.next_check_at IS NOT NULL
      AND (contact.lease_until IS NULL OR contact.lease_until <= ?)
      AND contact.next_check_at > ?
      AND contact.next_check_at <= ?

    UNION ALL

    SELECT task.profile_id, task.instance_id,
           'AI_TASK' AS source_kind,
           CASE
             WHEN TRIM(COALESCE(({_TASK_PREWARM_REF_SQL}), '')) <> ''
               THEN ({_TASK_PREWARM_REF_SQL})
             WHEN CAST(({_TASK_WAKEUP_ID_SQL}) AS INTEGER) > 0
               THEN 'instance-wakeup:'
                    || CAST(({_TASK_WAKEUP_ID_SQL}) AS INTEGER)
             WHEN TRIM(COALESCE(({_TASK_CONTACT_REF_SQL}), '')) <> ''
               THEN 'contact-attempt:' || ({_TASK_CONTACT_REF_SQL})
             WHEN TRIM(COALESCE(task.idempotency_key, '')) <> ''
               THEN 'ai-task-key:' || task.task_type || ':' || task.idempotency_key
             ELSE 'ai-task:' || CAST(task.task_id AS TEXT)
           END AS source_ref,
           task.due_at AS planned_at,
           ({_TASK_EFFECTIVE_PLANNED_SQL}) AS seed_planned_at,
           task.version AS record_version,
           0 AS contact_generation,
           NULL AS contact_last_success_at,
           '' AS contact_daily_bucket,
           0 AS contact_daily_success_count,
           0 AS contact_consecutive_unanswered
    FROM ai_tasks AS task
    WHERE task.task_type = 'MAIN_CORE'
      AND task.task_class = 'BACKGROUND'
      AND task.status IN ('SCHEDULED', 'RETRY_WAIT')
      AND ({_TASK_SOURCE_SQL}) IN ('PLUGIN_WAKE', 'TIMER')
      AND NOT (
        task.status = 'SCHEDULED'
        AND TRIM(COALESCE(({_TASK_CONTACT_REF_SQL}), '')) <> ''
      )
      AND task.due_at > ?
      AND task.due_at <= ?
)
SELECT candidate.*
FROM candidate
JOIN role_profiles AS profile
  ON profile.profile_id = candidate.profile_id
JOIN character_instances AS instance
  ON instance.profile_id = candidate.profile_id
 AND instance.instance_id = candidate.instance_id
JOIN background_instances AS background
  ON background.profile_id = candidate.profile_id
 AND background.instance_id = candidate.instance_id
LEFT JOIN instance_chat_policies AS chat_policy
  ON chat_policy.profile_id = candidate.profile_id
 AND chat_policy.instance_id = candidate.instance_id
WHERE profile.enabled = 1
  AND COALESCE(chat_policy.soulcore_enabled, 1) = 1
  AND instance.readiness = 'READY'
  AND instance.initialization_state = 'READY'
  AND background.enabled = 1
  AND background.proactive_frame_prewarm_enabled = 1
  AND background.initialization_state = 'READY'
  AND background.initialization_step = 'READY'
ORDER BY candidate.planned_at, candidate.source_kind, candidate.source_ref
LIMIT ?
"""


async def scan_predictable_proactive_sources(
    database: PredictableSourceDatabasePort,
    *,
    now: datetime,
    limit: int = PREDICTABLE_PROACTIVE_SOURCE_LIMIT,
) -> tuple[PredictableProactiveSource, ...]:
    """Return future sources close enough for their random cutoff to be due."""

    now = _utc_datetime(now, "predictable source scan time")
    scan_through = now + timedelta(minutes=PROACTIVE_FRAME_GAP_MINUTES[2])
    now_text = now.isoformat()
    through_text = scan_through.isoformat()
    bounded_limit = max(1, min(int(limit), PREDICTABLE_PROACTIVE_SOURCE_LIMIT))
    params: tuple[object, ...] = (
        # Timer occurrence window.
        now_text,
        through_text,
        # Durable wakeup window.
        now_text,
        through_text,
        # Contact lease recovery boundary followed by its schedule window.
        now_text,
        now_text,
        through_text,
        # Already-materialized proactive MainCore task window.
        now_text,
        through_text,
        bounded_limit,
    )
    rows = await database.fetch_all(PREDICTABLE_PROACTIVE_SOURCES_SQL, params)
    sources: list[PredictableProactiveSource] = []
    for row in rows:
        try:
            sources.append(_source_from_row(row))
        except (TypeError, ValueError) as exc:
            logger.warning(
                "invalid predictable proactive source skipped for %s/%s: %s",
                str(row.get("profile_id") or ""),
                str(row.get("instance_id") or ""),
                exc,
            )
    return _deduplicate_sources(tuple(sources))


def _source_from_row(row: Mapping[str, Any]) -> PredictableProactiveSource:
    values = dict(row)
    planned_at = _datetime(values.get("planned_at"), required=True)
    assert planned_at is not None
    seed_planned_at = _datetime(values.get("seed_planned_at"), required=True)
    assert seed_planned_at is not None
    source_kind = ProactiveFrameSourceKind(str(values.get("source_kind") or ""))
    source_ref = str(values.get("source_ref") or "")
    if source_kind is ProactiveFrameSourceKind.CONTACT:
        source_ref = (
            f"contact-check:{str(values.get('profile_id') or '')}:"
            f"{str(values.get('instance_id') or '')}:{planned_at.isoformat()}"
        )
    return PredictableProactiveSource(
        candidate=ProactiveFrameCandidate(
            profile_id=str(values.get("profile_id") or ""),
            instance_id=str(values.get("instance_id") or ""),
            source_kind=source_kind,
            source_ref=source_ref,
            planned_main_core_at=planned_at,
            seed_planned_main_core_at=seed_planned_at,
        ),
        record_version=max(0, int(values.get("record_version") or 0)),
        contact_generation=max(0, int(values.get("contact_generation") or 0)),
        contact_last_success_at=_datetime(values.get("contact_last_success_at")),
        contact_daily_bucket=str(values.get("contact_daily_bucket") or ""),
        contact_daily_success_count=max(0, int(values.get("contact_daily_success_count") or 0)),
        contact_consecutive_unanswered=max(
            0, int(values.get("contact_consecutive_unanswered") or 0)
        ),
    )


def _deduplicate_sources(
    sources: tuple[PredictableProactiveSource, ...],
) -> tuple[PredictableProactiveSource, ...]:
    unique: dict[tuple[str, str, str], PredictableProactiveSource] = {}
    for source in sources:
        candidate = source.candidate
        key = (candidate.profile_id, candidate.instance_id, candidate.source_ref)
        prior = unique.get(key)
        if prior is None or (candidate.planned_main_core_at < prior.candidate.planned_main_core_at):
            unique[key] = source
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.candidate.planned_main_core_at,
                item.candidate.source_kind.value,
                item.candidate.source_ref,
            ),
        )
    )


def _datetime(value: object, *, required: bool = False) -> datetime | None:
    if value in (None, ""):
        if required:
            raise ValueError("predictable proactive source is missing its planned time")
        return None
    parsed = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    return _utc_datetime(parsed, "predictable proactive source time")


def _require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")


def _utc_datetime(value: datetime, label: str) -> datetime:
    _require_aware(value, label)
    return value.astimezone(UTC)


__all__ = [
    "PREDICTABLE_PROACTIVE_SOURCE_LIMIT",
    "PREDICTABLE_PROACTIVE_SOURCES_SQL",
    "PredictableProactiveSource",
    "PredictableSourceDatabasePort",
    "scan_predictable_proactive_sources",
]
