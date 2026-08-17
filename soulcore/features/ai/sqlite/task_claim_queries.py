from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from ....contracts.initialization import INSTANCE_INITIALIZATION_PROGRESS_NOTICE_KIND
from ....storage.sqlite.foreground_continuity import (
    FOREGROUND_RUN_HAS_BACKGROUND_PROJECTION_SQL,
    foreground_message_is_background_cursor_target_sql,
)

_FOREGROUND_QUIET_SECONDS = 60

CLAIM_TASK_SQL = f"""SELECT t.* FROM ai_tasks t
LEFT JOIN ai_backends b ON b.backend_id = t.backend_id
WHERE t.status IN ('READY', 'SCHEDULED', 'RETRY_WAIT')
  AND t.due_at <= ? {{task_type_filter}} {{prerequisite_filter}}
  AND EXISTS (SELECT 1 FROM role_profiles runtime_profile
    WHERE runtime_profile.profile_id = t.profile_id
      AND runtime_profile.enabled = 1)
  AND NOT EXISTS (SELECT 1 FROM instance_chat_policies runtime_instance
    WHERE runtime_instance.profile_id = t.profile_id
      AND runtime_instance.instance_id = t.instance_id
      AND runtime_instance.soulcore_enabled = 0)
  AND NOT EXISTS (SELECT 1 FROM ai_manager_pauses p
    WHERE p.paused = 1 AND (
        (p.pause_scope = 'GLOBAL' AND p.scope_key = '')
        OR (p.pause_scope = 'BACKGROUND' AND t.task_class = 'BACKGROUND')
        OR (p.pause_scope = 'BACKEND'
            AND p.scope_key = COALESCE(t.backend_id, ''))
        OR (p.pause_scope = 'CAPABILITY'
            AND p.scope_key = COALESCE(t.capability, ''))))
  AND (b.backend_id IS NULL OR (b.enabled = 1 AND NOT (
    b.circuit_state = 'OPEN' AND b.next_probe_at IS NOT NULL
    AND b.next_probe_at > ?) AND NOT (
      b.circuit_state = 'HALF_OPEN' AND EXISTS (
        SELECT 1 FROM ai_tasks backend_probe
        WHERE backend_probe.backend_id = b.backend_id
          AND backend_probe.status IN ('RUNNING','PAUSE_REQUESTED','CANCEL_REQUESTED')
      )
    )))
  AND (t.mutex_key IS NULL OR NOT EXISTS (
    SELECT 1 FROM ai_tasks active
    WHERE active.task_id <> t.task_id
      AND active.profile_id = t.profile_id
      AND active.instance_id = t.instance_id
      AND active.mutex_key = t.mutex_key
      AND active.status IN (
        'RUNNING','PAUSE_REQUESTED','CANCEL_REQUESTED')))
  AND (
    t.task_type <> 'BACKGROUND_AUTHOR'
    OR EXISTS (
      SELECT 1
      FROM background_instances instance
      JOIN instance_core_state core
        ON core.profile_id = instance.profile_id
       AND core.instance_id = instance.instance_id
      JOIN background_author_states author
        ON author.profile_id = instance.profile_id
       AND author.instance_id = instance.instance_id
       AND author.author_kind =
         json_extract(t.input_json, '$.payload.author_kind')
      LEFT JOIN background_initialization_openings AS opening
        ON opening.profile_id = instance.profile_id
       AND opening.instance_id = instance.instance_id
      WHERE instance.profile_id = t.profile_id
        AND instance.instance_id = t.instance_id
        AND instance.enabled = 1
        AND instance.initialization_state <> 'UNINITIALIZED'
        AND instance.foreground_lease_count = 0
        AND (
          author.author_kind NOT IN ('ORDINARY', 'KEYFRAME')
          OR instance.initialization_step <> 'READY'
          OR instance.last_foreground_at IS NULL
          OR instance.last_foreground_at <= ?
        )
        AND instance.initialization_step =
          json_extract(
            t.input_json, '$.payload.initialization_step'
          )
        AND core.activity_epoch =
          CAST(json_extract(
            t.input_json, '$.payload.activity_epoch'
          ) AS INTEGER)
        AND instance.continuity_version =
          CAST(json_extract(
            t.input_json, '$.payload.continuity_version'
          ) AS INTEGER)
        AND instance.config_version =
          CAST(json_extract(
            t.input_json, '$.payload.config_version'
          ) AS INTEGER)
        AND instance.publication_version =
          CAST(json_extract(
            t.input_json, '$.payload.publication_version'
          ) AS INTEGER)
        AND instance.timeline_version =
          CAST(json_extract(
            t.input_json, '$.payload.timeline_version'
          ) AS INTEGER)
        AND instance.view_version =
          CAST(json_extract(
            t.input_json, '$.payload.view_version'
          ) AS INTEGER)
        AND instance.foreground_message_cursor =
          CAST(json_extract(
            t.input_json, '$.payload.foreground_message_cursor'
          ) AS INTEGER)
        AND instance.foreground_run_cursor =
          CAST(json_extract(
            t.input_json, '$.payload.foreground_run_cursor'
          ) AS INTEGER)
        AND COALESCE(instance.simulated_through_at, '') =
          COALESCE(json_extract(
            t.input_json, '$.payload.simulated_through_at'
          ), '')
        AND author.active_task_id = t.task_id
        AND author.generation = t.generation
        AND author.status = 'ENQUEUED'
        AND author.state_version =
          CAST(json_extract(
            t.input_json, '$.payload.author_state_version'
          ) AS INTEGER)
        AND (
          author.author_kind <> 'KEYFRAME'
          OR (
            instance.initialization_state = 'INITIALIZING'
            AND instance.initialization_step = 'ORDINARY_CURRENT'
            AND COALESCE(opening.keyframe_completed, 0) = 0
          )
          OR (
            instance.ordinary_since_keyframe > 0
            AND NOT EXISTS (
              SELECT 1 FROM instance_messages AS keyframe_message
              WHERE keyframe_message.profile_id = instance.profile_id
                AND keyframe_message.instance_id = instance.instance_id
                AND keyframe_message.message_id > instance.foreground_message_cursor
                AND {foreground_message_is_background_cursor_target_sql("keyframe_message")}
            )
            AND NOT EXISTS (
              SELECT 1 FROM instance_core_runs AS core_run
              WHERE core_run.profile_id = instance.profile_id
                AND core_run.instance_id = instance.instance_id
                AND core_run.run_id > instance.foreground_run_cursor
                AND core_run.status = 'COMPLETED'
                AND core_run.source IN ('FOREGROUND_MESSAGE', 'DEFERRED_MESSAGE')
                AND core_run.decision_json IS NOT NULL
                AND {FOREGROUND_RUN_HAS_BACKGROUND_PROJECTION_SQL}
            )
          )
        )
        AND NOT EXISTS (
          SELECT 1 FROM instance_messages message
          WHERE message.profile_id = t.profile_id
            AND message.instance_id = t.instance_id
            AND (
              (message.direction = 'INBOUND'
               AND message.knowledge_eligibility = 'HELD'
               AND NOT (
                 instance.initialization_state = 'INITIALIZING'
                 AND EXISTS (
                   SELECT 1
                   FROM deferred_message_items AS initialization_item
                   JOIN deferred_message_batches AS initialization_batch
                     ON initialization_batch.batch_id =
                        initialization_item.batch_id
                   WHERE initialization_item.profile_id = message.profile_id
                     AND initialization_item.instance_id = message.instance_id
                     AND initialization_item.message_id = message.message_id
                     AND initialization_batch.creation_key LIKE
                         'instance-initialization:%'
                     AND initialization_batch.status IN ('PENDING', 'CLAIMED')
                 )
               ))
              OR
              (message.direction = 'OUTBOUND'
               AND message.delivery_status = 'PENDING')
            )
        )
        AND NOT EXISTS (
          SELECT 1 FROM conversation_turn_buffer_batches batch
          WHERE batch.profile_id = t.profile_id
            AND batch.instance_id = t.instance_id
            AND batch.status IN (
              'PENDING','CLASSIFYING','WAITING','CLAIMED'
            )
        )
        AND NOT EXISTS (
          SELECT 1 FROM group_flow_windows window
          WHERE window.profile_id = t.profile_id
            AND window.instance_id = t.instance_id
            AND window.status IN (
              'COLLECTING','JUDGING','READY','RUNNING',
              'WAITING_FIRST_ATTEMPT'
            )
        )
        AND NOT EXISTS (
          SELECT 1 FROM instance_core_runs run
          WHERE run.profile_id = t.profile_id
            AND run.instance_id = t.instance_id
            AND run.status = 'RUNNING'
        )
        AND NOT EXISTS (
          SELECT 1 FROM instance_main_core_occupancies occupancy
          WHERE occupancy.profile_id = t.profile_id
            AND occupancy.instance_id = t.instance_id
            AND occupancy.status = 'ACTIVE'
        )
        AND NOT EXISTS (
          SELECT 1 FROM instance_expression_batches expression
          WHERE expression.profile_id = t.profile_id
            AND expression.instance_id = t.instance_id
            AND expression.status = 'ACTIVE'
        )
        AND NOT EXISTS (
          SELECT 1 FROM instance_outbox outbox
          WHERE outbox.profile_id = t.profile_id
            AND outbox.instance_id = t.instance_id
            AND outbox.status IN ('PENDING','SENDING')
            AND NOT (
              instance.initialization_state = 'INITIALIZING'
              AND outbox.origin_kind = 'SYSTEM_EVENT'
              AND json_extract(outbox.payload_json, '$.system_notice_kind')
                = '{INSTANCE_INITIALIZATION_PROGRESS_NOTICE_KIND}'
            )
        )
        AND NOT EXISTS (
          SELECT 1 FROM message_retraction_actions retraction
          WHERE retraction.profile_id = t.profile_id
            AND retraction.instance_id = t.instance_id
            AND retraction.status IN ('PENDING','SENDING')
        )
        AND NOT EXISTS (
          SELECT 1 FROM platform_send_permits permit
          WHERE permit.profile_id = t.profile_id
            AND permit.instance_id = t.instance_id
            AND permit.status IN ('RESERVED','DISPATCHING')
            AND (
              permit.status = 'DISPATCHING'
              OR permit.lease_until > ?
            )
        )
        AND NOT EXISTS (
          SELECT 1 FROM ai_tasks main_core
          WHERE main_core.task_id <> t.task_id
            {{main_core_exemption}}
            AND main_core.profile_id = t.profile_id
            AND main_core.instance_id = t.instance_id
            AND main_core.task_type = 'MAIN_CORE'
            AND (
              main_core.status IN (
                'RUNNING','PAUSE_REQUESTED','CANCEL_REQUESTED'
              )
              OR (
                main_core.status IN (
                  'READY','SCHEDULED','RETRY_WAIT'
                )
                AND main_core.due_at <= ?
              )
            )
        )
    )
  )
ORDER BY t.priority DESC, t.due_at, t.task_id LIMIT 1"""

_PREREQUISITE_REQUESTER_SQL = """EXISTS (
    SELECT 1 FROM ai_tasks requester
    WHERE requester.task_id = ?
      AND requester.lease_token = ?
      AND requester.lease_owner = ?
      AND requester.status = 'RUNNING'
      AND requester.lease_until > ?
      AND requester.profile_id = t.profile_id
      AND requester.instance_id = t.instance_id
      AND (
        requester.task_type = 'TIMER_RUN'
        OR (
          requester.task_type = 'MAIN_CORE'
          AND json_extract(
            requester.input_json, '$.payload.source'
          ) IN ('TIMER', 'PLUGIN_WAKE')
        )
      )
  )"""

_PREREQUISITE_FILTER_SQL = f"""
  AND t.task_id = ?
  AND t.task_type = 'BACKGROUND_AUTHOR'
  AND json_extract(
    t.input_json, '$.payload.author_kind'
  ) IN ('ORDINARY', 'KEYFRAME')
  AND json_type(
    t.input_json, '$.payload.proactive_frame'
  ) = 'object'
  AND {_PREREQUISITE_REQUESTER_SQL}"""

ACTIVE_PREREQUISITE_TASK_SQL = f"""SELECT t.* FROM ai_tasks t
WHERE t.task_id = ?
  AND t.status IN ('RUNNING', 'PAUSE_REQUESTED', 'CANCEL_REQUESTED')
  AND t.task_type = 'BACKGROUND_AUTHOR'
  AND json_extract(
    t.input_json, '$.payload.author_kind'
  ) IN ('ORDINARY', 'KEYFRAME')
  AND json_type(
    t.input_json, '$.payload.proactive_frame'
  ) = 'object'
  AND {_PREREQUISITE_REQUESTER_SQL}
LIMIT 1"""


@dataclass(frozen=True, slots=True)
class TaskClaimQuery:
    sql: str
    params: tuple[object, ...]


def build_task_claim_query(
    now_text: str,
    normalized_types: Sequence[str],
) -> TaskClaimQuery:
    type_filter = ""
    if normalized_types:
        placeholders = ",".join("?" for _ in normalized_types)
        type_filter = f" AND t.task_type IN ({placeholders})"
    quiet_cutoff = (
        datetime.fromisoformat(now_text) - timedelta(seconds=_FOREGROUND_QUIET_SECONDS)
    ).isoformat()
    params = (
        now_text,
        *normalized_types,
        now_text,
        quiet_cutoff,
        now_text,
        now_text,
    )
    return TaskClaimQuery(
        sql=CLAIM_TASK_SQL.format(
            task_type_filter=type_filter,
            prerequisite_filter="",
            main_core_exemption="",
        ),
        params=params,
    )


def build_prerequisite_task_claim_query(
    now_text: str,
    *,
    task_id: int,
    requester_task_id: int,
    requester_lease_token: int,
    worker_id: str,
) -> TaskClaimQuery:
    """Select one proactive frame while exempting only its live requester."""

    quiet_cutoff = (
        datetime.fromisoformat(now_text) - timedelta(seconds=_FOREGROUND_QUIET_SECONDS)
    ).isoformat()
    return TaskClaimQuery(
        sql=CLAIM_TASK_SQL.format(
            task_type_filter="",
            prerequisite_filter=_PREREQUISITE_FILTER_SQL,
            main_core_exemption="AND main_core.task_id <> ?",
        ),
        params=(
            now_text,
            int(task_id),
            int(requester_task_id),
            int(requester_lease_token),
            worker_id,
            now_text,
            now_text,
            quiet_cutoff,
            now_text,
            int(requester_task_id),
            now_text,
        ),
    )


def build_active_prerequisite_task_query(
    now_text: str,
    *,
    task_id: int,
    requester_task_id: int,
    requester_lease_token: int,
    worker_id: str,
) -> TaskClaimQuery:
    """Verify that the same requester's proactive frame is already active."""

    return TaskClaimQuery(
        sql=ACTIVE_PREREQUISITE_TASK_SQL,
        params=(
            int(task_id),
            int(requester_task_id),
            int(requester_lease_token),
            worker_id,
            now_text,
        ),
    )


__all__ = [
    "TaskClaimQuery",
    "build_active_prerequisite_task_query",
    "build_prerequisite_task_claim_query",
    "build_task_claim_query",
]
