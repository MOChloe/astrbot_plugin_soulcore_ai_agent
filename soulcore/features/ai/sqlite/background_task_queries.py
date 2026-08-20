from __future__ import annotations

from datetime import datetime, timedelta

from ....contracts.initialization import INSTANCE_INITIALIZATION_PROGRESS_NOTICE_KIND
from ....storage.sqlite.foreground_continuity import (
    FOREGROUND_RUN_HAS_BACKGROUND_PROJECTION_SQL,
    foreground_message_is_background_cursor_target_sql,
)

_FOREGROUND_QUIET_SECONDS = 60

DUE_BACKGROUND_SLOTS_SQL = f"""WITH eligible AS (
  SELECT state.*, instance.initialization_state,
      instance.initialization_step, instance.config_version,
      instance.continuity_version, instance.publication_version,
      instance.timeline_version, instance.view_version,
      instance.simulated_through_at,
      instance.foreground_message_cursor,
      instance.foreground_run_cursor,
      opening.anchor_at AS initialization_anchor_at,
      core.activity_epoch,
      ROW_NUMBER() OVER (
        PARTITION BY state.profile_id, state.instance_id
        ORDER BY
          CASE WHEN state.hard_due_at IS NULL THEN 1 ELSE 0 END,
          state.hard_due_at,
          CASE WHEN state.next_due_at IS NULL THEN 1 ELSE 0 END,
          state.next_due_at,
          CASE state.author_kind
            WHEN 'KEYFRAME' THEN 0
            WHEN 'ORDINARY' THEN 1
            WHEN 'STORY_SOURCE' THEN 2
            WHEN 'LIFE_DIRECTION' THEN 3
            WHEN 'WORLD' THEN 4
            ELSE 5
          END
      ) AS instance_rank
  FROM background_author_states AS state
  JOIN background_instances AS instance
    ON instance.profile_id = state.profile_id
   AND instance.instance_id = state.instance_id
  JOIN role_profiles AS profile
    ON profile.profile_id = state.profile_id
  JOIN instance_core_state AS core
    ON core.profile_id = state.profile_id
   AND core.instance_id = state.instance_id
  LEFT JOIN background_initialization_openings AS opening
    ON opening.profile_id = state.profile_id
   AND opening.instance_id = state.instance_id
  WHERE profile.background_life_enabled = 1
  AND instance.initialization_state <> 'UNINITIALIZED'
  AND instance.foreground_lease_count = 0
  AND (
    state.author_kind NOT IN ('ORDINARY', 'KEYFRAME')
    OR instance.initialization_step <> 'READY'
    OR instance.last_foreground_at IS NULL
    OR instance.last_foreground_at <= ?
  )
  AND state.status IN ('IDLE', 'FAILED')
  AND state.active_task_id IS NULL
  AND NOT EXISTS (
    SELECT 1 FROM background_author_states AS occupied
    WHERE occupied.profile_id = state.profile_id
      AND occupied.instance_id = state.instance_id
      AND occupied.active_task_id IS NOT NULL
  )
  AND NOT EXISTS (
    SELECT 1 FROM ai_tasks AS background_task
    WHERE background_task.profile_id = state.profile_id
      AND background_task.instance_id = state.instance_id
      AND background_task.task_type = 'BACKGROUND_AUTHOR'
      AND background_task.status NOT IN (
        'DEFERRED','SUCCEEDED','FAILED','CANCELLED'
      )
  )
  AND (
    state.author_kind <> 'KEYFRAME'
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
    WHERE message.profile_id = state.profile_id
      AND message.instance_id = state.instance_id
      AND (
        (message.direction = 'INBOUND'
         AND message.knowledge_eligibility = 'HELD'
         AND NOT (
           instance.initialization_state = 'INITIALIZING'
           AND EXISTS (
             SELECT 1
             FROM deferred_message_items AS initialization_item
             JOIN deferred_message_batches AS initialization_batch
               ON initialization_batch.batch_id = initialization_item.batch_id
             WHERE initialization_item.profile_id = message.profile_id
               AND initialization_item.instance_id = message.instance_id
               AND initialization_item.message_id = message.message_id
               AND initialization_batch.creation_key LIKE 'instance-initialization:%'
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
    WHERE batch.profile_id = state.profile_id
      AND batch.instance_id = state.instance_id
      AND batch.status IN (
        'PENDING','CLASSIFYING','WAITING','CLAIMED'
      )
  )
  AND NOT EXISTS (
    SELECT 1 FROM group_flow_windows window
    WHERE window.profile_id = state.profile_id
      AND window.instance_id = state.instance_id
      AND window.status IN (
        'COLLECTING','JUDGING','READY','RUNNING',
        'WAITING_FIRST_ATTEMPT'
      )
  )
  AND NOT EXISTS (
    SELECT 1 FROM instance_core_runs run
    WHERE run.profile_id = state.profile_id
      AND run.instance_id = state.instance_id
      AND run.status = 'RUNNING'
  )
  AND NOT EXISTS (
    SELECT 1 FROM instance_main_core_occupancies occupancy
    WHERE occupancy.profile_id = state.profile_id
      AND occupancy.instance_id = state.instance_id
      AND occupancy.status = 'ACTIVE'
  )
  AND NOT EXISTS (
    SELECT 1 FROM instance_expression_batches expression
    WHERE expression.profile_id = state.profile_id
      AND expression.instance_id = state.instance_id
      AND expression.status = 'ACTIVE'
  )
  AND NOT EXISTS (
    SELECT 1 FROM instance_outbox outbox
    WHERE outbox.profile_id = state.profile_id
      AND outbox.instance_id = state.instance_id
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
    WHERE retraction.profile_id = state.profile_id
      AND retraction.instance_id = state.instance_id
      AND retraction.status IN ('PENDING','SENDING')
  )
  AND NOT EXISTS (
    SELECT 1 FROM platform_send_permits permit
    WHERE permit.profile_id = state.profile_id
      AND permit.instance_id = state.instance_id
      AND permit.status IN ('RESERVED','DISPATCHING')
      AND (
        permit.status = 'DISPATCHING'
        OR permit.lease_until > ?
      )
  )
  AND NOT EXISTS (
    SELECT 1 FROM ai_tasks main_core
    WHERE main_core.profile_id = state.profile_id
      AND main_core.instance_id = state.instance_id
      AND main_core.task_type = 'MAIN_CORE'
      AND (
        main_core.status IN (
          'RUNNING','PAUSE_REQUESTED','CANCEL_REQUESTED'
        )
        OR (
          main_core.status IN ('READY','SCHEDULED','RETRY_WAIT')
          AND main_core.due_at <= ?
        )
      )
  )
  AND (
    (instance.initialization_step = 'READY'
     AND COALESCE(state.next_due_at, state.hard_due_at) IS NOT NULL
     AND (state.next_due_at <= ? OR state.hard_due_at <= ?))
    OR
    (instance.initialization_step <> 'READY'
     AND state.author_kind = CASE instance.initialization_step
        WHEN 'WORLD' THEN 'WORLD'
        WHEN 'LIFE_DIRECTION' THEN 'LIFE_DIRECTION'
        WHEN 'STORY_SOURCE' THEN 'STORY_SOURCE'
        WHEN 'ORDINARY_CURRENT' THEN CASE
          WHEN COALESCE(opening.keyframe_completed, 0) = 0 THEN 'KEYFRAME'
          ELSE 'ORDINARY'
        END
     END
     AND (
       state.next_due_at IS NULL
       OR state.next_due_at <= ?
       OR state.hard_due_at <= ?
     ))
  )
)
SELECT * FROM eligible
WHERE instance_rank = 1
ORDER BY
  CASE WHEN initialization_step = 'READY' THEN 1 ELSE 0 END,
  CASE WHEN hard_due_at IS NULL THEN 1 ELSE 0 END,
  hard_due_at,
  CASE WHEN next_due_at IS NULL THEN 1 ELSE 0 END,
  next_due_at,
  profile_id, instance_id,
  CASE author_kind
    WHEN 'KEYFRAME' THEN 0
    WHEN 'ORDINARY' THEN 1
    WHEN 'STORY_SOURCE' THEN 2
    WHEN 'LIFE_DIRECTION' THEN 3
    WHEN 'WORLD' THEN 4
    ELSE 5
  END
LIMIT ?"""


def due_background_slot_params(now: str, limit: int) -> tuple[str | int, ...]:
    quiet_cutoff = (
        datetime.fromisoformat(now) - timedelta(seconds=_FOREGROUND_QUIET_SECONDS)
    ).isoformat()
    return (quiet_cutoff, now, now, now, now, now, now, limit)


__all__ = ["DUE_BACKGROUND_SLOTS_SQL", "due_background_slot_params"]
