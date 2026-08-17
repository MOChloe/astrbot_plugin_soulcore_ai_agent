"""Sparse administrator policy for one private chat or group route."""

from __future__ import annotations

import sqlite3

from ....contracts.message_reference import normalize_private_fallback_player_name
from ....contracts.models import InstanceChatPolicy, OutboxStatus
from ....storage.sqlite.chat_policy_delivery import (
    _cancel_terminal_expression_suffix,
    _resolve_terminal_group_window,
    restore_cancelled_file_todos,
    schedule_outbox_voice_artifact_cleanup_sql,
)
from ....storage.sqlite.codec import _dt, _load, _now, _parse
from ....storage.sqlite.expression_batch_lifecycle import (
    settle_pending_outbox_row,
    sync_expression_batch_status,
)


class InstanceChatPolicyRecords:
    async def get_instance_chat_policy(
        self,
        profile_id: str,
        instance_id: str,
    ) -> InstanceChatPolicy:
        row = await self.db.fetch_one(
            """SELECT profile_id, instance_id, soulcore_enabled, image_send_enabled,
                private_fallback_player_name, private_name_override_enabled,
                version, created_at, updated_at
            FROM instance_chat_policies
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        )
        if row is None:
            return InstanceChatPolicy(profile_id=profile_id, instance_id=instance_id)
        return InstanceChatPolicy(
            profile_id=str(row["profile_id"]),
            instance_id=str(row["instance_id"]),
            soulcore_enabled=bool(row["soulcore_enabled"]),
            image_send_enabled=bool(row["image_send_enabled"]),
            private_fallback_player_name=str(row["private_fallback_player_name"] or ""),
            private_name_override_enabled=bool(row["private_name_override_enabled"]),
            version=int(row["version"]),
            created_at=_parse(row["created_at"]),
            updated_at=_parse(row["updated_at"]),
        )

    async def upsert_instance_chat_policy(
        self,
        profile_id: str,
        instance_id: str,
        *,
        soulcore_enabled: bool,
        image_send_enabled: bool,
        expected_version: int,
        private_fallback_player_name: str = "",
        private_name_override_enabled: bool = False,
    ) -> InstanceChatPolicy | None:
        if expected_version < 0:
            raise ValueError("instance chat policy version must not be negative")
        fallback_name = normalize_private_fallback_player_name(private_fallback_player_name)
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> int:
            return _upsert_instance_chat_policy(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                soulcore_enabled=soulcore_enabled,
                image_send_enabled=image_send_enabled,
                private_fallback_player_name=fallback_name,
                private_name_override_enabled=private_name_override_enabled,
                expected_version=expected_version,
                now=now,
            )

        changed = await self.uow.run(operation)
        if changed != 1:
            return None
        await self.db.publish_backup_after_commit()
        return await self.get_instance_chat_policy(profile_id, instance_id)


def _upsert_instance_chat_policy(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    soulcore_enabled: bool,
    image_send_enabled: bool,
    private_fallback_player_name: str,
    private_name_override_enabled: bool,
    expected_version: int,
    now: str,
) -> int:
    instance = conn.execute(
        """SELECT scope FROM character_instances
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    if instance is None:
        raise KeyError((profile_id, instance_id))
    if str(instance["scope"]) != "private" and (
        private_fallback_player_name or private_name_override_enabled
    ):
        raise ValueError("private chat names are only available for private chats")
    if private_name_override_enabled and not private_fallback_player_name:
        raise ValueError("private name override requires a configured private chat name")
    previous = conn.execute(
        """SELECT soulcore_enabled, image_send_enabled
        FROM instance_chat_policies
        WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    previous_soulcore_enabled = bool(previous["soulcore_enabled"]) if previous is not None else True
    previous_image_send_enabled = (
        bool(previous["image_send_enabled"]) if previous is not None else True
    )
    changed = _write_instance_chat_policy(
        conn,
        profile_id=profile_id,
        instance_id=instance_id,
        soulcore_enabled=soulcore_enabled,
        image_send_enabled=image_send_enabled,
        private_fallback_player_name=private_fallback_player_name,
        private_name_override_enabled=private_name_override_enabled,
        expected_version=expected_version,
        now=now,
    )
    if changed != 1:
        return changed
    if previous_soulcore_enabled and not soulcore_enabled:
        _cancel_disabled_instance_work(
            conn, profile_id=profile_id, instance_id=instance_id, now=now
        )
    if previous_image_send_enabled and not image_send_enabled:
        _cancel_disabled_image_delivery(
            conn, profile_id=profile_id, instance_id=instance_id, now=now
        )
    return changed


def _write_instance_chat_policy(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    soulcore_enabled: bool,
    image_send_enabled: bool,
    private_fallback_player_name: str,
    private_name_override_enabled: bool,
    expected_version: int,
    now: str,
) -> int:
    if expected_version == 0:
        cursor = conn.execute(
            """INSERT OR IGNORE INTO instance_chat_policies(
                profile_id, instance_id, soulcore_enabled, image_send_enabled,
                private_fallback_player_name, private_name_override_enabled,
                version, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (
                profile_id,
                instance_id,
                int(soulcore_enabled),
                int(image_send_enabled),
                private_fallback_player_name,
                int(private_name_override_enabled),
                now,
                now,
            ),
        )
    else:
        cursor = conn.execute(
            """UPDATE instance_chat_policies SET
                soulcore_enabled = ?, image_send_enabled = ?,
                private_fallback_player_name = ?, private_name_override_enabled = ?,
                version = version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND version = ?""",
            (
                int(soulcore_enabled),
                int(image_send_enabled),
                private_fallback_player_name,
                int(private_name_override_enabled),
                now,
                profile_id,
                instance_id,
                expected_version,
            ),
        )
    return int(cursor.rowcount)


def _cancel_disabled_instance_work(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    now: str,
) -> None:
    rows = list(
        conn.execute(
            """SELECT * FROM instance_outbox
            WHERE profile_id = ? AND instance_id = ? AND status = 'PENDING'""",
            (profile_id, instance_id),
        )
    )
    _cancel_pending_policy_rows(
        conn,
        rows,
        profile_id=profile_id,
        instance_id=instance_id,
        reason="instance_disabled",
        now=now,
    )
    conn.execute(
        """UPDATE instance_wakeups
        SET status = 'CANCELLED', last_error = 'instance_disabled',
            lease_until = NULL, version = version + 1, updated_at = ?
        WHERE profile_id = ? AND instance_id = ?
          AND status IN ('PENDING', 'CLAIMED')""",
        (now, profile_id, instance_id),
    )
    conn.execute(
        """UPDATE instance_core_state
        SET state_epoch = state_epoch + 1, activity_epoch = activity_epoch + 1,
            updated_at = ?
        WHERE profile_id = ? AND instance_id = ?""",
        (now, profile_id, instance_id),
    )
    _settle_all_pending_contact_attempts(
        conn,
        profile_id=profile_id,
        instance_id=instance_id,
        reason="instance_disabled",
        now=now,
    )


def _cancel_disabled_image_delivery(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    now: str,
) -> None:
    contact_attempts = _pending_image_contact_attempts(
        conn,
        profile_id=profile_id,
        instance_id=instance_id,
    )
    rows = list(
        conn.execute(
            """SELECT * FROM instance_outbox
        WHERE profile_id = ? AND instance_id = ? AND status = 'PENDING'
          AND (
            CASE WHEN json_valid(payload_json)
              THEN UPPER(COALESCE(json_extract(
                  payload_json, '$.expression_kind'
              ), '')) ELSE '' END = 'IMAGE'
            OR EXISTS (
                SELECT 1 FROM json_each(
                    CASE WHEN json_valid(payload_json) THEN payload_json ELSE '{}' END,
                    '$.components'
                ) component
                WHERE LOWER(COALESCE(json_extract(component.value, '$.type'), ''))
                    = 'image_asset'
            )
          )""",
            (profile_id, instance_id),
        )
    )
    _cancel_pending_policy_rows(
        conn,
        rows,
        profile_id=profile_id,
        instance_id=instance_id,
        reason="instance_image_send_disabled",
        now=now,
    )
    for attempt_ref, generation in contact_attempts:
        if _has_active_contact_outbox(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            attempt_ref=attempt_ref,
            generation=generation,
        ):
            continue
        _settle_one_pending_contact_attempt(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            attempt_ref=attempt_ref,
            generation=generation,
            reason="instance_image_send_disabled",
            now=now,
        )


def _cancel_pending_policy_rows(
    conn: sqlite3.Connection,
    rows: list[sqlite3.Row],
    *,
    profile_id: str,
    instance_id: str,
    reason: str,
    now: str,
) -> None:
    cancelled: list[sqlite3.Row] = []
    for row in rows:
        if not settle_pending_outbox_row(
            conn,
            row,
            status=OutboxStatus.CANCELLED,
            reason=reason,
            error_code=reason,
            now=now,
        ):
            continue
        cancelled.append(row)
        schedule_outbox_voice_artifact_cleanup_sql(
            conn,
            profile_id=profile_id,
            instance_id=instance_id,
            outbox_id=int(row["outbox_id"]),
            reason=f"voice_outbox_{reason}",
            now=now,
        )
        _cancel_terminal_expression_suffix(conn, row, OutboxStatus.CANCELLED, now)
        sync_expression_batch_status(conn, row["expression_batch_id"], now)
        _resolve_terminal_group_window(conn, row, OutboxStatus.CANCELLED, now)
    restore_cancelled_file_todos(
        conn,
        profile_id,
        instance_id,
        cancelled,
        now,
        reason,
        load_payload=_load,
    )


def _settle_all_pending_contact_attempts(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    reason: str,
    now: str,
) -> None:
    conn.execute(
        """UPDATE contact_attempts
        SET status = 'FINALIZED', attempted = 0, success = 0, answered = 0,
            finalized_at = ?
        WHERE profile_id = ? AND instance_id = ? AND status = 'READY'""",
        (now, profile_id, instance_id),
    )
    conn.execute(
        """UPDATE contact_evidence_reservations
        SET status = 'STALE', resolved_at = ?, resolution_reason = ?,
            version = version + 1
        WHERE profile_id = ? AND instance_id = ? AND status = 'RESERVED'""",
        (now, reason, profile_id, instance_id),
    )
    conn.execute(
        """UPDATE instance_contact_state
        SET deferred_evidence_json = '{}', lease_until = NULL,
            lease_token = lease_token + 1, generation = generation + 1,
            version = version + 1, updated_at = ?
        WHERE profile_id = ? AND instance_id = ?""",
        (now, profile_id, instance_id),
    )


def _pending_image_contact_attempts(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
) -> tuple[tuple[str, int], ...]:
    rows = conn.execute(
        """SELECT DISTINCT
            TRIM(COALESCE(json_extract(payload_json, '$.contact_attempt_ref'), '')),
            CAST(COALESCE(json_extract(payload_json, '$.contact_generation'), 0) AS INTEGER)
        FROM instance_outbox
        WHERE profile_id = ? AND instance_id = ? AND status = 'PENDING'
          AND json_valid(payload_json)
          AND (
            UPPER(COALESCE(json_extract(payload_json, '$.expression_kind'), '')) = 'IMAGE'
            OR EXISTS (
                SELECT 1 FROM json_each(payload_json, '$.components') component
                WHERE LOWER(COALESCE(json_extract(component.value, '$.type'), ''))
                    = 'image_asset'
            )
          )
          AND TRIM(COALESCE(json_extract(payload_json, '$.contact_attempt_ref'), '')) != ''
          AND CAST(COALESCE(json_extract(
              payload_json, '$.contact_generation'
          ), 0) AS INTEGER) >= 1""",
        (profile_id, instance_id),
    ).fetchall()
    return tuple((str(row[0]), int(row[1])) for row in rows)


def _has_active_contact_outbox(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    attempt_ref: str,
    generation: int,
) -> bool:
    return (
        conn.execute(
            """SELECT 1 FROM instance_outbox
            WHERE profile_id = ? AND instance_id = ?
              AND status IN ('PENDING', 'SENDING') AND json_valid(payload_json)
              AND TRIM(COALESCE(json_extract(
                  payload_json, '$.contact_attempt_ref'
              ), '')) = ?
              AND CAST(COALESCE(json_extract(
                  payload_json, '$.contact_generation'
              ), 0) AS INTEGER) = ?
            LIMIT 1""",
            (profile_id, instance_id, attempt_ref, generation),
        ).fetchone()
        is not None
    )


def _settle_one_pending_contact_attempt(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    attempt_ref: str,
    generation: int,
    reason: str,
    now: str,
) -> None:
    conn.execute(
        """UPDATE contact_attempts
        SET status = 'FINALIZED', attempted = 0, success = 0, answered = 0,
            finalized_at = ?
        WHERE profile_id = ? AND instance_id = ? AND attempt_ref = ?
          AND generation = ? AND status = 'READY'""",
        (now, profile_id, instance_id, attempt_ref, generation),
    )
    conn.execute(
        """UPDATE contact_evidence_reservations
        SET status = 'STALE', resolved_at = ?, resolution_reason = ?,
            version = version + 1
        WHERE profile_id = ? AND instance_id = ? AND attempt_ref = ?
          AND contact_generation = ? AND status = 'RESERVED'""",
        (now, reason, profile_id, instance_id, attempt_ref, generation),
    )
    conn.execute(
        """UPDATE instance_contact_state
        SET deferred_evidence_json = '{}', lease_until = NULL,
            lease_token = lease_token + 1, generation = generation + 1,
            version = version + 1, updated_at = ?
        WHERE profile_id = ? AND instance_id = ? AND generation = ?""",
        (now, profile_id, instance_id, generation),
    )


__all__ = ["InstanceChatPolicyRecords"]
