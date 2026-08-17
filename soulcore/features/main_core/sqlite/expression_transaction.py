"""Atomic persistence for ordered visible and retraction expression steps."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from .support import _dt, _parse, sqlite3


class ExpressionTransactionMixin:
    context: Any

    def _create_expression_batch(self, conn: sqlite3.Connection) -> None:
        batch = self.context.expression_batch
        if not batch:
            return
        batch_id = str(batch.get("batch_id") or "").strip()
        output_count = int(batch.get("output_count") or 0)
        steps = self._expression_steps(batch)
        self._validate_expression_batch_shape(conn, batch_id, output_count, steps)
        self._validate_expression_delays(conn, steps)
        self._validate_expression_actions(batch_id, steps)
        self._expression_delay_anchor()
        self._insert_expression_batch(conn, batch_id, output_count, batch)

    def _validate_expression_batch_shape(
        self,
        conn: sqlite3.Connection,
        batch_id: str,
        output_count: int,
        steps: list[Any],
    ) -> None:
        context = self.context
        segment_index = (context.expression_batch or {}).get("segment_index")
        if isinstance(segment_index, bool) or not isinstance(segment_index, int):
            raise ValueError("expression segment index must be an integer")
        if segment_index < 0:
            raise ValueError("expression segment index cannot be negative")
        if batch_id != f"expression-run:{context.run_id}:segment:{segment_index}":
            raise ValueError("expression batch id must be derived from its run and segment")
        retractions = sum(1 for step in steps if _step_kind(step) == "RETRACT")
        if output_count < 0:
            raise ValueError("expression batch visible-step count cannot be negative")
        previous_retractions = int(
            conn.execute(
                """SELECT COALESCE(SUM(retraction_count), 0)
                FROM instance_expression_batches
                WHERE profile_id = ? AND instance_id = ? AND source_run_id = ?
                  AND segment_index < ?""",
                (
                    context.profile_id,
                    context.instance_id,
                    context.run_id,
                    segment_index,
                ),
            ).fetchone()[0]
        )
        if not 0 <= previous_retractions + retractions <= 6:
            raise ValueError("expression batch supports at most six retractions")
        if not steps or output_count + retractions != len(steps):
            raise ValueError("expression batch requires an ordered expression timeline")
        if len(context.outbound_actions) != output_count:
            raise ValueError("expression batch visible-step count must match outbound actions")

    def _validate_expression_delays(
        self,
        conn: sqlite3.Connection,
        steps: list[Any],
    ) -> None:
        delays = [int(step.get("delay_after_previous_seconds") or 0) for step in steps]
        if any(not 0 <= delay <= 120 for delay in delays):
            raise ValueError("expression step delay must be between zero and 120")
        context = self.context
        segment_index = int((context.expression_batch or {}).get("segment_index") or 0)
        previous_delay = conn.execute(
            """SELECT COALESCE(SUM(delay_seconds), 0) FROM (
                SELECT CAST(
                    COALESCE(json_extract(item.payload_json,
                        '$.delay_after_previous_seconds'), 0) AS INTEGER
                ) AS delay_seconds
                FROM instance_expression_batches batch
                JOIN instance_outbox item ON item.expression_batch_id = batch.batch_id
                WHERE batch.profile_id = ? AND batch.instance_id = ?
                  AND batch.source_run_id = ? AND batch.segment_index < ?
                UNION ALL
                SELECT action.delay_after_previous_seconds AS delay_seconds
                FROM instance_expression_batches batch
                JOIN message_retraction_actions action
                  ON action.expression_batch_id = batch.batch_id
                WHERE batch.profile_id = ? AND batch.instance_id = ?
                  AND batch.source_run_id = ? AND batch.segment_index < ?
            )""",
            (
                context.profile_id,
                context.instance_id,
                context.run_id,
                segment_index,
                context.profile_id,
                context.instance_id,
                context.run_id,
                segment_index,
            ),
        ).fetchone()
        total_delay = int(previous_delay[0] if previous_delay is not None else 0) + sum(delays)
        if total_delay > 300:
            raise ValueError("expression run may span at most 300 seconds")

    def _validate_expression_actions(self, batch_id: str, steps: list[Any]) -> None:
        previous_key: str | None = None
        visible_steps = [index for index, step in enumerate(steps) if _step_kind(step) != "RETRACT"]
        for ordinal, raw in enumerate(self.context.outbound_actions):
            self._validate_expression_action(raw, batch_id, ordinal, visible_steps[ordinal])
            dependency = str(raw.get("depends_on_idempotency_key") or "").strip() or None
            if dependency != previous_key:
                raise ValueError("expression outbound dependency must reference the preceding item")
            previous_key = str(
                raw.get("idempotency_key")
                or f"instance-run:{self.context.run_id}:outbound:{ordinal}"
            )

    @staticmethod
    def _validate_expression_action(
        raw: Any, batch_id: str, ordinal: int, step_ordinal: int
    ) -> None:
        if not isinstance(raw, dict):
            raise ValueError("expression batch actions must use structured outbound records")
        if str(raw.get("expression_batch_id") or "") != batch_id:
            raise ValueError("expression outbound batch id mismatch")
        if int(raw.get("expression_ordinal", -1)) != ordinal:
            raise ValueError("expression outbound ordinals must be contiguous from zero")
        actual = int(raw.get("expression_step_ordinal", -1))
        if actual != step_ordinal:
            raise ValueError("expression outbound step ordinal must match the full timeline")

    def _create_expression_timeline(self, conn: sqlite3.Connection) -> None:
        steps = self._expression_steps(self.context.expression_batch or {})
        actions = {
            int(raw.get("expression_step_ordinal", -1)): raw
            for raw in self.context.outbound_actions
            if isinstance(raw, dict)
        }
        cumulative_delay = 0
        visible_index = 0
        for step_ordinal, step in enumerate(steps):
            cumulative_delay += int(step.get("delay_after_previous_seconds") or 0)
            if _step_kind(step) == "RETRACT":
                self._create_retraction(conn, step_ordinal, step, cumulative_delay)
            else:
                raw = actions.get(step_ordinal)
                if raw is None:
                    raise ValueError("visible expression step has no matching outbound action")
                self._create_outbound(conn, visible_index, raw)
                visible_index += 1

    def _expression_steps(self, batch: dict[str, Any]) -> list[dict[str, Any]]:
        explicit = list(batch.get("steps") or [])
        if not explicit:
            raise ValueError("expression batch requires an explicit expression timeline")
        return [dict(item) for item in explicit]

    def _create_retraction(
        self,
        conn: sqlite3.Connection,
        step_ordinal: int,
        step: dict[str, Any],
        cumulative_delay: int,
    ) -> None:
        target_ref = str(step.get("target_message_ref") or "").strip() or None
        if target_ref is not None:
            self._assert_retractable_target(conn, target_ref)
        else:
            self._assert_retractable_output(
                conn,
                int(step.get("target_output_ordinal")),
                step_ordinal,
            )
        not_before = self._expression_not_before(cumulative_delay)
        conn.execute(
            """INSERT INTO message_retraction_actions(
                profile_id, instance_id, source_run_id, expression_batch_id,
                step_ordinal, target_message_ref, target_output_ordinal,
                delay_after_previous_seconds, not_before_at, status,
                idempotency_key, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?)""",
            self._retraction_values(step_ordinal, step, target_ref, not_before),
        )

    def _assert_retractable_target(self, conn: sqlite3.Connection, target_ref: str) -> None:
        context = self.context
        target = conn.execute(
            """SELECT direction, self_retraction_supported, retractable_until,
                retraction_status, platform_message_id
            FROM instance_message_fragments
            WHERE profile_id = ? AND instance_id = ? AND message_ref = ?""",
            (context.profile_id, context.instance_id, target_ref),
        ).fetchone()
        if not _target_is_retractable(target, context.now):
            raise ValueError("target message fragment is no longer retractable")

    def _assert_retractable_output(
        self,
        conn: sqlite3.Connection,
        target_output_ordinal: int,
        step_ordinal: int,
    ) -> None:
        context = self.context
        current_segment = int((context.expression_batch or {}).get("segment_index") or 0)
        batches = conn.execute(
            """SELECT batch_id, segment_index, output_count
            FROM instance_expression_batches
            WHERE profile_id = ? AND instance_id = ? AND source_run_id = ?
            ORDER BY segment_index""",
            (context.profile_id, context.instance_id, context.run_id),
        ).fetchall()
        consumed = 0
        for batch in batches:
            output_count = int(batch["output_count"] or 0)
            if target_output_ordinal > consumed + output_count:
                consumed += output_count
                continue
            local_ordinal = target_output_ordinal - consumed - 1
            target = conn.execute(
                """SELECT expression_step_ordinal FROM instance_outbox
                WHERE profile_id = ? AND instance_id = ? AND expression_batch_id = ?
                  AND expression_ordinal = ?""",
                (
                    context.profile_id,
                    context.instance_id,
                    str(batch["batch_id"]),
                    local_ordinal,
                ),
            ).fetchone()
            if target is None:
                raise ValueError("retraction target output is unavailable")
            target_segment = int(batch["segment_index"])
            if target_segment > current_segment or (
                target_segment == current_segment
                and int(target["expression_step_ordinal"]) >= int(step_ordinal)
            ):
                raise ValueError("retraction target must be an earlier visible output")
            return
        raise ValueError("retraction target output is unavailable")

    def _retraction_values(
        self,
        step_ordinal: int,
        step: dict[str, Any],
        target_ref: str | None,
        not_before: str | None,
    ) -> tuple[Any, ...]:
        context = self.context
        target_ordinal = step.get("target_output_ordinal")
        batch_id = str((context.expression_batch or {}).get("batch_id") or "")
        return (
            context.profile_id,
            context.instance_id,
            context.run_id,
            batch_id,
            step_ordinal,
            target_ref,
            int(target_ordinal) if target_ordinal is not None else None,
            int(step.get("delay_after_previous_seconds") or 0),
            not_before,
            (
                f"core-run:{context.run_id}:segment:"
                f"{int((context.expression_batch or {}).get('segment_index') or 0)}:"
                f"retract:{step_ordinal}"
            ),
            context.now,
            context.now,
        )

    def _insert_expression_batch(
        self, conn: sqlite3.Connection, batch_id: str, output_count: int, batch: dict[str, Any]
    ) -> None:
        context = self.context
        segment_index = int(batch.get("segment_index") or 0)
        conn.execute(
            """INSERT INTO instance_expression_batches(
                batch_id, profile_id, instance_id, source_run_id, activity_epoch,
                segment_index, route_umo, status, output_count, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?)
            ON CONFLICT(profile_id, instance_id, source_run_id, segment_index) DO NOTHING""",
            (
                batch_id,
                context.profile_id,
                context.instance_id,
                context.run_id,
                context.expected_activity_epoch,
                segment_index,
                str(batch.get("route_umo") or context.instance.route_umo),
                output_count,
                context.now,
                context.now,
            ),
        )

    def _expression_not_before(self, delay_seconds: int) -> str | None:
        delay = max(0, int(delay_seconds))
        if delay == 0:
            return None
        return _dt(self._expression_delay_anchor() + timedelta(seconds=delay))

    def _expression_delay_anchor(self) -> datetime:
        context = self.context
        committed_at = _parse(context.now)
        if committed_at is None or committed_at.tzinfo is None:
            raise ValueError("expression commit time must include an explicit timezone offset")
        raw = (context.expression_batch or {}).get("delay_anchor_at")
        if raw is None or raw == "":
            return committed_at
        if not isinstance(raw, str):
            raise ValueError("expression delay anchor must be an ISO-8601 datetime string")
        try:
            anchor = _parse(raw)
        except ValueError as exc:
            raise ValueError("expression delay anchor must be a valid ISO-8601 datetime") from exc
        if anchor is None or anchor.tzinfo is None:
            raise ValueError("expression delay anchor must include an explicit timezone offset")
        return min(anchor, committed_at)


def _step_kind(step: Any) -> str:
    return str(step.get("kind") or "").upper()


def _target_is_retractable(target: Any, now: str) -> bool:
    if target is None or target["direction"] != "OUTBOUND":
        return False
    if not bool(target["self_retraction_supported"]):
        return False
    if not str(target["platform_message_id"] or "").strip():
        return False
    if target["retraction_status"] in {"PENDING", "SENDING", "RETRACTED"}:
        return False
    return target["retractable_until"] is None or target["retractable_until"] > now
