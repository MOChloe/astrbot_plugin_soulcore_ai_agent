from __future__ import annotations

from ....contracts.ai_task_payload import decode_task_payload
from ....storage.sqlite.contact_evidence_settlement import ContactEvidenceSettlement
from ..contact_models import ContactEvidenceKind, ContactOpportunity, TimelineEvidence
from .support import (
    Any,
    Mapping,
    Sequence,
    _dt,
    _load,
    _now,
    _parse,
    datetime,
    sqlite3,
    uuid,
)


def reconcile_failed_contact_task_handoffs(
    conn: sqlite3.Connection,
    *,
    limit: int,
    point: str,
) -> int:
    rows = conn.execute(
        """SELECT attempt.profile_id, attempt.instance_id,
        attempt.attempt_ref, attempt.generation, attempt.task_id
        FROM contact_attempts attempt
        JOIN ai_tasks task ON task.task_id = attempt.task_id
        WHERE attempt.status = 'READY' AND task.status = 'FAILED'
        ORDER BY task.finished_at, attempt.created_at LIMIT ?""",
        (max(1, min(int(limit), 100)),),
    ).fetchall()
    for row in rows:
        _release_failed_attempt(conn, row, point=point)
    return len(rows)


def _release_failed_attempt(conn: sqlite3.Connection, row: sqlite3.Row, *, point: str) -> None:
    ContactEvidenceSettlement(
        profile_id=str(row["profile_id"]),
        instance_id=str(row["instance_id"]),
        attempt_ref=str(row["attempt_ref"]),
        generation=int(row["generation"]),
        result="FAILED",
        target="RELEASED",
        point=point,
    )(conn)
    finalized = conn.execute(
        """UPDATE contact_attempts SET status = 'FINALIZED',
        attempted = 0, success = 0, answered = 0, finalized_at = ?
        WHERE profile_id = ? AND instance_id = ? AND attempt_ref = ?
        AND generation = ? AND task_id = ? AND status = 'READY'""",
        (
            point,
            row["profile_id"],
            row["instance_id"],
            row["attempt_ref"],
            int(row["generation"]),
            int(row["task_id"]),
        ),
    )
    if finalized.rowcount != 1:
        raise RuntimeError("contact attempt recovery settlement lost")


class ContactEvidenceRecords:
    async def list_contact_action_results(
        self,
        profile_id: str,
        instance_id: str,
        *,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT e.event_id, e.intent_id, e.to_status, e.reason,
            e.created_at, i.intent_kind, i.priority, r.summary, r.goal
            FROM character_intent_events e JOIN character_intents i
              ON i.intent_id = e.intent_id
            JOIN character_intent_revisions r ON r.intent_id = i.intent_id
              AND r.revision = i.current_revision
            WHERE e.profile_id = ? AND e.instance_id = ?
              AND i.intent_kind = 'ACTION_INTENT'
              AND e.to_status IN ('COMPLETED','CANCELLED','EXPIRED')
              AND NOT EXISTS (SELECT 1 FROM contact_evidence_reservations cer
                WHERE cer.profile_id = e.profile_id AND cer.instance_id = e.instance_id
                  AND cer.evidence_kind = 'ACTION_RESULT'
                  AND cer.evidence_ref = e.intent_id || ':' || e.event_id
                  AND cer.status IN ('RESERVED','CONSUMED','STALE'))
            ORDER BY e.event_id LIMIT ?""",
            (profile_id, instance_id, max(0, min(int(limit), 20))),
        )
        return [
            {
                **self._record(row, json_columns=()),
                "evidence_kind": "ACTION_RESULT",
                "evidence_ref": f"{row['intent_id']}:{row['event_id']}",
                "occurred_at": _parse(row["created_at"]),
                "importance": float(row["priority"]),
            }
            for row in rows
        ]

    async def reserve_contact_evidence(
        self,
        profile_id: str,
        instance_id: str,
        *,
        attempt_ref: str,
        generation: int,
        evidence: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        reference = str(attempt_ref).strip()
        if not reference:
            raise ValueError("attempt_ref cannot be empty")
        now = _dt(_now())

        def operation(conn: sqlite3.Connection) -> list[str]:
            result: list[str] = []
            for item in evidence:
                kind = str(item.get("evidence_kind") or "").upper()
                source = str(item.get("evidence_ref") or "").strip()
                if (
                    kind
                    not in {
                        "ROLE_TIMELINE_EVENT",
                        "ACTION_RESULT",
                    }
                    or not source
                ):
                    raise ValueError("invalid contact evidence reference")
                prior = conn.execute(
                    """SELECT status FROM contact_evidence_reservations
                    WHERE profile_id = ? AND instance_id = ? AND evidence_kind = ?
                      AND evidence_ref = ? AND status IN ('RESERVED','CONSUMED','STALE')
                    ORDER BY reserved_at DESC LIMIT 1""",
                    (profile_id, instance_id, kind, source),
                ).fetchone()
                if prior is not None:
                    raise ValueError("contact evidence is already reserved or consumed")
                reservation_id = f"contact-evidence:{uuid.uuid4().hex}"
                try:
                    conn.execute(
                        """INSERT INTO contact_evidence_reservations(
                            reservation_id, profile_id, instance_id, attempt_ref,
                            contact_generation, evidence_kind, evidence_ref,
                            reserved_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            reservation_id,
                            profile_id,
                            instance_id,
                            reference,
                            int(generation),
                            kind,
                            source,
                            now,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError("contact evidence is already reserved or consumed") from exc
                result.append(reservation_id)
            return result

        ids = await self.uow.run(operation)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        return [
            self._record(row, json_columns=())
            for row in await self.db.fetch_all(
                f"SELECT * FROM contact_evidence_reservations WHERE reservation_id IN ({placeholders})",
                ids,
            )
        ]

    async def list_pending_contact_task_handoffs(
        self,
        *,
        limit: int = 20,
    ) -> list[ContactOpportunity]:
        rows = await self.db.fetch_all(
            """SELECT attempt.profile_id, attempt.instance_id,
            attempt.attempt_ref, attempt.generation,
            state.deferred_evidence_json, instance.route_umo
            FROM contact_attempts attempt
            JOIN instance_contact_state state
              ON state.profile_id = attempt.profile_id
             AND state.instance_id = attempt.instance_id
            JOIN character_instances instance
              ON instance.profile_id = attempt.profile_id
             AND instance.instance_id = attempt.instance_id
            WHERE attempt.status = 'READY' AND attempt.task_id IS NULL
            ORDER BY attempt.created_at, attempt.attempt_ref LIMIT ?""",
            (max(1, min(int(limit), 100)),),
        )
        opportunities: list[ContactOpportunity] = []
        for row in rows:
            opportunity = _pending_contact_opportunity(row)
            if opportunity is not None:
                opportunities.append(opportunity)
        return opportunities

    async def reconcile_failed_contact_task_handoffs(
        self,
        *,
        limit: int = 20,
        now: datetime | None = None,
    ) -> int:
        """Release contact attempts whose bound durable task failed pre-delivery.

        Provider/input failures happen before the platform boundary.  They must
        neither count as a contact attempt nor leave the frozen evidence owned
        forever by a terminal task.
        """

        point = _dt(now or _now())
        return int(
            await self.uow.run(
                lambda conn: reconcile_failed_contact_task_handoffs(conn, limit=limit, point=point)
            )
        )

    async def publish_contact_task_handoff(
        self,
        profile_id: str,
        instance_id: str,
        *,
        attempt_ref: str,
        generation: int,
        task_id: int,
        now: datetime | None = None,
    ) -> bool:
        reference = str(attempt_ref).strip()
        if not reference or int(task_id) < 1:
            raise ValueError("contact task handoff identity is invalid")
        point = _dt(now or _now())

        def operation(conn: sqlite3.Connection) -> bool:
            attempt = conn.execute(
                """SELECT * FROM contact_attempts WHERE profile_id = ?
                AND instance_id = ? AND attempt_ref = ? AND generation = ?
                AND status = 'READY'""",
                (profile_id, instance_id, reference, int(generation)),
            ).fetchone()
            state = conn.execute(
                """SELECT deferred_evidence_json FROM instance_contact_state
                WHERE profile_id = ? AND instance_id = ?""",
                (profile_id, instance_id),
            ).fetchone()
            task = conn.execute(
                """SELECT * FROM ai_tasks WHERE task_id = ? AND profile_id = ?
                AND instance_id = ? AND task_type = 'MAIN_CORE'""",
                (int(task_id), profile_id, instance_id),
            ).fetchone()
            snapshot = _contact_handoff_snapshot(
                attempt,
                state,
                task,
                reference=reference,
                generation=int(generation),
                task_id=int(task_id),
            )
            if snapshot is None:
                return False
            evidence = _snapshot_evidence(snapshot)
            if not _reserve_handoff_evidence(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                reference=reference,
                generation=int(generation),
                evidence=evidence,
                point=point,
            ):
                return False
            _bind_and_publish_contact_task(
                conn,
                attempt=attempt,
                task=task,
                profile_id=profile_id,
                instance_id=instance_id,
                reference=reference,
                generation=int(generation),
                task_id=int(task_id),
                point=point,
            )
            return True

        return bool(await self.uow.run(operation))

    async def cancel_unpublished_contact_task(
        self,
        profile_id: str,
        instance_id: str,
        task_id: int,
        *,
        attempt_ref: str,
        generation: int,
        reason: str,
        now: datetime | None = None,
    ) -> bool:
        reference = str(attempt_ref).strip()
        if not reference or int(generation) < 1 or int(task_id) < 1:
            raise ValueError("contact task handoff identity is invalid")
        point = _dt(now or _now())

        def operation(conn: sqlite3.Connection) -> bool:
            task = conn.execute(
                """SELECT * FROM ai_tasks WHERE profile_id = ?
                AND instance_id = ? AND task_type = 'MAIN_CORE'
                AND status = 'SCHEDULED' AND idempotency_key = ?
                AND task_id = ?""",
                (profile_id, instance_id, reference, int(task_id)),
            ).fetchone()
            if task is None or not _task_matches_contact(task, reference, int(generation)):
                return False
            cancelled = conn.execute(
                """UPDATE ai_tasks SET status = 'CANCELLED', last_error = ?,
                finished_at = ?, updated_at = ?, version = version + 1
                WHERE task_id = ? AND version = ? AND status = 'SCHEDULED'""",
                (
                    str(reason or "contact_task_handoff_rejected")[:600],
                    point,
                    point,
                    int(task["task_id"]),
                    int(task["version"]),
                ),
            )
            if cancelled.rowcount != 1:
                raise RuntimeError("contact task cancellation lost")
            return True

        return bool(await self.uow.run(operation))

    async def cancel_pending_contact_task_handoff(
        self,
        profile_id: str,
        instance_id: str,
        *,
        attempt_ref: str,
        generation: int,
        reason: str,
        outcome: str,
        now: datetime | None = None,
    ) -> bool:
        reference = str(attempt_ref).strip()
        result = str(outcome).upper()
        targets = {
            "SUPERSEDED": "STALE",
            "SUPPRESSED": "RELEASED",
            "FAILED": "RELEASED",
        }
        if not reference or int(generation) < 1:
            raise ValueError("contact task handoff identity is invalid")
        if result not in targets:
            raise ValueError("unsupported pending contact handoff outcome")
        point = _dt(now or _now())

        def operation(conn: sqlite3.Connection) -> bool:
            attempt = conn.execute(
                """SELECT * FROM contact_attempts WHERE profile_id = ?
                AND instance_id = ? AND attempt_ref = ? AND generation = ?
                AND status = 'READY'""",
                (profile_id, instance_id, reference, int(generation)),
            ).fetchone()
            state = conn.execute(
                """SELECT deferred_evidence_json FROM instance_contact_state
                WHERE profile_id = ? AND instance_id = ?""",
                (profile_id, instance_id),
            ).fetchone()
            if not _pending_attempt_matches(
                attempt,
                state,
                reference,
                int(generation),
            ):
                return False
            task_lookup = _pending_contact_task(
                conn,
                attempt=attempt,
                profile_id=profile_id,
                instance_id=instance_id,
                reference=reference,
            )
            if task_lookup is None:
                return False
            task, bound_task_id = task_lookup
            if task is not None and not _task_matches_contact(task, reference, int(generation)):
                return False
            if task is not None:
                _cancel_pending_contact_task(conn, task=task, reason=reason, point=point)
            ContactEvidenceSettlement(
                profile_id=profile_id,
                instance_id=instance_id,
                attempt_ref=reference,
                generation=int(generation),
                result=result,
                target=targets[result],
                point=point,
            )(conn)
            _finalize_pending_contact_attempt(
                conn,
                profile_id=profile_id,
                instance_id=instance_id,
                reference=reference,
                generation=int(generation),
                bound_task_id=bound_task_id,
                point=point,
            )
            return True

        return bool(await self.uow.run(operation))

    async def settle_contact_evidence(
        self,
        profile_id: str,
        instance_id: str,
        *,
        attempt_ref: str,
        generation: int,
        outcome: str,
        now: datetime | None = None,
    ) -> int:
        result = str(outcome).upper()
        if result not in {
            "DELIVERED",
            "SUPERSEDED",
            "SUPPRESSED",
            "FAILED",
            "ATTEMPTED_UNKNOWN",
        }:
            raise ValueError("unsupported contact evidence outcome")
        target = {
            "DELIVERED": "CONSUMED",
            "SUPERSEDED": "STALE",
            "SUPPRESSED": "RELEASED",
            "FAILED": "RELEASED",
            "ATTEMPTED_UNKNOWN": "RELEASED",
        }[result]
        operation = ContactEvidenceSettlement(
            profile_id=profile_id,
            instance_id=instance_id,
            attempt_ref=str(attempt_ref),
            generation=int(generation),
            result=result,
            target=target,
            point=_dt(now or _now()),
        )
        return await self.uow.run(operation)

    async def claim_contact_evidence_for_foreground(
        self,
        profile_id: str,
        instance_id: str,
        *,
        activity_epoch: int,
        limit: int = 12,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Hand a fenced, not-yet-delivered contact snapshot to new foreground.

        The autonomous attempt loses ownership, reservations are released, and
        evidence is not consumed.  The caller receives only the already-bounded
        structured snapshot; no chat history or full Memory row is exposed.
        """

        point = _dt(now or _now())
        bounded = max(1, min(int(limit), 12))

        def operation(conn: sqlite3.Connection) -> dict[str, Any] | None:
            core = conn.execute(
                """SELECT activity_epoch FROM instance_core_state
                WHERE profile_id = ? AND instance_id = ?""",
                (profile_id, instance_id),
            ).fetchone()
            if core is None or int(core["activity_epoch"]) != int(activity_epoch):
                return None
            state = conn.execute(
                """SELECT deferred_evidence_json FROM instance_contact_state
                WHERE profile_id = ? AND instance_id = ?""",
                (profile_id, instance_id),
            ).fetchone()
            snapshot = _load(state["deferred_evidence_json"]) if state else {}
            if not isinstance(snapshot, dict):
                return None
            attempt_ref = str(snapshot.get("attempt_ref") or "").strip()
            generation = int(snapshot.get("generation") or 0)
            items = list(snapshot.get("items") or ())[:bounded]
            if not attempt_ref or generation < 1 or not items:
                return None
            reservations = conn.execute(
                """SELECT * FROM contact_evidence_reservations WHERE profile_id = ?
                AND instance_id = ? AND attempt_ref = ? AND contact_generation = ?
                AND status = 'RESERVED'""",
                (profile_id, instance_id, attempt_ref, generation),
            ).fetchall()
            for reservation in reservations:
                conn.execute(
                    """UPDATE contact_evidence_reservations SET status = 'RELEASED',
                    resolved_at = ?, resolution_reason = 'foreground_handoff',
                    version = version + 1 WHERE reservation_id = ?
                    AND status = 'RESERVED'""",
                    (point, reservation["reservation_id"]),
                )
            cursor = conn.execute(
                """UPDATE instance_contact_state SET deferred_evidence_json = '{}',
                last_result = 'FOREGROUND_HANDOFF',
                last_reason = 'new_player_activity', version = version + 1,
                updated_at = ? WHERE profile_id = ? AND instance_id = ?
                AND deferred_evidence_json = ?""",
                (point, profile_id, instance_id, state["deferred_evidence_json"]),
            )
            if cursor.rowcount != 1:
                return None
            return {
                "attempt_ref": attempt_ref,
                "generation": generation,
                "activity_epoch": int(activity_epoch),
                "items": items,
            }

        return await self.uow.run(operation)


def _contact_handoff_snapshot(
    attempt: Any,
    state: Any,
    task: Any,
    *,
    reference: str,
    generation: int,
    task_id: int,
) -> Mapping[str, Any] | None:
    if any(value is None for value in (attempt, state, task)):
        return None
    if attempt["task_id"] not in (None, task_id):
        return None
    snapshot = _load(state["deferred_evidence_json"])
    if not _snapshot_matches(snapshot, reference, generation):
        return None
    task_input = decode_task_payload("input", task["input_json"])
    metadata = task_input.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if (
        str(metadata.get("contact_attempt_ref") or "") != reference
        or int(metadata.get("contact_generation") or 0) != generation
        or str(task["status"]) not in {"SCHEDULED", "READY"}
    ):
        return None
    assert isinstance(snapshot, Mapping)
    return snapshot


def _reserve_handoff_evidence(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    reference: str,
    generation: int,
    evidence: Sequence[tuple[str, str]],
    point: str,
) -> bool:
    existing: set[tuple[str, str]] = set()
    for kind, source in evidence:
        prior = conn.execute(
            """SELECT attempt_ref, contact_generation, status
            FROM contact_evidence_reservations WHERE profile_id = ?
            AND instance_id = ? AND evidence_kind = ? AND evidence_ref = ?
            AND status IN ('RESERVED','CONSUMED','STALE')
            ORDER BY reserved_at DESC LIMIT 1""",
            (profile_id, instance_id, kind, source),
        ).fetchone()
        if prior is None:
            continue
        if (
            str(prior["attempt_ref"]) != reference
            or int(prior["contact_generation"]) != generation
            or str(prior["status"]) != "RESERVED"
        ):
            return False
        existing.add((kind, source))
    for kind, source in evidence:
        if (kind, source) in existing:
            continue
        conn.execute(
            """INSERT INTO contact_evidence_reservations(
            reservation_id, profile_id, instance_id, attempt_ref,
            contact_generation, evidence_kind, evidence_ref, reserved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"contact-evidence:{uuid.uuid4().hex}",
                profile_id,
                instance_id,
                reference,
                generation,
                kind,
                source,
                point,
            ),
        )
    return True


def _bind_and_publish_contact_task(
    conn: sqlite3.Connection,
    *,
    attempt: Any,
    task: Any,
    profile_id: str,
    instance_id: str,
    reference: str,
    generation: int,
    task_id: int,
    point: str,
) -> None:
    if attempt["task_id"] != task_id:
        bound = conn.execute(
            """UPDATE contact_attempts SET task_id = ?
            WHERE profile_id = ? AND instance_id = ? AND attempt_ref = ?
            AND generation = ? AND status = 'READY'
            AND task_id IS NULL""",
            (task_id, profile_id, instance_id, reference, generation),
        )
        if bound.rowcount != 1:
            raise RuntimeError("contact task binding lost")
    if str(task["status"]) == "SCHEDULED":
        published = conn.execute(
            """UPDATE ai_tasks SET status = 'READY', due_at = ?,
            updated_at = ?, version = version + 1
            WHERE task_id = ? AND status = 'SCHEDULED'""",
            (point, point, task_id),
        )
        if published.rowcount != 1:
            raise RuntimeError("contact task publication lost")


def _pending_attempt_matches(
    attempt: Any,
    state: Any,
    reference: str,
    generation: int,
) -> bool:
    return bool(
        attempt is not None
        and state is not None
        and _snapshot_matches(
            _load(state["deferred_evidence_json"]),
            reference,
            generation,
        )
    )


def _pending_contact_task(
    conn: sqlite3.Connection,
    *,
    attempt: Any,
    profile_id: str,
    instance_id: str,
    reference: str,
) -> tuple[Any | None, int | None] | None:
    bound_task_id = attempt["task_id"]
    if bound_task_id is not None:
        task = conn.execute(
            """SELECT * FROM ai_tasks WHERE task_id = ? AND profile_id = ?
            AND instance_id = ? AND task_type = 'MAIN_CORE'
            AND idempotency_key = ?""",
            (int(bound_task_id), profile_id, instance_id, reference),
        ).fetchone()
        if task is None or str(task["status"]) not in {"SCHEDULED", "READY"}:
            return None
        return task, int(bound_task_id)
    candidates = conn.execute(
        """SELECT * FROM ai_tasks WHERE profile_id = ? AND instance_id = ?
        AND task_type = 'MAIN_CORE' AND idempotency_key = ?
        AND status IN ('SCHEDULED','READY')""",
        (profile_id, instance_id, reference),
    ).fetchall()
    if len(candidates) > 1:
        return None
    task = candidates[0] if candidates else None
    if task is not None and str(task["status"]) != "SCHEDULED":
        return None
    return task, None


def _cancel_pending_contact_task(
    conn: sqlite3.Connection,
    *,
    task: Any,
    reason: str,
    point: str,
) -> None:
    task_status = str(task["status"])
    cancelled = conn.execute(
        """UPDATE ai_tasks SET status = 'CANCELLED', last_error = ?,
        finished_at = ?, updated_at = ?, version = version + 1
        WHERE task_id = ? AND version = ? AND status = ?""",
        (
            str(reason or "contact_task_handoff_cancelled")[:600],
            point,
            point,
            int(task["task_id"]),
            int(task["version"]),
            task_status,
        ),
    )
    if cancelled.rowcount != 1:
        raise RuntimeError("contact task cancellation lost")


def _finalize_pending_contact_attempt(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    reference: str,
    generation: int,
    bound_task_id: int | None,
    point: str,
) -> None:
    task_clause = "AND task_id IS NULL"
    task_params: tuple[int, ...] = ()
    if bound_task_id is not None:
        task_clause = "AND task_id = ?"
        task_params = (int(bound_task_id),)
    finalized = conn.execute(
        f"""UPDATE contact_attempts SET status = 'FINALIZED',
        attempted = 0, success = 0, answered = 0, finalized_at = ?
        WHERE profile_id = ? AND instance_id = ? AND attempt_ref = ?
        AND generation = ? AND status = 'READY' {task_clause}""",
        (
            point,
            profile_id,
            instance_id,
            reference,
            generation,
            *task_params,
        ),
    )
    if finalized.rowcount != 1:
        raise RuntimeError("contact attempt settlement lost")


def _snapshot_matches(value: Any, attempt_ref: str, generation: int) -> bool:
    return bool(
        isinstance(value, Mapping)
        and str(value.get("attempt_ref") or "") == attempt_ref
        and int(value.get("generation") or 0) == generation
    )


def _task_matches_contact(
    task: Mapping[str, Any],
    attempt_ref: str,
    generation: int,
) -> bool:
    task_input = decode_task_payload("input", task["input_json"])
    metadata = task_input.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return bool(
        str(metadata.get("contact_attempt_ref") or "") == attempt_ref
        and int(metadata.get("contact_generation") or 0) == generation
    )


def _snapshot_evidence(value: Mapping[str, Any]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    allowed = {kind.value for kind in ContactEvidenceKind}
    for item in list(value.get("items") or ())[:12]:
        if not isinstance(item, Mapping):
            raise ValueError("contact evidence snapshot item is invalid")
        kind = str(item.get("evidence_kind") or "").upper()
        source = str(item.get("evidence_id") or "").strip()
        if kind not in allowed or not source:
            raise ValueError("contact evidence snapshot reference is invalid")
        result.append((kind, source))
    if len(result) != len(set(result)):
        raise ValueError("contact evidence snapshot contains duplicates")
    return result


def _pending_contact_opportunity(row: Mapping[str, Any]) -> ContactOpportunity | None:
    snapshot = _load(row["deferred_evidence_json"])
    attempt_ref = str(row["attempt_ref"])
    generation = int(row["generation"])
    if not _snapshot_matches(snapshot, attempt_ref, generation):
        return None
    assert isinstance(snapshot, Mapping)
    evidence = tuple(_timeline_evidence(item) for item in list(snapshot.get("items") or ())[:12])
    proactive_frame_planned_at = _parse(snapshot.get("proactive_frame_planned_at"))
    return ContactOpportunity(
        profile_id=str(row["profile_id"]),
        instance_id=str(row["instance_id"]),
        generation=generation,
        activity_epoch=int(snapshot.get("activity_epoch") or 0),
        state_epoch=int(snapshot.get("state_epoch") or 0),
        route_umo=str(snapshot.get("route_umo") or row["route_umo"] or ""),
        evidence=evidence,
        attempt_ref=attempt_ref,
        failure_mode=str(snapshot.get("failure_mode") or "SKIP"),
        reroll_count=int(snapshot.get("reroll_count") or 0),
        proactive_frame_planned_at=proactive_frame_planned_at,
        proactive_frame_source_ref=str(snapshot.get("proactive_frame_source_ref") or ""),
    )


def _timeline_evidence(item: Any) -> TimelineEvidence:
    if not isinstance(item, Mapping):
        raise ValueError("contact evidence snapshot item is invalid")
    occurred_at = _parse(item.get("occurred_at"))
    if occurred_at is None:
        raise ValueError("contact evidence snapshot time is invalid")
    return TimelineEvidence(
        evidence_id=str(item.get("evidence_id") or ""),
        evidence_kind=ContactEvidenceKind(str(item.get("evidence_kind") or "").upper()),
        summary=str(item.get("summary") or ""),
        occurred_at=occurred_at,
        important=bool(item.get("important")),
        importance=float(item.get("importance") or 0.0),
        reason=str(item.get("reason") or ""),
    )
