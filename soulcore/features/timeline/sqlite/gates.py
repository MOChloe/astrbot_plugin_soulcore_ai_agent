from __future__ import annotations

from .deferred_batch_transactions import (
    DeferredBatchAppendContext,
    DeferredBatchAppendTransaction,
    DeferredBatchClaimContext,
    DeferredBatchClaimTransaction,
)
from .support import (
    STATE_GATE_POLICY_FIELDS,
    Any,
    Mapping,
    _dt,
    _now,
    datetime,
    sqlite3,
    timedelta,
    uuid,
)
from .temporary_absence_expiry import TemporaryAbsenceExpiryRecords


class StateGateRecords(TemporaryAbsenceExpiryRecords):
    async def get_state_gate_policy(
        self,
        profile_id: str,
        scope: str,
    ) -> dict[str, Any] | None:
        self._validate_scope(scope)
        row = await self.db.fetch_one(
            """SELECT * FROM scope_state_gate_policies
            WHERE profile_id = ? AND scope = ?""",
            (profile_id, scope),
        )
        return self._record(row, json_columns=()) if row else None

    async def update_state_gate_policy(
        self,
        profile_id: str,
        scope: str,
        patch: Mapping[str, Any],
        *,
        expected_version: int,
    ) -> dict[str, Any] | None:
        self._validate_scope(scope)
        unknown = set(patch) - set(STATE_GATE_POLICY_FIELDS)
        if unknown:
            raise ValueError(f"unsupported state gate policy fields: {sorted(unknown)}")
        if "max_gate_hours" in patch and not 1 <= int(patch["max_gate_hours"]) <= 24:
            raise ValueError("max_gate_hours must be between 1 and 24")
        assignments = [f"{key} = ?" for key in patch]
        if assignments:
            values = [
                int(bool(value)) if key in {"enabled", "silent_enabled"} else int(value)
                for key, value in patch.items()
            ]
            cursor = await self.db.call(
                lambda conn: conn.execute(
                    f"UPDATE scope_state_gate_policies SET {', '.join(assignments)}, "
                    "version = version + 1, updated_at = ? WHERE profile_id = ? "
                    "AND scope = ? AND version = ?",
                    (*values, _dt(_now()), profile_id, scope, int(expected_version)),
                ),
                transaction=True,
            )
            if cursor.rowcount != 1:
                return None
        return await self.get_state_gate_policy(profile_id, scope)

    async def get_instance_state_gate_override(
        self,
        profile_id: str,
        instance_id: str,
    ) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            """SELECT * FROM instance_state_gate_overrides
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        )
        return self._record(row, json_columns=()) if row else None

    async def upsert_instance_state_gate_override(
        self,
        profile_id: str,
        instance_id: str,
        patch: Mapping[str, Any],
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any] | None:
        unknown = set(patch) - set(STATE_GATE_POLICY_FIELDS)
        if unknown:
            raise ValueError(f"unsupported state gate override fields: {sorted(unknown)}")
        if patch.get("max_gate_hours") is not None and not 1 <= int(patch["max_gate_hours"]) <= 24:
            raise ValueError("max_gate_hours must be between 1 and 24")
        current = await self.get_instance_state_gate_override(profile_id, instance_id)
        now = _dt(_now())
        values = {
            key: (None if value is None else int(bool(value)))
            if key in {"enabled", "silent_enabled"}
            else value
            for key, value in patch.items()
        }
        if current is None:
            if expected_version not in (None, 0):
                return None
            columns = ["profile_id", "instance_id", *values, "created_at", "updated_at"]
            try:
                await self.db.call(
                    lambda conn: conn.execute(
                        f"INSERT INTO instance_state_gate_overrides({', '.join(columns)}) "
                        f"VALUES ({','.join('?' for _ in columns)})",
                        (profile_id, instance_id, *values.values(), now, now),
                    ),
                    transaction=True,
                )
            except sqlite3.IntegrityError:
                return None
        elif values:
            version = int(current["version"]) if expected_version is None else int(expected_version)
            cursor = await self.db.call(
                lambda conn: conn.execute(
                    f"UPDATE instance_state_gate_overrides SET "
                    f"{', '.join(f'{key} = ?' for key in values)}, "
                    "version = version + 1, updated_at = ? WHERE profile_id = ? "
                    "AND instance_id = ? AND version = ?",
                    (*values.values(), now, profile_id, instance_id, version),
                ),
                transaction=True,
            )
            if cursor.rowcount != 1:
                return None
        return await self.get_instance_state_gate_override(profile_id, instance_id)

    async def delete_instance_state_gate_override(
        self,
        profile_id: str,
        instance_id: str,
        *,
        expected_version: int,
    ) -> bool:
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """DELETE FROM instance_state_gate_overrides WHERE profile_id = ?
                AND instance_id = ? AND version = ?""",
                (profile_id, instance_id, int(expected_version)),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1

    async def resolve_state_gate_policy(
        self,
        profile_id: str,
        instance_id: str,
    ) -> dict[str, Any]:
        instance = await self.db.fetch_one(
            """SELECT scope FROM character_instances
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        )
        if instance is None:
            raise KeyError((profile_id, instance_id))
        base = await self.get_state_gate_policy(profile_id, str(instance["scope"]))
        if base is None:
            raise KeyError((profile_id, instance["scope"]))
        override = await self.get_instance_state_gate_override(profile_id, instance_id)
        result = {key: base[key] for key in STATE_GATE_POLICY_FIELDS}
        sources = dict.fromkeys(STATE_GATE_POLICY_FIELDS, "scope")
        if override:
            for key in STATE_GATE_POLICY_FIELDS:
                if override.get(key) is not None:
                    result[key], sources[key] = override[key], "instance"
        result.update(
            {
                "profile_id": profile_id,
                "instance_id": instance_id,
                "scope": str(instance["scope"]),
                "sources": sources,
            }
        )
        return result

    async def get_state_gate_snapshot(
        self,
        profile_id: str,
        instance_id: str,
    ) -> dict[str, Any]:
        row = await self.db.fetch_one(
            """SELECT * FROM instance_state_gate_snapshots
            WHERE profile_id = ? AND instance_id = ?""",
            (profile_id, instance_id),
        )
        if row is None:
            raise KeyError((profile_id, instance_id))
        return self._record(row, json_columns=())

    async def create_or_append_deferred_message_batch(
        self,
        profile_id: str,
        instance_id: str,
        *,
        message_id: int,
        due_at: datetime,
        activity_epoch: int,
        gate_generation: int,
        creation_key: str,
        batch_id: str | None = None,
        message_ref: str | None = None,
        idempotency_key: str | None = None,
        received_at: datetime | None = None,
    ) -> dict[str, Any]:
        normalized_key = str(creation_key).strip()
        if not normalized_key:
            raise ValueError("deferred batch creation_key cannot be empty")
        context = DeferredBatchAppendContext(
            profile_id=profile_id,
            instance_id=instance_id,
            message_id=int(message_id),
            due_at=due_at,
            activity_epoch=int(activity_epoch),
            gate_generation=int(gate_generation),
            creation_key=normalized_key,
            identifier=str(batch_id or f"defer:{uuid.uuid4().hex}"),
            message_ref=message_ref,
            idempotency_key=idempotency_key,
            received_at=received_at,
            now=_dt(_now()),
        )
        selected = await self.uow.run(DeferredBatchAppendTransaction(context))
        result = await self.get_deferred_message_batch(profile_id, instance_id, selected)
        assert result is not None
        return result

    async def get_deferred_message_batch(
        self,
        profile_id: str,
        instance_id: str,
        batch_id: str,
    ) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            """SELECT * FROM deferred_message_batches WHERE profile_id = ?
            AND instance_id = ? AND batch_id = ?""",
            (profile_id, instance_id, batch_id),
        )
        if row is None:
            return None
        result = self._record(row, json_columns=())
        result["items"] = [
            self._record(item, json_columns=())
            for item in await self.db.fetch_all(
                """SELECT * FROM deferred_message_items WHERE batch_id = ?
            ORDER BY message_id""",
                (batch_id,),
            )
        ]
        return result

    async def claim_due_deferred_message_batches(
        self,
        *,
        now: datetime | None = None,
        limit: int = 10,
        lease_seconds: int = 120,
        profile_id: str | None = None,
        instance_id: str | None = None,
        include_policy_disabled: bool = False,
    ) -> list[dict[str, Any]]:
        current = now or _now()
        context = DeferredBatchClaimContext(
            now_text=_dt(current),
            orphan_cutoff=_dt(current - timedelta(minutes=5)),
            lease_until=_dt(current + timedelta(seconds=max(1, lease_seconds))),
            limit=max(0, int(limit)),
            profile_id=profile_id,
            instance_id=instance_id,
            include_policy_disabled=bool(include_policy_disabled),
        )
        ids = await self.uow.run(DeferredBatchClaimTransaction(context))
        result: list[dict[str, Any]] = []
        for identifier in ids:
            row = await self.db.fetch_one(
                """SELECT profile_id, instance_id FROM deferred_message_batches
                WHERE batch_id = ?""",
                (identifier,),
            )
            if row is None:
                continue
            item = await self.get_deferred_message_batch(
                row["profile_id"], row["instance_id"], identifier
            )
            if item is not None:
                result.append(item)
        return result

    @staticmethod
    def _deferred_gate_projection(row: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(row)
        result["batch_ref"] = str(result.get("batch_id") or "")
        projected: list[dict[str, Any]] = []
        for item in result.get("items", ()) or ():
            message = dict(item)
            message["ledger_entry_id"] = int(message.get("message_id") or 0)
            projected.append(message)
        result["messages"] = projected
        return result

    async def get_state_message_gate_policy(
        self,
        profile_id: str,
        instance_id: str,
    ) -> dict[str, Any]:
        policy = await self.resolve_state_gate_policy(profile_id, instance_id)
        return {
            **policy,
            "state_message_gate_enabled": bool(policy["enabled"]),
            "state_message_silent_enabled": bool(policy["silent_enabled"]),
            "max_non_open_hours": int(policy["max_gate_hours"]),
        }

    async def get_state_message_gate_snapshot(
        self,
        profile_id: str,
        instance_id: str,
    ) -> dict[str, Any]:
        snapshot = await self.get_state_gate_snapshot(profile_id, instance_id)
        return {
            **snapshot,
            "mode": snapshot["action"],
            "effective_at": snapshot["not_before_at"],
            "expires_at": snapshot["until_at"],
        }

    async def append_or_merge_deferred_gate_message(
        self,
        profile_id: str,
        instance_id: str,
        *,
        message_ref: str,
        ledger_entry_id: int,
        idempotency_key: str,
        gate_generation: int,
        activity_epoch: int,
        received_at: datetime,
        due_at: datetime,
    ) -> dict[str, Any]:
        batch = await self.create_or_append_deferred_message_batch(
            profile_id,
            instance_id,
            message_id=int(ledger_entry_id),
            due_at=due_at,
            activity_epoch=int(activity_epoch),
            gate_generation=int(gate_generation),
            creation_key=f"state-gate:{int(gate_generation)}",
            message_ref=str(message_ref),
            idempotency_key=str(idempotency_key),
            received_at=received_at,
        )
        return self._deferred_gate_projection(batch)

    async def claim_due_deferred_gate_batches(
        self,
        *,
        now: datetime,
        limit: int = 10,
        lease_seconds: int = 120,
    ) -> list[dict[str, Any]]:
        rows = await self.claim_due_deferred_message_batches(
            now=now,
            limit=limit,
            lease_seconds=lease_seconds,
            include_policy_disabled=True,
        )
        return [self._deferred_gate_projection(row) for row in rows]

    async def claim_deferred_gate_batch_for_foreground(
        self,
        profile_id: str,
        instance_id: str,
        *,
        expected_activity_epoch: int,
        now: datetime,
        lease_seconds: int = 120,
    ) -> dict[str, Any] | None:
        point = _dt(now)
        lease = _dt(now + timedelta(seconds=max(1, int(lease_seconds))))

        def operation(conn: sqlite3.Connection) -> str | None:
            return self._claim_deferred_batch_in_transaction(
                conn,
                profile_id,
                instance_id,
                expected_activity_epoch=int(expected_activity_epoch),
                point=point,
                lease=lease,
                merge_reason="foreground-batch-coalesced",
            )

        batch_id = await self.uow.run(operation)
        if batch_id is None:
            return None
        batch = await self.get_deferred_message_batch(profile_id, instance_id, batch_id)
        assert batch is not None
        return self._deferred_gate_projection(batch)

    @staticmethod
    def _claim_deferred_batch_in_transaction(
        conn: sqlite3.Connection,
        profile_id: str,
        instance_id: str,
        *,
        expected_activity_epoch: int,
        point: str,
        lease: str,
        merge_reason: str,
        required_gate_generation: int | None = None,
    ) -> str | None:
        rows = conn.execute(
            """SELECT * FROM deferred_message_batches WHERE profile_id = ?
            AND instance_id = ? AND status IN ('PENDING', 'CLAIMED')
            ORDER BY created_at, batch_id""",
            (profile_id, instance_id),
        ).fetchall()
        if required_gate_generation is not None:
            rows = [
                row for row in rows if int(row["gate_generation"]) == int(required_gate_generation)
            ]
        if not rows:
            return None
        target = rows[0]
        # A newer scene may pre-empt an unexpired scheduler lease. Incrementing
        # both fences makes the former owner unable to resolve or release it.
        conn.execute(
            """UPDATE deferred_message_batches SET status = 'PENDING',
            lease_until = NULL, lease_token = lease_token + 1,
            version = version + 1, updated_at = ? WHERE batch_id = ?
            AND status = 'CLAIMED'""",
            (point, target["batch_id"]),
        )
        for extra in rows[1:]:
            conn.execute(
                """UPDATE deferred_message_batches SET status = 'PENDING',
                lease_until = NULL, lease_token = lease_token + 1,
                version = version + 1, updated_at = ? WHERE batch_id = ?
                AND status = 'CLAIMED'""",
                (point, extra["batch_id"]),
            )
            conn.execute(
                """UPDATE deferred_message_items SET batch_id = ?
                WHERE batch_id = ?""",
                (target["batch_id"], extra["batch_id"]),
            )
            conn.execute(
                """UPDATE deferred_message_batches SET status = 'MERGED',
                resolution_reason = ?, resolved_at = ?, version = version + 1,
                updated_at = ? WHERE batch_id = ? AND status = 'PENDING'""",
                (merge_reason, point, point, extra["batch_id"]),
            )
        cursor = conn.execute(
            """UPDATE deferred_message_batches SET status = 'CLAIMED',
            activity_epoch = ?, lease_until = ?, lease_token = lease_token + 1,
            version = version + 1, updated_at = ? WHERE batch_id = ?
            AND status = 'PENDING'""",
            (int(expected_activity_epoch), lease, point, target["batch_id"]),
        )
        return str(target["batch_id"]) if cursor.rowcount else None

    async def interrupt_temporary_absence_for_timer(
        self,
        profile_id: str,
        instance_id: str,
        *,
        now: datetime,
        lease_seconds: int = 120,
    ) -> dict[str, Any] | None:
        point = _dt(now)
        lease = _dt(now + timedelta(seconds=max(1, int(lease_seconds))))

        def operation(conn: sqlite3.Connection) -> dict[str, Any] | None:
            policy = conn.execute(
                """SELECT COALESCE(override.enabled, policy.enabled) AS enabled
                FROM character_instances AS instance
                JOIN scope_state_gate_policies AS policy
                  ON policy.profile_id = instance.profile_id AND policy.scope = instance.scope
                LEFT JOIN instance_state_gate_overrides AS override
                  ON override.profile_id = instance.profile_id
                 AND override.instance_id = instance.instance_id
                WHERE instance.profile_id = ? AND instance.instance_id = ?""",
                (profile_id, instance_id),
            ).fetchone()
            if policy is None or not bool(policy["enabled"]):
                return None
            snapshot = conn.execute(
                """SELECT * FROM instance_state_gate_snapshots
                WHERE profile_id = ? AND instance_id = ?
                  AND reason_code = 'TEMPORARY_ABSENCE'
                  AND not_before_at IS NOT NULL AND not_before_at <= ?
                  AND until_at IS NOT NULL AND until_at > ?
                  AND (
                    action = 'DEFER'
                    OR (
                      action = 'OPEN' AND generation > 0 AND EXISTS (
                        SELECT 1 FROM deferred_message_batches AS batch
                        WHERE batch.profile_id = instance_state_gate_snapshots.profile_id
                          AND batch.instance_id = instance_state_gate_snapshots.instance_id
                          AND batch.gate_generation = instance_state_gate_snapshots.generation - 1
                          AND batch.status IN ('PENDING', 'CLAIMED')
                      )
                    )
                  )""",
                (profile_id, instance_id, point, point),
            ).fetchone()
            if snapshot is None:
                return None
            active = str(snapshot["action"]) == "DEFER"
            gate_generation = (
                int(snapshot["generation"]) if active else int(snapshot["generation"]) - 1
            )
            snapshot_generation = int(snapshot["generation"])
            if active:
                changed = conn.execute(
                    """UPDATE instance_state_gate_snapshots SET action = 'OPEN',
                    generation = generation + 1, version = version + 1, updated_at = ?
                    WHERE profile_id = ? AND instance_id = ? AND version = ?
                      AND action = 'DEFER' AND reason_code = 'TEMPORARY_ABSENCE'""",
                    (point, profile_id, instance_id, int(snapshot["version"])),
                ).rowcount
                if changed != 1:
                    return None
                snapshot_generation += 1
            state = conn.execute(
                """SELECT activity_epoch FROM instance_core_state
                WHERE profile_id = ? AND instance_id = ?""",
                (profile_id, instance_id),
            ).fetchone()
            if state is None:
                raise KeyError((profile_id, instance_id))
            batch_id = self._claim_deferred_batch_in_transaction(
                conn,
                profile_id,
                instance_id,
                expected_activity_epoch=int(state["activity_epoch"]),
                point=point,
                lease=lease,
                merge_reason="timer-interrupt-batch-coalesced",
                required_gate_generation=gate_generation,
            )
            if batch_id is None:
                self._clear_finished_temporary_absence_in_transaction(
                    conn,
                    profile_id,
                    instance_id,
                    gate_generation=gate_generation,
                    snapshot_generation=snapshot_generation,
                    point=point,
                )
            return {
                "reason": str(snapshot["expression_context"] or ""),
                "started_at": str(snapshot["not_before_at"] or ""),
                "planned_until": str(snapshot["until_at"] or ""),
                "ended_at": point,
                "batch_id": batch_id,
            }

        result = await self.uow.run(operation)
        if result is None:
            return None
        batch_id = str(result.pop("batch_id") or "")
        if batch_id:
            batch = await self.get_deferred_message_batch(profile_id, instance_id, batch_id)
            if batch is None:
                raise RuntimeError("claimed deferred batch disappeared")
            result["batch"] = self._deferred_gate_projection(batch)
        return result

    async def resolve_deferred_gate_batch(
        self,
        profile_id: str,
        instance_id: str,
        batch_ref: str,
        *,
        expected_version: int,
        lease_token: int,
        expected_gate_generation: int,
        expected_activity_epoch: int,
        outcome: str,
        resolved_at: datetime,
    ) -> bool:
        point = _dt(resolved_at)

        def operation(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                """UPDATE deferred_message_batches SET status = 'RESOLVED',
                resolution_reason = ?, resolved_at = ?, lease_until = NULL,
                lease_token = lease_token + 1, version = version + 1,
                updated_at = ? WHERE profile_id = ? AND instance_id = ?
                AND batch_id = ? AND version = ? AND lease_token = ?
                AND gate_generation = ? AND activity_epoch = ?
                AND status = 'CLAIMED'""",
                (
                    str(outcome),
                    point,
                    point,
                    profile_id,
                    instance_id,
                    batch_ref,
                    int(expected_version),
                    int(lease_token),
                    int(expected_gate_generation),
                    int(expected_activity_epoch),
                ),
            )
            if cursor.rowcount != 1:
                return False
            conn.execute(
                """UPDATE deferred_message_items SET status = 'RESOLVED',
                resolved_at = ? WHERE batch_id = ? AND status = 'PENDING'""",
                (point, batch_ref),
            )
            self._clear_finished_temporary_absence_in_transaction(
                conn,
                profile_id,
                instance_id,
                gate_generation=int(expected_gate_generation),
                snapshot_generation=int(expected_gate_generation) + 1,
                point=point,
            )
            return True

        return await self.uow.run(operation)

    async def renew_deferred_gate_batch_lease(
        self,
        profile_id: str,
        instance_id: str,
        batch_ref: str,
        *,
        expected_version: int,
        lease_token: int,
        now: datetime,
        lease_seconds: int = 120,
    ) -> bool:
        point = _dt(now)
        lease = _dt(now + timedelta(seconds=max(1, int(lease_seconds))))
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE deferred_message_batches SET lease_until = ?, updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND batch_id = ?
                AND version = ? AND lease_token = ? AND status = 'CLAIMED'
                AND lease_until > ?""",
                (
                    lease,
                    point,
                    profile_id,
                    instance_id,
                    batch_ref,
                    int(expected_version),
                    int(lease_token),
                    point,
                ),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1

    async def release_deferred_gate_batch(
        self,
        profile_id: str,
        instance_id: str,
        batch_ref: str,
        *,
        expected_version: int,
        lease_token: int,
        retry_at: datetime,
        reason: str,
    ) -> bool:
        point = _dt(_now())
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE deferred_message_batches SET status = 'PENDING',
                due_at = ?, resolution_reason = ?, lease_until = NULL,
                lease_token = lease_token + 1, version = version + 1,
                updated_at = ? WHERE profile_id = ? AND instance_id = ?
                AND batch_id = ? AND version = ? AND lease_token = ?
                AND status = 'CLAIMED'""",
                (
                    _dt(retry_at),
                    str(reason),
                    point,
                    profile_id,
                    instance_id,
                    batch_ref,
                    int(expected_version),
                    int(lease_token),
                ),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1
