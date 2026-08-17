from __future__ import annotations

from .support import (
    Any,
    Wakeup,
    WakeupStatus,
    _dt,
    _now,
    datetime,
    sqlite3,
    timedelta,
)


class DeliveryWakeupRecords:
    async def claim_due_instance_wakeups(
        self,
        *,
        now: datetime | None = None,
        limit: int = 10,
        lease_seconds: int = 120,
        profile_id: str | None = None,
        instance_id: str | None = None,
        task_class: str = "BACKGROUND",
        capability: str | None = None,
    ) -> list[Wakeup]:
        current = now or _now()
        current_text = _dt(current)
        lease_until = _dt(current + timedelta(seconds=lease_seconds))

        def operation(conn: sqlite3.Connection) -> list[sqlite3.Row]:
            conn.execute(
                """UPDATE instance_wakeups SET status = 'PENDING',
                    lease_until = NULL, generation = generation + 1,
                    lease_token = lease_token + 1, version = version + 1, updated_at = ?
                WHERE status = 'CLAIMED' AND lease_until <= ?""",
                (current_text, current_text),
            )
            sql = """SELECT wakeup_id FROM instance_wakeups
                WHERE status = 'PENDING' AND due_at <= ?
                AND EXISTS (
                    SELECT 1 FROM role_profiles runtime_profile
                    WHERE runtime_profile.profile_id = instance_wakeups.profile_id
                      AND runtime_profile.enabled = 1
                )"""
            params: list[Any] = [current_text]
            if profile_id is not None:
                sql += " AND profile_id = ?"
                params.append(profile_id)
            if instance_id is not None:
                sql += " AND instance_id = ?"
                params.append(instance_id)
            sql += " ORDER BY due_at, wakeup_id LIMIT ?"
            params.append(max(0, limit))
            ids = [int(row[0]) for row in conn.execute(sql, params)]
            if not ids:
                return []
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"UPDATE instance_wakeups SET status = 'CLAIMED', attempts = attempts + 1, "
                f"lease_until = ?, lease_token = lease_token + 1, "
                f"version = version + 1, updated_at = ? WHERE wakeup_id IN ({placeholders}) "
                "AND status = 'PENDING'",
                (lease_until, current_text, *ids),
            )
            return list(
                conn.execute(
                    f"SELECT * FROM instance_wakeups WHERE wakeup_id IN ({placeholders}) "
                    "AND status = 'CLAIMED' ORDER BY due_at",
                    ids,
                )
            )

        return [self._wakeup(row) for row in await self.uow.run(operation)]

    async def complete_instance_wakeup(
        self,
        profile_id: str,
        instance_id: str,
        wakeup_id: int,
        *,
        expected_generation: int,
        lease_token: int,
        expected_version: int,
    ) -> bool:
        return await self._transition_instance_wakeup(
            profile_id,
            instance_id,
            wakeup_id,
            WakeupStatus.COMPLETED,
            allowed={WakeupStatus.CLAIMED},
            expected_generation=expected_generation,
            lease_token=lease_token,
            expected_version=expected_version,
        )

    async def retry_instance_wakeup(
        self,
        profile_id: str,
        instance_id: str,
        wakeup_id: int,
        due_at: datetime,
        *,
        error: str | None = None,
        expected_generation: int,
        lease_token: int,
        expected_version: int,
    ) -> bool:
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE instance_wakeups SET status = 'PENDING', due_at = ?,
                    lease_until = NULL, last_error = ?, version = version + 1, updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND wakeup_id = ?
                    AND status = 'CLAIMED' AND generation = ? AND lease_token = ?
                    AND version = ?""",
                (
                    _dt(due_at),
                    error,
                    _dt(_now()),
                    profile_id,
                    instance_id,
                    wakeup_id,
                    expected_generation,
                    lease_token,
                    expected_version,
                ),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1

    async def fail_instance_wakeup(
        self,
        profile_id: str,
        instance_id: str,
        wakeup_id: int,
        error: str,
        *,
        expected_generation: int,
        lease_token: int,
        expected_version: int,
    ) -> bool:
        return await self._transition_instance_wakeup(
            profile_id,
            instance_id,
            wakeup_id,
            WakeupStatus.FAILED,
            allowed={WakeupStatus.CLAIMED},
            error=error,
            expected_generation=expected_generation,
            lease_token=lease_token,
            expected_version=expected_version,
        )

    async def _transition_instance_wakeup(
        self,
        profile_id: str,
        instance_id: str,
        wakeup_id: int,
        status: WakeupStatus,
        *,
        allowed: set[WakeupStatus],
        error: str | None = None,
        expected_generation: int,
        lease_token: int,
        expected_version: int,
    ) -> bool:
        placeholders = ",".join("?" for _ in allowed)
        cursor = await self.db.call(
            lambda conn: conn.execute(
                f"UPDATE instance_wakeups SET status = ?, lease_until = NULL, "
                f"version = version + 1, "
                f"last_error = ?, updated_at = ? WHERE profile_id = ? "
                f"AND instance_id = ? AND wakeup_id = ? AND status IN ({placeholders}) "
                f"AND generation = ? AND lease_token = ? AND version = ?",
                (
                    status.value,
                    error,
                    _dt(_now()),
                    profile_id,
                    instance_id,
                    wakeup_id,
                    *(item.value for item in allowed),
                    expected_generation,
                    lease_token,
                    expected_version,
                ),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1
