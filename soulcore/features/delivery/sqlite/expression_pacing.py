from __future__ import annotations

from .support import Any, _dt, _now, _parse, datetime, sqlite3, timedelta


def resolve_expression_pacing_policy_conn(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    platform_instance_id: str,
) -> dict[str, Any]:
    instance = conn.execute(
        """SELECT scope FROM character_instances WHERE profile_id = ?
        AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    if instance is None:
        raise KeyError((profile_id, instance_id))
    scope = str(instance["scope"])
    base = conn.execute(
        """SELECT send_qpm_limit, version
        FROM scope_delivery_policies WHERE profile_id = ? AND scope = ?""",
        (profile_id, scope),
    ).fetchone()
    if base is None:
        raise KeyError((profile_id, scope))
    policy = _base_expression_policy(base)
    platform_version = _apply_platform_policy(conn, profile_id, scope, platform_instance_id, policy)
    override_version = _apply_instance_override(conn, profile_id, instance_id, policy)
    return {
        "profile_id": profile_id,
        "instance_id": instance_id,
        "scope": scope,
        "platform_instance_id": platform_instance_id,
        "send_qpm_limit": max(1, int(policy["send_qpm_limit"])),
        "account_send_qpm_limit": policy["account_send_qpm_limit"],
        "scope_version": int(base["version"]),
        "platform_version": platform_version,
        "override_version": override_version,
    }


def _base_expression_policy(base: sqlite3.Row) -> dict[str, Any]:
    return {
        "send_qpm_limit": int(base["send_qpm_limit"]),
        "account_send_qpm_limit": None,
    }


def _apply_platform_policy(
    conn: sqlite3.Connection,
    profile_id: str,
    scope: str,
    platform_instance_id: str,
    policy: dict[str, Any],
) -> int | None:
    if not platform_instance_id:
        return None
    row = conn.execute(
        """SELECT send_qpm_limit, account_send_qpm_limit, version
        FROM platform_connection_policies WHERE profile_id = ?
          AND scope = ? AND platform_instance_id = ?""",
        (profile_id, scope, platform_instance_id),
    ).fetchone()
    if row is None:
        return None
    for field in (
        "send_qpm_limit",
        "account_send_qpm_limit",
    ):
        if row[field] is not None:
            policy[field] = int(row[field])
    return int(row["version"])


def _apply_instance_override(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    policy: dict[str, Any],
) -> int | None:
    row = conn.execute(
        """SELECT send_qpm_limit, version
        FROM instance_delivery_overrides WHERE profile_id = ? AND instance_id = ?""",
        (profile_id, instance_id),
    ).fetchone()
    if row is None:
        return None
    if row["send_qpm_limit"] is not None:
        policy["send_qpm_limit"] = int(row["send_qpm_limit"])
    return int(row["version"])


class _ExpressionPermitReservation:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def __call__(self, conn: sqlite3.Connection) -> dict[str, Any]:
        policy = resolve_expression_pacing_policy_conn(
            conn,
            self.values["profile_id"],
            self.values["instance_id"],
            self.values["platform_instance_id"],
        )
        account_limit = self._account_limit(policy)
        self._release_expired(conn)
        key = self._reservation_key()
        existing = self._load_existing(conn, key)
        self._validate_route(existing)
        if self._is_active(existing):
            return self._reserved_result(existing, policy)
        release = self._capacity_release(conn, policy, account_limit)
        if release is not None:
            return self._blocked_result(release, policy)
        self._write_permit(conn, key, existing)
        permit = self._load_existing(conn, key)
        assert permit is not None
        return self._reserved_result(permit, policy)

    def _account_limit(self, policy: dict[str, Any]) -> int | None:
        if not self.values["account_key"]:
            return None
        requested = self.values["account_limit"]
        configured = policy.get("account_send_qpm_limit")
        if configured is None:
            return requested
        return int(configured) if requested is None else min(int(requested), int(configured))

    def _release_expired(self, conn: sqlite3.Connection) -> None:
        now_text = self.values["now_text"]
        conn.execute(
            """UPDATE platform_send_permits SET status = 'RELEASED', updated_at = ?
            WHERE status = 'RESERVED' AND (expires_at <= ? OR lease_until <= ?)""",
            (now_text, now_text, now_text),
        )

    def _reservation_key(self) -> str:
        return (
            f"{self.values['profile_id']}:{self.values['instance_id']}:"
            f"EXPRESSION_ITEM:{self.values['origin_id']}:0"
        )

    @staticmethod
    def _load_existing(conn: sqlite3.Connection, key: str) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM platform_send_permits WHERE reservation_key = ?",
            (key,),
        ).fetchone()

    def _validate_route(self, existing: sqlite3.Row | None) -> None:
        if existing is None:
            return
        actual = (
            str(existing["platform_instance_id"]),
            str(existing["target_id"]),
            str(existing["account_key"]),
        )
        expected = (
            self.values["platform_instance_id"],
            self.values["target_id"],
            self.values["account_key"],
        )
        if actual != expected:
            raise ValueError("expression permit is bound to another physical route")

    @staticmethod
    def _is_active(existing: sqlite3.Row | None) -> bool:
        return existing is not None and str(existing["status"]) in {
            "RESERVED",
            "DISPATCHING",
            "ATTEMPTED_UNKNOWN",
        }

    def _capacity_release(
        self,
        conn: sqlite3.Connection,
        policy: dict[str, Any],
        account_limit: int | None,
    ) -> str | None:
        target_used = self._used(conn, "target_id", self.values["target_id"])
        account_used = (
            0
            if account_limit is None
            else self._used(conn, "account_key", self.values["account_key"])
        )
        if target_used < int(policy["send_qpm_limit"]) and (
            account_limit is None or account_used < account_limit
        ):
            return None
        return self._next_available(conn)

    def _used(self, conn: sqlite3.Connection, column: str, value: str) -> int:
        if column not in {"target_id", "account_key"}:
            raise ValueError("unsupported QPM route column")
        return int(
            conn.execute(
                f"""SELECT COUNT(*) FROM platform_send_permits
                WHERE platform_instance_id = ? AND {column} = ?
                  AND status NOT IN ('RELEASED', 'FAILED_BEFORE_DISPATCH')
                  AND expires_at > ?""",
                (self.values["platform_instance_id"], value, self.values["now_text"]),
            ).fetchone()[0]
        )

    def _next_available(self, conn: sqlite3.Connection) -> str | None:
        row = conn.execute(
            """SELECT MIN(expires_at) FROM platform_send_permits
            WHERE platform_instance_id = ? AND (target_id = ? OR (? <> '' AND account_key = ?))
              AND status NOT IN ('RELEASED', 'FAILED_BEFORE_DISPATCH') AND expires_at > ?""",
            (
                self.values["platform_instance_id"],
                self.values["target_id"],
                self.values["account_key"],
                self.values["account_key"],
                self.values["now_text"],
            ),
        ).fetchone()
        return str(row[0]) if row is not None and row[0] is not None else None

    def _write_permit(
        self,
        conn: sqlite3.Connection,
        key: str,
        existing: sqlite3.Row | None,
    ) -> None:
        if existing is None:
            self._insert_permit(conn, key)
            return
        conn.execute(
            """UPDATE platform_send_permits SET status = 'RESERVED', reserved_at = ?,
            expires_at = ?, lease_until = ?, detail = '', dispatched_at = NULL,
            updated_at = ? WHERE permit_id = ?""",
            (
                self.values["now_text"],
                self.values["expires"],
                self.values["lease_until"],
                self.values["now_text"],
                int(existing["permit_id"]),
            ),
        )

    def _insert_permit(self, conn: sqlite3.Connection, key: str) -> None:
        conn.execute(
            """INSERT INTO platform_send_permits(profile_id, instance_id,
            platform_instance_id, target_id, account_key, origin_kind, origin_id,
            fragment_index, reservation_key, reserved_at, expires_at, lease_until, updated_at)
            VALUES (?, ?, ?, ?, ?, 'EXPRESSION_ITEM', ?, 0, ?, ?, ?, ?, ?)""",
            (
                self.values["profile_id"],
                self.values["instance_id"],
                self.values["platform_instance_id"],
                self.values["target_id"],
                self.values["account_key"],
                self.values["origin_id"],
                key,
                self.values["now_text"],
                self.values["expires"],
                self.values["lease_until"],
                self.values["now_text"],
            ),
        )

    @staticmethod
    def _reserved_result(permit: sqlite3.Row, policy: dict[str, Any]) -> dict[str, Any]:
        return {
            "reserved": True,
            "permit": dict(permit),
            "next_available_at": _parse(permit["expires_at"]),
            "policy": policy,
        }

    @staticmethod
    def _blocked_result(release: str | None, policy: dict[str, Any]) -> dict[str, Any]:
        return {
            "reserved": False,
            "permit": None,
            "next_available_at": _parse(release),
            "policy": policy,
        }


class ExpressionPacingRecords:
    async def get_instance_delivery_override(
        self,
        profile_id: str,
        instance_id: str,
    ) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            """SELECT * FROM instance_delivery_overrides WHERE profile_id = ?
            AND instance_id = ?""",
            (profile_id, instance_id),
        )
        return self._record(row, json_columns=()) if row else None

    async def upsert_instance_delivery_override(
        self,
        profile_id: str,
        instance_id: str,
        *,
        send_qpm_limit: int | None = None,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        self._validate_override(send_qpm_limit)
        operation = _UpsertInstanceDeliveryOverride(
            profile_id=profile_id,
            instance_id=instance_id,
            send_qpm_limit=send_qpm_limit,
            expected_version=expected_version,
            now=_dt(_now()),
        )
        row = await self.uow.run(operation)
        return self._record(row, json_columns=())

    @staticmethod
    def _validate_override(send_qpm_limit: int | None) -> None:
        if send_qpm_limit is None:
            raise ValueError("at least one delivery override is required")
        if send_qpm_limit is not None and int(send_qpm_limit) < 1:
            raise ValueError("send_qpm_limit must be positive")

    async def delete_instance_delivery_override(
        self,
        profile_id: str,
        instance_id: str,
        *,
        expected_version: int | None = None,
    ) -> bool:
        params: list[Any] = [profile_id, instance_id]
        sql = """DELETE FROM instance_delivery_overrides
        WHERE profile_id = ? AND instance_id = ?"""
        if expected_version is not None:
            sql += " AND version = ?"
            params.append(int(expected_version))
        cursor = await self.db.call(lambda conn: conn.execute(sql, params), transaction=True)
        return cursor.rowcount == 1

    async def resolve_expression_pacing_policy(
        self,
        profile_id: str,
        instance_id: str,
        *,
        platform_instance_id: str = "",
    ) -> dict[str, Any]:
        return await self.db.call(
            lambda conn: resolve_expression_pacing_policy_conn(
                conn, profile_id, instance_id, str(platform_instance_id)
            )
        )

    async def reserve_expression_send_permit(
        self,
        profile_id: str,
        instance_id: str,
        *,
        platform_instance_id: str,
        target_id: str,
        origin_id: str,
        account_key: str = "",
        account_limit: int | None = None,
        now: datetime | None = None,
        lease_seconds: int = 120,
    ) -> dict[str, Any]:
        values = self._expression_permit_values(
            profile_id,
            instance_id,
            platform_instance_id=platform_instance_id,
            target_id=target_id,
            origin_id=origin_id,
            account_key=account_key,
            account_limit=account_limit,
            now=now,
            lease_seconds=lease_seconds,
        )
        return await self.uow.run(_ExpressionPermitReservation(values))

    @staticmethod
    def _expression_permit_values(
        profile_id: str,
        instance_id: str,
        *,
        platform_instance_id: str,
        target_id: str,
        origin_id: str,
        account_key: str,
        account_limit: int | None,
        now: datetime | None,
        lease_seconds: int,
    ) -> dict[str, Any]:
        route = tuple(str(value).strip() for value in (platform_instance_id, target_id, origin_id))
        if not all(route):
            raise ValueError("physical route and expression origin are required")
        if account_limit is not None and int(account_limit) < 1:
            raise ValueError("account_limit must be positive")
        current = now or _now()
        return {
            "profile_id": profile_id,
            "instance_id": instance_id,
            "platform_instance_id": route[0],
            "target_id": route[1],
            "origin_id": route[2],
            "account_key": str(account_key),
            "account_limit": account_limit,
            "now_text": _dt(current),
            "expires": _dt(current + timedelta(minutes=1)),
            "lease_until": _dt(current + timedelta(seconds=max(1, int(lease_seconds)))),
        }


class _UpsertInstanceDeliveryOverride:
    def __init__(
        self,
        *,
        profile_id: str,
        instance_id: str,
        send_qpm_limit: int | None,
        expected_version: int | None,
        now: str,
    ) -> None:
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.send_qpm_limit = send_qpm_limit
        self.expected_version = expected_version
        self.now = now

    def __call__(self, conn: sqlite3.Connection) -> sqlite3.Row:
        current = conn.execute(
            """SELECT version FROM instance_delivery_overrides
            WHERE profile_id = ? AND instance_id = ?""",
            (self.profile_id, self.instance_id),
        ).fetchone()
        if current is None:
            self._insert(conn)
        else:
            self._update(conn, int(current["version"]))
        row = conn.execute(
            """SELECT * FROM instance_delivery_overrides WHERE profile_id = ?
            AND instance_id = ?""",
            (self.profile_id, self.instance_id),
        ).fetchone()
        assert row is not None
        return row

    def _insert(self, conn: sqlite3.Connection) -> None:
        if self.expected_version not in {None, 0}:
            raise ValueError("instance delivery override version conflict")
        conn.execute(
            """INSERT INTO instance_delivery_overrides(
                profile_id, instance_id, send_qpm_limit, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)""",
            (
                self.profile_id,
                self.instance_id,
                self.send_qpm_limit,
                self.now,
                self.now,
            ),
        )

    def _update(self, conn: sqlite3.Connection, version: int) -> None:
        if self.expected_version is not None and int(self.expected_version) != version:
            raise ValueError("instance delivery override version conflict")
        conn.execute(
            """UPDATE instance_delivery_overrides SET send_qpm_limit = ?,
            version = version + 1, updated_at = ?
            WHERE profile_id = ? AND instance_id = ?""",
            (
                self.send_qpm_limit,
                self.now,
                self.profile_id,
                self.instance_id,
            ),
        )


__all__ = ["ExpressionPacingRecords", "resolve_expression_pacing_policy_conn"]
