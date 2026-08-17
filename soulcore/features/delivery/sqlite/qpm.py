from __future__ import annotations

from .group_first_attempt import (
    BeginGroupAwareDispatchPermit,
    PrepareGroupDispatchAnchor,
    ResolveUndeliverableGroupWindow,
    mark_group_first_attempt_started,
)
from .support import (
    Any,
    _dt,
    _now,
    _parse,
    datetime,
    sqlite3,
    timedelta,
)


class _PlatformSendPermitResize:
    def __init__(
        self,
        *,
        profile_id: str,
        instance_id: str,
        platform_instance_id: str,
        target_id: str,
        origin_kind: str,
        origin_id: str,
        desired: int,
        group_limit: int,
        account_key: str,
        account_limit: int | None,
        now_text: str,
        expires: str,
        lease_until: str,
    ) -> None:
        self.profile_id = profile_id
        self.instance_id = instance_id
        self.platform_instance_id = platform_instance_id
        self.target_id = target_id
        self.origin_kind = origin_kind
        self.origin_id = origin_id
        self.desired = desired
        self.group_limit = int(group_limit)
        self.account_key = account_key
        self.account_limit = account_limit
        self.now_text = now_text
        self.expires = expires
        self.lease_until = lease_until

    def __call__(
        self,
        conn: sqlite3.Connection,
    ) -> list[sqlite3.Row] | None:
        conn.execute(
            """UPDATE platform_send_permits SET status = 'RELEASED',
            updated_at = ? WHERE status = 'RESERVED'
              AND (expires_at <= ? OR lease_until <= ?)""",
            (self.now_text, self.now_text, self.now_text),
        )
        rows = self._load_existing(conn)
        self._validate_route_binding(rows)
        if self._started_count(rows) > self.desired:
            return None
        active_by_index = self._active_by_index(rows)
        self._release_surplus(conn, active_by_index)
        missing = [index for index in range(self.desired) if index not in active_by_index]
        if not self._has_capacity(conn, len(missing)):
            return None
        self._restore_missing(conn, missing)
        self._renew_reserved(conn)
        return self._load_active(conn)

    def _load_existing(self, conn: sqlite3.Connection) -> list[sqlite3.Row]:
        return list(
            conn.execute(
                """SELECT * FROM platform_send_permits WHERE profile_id = ?
            AND instance_id = ? AND origin_kind = ? AND origin_id = ?
            ORDER BY fragment_index""",
                (
                    self.profile_id,
                    self.instance_id,
                    self.origin_kind,
                    self.origin_id,
                ),
            )
        )

    def _validate_route_binding(self, rows: list[sqlite3.Row]) -> None:
        expected = (
            self.platform_instance_id,
            self.target_id,
            self.account_key,
        )
        for row in rows:
            actual = (
                row["platform_instance_id"],
                row["target_id"],
                row["account_key"],
            )
            if actual != expected:
                raise ValueError("reservation is bound to another physical route")

    @staticmethod
    def _started_count(rows: list[sqlite3.Row]) -> int:
        started_statuses = {"DISPATCHING", "ATTEMPTED_UNKNOWN"}
        return sum(1 for row in rows if row["status"] in started_statuses)

    @staticmethod
    def _active_by_index(rows: list[sqlite3.Row]) -> dict[int, sqlite3.Row]:
        terminal_statuses = {"RELEASED", "FAILED_BEFORE_DISPATCH"}
        return {
            int(row["fragment_index"]): row
            for row in rows
            if row["status"] not in terminal_statuses
        }

    def _release_surplus(
        self,
        conn: sqlite3.Connection,
        active_by_index: dict[int, sqlite3.Row],
    ) -> None:
        for index, row in list(active_by_index.items()):
            if index < self.desired or row["status"] != "RESERVED":
                continue
            conn.execute(
                """UPDATE platform_send_permits SET status = 'RELEASED',
                updated_at = ? WHERE permit_id = ? AND status = 'RESERVED'""",
                (self.now_text, row["permit_id"]),
            )
            active_by_index.pop(index)

    def _has_capacity(self, conn: sqlite3.Connection, missing: int) -> bool:
        used_group = self._count_used(
            conn,
            "target_id",
            self.target_id,
        )
        if used_group + missing > self.group_limit:
            return False
        if self.account_limit is None:
            return True
        used_account = self._count_used(
            conn,
            "account_key",
            self.account_key,
        )
        return used_account + missing <= int(self.account_limit)

    def _count_used(
        self,
        conn: sqlite3.Connection,
        route_column: str,
        route_value: str,
    ) -> int:
        if route_column not in {"target_id", "account_key"}:
            raise ValueError("unsupported QPM route column")
        return int(
            conn.execute(
                f"""SELECT COUNT(*) FROM platform_send_permits
            WHERE platform_instance_id = ? AND {route_column} = ?
              AND status NOT IN ('RELEASED', 'FAILED_BEFORE_DISPATCH')
              AND expires_at > ?""",
                (self.platform_instance_id, route_value, self.now_text),
            ).fetchone()[0]
        )

    def _restore_missing(
        self,
        conn: sqlite3.Connection,
        missing: list[int],
    ) -> None:
        for index in missing:
            key = self._reservation_key(index)
            old = conn.execute(
                """SELECT permit_id, status FROM platform_send_permits
                WHERE reservation_key = ?""",
                (key,),
            ).fetchone()
            if old is None:
                self._insert_missing(conn, index, key)
            elif old["status"] in {"RELEASED", "FAILED_BEFORE_DISPATCH"}:
                self._reactivate_missing(conn, old)

    def _reservation_key(self, index: int) -> str:
        return f"{self.profile_id}:{self.instance_id}:{self.origin_kind}:{self.origin_id}:{index}"

    def _insert_missing(
        self,
        conn: sqlite3.Connection,
        index: int,
        key: str,
    ) -> None:
        conn.execute(
            """INSERT INTO platform_send_permits(
                profile_id, instance_id, platform_instance_id, target_id,
                account_key, origin_kind, origin_id, fragment_index,
                reservation_key, reserved_at, expires_at, lease_until,
                updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.profile_id,
                self.instance_id,
                self.platform_instance_id,
                self.target_id,
                self.account_key,
                self.origin_kind,
                self.origin_id,
                index,
                key,
                self.now_text,
                self.expires,
                self.lease_until,
                self.now_text,
            ),
        )

    def _reactivate_missing(
        self,
        conn: sqlite3.Connection,
        old: sqlite3.Row,
    ) -> None:
        conn.execute(
            """UPDATE platform_send_permits SET status = 'RESERVED',
            reserved_at = ?, expires_at = ?, lease_until = ?, detail = '',
            dispatched_at = NULL, updated_at = ? WHERE permit_id = ?""",
            (
                self.now_text,
                self.expires,
                self.lease_until,
                self.now_text,
                old["permit_id"],
            ),
        )

    def _renew_reserved(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """UPDATE platform_send_permits SET lease_until = ?, updated_at = ?
            WHERE profile_id = ? AND instance_id = ? AND origin_kind = ?
              AND origin_id = ? AND status = 'RESERVED'""",
            (
                self.lease_until,
                self.now_text,
                self.profile_id,
                self.instance_id,
                self.origin_kind,
                self.origin_id,
            ),
        )

    def _load_active(self, conn: sqlite3.Connection) -> list[sqlite3.Row]:
        return list(
            conn.execute(
                """SELECT * FROM platform_send_permits WHERE profile_id = ?
            AND instance_id = ? AND origin_kind = ? AND origin_id = ?
              AND status NOT IN ('RELEASED', 'FAILED_BEFORE_DISPATCH')
            ORDER BY fragment_index""",
                (
                    self.profile_id,
                    self.instance_id,
                    self.origin_kind,
                    self.origin_id,
                ),
            )
        )


class QpmRecords:
    async def reserve_platform_send_permits(
        self,
        profile_id: str,
        instance_id: str,
        *,
        platform_instance_id: str,
        target_id: str,
        origin_kind: str,
        origin_id: str,
        fragment_count: int,
        group_limit: int = 20,
        account_key: str = "",
        account_limit: int | None = None,
        now: datetime | None = None,
        lease_seconds: int = 120,
    ) -> list[dict[str, Any]] | None:
        count = int(fragment_count)
        if count < 1:
            raise ValueError("fragment_count must be positive")
        if int(group_limit) < 1 or (account_limit is not None and int(account_limit) < 1):
            raise ValueError("send limits must be positive")
        current = now or _now()
        now_text = _dt(current)
        expires = _dt(current + timedelta(minutes=1))
        lease_until = _dt(current + timedelta(seconds=max(1, int(lease_seconds))))

        rows = await self.uow.run(
            _PlatformSendPermitResize(
                profile_id=profile_id,
                instance_id=instance_id,
                platform_instance_id=platform_instance_id,
                target_id=target_id,
                origin_kind=origin_kind,
                origin_id=origin_id,
                desired=count,
                group_limit=group_limit,
                account_key=account_key,
                account_limit=account_limit,
                now_text=now_text,
                expires=expires,
                lease_until=lease_until,
            )
        )
        return [self._record(row, json_columns=()) for row in rows] if rows is not None else None

    async def resize_platform_send_permits(
        self,
        profile_id: str,
        instance_id: str,
        *,
        platform_instance_id: str,
        target_id: str,
        origin_kind: str,
        origin_id: str,
        fragment_count: int,
        group_limit: int = 20,
        account_key: str = "",
        account_limit: int | None = None,
        now: datetime | None = None,
        lease_seconds: int = 120,
    ) -> list[dict[str, Any]] | None:
        desired = int(fragment_count)
        if desired < 1:
            raise ValueError("fragment_count must be positive")
        if int(group_limit) < 1 or (account_limit is not None and int(account_limit) < 1):
            raise ValueError("send limits must be positive")
        current = now or _now()
        now_text = _dt(current)
        expires = _dt(current + timedelta(minutes=1))
        lease_until = _dt(current + timedelta(seconds=max(1, int(lease_seconds))))
        operation = _PlatformSendPermitResize(
            profile_id=profile_id,
            instance_id=instance_id,
            platform_instance_id=platform_instance_id,
            target_id=target_id,
            origin_kind=origin_kind,
            origin_id=origin_id,
            desired=desired,
            group_limit=group_limit,
            account_key=account_key,
            account_limit=account_limit,
            now_text=now_text,
            expires=expires,
            lease_until=lease_until,
        )
        rows = await self.uow.run(operation)
        return [self._record(row, json_columns=()) for row in rows] if rows is not None else None

    async def begin_dispatch_platform_send_permit(
        self,
        permit_id: int,
        *,
        now: datetime | None = None,
        profile_id: str = "",
        instance_id: str = "",
        group_window_id: str = "",
        outbox_id: int | None = None,
    ) -> bool:
        current_time = now or _now()
        current = _dt(current_time)
        if group_window_id and (not profile_id or not instance_id):
            raise ValueError("group send permit requires profile and instance identity")
        result = await self.uow.run(
            BeginGroupAwareDispatchPermit(
                permit_id,
                now=current,
                expires_at=_dt(current_time + timedelta(minutes=1)),
                profile_id=str(profile_id),
                instance_id=str(instance_id),
                group_window_id=str(group_window_id),
                outbox_id=outbox_id,
            )
        )
        return bool(result.get("started"))

    async def begin_group_expression_send_permit(
        self,
        permit_id: int,
        *,
        profile_id: str,
        instance_id: str,
        group_window_id: str,
        outbox_id: int,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if not all(
            (
                str(profile_id).strip(),
                str(instance_id).strip(),
                str(group_window_id).strip(),
            )
        ):
            raise ValueError("group expression send permit requires its full identity")
        current_time = now or _now()
        return dict(
            await self.uow.run(
                BeginGroupAwareDispatchPermit(
                    permit_id,
                    now=_dt(current_time),
                    expires_at=_dt(current_time + timedelta(minutes=1)),
                    profile_id=str(profile_id),
                    instance_id=str(instance_id),
                    group_window_id=str(group_window_id),
                    outbox_id=int(outbox_id),
                )
            )
        )

    async def prepare_group_expression_dispatch(
        self,
        permit_id: int,
        *,
        profile_id: str,
        instance_id: str,
        group_window_id: str,
        outbox_id: int,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if not all(
            (
                str(profile_id).strip(),
                str(instance_id).strip(),
                str(group_window_id).strip(),
            )
        ):
            raise ValueError("group expression dispatch requires its full identity")
        return dict(
            await self.uow.run(
                PrepareGroupDispatchAnchor(
                    permit_id,
                    now=_dt(now or _now()),
                    profile_id=str(profile_id),
                    instance_id=str(instance_id),
                    group_window_id=str(group_window_id),
                    outbox_id=int(outbox_id),
                )
            )
        )

    async def resolve_group_window_if_no_deliverable(
        self,
        profile_id: str,
        instance_id: str,
        group_window_id: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        if not str(group_window_id).strip():
            return False
        return bool(
            await self.uow.run(
                ResolveUndeliverableGroupWindow(
                    str(profile_id),
                    str(instance_id),
                    str(group_window_id),
                    _dt(now or _now()),
                )
            )
        )

    async def mark_platform_send_permit_attempted_unknown(
        self,
        permit_id: int,
        *,
        detail: str = "",
        now: datetime | None = None,
        profile_id: str = "",
        instance_id: str = "",
        group_window_id: str = "",
        outbox_id: int | None = None,
    ) -> bool:
        current = _dt(now or _now())

        def operation(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                """UPDATE platform_send_permits SET status = 'ATTEMPTED_UNKNOWN',
                detail = ?, updated_at = ? WHERE permit_id = ?
                  AND status = 'DISPATCHING'
                  AND (? IS NULL OR (
                    profile_id = ? AND instance_id = ?
                    AND origin_kind = 'EXPRESSION_ITEM' AND origin_id = ?
                  ))""",
                (
                    str(detail),
                    current,
                    int(permit_id),
                    outbox_id,
                    str(profile_id),
                    str(instance_id),
                    f"expression-outbox:{outbox_id}",
                ),
            )
            if cursor.rowcount != 1:
                return False
            if group_window_id:
                if not profile_id or not instance_id:
                    raise ValueError(
                        "group platform attempt requires profile and instance identity"
                    )
                mark_group_first_attempt_started(
                    conn,
                    profile_id=str(profile_id),
                    instance_id=str(instance_id),
                    group_window_id=str(group_window_id),
                    now=current,
                )
            return True

        return bool(await self.uow.run(operation))

    async def release_platform_send_permits(
        self,
        profile_id: str,
        instance_id: str,
        origin_kind: str,
        origin_id: str,
        *,
        now: datetime | None = None,
    ) -> int:
        current = _dt(now or _now())
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE platform_send_permits SET status = 'RELEASED',
                updated_at = ? WHERE profile_id = ? AND instance_id = ?
                  AND origin_kind = ? AND origin_id = ? AND status = 'RESERVED'""",
                (current, profile_id, instance_id, origin_kind, origin_id),
            ),
            transaction=True,
        )
        return int(cursor.rowcount)

    async def fail_platform_send_permit_before_dispatch(
        self,
        permit_id: int,
        *,
        detail: str = "",
        now: datetime | None = None,
    ) -> bool:
        current = _dt(now or _now())
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE platform_send_permits SET status = 'FAILED_BEFORE_DISPATCH',
                detail = ?, updated_at = ? WHERE permit_id = ?
                  AND status IN ('RESERVED', 'DISPATCHING')""",
                (str(detail), current, int(permit_id)),
            ),
            transaction=True,
        )
        return cursor.rowcount == 1

    async def renew_platform_send_permits(
        self,
        profile_id: str,
        instance_id: str,
        origin_kind: str,
        origin_id: str,
        *,
        lease_seconds: int = 120,
        now: datetime | None = None,
    ) -> int:
        current = now or _now()
        now_text = _dt(current)
        lease = _dt(current + timedelta(seconds=max(1, int(lease_seconds))))
        expires = _dt(current + timedelta(minutes=1))
        cursor = await self.db.call(
            lambda conn: conn.execute(
                """UPDATE platform_send_permits SET lease_until = ?, expires_at = ?,
                updated_at = ?
                WHERE profile_id = ? AND instance_id = ? AND origin_kind = ?
                  AND origin_id = ? AND status = 'RESERVED' AND expires_at > ?""",
                (
                    lease,
                    expires,
                    now_text,
                    profile_id,
                    instance_id,
                    origin_kind,
                    origin_id,
                    now_text,
                ),
            ),
            transaction=True,
        )
        return int(cursor.rowcount)

    async def list_platform_send_permits(
        self,
        profile_id: str,
        instance_id: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            """SELECT * FROM platform_send_permits WHERE profile_id = ?
            AND instance_id = ? ORDER BY permit_id DESC LIMIT ?""",
            (profile_id, instance_id, max(0, int(limit))),
        )
        return [self._record(row, json_columns=()) for row in rows]

    async def snapshot_platform_send_qpm(
        self,
        *,
        platform_instance_id: str,
        target_id: str,
        account_key: str = "",
        group_limit: int | None = None,
        account_limit: int | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = _dt(now or _now())
        row = await self.db.fetch_one(
            """SELECT
                SUM(CASE WHEN target_id = ? AND status = 'RESERVED' THEN 1 ELSE 0 END) AS group_reserved,
                SUM(CASE WHEN target_id = ? AND status IN ('DISPATCHING', 'ATTEMPTED_UNKNOWN') THEN 1 ELSE 0 END) AS group_attempted,
                SUM(CASE WHEN account_key = ? AND status = 'RESERVED' THEN 1 ELSE 0 END) AS account_reserved,
                SUM(CASE WHEN account_key = ? AND status IN ('DISPATCHING', 'ATTEMPTED_UNKNOWN') THEN 1 ELSE 0 END) AS account_attempted,
                MIN(expires_at) AS next_release_at
            FROM platform_send_permits
            WHERE platform_instance_id = ?
              AND status NOT IN ('RELEASED', 'FAILED_BEFORE_DISPATCH')
              AND expires_at > ?""",
            (target_id, target_id, account_key, account_key, platform_instance_id, current),
        )
        group_reserved = int(row["group_reserved"] or 0)
        group_attempted = int(row["group_attempted"] or 0)
        account_reserved = int(row["account_reserved"] or 0)
        account_attempted = int(row["account_attempted"] or 0)
        return {
            "platform_instance_id": platform_instance_id,
            "target_id": target_id,
            "account_key": account_key,
            "group_reserved": group_reserved,
            "group_attempted": group_attempted,
            "group_used": group_reserved + group_attempted,
            "account_reserved": account_reserved,
            "account_attempted": account_attempted,
            "account_used": account_reserved + account_attempted,
            "group_limit": group_limit,
            "account_limit": account_limit,
            "next_release_at": _parse(row["next_release_at"]),
            "as_of": _parse(current),
        }
