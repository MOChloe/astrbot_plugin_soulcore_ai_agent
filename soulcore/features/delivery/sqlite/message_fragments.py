from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from typing import Any

from ....contracts.models import (
    MessageDirection,
    MessageRetractionStatus,
    PlatformMessageFragment,
)
from ....storage.sqlite.codec import _dt, _now, _parse
from .message_retractions import MessageRetractionRecords


def _fragment(row: sqlite3.Row) -> PlatformMessageFragment:
    status = row["retraction_status"]
    return PlatformMessageFragment(
        message_ref=str(row["message_ref"]),
        profile_id=str(row["profile_id"]),
        instance_id=str(row["instance_id"]),
        ledger_message_id=int(row["ledger_message_id"]),
        fragment_ordinal=int(row["fragment_ordinal"]),
        platform_instance_id=str(row["platform_instance_id"]),
        route_umo=str(row["route_umo"]),
        platform_message_id=str(row["platform_message_id"]),
        direction=MessageDirection(row["direction"]),
        content_kind=str(row["content_kind"]),
        platform_reference_id=str(row["platform_reference_id"]),
        content_projection=str(row["content_projection"]),
        sender_id=str(row["sender_id"]),
        native_reply_supported=bool(row["native_reply_supported"]),
        member_mention_supported=bool(row["member_mention_supported"]),
        self_retraction_supported=bool(row["self_retraction_supported"]),
        returns_platform_message_id=bool(row["returns_platform_message_id"]),
        accepted_at=_parse(row["accepted_at"]),
        retractable_until=_parse(row["retractable_until"]),
        retraction_status=MessageRetractionStatus(status) if status else None,
        created_at=_parse(row["created_at"]),
        updated_at=_parse(row["updated_at"]),
    )


class MessageFragmentRecords:
    async def create_message_fragment(
        self,
        profile_id: str,
        instance_id: str,
        *,
        ledger_message_id: int,
        fragment_ordinal: int,
        platform_instance_id: str,
        route_umo: str,
        platform_message_id: str,
        direction: MessageDirection | str,
        content_kind: str,
        platform_reference_id: str = "",
        content_projection: str = "",
        sender_id: str = "",
        native_reply_supported: bool = False,
        member_mention_supported: bool = False,
        self_retraction_supported: bool = False,
        returns_platform_message_id: bool = True,
        accepted_at: datetime | None = None,
        retractable_until: datetime | None = None,
        message_ref: str | None = None,
    ) -> PlatformMessageFragment:
        values = self._fragment_values(
            profile_id,
            instance_id,
            ledger_message_id=ledger_message_id,
            fragment_ordinal=fragment_ordinal,
            platform_instance_id=platform_instance_id,
            route_umo=route_umo,
            platform_message_id=platform_message_id,
            direction=direction,
            content_kind=content_kind,
            platform_reference_id=platform_reference_id,
            content_projection=content_projection,
            sender_id=sender_id,
            native_reply_supported=native_reply_supported,
            member_mention_supported=member_mention_supported,
            self_retraction_supported=self_retraction_supported,
            returns_platform_message_id=returns_platform_message_id,
            accepted_at=accepted_at,
            retractable_until=retractable_until,
            message_ref=message_ref,
        )

        def operation(conn: sqlite3.Connection) -> sqlite3.Row:
            existing = self._existing_fragment(conn, values)
            if existing is not None:
                self._validate_fragment_replay(existing, values)
                return self._record_initial_platform_reference_id(conn, existing, values)
            conn.execute(
                """INSERT INTO instance_message_fragments(
                    message_ref, profile_id, instance_id, ledger_message_id,
                    fragment_ordinal, platform_instance_id, route_umo,
                    platform_message_id, direction, content_kind, platform_reference_id,
                    content_projection,
                    sender_id, native_reply_supported, member_mention_supported,
                    self_retraction_supported, returns_platform_message_id,
                    accepted_at, retractable_until, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(
                    values[key]
                    for key in (
                        "message_ref",
                        "profile_id",
                        "instance_id",
                        "ledger_message_id",
                        "fragment_ordinal",
                        "platform_instance_id",
                        "route_umo",
                        "platform_message_id",
                        "direction",
                        "content_kind",
                        "platform_reference_id",
                        "content_projection",
                        "sender_id",
                        "native_reply_supported",
                        "member_mention_supported",
                        "self_retraction_supported",
                        "returns_platform_message_id",
                        "accepted_at",
                        "retractable_until",
                        "created_at",
                        "updated_at",
                    )
                ),
            )
            row = conn.execute(
                "SELECT * FROM instance_message_fragments WHERE message_ref = ?",
                (values["message_ref"],),
            ).fetchone()
            assert row is not None
            return row

        return _fragment(await self.uow.run(operation))

    async def get_message_fragment(
        self, profile_id: str, instance_id: str, message_ref: str
    ) -> PlatformMessageFragment | None:
        row = await self.db.fetch_one(
            """SELECT * FROM instance_message_fragments
            WHERE profile_id = ? AND instance_id = ? AND message_ref = ?""",
            (profile_id, instance_id, str(message_ref)),
        )
        return _fragment(row) if row else None

    async def get_message_fragment_by_platform_locator(
        self,
        profile_id: str,
        instance_id: str,
        *,
        platform_instance_id: str,
        route_umo: str,
        platform_locator: str,
    ) -> PlatformMessageFragment | None:
        """Resolve either a platform message id or its native quote reference id.

        QQ official exposes the quoted target as a REFIDX value, while OneBot
        and other adapters normally expose the ordinary platform message id.
        Prefer the exact adapter route. Some adapters rewrite the same physical
        route between outbound and inbound events, so an exact miss falls back
        to a unique locator within the same SoulCore instance. Ambiguous
        locators are rejected instead of guessed.
        """

        locator = str(platform_locator or "").strip()
        if not locator:
            return None
        rows = await self.db.fetch_all(
            """SELECT * FROM instance_message_fragments
            WHERE profile_id = ? AND instance_id = ?
              AND platform_instance_id = ? AND route_umo = ?
              AND (platform_message_id = ? OR platform_reference_id = ?)
            LIMIT 2""",
            (
                profile_id,
                instance_id,
                str(platform_instance_id or "").strip(),
                str(route_umo or "").strip(),
                locator,
                locator,
            ),
        )
        if len(rows) == 1:
            return _fragment(rows[0])
        if len(rows) > 1:
            return None
        scoped_rows = await self.db.fetch_all(
            """SELECT * FROM instance_message_fragments
            WHERE profile_id = ? AND instance_id = ?
              AND platform_instance_id = ?
              AND (platform_message_id = ? OR platform_reference_id = ?)
            LIMIT 2""",
            (
                profile_id,
                instance_id,
                str(platform_instance_id or "").strip(),
                locator,
                locator,
            ),
        )
        return _fragment(scoped_rows[0]) if len(scoped_rows) == 1 else None

    async def list_message_fragments(
        self,
        profile_id: str,
        instance_id: str,
        *,
        ledger_message_id: int | None = None,
        ledger_message_ids: list[int] | tuple[int, ...] = (),
        direction: MessageDirection | str | None = None,
        include_retracted: bool = True,
        limit: int = 100,
    ) -> list[PlatformMessageFragment]:
        clauses = ["profile_id = ?", "instance_id = ?"]
        params: list[Any] = [profile_id, instance_id]
        if ledger_message_id is not None:
            clauses.append("ledger_message_id = ?")
            params.append(int(ledger_message_id))
        ids = tuple(dict.fromkeys(int(value) for value in ledger_message_ids))
        if ids:
            if any(value < 1 for value in ids):
                raise ValueError("ledger_message_ids must be positive")
            clauses.append(f"ledger_message_id IN ({','.join('?' for _ in ids)})")
            params.extend(ids)
        if direction is not None:
            clauses.append("direction = ?")
            params.append(self._direction(direction))
        if not include_retracted:
            clauses.append("COALESCE(retraction_status, '') != 'RETRACTED'")
        params.append(max(1, min(int(limit), 500)))
        rows = await self.db.fetch_all(
            f"""SELECT * FROM instance_message_fragments
            WHERE {" AND ".join(clauses)}
            ORDER BY ledger_message_id DESC, fragment_ordinal LIMIT ?""",
            params,
        )
        return [_fragment(row) for row in rows]

    async def list_message_fragments_for_expression_output(
        self,
        profile_id: str,
        instance_id: str,
        expression_batch_id: str,
        output_ordinal: int,
    ) -> list[PlatformMessageFragment]:
        ordinal = int(output_ordinal)
        if ordinal < 1:
            raise ValueError("output_ordinal is 1-based and must be positive")
        rows = await self.db.fetch_all(
            """SELECT fragment.* FROM instance_message_fragments fragment
            JOIN instance_messages message
              ON message.profile_id = fragment.profile_id
             AND message.instance_id = fragment.instance_id
             AND message.message_id = fragment.ledger_message_id
            WHERE fragment.profile_id = ? AND fragment.instance_id = ?
              AND message.expression_batch_id = ? AND message.expression_ordinal = ?
            ORDER BY fragment.fragment_ordinal""",
            (profile_id, instance_id, expression_batch_id, ordinal - 1),
        )
        return [_fragment(row) for row in rows]

    @classmethod
    def _fragment_values(cls, profile_id: str, instance_id: str, **raw: Any) -> dict[str, Any]:
        required = ("platform_instance_id", "route_umo", "platform_message_id")
        normalized = {key: str(raw[key] or "").strip() for key in required}
        if not all(normalized.values()):
            raise ValueError("platform instance, route, and platform message id are required")
        fragment_ordinal = int(raw["fragment_ordinal"])
        if fragment_ordinal < 0:
            raise ValueError("fragment_ordinal cannot be negative")
        kind = str(raw["content_kind"] or "").strip().upper()
        if kind not in {"TEXT", "IMAGE", "STICKER", "FILE", "OTHER"}:
            raise ValueError("unsupported platform message fragment content kind")
        current = _now()
        accepted = raw["accepted_at"] or current
        return {
            "message_ref": str(raw["message_ref"] or f"msgref:v1:{uuid.uuid4().hex}"),
            "profile_id": profile_id,
            "instance_id": instance_id,
            "ledger_message_id": int(raw["ledger_message_id"]),
            "fragment_ordinal": fragment_ordinal,
            **normalized,
            "direction": cls._direction(raw["direction"]),
            "content_kind": kind,
            "platform_reference_id": str(raw["platform_reference_id"] or "").strip(),
            "content_projection": str(raw["content_projection"] or ""),
            "sender_id": str(raw["sender_id"] or ""),
            "native_reply_supported": int(bool(raw["native_reply_supported"])),
            "member_mention_supported": int(bool(raw["member_mention_supported"])),
            "self_retraction_supported": int(bool(raw["self_retraction_supported"])),
            "returns_platform_message_id": int(bool(raw["returns_platform_message_id"])),
            "accepted_at": _dt(accepted),
            "retractable_until": _dt(raw["retractable_until"]),
            "created_at": _dt(current),
            "updated_at": _dt(current),
        }

    @staticmethod
    def _direction(value: MessageDirection | str) -> str:
        normalized = str(value.value if isinstance(value, MessageDirection) else value).upper()
        return MessageDirection(normalized).value

    @staticmethod
    def _existing_fragment(conn: sqlite3.Connection, values: dict[str, Any]) -> sqlite3.Row | None:
        row = conn.execute(
            "SELECT * FROM instance_message_fragments WHERE message_ref = ?",
            (values["message_ref"],),
        ).fetchone()
        return (
            row
            or conn.execute(
                """SELECT * FROM instance_message_fragments
            WHERE platform_instance_id = ? AND route_umo = ? AND platform_message_id = ?""",
                (
                    values["platform_instance_id"],
                    values["route_umo"],
                    values["platform_message_id"],
                ),
            ).fetchone()
        )

    @staticmethod
    def _validate_fragment_replay(row: sqlite3.Row, values: dict[str, Any]) -> None:
        identity = (
            "profile_id",
            "instance_id",
            "ledger_message_id",
            "fragment_ordinal",
            "platform_instance_id",
            "route_umo",
            "platform_message_id",
            "direction",
        )
        if any(str(row[key]) != str(values[key]) for key in identity):
            raise ValueError("platform message fragment replay conflicts with stored identity")

    @staticmethod
    def _record_initial_platform_reference_id(
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        values: dict[str, Any],
    ) -> sqlite3.Row:
        stored = str(row["platform_reference_id"] or "").strip()
        replayed = str(values["platform_reference_id"] or "").strip()
        if stored and replayed and stored != replayed:
            raise ValueError("platform message fragment replay conflicts with stored reference id")
        restore_native_reply = bool(values["native_reply_supported"]) and not bool(
            row["native_reply_supported"]
        )
        if (stored or not replayed) and not restore_native_reply:
            return row
        conn.execute(
            """UPDATE instance_message_fragments
            SET platform_reference_id = CASE
                    WHEN platform_reference_id = '' THEN ?
                    ELSE platform_reference_id
                END,
                native_reply_supported = CASE
                    WHEN ? = 1 THEN 1
                    ELSE native_reply_supported
                END,
                updated_at = ?
            WHERE message_ref = ?""",
            (
                replayed,
                int(bool(values["native_reply_supported"])),
                values["updated_at"],
                str(row["message_ref"]),
            ),
        )
        updated = conn.execute(
            "SELECT * FROM instance_message_fragments WHERE message_ref = ?",
            (str(row["message_ref"]),),
        ).fetchone()
        assert updated is not None
        return updated


__all__ = ["MessageFragmentRecords", "MessageRetractionRecords"]
