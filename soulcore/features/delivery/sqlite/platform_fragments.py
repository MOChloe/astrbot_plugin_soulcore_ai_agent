"""Platform receipt projection for durable delivery ledgers."""

from __future__ import annotations

import hashlib

from .support import _dt, _load, datetime, sqlite3, timedelta


def _fragment_content(outbox: sqlite3.Row, message: sqlite3.Row) -> tuple[str, str]:
    payload = _load(outbox["payload_json"]) or {}
    kind = str(payload.get("expression_kind") or "OTHER").upper()
    if kind not in {"TEXT", "IMAGE", "STICKER", "FILE"}:
        kind = "OTHER"
    projection = str(message["plain_text"] or "").strip()[:120]
    if projection:
        return kind, projection
    placeholders = {
        "IMAGE": "[图片]",
        "STICKER": "[表情包]",
        "FILE": "[文件]",
        "OTHER": "[媒体消息]",
    }
    return kind, placeholders.get(kind, "")


def _foreground_fragment_content(message: sqlite3.Row) -> tuple[str, str]:
    components = _load(message["components_json"]) or []
    component_types = {
        str(component.get("type") or "").lower()
        for component in components
        if isinstance(component, dict)
    }
    if "file_artifact" in component_types:
        kind = "FILE"
    elif "sticker" in component_types or "sticker_ref" in component_types:
        kind = "STICKER"
    elif "image_asset" in component_types or "image" in component_types:
        kind = "IMAGE"
    else:
        kind = "TEXT"
    projection = str(message["plain_text"] or "").strip()[:120]
    if projection:
        return kind, projection
    return kind, {
        "IMAGE": "[图片]",
        "STICKER": "[表情包]",
        "FILE": "[文件]",
        "TEXT": "",
    }[kind]


def _insert_platform_fragment(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    instance_id: str,
    message: sqlite3.Row,
    route_umo: str,
    kind: str,
    projection: str,
    accepted_at: datetime,
    platform_message_id: str,
    fragment_ordinal: int,
    platform_id: str,
    platform_reference_id: str,
    native_reply_supported: bool,
    member_mention_supported: bool,
    self_retraction_supported: bool,
    returns_platform_message_id: bool,
    retractable_for_seconds: int | None,
    now: str,
) -> None:
    identity = (
        f"{profile_id}\x1f{instance_id}\x1f{message['message_id']}\x1f"
        f"{fragment_ordinal}\x1f{platform_message_id}"
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()
    deadline = (
        accepted_at + timedelta(seconds=max(1, int(retractable_for_seconds)))
        if retractable_for_seconds is not None
        else None
    )
    conn.execute(
        """INSERT INTO instance_message_fragments(
            message_ref, profile_id, instance_id, ledger_message_id,
            fragment_ordinal, platform_instance_id, route_umo,
            platform_message_id, platform_reference_id,
            direction, content_kind, content_projection,
            sender_id, native_reply_supported, member_mention_supported,
            self_retraction_supported, returns_platform_message_id,
            accepted_at, retractable_until, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OUTBOUND', ?, ?, 'soulcore',
            ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(profile_id, instance_id, ledger_message_id, fragment_ordinal)
        DO NOTHING""",
        (
            f"msgref:v1:{digest}",
            profile_id,
            instance_id,
            int(message["message_id"]),
            fragment_ordinal,
            platform_id or route_umo.split(":", 1)[0],
            route_umo,
            platform_message_id,
            platform_reference_id,
            kind,
            projection,
            int(native_reply_supported),
            int(member_mention_supported),
            int(self_retraction_supported),
            int(returns_platform_message_id),
            _dt(accepted_at),
            _dt(deadline),
            now,
            now,
        ),
    )
