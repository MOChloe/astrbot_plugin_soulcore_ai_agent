from __future__ import annotations

from collections.abc import Sequence

from ....contracts.delivery_visibility import (
    FOREGROUND_DELIVERY_METADATA_KEY,
    foreground_delivery_boundary,
)
from ....contracts.message_reference import inbound_reply_projection
from ..turn_buffer import TurnBufferDialogueProjection, TurnBufferMessageProjection
from .support import (
    Any,
    MessageDirection,
    _dump,
    _load,
    _parse,
    sqlite3,
)


def normalize_direction(direction: MessageDirection | str) -> str:
    normalized = (
        str(direction.value if isinstance(direction, MessageDirection) else direction)
        .strip()
        .upper()
    )
    if normalized not in {item.value for item in MessageDirection}:
        raise ValueError("direction must be INBOUND or OUTBOUND")
    return normalized


def normalize_knowledge_eligibility(value: str | None) -> str:
    normalized = str(value or "ELIGIBLE").upper()
    if normalized not in {"ELIGIBLE", "HELD", "EXCLUDED"}:
        raise ValueError("unsupported knowledge eligibility")
    return normalized


def validate_record_shape(
    *,
    direction: str,
    role: str,
    internal_memo: str,
    plain_text: str,
    components: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    delivery_status: str,
    knowledge_eligibility: str,
) -> None:
    del plain_text, components, delivery_status, knowledge_eligibility
    if internal_memo and (direction != MessageDirection.OUTBOUND.value or role != "assistant"):
        raise ValueError("only assistant platform messages may carry an internal memo")


def required_text(value: object, field: str, *, upper: bool = False) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} cannot be empty")
    return normalized.upper() if upper else normalized.lower()


def normalize_expression_link(
    batch_id: str | None,
    ordinal: int | None,
) -> tuple[str | None, int | None]:
    normalized_batch = str(batch_id).strip() if batch_id is not None else None
    normalized_batch = normalized_batch or None
    normalized_ordinal = int(ordinal) if ordinal is not None else None
    if normalized_ordinal is not None and normalized_ordinal < 0:
        raise ValueError("expression_ordinal cannot be negative")
    if normalized_ordinal is not None and normalized_batch is None:
        raise ValueError("expression_ordinal requires expression_batch_id")
    return normalized_batch, normalized_ordinal


def knowledge_reason(reason: str) -> str:
    return str(reason or "")


def transition_foreground_delivery_boundary(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    message_id: int,
    *,
    expected: str,
    target: str,
) -> bool:
    row = conn.execute(
        """SELECT direction, delivery_status, metadata_json
        FROM instance_messages
        WHERE profile_id = ? AND instance_id = ? AND message_id = ?""",
        (profile_id, instance_id, int(message_id)),
    ).fetchone()
    if (
        row is None
        or str(row["direction"]).upper() != "OUTBOUND"
        or str(row["delivery_status"]).upper() != "PENDING"
    ):
        return False
    metadata = _load(row["metadata_json"]) or {}
    if not isinstance(metadata, dict) or foreground_delivery_boundary(metadata) != expected:
        return False
    protocol = metadata.get(FOREGROUND_DELIVERY_METADATA_KEY)
    assert isinstance(protocol, dict)
    transitioned = dict(protocol)
    transitioned["platform_boundary"] = target
    metadata[FOREGROUND_DELIVERY_METADATA_KEY] = transitioned
    cursor = conn.execute(
        """UPDATE instance_messages SET metadata_json = ?
        WHERE profile_id = ? AND instance_id = ? AND message_id = ?
          AND direction = 'OUTBOUND' AND delivery_status = 'PENDING'""",
        (
            _dump(metadata),
            profile_id,
            instance_id,
            int(message_id),
        ),
    )
    return cursor.rowcount == 1


def existing_message(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    key: str | None,
) -> sqlite3.Row | None:
    if key is None:
        return None
    return conn.execute(
        """SELECT * FROM instance_messages
        WHERE profile_id = ? AND instance_id = ? AND idempotency_key = ?""",
        (profile_id, instance_id, key),
    ).fetchone()


def _classifier_media_types(components: object) -> tuple[str, ...]:
    """Collapse component payloads to a small allow-listed vocabulary."""

    result: list[str] = []
    if not isinstance(components, list):
        return ()
    for component in components:
        if not isinstance(component, dict):
            continue
        kind = str(component.get("type") or "").strip().lower()
        projected = (
            "STICKER"
            if "sticker" in kind or "emoji" in kind
            else "IMAGE"
            if "image" in kind or kind in {"photo", "picture"}
            else "FILE"
            if "file" in kind or "attachment" in kind
            else "MEDIA"
            if kind in {"audio", "record", "video"}
            else ""
        )
        if projected:
            result.append(projected)
    return tuple(result)


def turn_message_projections(
    rows: Sequence[sqlite3.Row],
) -> tuple[TurnBufferMessageProjection, ...]:
    projections: list[TurnBufferMessageProjection] = []
    for row in rows:
        occurred_at = _parse(row["occurred_at"])
        if occurred_at is None:
            continue
        components = _load(row["components_json"]) or []
        projections.append(
            TurnBufferMessageProjection(
                message_id=int(row["message_id"]),
                sender_id=str(row["sender_id"] or ""),
                sender_name=str(row["sender_name"] or ""),
                plain_text=str(row["plain_text"] or ""),
                media_types=_classifier_media_types(components),
                occurred_at=occurred_at,
                reply_reference=inbound_reply_projection(components),
            )
        )
    return tuple(projections)


def turn_buffer_dialogue_projections(
    rows: Sequence[sqlite3.Row],
) -> tuple[TurnBufferDialogueProjection, ...]:
    projections: list[TurnBufferDialogueProjection] = []
    for row in rows:
        occurred_at = _parse(row["occurred_at"])
        if occurred_at is None:
            continue
        components = _load(row["components_json"]) or []
        projections.append(
            TurnBufferDialogueProjection(
                message_id=int(row["message_id"]),
                is_character=str(row["direction"] or "").upper() == "OUTBOUND",
                sender_id=str(row["sender_id"] or ""),
                plain_text=str(row["plain_text"] or ""),
                media_types=_classifier_media_types(components),
                occurred_at=occurred_at,
            )
        )
    return tuple(projections)
