"""One authoritative interpretation of ledger delivery states.

QQ adapters acknowledge platform acceptance, not a recipient read/delivery
receipt.  The product nevertheless keeps accepted outbound speech in dialogue
continuity.  Consumers must preserve that uncertainty instead of treating it
as a confirmed shared conversation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum


class DeliveryVisibility(StrEnum):
    """What a ledger row proves for dialogue-continuity consumers."""

    INBOUND = "INBOUND"
    CONFIRMED_VISIBLE = "CONFIRMED_VISIBLE"
    PLATFORM_ACCEPTED_UNCONFIRMED = "PLATFORM_ACCEPTED_UNCONFIRMED"
    NOT_VISIBLE = "NOT_VISIBLE"


CONFIRMED_OUTBOUND_STATUSES = frozenset({"SENT", "DELIVERED"})
DIALOGUE_CONTINUITY_OUTBOUND_STATUSES = (
    DeliveryVisibility.PLATFORM_ACCEPTED_UNCONFIRMED.value,
    *sorted(CONFIRMED_OUTBOUND_STATUSES),
)
FOREGROUND_DELIVERY_METADATA_KEY = "foreground_delivery"
FOREGROUND_DELIVERY_PROTOCOL_VERSION = 1
FOREGROUND_DELIVERY_BOUNDARY_PREPARED = "PREPARED"
FOREGROUND_DELIVERY_BOUNDARY_PREPARING = "PREPARING"
FOREGROUND_DELIVERY_BOUNDARY_ENTERED = "ENTERED"
_FOREGROUND_DELIVERY_BOUNDARIES = frozenset(
    {
        FOREGROUND_DELIVERY_BOUNDARY_PREPARED,
        FOREGROUND_DELIVERY_BOUNDARY_PREPARING,
        FOREGROUND_DELIVERY_BOUNDARY_ENTERED,
    }
)


def delivery_visibility(direction: str, status: str) -> DeliveryVisibility:
    """Classify exactly what a persisted message proves to later dialogue."""

    normalized_direction = str(direction or "").upper()
    normalized_status = str(status or "").upper()
    if normalized_direction == "INBOUND" and normalized_status == "RECEIVED":
        return DeliveryVisibility.INBOUND
    if normalized_direction != "OUTBOUND":
        return DeliveryVisibility.NOT_VISIBLE
    if normalized_status == DeliveryVisibility.PLATFORM_ACCEPTED_UNCONFIRMED.value:
        return DeliveryVisibility.PLATFORM_ACCEPTED_UNCONFIRMED
    if normalized_status in CONFIRMED_OUTBOUND_STATUSES:
        return DeliveryVisibility.CONFIRMED_VISIBLE
    return DeliveryVisibility.NOT_VISIBLE


def is_dialogue_continuity_visible(direction: str, status: str) -> bool:
    return delivery_visibility(direction, status) is not DeliveryVisibility.NOT_VISIBLE


def foreground_delivery_metadata(
    boundary: str,
    *,
    important_todo_ids: Iterable[str] = (),
) -> dict[str, object]:
    """Build the private ledger marker used to classify interrupted foreground sends."""

    normalized = str(boundary or "").strip().upper()
    if normalized not in _FOREGROUND_DELIVERY_BOUNDARIES:
        raise ValueError("unsupported foreground delivery boundary")
    todo_ids = tuple(
        dict.fromkeys(str(value).strip() for value in important_todo_ids if str(value).strip())
    )
    protocol: dict[str, object] = {
        "protocol_version": FOREGROUND_DELIVERY_PROTOCOL_VERSION,
        "platform_boundary": normalized,
    }
    if todo_ids:
        protocol["important_todo_ids"] = list(todo_ids)
    return {FOREGROUND_DELIVERY_METADATA_KEY: protocol}


def foreground_delivery_boundary(metadata: Mapping[str, object]) -> str | None:
    """Read a known foreground boundary, leaving legacy or malformed rows conservative."""

    protocol = metadata.get(FOREGROUND_DELIVERY_METADATA_KEY)
    if not isinstance(protocol, Mapping):
        return None
    if protocol.get("protocol_version") != FOREGROUND_DELIVERY_PROTOCOL_VERSION:
        return None
    boundary = str(protocol.get("platform_boundary") or "").strip().upper()
    return boundary if boundary in _FOREGROUND_DELIVERY_BOUNDARIES else None


def foreground_delivery_todo_ids(metadata: Mapping[str, object]) -> tuple[str, ...]:
    """Read the scoped todo ownership recorded by the current foreground protocol."""

    protocol = metadata.get(FOREGROUND_DELIVERY_METADATA_KEY)
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("protocol_version") != FOREGROUND_DELIVERY_PROTOCOL_VERSION
    ):
        return ()
    values = protocol.get("important_todo_ids")
    if not isinstance(values, list):
        return ()
    return tuple(
        dict.fromkeys(
            str(value).strip() for value in values if isinstance(value, str) and value.strip()
        )
    )


def outbox_todo_ids(payload: Mapping[str, object]) -> tuple[str, ...]:
    """Read the ordered durable file-todo references carried by an outbox payload."""

    values = payload.get("important_todo_ids")
    if not isinstance(values, list):
        return ()
    return tuple(
        dict.fromkeys(
            str(value).strip() for value in values if isinstance(value, str) and value.strip()
        )
    )


def sql_status_values(statuses: Iterable[str]) -> str:
    """Render the trusted, closed delivery-status vocabulary for a SQL IN list."""

    values = tuple(sorted({str(status).upper() for status in statuses}))
    if not values:
        raise ValueError("delivery status SQL list cannot be empty")
    return ", ".join(f"'{status}'" for status in values)


__all__ = [
    "CONFIRMED_OUTBOUND_STATUSES",
    "DIALOGUE_CONTINUITY_OUTBOUND_STATUSES",
    "DeliveryVisibility",
    "FOREGROUND_DELIVERY_BOUNDARY_ENTERED",
    "FOREGROUND_DELIVERY_BOUNDARY_PREPARED",
    "FOREGROUND_DELIVERY_BOUNDARY_PREPARING",
    "FOREGROUND_DELIVERY_METADATA_KEY",
    "FOREGROUND_DELIVERY_PROTOCOL_VERSION",
    "delivery_visibility",
    "foreground_delivery_boundary",
    "foreground_delivery_metadata",
    "foreground_delivery_todo_ids",
    "is_dialogue_continuity_visible",
    "outbox_todo_ids",
    "sql_status_values",
]
