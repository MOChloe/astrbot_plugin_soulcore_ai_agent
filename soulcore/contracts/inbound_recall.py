"""Cross-domain commit fence for a released inbound-recall grace message."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InboundRecallCommitFence:
    """Exact released grace claim that one Main Core commit must settle."""

    ledger_message_id: int
    lease_token: int
    activity_epoch: int

    def __post_init__(self) -> None:
        if self.ledger_message_id < 1 or self.lease_token < 1:
            raise ValueError("inbound-recall fence identifiers must be positive")
        if self.activity_epoch < 0:
            raise ValueError("inbound-recall fence activity epoch cannot be negative")

    def as_metadata(self) -> dict[str, int]:
        return {
            "ledger_message_id": self.ledger_message_id,
            "lease_token": self.lease_token,
            "activity_epoch": self.activity_epoch,
        }

    @classmethod
    def from_metadata(cls, value: object) -> InboundRecallCommitFence | None:
        if not isinstance(value, Mapping):
            return None
        fields = ("ledger_message_id", "lease_token", "activity_epoch")
        if any(
            isinstance(value.get(field), bool) or not isinstance(value.get(field), int)
            for field in fields
        ):
            return None
        try:
            return cls(
                ledger_message_id=int(value["ledger_message_id"]),
                lease_token=int(value["lease_token"]),
                activity_epoch=int(value["activity_epoch"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


__all__ = ["InboundRecallCommitFence"]
