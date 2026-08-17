"""Cross-domain command contract for an atomic turn-buffer gate transfer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True, slots=True)
class DeferredTurnBufferMessage:
    message_id: int
    message_ref: str
    received_at: datetime


@dataclass(frozen=True, slots=True)
class TurnBufferCommitFence:
    """Exact durable claim that authorizes one buffered Main Core commit."""

    batch_id: str
    generation: int
    activity_epoch: int
    lease_token: int
    version: int
    main_core_task_ref: str

    def __post_init__(self) -> None:
        if not self.batch_id.strip() or not self.main_core_task_ref.strip():
            raise ValueError("turn-buffer commit fence identifiers cannot be empty")
        if self.generation < 1 or self.version < 1:
            raise ValueError("turn-buffer commit fence generation/version must be positive")
        if self.activity_epoch < 0 or self.lease_token < 0:
            raise ValueError("turn-buffer commit fence epochs/tokens cannot be negative")

    def as_metadata(self) -> dict[str, str | int]:
        return {
            "batch_id": self.batch_id,
            "generation": self.generation,
            "activity_epoch": self.activity_epoch,
            "lease_token": self.lease_token,
            "version": self.version,
            "main_core_task_ref": self.main_core_task_ref,
        }

    @classmethod
    def from_metadata(cls, value: object) -> TurnBufferCommitFence | None:
        if not isinstance(value, Mapping):
            return None
        integer_fields = ("generation", "activity_epoch", "lease_token", "version")
        if any(
            isinstance(value.get(field), bool) or not isinstance(value.get(field), int)
            for field in integer_fields
        ):
            return None
        try:
            return cls(
                batch_id=str(value.get("batch_id") or ""),
                generation=int(value["generation"]),
                activity_epoch=int(value["activity_epoch"]),
                lease_token=int(value["lease_token"]),
                version=int(value["version"]),
                main_core_task_ref=str(value.get("main_core_task_ref") or ""),
            )
        except (KeyError, TypeError, ValueError):
            return None


class TurnBufferGateTransferPort(Protocol):
    async def transfer_turn_buffer_to_state_gate(
        self,
        profile_id: str,
        instance_id: str,
        batch_id: str,
        *,
        expected_generation: int,
        expected_version: int,
        lease_token: int,
        expected_activity_epoch: int,
        gate_generation: int,
        due_at: datetime,
        messages: Sequence[DeferredTurnBufferMessage],
        transferred_at: datetime,
    ) -> bool: ...


__all__ = [
    "DeferredTurnBufferMessage",
    "TurnBufferCommitFence",
    "TurnBufferGateTransferPort",
]
