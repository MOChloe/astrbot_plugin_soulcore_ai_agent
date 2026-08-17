"""Cross-domain commit fence for a foreground-claimed deferred gate batch."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeferredGateCommitFence:
    """Exact deferred batch claim that authorizes one Main Core commit."""

    batch_ref: str
    gate_generation: int
    activity_epoch: int
    version: int
    lease_token: int

    def __post_init__(self) -> None:
        if not self.batch_ref.strip():
            raise ValueError("deferred-gate fence batch_ref cannot be empty")
        if self.gate_generation < 1 or self.version < 1 or self.lease_token < 1:
            raise ValueError("deferred-gate fence generations must be positive")
        if self.activity_epoch < 0:
            raise ValueError("deferred-gate fence activity epoch cannot be negative")

    def as_metadata(self) -> dict[str, str | int]:
        return {
            "batch_ref": self.batch_ref,
            "gate_generation": self.gate_generation,
            "activity_epoch": self.activity_epoch,
            "version": self.version,
            "lease_token": self.lease_token,
        }

    @classmethod
    def from_metadata(cls, value: object) -> DeferredGateCommitFence | None:
        if not isinstance(value, Mapping):
            return None
        fields = ("gate_generation", "activity_epoch", "version", "lease_token")
        if any(
            isinstance(value.get(field), bool) or not isinstance(value.get(field), int)
            for field in fields
        ):
            return None
        try:
            return cls(
                batch_ref=str(value.get("batch_ref") or ""),
                gate_generation=int(value["gate_generation"]),
                activity_epoch=int(value["activity_epoch"]),
                version=int(value["version"]),
                lease_token=int(value["lease_token"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


__all__ = ["DeferredGateCommitFence"]
