"""SoulCore-owned contracts implemented by the active platform adapter."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .capabilities import DeliveryCapability, PhysicalDeliveryReceipt
from .routes import CapturedUMO


class DeliveryStatus(StrEnum):
    ATTEMPTED_UNKNOWN = "attempted_unknown"
    PARTIALLY_ATTEMPTED = "partially_attempted"
    FAILED = "failed"
    ROUTE_NOT_READY = "route_not_ready"
    RATE_LIMITED = "rate_limited"


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    status: DeliveryStatus
    platform_found: bool
    platform_called: bool = False
    diagnostic_code: str = ""
    error_code: str = ""
    error_message: str = ""
    receipts: tuple[PhysicalDeliveryReceipt, ...] = ()

    @property
    def attempted(self) -> bool:
        return self.status in {
            DeliveryStatus.ATTEMPTED_UNKNOWN,
            DeliveryStatus.PARTIALLY_ATTEMPTED,
        }

    @classmethod
    def accepted_unconfirmed(
        cls,
        diagnostic_code: str,
        *,
        platform_found: bool = True,
        receipts: tuple[PhysicalDeliveryReceipt, ...] = (),
    ) -> DeliveryResult:
        return cls(
            DeliveryStatus.ATTEMPTED_UNKNOWN,
            platform_found,
            True,
            diagnostic_code=str(diagnostic_code),
            receipts=tuple(receipts),
        )

    @classmethod
    def partially_attempted(
        cls,
        diagnostic_code: str,
        *,
        platform_found: bool = True,
        receipts: tuple[PhysicalDeliveryReceipt, ...] = (),
    ) -> DeliveryResult:
        return cls(
            DeliveryStatus.PARTIALLY_ATTEMPTED,
            platform_found,
            True,
            diagnostic_code=str(diagnostic_code),
            receipts=tuple(receipts),
        )

    @classmethod
    def deferred(
        cls,
        status: DeliveryStatus,
        diagnostic_code: str,
        *,
        platform_found: bool = True,
    ) -> DeliveryResult:
        if status not in {DeliveryStatus.ROUTE_NOT_READY, DeliveryStatus.RATE_LIMITED}:
            raise ValueError("deferred delivery result requires a waiting status")
        return cls(status, platform_found, diagnostic_code=str(diagnostic_code))

    @classmethod
    def failed(
        cls,
        error_code: str,
        *,
        error_message: str = "",
        platform_found: bool,
        platform_called: bool = False,
    ) -> DeliveryResult:
        raw_code = str(error_code or "").strip()
        stable_code = raw_code.split(":", 1)[0]
        return cls(
            DeliveryStatus.FAILED,
            platform_found,
            platform_called,
            error_code=stable_code or "delivery_failed",
            error_message=str(error_message or raw_code or "delivery_failed"),
        )


class SelfRetractionStatus(StrEnum):
    RETRACTED = "retracted"
    ATTEMPTED_UNKNOWN = "attempted_unknown"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SelfRetractionResult:
    status: SelfRetractionStatus
    detail: str
    platform_found: bool

    @property
    def retracted(self) -> bool:
        return self.status is SelfRetractionStatus.RETRACTED

    @property
    def attempted(self) -> bool:
        return self.status in {
            SelfRetractionStatus.RETRACTED,
            SelfRetractionStatus.ATTEMPTED_UNKNOWN,
        }


@dataclass(frozen=True, slots=True)
class DeliveryMessage:
    """Platform-neutral physical message assembled by SoulCore."""

    content: str = ""
    components: tuple[dict[str, object], ...] = ()


class DeliveryTransportPort(Protocol):
    def capability_for(
        self, umo: str | CapturedUMO, **values: object
    ) -> DeliveryCapability | None: ...

    def route_ready(self, umo: str | CapturedUMO) -> bool: ...

    def voice_ready(self, umo: str | CapturedUMO) -> bool: ...

    async def send(
        self, umo: str | CapturedUMO, message: DeliveryMessage, **values: object
    ) -> DeliveryResult: ...

    async def retract_self(
        self, umo: str | CapturedUMO, platform_message_id: str
    ) -> SelfRetractionResult: ...


__all__ = [
    "DeliveryResult",
    "DeliveryMessage",
    "DeliveryStatus",
    "DeliveryTransportPort",
    "SelfRetractionResult",
    "SelfRetractionStatus",
]
