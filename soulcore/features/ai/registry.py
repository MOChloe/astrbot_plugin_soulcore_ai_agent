"""AI backend/capability registries and circuit state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ...contracts.ai_models import (
    AIBackendAdapter,
    AIBackendCircuit,
    AIBackendDescriptor,
    AIBackendHealth,
    AIBackendState,
    AICapabilityAdapter,
    AIErrorCode,
    AIErrorInfo,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class BackendRegistration:
    descriptor: AIBackendDescriptor
    adapter: AIBackendAdapter


class BackendPool:
    def __init__(self) -> None:
        self._items: dict[str, BackendRegistration] = {}

    def register(
        self,
        descriptor: AIBackendDescriptor,
        adapter: AIBackendAdapter,
    ) -> None:
        backend_id = str(descriptor.backend_id or "").strip()
        if not backend_id:
            raise ValueError("backend_id is required")
        if str(adapter.adapter_id) != descriptor.adapter_id:
            raise ValueError("backend adapter_id does not match its descriptor")
        self._items[backend_id] = BackendRegistration(descriptor, adapter)

    def get(self, backend_id: str) -> BackendRegistration | None:
        return self._items.get(str(backend_id))

    def candidates(self, backend_ids: Sequence[str] = ()) -> list[BackendRegistration]:
        if backend_ids:
            values = [self._items[value] for value in backend_ids if value in self._items]
        else:
            values = list(self._items.values())
        return sorted(
            (value for value in values if value.descriptor.enabled),
            key=lambda value: (value.descriptor.priority, value.descriptor.backend_id),
        )

    def list(self) -> list[AIBackendDescriptor]:
        return [value.descriptor for value in self.candidates()]


@dataclass(frozen=True, slots=True)
class CircuitPolicy:
    transient_failure_threshold: int = 3
    transient_open_seconds: float = 120.0
    immediate_open_seconds: float = 5 * 60 * 60


class CircuitBreaker:
    def __init__(self, policy: CircuitPolicy | None = None) -> None:
        self.policy = policy or CircuitPolicy()
        self._circuits: dict[str, AIBackendCircuit] = {}
        self._health: dict[str, AIBackendHealth] = {}
        self._half_open_inflight: set[str] = set()

    def allow(self, backend_id: str, *, now: datetime | None = None) -> bool:
        current = now or _utcnow()
        circuit = self._circuits.get(backend_id)
        if circuit is None or circuit.state in {
            AIBackendState.HEALTHY,
            AIBackendState.DEGRADED,
        }:
            return True
        if circuit.state is AIBackendState.HALF_OPEN:
            if backend_id in self._half_open_inflight:
                return False
            self._half_open_inflight.add(backend_id)
            return True
        if circuit.state is AIBackendState.DISABLED:
            return False
        if circuit.opened_until is not None and circuit.opened_until <= current:
            circuit.state = AIBackendState.HALF_OPEN
            circuit.updated_at = current
            self._health_for(backend_id).state = AIBackendState.HALF_OPEN
            self._half_open_inflight.add(backend_id)
            return True
        return False

    def can_attempt(self, backend_id: str, *, now: datetime | None = None) -> bool:
        """Inspect whether ``allow`` could grant a probe without reserving it."""

        current = now or _utcnow()
        circuit = self._circuits.get(backend_id)
        if circuit is None or circuit.state in {
            AIBackendState.HEALTHY,
            AIBackendState.DEGRADED,
        }:
            return True
        if circuit.state is AIBackendState.HALF_OPEN:
            return backend_id not in self._half_open_inflight
        if circuit.state is AIBackendState.DISABLED:
            return False
        return bool(circuit.opened_until is not None and circuit.opened_until <= current)

    def record_success(self, backend_id: str) -> AIBackendHealth:
        now = _utcnow()
        circuit = self._circuits.setdefault(backend_id, AIBackendCircuit(backend_id=backend_id))
        circuit.state = AIBackendState.HEALTHY
        circuit.failure_count = 0
        circuit.opened_until = None
        circuit.last_error_code = ""
        circuit.updated_at = now
        self._half_open_inflight.discard(backend_id)
        health = self._health_for(backend_id)
        health.state = AIBackendState.HEALTHY
        health.success_count += 1
        health.last_success_at = now
        health.circuit = circuit
        health.updated_at = now
        return health

    def record_failure(self, backend_id: str, error: AIErrorInfo) -> AIBackendHealth:
        now = _utcnow()
        circuit = self._circuits.setdefault(backend_id, AIBackendCircuit(backend_id=backend_id))
        circuit.failure_count += 1
        circuit.last_error_code = error.code.value
        immediate = error.open_circuit or error.code in {
            AIErrorCode.AUTHENTICATION,
            AIErrorCode.PERMISSION,
            AIErrorCode.QUOTA_EXHAUSTED,
            AIErrorCode.RATE_LIMIT,
        }
        if immediate or circuit.failure_count >= max(
            1, int(self.policy.transient_failure_threshold)
        ):
            duration = (
                error.retry_after_seconds
                if error.retry_after_seconds is not None
                else (
                    self.policy.immediate_open_seconds
                    if immediate
                    else self.policy.transient_open_seconds
                )
            )
            circuit.state = AIBackendState.OPEN
            circuit.opened_until = now + timedelta(seconds=max(1.0, float(duration)))
        else:
            circuit.state = AIBackendState.DEGRADED
        circuit.updated_at = now
        self._half_open_inflight.discard(backend_id)
        health = self._health_for(backend_id)
        health.state = circuit.state
        health.failure_count += 1
        health.last_failure_at = now
        health.last_error_code = error.code.value
        health.circuit = circuit
        health.updated_at = now
        return health

    def get(self, backend_id: str) -> AIBackendHealth:
        return self._health_for(backend_id)

    def force_half_open(self, backend_id: str) -> AIBackendHealth:
        """Open exactly one probe slot, even when the normal cooldown is active."""

        now = _utcnow()
        circuit = self._circuits.setdefault(backend_id, AIBackendCircuit(backend_id=backend_id))
        circuit.state = AIBackendState.HALF_OPEN
        circuit.opened_until = None
        circuit.updated_at = now
        self._half_open_inflight.discard(backend_id)
        health = self._health_for(backend_id)
        health.state = AIBackendState.HALF_OPEN
        health.circuit = circuit
        health.updated_at = now
        return health

    def release_probe(self, backend_id: str) -> None:
        self._half_open_inflight.discard(backend_id)

    def list(self) -> list[AIBackendHealth]:
        return [self._health[key] for key in sorted(self._health)]

    def restore(
        self,
        backend_id: str,
        *,
        state: str,
        failure_count: int = 0,
        opened_until: datetime | None = None,
        last_error_code: str = "",
    ) -> None:
        normalized = str(state or "CLOSED").upper()
        mapped = {
            "CLOSED": AIBackendState.HEALTHY,
            "HEALTHY": AIBackendState.HEALTHY,
            "DEGRADED": AIBackendState.DEGRADED,
            "OPEN": AIBackendState.OPEN,
            "HALF_OPEN": AIBackendState.HALF_OPEN,
            "DISABLED": AIBackendState.DISABLED,
        }.get(normalized, AIBackendState.DEGRADED)
        circuit = AIBackendCircuit(
            backend_id=backend_id,
            state=mapped,
            failure_count=max(0, int(failure_count)),
            opened_until=opened_until,
            last_error_code=str(last_error_code or ""),
            updated_at=_utcnow(),
        )
        self._circuits[backend_id] = circuit
        health = self._health_for(backend_id)
        health.state = mapped
        health.failure_count = max(0, int(failure_count))
        health.last_error_code = str(last_error_code or "")
        health.circuit = circuit
        health.updated_at = _utcnow()

    def _health_for(self, backend_id: str) -> AIBackendHealth:
        return self._health.setdefault(
            backend_id,
            AIBackendHealth(
                backend_id=backend_id,
                circuit=self._circuits.get(backend_id),
            ),
        )


@dataclass(frozen=True, slots=True)
class CapabilityRegistration:
    descriptor: AIBackendDescriptor
    adapter: AICapabilityAdapter


class CapabilityAdapterRegistry:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], CapabilityRegistration] = {}

    def register(
        self,
        descriptor: AIBackendDescriptor,
        adapter: AICapabilityAdapter,
    ) -> None:
        if descriptor.adapter_id != str(adapter.adapter_id):
            raise ValueError("capability adapter_id does not match backend descriptor")
        for capability in adapter.capabilities:
            value = str(capability or "").strip()
            if value:
                self._items[(value, descriptor.backend_id)] = CapabilityRegistration(
                    descriptor, adapter
                )

    def candidates(
        self, capability: str, backend_ids: Sequence[str] = ()
    ) -> list[CapabilityRegistration]:
        allowed = set(backend_ids)
        values = [
            registration
            for (registered_capability, backend_id), registration in self._items.items()
            if registered_capability == capability
            and (not allowed or backend_id in allowed)
            and registration.descriptor.enabled
        ]
        return sorted(
            values,
            key=lambda value: (
                value.descriptor.priority,
                value.descriptor.backend_id,
            ),
        )


__all__ = [
    "BackendPool",
    "BackendRegistration",
    "CapabilityAdapterRegistry",
    "CapabilityRegistration",
    "CircuitBreaker",
    "CircuitPolicy",
]
