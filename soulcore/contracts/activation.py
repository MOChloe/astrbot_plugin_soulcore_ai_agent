"""One-way publication gate for irreversible runtime effects."""

from __future__ import annotations

import asyncio


class RuntimeActivationGate:
    """Keep provider and platform calls dormant until startup commits."""

    def __init__(self, *, active: bool = False) -> None:
        self._active = asyncio.Event()
        self._aborted = False
        if active:
            self._active.set()

    @classmethod
    def already_active(cls) -> RuntimeActivationGate:
        return cls(active=True)

    @property
    def active(self) -> bool:
        return self._active.is_set() and not self._aborted

    async def wait(self) -> None:
        await self._active.wait()
        if self._aborted:
            raise RuntimeError("SoulCore startup did not commit")

    def commit(self) -> None:
        if self._aborted:
            raise RuntimeError("cannot commit an aborted SoulCore runtime")
        self._active.set()

    def abort(self) -> None:
        self._aborted = True
        self._active.set()


__all__ = ["RuntimeActivationGate"]
