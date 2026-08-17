"""Public persistence boundary for portable role packages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from .domain import (
    ApplyResult,
    ImportState,
    PortraitMutation,
    RoleDatabaseSnapshot,
)


class RolePackageRepositoryPort(Protocol):
    async def snapshot(self, profile_id: str) -> RoleDatabaseSnapshot: ...

    async def apply(
        self,
        *,
        profile_id: str,
        expected: RoleDatabaseSnapshot,
        state: ImportState,
        portrait_mutations: Mapping[str, PortraitMutation],
        package_sha256: str,
        idempotency_key: str,
    ) -> ApplyResult: ...


__all__ = ["RolePackageRepositoryPort"]
