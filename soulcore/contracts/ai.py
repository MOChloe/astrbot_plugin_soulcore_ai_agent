from __future__ import annotations

from typing import Protocol


class TaskExecutor(Protocol):
    async def __call__(
        self,
        task: dict[str, object],
        control: object,
    ) -> dict[str, object] | None: ...
