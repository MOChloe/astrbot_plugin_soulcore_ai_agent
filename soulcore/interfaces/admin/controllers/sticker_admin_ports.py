"""Narrow controller-local contracts for sticker AI task administration."""

from __future__ import annotations

from typing import Any, Protocol


class StickerAITaskPort(Protocol):
    async def create_ai_task(self, *values: object, **named: object) -> Any: ...

    async def expedite_ai_task(self, *values: object, **named: object) -> Any: ...

    async def get_ai_task(self, *values: object, **named: object) -> Any: ...

    async def list_ai_tasks(self, *values: object, **named: object) -> list[Any]: ...


__all__ = ["StickerAITaskPort"]
