"""Shared invocation contract for Main Core consumers."""

from typing import Protocol

from .models import CoreRunResult, CoreWakeRequest


class MainCoreHandlePort(Protocol):
    async def handle(
        self,
        request: CoreWakeRequest,
        **values: object,
    ) -> CoreRunResult: ...


__all__ = ["MainCoreHandlePort"]
