from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping

from ...contracts.json import JSONObject, JSONValue

ActionHandler = Callable[[JSONObject], Mapping[str, JSONValue] | Awaitable[Mapping[str, JSONValue]]]


class AdminActionRouter:
    """Dispatch stable Page action names without coupling callbacks to the plugin class."""

    def __init__(self, handlers: Mapping[str, ActionHandler]) -> None:
        self._handlers = dict(handlers)

    async def dispatch(self, action: str, payload: JSONObject) -> Mapping[str, JSONValue]:
        handler = self._handlers.get(action)
        if handler is None:
            raise ValueError(f"unknown page action: {action}")
        result = handler(payload)
        if inspect.isawaitable(result):
            result = await result
        return result
