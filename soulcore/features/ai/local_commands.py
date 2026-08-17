"""SoulCore-owned text-command collection without native function schemas."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .command_protocol_types import CommandSpec


class MainCoreCommandSet:
    def __init__(
        self,
        commands: Iterable[CommandSpec] = (),
        *,
        terminal_handler: Any = None,
        disabled_terminal_send_kinds: Iterable[str] = (),
    ) -> None:
        self.commands = tuple(commands)
        self.terminal_handler = terminal_handler
        self.disabled_terminal_send_kinds = frozenset(
            str(item or "").strip().upper()
            for item in disabled_terminal_send_kinds
            if str(item or "").strip()
        )


__all__ = ["MainCoreCommandSet"]
