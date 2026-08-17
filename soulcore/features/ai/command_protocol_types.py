from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class CommandProtocolError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    name: str
    parameters: Mapping[str, str]
    ordinal: int
    raw_text: str
    unlabeled_content: str = ""


@dataclass(frozen=True, slots=True)
class ParsedModelTurn:
    working_text: str
    commands: tuple[ParsedCommand, ...]
    errors: tuple[str, ...] = ()
    raw_text: str = ""

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class CommandParameter:
    label: str
    internal_name: str
    required: bool = False
    choices: tuple[str, ...] = ()
    prompt_hint: str = ""
    identity_mode: str = "render"
    validator: Callable[[str, Mapping[str, Any]], str] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.identity_mode not in {"literal", "render", "template"}:
            raise ValueError(f"unknown command identity mode: {self.identity_mode}")


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    internal_name: str
    description: str
    parameters: tuple[CommandParameter, ...] = ()
    terminal: bool = False
    send_kind: str = ""
    serial: bool = False
    handler: Any = field(default=None, repr=False, compare=False)
    usage_guidance: str = ""
    prompt_visible: bool = True
    body_parameter: str = ""


@dataclass(frozen=True, slots=True)
class ValidatedCommand:
    parsed: ParsedCommand
    spec: CommandSpec
    arguments: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class CommandExecutionResult:
    ordinal: int
    command_name: str
    ok: bool
    content: str
    media_asset_ids: tuple[str, ...] = ()
    references: tuple[tuple[str, str], ...] = ()
    diagnostic: Mapping[str, Any] = field(default_factory=dict)
    public_references: tuple[str, ...] = ()
    model_input_images: tuple[str, ...] = field(default=(), repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ModelVisibleCommandResult:
    """Producer-owned natural-language projection for the next model turn."""

    content: str
    media_asset_ids: tuple[str, ...] = ()
    reference_hints: tuple[tuple[str, str], ...] = ()
    content_parts: tuple[Mapping[str, Any], ...] = field(default=(), repr=False, compare=False)


class CommandSetLike(Protocol):
    commands: Sequence[CommandSpec]
    terminal_handler: Callable[..., object] | None
    disabled_terminal_send_kinds: frozenset[str]


__all__ = [
    "CommandExecutionResult",
    "ModelVisibleCommandResult",
    "CommandParameter",
    "CommandProtocolError",
    "CommandSpec",
    "CommandSetLike",
    "ParsedCommand",
    "ParsedModelTurn",
    "ValidatedCommand",
]
