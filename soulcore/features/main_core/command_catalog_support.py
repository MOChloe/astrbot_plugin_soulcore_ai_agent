"""Small constructors shared by Main Core command catalogs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..ai.service import CommandParameter, CommandSpec
from ..identity import IDENTITY_MODES


def parameter(
    label: str,
    internal_name: str,
    *,
    required: bool = False,
    choices: Sequence[str] = (),
    prompt_hint: str = "",
    identity_mode: str = "render",
    validator: Callable[[str, Mapping[str, Any]], str] | None = None,
) -> CommandParameter:
    if identity_mode not in IDENTITY_MODES:
        raise ValueError(f"unknown command identity mode: {identity_mode}")
    return CommandParameter(
        label=label,
        internal_name=internal_name,
        required=required,
        choices=tuple(str(value) for value in choices),
        prompt_hint=str(prompt_hint or "").strip(),
        identity_mode=identity_mode,
        validator=validator,
    )


def command(
    name: str,
    internal_name: str,
    description: str,
    handler: Any,
    *parameters: CommandParameter,
    serial: bool = False,
    usage_guidance: str = "",
    prompt_visible: bool = True,
) -> CommandSpec:
    return CommandSpec(
        name=name,
        internal_name=internal_name,
        description=description,
        parameters=tuple(parameters),
        serial=serial,
        handler=handler,
        usage_guidance=usage_guidance,
        prompt_visible=prompt_visible,
    )


__all__ = ["command", "parameter"]
