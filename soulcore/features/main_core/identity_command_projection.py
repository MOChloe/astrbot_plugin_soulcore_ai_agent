"""Parse one Main Core turn and apply per-parameter identity boundaries."""

from __future__ import annotations

from typing import Any

from ..ai.service import (
    MainCoreCommandRegistry,
    ParsedCommand,
    ParsedModelTurn,
    parse_model_turn,
)
from ..identity import decode_model_parameter_map


def parse_identity_turn(
    text: str,
    registry: MainCoreCommandRegistry,
    identity_catalog: Any | None,
    identity_scope: str,
    identity_context: Any | None,
) -> tuple[ParsedModelTurn, str]:
    parsed = parse_model_turn(text)
    if not parsed.valid:
        return parsed, "；".join(parsed.errors)
    if identity_catalog is None:
        return parsed, ""
    try:
        return (
            decode_identity_turn(
                parsed,
                registry,
                identity_catalog,
                identity_scope,
                identity_context,
            ),
            "",
        )
    except ValueError as exc:
        return parsed, str(exc)


def decode_identity_turn(
    parsed: ParsedModelTurn,
    registry: MainCoreCommandRegistry,
    identity_catalog: Any,
    identity_scope: str,
    identity_context: Any = None,
) -> ParsedModelTurn:
    commands = []
    for command in parsed.commands:
        parameters = decode_model_parameter_map(
            command.parameters,
            identity_catalog,
            scope=identity_scope,
            identity_context=identity_context,
            identity_modes=registry.identity_modes(command.name),
        )
        commands.append(
            ParsedCommand(
                name=command.name,
                parameters=parameters,
                ordinal=command.ordinal,
                raw_text=command.raw_text,
            )
        )
    return ParsedModelTurn(
        working_text=parsed.working_text,
        commands=tuple(commands),
        errors=parsed.errors,
        raw_text=parsed.raw_text,
    )


__all__ = ["decode_identity_turn", "parse_identity_turn"]
