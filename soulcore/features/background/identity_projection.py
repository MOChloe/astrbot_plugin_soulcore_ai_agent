"""Request-local identity projection for background authors."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote

from ...shared.prompt_document import (
    TrustedPromptMarkup,
    join_prompt_markup,
    prompt_markup_block,
)
from ..identity import (
    PRIVATE_USER_PLACEHOLDER,
    IdentityCatalog,
    IdentityRenderContext,
    decode_model_identity_text,
    group_user_placeholder,
    internal_identity_placeholders,
)
from .domain import ForegroundContinuityMessage
from .prompt_budget import FrozenBackgroundProjection, outer_prompt_blocks
from .prompt_rendering import RECENT_FOREGROUND_BLOCK_NAME

IDENTITY_BLOCK_NAME = "身份引用"

_GROUP_USER_TEMPLATE = re.compile(r"^\{\[user:([^\]\}]+)\]\}$", re.IGNORECASE)
_DIALOGUE_PERSON = re.compile(
    r"^<对话>\n\[\[人物\]\]:\s*(C|P[1-9]\d*)\s*\n\[\[内容\]\]:",
    re.MULTILINE,
)
_PARTICIPANT_REFERENCE = re.compile(r"^P[1-9]\d*$")


@dataclass(frozen=True, slots=True)
class BackgroundIdentityMaterial:
    """Real participant identities present in one frozen author candidate."""

    participant_ids: tuple[str, ...]
    has_real_participants: bool


def background_identity_material(
    frozen: FrozenBackgroundProjection,
    *,
    additional_trusted_values: Sequence[Any] = (),
) -> BackgroundIdentityMaterial:
    participant_ids, trusted_identity_values = _foreground_identity_inputs(frozen.snapshot)
    trusted_identity_values.extend(additional_trusted_values)
    referenced_ids, has_template_participant = _template_participant_ids(trusted_identity_values)
    participant_ids.extend(referenced_ids)
    return BackgroundIdentityMaterial(
        participant_ids=tuple(dict.fromkeys(participant_ids)),
        has_real_participants=bool(participant_ids) or has_template_participant,
    )


def _foreground_identity_inputs(
    snapshot: Mapping[str, Any],
) -> tuple[list[str], list[Any]]:
    participant_ids: list[str] = []
    trusted_identity_values: list[Any] = [
        snapshot.get("world_state"),
        snapshot.get("life_state"),
        snapshot.get("story_sources"),
        snapshot.get("recent_timeline"),
        snapshot.get("character_view"),
    ]
    for message in snapshot.get("foreground_messages") or ():
        if not isinstance(message, ForegroundContinuityMessage):
            continue
        outbound_role = (
            str(message.direction).upper() == "OUTBOUND"
            and str(message.role).strip().lower() == "assistant"
        )
        participant_id = str(message.participant_id or "").strip()
        if not outbound_role and participant_id:
            participant_ids.append(participant_id)
        if outbound_role and message.internal_memo:
            trusted_identity_values.append(message.internal_memo)
    return participant_ids, trusted_identity_values


def _template_participant_ids(values: Sequence[Any]) -> tuple[list[str], bool]:
    participant_ids: list[str] = []
    has_real_participants = False
    for text in _identity_texts(values):
        for placeholder in internal_identity_placeholders(text):
            if placeholder.casefold() == PRIVATE_USER_PLACEHOLDER.casefold():
                has_real_participants = True
                continue
            match = _GROUP_USER_TEMPLATE.fullmatch(placeholder)
            if match is None:
                continue
            participant_id = unquote(match.group(1)).strip()
            if participant_id:
                participant_ids.append(participant_id)
                has_real_participants = True
    return participant_ids, has_real_participants


def participant_reference_map(
    context: IdentityRenderContext,
    catalog: IdentityCatalog,
    participant_ids: Sequence[str],
) -> Mapping[str, str]:
    result: dict[str, str] = {}
    scope = str(context.scope or "").strip().lower()
    for raw_participant_id in dict.fromkeys(str(value or "").strip() for value in participant_ids):
        if not raw_participant_id:
            continue
        placeholder = (
            PRIVATE_USER_PLACEHOLDER
            if scope == "private"
            else group_user_placeholder(raw_participant_id)
        )
        reference = next(
            (
                str(catalog.token_to_reference.get(token) or "").strip()
                for token, candidate in catalog.token_to_placeholder.items()
                if candidate == placeholder
            ),
            "",
        )
        if _PARTICIPANT_REFERENCE.fullmatch(reference):
            result[raw_participant_id] = reference
    return MappingProxyType(result)


def provisional_identity_catalog_text(
    material: BackgroundIdentityMaterial,
    catalog: IdentityCatalog,
) -> str:
    if not material.has_real_participants:
        return ""
    if not any(
        _PARTICIPANT_REFERENCE.fullmatch(str(reference or ""))
        for reference in catalog.token_to_reference.values()
    ):
        return ""
    return catalog.prompt_text()


def finalize_identity_directory(
    task_input: TrustedPromptMarkup,
    catalog: IdentityCatalog,
) -> TrustedPromptMarkup:
    """Prune the provisional directory against the final non-directory payload."""

    blocks = outer_prompt_blocks(task_input)
    directories = _named_blocks(blocks, IDENTITY_BLOCK_NAME)
    if not directories:
        return task_input
    if len(directories) != 1:
        raise ValueError("background author prompt must contain at most one identity directory")

    participant_tokens = _final_participant_tokens(blocks, catalog)
    replacement = _final_identity_directory(catalog, participant_tokens)
    return _replace_identity_directory(blocks, replacement)


def _named_blocks(
    blocks: Sequence[tuple[str, TrustedPromptMarkup]],
    name: str,
) -> tuple[TrustedPromptMarkup, ...]:
    return tuple(block for block_name, block in blocks if block_name == name)


def _final_participant_tokens(
    blocks: Sequence[tuple[str, TrustedPromptMarkup]],
    catalog: IdentityCatalog,
) -> set[str]:
    non_directory = join_prompt_markup(
        block for name, block in blocks if name != IDENTITY_BLOCK_NAME
    )
    non_directory_text = str(non_directory)
    foreground_text = "\n".join(
        str(block) for name, block in blocks if name == RECENT_FOREGROUND_BLOCK_NAME
    )
    visible_person_refs = set(_DIALOGUE_PERSON.findall(foreground_text))
    return {
        token
        for token, reference in catalog.token_to_reference.items()
        if _PARTICIPANT_REFERENCE.fullmatch(str(reference or ""))
        and (str(reference) in visible_person_refs or token in non_directory_text)
    }


def _final_identity_directory(
    catalog: IdentityCatalog,
    participant_tokens: set[str],
) -> TrustedPromptMarkup:
    if not participant_tokens:
        return TrustedPromptMarkup("")
    live_tokens = set(participant_tokens)
    live_tokens.update(
        token
        for token, reference in catalog.token_to_reference.items()
        if str(reference or "") == "C"
    )
    return prompt_markup_block(
        IDENTITY_BLOCK_NAME,
        _catalog_subset(catalog, live_tokens).prompt_text(),
    )


def _replace_identity_directory(
    blocks: Sequence[tuple[str, TrustedPromptMarkup]],
    replacement: TrustedPromptMarkup,
) -> TrustedPromptMarkup:
    rendered: list[TrustedPromptMarkup] = []
    for name, block in blocks:
        if name != IDENTITY_BLOCK_NAME:
            rendered.append(block)
        elif str(replacement).strip():
            rendered.append(replacement)
    return join_prompt_markup(rendered)


def visible_identity_catalog(
    catalog: IdentityCatalog,
    surfaces: Sequence[str],
) -> IdentityCatalog:
    """Return only exact full markers present in the final model request."""

    rendered = "\n".join(str(value or "") for value in surfaces)
    return _catalog_subset(
        catalog,
        {token for token in catalog.token_to_placeholder if token in rendered},
    )


def decode_visible_identity_data(
    value: Any,
    catalog: IdentityCatalog,
    *,
    scope: str,
) -> Any:
    """Single-pass exact-token decoding over parsed author output."""

    if isinstance(value, str):
        return str(decode_model_identity_text(value, catalog, scope=scope))
    if isinstance(value, Mapping):
        return {
            key: decode_visible_identity_data(item, catalog, scope=scope)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [decode_visible_identity_data(item, catalog, scope=scope) for item in value]
    if isinstance(value, tuple):
        return tuple(decode_visible_identity_data(item, catalog, scope=scope) for item in value)
    return value


def _catalog_subset(catalog: IdentityCatalog, tokens: set[str]) -> IdentityCatalog:
    return IdentityCatalog(
        token_to_placeholder={
            token: placeholder
            for token, placeholder in catalog.token_to_placeholder.items()
            if token in tokens
        },
        token_to_label={
            token: label for token, label in catalog.token_to_label.items() if token in tokens
        },
        token_to_reference={
            token: reference
            for token, reference in catalog.token_to_reference.items()
            if token in tokens
        },
    )


def _identity_texts(value: Any) -> tuple[str, ...]:
    result: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            result.append(item)
            return
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple, set, frozenset)):
            for nested in item:
                visit(nested)
            return
        if is_dataclass(item):
            for field in fields(item):
                visit(getattr(item, field.name))

    visit(value)
    return tuple(result)


__all__ = [
    "BackgroundIdentityMaterial",
    "IDENTITY_BLOCK_NAME",
    "background_identity_material",
    "decode_visible_identity_data",
    "finalize_identity_directory",
    "participant_reference_map",
    "provisional_identity_catalog_text",
    "visible_identity_catalog",
]
