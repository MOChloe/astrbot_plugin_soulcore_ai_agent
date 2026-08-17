"""Pure contracts for SoulCore's trusted identity-template language."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import NewType
from urllib.parse import quote, unquote

from ...contracts.message_reference import safe_model_identity
from ...shared.identity_syntax import escape_untrusted_identity_syntax

CHARACTER_PLACEHOLDER = "{[character]}"
PRIVATE_USER_PLACEHOLDER = "{[User]}"
IdentityTemplate = NewType("IdentityTemplate", str)

_INTERNAL_PATTERN = re.compile(
    r"\{\[(character|user(?::([^\]\}]+))?)\]\}",
    re.IGNORECASE,
)
_PERSON_REFERENCE_PATTERN = re.compile(r"[A-Z][1-9]\d*\Z")
_GENERIC_PRIVATE_DISPLAY_NAMES = frozenset({"", "对方", "当前对方", "当前好友", "一位联系人"})
_MODEL_NAME_TRANSLATION = str.maketrans(
    {
        "{": "｛",
        "}": "｝",
        "[": "［",
        "]": "］",
        "#": "＃",
        "@": "＠",
        ":": "：",
    }
)

IDENTITY_MODE_LITERAL = "literal"
IDENTITY_MODE_RENDER = "render"
IDENTITY_MODE_TEMPLATE = "template"
IDENTITY_MODES = frozenset({IDENTITY_MODE_LITERAL, IDENTITY_MODE_RENDER, IDENTITY_MODE_TEMPLATE})

IDENTITY_REFERENCE_GUIDANCE = (
    "C、P1、P2 等人物引用用于区分发言人。写下的内容需要明确指向下列现实聊天身份时，"
    "逐字使用对应的完整身份标记；虚构人物和普通名称直接写文字。"
)


@dataclass(frozen=True, slots=True)
class IdentityParticipant:
    participant_id: str
    display_name: str
    name_source: str = "OBSERVED"
    last_message_id: int | None = None


@dataclass(frozen=True, slots=True)
class IdentityRenderContext:
    profile_id: str
    instance_id: str
    scope: str
    character_name: str
    participants: tuple[IdentityParticipant, ...] = ()
    private_fallback_player_name: str = ""
    private_name_override_enabled: bool = False

    @property
    def participant_by_id(self) -> dict[str, IdentityParticipant]:
        return {item.participant_id: item for item in self.participants}

    @property
    def private_display_name(self) -> str:
        observed = self.participants[0].display_name if self.participants else ""
        return resolve_private_display_name(
            observed,
            self.private_fallback_player_name,
            override_enabled=self.private_name_override_enabled,
        )


@dataclass(frozen=True, slots=True)
class IdentityCatalog:
    token_to_placeholder: dict[str, str]
    token_to_label: dict[str, str]
    token_to_reference: dict[str, str] = field(default_factory=dict)

    def prompt_text(self) -> str:
        rows = "\n".join(
            f"{self._prompt_row_reference(token)}：{self.token_to_label[token]}"
            for token in self.token_to_placeholder
        )
        reference_note = (
            "\n对话行中的 [C]、[P1] 等人物引用与下方同一行的身份标记指向同一人；"
            "人物引用用于判断发言人，完整身份标记用于在写下的内容中稳定指向相应人物。"
            if self.token_to_reference
            else ""
        )
        return f"{IDENTITY_REFERENCE_GUIDANCE}{reference_note}\n\n{rows}".strip()

    def _prompt_row_reference(self, token: str) -> str:
        reference = str(self.token_to_reference.get(token) or "").strip()
        return f"{reference} / {token}" if reference else token

    @property
    def group_participant_references(self) -> tuple[str, ...]:
        return tuple(
            reference
            for token, reference in self.token_to_reference.items()
            if reference != "C"
            and str(self.token_to_placeholder.get(token) or "").startswith("{[User:")
        )

    def group_participant_reference(self, participant_id: str) -> str:
        placeholder = group_user_placeholder(participant_id)
        for token, candidate in self.token_to_placeholder.items():
            if candidate != placeholder:
                continue
            return str(self.token_to_reference.get(token) or "")
        return ""


@dataclass(frozen=True, slots=True)
class IdentityAnnotationCandidate:
    candidate_id: str
    field_path: str
    start: int
    end: int
    matched_text: str
    replacement: str
    context: str


def group_user_placeholder(participant_id: str) -> str:
    normalized = str(participant_id or "").strip()
    if not normalized:
        raise ValueError("group identity requires a platform participant id")
    encoded = quote(normalized, safe="A-Za-z0-9._~:-")
    return f"{{[User:{encoded}]}}"


def resolve_private_display_name(
    observed: str,
    fallback: str = "",
    *,
    override_enabled: bool = False,
) -> str:
    """Resolve one private-chat name using the configured precedence."""

    fallback_name = safe_model_identity(fallback)
    if override_enabled and fallback_name:
        return fallback_name
    observed_name = safe_model_identity(observed)
    if observed_name not in _GENERIC_PRIVATE_DISPLAY_NAMES:
        return observed_name
    return fallback_name or "对方"


def internal_identity_placeholders(value: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in _INTERNAL_PATTERN.finditer(str(value or "")))


def validate_identity_template(value: str, *, scope: str) -> IdentityTemplate:
    normalized_scope = str(scope or "").strip().lower()
    if normalized_scope not in {"profile", "private", "group"}:
        raise ValueError("identity template scope must be profile, private, or group")

    def normalize(match: re.Match[str]) -> str:
        kind, encoded = match.group(1).casefold(), match.group(2)
        if kind == "character":
            return CHARACTER_PLACEHOLDER
        if normalized_scope == "profile":
            raise ValueError("profile-scoped text cannot reference a conversation participant")
        if normalized_scope == "private" and encoded is not None:
            raise ValueError("private-scoped text must use {[User]}")
        if normalized_scope == "group" and encoded is None:
            raise ValueError("group-scoped text must use a platform-bound user identity")
        if encoded is not None and not unquote(encoded).strip():
            raise ValueError("group identity placeholder is empty")
        return (
            PRIVATE_USER_PLACEHOLDER
            if encoded is None
            else group_user_placeholder(unquote(encoded))
        )

    return IdentityTemplate(_INTERNAL_PATTERN.sub(normalize, str(value or "")))


def render_identity_text(value: str, context: IdentityRenderContext) -> str:
    participants = context.participant_by_id

    def replace(match: re.Match[str]) -> str:
        kind, encoded = match.group(1).casefold(), match.group(2)
        if kind == "character":
            return context.character_name or "角色本人"
        if encoded is None:
            return context.private_display_name if context.scope == "private" else "一位群成员"
        participant = participants.get(unquote(encoded))
        if participant is None:
            return "一位群成员"
        return safe_model_identity(participant.display_name) or "一位群成员"

    return _INTERNAL_PATTERN.sub(replace, str(value or ""))


def encode_identity_template_for_model(value: str, catalog: IdentityCatalog) -> str:
    """Project internal templates to readable, run-scoped model identity marks."""

    placeholder_to_token = {
        placeholder: token for token, placeholder in catalog.token_to_placeholder.items()
    }

    def replace(match: re.Match[str]) -> str:
        placeholder = str(
            validate_identity_template(match.group(0), scope=_placeholder_scope(match))
        )
        token = placeholder_to_token.get(placeholder)
        if token is not None:
            return token
        kind, encoded = match.group(1).casefold(), match.group(2)
        if kind == "character":
            return "角色本人"
        if encoded is None:
            return "对方"
        return "一位群成员"

    return _INTERNAL_PATTERN.sub(replace, str(value or ""))


def build_identity_catalog(
    context: IdentityRenderContext,
    *,
    participant_ids: tuple[str, ...] | None = None,
    participant_references: Mapping[str, str] | None = None,
) -> IdentityCatalog:
    rows = _identity_catalog_rows(context, participant_ids, participant_references)
    projected = _project_identity_catalog_rows(rows)
    return IdentityCatalog(
        token_to_placeholder={
            token: placeholder for _name, token, placeholder, _label in projected
        },
        token_to_label={token: label for _name, token, _placeholder, label in projected},
        token_to_reference={
            projected_row[1]: source_row[1]
            for source_row, projected_row in zip(rows, projected, strict=True)
        },
    )


def _identity_catalog_rows(
    context: IdentityRenderContext,
    participant_ids: tuple[str, ...] | None,
    participant_references: Mapping[str, str] | None,
) -> tuple[tuple[str, str, str, str], ...]:
    selected = _selected_identity_participants(context, participant_ids)
    if context.scope == "private" and not selected:
        selected.append(IdentityParticipant("", context.private_display_name or "对方"))
    rows = [(context.character_name or "角色本人", "C", CHARACTER_PLACEHOLDER, "你本人")]
    rows.extend(_participant_catalog_rows(context, selected, participant_references))
    return tuple(rows)


def _selected_identity_participants(
    context: IdentityRenderContext,
    participant_ids: tuple[str, ...] | None,
) -> list[IdentityParticipant]:
    if participant_ids is None:
        return list(context.participants)
    by_id = context.participant_by_id
    selected: list[IdentityParticipant] = []
    for index, participant_id in enumerate(dict.fromkeys(participant_ids), start=1):
        participant = by_id.get(participant_id)
        if participant is not None:
            selected.append(participant)
        elif context.scope == "group":
            selected.append(IdentityParticipant(participant_id, f"群成员{index}"))
    return selected


def _participant_catalog_rows(
    context: IdentityRenderContext,
    selected: Sequence[IdentityParticipant],
    participant_references: Mapping[str, str] | None,
) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    used_references = {"C"}
    next_fallback = 1
    for index, item in enumerate(selected, start=1):
        placeholder = (
            PRIVATE_USER_PLACEHOLDER
            if context.scope == "private"
            else group_user_placeholder(item.participant_id)
        )
        label = "当前对方" if context.scope == "private" else "当前群成员"
        preferred = str((participant_references or {}).get(item.participant_id) or "").strip()
        reference, next_fallback = _allocate_person_reference(
            preferred,
            used_references,
            next_fallback,
        )
        used_references.add(reference)
        fallback = "对方" if context.scope == "private" else f"群成员{index}"
        display_name = (
            context.private_display_name
            if context.scope == "private"
            else safe_model_identity(item.display_name) or fallback
        )
        rows.append((display_name, reference, placeholder, label))
    return rows


def _allocate_person_reference(
    preferred: str,
    used_references: set[str],
    next_fallback: int,
) -> tuple[str, int]:
    if _PERSON_REFERENCE_PATTERN.fullmatch(preferred) and preferred not in used_references:
        return preferred, next_fallback
    while True:
        candidate = f"P{next_fallback}"
        next_fallback += 1
        if candidate not in used_references:
            return candidate, next_fallback


def _project_identity_catalog_rows(
    rows: tuple[tuple[str, str, str, str], ...],
) -> tuple[tuple[str, str, str, str], ...]:
    names = tuple(_model_identity_name(name, label) for name, _suffix, _placeholder, label in rows)
    counts = Counter(name.casefold() for name in names)
    return tuple(
        (
            name,
            _catalog_model_token(
                model_name,
                suffix,
                force_suffix=(
                    counts[model_name.casefold()] > 1 or _internal_model_token_conflict(model_name)
                ),
            ),
            placeholder,
            label,
        )
        for (name, suffix, placeholder, label), model_name in zip(rows, names, strict=True)
    )


def _model_identity_name(name: str, label: str) -> str:
    fallback = "角色" if label == "你本人" else "对方" if label == "当前对方" else "群成员"
    normalized = safe_model_identity(str(name or "")) or fallback
    return normalized.translate(_MODEL_NAME_TRANSLATION) or fallback


def _catalog_model_token(name: str, suffix: str, *, force_suffix: bool) -> str:
    qualifier = f"#{suffix}" if force_suffix else ""
    return f"{{[{name}{qualifier}]}}"


def _internal_model_token_conflict(name: str) -> bool:
    return _INTERNAL_PATTERN.fullmatch(f"{{[{name}]}}") is not None


def decode_model_identity_text(
    value: str,
    catalog: IdentityCatalog,
    *,
    scope: str,
) -> IdentityTemplate:
    """Decode explicit model identity marks without interpreting ordinary names."""

    del scope
    text = str(value or "")
    tokens = sorted(
        (token for token in catalog.token_to_placeholder if token),
        key=len,
        reverse=True,
    )
    if not tokens:
        return IdentityTemplate(text)
    pattern = re.compile("|".join(re.escape(token) for token in tokens))
    return IdentityTemplate(
        pattern.sub(
            lambda match: catalog.token_to_placeholder[match.group(0)],
            text,
        )
    )


def render_model_identity_text(
    value: str,
    catalog: IdentityCatalog,
    context: IdentityRenderContext,
    *,
    scope: str,
) -> str:
    """Resolve explicit model identity marks for one transient consumer."""

    template = decode_model_identity_text(value, catalog, scope=scope)
    return render_identity_text(str(template), context)


def project_identity_text_for_model(
    value: str,
    catalog: IdentityCatalog,
    *,
    scope: str,
) -> str:
    """Project trusted internal templates to readable model identity marks."""

    del scope
    return encode_identity_template_for_model(value, catalog)


def decode_model_parameter_map(
    parameters: Mapping[str, str],
    catalog: IdentityCatalog,
    *,
    scope: str,
    identity_context: IdentityRenderContext | None = None,
    identity_modes: Mapping[str, str] | None = None,
    evidence_labels: tuple[str, ...] = ("证据", "原文", "引用"),
) -> dict[str, str]:
    """Normalize command fields according to their real downstream data boundary."""

    result: dict[str, str] = {}
    modes = {str(label): str(mode) for label, mode in (identity_modes or {}).items()}
    for label, value in parameters.items():
        evidence_field = any(marker in str(label) for marker in evidence_labels)
        mode = modes.get(
            str(label),
            IDENTITY_MODE_LITERAL if evidence_field else IDENTITY_MODE_TEMPLATE,
        )
        if mode not in IDENTITY_MODES:
            raise ValueError(f"unknown identity parameter mode: {mode}")
        raw = str(value or "")
        if mode == IDENTITY_MODE_LITERAL:
            result[str(label)] = raw
        elif mode == IDENTITY_MODE_RENDER and identity_context is not None:
            result[str(label)] = render_model_identity_text(
                raw,
                catalog,
                identity_context,
                scope=scope,
            )
        elif mode == IDENTITY_MODE_RENDER:
            result[str(label)] = raw
        else:
            result[str(label)] = str(decode_model_identity_text(raw, catalog, scope=scope))
    return result


def _placeholder_scope(match: re.Match[str]) -> str:
    kind, encoded = match.group(1).casefold(), match.group(2)
    if kind == "character":
        return "profile"
    return "private" if encoded is None else "group"


def identity_annotation_candidates(
    value: str,
    *,
    field_path: str,
    identities: tuple[tuple[str, str], ...],
) -> tuple[IdentityAnnotationCandidate, ...]:
    text = str(value or "")
    result: list[IdentityAnnotationCandidate] = []
    occupied: set[int] = set()
    for name, replacement in sorted(identities, key=lambda item: len(item[0]), reverse=True):
        if not name:
            continue
        for match in re.finditer(re.escape(name), text):
            if any(index in occupied for index in range(match.start(), match.end())):
                continue
            occupied.update(range(match.start(), match.end()))
            start, end = max(0, match.start() - 24), min(len(text), match.end() + 24)
            candidate_key = f"{field_path}\0{match.start()}\0{match.end()}\0{replacement}"
            result.append(
                IdentityAnnotationCandidate(
                    candidate_id="identity-"
                    + sha256(candidate_key.encode("utf-8")).hexdigest()[:20],
                    field_path=field_path,
                    start=match.start(),
                    end=match.end(),
                    matched_text=match.group(0),
                    replacement=replacement,
                    context=text[start:end],
                )
            )
    return tuple(sorted(result, key=lambda item: (item.start, item.end)))


__all__ = [
    "CHARACTER_PLACEHOLDER",
    "IDENTITY_REFERENCE_GUIDANCE",
    "IDENTITY_MODE_LITERAL",
    "IDENTITY_MODES",
    "IDENTITY_MODE_RENDER",
    "IDENTITY_MODE_TEMPLATE",
    "PRIVATE_USER_PLACEHOLDER",
    "IdentityAnnotationCandidate",
    "IdentityCatalog",
    "IdentityParticipant",
    "IdentityRenderContext",
    "resolve_private_display_name",
    "IdentityTemplate",
    "build_identity_catalog",
    "decode_model_identity_text",
    "decode_model_parameter_map",
    "encode_identity_template_for_model",
    "escape_untrusted_identity_syntax",
    "group_user_placeholder",
    "identity_annotation_candidates",
    "internal_identity_placeholders",
    "project_identity_text_for_model",
    "render_identity_text",
    "render_model_identity_text",
    "validate_identity_template",
]
