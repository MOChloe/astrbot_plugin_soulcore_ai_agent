"""Bounded public-character draft used by guided Persona import."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..identity import CHARACTER_PLACEHOLDER
from .domain import (
    CapabilityProfile,
    CharacterIdentity,
    CharacterModel,
    LanguageProfile,
    PersonalityProfile,
    PreferenceProfile,
    SocialProfile,
    VisualProfile,
    model_to_payload,
    normalize_character_model,
)

PUBLIC_CHARACTER_FIELDS: dict[str, tuple[str, ...]] = {
    "identity": ("name", "aliases", "overview", "facts"),
    "personality": (
        "traits_and_values",
        "thinking_and_behavior",
        "habits_and_emotions",
    ),
    "social": ("interaction_style", "boundaries"),
    "preferences": ("likes_and_interests", "dislikes"),
    "language": ("speaking_style", "messaging_habits", "address_habits"),
    "visual": ("appearance", "clothing", "visual_boundaries"),
    "capabilities": ("abilities", "knowledge_scope", "limitations"),
}
PUBLIC_CHARACTER_TEXT_FIELDS = frozenset(
    {
        ("identity", "name"),
        ("identity", "overview"),
    }
)
_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def empty_public_character_sections() -> dict[str, dict[str, Any]]:
    return {
        section: {
            field: "" if (section, field) in PUBLIC_CHARACTER_TEXT_FIELDS else []
            for field in fields
        }
        for section, fields in PUBLIC_CHARACTER_FIELDS.items()
    }


def public_character_sections(value: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Project a complete character-model payload into the simple editor shape."""

    source = value if isinstance(value, Mapping) else {}
    result = empty_public_character_sections()
    identity = source.get("identity")
    character_name = (
        str(identity.get("name") or "").strip() if isinstance(identity, Mapping) else ""
    )

    def display_text(raw: Any) -> str:
        return str(raw or "").replace(CHARACTER_PLACEHOLDER, character_name or "角色")

    for section, fields in PUBLIC_CHARACTER_FIELDS.items():
        raw_section = source.get(section)
        if not isinstance(raw_section, Mapping):
            continue
        for field in fields:
            raw = raw_section.get(field)
            if (section, field) in PUBLIC_CHARACTER_TEXT_FIELDS:
                result[section][field] = display_text(raw)
            elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                result[section][field] = [display_text(item) for item in raw]
    return result


def parse_generated_public_character(text: str) -> dict[str, dict[str, Any]]:
    """Parse one model response and normalize it through the character domain."""

    value = _generated_character_json(text)
    sections = {
        section: _generated_character_section(section, fields, value.get(section, {}))
        for section, fields in PUBLIC_CHARACTER_FIELDS.items()
    }
    normalized = normalize_character_model(_character_model_from_public_sections(sections))
    return public_character_sections(model_to_payload(normalized))


def _generated_character_json(text: str) -> Mapping[str, Any]:
    raw = str(text or "").strip()
    fenced = _FENCED_JSON.fullmatch(raw)
    if fenced is not None:
        raw = fenced.group(1).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("输出必须是一个完整的 JSON 对象，不能包含说明文字") from exc
    if not isinstance(value, Mapping):
        raise ValueError("输出根节点必须是 JSON 对象")
    unknown_sections = sorted(str(key) for key in value if key not in PUBLIC_CHARACTER_FIELDS)
    if unknown_sections:
        raise ValueError(f"包含不支持的角色资料分组：{', '.join(unknown_sections)}")
    return value


def _generated_character_section(section: str, fields: Sequence[str], value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{section} 必须是 JSON 对象")
    unknown_fields = sorted(str(key) for key in value if key not in fields)
    if unknown_fields:
        raise ValueError(f"{section} 包含不支持的字段：{', '.join(unknown_fields)}")
    result = empty_public_character_sections()[section]
    for field in fields:
        if field in value:
            result[field] = _generated_character_field(section, field, value[field])
    return result


def _generated_character_field(section: str, field: str, value: Any) -> Any:
    path = f"{section}.{field}"
    if (section, field) in PUBLIC_CHARACTER_TEXT_FIELDS:
        if not isinstance(value, str):
            raise ValueError(f"{path} 必须是字符串")
        _reject_internal_marker(value, path)
        return value
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} 必须是字符串数组")
    if any(not isinstance(child, str) for child in value):
        raise ValueError(f"{path} 只能包含字符串")
    for index, child in enumerate(value):
        _reject_internal_marker(child, f"{path}[{index}]")
    return list(value)


def _character_model_from_public_sections(
    sections: Mapping[str, Mapping[str, Any]],
) -> CharacterModel:
    return CharacterModel(
        identity=CharacterIdentity(
            name=sections["identity"]["name"],
            aliases=tuple(sections["identity"]["aliases"]),
            overview=sections["identity"]["overview"],
            facts=tuple(sections["identity"]["facts"]),
        ),
        personality=PersonalityProfile(
            traits_and_values=tuple(sections["personality"]["traits_and_values"]),
            thinking_and_behavior=tuple(sections["personality"]["thinking_and_behavior"]),
            habits_and_emotions=tuple(sections["personality"]["habits_and_emotions"]),
        ),
        social=SocialProfile(
            interaction_style=tuple(sections["social"]["interaction_style"]),
            boundaries=tuple(sections["social"]["boundaries"]),
        ),
        preferences=PreferenceProfile(
            likes_and_interests=tuple(sections["preferences"]["likes_and_interests"]),
            dislikes=tuple(sections["preferences"]["dislikes"]),
        ),
        language=LanguageProfile(
            speaking_style=tuple(sections["language"]["speaking_style"]),
            messaging_habits=tuple(sections["language"]["messaging_habits"]),
            address_habits=tuple(sections["language"]["address_habits"]),
        ),
        visual=VisualProfile(
            appearance=tuple(sections["visual"]["appearance"]),
            clothing=tuple(sections["visual"]["clothing"]),
            visual_boundaries=tuple(sections["visual"]["visual_boundaries"]),
        ),
        capabilities=CapabilityProfile(
            abilities=tuple(sections["capabilities"]["abilities"]),
            knowledge_scope=tuple(sections["capabilities"]["knowledge_scope"]),
            limitations=tuple(sections["capabilities"]["limitations"]),
        ),
    )


def _reject_internal_marker(value: str, field: str) -> None:
    if "{[" in value:
        raise ValueError(f"{field} 包含程序内部格式，请改用普通姓名或普通文字")


__all__ = [
    "PUBLIC_CHARACTER_FIELDS",
    "PUBLIC_CHARACTER_TEXT_FIELDS",
    "empty_public_character_sections",
    "parse_generated_public_character",
    "public_character_sections",
]
