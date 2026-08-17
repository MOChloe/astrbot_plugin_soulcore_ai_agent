"""Player-facing presentation helpers for the SoulCore Page.

These projections deliberately speak about a character, people and lived
events.  Repository identities and engine terminology stay behind this
boundary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from ...features.character_model.importing import public_character_sections
from ...features.identity import resolve_private_display_name
from ...release_notes import load_public_release_notes
from .presentation import jsonable


def player_role_ref(profile_id: str) -> str:
    return _stable_ref("role", profile_id)


def player_contact_ref(profile_id: str, instance_id: str) -> str:
    return _stable_ref("contact", profile_id, instance_id)


def player_role_view(row: Mapping[str, Any], *, selected: bool) -> dict[str, Any]:
    profile_id = str(row.get("profile_id") or row.get("id") or "")
    display_name = str(row.get("display_name") or row.get("name") or "").strip()
    if not display_name or display_name == profile_id or display_name.casefold() == "default":
        display_name = "未命名角色"
    return {
        "role_ref": player_role_ref(profile_id),
        "display_name": display_name,
        "selected": bool(selected),
    }


def player_contact_view(
    profile_id: str,
    instance: Mapping[str, Any],
    *,
    latest_at: Any = None,
    problem_count: int = 0,
) -> dict[str, Any]:
    scope = "group" if str(instance.get("scope") or "").lower() == "group" else "private"
    display_name = _text(instance.get("display_name"))
    if scope == "private":
        identifier_label = _text(instance.get("identifier_label") or instance.get("target_label"))
        display_name = resolve_private_display_name(display_name, identifier_label)
        if display_name == "对方":
            display_name = identifier_label or "一位好友"
    return {
        "contact_ref": player_contact_ref(profile_id, str(instance.get("instance_id") or "")),
        "display_name": display_name
        or _text(instance.get("target_label"))
        or ("一个群聊" if scope == "group" else "一位好友"),
        "kind": scope,
        "kind_label": "群聊" if scope == "group" else "好友",
        "latest_at": jsonable(latest_at),
        "problem_count": max(0, int(problem_count)),
    }


def player_current_life_view(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "narrative_time": _text(value.get("narrative_time")),
        "location": _text(value.get("location")),
        "doing": _text(value.get("doing")),
        "body_state": _text(value.get("body_state")),
        "mood": _text(value.get("mood")),
        "intention": _text(value.get("intention")),
        "concern": _text(value.get("current_concern")),
        "as_of": jsonable(value.get("as_of") or value.get("updated_at")),
    }


def player_life_events_view(values: Any, *, limit: int = 12) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in _sequence(values)[: max(1, min(limit, 30))]:
        content = _text(item.get("content"))
        if not content:
            continue
        result.append(
            {
                "content": content,
                "started_at": jsonable(item.get("frame_start_at")),
                "ended_at": jsonable(item.get("frame_end_at") or item.get("created_at")),
            }
        )
    return result


def player_intents_view(values: Any, *, limit: int = 12) -> list[dict[str, Any]]:
    terminal = {"COMPLETED", "CANCELLED", "CONSUMED", "EXPIRED", "SUPERSEDED"}
    labels = {
        "OPEN": "还在想着",
        "PLANNED": "已经打算去做",
        "IN_PROGRESS": "正在做",
        "BLOCKED": "暂时做不了",
    }
    result: list[dict[str, Any]] = []
    for item in _sequence(values):
        status = _text(item.get("status")).upper()
        if status in terminal:
            continue
        summary = _text(item.get("summary") or item.get("goal"))
        if not summary:
            continue
        result.append(
            {
                "summary": summary,
                "status_label": labels.get(status, "还记着"),
                "target_at": jsonable(item.get("target_at") or item.get("not_before_at")),
                "updated_at": jsonable(item.get("updated_at") or item.get("created_at")),
            }
        )
        if len(result) >= max(1, min(limit, 30)):
            break
    return result


def player_memories_view(values: Any, *, limit: int = 30) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in _sequence(values)[: max(1, min(limit, 100))]:
        title = _text(item.get("ultra_brief") or item.get("brief"))
        description = _text(item.get("brief"))
        if not title and not description:
            continue
        result.append(
            {
                "title": title or description,
                "description": description if description != title else "",
                "updated_at": jsonable(item.get("updated_at") or item.get("created_at")),
            }
        )
    return result


def player_people_view(values: Any) -> list[dict[str, Any]]:
    return [
        {
            "person_ref": _text(item.get("person_ref")),
            "display_name": _text(item.get("display_name")) or "一位群成员",
            "active_count": max(0, int(item.get("active_count") or 0)),
            "selected": bool(item.get("selected")),
        }
        for item in _sequence(values)
        if _text(item.get("person_ref"))
    ]


def player_portrait_view(values: Any) -> list[dict[str, Any]]:
    result = []
    for item in _sequence(values):
        if _text(item.get("status")).upper() != "ACTIVE":
            continue
        text = _text(item.get("text"))
        if not text:
            continue
        result.append(
            {
                "entry_ref": _text(item.get("entry_ref")),
                "category_label": _text(item.get("category_label")) or "了解",
                "text": text,
                "layer_label": _text(item.get("layer_label")),
                "updated_at": jsonable(item.get("updated_at")),
            }
        )
    return result


def player_character_view(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    character = snapshot.get("character_model")
    source = character if isinstance(character, Mapping) else snapshot
    model = _mapping(source.get("model"))
    return {
        "revision": int(source.get("revision") or 0),
        "sections": public_character_sections(model),
    }


def player_world_view(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    definition = _mapping(snapshot.get("definition"))
    boundaries = []
    for item in _sequence(snapshot.get("boundaries")):
        if not bool(item.get("enabled", True)):
            continue
        boundaries.append(
            {
                "rule": _text(item.get("rule_text")),
                "preferred": _text(item.get("positive_space")),
                "importance": (
                    "不能突破" if _text(item.get("severity")).upper() == "HARD" else "优先遵循"
                ),
            }
        )
    return {
        "revision": int(definition.get("revision") or 0),
        "world_brief": _text(definition.get("world_brief")),
        "world_rules": _text(definition.get("world_rules")),
        "life_direction": _text(definition.get("life_direction")),
        "world_texture": _text(definition.get("world_texture")),
        "expansion_policy": _text(definition.get("expansion_policy")) or "OPEN",
        "boundaries": boundaries,
    }


def player_release_notes_view() -> dict[str, Any]:
    return {"items": load_public_release_notes()}


def _stable_ref(kind: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join((kind, *parts)).encode("utf-8")).hexdigest()[:24]
    return f"{kind}-{digest}"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _text(value: Any) -> str:
    return str(value or "").strip()


__all__ = [
    "player_character_view",
    "player_contact_ref",
    "player_contact_view",
    "player_current_life_view",
    "player_intents_view",
    "player_life_events_view",
    "player_memories_view",
    "player_people_view",
    "player_portrait_view",
    "player_release_notes_view",
    "player_role_ref",
    "player_role_view",
    "player_world_view",
]
