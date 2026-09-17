"""One administrator-owned rule for admitting group messages."""

from collections.abc import Mapping
from typing import Any

UNCHANGED_GROUP_WAKE = object()


def default_group_wake_rule() -> dict[str, Any]:
    return {"enabled": False, "keywords": []}


def normalize_group_wake_rule(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"enabled", "keywords"}:
        raise ValueError("group wake rule requires enabled and keywords")
    if not isinstance(value["enabled"], bool):
        raise ValueError("group wake enabled must be a boolean")
    raw = value["keywords"]
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ValueError("group wake keywords must be a list of strings")
    keywords: list[str] = []
    seen: set[str] = set()
    for item in raw:
        word = item.strip()
        if not word or word.casefold() in seen:
            continue
        if len(word) > 200 or "\n" in word or "\r" in word:
            raise ValueError("each group wake keyword must be one line of at most 200 characters")
        keywords.append(word)
        seen.add(word.casefold())
    if len(keywords) > 100:
        raise ValueError("group wake keywords must contain at most 100 entries")
    return {"enabled": value["enabled"], "keywords": keywords}


def resolve_group_wake_rule(default: Mapping[str, Any], override: Any) -> dict[str, Any]:
    return normalize_group_wake_rule(default if override is None else override)


def group_wake_matches(rule: Mapping[str, Any], *, mentioned: bool, text: str) -> bool:
    if not rule["enabled"]:
        return True
    return mentioned and (
        not rule["keywords"] or any(word.casefold() in text.casefold() for word in rule["keywords"])
    )
