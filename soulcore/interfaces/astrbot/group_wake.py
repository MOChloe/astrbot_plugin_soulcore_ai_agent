"""Check raw group addressing before initialization or any model work."""

from typing import Any

from ...contracts.group_wake import group_wake_matches
from .context_message import group_wake_input


def event_matches_group_wake(event: Any, rule: dict) -> bool:
    if not rule["enabled"]:
        return True
    mentioned, text = group_wake_input(event)
    return group_wake_matches(rule, mentioned=mentioned, text=text)
