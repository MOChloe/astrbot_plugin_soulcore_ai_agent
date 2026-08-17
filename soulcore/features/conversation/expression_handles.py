"""Build run-scoped opaque message and group-member addressing handles."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from ...contracts.models import CharacterInstance, ConversationMessage, PlatformMessageFragment


async def load_expression_handles(
    platform_messages: Any,
    profile_id: str,
    instance_id: str,
    *,
    instance: CharacterInstance | None,
    messages: list[ConversationMessage],
    core_run_id: int,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    fragments = await _load_fragments(platform_messages, profile_id, instance_id, messages)
    message_allowlist = _message_handles(profile_id, instance_id, messages, fragments)
    member_allowlist, refs_by_sender = _member_handles(
        profile_id, instance_id, instance, messages, core_run_id
    )
    return message_allowlist, member_allowlist, refs_by_sender


async def _load_fragments(
    platform_messages: Any,
    profile_id: str,
    instance_id: str,
    messages: list[ConversationMessage],
) -> list[PlatformMessageFragment]:
    message_ids = tuple(
        dict.fromkeys(int(item.message_id) for item in messages if int(item.message_id) > 0)
    )
    if not message_ids:
        return []
    return await platform_messages.list_message_fragments(
        profile_id,
        instance_id,
        ledger_message_ids=message_ids,
        include_retracted=True,
        limit=min(500, max(100, len(message_ids) * 4)),
    )


def _message_handles(
    profile_id: str,
    instance_id: str,
    messages: list[ConversationMessage],
    fragments: list[PlatformMessageFragment],
) -> dict[str, dict[str, Any]]:
    allowlist: dict[str, dict[str, Any]] = {}
    messages_by_id = {int(item.message_id): item for item in messages}
    now = datetime.now(UTC)
    for fragment in fragments:
        value = _fragment_handle(profile_id, instance_id, fragment, messages_by_id, now)
        if value is None:
            continue
        if value["retraction_status"] not in {
            "PENDING",
            "SENDING",
            "RETRACTED",
            "UNKNOWN_AFTER_CRASH",
        }:
            allowlist[value["message_ref"]] = value
    return allowlist


def _fragment_handle(
    profile_id: str,
    instance_id: str,
    fragment: PlatformMessageFragment,
    messages_by_id: dict[int, ConversationMessage],
    now: datetime,
) -> dict[str, Any] | None:
    platform_message_id = str(fragment.platform_message_id or "").strip()
    platform_reference_id = str(fragment.platform_reference_id or "").strip()
    message_ref = str(fragment.message_ref or "").strip()
    if not message_ref or not platform_message_id:
        return None
    status = fragment.retraction_status.value if fragment.retraction_status is not None else ""
    deadline = fragment.retractable_until
    direction = fragment.direction.value.upper()
    can_retract = _can_retract_fragment(fragment, status, direction, deadline, now)
    ledger_id = int(fragment.ledger_message_id)
    message = messages_by_id.get(ledger_id)
    return {
        "message_ref": message_ref,
        "profile_id": profile_id,
        "instance_id": instance_id,
        "ledger_message_id": ledger_id,
        "fragment_ordinal": int(fragment.fragment_ordinal),
        "platform_instance_id": str(fragment.platform_instance_id),
        "route_umo": str(fragment.route_umo),
        "platform_message_id": platform_message_id,
        "platform_reference_id": platform_reference_id,
        "native_reply_target_id": (
            platform_reference_id or platform_message_id if fragment.native_reply_supported else ""
        ),
        "content_kind": str(fragment.content_kind),
        "content_projection": str(fragment.content_projection or "")[:120],
        "sender_id": str(fragment.sender_id or ""),
        "sender_display_name": str(message.sender_name if message is not None else "")[:80],
        "native_reply_supported": bool(fragment.native_reply_supported),
        "reply_allowed": True,
        "retract_allowed": can_retract,
        "retractable_until": deadline,
        "retraction_status": status,
    }


def _can_retract_fragment(
    fragment: PlatformMessageFragment,
    status: str,
    direction: str,
    deadline: datetime | None,
    now: datetime,
) -> bool:
    return bool(
        direction == "OUTBOUND"
        and fragment.self_retraction_supported
        and not status
        and (deadline is None or deadline > now)
    )


def _member_handles(
    profile_id: str,
    instance_id: str,
    instance: CharacterInstance | None,
    messages: list[ConversationMessage],
    core_run_id: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    if instance is None or instance.scope != "group":
        return {}, {}
    members: dict[str, dict[str, Any]] = {}
    refs_by_sender: dict[str, str] = {}
    for message in reversed(_recent_group_members(messages)):
        sender_id = str(message.sender_id).strip()
        member_ref = _member_ref(profile_id, instance_id, core_run_id, sender_id)
        refs_by_sender[sender_id] = member_ref
        members[member_ref] = _member_value(member_ref, profile_id, instance_id, instance, message)
    for message in messages:
        member_ref = refs_by_sender.get(str(message.sender_id or ""))
        if member_ref:
            members[member_ref]["ledger_message_ids"].append(int(message.message_id))
    return members, refs_by_sender


def _recent_group_members(messages: list[ConversationMessage]) -> list[ConversationMessage]:
    by_sender: dict[str, ConversationMessage] = {}
    for message in messages:
        sender_id = str(message.sender_id or "").strip()
        if sender_id and message.role == "user":
            by_sender[sender_id] = message
    pinned = _pinned_member_ids(messages)
    recent: list[ConversationMessage] = [by_sender[value] for value in pinned if value in by_sender]
    seen: set[str] = set()
    seen.update(str(item.sender_id or "").strip() for item in recent)
    for message in reversed(messages):
        sender_id = str(message.sender_id or "").strip()
        if not _eligible_group_member(message, sender_id, seen):
            continue
        seen.add(sender_id)
        recent.append(message)
    return recent


def _eligible_group_member(message: ConversationMessage, sender_id: str, seen: set[str]) -> bool:
    return bool(
        sender_id
        and str(message.sender_name or "").strip()
        and sender_id != "soulcore"
        and sender_id not in seen
        and message.role == "user"
    )


def _pinned_member_ids(messages: list[ConversationMessage]) -> tuple[str, ...]:
    values: list[str] = []
    for message in reversed(messages[-8:]):
        sender_id = str(message.sender_id or "").strip()
        if sender_id and message.role == "user":
            values.append(sender_id)
        for component in message.components:
            if not isinstance(component, dict):
                continue
            kind = str(component.get("type") or "").strip().lower()
            if kind == "at":
                values.append(str(component.get("qq") or component.get("sender_id") or "").strip())
    return tuple(value for value in dict.fromkeys(values) if value)


def _member_ref(profile_id: str, instance_id: str, run_id: int, sender_id: str) -> str:
    source = f"{profile_id}\x1f{instance_id}\x1f{run_id}\x1f{sender_id}"
    return f"member_ref:v1:{hashlib.sha256(source.encode()).hexdigest()[:20]}"


def _member_value(
    member_ref: str,
    profile_id: str,
    instance_id: str,
    instance: CharacterInstance,
    message: ConversationMessage,
) -> dict[str, Any]:
    return {
        "member_ref": member_ref,
        "profile_id": profile_id,
        "instance_id": instance_id,
        "route_umo": str(instance.route_umo or ""),
        "sender_id": str(message.sender_id).strip(),
        "display_name": str(message.sender_name).strip()[:80],
        "ledger_message_ids": [],
    }
