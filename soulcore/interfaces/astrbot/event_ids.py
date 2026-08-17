"""Stable platform message identity extraction for AstrBot events."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from astrbot.api.event import AstrMessageEvent


def event_message_id(event: AstrMessageEvent) -> str:
    message = getattr(event, "message_obj", None)
    value = getattr(message, "message_id", None)
    if value is None:
        getter = getattr(event, "get_message_id", None)
        value = getter() if callable(getter) else ""
    return str(value or "").strip()


def event_reference_message_id(event: AstrMessageEvent) -> str:
    return event_reference_message_probe(event).value


@dataclass(frozen=True, slots=True)
class ReferenceMessageProbe:
    """Privacy-safe description of how one quoted-target locator was selected."""

    value: str
    source: str
    event_message_id: dict[str, object]
    reference_candidates: tuple[dict[str, object], ...]
    component_types: tuple[str, ...]
    reply_candidates: tuple[dict[str, object], ...]

    def safe_details(self) -> dict[str, object]:
        return {
            "selected_source": self.source or "none",
            "selected_locator": opaque_identifier_shape(self.value),
            "event_message_id": self.event_message_id,
            "reference_candidates": list(self.reference_candidates),
            "component_types": list(self.component_types),
            "reply_candidates": list(self.reply_candidates),
        }


def event_reference_message_probe(event: AstrMessageEvent) -> ReferenceMessageProbe:
    message = getattr(event, "message_obj", None)
    candidates = (
        ("event.message_reference", event),
        ("event.message_obj.message_reference", message),
        ("event.message_obj.raw_message.message_reference", getattr(message, "raw_message", None)),
        ("event.message_obj.raw_event.message_reference", getattr(message, "raw_event", None)),
        ("event.raw_message.message_reference", getattr(event, "raw_message", None)),
    )
    reference_probes = [_reference_candidate_probe(*candidate) for candidate in candidates]
    selected_value, selected_source = _first_reference_locator(reference_probes)
    reference_candidates = [details for _, _, details in reference_probes]
    components = _event_message_components(event, message)
    component_types = tuple(type(component).__name__ for component in components)
    reply_probes = _reply_component_probes(components)
    reply_candidates = [details for _, _, details in reply_probes]
    if not selected_value:
        selected_value, selected_source = _first_reference_locator(reply_probes)
    captured_reply_reference = _captured_reply_reference(event)
    reference_candidates.append(_capture_candidate_details(captured_reply_reference))
    if captured_reply_reference and not selected_value:
        selected_value = captured_reply_reference
        selected_source = "qq_parser_capture.reply_reference"
    return ReferenceMessageProbe(
        value=selected_value,
        source=selected_source,
        event_message_id=opaque_identifier_shape(event_message_id(event)),
        reference_candidates=tuple(reference_candidates),
        component_types=component_types,
        reply_candidates=tuple(reply_candidates),
    )


def _reference_candidate_probe(source: str, candidate: Any) -> tuple[str, str, dict[str, object]]:
    reference = (
        candidate.get("message_reference")
        if isinstance(candidate, Mapping)
        else getattr(candidate, "message_reference", None)
    )
    value = (
        reference.get("message_id")
        if isinstance(reference, Mapping)
        else getattr(reference, "message_id", None)
    )
    normalized = str(value or "").strip()
    return (
        normalized,
        source,
        {
            "source": source,
            "container_type": type(reference).__name__ if reference is not None else "none",
            "message_id": opaque_identifier_shape(normalized),
        },
    )


def _first_reference_locator(
    probes: list[tuple[str, str, dict[str, object]]],
) -> tuple[str, str]:
    for value, source, _details in probes:
        if value:
            return value, source
    return "", ""


def _event_message_components(event: AstrMessageEvent, message: Any) -> list[Any]:
    getter = getattr(event, "get_messages", None)
    try:
        components = list(getter() or []) if callable(getter) else []
    except Exception:
        components = []
    if not components:
        components = list(getattr(message, "message", None) or [])
    return components


def _reply_component_probes(
    components: list[Any],
) -> list[tuple[str, str, dict[str, object]]]:
    probes: list[tuple[str, str, dict[str, object]]] = []
    for index, component in enumerate(components):
        if component.__class__.__name__.lower() != "reply":
            continue
        value = getattr(component, "id", None) or getattr(component, "message_id", None)
        normalized = str(value or "").strip()
        source = f"event.messages[{index}].Reply.id"
        probes.append(
            (
                normalized,
                source,
                {"source": source, "message_id": opaque_identifier_shape(normalized)},
            )
        )
    return probes


def _captured_reply_reference(event: AstrMessageEvent) -> str:
    try:
        from .qq_reference_ids import event_reply_reference_id

        return event_reply_reference_id(event)
    except Exception:
        return ""


def _capture_candidate_details(value: str) -> dict[str, object]:
    return {
        "source": "qq_parser_capture.reply_reference",
        "container_type": "capture_cache" if value else "none",
        "message_id": opaque_identifier_shape(value),
    }


def opaque_identifier_shape(value: Any) -> dict[str, object]:
    """Describe an opaque identifier without persisting the identifier itself."""

    normalized = str(value or "").strip()
    if not normalized:
        return {"present": False, "kind": "empty", "length": 0, "digest": ""}
    kind = "qq_reference_index" if normalized.upper().startswith("REFIDX") else "opaque_id"
    return {
        "present": True,
        "kind": kind,
        "length": len(normalized),
        "digest": hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12],
    }


__all__ = [
    "ReferenceMessageProbe",
    "event_message_id",
    "event_reference_message_id",
    "event_reference_message_probe",
    "opaque_identifier_shape",
]
